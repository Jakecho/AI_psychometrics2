"""Response generation and CSV upload for the dichotomous Rasch model."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io

import numpy as np
import pandas as pd

from .model import probability


@dataclass
class Dataset:
    """A persistent response matrix shared by every page.

    X uses np.nan for missing responses. true_theta / true_b exist only for simulated
    data; uploaded operational data never gets invented generating parameters.
    """
    X: np.ndarray
    item_ids: list
    person_ids: list
    source: str = "Simulation"
    true_theta: np.ndarray | None = None
    true_b: np.ndarray | None = None
    settings: dict = field(default_factory=dict)

    @property
    def identity(self):
        h = hashlib.sha256()
        h.update(np.nan_to_num(self.X, nan=-1).astype(np.int8).tobytes())
        h.update(repr((self.item_ids, self.source, sorted(self.settings.items()))).encode())
        return h.hexdigest()[:12]

    @property
    def mask(self):
        return ~np.isnan(self.X)

    @property
    def n_persons(self):
        return self.X.shape[0]

    @property
    def n_items(self):
        return self.X.shape[1]

    @property
    def has_truth(self):
        return self.true_theta is not None and self.true_b is not None

    def frame(self):
        return pd.DataFrame(self.X, columns=self.item_ids, index=self.person_ids)


def simulate(n_persons=500, n_items=20, theta_mean=0.0, theta_sd=1.0, b_mean=0.0, b_sd=1.0,
             missing=0.0, seed=2026, true_b=None):
    """theta ~ N(mu_theta, sd_theta^2), b ~ N(mu_b, sd_b^2), X ~ Bernoulli(P).

    `true_b` fixes the item difficulties (Monte Carlo: items held constant, persons redrawn).

    Missingness (if requested) is applied completely at random after generation, so the
    realized responses and the observation mask are both preserved.
    """
    if n_persons < 2 or n_items < 2:
        raise ValueError("Use at least two persons and two items.")
    if theta_sd <= 0 or b_sd <= 0:
        raise ValueError("Standard deviations must be positive.")
    if not 0 <= missing < 1:
        raise ValueError("Missing proportion must be in [0, 1).")
    rng = np.random.default_rng(int(seed))
    theta = rng.normal(theta_mean, theta_sd, n_persons)
    b = rng.normal(b_mean, b_sd, n_items)
    if true_b is not None:
        b = np.asarray(true_b, float)
        if b.shape != (n_items,):
            raise ValueError("true_b must have one value per item.")
    X = (rng.random((n_persons, n_items)) < probability(theta, b)).astype(float)
    if missing > 0:
        drop = rng.random(X.shape) < missing
        X[drop] = np.nan
        # Every person keeps at least one observed response (a person with none has no data).
        empty = np.flatnonzero(np.isnan(X).all(axis=1))
        for i in empty:
            j = rng.integers(n_items)
            X[i, j] = float(rng.random() < probability([theta[i]], [b[j]])[0, 0])
    settings = dict(n_persons=n_persons, n_items=n_items, theta_mean=theta_mean, theta_sd=theta_sd,
                    b_mean=b_mean, b_sd=b_sd, missing=missing, seed=int(seed))
    return Dataset(X=X, item_ids=[f"I{j+1:02d}" for j in range(n_items)],
                   person_ids=[f"P{i+1:04d}" for i in range(n_persons)], source="Simulation",
                   true_theta=theta, true_b=b, settings=settings)


def load_csv(content, id_column=None):
    """Validate an uploaded persons-by-items CSV of 0/1 responses (blank = missing).

    `id_column` names a person-identifier column; it is removed, never used as an item.
    Non-numeric columns other than the declared identifier are rejected rather than guessed.
    """
    frame = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else content)
    if frame.empty:
        raise ValueError("The CSV has no rows.")
    columns = [str(c) for c in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("Item identifiers (column names) must be unique.")
    frame.columns = columns
    if id_column:
        if id_column not in frame.columns:
            raise ValueError(f"Identifier column '{id_column}' not found.")
        person_ids = frame.pop(id_column).astype(str).tolist()
        if len(set(person_ids)) != len(person_ids):
            raise ValueError("Person identifiers must be unique.")
    else:
        person_ids = [f"P{i+1:04d}" for i in range(len(frame))]
    bad = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if bad:
        raise ValueError("Non-numeric columns found: " + ", ".join(bad[:5]) +
                         ". Declare a person-identifier column or remove non-item columns.")
    X = frame.to_numpy(float)
    values = X[~np.isnan(X)]
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Responses must be 0, 1, or blank (missing).")
    if X.shape[1] < 3:
        raise ValueError("At least three items are required.")
    if X.shape[0] < 10:
        raise ValueError("At least ten persons are required.")
    empty_persons = np.isnan(X).all(axis=1)
    if empty_persons.any():
        raise ValueError(f"{int(empty_persons.sum())} person(s) have no observed responses; remove them.")
    empty_items = np.isnan(X).all(axis=0)
    if empty_items.any():
        raise ValueError("Items with no observed responses: " +
                         ", ".join(np.array(frame.columns)[empty_items][:5]))
    return Dataset(X=X, item_ids=list(frame.columns), person_ids=person_ids, source="Upload",
                   settings=dict(rows=X.shape[0], items=X.shape[1]))


def summaries(data):
    """Raw scores, item p-values and extreme-score counts on observed responses."""
    mask = data.mask
    x = np.where(mask, data.X, 0.0)
    scores, n_obs = x.sum(axis=1), mask.sum(axis=1)
    p_values = x.sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    return dict(raw_scores=scores, n_observed=n_obs, p_values=p_values,
                zero_scores=int(np.sum(scores == 0)), perfect_scores=int(np.sum(scores == n_obs)),
                items_all_correct=int(np.sum(p_values == 1)), items_all_wrong=int(np.sum(p_values == 0)),
                missing_rate=float(1 - mask.mean()))


def connectivity_issues(data):
    """Detect designs where items or persons cannot be placed on one scale.

    Items are connected if some person answers both; a disconnected item graph means the
    relative difficulties of the two blocks are not identified.
    """
    mask = data.mask.astype(int)
    co = (mask.T @ mask) > 0
    J = co.shape[0]
    seen = np.zeros(J, bool)
    stack = [0]
    seen[0] = True
    while stack:
        j = stack.pop()
        for k in np.flatnonzero(co[j] & ~seen):
            seen[k] = True
            stack.append(k)
    if not seen.all():
        return [f"Items are split into disconnected blocks ({int(seen.sum())} connected to {data.item_ids[0]}); "
                "relative difficulties across blocks are not identified."]
    return []
