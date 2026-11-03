"""Response datasets: responses (NaN = not administered), groups, item structure, bank anchors, truth."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import Item, ItemSpec, make_item


@dataclass
class Dataset:
    name: str
    Y: np.ndarray                        # (N, J) float, NaN = missing by design / not presented
    group: np.ndarray                    # (N,) int group index
    group_names: list
    specs: list                          # ItemSpec per column
    anchors: dict = field(default_factory=dict)   # item id -> Item with bank (fixed) parameters
    truth: dict | None = None            # generating values: items {id: Item}, theta, groups [dict]
    description: str = ""

    def __post_init__(self):
        self.Y = np.asarray(self.Y, float)
        self.group = np.asarray(self.group, int)
        if self.Y.ndim != 2:
            raise ValueError("Responses must be a two-dimensional table.")
        if len(self.group) != self.Y.shape[0]:
            raise ValueError("Group vector length does not match the number of examinees.")
        if len(self.specs) != self.Y.shape[1]:
            raise ValueError("Item structure does not match the number of response columns.")
        self.group_names = [str(g) for g in self.group_names]
        if self.group.min(initial=0) < 0 or self.group.max(initial=0) >= len(self.group_names):
            raise ValueError("Group codes must index the list of group names.")

    @property
    def N(self):
        return self.Y.shape[0]

    @property
    def J(self):
        return self.Y.shape[1]

    @property
    def G(self):
        return len(self.group_names)

    @property
    def ids(self):
        return [s.id for s in self.specs]

    @property
    def observed(self):
        return ~np.isnan(self.Y)

    @property
    def ncats(self):
        return [s.ncat for s in self.specs]

    @property
    def identity(self):
        h = hashlib.sha256()
        h.update(np.nan_to_num(self.Y, nan=-9).astype(np.int16).tobytes())
        h.update(self.group.astype(np.int16).tobytes())
        h.update(json.dumps([self.ids, self.ncats, self.group_names,
                             {k: [v.model, v.a, list(v.d)] for k, v in sorted(self.anchors.items())}]).encode())
        return h.hexdigest()[:16]

    def group_sizes(self):
        return np.bincount(self.group, minlength=self.G)

    def counts(self, j, g=None):
        """Observed category counts for item j (optionally within group g)."""
        y = self.Y[:, j]
        m = ~np.isnan(y)
        if g is not None:
            m &= self.group == g
        return np.bincount(y[m].astype(int), minlength=self.specs[j].ncat)

    def administered(self):
        """(G, J) boolean: item j was presented to at least one examinee in group g."""
        return np.array([self.observed[self.group == g].any(axis=0) for g in range(self.G)])

    def summary(self):
        return dict(name=self.name, examinees=self.N, items=self.J, groups=self.G,
                    anchors=len(self.anchors), missing=float(np.mean(~self.observed)))

    def subset_groups(self, groups, name=None):
        """Examinees of the listed groups only; items nobody in them saw are dropped."""
        groups = list(groups)
        keep = np.isin(self.group, groups)
        cols = np.flatnonzero(self.observed[keep].any(axis=0))
        remap = {g: k for k, g in enumerate(groups)}
        ids = [self.ids[j] for j in cols]
        truth = None
        if self.truth:
            truth = dict(self.truth, theta=self.truth["theta"][keep], groups=[self.truth["groups"][g] for g in groups])
        return Dataset(name or f"{self.name} · groups {', '.join(self.group_names[g] for g in groups)}",
                       self.Y[np.ix_(keep, cols)], np.array([remap[g] for g in self.group[keep]]),
                       [self.group_names[g] for g in groups], [self.specs[j] for j in cols],
                       anchors={k: v for k, v in self.anchors.items() if k in ids}, truth=truth,
                       description=self.description)

    def to_csv(self):
        frame = pd.DataFrame(self.Y, columns=self.ids)
        frame.insert(0, "group", [self.group_names[g] for g in self.group])
        return frame.to_csv(index=False, na_rep="")


def validate(data: Dataset, model=None, anchored=False):
    """Data problems → (errors, warnings). Errors stop calibration; warnings are shown to the learner."""
    errors, warnings = [], []
    ids = data.ids
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errors.append(f"Duplicated item IDs: {', '.join(dup)}.")
    if any(not str(i).strip() for i in ids):
        errors.append("Every item needs a non-empty ID.")
    Y = data.Y
    obs = data.observed
    for j, s in enumerate(data.specs):
        y = Y[obs[:, j], j]
        if len(y) == 0:
            errors.append(f"Item {s.id} has no observed responses.")
            continue
        if np.any(y != np.round(y)):
            errors.append(f"Item {s.id} has non-integer responses.")
            continue
        if y.min() < 0 or y.max() > s.ncat - 1:
            errors.append(f"Item {s.id}: responses must lie in 0…{s.ncat - 1}.")
            continue
        c = np.bincount(y.astype(int), minlength=s.ncat)
        empty = [str(k) for k in range(s.ncat) if c[k] == 0]
        if empty and not (anchored and s.id in data.anchors):
            warnings.append(f"Item {s.id}: category {', '.join(empty)} never observed — the adjacent step/threshold "
                            "is not identified by the data; the estimate stops at the optimizer bound.")
        elif not empty and c.min() < 5 and not (anchored and s.id in data.anchors):
            warnings.append(f"Item {s.id}: sparse category (only {c.min()} responses) — expect a large standard error.")
    if (obs.sum(axis=1) == 0).any():
        warnings.append(f"{int((obs.sum(axis=1) == 0).sum())} examinee(s) answered no items; they only carry the prior.")
    for k, it in data.anchors.items():
        if k not in ids:
            errors.append(f"Anchor {k} is not a column of the response data.")
            continue
        j = ids.index(k)
        if it.ncat != data.specs[j].ncat:
            errors.append(f"Anchor {k}: bank parameters imply {it.ncat} categories but the data declare "
                          f"{data.specs[j].ncat}.")
        if anchored and model is not None and it.model != model:
            errors.append(f"Anchor {k} was calibrated with {it.model}, not {model}; anchors must use the fitted model.")
    if model == "RSM":
        by_scale = {}
        for s in data.specs:
            by_scale.setdefault(s.scale, set()).add(s.ncat)
        for sc, ks in by_scale.items():
            if len(ks) > 1:
                errors.append(f"RSM scale group '{sc}' mixes items with {sorted(ks)} categories; items sharing "
                              "rating-scale steps need the same category definitions.")
    return errors, warnings


def from_frame(frame: pd.DataFrame, name="Uploaded data", group_column=None, ncats=None, anchors=None):
    """Build a Dataset from a data frame of integer responses (blank = missing)."""
    frame = frame.copy()
    if group_column and group_column in frame.columns:
        labels = frame.pop(group_column).astype(str)
        names = list(dict.fromkeys(labels))
        group = labels.map({n: i for i, n in enumerate(names)}).to_numpy()
    else:
        names, group = ["All examinees"], np.zeros(len(frame), int)
    Y = frame.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cols = [str(c) for c in frame.columns]
    if ncats is None:
        ncats = [int(np.nanmax(Y[:, j])) + 1 if np.isfinite(np.nanmax(Y[:, j])) else 2 for j in range(Y.shape[1])]
    specs = [ItemSpec(c, max(2, k)) for c, k in zip(cols, ncats)]
    return Dataset(name, Y, group, names, specs, anchors=anchors or {})


def read_csv(text_or_bytes, **kw):
    if isinstance(text_or_bytes, bytes):
        text_or_bytes = text_or_bytes.decode("utf-8-sig")
    return from_frame(pd.read_csv(io.StringIO(text_or_bytes)), **kw)


def anchors_from_frame(frame: pd.DataFrame):
    """Bank table: columns item, model, a, then d1..dK (steps or thresholds). RSM may give b and tau1..tauK."""
    out = {}
    for _, r in frame.iterrows():
        model = str(r["model"]).upper()
        d = [float(r[c]) for c in frame.columns if str(c).lower().startswith(("d", "δ")) and pd.notna(r[c])
             and str(c).lower() not in ("d",)]
        if model == "RSM" and "b" in frame.columns and pd.notna(r.get("b")):
            tau = [float(r[c]) for c in frame.columns if str(c).lower().startswith("tau") and pd.notna(r[c])]
            it = Item(str(r["item"]), "RSM", 1.0, (), b=float(r["b"]), tau=tuple(tau))
        else:
            it = make_item(str(r["item"]), model, float(r.get("a", 1.0)), tuple(d))
        out[it.id] = it
    return out


def walkthrough_dataset():
    """Deterministic teaching data: 5 examinees × 3 items, categories 0/1/2 (every category observed)."""
    Y = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 2],
                  [2, 1, 1],
                  [2, 2, 2]], float)
    specs = [ItemSpec(f"I{j + 1}", 3) for j in range(3)]
    return Dataset("Walkthrough · 5 × 3 teaching data", Y, np.zeros(5, int), ["All examinees"], specs,
                   description="Small deterministic dataset used by the numerical walkthrough.")
