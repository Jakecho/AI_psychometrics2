"""Datasets: built-in numerical examples A–D and simulated (mixed-format) tests.

A Dataset holds known item parameters, a response matrix (NaN = missing) and, for simulated data,
the generating abilities. Everything is reproducible from the seed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from irt_models import DICHOTOMOUS, Item, make_item


@dataclass
class Dataset:
    name: str
    items: list
    Y: np.ndarray
    theta_true: np.ndarray | None = None
    person_ids: list = field(default_factory=list)
    description: str = ""
    spec: dict = field(default_factory=dict)

    def __post_init__(self):
        self.Y = np.atleast_2d(np.asarray(self.Y, float))
        if not self.person_ids:
            self.person_ids = [f"P{i + 1:03d}" for i in range(self.Y.shape[0])]

    @property
    def n_persons(self):
        return self.Y.shape[0]

    @property
    def n_items(self):
        return len(self.items)

    @property
    def identity(self):
        payload = json.dumps([[it.__dict__ for it in self.items], np.nan_to_num(self.Y, nan=-1).tolist()],
                             default=str, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]

    @property
    def has_truth(self):
        return self.theta_true is not None

    @property
    def models(self):
        return sorted({it.model for it in self.items}, key=lambda m: ("Rasch 2PL 3PL 4PL PCM GPCM RSM GRM".split()).index(m))

    def with_items(self, items):
        return Dataset(self.name + " (edited items)", list(items), self.Y.copy(), self.theta_true, list(self.person_ids),
                       self.description, dict(self.spec))

    def with_pattern(self, person, y):
        Y = self.Y.copy()
        Y[person] = y
        name = self.name if self.name.endswith("(edited responses)") else self.name + " (edited responses)"
        return Dataset(name, list(self.items), Y, self.theta_true, list(self.person_ids), self.description,
                       dict(self.spec))


# ---- numerical examples from the specification --------------------------------------------

def example_a():
    items = [make_item(f"R{j + 1}", "Rasch", b=b) for j, b in enumerate((-1.0, 0.0, 1.0))]
    pats = [(1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 0, 0), (1, 1, 1), (0, 1, 0)]
    return Dataset("Example A · Rasch (b = −1, 0, 1)", items, np.array(pats, float),
                   person_ids=["x=" + "".join(map(str, p)) for p in pats],
                   description="Three Rasch items, b = (−1, 0, 1). The first examinee answers x = (1, 0, 1). All eight "
                               "patterns are included: patterns with the same raw score share every Rasch MLE, and "
                               "000 / 111 have no finite MLE.")


def example_b():
    items = [make_item("I1", "2PL", a=1.0, b=0.0), make_item("I2", "2PL", a=2.0, b=0.0)]
    pats = [(1, 0), (0, 1), (0, 0), (1, 1)]
    return Dataset("Example B · 2PL, same raw score", items, np.array(pats, float),
                   person_ids=["A=(1,0)", "B=(0,1)", "(0,0)", "(1,1)"],
                   description="(a, b) = (1, 0) and (2, 0), D = 1. Patterns A = (1, 0) and B = (0, 1) have the same raw "
                               "score but MLEs −0.4196 and +0.4196; TCC inversion gives 0 for both.")


def example_c():
    items = [make_item("C1", "PCM", steps=(-1.0, 0.5)),
             make_item("C2", "PCM", steps=(-0.5, 1.0)),
             make_item("C3", "PCM", steps=(0.0, -0.3, 1.2))]
    pats = [(1, 2, 1), (2, 1, 1), (0, 1, 2), (2, 2, 3), (0, 0, 0), (1, 1, 1), (2, 0, 3), (1, 0, 0)]
    return Dataset("Example C · Partial credit (3 items)", items, np.array(pats, float),
                   person_ids=["y=" + "".join(map(str, p)) for p in pats],
                   description="PCM items with step difficulties (−1, 0.5), (−0.5, 1) and (0, −0.3, 1.2): two items "
                               "with 3 categories and one with 4. Item C3 has a reversed step (δ₂ < δ₁).")


def example_d():
    items = [make_item("G1", "GRM", a=1.0, steps=(-1.0, 1.0)),
             make_item("G2", "GRM", a=1.5, steps=(-0.5, 0.8)),
             make_item("G3", "GRM", a=0.8, steps=(-1.5, 0.0, 1.5)),
             make_item("G4", "GRM", a=1.2, steps=(0.0, 1.2))]
    pats = [(1, 1, 2, 1), (2, 1, 1, 0), (0, 0, 1, 0), (2, 2, 3, 2), (0, 0, 0, 0), (1, 2, 1, 1)]
    return Dataset("Example D · Graded response (4 items)", items, np.array(pats, float),
                   person_ids=["y=" + "".join(map(str, p)) for p in pats],
                   description="Item G1 is the three-category GRM item with a = 1 and thresholds (−1, 1): at θ = 0, "
                               "(P₀, P₁, P₂) = (0.26894, 0.46212, 0.26894). Three more GRM items extend it.")


EXAMPLES = {"Example A · Rasch": example_a, "Example B · 2PL": example_b, "Example C · PCM": example_c,
            "Example D · GRM": example_d}


# ---- simulation -----------------------------------------------------------------------------

def draw_theta(n, dist="normal", mean=0.0, sd=1.0, rng=None):
    rng = rng or np.random.default_rng()
    if dist == "uniform":
        half = np.sqrt(3) * sd
        return rng.uniform(mean - half, mean + half, n)
    if dist == "skewed":                       # standardised chi-square(4): skewness √2
        x = rng.chisquare(4, n)
        return mean + sd * (x - 4) / np.sqrt(8)
    return rng.normal(mean, sd, n)


def make_items(n_dich=15, dich_model="2PL", n_poly=5, poly_model="GPCM", n_cat=4, b_mean=0.0, b_sd=1.0,
               a_meanlog=0.0, a_sdlog=0.25, c_range=(0.1, 0.25), d_range=(0.9, 0.99), step_spread=1.0, step_sd=0.3,
               D=1.0, rng=None):
    """Item parameters for a (possibly mixed-format) test. Each polytomous item has n_cat categories."""
    rng = rng or np.random.default_rng()
    items = []
    for j in range(n_dich):
        a = float(np.exp(rng.normal(a_meanlog, a_sdlog))) if dich_model != "Rasch" else 1.0
        c = float(rng.uniform(*c_range)) if dich_model in ("3PL", "4PL") else 0.0
        d = float(rng.uniform(*d_range)) if dich_model == "4PL" else 1.0
        items.append(make_item(f"D{j + 1:02d}", dich_model, a=a, b=rng.normal(b_mean, b_sd), c=c, d=d, D=D))
    m = n_cat - 1
    base = np.linspace(-step_spread, step_spread, m) if m > 1 else np.zeros(1)
    tau = base - base.mean()                      # RSM thresholds shared by all RSM items
    for j in range(n_poly):
        loc = rng.normal(b_mean, b_sd)
        a = float(np.exp(rng.normal(a_meanlog, a_sdlog))) if poly_model in ("GPCM", "GRM") else 1.0
        if poly_model == "RSM":
            items.append(make_item(f"P{j + 1:02d}", "RSM", b=loc, steps=tau, D=D))
            continue
        jitter = base + rng.normal(0, step_sd, m)
        if poly_model == "GRM":
            steps = np.sort(loc + jitter)
            steps = steps + np.arange(m) * 1e-3            # strictly increasing
        else:
            steps = loc + jitter
        items.append(make_item(f"P{j + 1:02d}", poly_model, a=a, steps=steps, D=D))
    return items


def simulate_responses(items, theta, missing_rate=0.0, rng=None):
    rng = rng or np.random.default_rng()
    Y = np.empty((len(theta), len(items)))
    for j, it in enumerate(items):
        cum = np.cumsum(it.probs(theta), axis=1)
        Y[:, j] = (rng.random(len(theta))[:, None] > cum[:, :-1]).sum(1)
    if missing_rate > 0:
        Y[rng.random(Y.shape) < missing_rate] = np.nan
        empty = np.isnan(Y).all(1)                    # keep at least one response per person
        if empty.any():
            j = rng.integers(0, len(items), empty.sum())
            Y[np.flatnonzero(empty), j] = 0
    return Y


def simulate(n_persons=300, seed=2026, theta_dist="normal", theta_mean=0.0, theta_sd=1.0, missing_rate=0.0, **item_kw):
    rng = np.random.default_rng(seed)
    items = make_items(rng=rng, **item_kw)
    theta = draw_theta(n_persons, theta_dist, theta_mean, theta_sd, rng)
    Y = simulate_responses(items, theta, missing_rate, rng)
    spec = dict(n_persons=n_persons, seed=seed, theta_dist=theta_dist, theta_mean=theta_mean, theta_sd=theta_sd,
                missing_rate=missing_rate, **item_kw)
    kinds = []
    if item_kw.get("n_dich", 15):
        kinds.append(f"{item_kw.get('n_dich', 15)} {item_kw.get('dich_model', '2PL')}")
    if item_kw.get("n_poly", 5):
        kinds.append(f"{item_kw.get('n_poly', 5)} {item_kw.get('poly_model', 'GPCM')}")
    return Dataset(f"Simulated · {' + '.join(kinds)} · N={n_persons}", items, Y, theta, spec=spec,
                   description=f"Simulated with seed {seed}; θ ~ {theta_dist}(mean {theta_mean}, SD {theta_sd}); "
                               f"missing rate {missing_rate:.0%}.")


def items_frame(items):
    rows = []
    for it in items:
        row = dict(id=it.id, model=it.model, a=it.a, b=it.b if it.model in DICHOTOMOUS + ("RSM",) else np.nan,
                   c=it.c if it.model in ("3PL", "4PL") else np.nan, d=it.d if it.model == "4PL" else np.nan,
                   steps=", ".join(f"{s:.3f}" for s in it.steps), categories=it.n_cat, D=it.D)
        rows.append(row)
    return rows


def items_from_frame(frame):
    """Inverse of items_frame (used by the item editor). Raises ValueError with readable messages."""
    items, problems = [], []
    for row in frame.to_dict("records"):
        steps = tuple(float(s) for s in str(row.get("steps") or "").replace(";", ",").split(",") if s.strip())
        val = lambda k, default: default if row.get(k) is None or (isinstance(row.get(k), float) and np.isnan(row.get(k))) else float(row[k])  # noqa: E731
        it = make_item(str(row["id"]), str(row["model"]), a=val("a", 1.0), b=val("b", 0.0), c=val("c", 0.0),
                       d=val("d", 1.0), steps=steps, D=val("D", 1.0))
        problems += [f"{it.id}: {p}" for p in it.problems()]
        items.append(it)
    if problems:
        raise ValueError("; ".join(problems))
    return items
