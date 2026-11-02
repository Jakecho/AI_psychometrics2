"""Item definition and the quantities every scoring method needs.

An `Item` holds one item's model and fixed parameters. Category k = 0..m scores k points.
For any θ (scalar or array) the module returns, per category,

    P_k(θ),  d log P_k / dθ,  d² log P_k / dθ²

from which all derived quantities follow:

    P'_k  = P_k · dlogP_k
    P''_k = P_k · (d2logP_k + dlogP_k²)
    I(θ)  = Σ_k P'_k² / P_k      = Σ_k P_k dlogP_k²            (Fisher information)
    J(θ)  = Σ_k P'_k P''_k / P_k = Σ_k P_k dlogP_k (d2logP_k + dlogP_k²)   (Warm's J)
    E(θ)  = Σ_k k P_k            (expected item score)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import dichotomous, grm, partial_credit

DICHOTOMOUS = ("Rasch", "2PL", "3PL", "4PL")
POLYTOMOUS = ("PCM", "GPCM", "RSM", "GRM")
MODELS = DICHOTOMOUS + POLYTOMOUS
RASCH_FAMILY = ("Rasch", "PCM", "RSM")
# Models whose log-likelihood is an exponential family in θ (observed = expected information).
EXPONENTIAL_FAMILY = ("Rasch", "2PL", "PCM", "GPCM", "RSM")
# Models with J(θ) = dI/dθ, so Warm's WLE equals the maximiser of L·√I: the exponential family and also the GRM
# (Σ_k P_k dlogP_k d²logP_k telescopes to 0 for the GRM). Only items with asymptotes (3PL, 4PL) break it.
J_EQUALS_IPRIME = EXPONENTIAL_FAMILY + ("GRM",)
MODEL_NAMES = {"Rasch": "Rasch / 1PL", "2PL": "2PL", "3PL": "3PL", "4PL": "4PL",
               "PCM": "Partial credit (PCM)", "GPCM": "Generalized partial credit (GPCM)",
               "RSM": "Rating scale (RSM)", "GRM": "Graded response (GRM)"}


@dataclass(frozen=True)
class Item:
    """One item with known parameters.

    a: discrimination (slope); 1 for Rasch, PCM, RSM.
    b: difficulty (dichotomous) or location (RSM). Unused by PCM, GPCM and GRM.
    c: lower asymptote (3PL/4PL), d: upper asymptote (4PL).
    steps: PCM/GPCM step difficulties δ_v, GRM thresholds b_k, or RSM thresholds τ_v.
    D: logistic scaling constant (1 by default; 1.702 approximates the normal ogive).
    """
    id: str
    model: str
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    steps: tuple = field(default_factory=tuple)
    D: float = 1.0

    @property
    def n_cat(self):
        return 2 if self.model in DICHOTOMOUS else len(self.steps) + 1

    @property
    def max_score(self):
        return self.n_cat - 1

    @property
    def deltas(self):
        """Step difficulties of a divide-by-total item (RSM: b + τ_v)."""
        if self.model == "RSM":
            return tuple(self.b + t for t in self.steps)
        return tuple(self.steps)

    def problems(self):
        """Constraint violations as plain sentences (empty list = valid)."""
        out = []
        if self.model not in MODELS:
            return [f"unknown model {self.model!r}"]
        if not self.a > 0:
            out.append("discrimination a must be positive")
        if self.model in RASCH_FAMILY and abs(self.a - 1.0) > 1e-12:
            out.append(f"{self.model} fixes a = 1")
        if self.model in ("Rasch", "2PL") and (self.c != 0 or self.d != 1):
            out.append(f"{self.model} fixes c = 0 and d = 1")
        if self.model == "3PL" and self.d != 1:
            out.append("3PL fixes d = 1")
        if self.model in ("3PL", "4PL") and not (0 <= self.c < self.d <= 1):
            out.append("need 0 ≤ c < d ≤ 1")
        if self.model in POLYTOMOUS and len(self.steps) < 1:
            out.append("a polytomous item needs at least one step/threshold")
        if self.model == "GRM" and np.any(np.diff(self.steps) <= 0):
            out.append("GRM thresholds must be strictly increasing")
        if not np.all(np.isfinite([self.a, self.b, self.c, self.d, self.D, *self.steps])):
            out.append("parameters must be finite")
        return out

    def with_(self, **kw):
        return replace(self, **kw)

    # ---- category-level functions -------------------------------------------------
    def log_probs(self, theta):
        if self.model in DICHOTOMOUS:
            return dichotomous.log_probs(theta, self.a, self.b, self.c, self.d, self.D)
        if self.model == "GRM":
            return grm.log_probs(theta, self.a, self.steps, self.D)
        return partial_credit.log_probs(theta, self.a, self.deltas, self.D)

    def probs(self, theta):
        return np.exp(self.log_probs(theta))

    def log_derivs(self, theta):
        """(P, d log P/dθ, d² log P/dθ²), each of shape (..., n_cat)."""
        if self.model in DICHOTOMOUS:
            return dichotomous.log_derivs(theta, self.a, self.b, self.c, self.d, self.D)
        if self.model == "GRM":
            return grm.log_derivs(theta, self.a, self.steps, self.D)
        return partial_credit.log_derivs(theta, self.a, self.deltas, self.D)

    def derivatives(self, theta):
        """(P, P', P'') per category."""
        P, dl, d2l = self.log_derivs(theta)
        return P, P * dl, P * (d2l + dl ** 2)

    def info(self, theta):
        P, dl, _ = self.log_derivs(theta)
        return (P * dl ** 2).sum(-1)

    def warm_j(self, theta):
        P, dl, d2l = self.log_derivs(theta)
        return (P * dl * (d2l + dl ** 2)).sum(-1)

    def expected(self, theta):
        return (self.probs(theta) * np.arange(self.n_cat)).sum(-1)

    def score_variance(self, theta):
        P = self.probs(theta)
        k = np.arange(self.n_cat)
        E = (P * k).sum(-1, keepdims=True)
        return (P * (k - E) ** 2).sum(-1)

    def expected_slope(self, theta):
        """dE/dθ = Σ_k k P'_k."""
        _, dP, _ = self.derivatives(theta)
        return (dP * np.arange(self.n_cat)).sum(-1)

    def limits(self):
        """Expected score as θ → −∞ and θ → +∞."""
        if self.model in DICHOTOMOUS:
            return self.c, self.d
        return 0.0, float(self.max_score)

    def describe(self):
        if self.model in DICHOTOMOUS:
            parts = [f"a={self.a:.3g}", f"b={self.b:.3g}"]
            if self.model in ("3PL", "4PL"):
                parts.append(f"c={self.c:.3g}")
            if self.model == "4PL":
                parts.append(f"d={self.d:.3g}")
        else:
            name = {"GRM": "thresholds", "RSM": "τ"}.get(self.model, "steps")
            parts = ([f"a={self.a:.3g}"] if self.model in ("GPCM", "GRM") else []) + \
                    ([f"b={self.b:.3g}"] if self.model == "RSM" else []) + \
                    [f"{name}=(" + ", ".join(f"{s:.3g}" for s in self.steps) + ")"]
        return f"{self.model}: " + ", ".join(parts)


def make_item(id, model, a=1.0, b=0.0, c=0.0, d=1.0, steps=(), D=1.0):
    if model in RASCH_FAMILY:
        a = 1.0
    if model in ("Rasch", "2PL"):
        c, d = 0.0, 1.0
    if model == "3PL":
        d = 1.0
    if model in DICHOTOMOUS:
        steps = ()
    return Item(id=id, model=model, a=float(a), b=float(b), c=float(c), d=float(d),
                steps=tuple(float(s) for s in steps), D=float(D))


# ---- pattern-level quantities (a set of items and one response vector per person) ---------

def observed_mask(Y):
    return ~np.isnan(np.asarray(Y, float))


def pattern_terms(items, Y, theta):
    """Sums over each person's observed items at that person's θ.

    Y: (N, J) responses with NaN for missing; theta: (N,).
    Returns dict of (N,) arrays: loglik, score (ℓ'), hessian (ℓ''), info (I), J, Iprime (dI/dθ),
    expected (T), variance (Σ Var), slope (T'), and raw (score on observed items).
    dI/dθ = Σ_k P_k dlogP_k (dlogP_k² + 2 d2logP_k); it equals J except for 3PL/4PL items.
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    out = {k: np.zeros(len(theta)) for k in ("loglik", "score", "hessian", "info", "J", "expected", "variance",
                                             "slope", "raw", "Iprime")}
    for j, item in enumerate(items):
        y = Y[:, j]
        obs = ~np.isnan(y)
        if not obs.any():
            continue
        P, dl, d2l = item.log_derivs(theta[obs])
        lp = item.log_probs(theta[obs])
        k = y[obs].astype(int)
        rows = np.arange(len(k))
        kk = np.arange(item.n_cat)
        E = (P * kk).sum(-1)
        out["loglik"][obs] += lp[rows, k]
        out["score"][obs] += dl[rows, k]
        out["hessian"][obs] += d2l[rows, k]
        out["info"][obs] += (P * dl ** 2).sum(-1)
        out["J"][obs] += (P * dl * (d2l + dl ** 2)).sum(-1)
        out["expected"][obs] += E
        out["variance"][obs] += (P * (kk - E[:, None]) ** 2).sum(-1)
        out["slope"][obs] += (P * dl * kk).sum(-1)
        out["Iprime"][obs] += (P * dl * (dl ** 2 + 2 * d2l)).sum(-1)
        out["raw"][obs] += k
    return out


def item_table(items, y, theta):
    """Item-by-item calculations for one response vector at one θ (educational tables)."""
    rows = []
    theta = float(theta)
    for item, yj in zip(items, y):
        P, dl, d2l = item.log_derivs(np.array([theta]))
        P, dl, d2l = P[0], dl[0], d2l[0]
        k = np.arange(item.n_cat)
        row = dict(item=item.id, model=item.model)
        if np.isnan(yj):
            row.update(response="missing")
            rows.append(row)
            continue
        y_int = int(yj)
        row.update(response=y_int, P_observed=P[y_int], log_P=float(item.log_probs(np.array([theta]))[0, y_int]),
                   score_contrib=dl[y_int], hessian_contrib=d2l[y_int],
                   information=float((P * dl ** 2).sum()), J_contrib=float((P * dl * (d2l + dl ** 2)).sum()),
                   expected_score=float((P * k).sum()))
        rows.append(row)
    return rows


def category_table(items, theta):
    """Every category probability of every item at θ."""
    rows = []
    for item in items:
        P = item.probs(np.array([float(theta)]))[0]
        for k, p in enumerate(P):
            rows.append(dict(item=item.id, model=item.model, category=k, probability=p))
    return rows


def loglik_grid(items, Y, grid):
    """ℓ_i(θ_q) for every person i and grid point q: shape (N, Q). Missing responses are skipped."""
    Y = np.atleast_2d(np.asarray(Y, float))
    grid = np.asarray(grid, float)
    out = np.zeros((Y.shape[0], len(grid)))
    for j, item in enumerate(items):
        lp = item.log_probs(grid)                        # (Q, K)
        y = Y[:, j]
        obs = ~np.isnan(y)
        out[obs] += lp[:, y[obs].astype(int)].T
    return out


def extreme_status(items, Y):
    """'min' / 'max' / '' per person: all observed responses in the lowest / highest category."""
    Y = np.atleast_2d(np.asarray(Y, float))
    lo = np.zeros(Y.shape)
    hi = np.array([[it.max_score for it in items]] * Y.shape[0], float)
    obs = ~np.isnan(Y)
    n_obs = obs.sum(1)
    at_min = ((Y == lo) | ~obs).all(1) & (n_obs > 0)
    at_max = ((Y == hi) | ~obs).all(1) & (n_obs > 0)
    return np.where(at_max, "max", np.where(at_min, "min", ""))
