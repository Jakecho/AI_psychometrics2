"""Polytomous IRT models: PCM, RSM, GPCM and GRM.

Canonical parameterisation used everywhere in this lab (logistic scale D, default D = 1):

* PCM   P_k ∝ exp[ D Σ_{h≤k} (θ − δ_h) ]                      a fixed at 1
* RSM   δ_h = b + τ_h with Σ_h τ_h = 0 (τ shared by items with the same number of categories)
* GPCM  P_k ∝ exp[ D a Σ_{h≤k} (θ − δ_h) ],  a > 0
* GRM   P(Y ≥ k) = logistic[ D a (θ − b_k) ],  b_1 < … < b_K

`Item.d` holds the K = ncat − 1 step difficulties δ_h (PCM, RSM, GPCM) or ordered thresholds b_k (GRM).
All probabilities are computed on the log scale (log-sum-exp / log-sigmoid), so extreme θ or thresholds do
not underflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import expit, log_expit, logsumexp

MODELS = ("PCM", "RSM", "GPCM", "GRM")
RASCH = frozenset({"PCM", "RSM"})
ADJACENT = frozenset({"PCM", "RSM", "GPCM"})
MODEL_NAMES = dict(PCM="Partial Credit Model", RSM="Rating Scale Model",
                   GPCM="Generalized Partial Credit Model", GRM="Graded Response Model")
LOG_FLOOR = -700.0


@dataclass
class Item:
    """One polytomous item. `d` = step difficulties (adjacent-category models) or thresholds (GRM)."""
    id: str
    model: str
    a: float = 1.0
    d: tuple = ()
    b: float | None = None            # RSM location
    tau: tuple | None = None          # RSM shared steps (sum to zero)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model {self.model!r}; choose one of {MODELS}.")
        if self.model == "RSM":
            if self.b is None or self.tau is None:
                raise ValueError(f"RSM item {self.id} needs a location b and shared steps tau.")
            tau = np.asarray(self.tau, float)
            if abs(tau.sum()) > 1e-8:
                raise ValueError(f"RSM item {self.id}: steps tau must sum to zero (sum = {tau.sum():.3g}).")
            self.tau = tuple(float(t) for t in tau)
            self.b = float(self.b)
            self.d = tuple(float(self.b + t) for t in tau)
        self.d = tuple(float(v) for v in self.d)
        self.a = float(self.a)
        if len(self.d) < 1:
            raise ValueError(f"Item {self.id} needs at least two categories.")
        if self.model in RASCH and abs(self.a - 1) > 1e-12:
            raise ValueError(f"{self.model} item {self.id}: discrimination is fixed at 1.")
        if self.a <= 0:
            raise ValueError(f"Item {self.id}: discrimination must be positive.")
        if self.model == "GRM" and np.any(np.diff(self.d) <= 0):
            raise ValueError(f"GRM item {self.id}: thresholds must be strictly increasing.")

    @property
    def ncat(self):
        return len(self.d) + 1

    @property
    def K(self):
        return len(self.d)

    def location(self):
        return float(np.mean(self.d))

    def vector(self):
        """Canonical comparable vector (a, d_1..d_K) used for change and recovery statistics."""
        return np.r_[self.a, self.d]

    def labels(self):
        return ["a"] + [("b" if self.model == "GRM" else "δ") + str(h + 1) for h in range(self.K)]

    def as_row(self):
        row = dict(item=self.id, model=self.model, categories=self.ncat, a=self.a)
        for h, v in enumerate(self.d):
            row[("b" if self.model == "GRM" else "δ") + str(h + 1)] = v
        if self.model == "RSM":
            row["b (location)"] = self.b
            row.update({f"τ{h + 1}": t for h, t in enumerate(self.tau)})
        return row

    def with_params(self, **kw):
        return replace(self, **kw)


def make_item(id, model, a=1.0, d=None, b=None, tau=None):
    """Convenience constructor accepting RSM as (b, tau) or as step vector d (then split into b + τ)."""
    if model == "RSM" and (b is None or tau is None):
        d = np.asarray(d, float)
        b, tau = float(d.mean()), tuple(d - d.mean())
    return Item(id=str(id), model=model, a=1.0 if model in RASCH else float(a),
                d=tuple(d) if d is not None else (), b=b, tau=tau)


# ---- probabilities -------------------------------------------------------------------------------

def adjacent_logits(theta, a, d, D=1.0):
    """z_k = D a (k θ − Σ_{h≤k} δ_h), z_0 = 0. Shape (Q, K+1)."""
    theta = np.atleast_1d(np.asarray(theta, float))
    d = np.asarray(d, float)
    k = np.arange(len(d) + 1)
    cum = np.r_[0.0, np.cumsum(d)]
    return D * a * (theta[:, None] * k[None, :] - cum[None, :])


def log_probs_adjacent(theta, a, d, D=1.0):
    z = adjacent_logits(theta, a, d, D)
    return z - logsumexp(z, axis=1, keepdims=True)


def log_probs_grm(theta, a, b, D=1.0):
    """log P(Y = k) from cumulative boundaries; stable difference of logistic functions."""
    theta = np.atleast_1d(np.asarray(theta, float))
    b = np.asarray(b, float)
    x = D * a * (theta[:, None] - b[None, :])            # (Q, K): logit of P(Y ≥ k)
    K = len(b)
    out = np.empty((len(theta), K + 1))
    out[:, 0] = log_expit(-x[:, 0])
    out[:, K] = log_expit(x[:, K - 1])
    for k in range(1, K):
        gap = x[:, k] - x[:, k - 1]                        # = −D a (b_{k+1} − b_k) < 0
        out[:, k] = log_expit(x[:, k - 1]) + log_expit(-x[:, k]) + np.log(-np.expm1(np.minimum(gap, -1e-300)))
    return np.maximum(out, LOG_FLOOR)


def log_probs(item: Item, theta, D=1.0):
    if item.model == "GRM":
        return log_probs_grm(theta, item.a, item.d, D)
    return np.maximum(log_probs_adjacent(theta, item.a, item.d, D), LOG_FLOOR)


def probs(item: Item, theta, D=1.0):
    return np.exp(log_probs(item, theta, D))


def cumulative_probs(item: Item, theta, D=1.0):
    """P(Y ≥ k), k = 1..K, for any model (GRM: the boundary curves themselves)."""
    p = probs(item, theta, D)
    return np.cumsum(p[:, ::-1], axis=1)[:, ::-1][:, 1:]


def expected_score(item: Item, theta, D=1.0):
    p = probs(item, theta, D)
    return p @ np.arange(item.ncat)


def information(item: Item, theta, D=1.0):
    """Fisher information I(θ) = Σ_k (∂P_k/∂θ)² / P_k (exact for every model)."""
    theta = np.atleast_1d(np.asarray(theta, float))
    p = probs(item, theta, D)
    k = np.arange(item.ncat)
    if item.model in ADJACENT:
        mean = p @ k
        return (D * item.a) ** 2 * (p @ k ** 2 - mean ** 2)
    x = D * item.a * (theta[:, None] - np.asarray(item.d)[None, :])
    w = expit(x) * expit(-x) * D * item.a                       # ∂P*_k/∂θ
    w = np.c_[np.zeros(len(theta)), w, np.zeros(len(theta))]
    dp = w[:, :-1] - w[:, 1:]
    return np.sum(dp ** 2 / np.maximum(p, 1e-300), axis=1)


def adjacent_odds(item: Item, theta, D=1.0):
    """log[P_k / P_{k−1}] for k = 1..K (equals D a (θ − δ_k) for adjacent-category models)."""
    lp = log_probs(item, theta, D)
    return lp[:, 1:] - lp[:, :-1]


def sample_responses(items, theta, rng, D=1.0):
    """Draw categories by inverse-CDF sampling. theta: (N,). Returns int array (N, J)."""
    theta = np.asarray(theta, float)
    out = np.empty((len(theta), len(items)), int)
    u = rng.random((len(theta), len(items)))
    for j, it in enumerate(items):
        cdf = np.cumsum(probs(it, theta, D), axis=1)
        out[:, j] = (u[:, [j]] > cdf[:, :-1]).sum(axis=1)
    return out


# ---- objective pieces for the item M-step ----------------------------------------------------------
# C is an expected-count array (Q, K+1): C[q, k] = expected number of examinees at node q answering k.

def adjacent_objective(theta, C, a, d, D=1.0):
    """Q_j = Σ_qk C log P, its gradient with respect to log a and each δ_h."""
    z = adjacent_logits(theta, a, d, D)
    lp = z - logsumexp(z, axis=1, keepdims=True)
    P = np.exp(lp)
    n = C.sum(axis=1, keepdims=True)
    R = C - n * P                                           # ∂Q/∂z_k
    tail = np.cumsum(R[:, ::-1], axis=1)[:, ::-1][:, 1:]   # Σ_{k≥h} R_k for h = 1..K
    g_d = -D * a * tail.sum(axis=0)
    g_loga = float(np.sum(R * z))
    return float(np.sum(C * np.maximum(lp, LOG_FLOOR))), g_loga, g_d


def grm_objective(theta, C, a, b, D=1.0):
    """Q_j for the GRM with gradients with respect to a and each threshold b_k."""
    theta = np.atleast_1d(theta)
    b = np.asarray(b, float)
    lp = log_probs_grm(theta, a, b, D)
    P = np.exp(lp)
    r = C / np.maximum(P, 1e-300)                            # C_k / P_k
    x = D * a * (theta[:, None] - b[None, :])
    W = expit(x) * expit(-x)                                 # P*(1 − P*)
    g_b = D * a * np.sum(W * (r[:, :-1] - r[:, 1:]), axis=0)
    dPa = D * (theta[:, None] - b[None, :]) * W             # ∂P*_k/∂a
    dPa = np.c_[np.zeros(len(theta)), dPa, np.zeros(len(theta))]
    g_a = float(np.sum(r * (dPa[:, :-1] - dPa[:, 1:])))
    return float(np.sum(C * lp)), g_a, g_b


def item_objective(item: Item, theta, C, D=1.0):
    if item.model == "GRM":
        return grm_objective(theta, C, item.a, item.d, D)[0]
    return adjacent_objective(theta, C, item.a, item.d, D)[0]


# ---- parameter conversions -------------------------------------------------------------------------

def to_intercepts(item: Item, D=1.0):
    """Slope-intercept form used by mirt.

    Adjacent-category models: P_k ∝ exp(k·α θ + γ_k), α = D a, γ_k = −D a Σ_{h≤k} δ_h  (γ_0 = 0).
    GRM: P(Y ≥ k) = logistic(α θ + c_k), c_k = −D a b_k (mirt's d_k)."""
    if item.model == "GRM":
        return dict(alpha=D * item.a, intercepts=tuple(-D * item.a * np.asarray(item.d)))
    gam = -D * item.a * np.cumsum(item.d)
    return dict(alpha=D * item.a, intercepts=(0.0,) + tuple(gam))


def from_intercepts(id, model, alpha, intercepts, D=1.0):
    """Inverse of to_intercepts (returns a canonical Item)."""
    a = alpha / D
    c = np.asarray(intercepts, float)
    if model == "GRM":
        return make_item(id, model, a, tuple(-c / (D * a)))
    gam = c[1:] - c[0]
    d = -np.diff(np.r_[0.0, gam]) / (D * a)
    return make_item(id, model, a, tuple(d))


def to_tam(item: Item):
    """TAM's PCM/RSM parameters (D = 1). PCM: xsi_{jh} = δ_jh. RSM: item location b_j and steps τ_h."""
    if item.model == "RSM":
        return dict(xsi_item=item.b, tau=item.tau)
    return dict(xsi_steps=item.d)


def to_irtq(item: Item):
    """irtQ shape_df row: GPCM/PCM use d = step difficulties, GRM uses d = thresholds (same as canonical)."""
    if item.model == "RSM":
        raise ValueError("irtQ has no RSM; an RSM item can be written as a PCM item with δ_h = b + τ_h.")
    return dict(model="GRM" if item.model == "GRM" else "GPCM", a=item.a, d=item.d)


def to_mirt(item: Item, D=1.0):
    """mirt 'gpcm' / 'Rasch' (polytomous) / 'graded' coefficient names."""
    ic = to_intercepts(item, D)
    if item.model == "GRM":
        return {"a1": ic["alpha"], **{f"d{k + 1}": v for k, v in enumerate(ic["intercepts"])}}
    return {"a1": ic["alpha"], **{f"d{k}": v for k, v in enumerate(ic["intercepts"])}}


def reverse_params(item: Item):
    """Utility used in tests: parameters packed for re-creating the item."""
    return dict(id=item.id, model=item.model, a=item.a, d=item.d, b=item.b, tau=item.tau)


@dataclass
class ItemSpec:
    """Item structure in a dataset (no parameters): id, number of categories, optional RSM scale group."""
    id: str
    ncat: int
    scale: str = field(default="")

    def __post_init__(self):
        self.id = str(self.id)
        self.ncat = int(self.ncat)
        if self.ncat < 2:
            raise ValueError(f"Item {self.id} needs at least two categories.")
        if not self.scale:
            self.scale = f"{self.ncat}-category scale"
