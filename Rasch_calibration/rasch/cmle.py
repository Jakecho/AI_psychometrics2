"""Educational conditional maximum likelihood (CMLE) for Rasch item difficulties.

The raw score r_i is sufficient for theta_i, so conditioning on it removes ability:

    P(x_i | r_i, b) = exp(-sum_{j in O_i} x_ij b_j) / gamma_{r_i}(b; O_i),

where gamma_r is the elementary symmetric function (ESF) of eps_j = exp(-b_j) over the
person's observed items O_i. ESFs are computed in log space by the recursion
    gamma_r^{(k)} = gamma_r^{(k-1)} + eps_k * gamma_{r-1}^{(k-1)},
never by enumerating 2^J patterns. A missing item has eps_j = 0 for that person, which is
exactly "condition on the observed item subset".

Identification: the conditional likelihood depends only on differences of b, so the sum
of difficulties is fixed at 0 (as in eRm::RM with sum0 = TRUE). Persons with r = 0 or
r = |O_i| have conditional probability 1 and carry no information; they are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import time

import numpy as np
from scipy.optimize import minimize

from .model import observed_values

NEG = -np.inf


def log_esf(log_eps):
    """log gamma_0..gamma_n for one or many rows of log eps (missing items = -inf).

    log_eps: (..., n). Returns (..., n + 1).
    """
    log_eps = np.atleast_1d(np.asarray(log_eps, float))
    shape = log_eps.shape[:-1]
    n = log_eps.shape[-1]
    g = np.full(shape + (n + 1,), NEG)
    g[..., 0] = 0.0
    with np.errstate(invalid="ignore"):
        for k in range(n):
            g[..., 1:] = np.logaddexp(g[..., 1:], log_eps[..., k, None] + g[..., :-1])
    return g


def log_esf_leave_one_out(log_eps):
    """log gamma^{(-j)}_r for every item j (each item removed in turn).

    log_eps: (P, n). Returns (P, n, n + 1). Item j is skipped by giving it eps = 0.
    """
    log_eps = np.asarray(log_eps, float)
    P, n = log_eps.shape
    loo = np.full((P, n, n + 1), NEG)
    loo[:, :, 0] = 0.0
    keep = ~np.eye(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        for k in range(n):
            update = np.logaddexp(loo[:, :, 1:], log_eps[:, k, None, None] + loo[:, :, :-1])
            loo[:, keep[:, k], 1:] = update[:, keep[:, k], :]
    return loo


@dataclass
class CMLEResult:
    b: np.ndarray
    se_b: np.ndarray
    cond_loglik: float
    converged: bool
    status: str
    iterations: int
    n_informative: int
    n_extreme: int
    history: list = field(default_factory=list)
    seconds: float = 0.0


class _Data:
    """Persons grouped by (observed item set, raw score), with frequencies; extremes removed.

    The conditional likelihood depends on a person's responses only through the item totals
    (sum x b) and gamma_r over the observed items, so persons who took the same items and have
    the same raw score share every ESF term. Grouping on (mask, r) rather than on whole
    response patterns gives at most (forms x scores) rows.
    """

    def __init__(self, X):
        x, mask = observed_values(X)
        r, n_obs = x.sum(axis=1), mask.sum(axis=1)
        informative = (r > 0) & (r < n_obs)
        self.n_extreme = int((~informative).sum())
        x, mask, r = x[informative], mask[informative], r[informative].astype(int)
        J = x.shape[1]
        key = np.concatenate([mask.astype(int), r[:, None]], axis=1)
        uniq, counts = np.unique(key, axis=0, return_counts=True)
        self.mask = uniq[:, :J].astype(bool)
        self.r = uniq[:, J].astype(int)
        self.w = counts.astype(float)
        self.J = J
        self.n_informative = int(informative.sum())
        self.item_totals = x.sum(axis=0)
        self.item_exposed = mask.sum(axis=0).astype(float)


def _cll_and_grad(b, d: _Data, chunk=512):
    """Conditional log-likelihood and its gradient with respect to b."""
    ll = -float(d.item_totals @ b)
    grad = -d.item_totals.copy()
    for start in range(0, len(d.w), chunk):
        sl = slice(start, start + chunk)
        le = np.where(d.mask[sl], -b[None, :], NEG)
        g = log_esf(le)                                   # (P, J+1)
        rr = d.r[sl]
        rows = np.arange(len(rr))
        lg_r = g[rows, rr]
        ll -= float(d.w[sl] @ lg_r)
        loo = log_esf_leave_one_out(le)                   # (P, J, J+1)
        # P(x_j = 1 | r) = eps_j * gamma^{(-j)}_{r-1} / gamma_r
        lp = le + loo[rows, :, rr - 1] - lg_r[:, None]
        pj = np.where(d.mask[sl], np.exp(lp), 0.0)
        grad += (d.w[sl, None] * pj).sum(axis=0)
    return ll, grad


def conditional_loglik(X, b):
    return _cll_and_grad(np.asarray(b, float), _Data(X))[0]


def cmle(X, tol=1e-7, max_iter=500, callback=None, fixed_b=None, compute_se=True):
    """Maximise the conditional likelihood subject to sum(b) = 0, or with some items fixed.

    Free parameters z (J - 1 of them) map to b = A z with the last difficulty equal to
    minus the sum of the others. L-BFGS-B uses the analytic gradient; standard errors come
    from the inverse of a finite-difference Hessian of that exact gradient.

    With `fixed_b` (a difficulty for each fixed item, NaN for free items) the fixed items set
    the scale: b = c + A z, where c holds the fixed values and A selects the free items.
    """
    started = time.perf_counter()
    d = _Data(X)
    J = d.J
    if d.n_informative < 2:
        raise ValueError("Fewer than two persons have non-extreme raw scores; CMLE has no information.")
    zero_items = d.item_totals == 0
    full_items = d.item_totals == d.item_exposed
    if zero_items.any() or full_items.any():
        raise ValueError("Among non-extreme persons, some item is answered all-correct or all-incorrect; "
                         "its conditional MLE is infinite. Remove the item or add data.")
    fixed = np.zeros(J, bool) if fixed_b is None else np.isfinite(np.asarray(fixed_b, float))
    if fixed_b is not None and np.shape(fixed_b) != (J,):
        raise ValueError("fixed_b needs one entry per item (NaN = free).")
    if fixed.all():
        raise ValueError("At least one item must be free.")
    if fixed.any():
        free = np.flatnonzero(~fixed)
        A = np.zeros((J, len(free)))
        A[free, np.arange(len(free))] = 1.0
        c = np.where(fixed, np.asarray(fixed_b, float), 0.0)
    else:
        A = np.vstack([np.eye(J - 1), -np.ones((1, J - 1))])
        c = np.zeros(J)
    k_free = A.shape[1]
    history = []

    def objective(z):
        ll, g = _cll_and_grad(c + A @ z, d)
        return -ll, -(A.T @ g)

    p0 = np.clip(d.item_totals / d.item_exposed, 1e-3, 1 - 1e-3)
    b0 = -np.log(p0 / (1 - p0))
    if fixed.any():
        b0 += np.mean(c[fixed] - b0[fixed])
        z0 = b0[~fixed]
    else:
        b0 -= b0.mean()
        z0 = b0[:-1]
    history.append(dict(iteration=0, b=c + A @ z0, cond_loglik=-objective(z0)[0], max_change=np.nan,
                        grad_norm=float(np.max(np.abs(objective(z0)[1])))))

    def record(zk):
        b_k = c + A @ zk
        val, gr = objective(zk)
        history.append(dict(iteration=len(history), b=b_k, cond_loglik=-val,
                            max_change=float(np.max(np.abs(b_k - history[-1]["b"]))),
                            grad_norm=float(np.max(np.abs(gr)))))
        if callback:
            callback(history[-1])

    opt = minimize(objective, z0, jac=True, method="L-BFGS-B", callback=record,
                   options=dict(maxiter=max_iter, ftol=1e-15, gtol=tol))
    b = c + A @ opt.x
    # Observed information: finite differences of the exact gradient in the free parameters.
    h = 1e-5
    se = np.full(J, np.nan)
    if compute_se:
        H = np.zeros((k_free, k_free))
        for k in range(k_free):
            e = np.zeros(k_free)
            e[k] = h
            H[:, k] = (objective(opt.x + e)[1] - objective(opt.x - e)[1]) / (2 * h)
        H = (H + H.T) / 2
        try:
            cov = A @ np.linalg.inv(H) @ A.T
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            pass
    se[fixed] = np.nan
    grad_max = float(np.max(np.abs(objective(opt.x)[1])))
    # Convergence is judged on the gradient itself; the optimizer's own message is kept for reference
    # (L-BFGS-B can report a line-search stop once the gradient is already at numerical zero).
    converged = bool(grad_max < 1e-4)
    status = (f"Converged: max |conditional score| = {grad_max:.1e}" if converged else
              f"Not converged: max |conditional score| = {grad_max:.1e}") + f" · optimizer: {opt.message}"
    return CMLEResult(b=b, se_b=se, cond_loglik=-float(opt.fun), converged=converged,
                      status=status,
                      iterations=int(opt.nit), n_informative=d.n_informative, n_extreme=d.n_extreme,
                      history=history, seconds=time.perf_counter() - started)


def displacement(X, b, h=1e-5):
    """One Newton step each item would take if freed: conditional score / conditional information."""
    d = _Data(X)
    b = np.asarray(b, float)
    g = _cll_and_grad(b, d)[1]
    out = np.empty(len(b))
    for j in range(len(b)):
        e = np.zeros(len(b))
        e[j] = h
        info = -(_cll_and_grad(b + e, d)[1][j] - _cll_and_grad(b - e, d)[1][j]) / (2 * h)
        out[j] = g[j] / info
    return out


def demonstration(b_small, r):
    """Small-example table (n <= 6 items): every pattern with raw score r.

    Shows numerator exp(-sum x_j b_j), the ESF denominator from the recursion, and the
    resulting conditional probabilities, which sum to one.
    """
    b_small = np.asarray(b_small, float)
    n = len(b_small)
    if n > 6:
        raise ValueError("The enumeration demonstration is limited to 6 items.")
    eps = np.exp(-b_small)
    gamma = np.exp(log_esf(np.log(eps)))
    rows = []
    for combo in combinations(range(n), r):
        pattern = np.zeros(n, int)
        pattern[list(combo)] = 1
        numer = float(np.prod(eps[list(combo)])) if combo else 1.0
        rows.append(dict(pattern="".join(map(str, pattern)), numerator=numer,
                         conditional_probability=numer / gamma[r]))
    # Polynomial check: coefficients of prod_j (1 + t * eps_j).
    poly = np.array([1.0])
    for e in eps:
        poly = np.convolve(poly, [1.0, e])
    return dict(rows=rows, gamma=gamma, polynomial=poly, eps=eps)
