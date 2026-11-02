"""Test characteristic curve (TCC) inversion.

    T(θ) = Σ_{j∈O_i} Σ_k k P_jk(θ)          expected summed score on the person's observed items
    solve T(θ̂) = r_i                       (r_i: observed summed score)

T is strictly increasing when every slope a_j > 0, so a root is unique when it exists; it exists
only for T(−∞) < r < T(+∞), where T(−∞) = Σ c_j (lower asymptotes of 3PL/4PL items, 0 otherwise)
and T(+∞) = Σ (d_j or m_j). Scores outside that interval have no finite TCC estimate.

Uncertainty (approximation): delta method SE = √(Σ_j Var_j(θ̂)) / T'(θ̂). This is not irtQ's
INV.TCC SE (irtQ reports a summed-score posterior-type spread), so only θ is compared with irtQ.
"""
from __future__ import annotations

import time

import numpy as np

from irt_models import pattern_terms

from .result import ScoreResult, skip_empty
from .solvers import find_root


def tcc_limits(items, Y):
    Y = np.atleast_2d(np.asarray(Y, float))
    obs = ~np.isnan(Y)
    lo = np.array([it.limits()[0] for it in items])
    hi = np.array([it.limits()[1] for it in items])
    return (obs * lo).sum(1), (obs * hi).sum(1)


@skip_empty
def tcc_inverse(items, Y, settings, start=None, record=False):
    t0 = time.perf_counter()
    Y = np.atleast_2d(np.asarray(Y, float))
    N = Y.shape[0]
    r = np.nansum(Y, axis=1)

    def fun(theta, subset):
        t = pattern_terms(items, Y[subset], theta[subset])
        return dict(U=r[subset] - t["expected"], dU=-t["slope"], expected=t["expected"], slope=t["slope"])

    start = np.full(N, settings.start) if start is None else np.broadcast_to(np.asarray(start, float), (N,)).copy()
    theta, status, n_iter, hist = find_root(fun, start, settings, record)
    lo, hi = tcc_limits(items, Y)
    t = pattern_terms(items, Y, theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(t["variance"]) / t["slope"]
    estimate = theta.copy()
    words = np.array(["converged" if s == "converged" else s for s in status], dtype=object)
    below, above = r <= lo + 1e-12, r >= hi - 1e-12
    estimate[below], estimate[above] = -np.inf, np.inf
    se[below | above] = np.nan
    words[below] = [f"no finite solution: r = {ri:g} ≤ T(−∞) = {l:.3g}" for ri, l in zip(r[below], lo[below])]
    words[above] = [f"no finite solution: r = {ri:g} ≥ T(+∞) = {h:.3g}" for ri, h in zip(r[above], hi[above])]
    oob = np.isin(status, ["no root above the lower bound", "no root below the upper bound"]) & ~(below | above)
    words[oob] = "root lies outside the search range"
    estimate[oob] = np.sign(theta[oob]) * np.inf
    return ScoreResult("TCC", estimate, se, words, n_iter, "SE ≈ √ΣVar / T′ (delta method)",
                       extra=dict(raw=r, tcc_min=lo, tcc_max=hi, theta_bounded=theta), history=hist,
                       seconds=time.perf_counter() - t0)


def monotonicity(items, grid):
    """T'(θ) on a grid for the complete test: (all positive?, minimum slope)."""
    y = np.zeros(len(items))
    t = pattern_terms(items, np.tile(y, (len(grid), 1)), grid)
    return bool((t["slope"] > 0).all()), float(t["slope"].min())
