"""Advanced scoring methods (optional modules).

posterior_median   — 50% point of the posterior (fine grid, piecewise-linear CDF).
plausible_values   — random draws from each person's posterior, for population-level analysis
                     only; they are not point estimates for individual reporting.
prox               — Rasch PROX normal approximation (Wright & Stone 1979), dichotomous Rasch only:
                     θ ≈ b̄ + √(1 + s_b²/2.89) · ln(r / (n − r)).
adjusted_mle       — MLE, except for patterns at the lowest/highest score, which are scored by
                     solving T(θ) = r ± ε on the observed items (ε = 0.3 by default, the common
                     "extreme score adjustment"). For Rasch-family items this is the MLE at the
                     fractional score; for other models it is a TCC-based value. It is a
                     convention, not an estimate the data identify.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from irt_models import loglik_grid, pattern_terms

from .eap import posterior_quantiles
from .mle import mle
from .result import ScoreResult
from .solvers import find_root


def posterior_median(items, Y, settings):
    q = posterior_quantiles(items, Y, settings, probs=(0.025, 0.5, 0.975))
    return ScoreResult("Posterior median", q[:, 1], (q[:, 2] - q[:, 0]) / (2 * 1.959964),
                       np.array(["fine-grid posterior"] * len(q), dtype=object), np.zeros(len(q), int),
                       "95% interval half-width / 1.96", extra=dict(lower=q[:, 0], upper=q[:, 2]))


def plausible_values(items, Y, settings, draws=5, seed=1, n_grid=801):
    """(N, draws) array: inverse-CDF draws from a piecewise-constant posterior on a fine grid."""
    rng = np.random.default_rng(seed)
    mu, sd = settings.prior_mean, settings.prior_sd
    grid = np.linspace(mu - 8 * sd, mu + 8 * sd, n_grid)
    h = grid[1] - grid[0]
    lp = loglik_grid(items, np.atleast_2d(Y), grid) + norm.logpdf(grid, mu, sd)
    post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    cdf = np.cumsum(post, axis=1)
    u = rng.random((post.shape[0], draws))
    out = np.empty_like(u)
    for i in range(post.shape[0]):
        k = np.searchsorted(cdf[i], u[i])
        k = np.minimum(k, n_grid - 1)
        out[i] = grid[k] + (rng.random(draws) - 0.5) * h      # uniform within the grid cell
    return out


def prox(items, Y):
    """Rasch PROX. Returns estimates (±inf for extreme scores) or raises for non-Rasch items."""
    if any(it.model != "Rasch" for it in items):
        raise ValueError("PROX is defined here for dichotomous Rasch items only.")
    Y = np.atleast_2d(np.asarray(Y, float))
    b = np.array([it.b for it in items])
    obs = ~np.isnan(Y)
    est = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        bb = b[obs[i]]
        n, r = len(bb), np.nansum(Y[i])
        if n == 0:
            continue
        if r <= 0 or r >= n:
            est[i] = -np.inf if r <= 0 else np.inf
            continue
        est[i] = bb.mean() + np.sqrt(1 + bb.var() / 2.89) * np.log(r / (n - r))
    return est


def adjusted_mle(items, Y, settings, epsilon=0.3):
    Y = np.atleast_2d(np.asarray(Y, float))
    res = mle(items, Y, settings)
    ext = res.extra["extreme"]
    theta, se, status = res.theta.copy(), res.se.copy(), res.status.copy()
    rows = np.flatnonzero(ext != "")
    if len(rows):
        obs = ~np.isnan(Y[rows])
        hi = np.array([[it.max_score for it in items]] * len(rows)) * obs
        target = np.where(ext[rows] == "max", hi.sum(1) - epsilon, epsilon)
        Ysub = Y[rows]

        def fun(th, subset):
            t = pattern_terms(items, Ysub[subset], th[subset])
            return dict(U=target[subset] - t["expected"], dU=-t["slope"])

        th, st, _, _ = find_root(fun, np.zeros(len(rows)), settings, record=False)
        theta[rows] = th
        se[rows] = 1 / np.sqrt(pattern_terms(items, Ysub, th)["info"])
        status[rows] = [f"extreme score: solved T(θ) = {t:.2f} (r ± {epsilon})" for t in target]
    return ScoreResult("Adjusted MLE", theta, se, status, res.iterations, "SE = 1/√I(θ̂)")
