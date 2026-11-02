"""Maximum a posteriori (posterior mode) ability estimation with a normal prior N(μ, σ²).

    log posterior   = ℓ(θ) − (θ − μ)² / (2σ²) + const
    U_MAP(θ)        = ℓ'(θ) − (θ − μ) / σ²
    curvature       = ℓ''(θ) − 1/σ²
    θ ← θ − U_MAP / curvature

Uncertainty: SE = 1/√(I(θ̂) + 1/σ²) (expected posterior information; irtQ) and the observed
version 1/√(−ℓ''(θ̂) + 1/σ²) (mirt) are both kept.
"""
from __future__ import annotations

import time

import numpy as np

from irt_models import pattern_terms

from .likelihood import global_check, nonconcave
from .result import ScoreResult, skip_empty
from .solvers import maximize


def objective(items, Y, mu, sd):
    def fun(theta, subset):
        th = theta[subset]
        t = pattern_terms(items, Y[subset], th)
        return dict(obj=t["loglik"] - (th - mu) ** 2 / (2 * sd ** 2), grad=t["score"] - (th - mu) / sd ** 2,
                    hess=t["hessian"] - 1 / sd ** 2, fisher=t["info"] + 1 / sd ** 2,
                    lik_score=t["score"], prior_score=-(th - mu) / sd ** 2, lik_hess=t["hessian"])
    return fun


@skip_empty
def map_estimate(items, Y, settings, start=None, record=False):
    t0 = time.perf_counter()
    Y = np.atleast_2d(np.asarray(Y, float))
    N = Y.shape[0]
    mu, sd = settings.prior_mean, settings.prior_sd
    start = np.full(N, settings.start) if start is None else np.broadcast_to(np.asarray(start, float), (N,)).copy()
    theta, status, n_iter, hist = maximize(objective(items, Y, mu, sd), start, settings, record)
    words = np.array(["converged" if s == "converged" else
                      ("stopped at the search bound" if s == "bounded" else "did not converge") for s in status],
                     dtype=object)
    if nonconcave(items):
        # The log posterior can be multimodal with 3PL/4PL items: compare with a grid and restart if needed.
        loglik_now = pattern_terms(items, Y, theta)["loglik"] - (theta - mu) ** 2 / (2 * sd ** 2)
        msg, top, edge = global_check(items, Y, theta, loglik_now, settings.bound,
                                      log_prior=lambda g: -(g - mu) ** 2 / (2 * sd ** 2))
        idx = np.flatnonzero(msg != "")
        if len(idx):
            th2, st2, it2, _ = maximize(objective(items, Y[idx], mu, sd), top[idx], settings, False)
            lp2 = pattern_terms(items, Y[idx], th2)["loglik"] - (th2 - mu) ** 2 / (2 * sd ** 2)
            for k, i in enumerate(idx):
                if st2[k] == "converged" and lp2[k] > loglik_now[i] + 1e-10:
                    words[i] = (f"converged after a restart from the grid maximum (the run from θ⁽⁰⁾ reached a local "
                                f"mode at θ = {theta[i]:.3f})")
                    theta[i], n_iter[i] = th2[k], n_iter[i] + it2[k]
    final = pattern_terms(items, Y, theta)
    se = 1 / np.sqrt(final["info"] + 1 / sd ** 2)
    curv = -final["hessian"] + 1 / sd ** 2
    se_obs = np.where(curv > 0, 1 / np.sqrt(np.maximum(curv, 1e-300)), np.nan)
    return ScoreResult("MAP", theta, se, words, n_iter, "SE = 1/√(I + 1/σ²)",
                       extra=dict(se_observed=se_obs, log_posterior=final["loglik"] - (theta - mu) ** 2 / (2 * sd ** 2),
                                  info=final["info"], raw=final["raw"]),
                       history=hist, seconds=time.perf_counter() - t0)
