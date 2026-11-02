"""Warm's weighted likelihood estimation (WLE).

Warm (1989): the MLE has first-order bias  B(θ) = J(θ) / (2 I(θ)²),  with
    I(θ) = Σ_j Σ_k P'_jk² / P_jk         (test information)
    J(θ) = Σ_j Σ_k P'_jk P''_jk / P_jk
Solving the adjusted score equation removes it:

    U_W(θ) = ℓ'(θ) + J(θ) / (2 I(θ)) = 0

This is the general form used by irtQ (method = "WL"); it is valid for 3PL, 4PL, GRM and the
divide-by-total models alike. For Rasch, 2PL, PCM, GPCM, RSM and GRM items J = dI/dθ,
so U_W is the derivative of ℓ + ½ log I and the WLE maximises L(θ)·√I(θ). That simpler form is
what mirt::fscores(method = "WLE") maximises for every model; `weighted_likelihood_mode` computes it
so the two can be compared. They differ only when J ≠ dI/dθ, i.e. with 3PL / 4PL items.

SE: 1/√I(θ̂_W) (irtQ, TAM).
"""
from __future__ import annotations

import time

import numpy as np

from irt_models import extreme_status, pattern_terms

from .result import ScoreResult, skip_empty
from .solvers import find_root

H = 1e-5


def _warm_parts(items, Y, theta):
    t = pattern_terms(items, Y, theta)
    adj = t["J"] / (2 * t["info"])
    return t, adj


def equation(items, Y, kind="warm"):
    key = "J" if kind == "warm" else "Iprime"

    def U_of(th, Ysub):
        t = pattern_terms(items, Ysub, th)
        return t["score"] + t[key] / (2 * t["info"]), t

    def fun(theta, subset):
        th = theta[subset]
        Ysub = Y[subset]
        U, t = U_of(th, Ysub)
        up, _ = U_of(th + H, Ysub)
        dn, _ = U_of(th - H, Ysub)
        adj = t[key] / (2 * t["info"])
        return dict(U=U, dU=(up - dn) / (2 * H), score=t["score"], adjustment=adj, info=t["info"],
                    J=t[key], hessian=t["hessian"])
    return fun


def _run(items, Y, settings, start, record, kind, name):
    t0 = time.perf_counter()
    Y = np.atleast_2d(np.asarray(Y, float))
    N = Y.shape[0]
    start = np.full(N, settings.start) if start is None else np.broadcast_to(np.asarray(start, float), (N,)).copy()
    theta, status, n_iter, hist = find_root(equation(items, Y, kind), start, settings, record)
    t = pattern_terms(items, Y, theta)
    se = 1 / np.sqrt(t["info"])
    words = np.array([s if s != "converged" else "converged" for s in status], dtype=object)
    out_of_range = np.isin(status, ["no root above the lower bound", "no root below the upper bound"])
    words[out_of_range] = [f"{s} (θ = {th:.0f}); estimate set to the bound" for s, th in
                           zip(status[out_of_range], theta[out_of_range])]
    return ScoreResult(name, theta, se, words, n_iter, "SE = 1/√I(θ̂)",
                       extra=dict(info=t["info"], adjustment=(t["J"] if kind == "warm" else t["Iprime"]) / (2 * t["info"]),
                                  extreme=extreme_status(items, Y), raw=t["raw"]),
                       history=hist, seconds=time.perf_counter() - t0)


@skip_empty
def wle(items, Y, settings, start=None, record=False):
    """Warm's WLE: root of ℓ' + J/(2I)."""
    return _run(items, Y, settings, start, record, "warm", "WLE")


@skip_empty
def weighted_likelihood_mode(items, Y, settings, start=None, record=False):
    """Maximiser of L·√I: root of ℓ' + I'/(2I) (mirt's 'WLE'). Equals Warm's WLE when J = I'."""
    return _run(items, Y, settings, start, record, "jeffreys", "L·√I mode")


def adjustment_curves(items, y, grid):
    t = pattern_terms(items, np.tile(np.asarray(y, float), (len(grid), 1)), grid)
    return dict(score=t["score"], warm=t["J"] / (2 * t["info"]), jeffreys=t["Iprime"] / (2 * t["info"]),
                info=t["info"], J=t["J"], Iprime=t["Iprime"])
