"""EAP based on summed scores (EAPsum).

The likelihood of a summed score r at θ comes from the Lord–Wingersky recursion: start with
f_0(0 | θ) = 1 and add one item at a time,

    f_j(s | θ) = Σ_k f_{j−1}(s − k | θ) · P_jk(θ)

so f_J(r | θ) = P(R = r | θ). Then, over quadrature nodes,

    θ̂_EAPsum(r) = Σ_q θ_q f_J(r | θ_q) w_q / Σ_q f_J(r | θ_q) w_q

Every response pattern with the same summed score (on the same set of items) gets the same
estimate. Persons with missing responses are scored on the items they answered.
"""
from __future__ import annotations

import time

import numpy as np

from .eap import quadrature
from .result import ScoreResult, skip_empty


def score_distribution(items, nodes):
    """P(R = r | θ_q): shape (Q, R_max + 1)."""
    nodes = np.atleast_1d(np.asarray(nodes, float))
    f = np.ones((len(nodes), 1))
    for it in items:
        P = it.probs(nodes)                       # (Q, K)
        new = np.zeros((len(nodes), f.shape[1] + it.max_score))
        for k in range(it.n_cat):
            new[:, k:k + f.shape[1]] += f * P[:, [k]]
        f = new
    return f


def recursion_trace(items, theta):
    """The recursion at one θ: list of (item id, category probabilities, distribution after the item)."""
    f = np.ones(1)
    steps = [("start", None, f.copy())]
    for it in items:
        P = it.probs(np.array([float(theta)]))[0]
        new = np.zeros(len(f) + it.max_score)
        for k in range(it.n_cat):
            new[k:k + len(f)] += f * P[k]
        f = new
        steps.append((it.id, P, f.copy()))
    return steps


def score_table(items, settings, rule=None, n=None):
    """EAPsum, posterior SD and marginal probability for every summed score of a complete test."""
    nodes, w = quadrature(settings, rule, n)
    f = score_distribution(items, nodes)          # (Q, S)
    joint = f * w[:, None]
    marg = joint.sum(0)
    post = joint / marg
    m = nodes @ post
    sd = np.sqrt(np.maximum(nodes ** 2 @ post - m ** 2, 0))
    return dict(score=np.arange(f.shape[1]), marginal_probability=marg, EAPsum=m, PSD=sd), (nodes, w, f)


@skip_empty
def eapsum(items, Y, settings, rule=None, n=None):
    t0 = time.perf_counter()
    Y = np.atleast_2d(np.asarray(Y, float))
    N = Y.shape[0]
    theta, psd = np.full(N, np.nan), np.full(N, np.nan)
    obs = ~np.isnan(Y)
    r = np.nansum(Y, 1).astype(int)
    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    for g, mask in enumerate(patterns):
        rows = inverse == g
        if not mask.any():
            continue
        tab, _ = score_table([it for it, m in zip(items, mask) if m], settings, rule, n)
        theta[rows] = tab["EAPsum"][r[rows]]
        psd[rows] = tab["PSD"][r[rows]]
    status = np.where(obs.all(1), "summed-score posterior", "summed-score posterior on observed items")
    return ScoreResult("EAPsum", theta, psd, status.astype(object), np.zeros(N, int), "posterior SD",
                       extra=dict(raw=r), seconds=time.perf_counter() - t0)
