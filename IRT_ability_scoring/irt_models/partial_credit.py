"""Divide-by-total (adjacent-category) models: PCM, GPCM and RSM.

    P_jk(θ) = exp(η_jk) / Σ_h exp(η_jh),   η_jk = Σ_{v=1..k} D a_j (θ − δ_jv),   η_j0 = 0

PCM:  a_j = 1, step difficulties δ_jv free per item.
GPCM: slope a_j free, step difficulties δ_jv free per item.
RSM:  a_j = 1, δ_jv = b_j + τ_v with thresholds τ_v shared by all RSM items.

Every model in this family is an exponential family in θ: the category score k is the
sufficient statistic, which gives the closed forms

    d log P_jk / dθ   = D a_j (k − E_j(θ))
    d² log P_jk / dθ² = −(D a_j)² Var_j(θ)

with E_j and Var_j the mean and variance of the category score at θ.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _eta(theta, a, deltas, D):
    theta = np.asarray(theta, float)
    deltas = np.asarray(deltas, float)
    k = np.arange(len(deltas) + 1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    return D * a * (np.multiply.outer(theta, k) - cum)          # (..., K)


def log_probs(theta, a, deltas, D):
    eta = _eta(theta, a, deltas, D)
    return eta - logsumexp(eta, axis=-1, keepdims=True)


def log_derivs(theta, a, deltas, D):
    lp = log_probs(theta, a, deltas, D)
    P = np.exp(lp)
    k = np.arange(P.shape[-1])
    E = (P * k).sum(-1, keepdims=True)
    V = (P * (k - E) ** 2).sum(-1, keepdims=True)
    dl = D * a * (k - E)
    d2l = np.broadcast_to(-(D * a) ** 2 * V, P.shape)
    return P, dl, np.array(d2l)
