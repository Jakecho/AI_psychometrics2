"""Samejima's graded response model (GRM).

Cumulative ("boundary") curves:  P*_jk(θ) = Ψ(D a_j (θ − b_jk)),  k = 1..m,  with P*_j0 = 1 and
P*_j,m+1 = 0. Category probabilities are differences of adjacent boundaries:

    P_jk(θ) = P*_jk(θ) − P*_j,k+1(θ)

Thresholds must be strictly increasing so every category probability is positive.
Using P*(1 − P*) differences:

    d log P_jk / dθ   = D a (1 − P*_k − P*_k+1)
    d² log P_jk / dθ² = −(D a)² [P*_k Q*_k + P*_k+1 Q*_k+1]

Log probabilities are taken from whichever tail avoids cancellation:
Ψ(z_k) − Ψ(z_k+1) = Ψ(−z_k+1) − Ψ(−z_k).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit


def _z(theta, a, thresholds, D):
    return D * a * np.subtract.outer(np.asarray(theta, float), np.asarray(thresholds, float))   # (..., m)


def log_probs(theta, a, thresholds, D):
    z = _z(theta, a, thresholds, D)
    shape = z.shape[:-1]
    inf = np.full(shape + (1,), np.inf)
    zk = np.concatenate([inf, z], axis=-1)            # boundary k   (z_0 = +∞)
    zk1 = np.concatenate([z, -inf], axis=-1)          # boundary k+1 (z_{m+1} = −∞)
    with np.errstate(invalid="ignore", divide="ignore"):
        upper = zk1 > 0                                # both boundaries near 1 → use the lower tail
        A = np.where(upper, log_expit(-zk1), log_expit(zk))
        B = np.where(upper, log_expit(-zk), log_expit(zk1))
        lp = A + np.log1p(-np.exp(B - A))
    return lp


def log_derivs(theta, a, thresholds, D):
    z = _z(theta, a, thresholds, D)
    shape = z.shape[:-1]
    ps = expit(z)
    star = np.concatenate([np.ones(shape + (1,)), ps, np.zeros(shape + (1,))], axis=-1)   # P*_0..P*_{m+1}
    pk, pk1 = star[..., :-1], star[..., 1:]
    P = np.exp(log_probs(theta, a, thresholds, D))
    dl = D * a * (1.0 - pk - pk1)
    d2l = -(D * a) ** 2 * (pk * (1 - pk) + pk1 * (1 - pk1))
    return P, dl, d2l
