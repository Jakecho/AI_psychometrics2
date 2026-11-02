"""Dichotomous logistic models: Rasch/1PL, 2PL, 3PL and 4PL.

    P_j(θ) = c_j + (d_j − c_j) Ψ(z),   z = D a_j (θ − b_j),   Ψ(z) = 1 / (1 + e^(−z))

Rasch: a = 1, c = 0, d = 1.  2PL: c = 0, d = 1.  3PL: d = 1.  4PL: all four free.
Category 0 is an incorrect response, category 1 a correct response.

Returned arrays have shape (T, 2): one row per θ value, one column per category.
`log_derivs` returns log P, d log P / dθ and d² log P / dθ² for each category. They are
computed from Ψ and 1 − Ψ = Ψ(−z) separately, so neither tail loses precision.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit


def log_probs(theta, a, b, c, d, D):
    z = D * a * (np.asarray(theta, float) - b)
    if c == 0.0 and d == 1.0:
        lp1, lp0 = log_expit(z), log_expit(-z)
    else:
        lp1 = np.log(c + (d - c) * expit(z))
        lp0 = np.log((1.0 - d) + (d - c) * expit(-z))
    return np.stack([lp0, lp1], axis=-1)


def log_derivs(theta, a, b, c, d, D):
    theta = np.asarray(theta, float)
    z = D * a * (theta - b)
    psi, q = expit(z), expit(-z)
    p1 = c + (d - c) * psi
    p0 = (1.0 - d) + (d - c) * q
    g = (d - c) * D * a * psi * q                       # P'(θ)
    h = (d - c) * (D * a) ** 2 * psi * q * (q - psi)    # P''(θ); 1 − 2Ψ = q − Ψ
    if c == 0.0 and d == 1.0:                           # exact simplifications: g/p1 = Da·q, g/p0 = Da·Ψ
        dl1, dl0 = D * a * q, -D * a * psi
    else:
        dl1, dl0 = g / p1, -g / p0
    d2l1 = h / p1 - dl1 ** 2
    d2l0 = -h / p0 - dl0 ** 2
    P = np.stack([p0, p1], axis=-1)
    return P, np.stack([dl0, dl1], axis=-1), np.stack([d2l0, d2l1], axis=-1)
