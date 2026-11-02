"""Person scoring with fixed (already calibrated) item difficulties.

MLE  solves  U(theta) = r - sum_j P_j(theta) = 0. It is infinite for zero and perfect
     raw scores (reported as -inf / +inf, never silently clipped).
WLE  (Warm, 1989) solves  U(theta) + I'(theta) / (2 I(theta)) = 0, with
     I(theta) = sum_j P_j (1 - P_j)  and  I'(theta) = sum_j P_j (1 - P_j)(1 - 2 P_j).
     For the Rasch model the WLE is finite for every raw score, including 0 and perfect.
Both use only the person's observed items. SE = 1 / sqrt(I(theta)) at the estimate.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .model import observed_values


def _parts(theta, b):
    p = expit(theta - b)
    info = float(np.sum(p * (1 - p)))
    dinfo = float(np.sum(p * (1 - p) * (1 - 2 * p)))
    return p, info, dinfo


def mle_score(theta, r, b):
    p, _, _ = _parts(theta, b)
    return r - p.sum()


def wle_score(theta, r, b):
    p, info, dinfo = _parts(theta, b)
    return r - p.sum() + dinfo / (2 * info)


def _root(fun, lo=-30.0, hi=30.0):
    return brentq(fun, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)


def score_persons(X, b, method="WLE"):
    """Ability estimates and SEs for every person (method: 'MLE' or 'WLE')."""
    x, mask = observed_values(X)
    b = np.asarray(b, float)
    theta = np.empty(x.shape[0])
    se = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        bi = b[mask[i]]
        r = x[i, mask[i]].sum()
        n = len(bi)
        if method == "MLE" and (r == 0 or r == n):
            theta[i] = -np.inf if r == 0 else np.inf
            se[i] = np.inf
            continue
        fun = (lambda t, r=r, bi=bi: mle_score(t, r, bi)) if method == "MLE" else (lambda t, r=r, bi=bi: wle_score(t, r, bi))
        theta[i] = _root(fun)
        se[i] = 1 / np.sqrt(_parts(theta[i], bi)[1])
    return theta, se


def score_table(b, method="WLE"):
    """Estimate for each possible raw score on the complete test (r = 0..J)."""
    b = np.asarray(b, float)
    rows = []
    for r in range(len(b) + 1):
        if method == "MLE" and (r == 0 or r == len(b)):
            rows.append((r, -np.inf if r == 0 else np.inf, np.inf))
            continue
        fun = (lambda t, r=r: mle_score(t, r, b)) if method == "MLE" else (lambda t, r=r: wle_score(t, r, b))
        t = _root(fun)
        rows.append((r, t, 1 / np.sqrt(_parts(t, b)[1])))
    return np.array(rows, float)


def newton_trace(r, b, method="WLE", start=0.0, steps=25, tol=1e-10):
    """Actual Newton iterations for one raw score (for the walkthrough).

    MLE derivative: -I(theta). WLE derivative of the adjusted score is taken numerically.
    Steps are limited to |1| logit; extreme MLE scores diverge and are flagged.
    """
    b = np.asarray(b, float)
    fun = (lambda t: mle_score(t, r, b)) if method == "MLE" else (lambda t: wle_score(t, r, b))
    t = float(start)
    rows = []
    for k in range(1, steps + 1):
        p, info, dinfo = _parts(t, b)
        u = fun(t)
        h = 1e-6
        deriv = -info if method == "MLE" else (fun(t + h) - fun(t - h)) / (2 * h)
        step = float(np.clip(-u / deriv, -1, 1))
        rows.append(dict(step=k, theta=t, expected_score=float(p.sum()), residual=float(r - p.sum()),
                         information=info, adjustment=dinfo / (2 * info) if method == "WLE" else 0.0,
                         score_function=float(u), update=step, theta_next=t + step))
        t += step
        if abs(step) < tol:
            break
    return rows
