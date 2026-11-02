"""Recovery statistics and explicit (post-hoc) location alignment.

Rasch difficulties are identified only up to a common shift, and each estimator fixes
the location differently (JMLE "cases": mean ability 0; CMLE: sum b = 0; MML: ability
mean 0). Before comparing with true difficulties, estimates are shifted so that their
mean equals the mean of the true difficulties. This alignment is reported separately and
never changes the fitted (native) parameters.
"""
from __future__ import annotations

import numpy as np


def align_location(estimate, target):
    estimate = np.asarray(estimate, float)
    shift = float(np.mean(target) - np.mean(estimate))
    return estimate + shift, shift


def recovery(estimate, truth):
    est = np.asarray(estimate, float)
    tru = np.asarray(truth, float)
    ok = np.isfinite(est) & np.isfinite(tru)
    e = est[ok] - tru[ok]
    if e.size == 0:
        return dict(n=0, bias=np.nan, rmse=np.nan, mad=np.nan, max_abs=np.nan, corr=np.nan)
    corr = float(np.corrcoef(est[ok], tru[ok])[0, 1]) if e.size > 2 and np.std(est[ok]) > 0 else np.nan
    return dict(n=int(e.size), bias=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                mad=float(np.abs(e).mean()), max_abs=float(np.abs(e).max()), corr=corr)
