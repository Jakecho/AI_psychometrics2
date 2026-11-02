"""Maximum likelihood ability estimation with known item parameters.

    ℓ_i(θ) = Σ_{j∈O_i} log P_{j,y_ij}(θ)
    θ ← θ − ℓ'(θ) / ℓ''(θ)                         (Newton–Raphson)

Safeguards (see scoring.solvers.maximize): Fisher scoring when ℓ'' ≥ 0, a step cap, step halving
when ℓ decreases, and the search range [−bound, bound]. With 3PL/4PL items ℓ can have several maxima:
the solution is compared with an 801-point grid and, if the grid is higher, Newton is restarted from the
grid maximum (reported in the status). If ℓ is highest at the edge of the range, no finite MLE is reported.

Patterns with every observed response in the lowest (highest) category have no finite MLE: ℓ
increases monotonically as θ → −∞ (+∞). They are reported as ±∞; the bounded optimiser result is
kept separately in extra["theta_bounded"] and is *not* an MLE.
"""
from __future__ import annotations

import time

import numpy as np

from irt_models import extreme_status, pattern_terms

from .likelihood import global_check, nonconcave
from .result import ScoreResult, skip_empty
from .solvers import maximize


def objective(items, Y):
    def fun(theta, subset):
        t = pattern_terms(items, Y[subset], theta[subset])
        return dict(obj=t["loglik"], grad=t["score"], hess=t["hessian"], fisher=t["info"])
    return fun


@skip_empty
def mle(items, Y, settings, start=None, record=False):
    t0 = time.perf_counter()
    Y = np.atleast_2d(np.asarray(Y, float))
    N = Y.shape[0]
    start = np.full(N, settings.start) if start is None else np.broadcast_to(np.asarray(start, float), (N,)).copy()
    theta, status, n_iter, hist = maximize(objective(items, Y), start, settings, record)
    final = pattern_terms(items, Y, theta)
    info, hess = final["info"], final["hessian"]
    with np.errstate(divide="ignore", invalid="ignore"):
        se = 1 / np.sqrt(info)
        se_obs = np.where(hess < 0, 1 / np.sqrt(-hess), np.nan)
    estimate = theta.copy()
    words = np.array(["converged" if s == "converged" else s for s in status], dtype=object)
    ext = extreme_status(items, Y)
    for side, sign in (("min", -1), ("max", 1)):
        e = ext == side
        estimate[e] = sign * np.inf
        se[e] = np.nan
        se_obs[e] = np.nan
        words[e] = ("no finite MLE: all responses in the lowest category (ℓ rises as θ → −∞)" if side == "min" else
                    "no finite MLE: all responses in the highest category (ℓ rises as θ → +∞)")
    bounded = (status == "bounded") & (ext == "")
    words[bounded] = "stopped at the search bound (no interior maximum found)"
    estimate[bounded] = np.sign(theta[bounded]) * np.inf
    se[bounded] = np.nan
    restarted = np.zeros(N, bool)
    if nonconcave(items):
        # 3PL/4PL: ℓ may have several maxima. Compare with a grid; if the grid is higher, restart Newton there.
        msg, top, edge = global_check(items, Y, theta, final["loglik"], settings.bound)
        redo = (msg != "") & (ext == "")
        interior = redo & ~edge
        if interior.any():
            idx = np.flatnonzero(interior)
            th2, st2, it2, _ = maximize(objective(items, Y[idx]), top[idx], settings, False)
            f2 = pattern_terms(items, Y[idx], th2)
            ok = (st2 == "converged") & (f2["loglik"] > final["loglik"][idx] + 1e-10)
            for k, i in enumerate(idx):
                if not ok[k]:
                    continue
                first = (f"a local maximum at θ = {theta[i]:.3f}" if status[i] == "converged" else
                         f"the search bound θ = {theta[i]:.0f}")
                words[i] = f"converged after a restart from the grid maximum (the run from θ⁽⁰⁾ reached {first})"
                theta[i], estimate[i], n_iter[i] = th2[k], th2[k], n_iter[i] + it2[k]
                info[i], hess[i] = f2["info"][k], f2["hessian"][k]
                final["loglik"][i] = f2["loglik"][k]
                with np.errstate(divide="ignore", invalid="ignore"):
                    se[i] = 1 / np.sqrt(info[i])
                    se_obs[i] = 1 / np.sqrt(-hess[i]) if hess[i] < 0 else np.nan
                restarted[i] = True
        at_edge = redo & edge
        words[at_edge] = msg[at_edge]
        estimate[at_edge] = np.sign(top[at_edge]) * np.inf
        se[at_edge] = se_obs[at_edge] = np.nan
    no_items = (~np.isnan(Y)).sum(1) == 0
    estimate[no_items], se[no_items], words[no_items] = np.nan, np.nan, "no observed responses"
    return ScoreResult("MLE", estimate, se, words, n_iter, "SE = 1/√I(θ̂)",
                       extra=dict(theta_bounded=theta, se_observed=se_obs, loglik=final["loglik"],
                                  info=info, extreme=ext, raw=final["raw"], restarted=restarted),
                       history=hist, seconds=time.perf_counter() - t0)
