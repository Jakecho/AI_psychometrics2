"""Safeguarded one-dimensional solvers, vectorised over persons, with full iteration records.

`maximize`  — Newton–Raphson on a log-objective (MLE, MAP):
    θ_new = θ + ℓ'(θ) / c(θ),  c = −ℓ''(θ) if ℓ'' < 0, otherwise the expected information
    (Fisher scoring). The step is capped at `max_step`, kept inside the search bounds, and halved
    while the objective decreases.

`find_root` — safeguarded Newton (rtsafe) for a decreasing estimating equation U(θ) = 0
    (WLE, TCC inversion): a bracket [lo, hi] with U(lo) > 0 > U(hi) is kept; a Newton step that
    leaves the bracket is replaced by bisection.

Every iteration is recorded as arrays over persons; `rows(history, i)` extracts one person.
"""
from __future__ import annotations

import numpy as np


def maximize(fun, start, settings, record=True):
    """fun(theta) -> dict(obj, grad, hess, fisher) of (N,) arrays. Returns (theta, status, n_iter, history)."""
    theta = np.array(start, float)
    N = len(theta)
    B, tol = settings.bound, settings.tol
    status = np.array(["max_iter"] * N, dtype=object)
    n_iter = np.zeros(N, int)
    active = np.ones(N, bool)
    history = []
    cur = fun(theta, subset=np.arange(N))
    for it in range(1, settings.max_iter + 1):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        g, h, f, obj = cur["grad"][idx], cur["hess"][idx], cur["fisher"][idx], cur["obj"][idx]
        newton = h < 0
        curv = np.where(newton, -h, f)
        raw = g / curv
        step = np.clip(raw, -settings.max_step, settings.max_step)
        limited = np.abs(raw) > settings.max_step
        new = np.clip(theta[idx] + step, -B, B)
        trial = fun_subset(fun, theta, idx, new)
        halvings = np.zeros(len(idx), int)
        worse = trial["obj"] < obj - 1e-12
        while worse.any() and halvings.max() < 40:
            new = np.where(worse, theta[idx] + (new - theta[idx]) / 2, new)
            halvings += worse
            trial = fun_subset(fun, theta, idx, new)
            worse = trial["obj"] < obj - 1e-12
        kind = np.where(newton, "Newton", "Fisher scoring (ℓ″ ≥ 0)").astype(object)
        kind = np.where(limited, kind + ", step capped", kind)
        kind = np.where(halvings > 0, kind + ", halved ×" + halvings.astype(str), kind)
        change = new - theta[idx]
        if record:
            history.append(dict(iteration=it, idx=idx.copy(), theta=theta[idx].copy(), obj=obj.copy(), grad=g.copy(),
                                hess=h.copy(), curvature=curv.copy(), step_type=kind, raw_step=raw.copy(),
                                step=change.copy(), theta_new=new.copy()))
        theta[idx] = new
        for key in cur:
            cur[key][idx] = trial[key]
        n_iter[idx] = it
        done = np.abs(change) < tol
        bounded = (np.abs(new) >= B) & (np.sign(trial["grad"]) == np.sign(new))
        status[idx[done]] = "converged"
        status[idx[bounded & ~done]] = "bounded"
        active[idx[done | bounded]] = False
    return theta, status, n_iter, history


def fun_subset(fun, theta, idx, new):
    full = theta.copy()
    full[idx] = new
    out = fun(full, subset=idx)
    return out


def find_root(fun, start, settings, record=True, lo=None, hi=None):
    """fun(theta, subset) -> dict(U, dU) with U decreasing through the root.

    Returns (theta, status, n_iter, history). status: converged / no root below lower bound /
    no root above upper bound / max_iter.
    """
    theta = np.array(start, float)
    N = len(theta)
    B, tol = settings.bound, settings.tol
    lo = np.full(N, -B) if lo is None else np.array(lo, float)
    hi = np.full(N, B) if hi is None else np.array(hi, float)
    all_idx = np.arange(N)
    u_lo = fun(lo, subset=all_idx)["U"]
    u_hi = fun(hi, subset=all_idx)["U"]
    status = np.array(["max_iter"] * N, dtype=object)
    below = u_lo <= 0          # U already ≤ 0 at the lower bound: root (if any) lies below −B
    above = u_hi >= 0          # U still ≥ 0 at the upper bound: root lies above +B
    status[below] = "no root above the lower bound"
    status[above & ~below] = "no root below the upper bound"
    theta[below] = lo[below]
    theta[above & ~below] = hi[above & ~below]
    active = ~(below | above)
    theta = np.clip(theta, lo, hi)
    n_iter = np.zeros(N, int)
    history = []
    for it in range(1, settings.max_iter + 1):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        cur = fun(theta, subset=idx)
        U, dU = cur["U"], cur["dU"]
        lo[idx] = np.where(U > 0, theta[idx], lo[idx])
        hi[idx] = np.where(U <= 0, theta[idx], hi[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = theta[idx] - U / dU
        exact = np.abs(U) <= 1e-14
        ok = (dU < 0) & (newton > lo[idx]) & (newton < hi[idx])
        new = np.where(exact, theta[idx], np.where(ok, newton, (lo[idx] + hi[idx]) / 2))
        kind = np.where(ok | exact, "Newton", "bisection (Newton step left the bracket)").astype(object)
        change = new - theta[idx]
        if record:
            hist = dict(iteration=it, idx=idx.copy(), theta=theta[idx].copy(), U=U.copy(), dU=dU.copy(),
                        lo=lo[idx].copy(), hi=hi[idx].copy(), step_type=kind, step=change.copy(), theta_new=new.copy())
            hist.update({k: v.copy() for k, v in cur.items() if k not in ("U", "dU")})
            history.append(hist)
        theta[idx] = new
        n_iter[idx] = it
        done = np.abs(change) < tol
        status[idx[done]] = "converged"
        active[idx[done]] = False
    return theta, status, n_iter, history


def rows(history, person):
    """Iteration records of one person as a list of dicts (for tables and step-through views)."""
    out = []
    for h in history:
        where = np.flatnonzero(h["idx"] == person)
        if not len(where):
            continue
        k = where[0]
        out.append({key: (val[k] if isinstance(val, np.ndarray) and key != "idx" else val)
                    for key, val in h.items() if key != "idx"})
    return out
