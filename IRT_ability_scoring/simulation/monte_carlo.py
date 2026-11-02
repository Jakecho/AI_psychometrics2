"""Monte Carlo evaluation of scoring methods against known generating abilities.

Per replication: draw θ, draw item parameters (fixed across replications unless `new_items`), simulate
responses, score with every selected method, and summarise against the true θ. Persons whose estimate
is not finite (e.g. MLE at extreme patterns) are excluded from bias / RMSE / MAD and counted in
`nonfinite_rate`; this is reported, never hidden.

Summaries per method (averaged over replications, with Monte Carlo SE = SD across replications / √R):
    bias  = mean(θ̂ − θ),  RMSE = √mean((θ̂ − θ)²),  MAD = mean|θ̂ − θ|,
    empirical SD of the errors, mean reported uncertainty, coverage of θ̂ ± 1.96·uncertainty,
    convergence-failure rate, non-finite rate, seconds.
Conditional bias and RMSE are computed in bins of true θ (pooled over replications).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from scoring import FUNCTIONS

from .data import draw_theta, make_items, simulate_responses

FAIL_WORDS = ("did not converge", "max_iter", "local maximum", "outside the search range", "stopped at")


def run(methods, settings, n_persons=500, replications=10, seed=7, theta_dist="normal", theta_mean=0.0, theta_sd=1.0,
        missing_rate=0.0, new_items=False, bins=(-np.inf, -2, -1, 0, 1, 2, np.inf), time_limit=None, **item_kw):
    rng = np.random.default_rng(seed)
    items = make_items(rng=np.random.default_rng(seed + 10_000), **item_kw)
    rep_rows, person_rows = [], []
    t_start = time.perf_counter()
    done = 0
    for r in range(replications):
        if time_limit and time.perf_counter() - t_start > time_limit and r > 0:
            break
        if new_items and r > 0:
            items = make_items(rng=rng, **item_kw)
        theta = draw_theta(n_persons, theta_dist, theta_mean, theta_sd, rng)
        Y = simulate_responses(items, theta, missing_rate, rng)
        for m in methods:
            res = FUNCTIONS[m](items, Y, settings)
            est, unc = res.theta, res.se
            fin = np.isfinite(est)
            err = est[fin] - theta[fin]
            fail = np.array([any(w in str(s) for w in FAIL_WORDS) for s in res.status])
            ok_unc = fin & np.isfinite(unc)
            cover = np.abs(est[ok_unc] - theta[ok_unc]) <= 1.959964 * unc[ok_unc]
            rep_rows.append(dict(replication=r + 1, method=m, n=int(fin.sum()), bias=err.mean(),
                                 rmse=np.sqrt((err ** 2).mean()), mad=np.abs(err).mean(), emp_sd=err.std(ddof=1),
                                 mean_uncertainty=float(np.mean(unc[ok_unc])) if ok_unc.any() else np.nan,
                                 coverage=float(cover.mean()) if len(cover) else np.nan,
                                 failure_rate=float(fail.mean()), nonfinite_rate=float((~fin).mean()),
                                 seconds=res.seconds))
            person_rows.append(pd.DataFrame(dict(replication=r + 1, method=m, theta=theta, estimate=est,
                                                 uncertainty=unc, raw=np.nansum(Y, 1))))
        done += 1
    reps = pd.DataFrame(rep_rows)
    persons = pd.concat(person_rows, ignore_index=True)
    return dict(replications=reps, persons=persons, summary=summarise(reps), conditional=conditional(persons, bins),
                items=items, completed=done, requested=replications, seconds=time.perf_counter() - t_start)


def summarise(reps):
    cols = ["bias", "rmse", "mad", "emp_sd", "mean_uncertainty", "coverage", "failure_rate", "nonfinite_rate", "seconds"]
    g = reps.groupby("method", sort=False)
    out = g[cols].mean()
    R = g.size()
    for c in ("bias", "rmse", "coverage"):
        out[f"mcse_{c}"] = g[c].std(ddof=1) / np.sqrt(R) if R.min() > 1 else np.nan
    out["replications"] = R
    return out.reset_index()


def conditional(persons, bins):
    p = persons[np.isfinite(persons["estimate"])].copy()
    p["theta_bin"] = pd.cut(p["theta"], bins).astype(str)
    p["err"] = p["estimate"] - p["theta"]
    g = p.groupby(["method", "theta_bin"], sort=False)
    out = g["err"].agg(bias="mean", rmse=lambda e: float(np.sqrt((e ** 2).mean())), n="size").reset_index()
    order = {str(c): k for k, c in enumerate(pd.cut(pd.Series([]), bins).cat.categories)}
    out["order"] = out["theta_bin"].map(order)
    return out.sort_values(["method", "order"]).drop(columns="order")
