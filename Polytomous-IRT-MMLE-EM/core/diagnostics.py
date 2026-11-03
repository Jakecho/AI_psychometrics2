"""Model fit, parameter recovery, Monte Carlo replications and drift analysis."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .data import Dataset
from .em import FitConfig, fit
from .models import probs
from .quadrature import moments
from .simulation import SimSpec, population_weights, simulate


# ---- model fit ----------------------------------------------------------------------------------

def category_fit(data: Dataset, res):
    """Observed category proportions vs expected marginal probabilities p̂_gjk = Σ_q π̂_gq P_jk(θ_q)."""
    rows = []
    adm = data.administered()
    for g in range(data.G):
        for j, it in enumerate(res.items):
            if not adm[g, j]:
                continue
            c = data.counts(j, g)
            n = c.sum()
            p_hat = res.weights[g] @ probs(it, res.nodes[g], res.config.D)
            for k in range(it.ncat):
                obs = c[k] / n
                se = np.sqrt(max(p_hat[k] * (1 - p_hat[k]), 1e-12) / n)
                rows.append(dict(group=data.group_names[g], item=it.id, category=k, n=int(n), observed=obs,
                                 expected=p_hat[k], difference=obs - p_hat[k], std_residual=(obs - p_hat[k]) / se,
                                 fixed=bool(res.fixed[j])))
    return pd.DataFrame(rows)


def fit_summary(table: pd.DataFrame):
    d = table["difference"].to_numpy()
    return dict(RMSE=float(np.sqrt(np.mean(d ** 2))), MAD=float(np.mean(np.abs(d))),
                max_abs=float(np.max(np.abs(d))), large_residuals=int((np.abs(table["std_residual"]) > 3).sum()))


def lord_wingersky(items, theta, D=1.0):
    """P(summed score = s | θ_q) for each node. Returns (Q, max score + 1)."""
    theta = np.atleast_1d(theta)
    dist = np.ones((len(theta), 1))
    for it in items:
        p = probs(it, theta, D)
        new = np.zeros((len(theta), dist.shape[1] + it.ncat - 1))
        for k in range(it.ncat):
            new[:, k:k + dist.shape[1]] += dist * p[:, [k]]
        dist = new
    return dist


def summed_score_fit(data: Dataset, res, g=0):
    """Observed vs expected summed-score distribution for examinees in group g who answered every item of
    their form (the items administered to the group)."""
    cols = np.flatnonzero(data.administered()[g])
    m = (data.group == g) & data.observed[:, cols].all(axis=1)
    if m.sum() == 0:
        return pd.DataFrame()
    scores = data.Y[np.ix_(m, cols)].sum(axis=1).astype(int)
    exp = res.weights[g] @ lord_wingersky([res.items[j] for j in cols], res.nodes[g], res.config.D)
    obs = np.bincount(scores, minlength=len(exp)) / m.sum()
    return pd.DataFrame(dict(score=np.arange(len(exp)), observed=obs, expected=exp, n=int(m.sum())))


def information_criteria(res, N):
    k = res.n_parameters()
    return dict(loglik=res.loglik, parameters=k, deviance=-2 * res.loglik, AIC=-2 * res.loglik + 2 * k,
                BIC=-2 * res.loglik + np.log(N) * k)


# ---- recovery ---------------------------------------------------------------------------------

def item_recovery(data: Dataset, res):
    """Estimate − generating value for every free parameter (canonical a, δ / b)."""
    if not data.truth:
        return pd.DataFrame()
    rows = []
    for it, f in zip(res.items, res.fixed):
        tr = data.truth["items"].get(it.id)
        if tr is None:
            continue
        for lab, e, t in zip(it.labels(), it.vector(), tr.vector()):
            if lab == "a" and it.model in ("PCM", "RSM"):
                continue
            se = None
            if res.se and it.id in res.se.get("items", {}):
                se = res.se["items"][it.id].get(lab)
            rows.append(dict(item=it.id, parameter=lab, kind="a" if lab == "a" else "location", fixed=bool(f),
                             true=t, estimate=e, error=e - t, se=se))
    return pd.DataFrame(rows)


def group_recovery(data: Dataset, res):
    if not data.truth:
        return pd.DataFrame()
    mm = res.group_moments
    rows = []
    for g, gt in enumerate(data.truth["groups"]):
        tw = population_weights(gt["shape"], res.nodes[g], gt["mean"], gt["sd"],
                                data.truth["spec"].get("discrete_nodes"), data.truth["spec"].get("discrete_weights"))
        rows.append(dict(group=data.group_names[g], shape=gt["shape"], true_mean=gt["mean"], est_mean=mm[g, 0],
                         true_sd=gt["sd"], est_sd=mm[g, 1], mean_error=mm[g, 0] - gt["mean"],
                         sd_error=mm[g, 1] - gt["sd"],
                         density_TV=0.5 * float(np.abs(tw - res.weights[g]).sum())))
    return pd.DataFrame(rows)


def summarize_errors(frame: pd.DataFrame, by):
    """Bias, RMSE, MAD (mean absolute error), empirical SD, mean model SE and 95% coverage."""
    def agg(d):
        e = d["error"].to_numpy(float)
        out = dict(n=len(e), bias=e.mean(), RMSE=np.sqrt(np.mean(e ** 2)), MAD=np.mean(np.abs(e)),
                   emp_SD=d.groupby(["item", "parameter"])["estimate"].std().mean() if "rep" in d else np.nan)
        if "se" in d and d["se"].notna().any():
            se = d["se"].to_numpy(float)
            ok = np.isfinite(se)
            out["mean_SE"] = np.nanmean(se)
            out["coverage95"] = float(np.mean(np.abs(e[ok]) <= 1.959964 * se[ok])) if ok.any() else np.nan
        return pd.Series(out)
    return frame.groupby(by).apply(agg, include_groups=False).reset_index()


def monte_carlo(spec: SimSpec, cfg: FitConfig, reps=30, seed0=1000, callback=None):
    """Replicate simulate → fit; every replication uses seed0 + r, so results are reproducible."""
    items, groups, runs = [], [], []
    for r in range(reps):
        d = simulate(replace(spec, seed=seed0 + r))
        t0 = time.perf_counter()
        try:
            res = fit(d, cfg, record=False)
            ok, conv, msg = True, res.converged, res.reason
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
            ok, conv, msg, res = False, False, str(e), None
        sec = time.perf_counter() - t0
        runs.append(dict(rep=r + 1, seed=seed0 + r, ok=ok, converged=conv, cycles=res.cycles if res else 0,
                         seconds=sec, loglik=res.loglik if res else np.nan, message=msg))
        if res is not None:
            ir = item_recovery(d, res)
            ir["rep"] = r + 1
            items.append(ir)
            gr = group_recovery(d, res)
            gr["rep"] = r + 1
            groups.append(gr)
        if callback:
            callback(r + 1, reps)
    return dict(items=pd.concat(items, ignore_index=True) if items else pd.DataFrame(),
                groups=pd.concat(groups, ignore_index=True) if groups else pd.DataFrame(),
                runs=pd.DataFrame(runs))


# ---- drift ----------------------------------------------------------------------------------

DRIFT_SCENARIOS = {
    "Stable anchors, equal populations": dict(anchor_drift=0.0, n_drift=0, group_means=[0.0, 0.0], shapes=None),
    "Stable anchors, changing population (+0.5)": dict(anchor_drift=0.0, n_drift=0, group_means=[0.0, 0.5], shapes=None),
    "Anchors become harder (+0.5 for 2 anchors)": dict(anchor_drift=0.5, n_drift=2, group_means=[0.0, 0.3], shapes=None),
    "Anchors become easier (−0.5 for 2 anchors)": dict(anchor_drift=-0.5, n_drift=2, group_means=[0.0, 0.3], shapes=None),
    "Group ability shift with drifting anchors": dict(anchor_drift=0.5, n_drift=3, group_means=[0.0, 0.6], shapes=None),
    "Changed latent shape (year 2 skewed)": dict(anchor_drift=0.0, n_drift=0, group_means=[0.0, 0.3],
                                                 shapes=["normal", "skewed"]),
}


def drift_spec(model, scenario, n=1000, n_anchor=6, n_items=10, ncat=4, seed=77):
    s = DRIFT_SCENARIOS[scenario]
    return SimSpec(model=model, design="mgfipc", n_per_group=n, n_items=n_items, n_common=0, n_anchor=n_anchor,
                   ncat=ncat, groups=2, group_means=s["group_means"], group_sds=[1.0, 1.0],
                   group_shapes=s["shapes"], anchor_drift=s["anchor_drift"], n_drift=s["n_drift"], seed=seed)


def drift_analysis(data: Dataset, model, prior="estimated_normal", n_nodes=41):
    """Compare calibrations of the same two-administration data and the anchors' displacement.

    * Free concurrent: anchors treated as ordinary common items (year 1 = reference N(0,1)).
    * FIPC: year 2 only, anchors fixed at bank values.
    * MG-FIPC: both years, anchors fixed.
    Displacement: each anchor re-estimated freely in year 2 with the other anchors fixed (leave-one-out)."""
    out = []
    true_g = data.truth["groups"] if data.truth else None
    designs = [("Free concurrent", data, "concurrent"), ("FIPC (year 2 only)", data.subset_groups([1]), "fipc"),
               ("MG-FIPC", data, "mgfipc")]
    fits = {}
    for label, d, design in designs:
        cfg = FitConfig(model=model, design=design, prior=prior, n_nodes=n_nodes, max_cycles=1000)
        res = fit(d, cfg, record=False)
        fits[label] = res
        mm = res.group_moments
        g2 = mm[-1]
        out.append(dict(calibration=label, year2_mean=g2[0], year2_sd=g2[1],
                        true_year2_mean=true_g[1]["mean"] if true_g else np.nan,
                        mean_error=g2[0] - true_g[1]["mean"] if true_g else np.nan,
                        converged=res.converged, cycles=res.cycles))
    # Leave-one-out displacement on year-2 data.
    y2 = data.subset_groups([1])
    disp = []
    drifted = set(data.truth.get("drifted", [])) if data.truth else set()
    for aid in list(y2.anchors):
        d = Dataset(y2.name, y2.Y, y2.group, y2.group_names, y2.specs,
                    anchors={k: v for k, v in y2.anchors.items() if k != aid}, truth=y2.truth)
        res = fit(d, FitConfig(model=model, design="fipc", prior=prior, n_nodes=n_nodes, max_cycles=1000), record=False)
        est = res.items[d.ids.index(aid)]
        bank = data.anchors[aid]
        gen = data.truth["generating"][aid] if data.truth else bank
        disp.append(dict(anchor=aid, drifted=aid in drifted, bank_location=bank.location(),
                         estimated_location=est.location(), estimated_drift=est.location() - bank.location(),
                         generating_drift=gen.location() - bank.location()))
    return pd.DataFrame(out), pd.DataFrame(disp), fits


def density_distance(w1, w2):
    return 0.5 * float(np.abs(np.asarray(w1) - np.asarray(w2)).sum())


def moments_table(res):
    mm = res.group_moments
    im = np.array([moments(res.nodes[g], res.initial_weights[g]) for g in range(len(mm))])
    return pd.DataFrame(dict(group=res.group_names, start_mean=im[:, 0], start_sd=im[:, 1],
                             mean=mm[:, 0], sd=mm[:, 1]))
