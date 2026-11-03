"""Step-by-step EM cycle: every intermediate quantity as a table, computed by the engine functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .em import (FitConfig, build_grid, estep, expected_counts, marginal_loglik, mstep_items, mstep_prior,
                 node_masses, plan, start_items)
from .models import log_probs
from .quadrature import moments

STEPS = ["View response data", "Choose initial item parameters", "Choose prior distribution",
         "Category probabilities", "Response-pattern likelihoods", "Unnormalised posterior",
         "Normalised posterior", "Expected category counts", "Optimise item parameters",
         "Update prior parameters", "Recalculate marginal likelihood", "Evaluate convergence",
         "Continue to the next iteration"]


def walkthrough_config(model="PCM", prior="fixed_normal"):
    """Three-point Gauss–Hermite (θ = −√3, 0, √3; π = 1/6, 2/3, 1/6). Estimated priors use the same three nodes as a
    fixed custom grid starting from the same masses."""
    if prior == "fixed_normal":
        return FitConfig(model=model, prior=prior, grid="gauss_hermite", n_nodes=3)
    nodes = [-np.sqrt(3), 0.0, np.sqrt(3)]
    w = [1 / 6, 2 / 3, 1 / 6]
    cfg = FitConfig(model=model, prior=prior, grid="custom", n_nodes=3, custom_nodes=nodes)
    if prior == "fixed_empirical":
        cfg.fixed_weights = w
    elif prior == "estimated_empirical":
        cfg.initial_weights = w
    return cfg


def default_start(data, model):
    """Readable starting values: steps/thresholds −0.5, 0.5 (spread for more categories), a = 1."""
    from .models import make_item
    items = []
    for s in data.specs:
        K = s.ncat - 1
        d = tuple(np.linspace(-0.5, 0.5, K)) if K > 1 else (0.0,)
        items.append(make_item(s.id, model, 1.0, d))
    if model == "RSM":
        from .models import Item
        tau = np.linspace(-0.5, 0.5, data.specs[0].ncat - 1)
        items = [Item(s.id, "RSM", 1.0, (), b=0.0, tau=tuple(tau - tau.mean())) for s in data.specs]
    return items


def cycle(data: Dataset, cfg: FitConfig, items, weights=None, previous_ll=None):
    """One full EM cycle; returns every intermediate quantity."""
    pl = plan(data, cfg)
    if pl["errors"]:
        raise ValueError(" ".join(pl["errors"]))
    nodes, w0 = build_grid(cfg, data.G)
    weights = w0 if weights is None else np.asarray(weights, float)
    es = estep(data, items, nodes, weights, cfg.D)
    counts = expected_counts(data, es["post"])
    new_items, statuses = mstep_items(data, items, counts, nodes, pl["fixed"], cfg.D)
    n_g = node_masses(data, es["post"])
    if any(u.estimated for u in pl["units"]):
        new_w, pinfo = mstep_prior(pl["units"], n_g, nodes, weights, cfg)
    else:
        new_w, pinfo = weights.copy(), []
    ll_before = es["loglik"]
    ll_after = marginal_loglik(data, new_items, nodes, new_w, cfg.D)
    dp = max(np.max(np.abs(a.vector() - b.vector())) for a, b in zip(items, new_items))
    dw = float(np.max(np.abs(new_w - weights)))
    return dict(nodes=nodes, weights=weights, estep=es, counts=counts, items=items, new_items=new_items,
                statuses=statuses, n_g=n_g, new_weights=new_w, prior_info=pinfo, ll_before=ll_before,
                ll_after=ll_after, item_change=float(dp), weight_change=dw, units=pl["units"], fixed=pl["fixed"],
                previous_ll=previous_ll)


# ---- tables -------------------------------------------------------------------------------------

def table_responses(data):
    f = pd.DataFrame(data.Y, columns=data.ids)
    f.insert(0, "examinee", [f"P{i + 1}" for i in range(data.N)])
    if data.G > 1:
        f.insert(1, "group", [data.group_names[g] for g in data.group])
    return f


def table_items(items):
    return pd.DataFrame([it.as_row() for it in items])


def table_prior(nodes, weights, names):
    rows = []
    for g in range(len(weights)):
        for q in range(nodes.shape[1]):
            rows.append(dict(group=names[g], q=q + 1, theta=nodes[g, q], weight=weights[g, q]))
    return pd.DataFrame(rows)


def table_category_probs(items, nodes, D=1.0):
    rows = []
    x = nodes[0]
    for it in items:
        P = np.exp(log_probs(it, x, D))
        for q, t in enumerate(x):
            r = dict(item=it.id, q=q + 1, theta=t)
            r.update({f"P(k={k})": P[q, k] for k in range(it.ncat)})
            r["sum"] = P[q].sum()
            rows.append(r)
    return pd.DataFrame(rows)


def table_likelihoods(data, items, cyc, D=1.0, limit=None):
    """L_i(θ_q) = Π_j P_{j, y_ij}(θ_q), with the factors written out."""
    nodes = cyc["nodes"]
    rows = []
    for i in range(data.N if limit is None else min(limit, data.N)):
        g = data.group[i]
        for q, t in enumerate(nodes[g]):
            factors = []
            for j, it in enumerate(items):
                if np.isnan(data.Y[i, j]):
                    continue
                k = int(data.Y[i, j])
                factors.append(np.exp(log_probs(it, [t], D)[0, k]))
            rows.append(dict(examinee=f"P{i + 1}", q=q + 1, theta=t,
                             factors=" × ".join(f"{f:.4f}" for f in factors),
                             likelihood=float(np.exp(cyc["estep"]["logL"][i, q]))))
    return pd.DataFrame(rows)


def table_posterior(data, cyc, limit=None):
    es = cyc["estep"]
    w = cyc["weights"]
    rows = []
    for i in range(data.N if limit is None else min(limit, data.N)):
        g = data.group[i]
        L = np.exp(es["logL"][i])
        un = w[g] * L
        for q in range(len(L)):
            rows.append(dict(examinee=f"P{i + 1}", q=q + 1, theta=cyc["nodes"][g, q], prior=w[g, q], likelihood=L[q],
                             unnormalised=un[q], marginal=un.sum(), posterior=es["post"][i, q]))
    return pd.DataFrame(rows)


def table_counts(data, cyc):
    rows = []
    for j, s in enumerate(data.specs):
        C = cyc["counts"][j]
        for g in range(data.G):
            for q in range(C.shape[1]):
                r = dict(item=s.id, group=data.group_names[g], q=q + 1, theta=cyc["nodes"][g, q])
                r.update({f"C(k={k})": C[g, q, k] for k in range(s.ncat)})
                rows.append(r)
    return pd.DataFrame(rows)


def reconcile_counts(data, cyc):
    """Σ_q C_jkq must equal the observed number of responses in category k (per item)."""
    rows = []
    for j, s in enumerate(data.specs):
        tot = cyc["counts"][j].sum(axis=(0, 1))
        obs = data.counts(j)
        for k in range(s.ncat):
            rows.append(dict(item=s.id, category=k, expected_total=tot[k], observed=int(obs[k]),
                             difference=tot[k] - obs[k]))
    return pd.DataFrame(rows)


def count_contributions(data, cyc, j, k, limit=None):
    """Each examinee's contribution I(Y_ij = k) w_iq to C_jkq."""
    post = cyc["estep"]["post"]
    rows = []
    for i in range(data.N if limit is None else min(limit, data.N)):
        if np.isnan(data.Y[i, j]):
            continue
        ind = int(data.Y[i, j] == k)
        r = dict(examinee=f"P{i + 1}", response=int(data.Y[i, j]), indicator=ind)
        r.update({f"q{q + 1}": ind * post[i, q] for q in range(post.shape[1])})
        rows.append(r)
    return pd.DataFrame(rows)


def table_moments(nodes, before, after, names):
    mb = np.array([moments(nodes[g], before[g]) for g in range(len(before))])
    ma = np.array([moments(nodes[g], after[g]) for g in range(len(after))])
    return pd.DataFrame(dict(group=names, mean_before=mb[:, 0], sd_before=mb[:, 1], mean_after=ma[:, 0],
                             sd_after=ma[:, 1]))
