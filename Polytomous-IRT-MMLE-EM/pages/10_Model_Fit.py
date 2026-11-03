import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, ORANGE, config, data, fig_layout, guided, interpret, latest_fit, lines, page_setup,
                           show, table)
from core.diagnostics import category_fit, fit_summary, information_criteria, summed_score_fit
from core.em import fit
from core.models import probs
from core.priors import PRIOR_NAMES

page_setup("10 · Model Fit", "Observed vs model-implied category proportions, residuals, summed-score distributions and "
                             "information criteria — including normal vs empirical prior comparisons.")
guided(["Read the observed-vs-expected scatter: points near the diagonal mean the marginal proportions are reproduced.",
        "Sort the residual table: |standardised residual| > 3 flags a category the model misses.",
        "Check the summed-score distribution — sensitive to latent-distribution shape.",
        "Fit a normal and an empirical prior and compare AIC/BIC and the summed-score fit."],
       "If observed and predicted category proportions agree closely, the model…",
       ["may still misfit — marginal agreement is necessary, not sufficient",
        "is proven to fit", "has no latent-distribution misspecification", "must be the GPCM"],
       "may still misfit — marginal agreement is necessary, not sufficient",
       "Item-level marginals are often reproduced almost exactly by MML; misfit shows in conditional and joint patterns "
       "(summed scores, item pairs, θ-conditional curves).", key="fit")
d = data()
res = latest_fit()
if res is None:
    st.info("Run a calibration on page 7 first.")
    st.stop()
tab = category_fit(d, res)
s = fit_summary(tab)
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE (proportions)", f"{s['RMSE']:.4f}")
c2.metric("MAD", f"{s['MAD']:.4f}")
c3.metric("Largest |difference|", f"{s['max_abs']:.4f}")
c4.metric("|std. residual| > 3", s["large_residuals"])
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_scatter(x=tab["expected"], y=tab["observed"], mode="markers", marker=dict(color=BLUE, size=7),
                    text=[f"{r.item} k={r.category} {r.group}" for r in tab.itertuples()], name="item × category")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#8a8f98", dash="dash"), name="y = x")
    show(fig_layout(fig, "Observed vs predicted category proportions", "predicted p̂_gjk", "observed"), "sc")
with c2:
    fig = go.Figure()
    fig.add_bar(x=[f"{r.item}·{r.category}" + (f"·{r.group}" if d.G > 1 else "") for r in tab.itertuples()],
                y=tab["std_residual"], marker_color=np.where(np.abs(tab["std_residual"]) > 3, ORANGE, BLUE))
    show(fig_layout(fig, "Standardised category residuals", "item·category", "z", legend=False), "res")
table(tab.sort_values("std_residual", key=np.abs, ascending=False), digits=4, height=280)

st.subheader("Summed-score distribution")
g = st.selectbox("Group", range(d.G), format_func=lambda g: d.group_names[g])
ss_fit = summed_score_fit(d, res, g)
if len(ss_fit):
    lines(ss_fit["score"].to_numpy(), {"observed": ss_fit["observed"].to_numpy(), "expected": ss_fit["expected"].to_numpy()},
          f"Observed vs expected summed scores ({int(ss_fit['n'].iloc[0])} complete examinees)", xl="summed score",
          yl="proportion", key="ss")
    n = int(ss_fit["n"].iloc[0])
    e = ss_fit["expected"].to_numpy() * n
    o = ss_fit["observed"].to_numpy() * n
    keep = e > 1e-9
    st.caption(f"Pearson X² over score points = {np.sum((o[keep] - e[keep]) ** 2 / e[keep]):.2f} on up to "
               f"{int(keep.sum()) - 1} cells (descriptive; sparse cells inflate it). Expected uses the Lord–Wingersky "
               "recursion over the fitted latent distribution.")

st.subheader("Item characteristic check")
j = st.selectbox("Item", range(d.J), format_func=lambda j: d.ids[j])
it = res.items[j]
x = res.nodes[0]
P = probs(it, x, res.config.D)
from core.em import estep  # noqa: E402

es = estep(d, res.items, res.nodes, res.weights, res.config.D)
m = d.observed[:, j]
post = es["post"][m]
y = d.Y[m, j].astype(int)
ncount = post.sum(axis=0)
emp = np.stack([post[y == k].sum(axis=0) for k in range(it.ncat)], axis=1) / np.maximum(ncount[:, None], 1e-12)
sel = ncount > 0.01 * m.sum() / len(x)
series = {**{f"model k={k}": P[sel, k] for k in range(it.ncat)}, **{f"posterior-weighted observed k={k}": emp[sel, k]
                                                                 for k in range(it.ncat)}}
lines(x[sel], series, f"Model curves vs posterior-weighted observed proportions · {it.id}", yl="probability",
      dash={f"posterior-weighted observed k={k}": "dot" for k in range(it.ncat)}, key="icc")
st.caption("Posterior-weighted observed proportions use each examinee's posterior over θ (pseudo-counts). Because "
           "the posterior depends on the model, this is a descriptive check, not a formal test.")

st.subheader("Normal vs empirical prior: latent-distribution misspecification")
if st.button("Fit fixed-normal and estimated-empirical priors and compare"):
    rows = []
    for prior in ("fixed_normal", "estimated_normal", "estimated_empirical"):
        try:
            r = fit(d, config(prior=prior, grid="equal"), record=False)
        except ValueError as e:
            rows.append(dict(prior=PRIOR_NAMES[prior], note=str(e)[:100]))
            continue
        ic = information_criteria(r, d.N)
        cf = fit_summary(category_fit(d, r))
        sf = summed_score_fit(d, r, 0)
        if len(sf):
            n = int(sf["n"].iloc[0])
            e, o = sf["expected"].to_numpy() * n, sf["observed"].to_numpy() * n
            k = e > 1e-9
            x2 = float(np.sum((o[k] - e[k]) ** 2 / e[k]))
        else:
            x2 = np.nan
        rows.append(dict(prior=PRIOR_NAMES[prior], loglik=r.loglik, parameters=ic["parameters"], AIC=ic["AIC"],
                         BIC=ic["BIC"], category_RMSE=cf["RMSE"], summed_score_X2=x2, converged=r.converged))
    st.session_state.fit_cmp = (pd.DataFrame(rows), d.identity)
if "fit_cmp" in st.session_state and st.session_state.fit_cmp[1] == d.identity:
    table(st.session_state.fit_cmp[0], digits=4)

info = information_criteria(res, d.N)
st.caption(f"Current fit: −2ℓ = {info['deviance']:.2f}, {info['parameters']} free parameters, AIC = {info['AIC']:.2f}, "
           f"BIC = {info['BIC']:.2f}. Parameter counts include free latent-distribution parameters (Q − 1 − constraints "
           "per free histogram).")
interpret("Close agreement of marginal category proportions alone does not establish fit: MML nearly reproduces them by "
          "construction. A skewed or bimodal true distribution typically shows up first in the summed-score "
          "distribution and in AIC/BIC favouring the empirical prior; item-level misfit shows in θ-conditional curves.")
