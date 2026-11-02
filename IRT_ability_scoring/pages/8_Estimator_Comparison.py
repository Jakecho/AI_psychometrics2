import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.ui import (GREY, METHOD_COLORS, cached, data, download, fig_layout, interpret, labels, page_setup,
                           pattern_text, person, settings, show, table)
from r_bridge.compare import compare_all
from scoring import FUNCTIONS, METHODS, UNCERTAINTY, score_all
from scoring.extras import adjusted_mle, plausible_values, posterior_median, prox

page_setup("8 · Estimator Comparison", "The same examinees, item parameters and settings — every scoring method side by side.")
d, s = data(), settings()
p = person()
y = d.Y[p]
items = d.items
st.markdown(f"Examinee **{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · {d.name}",
            unsafe_allow_html=True)
tabs = st.tabs(["Selected examinee", "Whole dataset", "Sensitivity", "SE vs posterior SD", "Advanced methods",
                "R validation"])

with tabs[0]:
    labels("edu")
    one = score_all(items, y[None, :], s)
    rows = []
    for m in METHODS:
        r = one[m]
        rows.append(dict(method=m, theta=r.theta[0], uncertainty=r.se[0], uncertainty_type=UNCERTAINTY[m],
                         convergence=r.status[0], iterations=int(r.iterations[0]) if r.iterations[0] else "N/A"))
    frame = pd.DataFrame(rows)
    table(frame, 4)
    download(frame, "estimates_selected_examinee.csv")
    fin = frame[np.isfinite(frame["theta"].astype(float))]
    fig = go.Figure()
    for _, row in fin.iterrows():
        u = row["uncertainty"] if np.isfinite(row["uncertainty"]) else 0
        fig.add_trace(go.Scatter(x=[row["theta"]], y=[row["method"]], mode="markers", name=row["method"],
                                 marker=dict(color=METHOD_COLORS[row["method"]], size=11),
                                 error_x=dict(type="data", array=[1.96 * u], color=METHOD_COLORS[row["method"]], thickness=2)))
    if d.has_truth:
        fig.add_vline(x=d.theta_true[p], line=dict(color=GREY, dash="dot"), annotation_text="true θ")
    show(fig_layout(fig, "Estimate ± 1.96 × uncertainty", "θ", "", 320, legend=False))
    ml = one["MLE"].theta[0]
    missing = [m for m in METHODS if not np.isfinite(one[m].theta[0])]
    interpret(("Methods without a finite estimate for this pattern: " + ", ".join(missing) + ". " if missing else "")
              + (f"Differences from the MLE ({ml:.3f}): " + ", ".join(
                  f"{m} {one[m].theta[0] - ml:+.3f}" for m in METHODS if m != "MLE" and np.isfinite(one[m].theta[0]))
                 + ". " if np.isfinite(ml) else "")
              + "MAP, EAP and EAPsum are pulled toward the prior mean; WLE is pulled toward the test's centre by Warm's "
                "correction; TCC and EAPsum ignore which items were answered correctly.")

with tabs[1]:
    labels("edu")
    allres = cached("score_all", lambda: score_all(items, d.Y, s))
    est = pd.DataFrame({m: allres[m].theta for m in METHODS})
    unc = pd.DataFrame({m: allres[m].se for m in METHODS})
    raw = np.nansum(d.Y, 1)
    c = st.columns(len(METHODS))
    for k, m in enumerate(METHODS):
        c[k].metric(f"{m}: finite", f"{np.isfinite(est[m]).mean():.1%}", help=f"Share of examinees with a finite {m}")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for m in METHODS:
            v = est[m][np.isfinite(est[m])]
            fig.add_trace(go.Box(y=v, name=m, marker_color=METHOD_COLORS[m], boxmean=True))
        show(fig_layout(fig, "Estimates by method (finite values)", "", "θ̂", 340, legend=False))
    with right:
        fig = go.Figure()
        for m in METHODS:
            g = pd.DataFrame(dict(raw=raw, v=est[m]))
            g = g[np.isfinite(g["v"])].groupby("raw")["v"].mean()
            fig.add_trace(go.Scatter(x=g.index, y=g.values, mode="lines+markers", name=m, line=dict(color=METHOD_COLORS[m])))
        show(fig_layout(fig, "Mean estimate by summed score", "summed score", "mean θ̂", 340))
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        base = est["MLE"]
        for m in METHODS[1:]:
            ok = np.isfinite(base) & np.isfinite(est[m])
            fig.add_trace(go.Scatter(x=base[ok], y=(est[m] - base)[ok], mode="markers", name=m,
                                     marker=dict(color=METHOD_COLORS[m], size=6, opacity=0.6)))
        fig.add_hline(y=0, line=dict(color=GREY))
        show(fig_layout(fig, "Difference from the MLE", "MLE", "estimate − MLE", 340))
    with right:
        fig = go.Figure()
        for m in METHODS:
            g = pd.DataFrame(dict(raw=raw, u=unc[m], v=est[m]))
            g = g[np.isfinite(g["u"]) & np.isfinite(g["v"])].groupby("raw")["u"].mean()
            fig.add_trace(go.Scatter(x=g.index, y=g.values, mode="lines+markers", name=m, line=dict(color=METHOD_COLORS[m])))
        show(fig_layout(fig, "Reported uncertainty by summed score", "summed score", "mean SE / PSD", 340))
    if d.has_truth:
        st.markdown("**Recovery of the generating abilities** (this one dataset — see the Monte Carlo page for repeated samples)")
        fig = make_subplots(rows=2, cols=3, subplot_titles=METHODS, shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=0.12, horizontal_spacing=0.04)
        rec = []
        for k, m in enumerate(METHODS):
            ok = np.isfinite(est[m])
            e = est[m][ok] - d.theta_true[ok]
            rec.append(dict(method=m, n_finite=int(ok.sum()), bias=e.mean(), RMSE=np.sqrt((e ** 2).mean()),
                            MAD=np.abs(e).mean(), correlation=np.corrcoef(est[m][ok], d.theta_true[ok])[0, 1]))
            fig.add_trace(go.Scatter(x=d.theta_true[ok], y=est[m][ok], mode="markers", name=m,
                                     marker=dict(color=METHOD_COLORS[m], size=4, opacity=0.5)), row=k // 3 + 1, col=k % 3 + 1)
            fig.add_trace(go.Scatter(x=[-3.5, 3.5], y=[-3.5, 3.5], mode="lines", line=dict(color=GREY, dash="dot"),
                                     showlegend=False), row=k // 3 + 1, col=k % 3 + 1)
        fig.update_layout(height=520, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(range=[-3.5, 3.5], title_text="true θ", row=2)
        fig.update_yaxes(range=[-4, 4])
        show(fig)
        table(pd.DataFrame(rec), 4)
        bins = pd.cut(d.theta_true, [-np.inf, -2, -1, 0, 1, 2, np.inf])
        cond = []
        for m in METHODS:
            ok = np.isfinite(est[m])
            g = pd.DataFrame(dict(bin=bins[ok].astype(str), e=(est[m] - d.theta_true)[ok])).groupby("bin", sort=False)["e"].mean()
            cond.append(go.Bar(x=g.index, y=g.values, name=m, marker_color=METHOD_COLORS[m]))
        fig = go.Figure(cond)
        fig.update_xaxes(categoryorder="array", categoryarray=[str(c) for c in pd.cut(pd.Series([0.0]), [-np.inf, -2, -1, 0, 1, 2, np.inf]).cat.categories])
        fig.add_hline(y=0, line=dict(color=GREY))
        show(fig_layout(fig, "Mean error by true-ability range (finite estimates only)", "true θ range", "mean(θ̂ − θ)", 340))
        st.caption("Shrinkage shows as positive error at low θ and negative error at high θ (EAP, MAP, EAPsum); the "
                   "MLE's outward bias shows the opposite pattern. Extreme patterns without a finite MLE/TCC are excluded "
                   "from those two methods, which favours them at the extremes.")
    export = pd.DataFrame(dict(examinee=d.person_ids, summed_score=raw))
    for m in METHODS:
        export[m] = est[m]
        export[f"{m}_uncertainty"] = unc[m]
    if d.has_truth:
        export["true_theta"] = d.theta_true
    download(export, "estimates_all_methods.csv", "Download all estimates (CSV)")

with tabs[2]:
    labels("edu")
    st.markdown("**Prior sensitivity for the selected examinee** (Bayesian methods only; MLE, WLE and TCC do not use a prior)")
    rows = []
    for sd_ in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        ss = s.with_(prior_sd=sd_)
        rows.append(dict(prior_sd=sd_, **{m: FUNCTIONS[m](items, y[None, :], ss).theta[0] for m in ("EAP", "MAP", "EAPsum")}))
    sens = pd.DataFrame(rows)
    fig = go.Figure([go.Scatter(x=sens["prior_sd"], y=sens[m], mode="lines+markers", name=m, line=dict(color=METHOD_COLORS[m]))
                     for m in ("EAP", "MAP", "EAPsum")])
    for m in ("MLE", "WLE"):
        v = FUNCTIONS[m](items, y[None, :], s).theta[0]
        if np.isfinite(v):
            fig.add_hline(y=v, line=dict(color=METHOD_COLORS[m], dash="dot"), annotation_text=m)
    show(fig_layout(fig, "Estimate as the prior SD changes", "prior SD", "θ̂", 320))
    st.markdown("**Extreme-score behaviour**: patterns with every response in the lowest or highest category")
    E = np.vstack([np.zeros(len(items)), [it.max_score for it in items]])
    er = score_all(items, E, s)
    table(pd.DataFrame([dict(pattern=lab, **{m: er[m].theta[k] for m in METHODS}) for k, lab in
                        enumerate(["all lowest", "all highest"])]), 4)
    st.caption(f"MLE and TCC have no finite value (±∞). WLE is finite without a prior; MAP/EAP/EAPsum are finite because of "
               f"the prior N({s.prior_mean:g}, {s.prior_sd:g}²).")
    if d.has_truth and d.n_items >= 4:
        st.markdown("**Test-length effect**: score the same examinees with only the first k items")
        ks = sorted({max(2, int(round(f * d.n_items))) for f in (0.2, 0.35, 0.5, 0.75, 1.0)})

        def length_study():
            out = []
            for k in ks:
                sub = score_all(items[:k], d.Y[:, :k], s)
                for m in METHODS:
                    ok = np.isfinite(sub[m].theta)
                    e = sub[m].theta[ok] - d.theta_true[ok]
                    out.append(dict(items=k, method=m, RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(),
                                    mean_uncertainty=np.nanmean(np.where(np.isfinite(sub[m].se), sub[m].se, np.nan)[ok]),
                                    finite=ok.mean()))
            return pd.DataFrame(out)
        ls = cached("length", length_study, ks)
        left, right = st.columns(2)
        for col, metric, title in ((left, "RMSE", "RMSE against true θ"), (right, "finite", "Share with a finite estimate")):
            with col:
                fig = go.Figure([go.Scatter(x=ls[ls.method == m]["items"], y=ls[ls.method == m][metric], mode="lines+markers",
                                            name=m, line=dict(color=METHOD_COLORS[m])) for m in METHODS])
                show(fig_layout(fig, title, "number of items", metric, 320))
        table(ls, 4, height=260)

with tabs[3]:
    st.markdown("""
| | Frequentist SE (MLE, WLE, TCC) | Bayesian posterior SD (EAP, EAPsum; MAP's curvature SE) |
|---|---|---|
| Question answered | How much would θ̂ vary over repeated testing of **this examinee at a fixed θ**? | How uncertain is θ **given these responses and the population prior**? |
| Computed from | Test information at θ̂: 1/√I(θ̂) (TCC: delta method) | Spread of the posterior (or its curvature at the mode) |
| Depends on the prior | No | Yes — a narrower prior gives a smaller PSD |
| Extreme patterns | Undefined for MLE/TCC (no finite θ̂) | Always finite |
| Interval interpretation | θ̂ ± 1.96 SE covers the fixed true θ in ≈95% of repetitions (large-sample) | 95% posterior interval: 95% posterior probability, averaged over the population |

A posterior SD is typically **smaller** than the SE at the same location because the prior adds information
(1/σ² for MAP). That does not make Bayesian estimates "more precise" for an individual far from the prior mean: their
bias toward μ is not part of the PSD. The Monte Carlo page reports coverage separately for each method.
""")

with tabs[4]:
    st.warning("Optional modules. Plausible values are random draws for **population-level** analysis (means, variances, "
               "regressions); they are not point estimates and must not be reported for individual examinees.")
    labels("edu")
    pm = posterior_median(items, y[None, :], s)
    am = adjusted_mle(items, y[None, :], s)
    pv = plausible_values(items, y[None, :], s, draws=5, seed=11)[0]
    rows = [dict(method="Posterior median", theta=pm.theta[0], note=f"95% posterior interval [{pm.extra['lower'][0]:.3f}, {pm.extra['upper'][0]:.3f}]"),
            dict(method="Adjusted MLE (extreme score ± 0.3)", theta=am.theta[0], note=str(am.status[0])),
            dict(method="Plausible values (5 draws)", theta=np.nan, note=", ".join(f"{v:.3f}" for v in pv))]
    if all(it.model == "Rasch" for it in items):
        rows.append(dict(method="Rasch PROX", theta=prox(items, y[None, :])[0],
                         note="normal approximation from the raw score and item difficulties"))
    else:
        rows.append(dict(method="Rasch PROX", theta=np.nan, note="not applicable: PROX needs dichotomous Rasch items"))
    table(pd.DataFrame(rows), 4)
    st.markdown("**Why plausible values, not EAPs, for population statistics** — whole dataset, 5 draws per examinee")

    def pv_study():
        P = plausible_values(items, d.Y, s, draws=5, seed=3)
        e = FUNCTIONS["EAP"](items, d.Y, s).theta
        out = [dict(statistic="variance of EAPs", value=float(np.var(e, ddof=1))),
               dict(statistic="mean over draws of the variance of plausible values",
                    value=float(np.mean([np.var(P[:, k], ddof=1) for k in range(P.shape[1])])))]
        if d.has_truth:
            out.append(dict(statistic="variance of the generating θ", value=float(np.var(d.theta_true, ddof=1))))
        return pd.DataFrame(out)
    table(cached("pv", pv_study), 4)
    st.caption("EAPs are shrunk, so their variance understates the population variance; plausible values, drawn from "
               "each posterior, reproduce it (when the prior matches the population).")

with tabs[5]:
    labels("ref", "edu")
    st.markdown("Score **every examinee** in the current dataset with irtQ (primary), mirt and TAM under matched settings "
                "and compare with the Python results. Packages that cannot score a model report *Unavailable*.")
    n_max = st.number_input("Examinees to include (first n)", 1, d.n_persons, min(d.n_persons, 300), key="val_n")
    if st.button("Run the reference comparison", type="primary"):
        with st.spinner("Running irtQ, mirt and TAM via Rscript…"):
            _, rows = compare_all(items, d.Y[: int(n_max)], s)
        st.session_state["val_rows"] = (rows, d.identity, s.as_dict(), int(n_max))
    saved = st.session_state.get("val_rows")
    if saved and saved[1:] == (d.identity, s.as_dict(), int(n_max)):
        frame = pd.DataFrame(saved[0])
        table(frame, 6)
        download(frame, "reference_comparison.csv")
        bad = frame[frame["result"] == "Fail"]
        st.markdown(("**All executed comparisons are within tolerance.**" if bad.empty else
                     f"**{len(bad)} comparison(s) outside tolerance** — see the rows marked Fail."))
    elif saved:
        st.caption("The dataset or settings changed since the last run; run the comparison again.")
