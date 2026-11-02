import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (GREY, METHOD_COLORS, download, fig_layout, interpret, labels, page_setup, settings, show, signature,
                           table, teaching_tabs)
from irt_models import DICHOTOMOUS, POLYTOMOUS
from scoring import METHODS
from simulation import monte_carlo

page_setup("9 · Monte Carlo Simulation", "Evaluate the scoring methods against known generating abilities over repeated samples.")
s = settings()

teaching_tabs([r"\text{Bias}=\frac1N\sum_i(\hat\theta_i-\theta_i),\quad \text{RMSE}=\sqrt{\frac1N\sum_i(\hat\theta_i-\theta_i)^2},"
               r"\quad \text{MAD}=\frac1N\sum_i|\hat\theta_i-\theta_i|",
               r"\text{MCSE}(\overline{\text{Bias}})=\frac{SD_r(\text{Bias}_r)}{\sqrt R},\qquad "
               r"\text{coverage}=\frac1N\sum_i\mathbf 1\{|\hat\theta_i-\theta_i|\le1.96\,u_i\}"],
              "Each replication draws new abilities (and optionally new items), simulates responses, and scores them with "
              "every selected method. Statistics are computed per replication and then averaged; the Monte Carlo SE "
              "(MCSE) tells you how much those averages would move with a different simulation seed. **Non-finite "
              "estimates** (MLE / TCC at extreme patterns) cannot enter bias or RMSE: they are excluded and their rate is "
              "reported, which flatters those methods at the extremes.",
              "- θ_i: generating ability · θ̂_i: estimate · u_i: the method's reported uncertainty\n"
              "- R: completed replications · conditional statistics: pooled over replications within bins of true θ")

st.subheader("A · Design")
preset = st.radio("Preset", ["Quick demonstration", "Short test (shrinkage vs. bias)", "Research (slower)"], horizontal=True)
P = {"Quick demonstration": dict(n=300, reps=10, nd=15, np_=5),
     "Short test (shrinkage vs. bias)": dict(n=500, reps=20, nd=8, np_=0),
     "Research (slower)": dict(n=2000, reps=100, nd=30, np_=10)}[preset]
with st.form("mc"):
    c1, c2, c3, c4 = st.columns(4)
    n_persons = c1.number_input("Examinees per replication", 20, 20000, P["n"], 50)
    reps = c1.number_input("Replications", 1, 1000, P["reps"])
    seed = c1.number_input("Random seed", 0, 10**6, 7)
    dist = c2.selectbox("True ability distribution", ["normal", "uniform", "skewed"])
    tm = c2.number_input("Ability mean", -3.0, 3.0, 0.0, 0.1)
    tsd = c2.number_input("Ability SD", 0.2, 3.0, 1.0, 0.1)
    n_dich = c3.number_input("Dichotomous items", 0, 200, P["nd"])
    dich_model = c3.selectbox("Dichotomous model", DICHOTOMOUS, index=1)
    n_poly = c3.number_input("Polytomous items", 0, 100, P["np_"])
    poly_model = c4.selectbox("Polytomous model", POLYTOMOUS, index=1)
    n_cat = c4.number_input("Categories", 2, 7, 4)
    missing = c4.slider("Missing-response rate", 0.0, 0.5, 0.0, 0.05)
    e1, e2, e3, e4 = st.columns(4)
    b_mean = e1.number_input("Difficulty mean", -3.0, 3.0, 0.0, 0.1)
    b_sd = e1.number_input("Difficulty SD", 0.0, 3.0, 1.0, 0.1)
    a_sdlog = e2.number_input("log-discrimination SD", 0.0, 1.0, 0.25, 0.05)
    spread = e2.number_input("Step / threshold spread", 0.1, 3.0, 1.0, 0.1)
    prior_mean = e3.number_input("Prior mean (EAP, MAP, EAPsum)", -3.0, 3.0, float(s.prior_mean), 0.1)
    prior_sd = e3.number_input("Prior SD", 0.1, 5.0, float(s.prior_sd), 0.1)
    methods = e4.multiselect("Methods", METHODS, default=list(METHODS))
    new_items = e4.checkbox("Draw new items every replication", value=False)
    limit = e4.number_input("Time limit (s, 0 = none)", 0, 3600, 120)
    run = st.form_submit_button("Run simulation", type="primary")
design = dict(n_persons=int(n_persons), replications=int(reps), seed=int(seed), theta_dist=dist, theta_mean=tm, theta_sd=tsd,
              missing_rate=missing, new_items=new_items, n_dich=int(n_dich), dich_model=dich_model, n_poly=int(n_poly),
              poly_model=poly_model, n_cat=int(n_cat), b_mean=b_mean, b_sd=b_sd, a_sdlog=a_sdlog, step_spread=spread)
ss = s.with_(prior_mean=prior_mean, prior_sd=prior_sd)
sig = signature([design, ss.as_dict(), methods])
if run:
    if not methods or n_dich + n_poly == 0:
        st.error("Choose at least one method and one item.")
    else:
        with st.spinner("Simulating and scoring…"):
            out = monte_carlo.run(methods, ss, time_limit=limit or None, **design)
        st.session_state["mc_result"] = dict(sig=sig, out=out, methods=methods, design=design, settings=ss.as_dict())
saved = st.session_state.get("mc_result")
if not saved:
    st.info("Choose a design and press **Run simulation**. The quick preset finishes in a few seconds.")
    st.stop()
if saved["sig"] != sig:
    st.warning("The design or settings changed since the results below were computed; press **Run simulation** to update.")
out = saved["out"]
methods = saved["methods"]
summ = out["summary"]

st.subheader("B · Results")
labels("edu")
st.caption(f"{out['completed']} of {out['requested']} replication(s) completed in {out['seconds']:.1f} s"
           + (" (stopped at the time limit)" if out["completed"] < out["requested"] else "")
           + f" · {saved['design']['n_persons']} examinees each · prior N({saved['settings']['prior_mean']:g}, "
             f"{saved['settings']['prior_sd']:g}²) · items fixed across replications: {not saved['design']['new_items']}")
table(summ, 4)
left, right = st.columns(2)
for col, metric, title in ((left, "bias", "Bias ± 1.96 MCSE"), (right, "rmse", "RMSE ± 1.96 MCSE")):
    with col:
        err = summ.get(f"mcse_{metric}", pd.Series(np.zeros(len(summ)))).fillna(0)
        fig = go.Figure(go.Bar(x=summ["method"], y=summ[metric], marker_color=[METHOD_COLORS[m] for m in summ["method"]],
                               error_y=dict(type="data", array=1.96 * err)))
        fig.add_hline(y=0, line=dict(color=GREY))
        show(fig_layout(fig, title, "", metric, 300, legend=False))
cond = out["conditional"]
left, right = st.columns(2)
for col, metric, title in ((left, "bias", "Conditional bias by true θ"), (right, "rmse", "Conditional RMSE by true θ")):
    with col:
        fig = go.Figure([go.Scatter(x=cond[cond.method == m]["theta_bin"], y=cond[cond.method == m][metric],
                                    mode="lines+markers", name=m, line=dict(color=METHOD_COLORS[m])) for m in methods])
        fig.add_hline(y=0, line=dict(color=GREY))
        show(fig_layout(fig, title, "true θ range", metric, 320))
left, right = st.columns(2)
with left:
    fig = go.Figure(go.Bar(x=summ["method"], y=summ["coverage"], marker_color=[METHOD_COLORS[m] for m in summ["method"]],
                           error_y=dict(type="data", array=1.96 * summ.get("mcse_coverage", pd.Series(np.zeros(len(summ)))).fillna(0))))
    fig.add_hline(y=0.95, line=dict(color=GREY, dash="dot"), annotation_text="0.95")
    fig.update_yaxes(range=[0, 1.05])
    show(fig_layout(fig, "Coverage of θ̂ ± 1.96 × reported uncertainty", "", "coverage", 300, legend=False))
with right:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summ["method"], y=summ["emp_sd"], name="empirical SD of errors", marker_color="#9fbbea"))
    fig.add_trace(go.Bar(x=summ["method"], y=summ["mean_uncertainty"], name="mean reported uncertainty", marker_color="#17273f"))
    show(fig_layout(fig, "Is the reported uncertainty honest?", "", "logits", 300))
table(summ[["method", "failure_rate", "nonfinite_rate", "seconds", "replications"]], 4)

st.subheader("C · Agreement with a reference estimator (not with truth)")
ref = st.selectbox("Reference estimator", methods, index=0)
pers = out["persons"]
wide = pers.pivot_table(index=[pers["replication"], pers.groupby(["replication", "method"]).cumcount()], columns="method",
                        values="estimate")
agree = []
for m in methods:
    ok = np.isfinite(wide[m]) & np.isfinite(wide[ref])
    dlt = (wide[m] - wide[ref])[ok]
    agree.append(dict(method=m, compared=int(ok.sum()), mean_difference=dlt.mean(), rms_difference=np.sqrt((dlt ** 2).mean())))
table(pd.DataFrame(agree), 4)
st.caption("Agreement measures how similar two methods are; it says nothing about which is closer to the true θ. "
           "Section B is the evaluation against the generating abilities.")
best = summ.loc[summ["rmse"].idxmin(), "method"]
shrink = [m for m in ("EAP", "MAP", "EAPsum") if m in methods]
interpret(f"Lowest RMSE: **{best}** ({summ['rmse'].min():.3f}). "
          + (f"The Bayesian methods ({', '.join(shrink)}) usually win on RMSE because shrinkage trades a little bias for a "
             "large variance reduction; their conditional bias is positive for low θ and negative for high θ. "
             if shrink else "")
          + ("MLE and TCC lose extreme patterns (non-finite rate in the last table), which hides their worst cases. "
             if any(m in methods for m in ("MLE", "TCC")) else "")
          + "Coverage near 0.95 means the reported uncertainty matches the actual error spread; Bayesian intervals reach "
            "it on average over the population, not for every θ.")

st.subheader("D · Downloads")
c1, c2, c3 = st.columns(3)
with c1:
    download(summ, "mc_summary.csv", "Summary (CSV)", key="dl1")
with c2:
    download(out["replications"], "mc_replications.csv", "Per replication (CSV)", key="dl2")
with c3:
    download(pers, "mc_person_estimates.csv", "Every estimate (CSV)", key="dl3")
st.caption("Reproducible: the same design, settings and seed give identical results.")
