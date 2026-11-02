import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREY, ORANGE, data, download, fig_layout, interpret, page_setup, show, table,
                           teaching_tabs)
from rasch.simulation import connectivity_issues, load_csv, simulate, summaries

page_setup("2 · Data & Simulation", "Generate Rasch responses with known truth, or upload operational 0/1 data.")

st.subheader("1 · Generate or upload")
gen_tab, up_tab = st.tabs(["Simulate", "Upload CSV"])
with gen_tab:
    teaching_tabs([r"\theta_i\sim N(\mu_\theta,\sigma_\theta^2),\qquad b_j\sim N(\mu_b,\sigma_b^2)",
                   r"X_{ij}\sim\operatorname{Bernoulli}(P_{ij}),\qquad P_{ij}=\frac{e^{\theta_i-b_j}}{1+e^{\theta_i-b_j}}"],
                  "Abilities and difficulties are drawn first; each response is then a coin flip with the model "
                  "probability. Optional missingness is applied completely at random afterwards; missing "
                  "responses are excluded from estimation, never scored as wrong. The same seed always gives "
                  "the same data.",
                  "- $N$ persons, $J$ items\n- $\\mu_\\theta,\\sigma_\\theta$: ability mean/SD\n"
                  "- $\\mu_b,\\sigma_b$: difficulty mean/SD")
    with st.form("simulate"):
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Persons N", 50, 5000, 500, 50)
        J = c2.number_input("Items J", 5, 100, 20, 1)
        seed = c3.number_input("Random seed", 0, 2**31 - 1, 2026)
        missing = c4.slider("Missing response proportion", 0.0, 0.20, 0.0, 0.01)
        c1, c2, c3, c4 = st.columns(4)
        tm = c1.slider("Ability mean", -2.0, 2.0, 0.0, 0.1)
        ts = c2.slider("Ability SD", 0.3, 2.0, 1.0, 0.1)
        bm = c3.slider("Difficulty mean", -2.0, 2.0, 0.0, 0.1)
        bsd = c4.slider("Difficulty SD", 0.3, 2.0, 1.0, 0.1)
        st.caption("Tip: a short test (J ≈ 8) or a large gap between ability and difficulty means creates many "
                   "zero/perfect scores — useful for seeing how estimators treat extremes.")
        go_sim = st.form_submit_button("Generate new dataset", type="primary")
    if go_sim:
        st.session_state.dataset = simulate(int(n), int(J), tm, ts, bm, bsd, missing, int(seed))
        st.success(f"Generated dataset {data().identity}. Calibrations of the previous data are no longer shown.")
with up_tab:
    st.write("Rows = persons, columns = items. Cells: 0, 1, or blank (missing). An optional person-identifier "
             "column must be declared so it is never treated as an item.")
    up = st.file_uploader("Responses CSV", type="csv")
    if up is not None:
        preview = pd.read_csv(up)
        up.seek(0)
        id_col = st.selectbox("Person identifier column", ["(none)"] + list(preview.columns.astype(str)))
        if st.button("Load uploaded data", type="primary"):
            try:
                st.session_state.dataset = load_csv(up.getvalue(), None if id_col == "(none)" else id_col)
                st.success(f"Loaded {data().n_persons} persons × {data().n_items} items. "
                           "No true parameters exist for uploaded data, so recovery statistics are not shown.")
            except ValueError as e:
                st.error(f"Upload rejected: {e}")

d = data()
s = summaries(d)
st.subheader("2 · Current dataset")
c = st.columns(6)
c[0].metric("Persons", f"{d.n_persons:,}")
c[1].metric("Items", d.n_items)
c[2].metric("Missing", f"{s['missing_rate']:.1%}")
c[3].metric("Zero scores", s["zero_scores"])
c[4].metric("Perfect scores", s["perfect_scores"])
c[5].metric("Items all right/wrong", s["items_all_correct"] + s["items_all_wrong"])
for issue in connectivity_issues(d):
    st.error(issue)

left, right = st.columns(2)
with left:
    order = np.argsort(s["raw_scores"] / s["n_observed"])[::-1][:100]
    z = np.where(np.isnan(d.X[order]), -1, d.X[order])
    fig = go.Figure(go.Heatmap(z=z, x=d.item_ids, colorscale=[[0, "#e9ecf2"], [0.5, "#fafcff"], [1, BLUE]], zmin=-1,
                               zmax=1, colorbar=dict(tickvals=[-1, 0, 1], ticktext=["missing", "0", "1"])))
    show(fig_layout(fig, "Response matrix (top 100 persons by proportion correct)", "Item", "Person", 360, False))
with right:
    fig = go.Figure(go.Histogram(x=s["raw_scores"], marker_color=BLUE, nbinsx=int(d.n_items) + 1))
    show(fig_layout(fig, "Raw-score distribution (observed items)", "Raw score", "Persons", 360, False))
left, right = st.columns(2)
with left:
    fig = go.Figure(go.Bar(x=d.item_ids, y=s["p_values"], marker_color=ORANGE))
    show(fig_layout(fig, "Item p-values (proportion correct)", "Item", "p-value", 320, False))
with right:
    if d.has_truth:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=d.true_theta, name="True ability θ", marker_color=BLUE, opacity=.6, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=d.true_b, name="True difficulty b", marker_color=ORANGE, opacity=.6, histnorm="probability density"))
        show(fig_layout(fig.update_layout(barmode="overlay"), "True ability and difficulty distributions", "Logits", "Density", 320))
    else:
        st.info("Uploaded data: generating parameters are unknown, so no true distributions are shown.")
miss_by_item = 1 - d.mask.mean(axis=0)
table(pd.DataFrame(dict(item=d.item_ids, p_value=s["p_values"], observed=d.mask.sum(axis=0), missing_rate=miss_by_item,
                        **({"true_b": d.true_b} if d.has_truth else {}))), height=260)
interpret(f"Persons with zero ({s['zero_scores']}) or perfect ({s['perfect_scores']}) raw scores carry no information "
          "about item differences under CMLE and have infinite unadjusted maximum-likelihood abilities. JMLE adjusts "
          "their scores, MML-EM uses them through the ability distribution, and WLE gives them finite abilities. "
          "Items everyone gets right or wrong cannot be calibrated by any method.")
frame = d.frame().reset_index(names="person")
download(frame, f"rasch_responses_{d.identity}.csv", "Download responses CSV")
if d.has_truth:
    download(pd.DataFrame(dict(item=d.item_ids, true_b=d.true_b)), f"rasch_true_items_{d.identity}.csv",
             "Download true item difficulties")
    download(pd.DataFrame(dict(person=d.person_ids, true_theta=d.true_theta)), f"rasch_true_persons_{d.identity}.csv",
             "Download true abilities")
