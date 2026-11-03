import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, ORANGE, add_dataset, fig_layout, guided, interpret, page_setup, show, table)
from core.diagnostics import DRIFT_SCENARIOS, drift_analysis, drift_spec
from core.models import MODEL_NAMES, MODELS
from core.priors import PRIOR_NAMES
from core.simulation import simulate

page_setup("11 · Drift Analysis", "Anchor drift, population change and scale maintenance: two administrations, the same "
                                  "data calibrated freely, with fixed anchors, and concurrently with fixed anchors.")
guided(["Pick a scenario: stable anchors, drifting anchors, ability shift, or a changed distribution shape.",
        "Run: three calibrations of the same data plus leave-one-out displacement for every anchor.",
        "Compare the estimated year-2 mean with the generating mean under each calibration.",
        "Compare each anchor's estimated drift (displacement) with its generating drift."],
       "If two anchors become harder in year 2 but are held at their bank values, the year-2 mean tends to be…",
       ["underestimated", "overestimated", "unaffected", "undefined"], "underestimated",
       "Year-2 examinees score lower on the drifted anchors; with those anchors fixed at their old (easier) values, the "
       "model attributes the lower scores to lower ability.", key="drift")

c1, c2, c3, c4 = st.columns(4)
model = c1.selectbox("Model", MODELS, index=2, format_func=lambda m: f"{m} · {MODEL_NAMES[m]}")
scenario = c2.selectbox("Scenario", list(DRIFT_SCENARIOS))
prior = c3.selectbox("Ability prior", ["estimated_normal", "estimated_empirical"], format_func=PRIOR_NAMES.get)
n = c4.select_slider("N per year", [500, 1000, 2000], 1000)
c1, c2, c3 = st.columns(3)
n_anchor = c1.number_input("Anchors", 3, 12, 6)
n_items = c2.number_input("Unique items per year", 4, 20, 8)
seed = c3.number_input("Seed", 0, 10 ** 6, 77)
st.caption(f"Scenario settings: {DRIFT_SCENARIOS[scenario]}")
if st.button("Simulate and analyse", type="primary"):
    spec = drift_spec(model, scenario, int(n), int(n_anchor), int(n_items), 4, int(seed))
    d = simulate(spec, f"Drift · {model} · {scenario}")
    with st.spinner("Calibrating three ways and estimating displacement for each anchor…"):
        try:
            summary, disp, fits = drift_analysis(d, model, prior)
            st.session_state.drift = (summary, disp, scenario, model, prior, d)
        except ValueError as e:
            st.error(str(e))
if "drift" in st.session_state:
    summary, disp, sc, m, p, d = st.session_state.drift
    st.markdown(f"**{sc}** · {m} · {PRIOR_NAMES[p]}")
    table(summary, digits=4)
    fig = go.Figure()
    fig.add_bar(x=summary["calibration"], y=summary["year2_mean"], name="estimated year-2 mean", marker_color=BLUE)
    fig.add_scatter(x=summary["calibration"], y=summary["true_year2_mean"], mode="markers", name="generating",
                    marker=dict(color=ORANGE, size=14, symbol="diamond"))
    show(fig_layout(fig, "Year-2 mean by calibration", "", "mean"), "drift_mean")
    st.subheader("Anchor displacement (leave-one-out)")
    table(disp, digits=4)
    fig = go.Figure()
    fig.add_scatter(x=disp["generating_drift"], y=disp["estimated_drift"], mode="markers+text", text=disp["anchor"],
                    textposition="top center", marker=dict(color=np.where(disp["drifted"], ORANGE, BLUE), size=11))
    lo = float(min(disp["generating_drift"].min(), disp["estimated_drift"].min(), -0.1)) - 0.1
    hi = float(max(disp["generating_drift"].max(), disp["estimated_drift"].max(), 0.1)) + 0.1
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#8a8f98"))
    show(fig_layout(fig, "Estimated vs generating anchor drift (location)", "generating drift", "estimated drift",
                    legend=False), "disp")
    st.caption("Displacement: each anchor is freed in a year-2 FIPC run while the other anchors stay fixed; its "
               "estimated location minus its bank location estimates its drift. Drifted anchors are orange.")
    if st.button("Make this dataset active (for pages 7–10)"):
        add_dataset(d)
        st.success("Activated. Calibration settings switched to MG-FIPC.")
interpret("**Free concurrent** calibration treats the anchors as ordinary common items: drift is spread across "
          "the year-2 mean and the items. **FIPC/MG-FIPC** hold anchors at bank values, so undetected drift moves "
          "the year-2 distribution (an easier anchor makes the group look more able). With stable anchors, a "
          "genuine population change is recovered by all three. Changes in item parameters and in the latent "
          "distribution cannot be separated unless enough anchors are stable — that is the identification problem "
          "scale maintenance must manage. A historical passing score or raw-score standard is not by itself evidence "
          "that the scale is invariant.")
