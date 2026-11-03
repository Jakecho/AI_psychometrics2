from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from components.ui import config, guided, interpret, page_setup, table, bars
from core.diagnostics import monte_carlo, summarize_errors
from core.em import DESIGN_NAMES
from core.models import MODEL_NAMES, MODELS
from core.priors import PRIOR_NAMES, PRIOR_TYPES
from core.simulation import SimSpec

page_setup("9 · Parameter Recovery", "Monte Carlo experiments: generate, calibrate, compare with the truth — bias, RMSE, "
                                     "MAD, empirical SD, SE coverage, convergence rate and runtime.")
guided(["Pick the generating model, design and ability shape.",
        "Pick one or more fitting priors: matching and mismatching the generating shape.",
        "Run (seeds seed₀, seed₀+1, … make every replication reproducible).",
        "Compare RMSE and group-moment bias across priors; check coverage only when SEs are computed."],
       "The empirical SD of the estimates across replications measures…",
       ["sampling variability of the estimator", "the model-based standard error", "bias", "convergence speed"],
       "sampling variability of the estimator",
       "Model-based SEs estimate that same variability from one dataset; comparing the two (and coverage) checks the SEs.",
       key="rec")
ss = st.session_state
base = config()
c1, c2, c3, c4 = st.columns(4)
model = c1.selectbox("Model", MODELS, index=MODELS.index(base.model), format_func=lambda m: f"{m} · {MODEL_NAMES[m]}")
design = c2.selectbox("Design", list(DESIGN_NAMES), format_func=DESIGN_NAMES.get)
N = c3.select_slider("N per group", [250, 500, 1000, 2000, 5000], 500)
J = c4.select_slider("Items per form", [5, 10, 20, 40], 10)
c1, c2, c3, c4 = st.columns(4)
ncat = c1.selectbox("Categories", [3, 4, 5], index=1)
shape = c2.selectbox("Generating ability shape", ["normal", "skewed", "bimodal", "discrete"])
reps = c3.number_input("Replications", 2, 200, 30)
seed0 = c4.number_input("Seed₀", 0, 10 ** 6, 1000)
priors = st.multiselect("Fitting priors", PRIOR_TYPES, default=["fixed_normal", "estimated_empirical"],
                        format_func=PRIOR_NAMES.get)
c1, c2 = st.columns(2)
se = c1.checkbox("Compute standard errors (enables coverage; slower)", value=False)
c2.caption(f"Approximate work: {len(priors)} prior(s) × {reps} replications. Empirical priors take more EM cycles.")
spec = SimSpec(model=model, design=design, n_per_group=int(N), n_items=int(J), ncat=int(ncat), shape=shape,
               groups=2, n_common=max(2, int(J) // 3), n_anchor=max(3, int(J) // 3),
               group_means=[0.0, 0.5], group_sds=[1.0, 1.1] if model in ("PCM", "RSM") else [1.0, 1.1],
               group_shapes=None, seed=int(seed0))
if design == "fipc":
    spec = replace(spec, group_means=[0.4], group_sds=[1.1])
if st.button("Run Monte Carlo", type="primary", disabled=not priors):
    results = {}
    bar = st.progress(0.0)
    for k, prior in enumerate(priors):
        cfg = replace(base, model=model, design=design, prior=prior, grid="equal", se=se, pooled=False,
                      fixed_weights=None, initial_weights=None, max_cycles=min(base.max_cycles, 1000))
        if prior == "fixed_empirical":
            from core.quadrature import illustrative_distribution, make_grid
            cfg.fixed_weights = list(illustrative_distribution(make_grid("equal", cfg.n_nodes, cfg.bound).nodes, shape
                                                               if shape != "discrete" else "normal"))
        results[prior] = monte_carlo(spec, cfg, int(reps), int(seed0),
                                     callback=lambda r, R, k=k: bar.progress((k + r / R) / len(priors),
                                                                             text=f"{PRIOR_NAMES[prior]} · replication {r}/{R}"))
    ss.recovery = (results, spec, se)
if "recovery" in ss:
    results, spec_used, se_used = ss.recovery
    st.caption(f"Generating: {spec_used.model} · {DESIGN_NAMES[spec_used.design]} · {spec_used.shape} · "
               f"N = {spec_used.n_per_group}/group · J = {spec_used.n_items} · {spec_used.ncat} categories")
    rows, grows, runs = [], [], []
    for prior, r in results.items():
        if len(r["items"]):
            s = summarize_errors(r["items"][~r["items"]["fixed"]], ["kind"])
            s.insert(0, "prior", PRIOR_NAMES[prior])
            rows.append(s)
        if len(r["groups"]):
            g = r["groups"]
            gs = g.groupby("group").agg(mean_bias=("mean_error", "mean"), mean_RMSE=("mean_error", lambda e: np.sqrt(np.mean(e ** 2))),
                                        sd_bias=("sd_error", "mean"), sd_RMSE=("sd_error", lambda e: np.sqrt(np.mean(e ** 2))),
                                        density_TV=("density_TV", "mean")).reset_index()
            gs.insert(0, "prior", PRIOR_NAMES[prior])
            grows.append(gs)
        rr = r["runs"]
        runs.append(dict(prior=PRIOR_NAMES[prior], replications=len(rr), fitted=int(rr["ok"].sum()),
                         convergence_rate=float(rr["converged"].mean()), mean_cycles=float(rr["cycles"].mean()),
                         mean_seconds=float(rr["seconds"].mean()), total_seconds=float(rr["seconds"].sum())))
    st.subheader("Item parameters (free items)")
    if rows:
        table(pd.concat(rows), digits=4)
    st.caption("bias = mean(estimate − true); RMSE = √mean(error²); MAD = mean |error|; emp_SD = SD of estimates "
               "across replications (averaged over parameters); mean_SE and coverage95 appear when SEs are computed.")
    st.subheader("Group distributions")
    if grows:
        table(pd.concat(grows), digits=4)
    st.subheader("Convergence and runtime")
    table(pd.DataFrame(runs), digits=3)
    if rows:
        comb = pd.concat(rows)
        loc = comb[comb["kind"] == "location"]
        bars(loc["prior"].tolist(), {"RMSE location": loc["RMSE"].to_numpy()}, "Location RMSE by fitting prior",
             xl="prior", yl="RMSE", key="rmse_bar")
    with st.expander("Per-parameter recovery (all replications)"):
        prior = st.selectbox("Prior", list(results), format_func=PRIOR_NAMES.get)
        it = results[prior]["items"]
        if len(it):
            per = it.groupby(["item", "parameter"]).agg(true=("true", "first"), mean_est=("estimate", "mean"),
                                                        bias=("error", "mean"), emp_SD=("estimate", "std"),
                                                        mean_SE=("se", "mean")).reset_index()
            table(per, digits=4, height=300)
        table(results[prior]["runs"], digits=3, height=200)
    interpret("When the generating distribution is skewed or bimodal, a fixed normal prior biases item locations and "
              "slopes and the group moments; an estimated empirical prior usually removes most of that bias at some "
              "cost in variability. When the truth is normal, both perform about the same and the histogram is "
              "noisier. Empirical SD is the actual sampling variability; mean SE is what a single calibration "
              "reports — coverage near 0.95 means the reported SEs are trustworthy for that condition.")
