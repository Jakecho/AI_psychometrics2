import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (GREY, METHOD_COLORS, data, download, fig_layout, interpret, labels, page_setup, show,
                           signature, table)
from rasch.cmle import cmle
from rasch.jmle import JMLESettings, jmle
from rasch.metrics import align_location, recovery
from rasch.mml_em import MMLSettings, mml_em
from rasch.scoring import score_persons
from rasch.simulation import simulate

page_setup("7 · Estimator Comparison", "JMLE, CMLE and MML-EM on the same responses: recovery, uncertainty, runtime and convergence.")
d = data()
NAMES = dict(JMLE="JMLE (TAM defaults)", JMLE_raw="JMLE (no bias correction)", CMLE="CMLE", MML="MML-EM")


def run_all(X, include_raw_jmle=False):
    out = {}
    fits = [("JMLE", lambda: jmle(X, JMLESettings()))]
    if include_raw_jmle:
        fits.append(("JMLE_raw", lambda: jmle(X, JMLESettings(bias=False))))
    fits += [("CMLE", lambda: cmle(X)), ("MML", lambda: mml_em(X, MMLSettings()))]
    for key, fn in fits:
        t = time.perf_counter()
        try:
            r = fn()
            out[key] = dict(result=r, seconds=time.perf_counter() - t, error=None)
        except ValueError as e:
            out[key] = dict(result=None, seconds=time.perf_counter() - t, error=str(e))
    return out


st.subheader("1 · Single run on the current dataset")
labels("edu")
st.caption("Each method uses its default settings (JMLE: adj 0.3, bias correction, mean ability 0; CMLE: sum b = 0; "
           "MML-EM: 21 nodes, variance estimated, mean ability 0).")
if st.button("Run all three estimators", type="primary"):
    with st.spinner("Running JMLE, CMLE and MML-EM…"):
        st.session_state.comparison = dict(data_id=d.identity, runs=run_all(d.X))
comp = st.session_state.get("comparison")
if comp and comp["data_id"] != d.identity:
    st.warning("The dataset changed since this comparison was run. Run it again.")
    comp = None
if comp:
    runs = comp["runs"]
    rows, est = [], {}
    for key, run in runs.items():
        r = run["result"]
        if r is None:
            rows.append(dict(method=NAMES[key], status="Failed: " + run["error"], seconds=run["seconds"]))
            continue
        est[key] = r.b
        row = dict(method=NAMES[key], status="Converged" if r.converged else "Not converged", iterations=r.iterations,
                   seconds=run["seconds"], mean_SE=float(np.nanmean(r.se_b)), native_mean_b=float(np.mean(r.b)))
        if d.has_truth:
            aligned, shift = align_location(r.b, d.true_b)
            row.update({f"{k}": v for k, v in recovery(aligned, d.true_b).items() if k != "n"}, alignment_shift=shift)
        rows.append(row)
    st.markdown("**Method comparison table**" + (" (recovery after aligning each method's mean to the true mean)" if d.has_truth else ""))
    table(rows)
    download(pd.DataFrame(rows), f"comparison_{d.identity}.csv")
    item_frame = pd.DataFrame(dict(item=d.item_ids, **{f"b_{k}": v for k, v in est.items()}))
    if d.has_truth:
        item_frame["true_b"] = d.true_b
    with st.expander("Native item estimates (each method's own identification)"):
        table(item_frame, height=260)
        download(item_frame, f"comparison_items_{d.identity}.csv")
    if est:
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            ref_vals = d.true_b if d.has_truth else est.get("CMLE", next(iter(est.values())))
            for k, b in est.items():
                aligned, _ = align_location(b, ref_vals)
                fig.add_trace(go.Scatter(x=ref_vals, y=aligned, mode="markers", name=NAMES[k], marker=dict(color=METHOD_COLORS[k], size=8)))
            lim = [float(np.min(ref_vals)) - .3, float(np.max(ref_vals)) + .3]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=GREY), name="identity"))
            show(fig_layout(fig, ("True" if d.has_truth else "CMLE") + " vs estimated difficulty (aligned)",
                            "True b" if d.has_truth else "CMLE b", "Estimate", 360))
        with right:
            if "CMLE" in est:
                fig = go.Figure()
                base, _ = align_location(est["CMLE"], est["CMLE"])
                for k, b in est.items():
                    if k == "CMLE":
                        continue
                    aligned, _ = align_location(b, est["CMLE"])
                    fig.add_trace(go.Scatter(x=est["CMLE"], y=aligned - est["CMLE"], mode="markers", name=f"{NAMES[k]} − CMLE",
                                             marker=dict(color=METHOD_COLORS[k], size=8)))
                fig.add_hline(y=0, line_dash="dot")
                show(fig_layout(fig, "Difference from CMLE (aligned)", "CMLE b", "Difference", 360))
        if d.has_truth:
            fig = go.Figure()
            for k, b in est.items():
                aligned, _ = align_location(b, d.true_b)
                fig.add_trace(go.Box(y=aligned - d.true_b, name=NAMES[k], marker_color=METHOD_COLORS[k], boxpoints="all"))
            show(fig_layout(fig, "Estimation error distribution (aligned estimate − truth)", "", "Error", 340, False))
            prow = []
            for k, b in est.items():
                wle, _ = score_persons(d.X, b, "WLE")
                _, shift = align_location(b, d.true_b)
                prow.append(dict(method=NAMES[k], **recovery(wle + shift, d.true_theta)))
            st.markdown("**Person-ability recovery** (WLE with each method's difficulties, same alignment shift)")
            table(prow)
        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Bar(x=[NAMES[k] for k in runs], y=[runs[k]["seconds"] for k in runs],
                                   marker_color=[METHOD_COLORS[k] for k in runs]))
            show(fig_layout(fig, "Runtime (seconds, Python)", "", "seconds", 300, False))
        with right:
            fig = go.Figure()
            for k, run in runs.items():
                r = run["result"]
                if r is None:
                    continue
                if k == "CMLE":
                    y = [h["grad_norm"] for h in r.history]
                    label = "CMLE max |score|"
                elif k == "JMLE":
                    y = [max(h["max_theta_change"], h["max_b_change"]) for h in r.history]
                    label = "JMLE max |change|"
                else:
                    y = [max(h["max_b_change"], h["sd_change"]) for h in r.history]
                    label = "MML-EM max |change|"
                fig.add_trace(go.Scatter(y=y, mode="lines+markers", name=label, line=dict(color=METHOD_COLORS[k])))
            show(fig_layout(fig.update_yaxes(type="log"), "Convergence trajectories", "Iteration", "criterion", 300))
    interpret("- **Why estimates differ:** JMLE estimates every ability as well, which stretches the difficulty "
              "scale by roughly J/(J−1) on short tests; TAM's (J−1)/J correction (on by default) largely removes "
              "this. CMLE removes ability exactly and is consistent; MML-EM is consistent if the normal ability "
              "distribution is right.\n- **Identification:** each method fixes the origin "
              "differently (native means shown), so estimates are aligned by a location shift before comparing; the "
              "shift does not change any fitted probability.\n- Standard errors are model-based for one run; their "
              "accuracy is checked by repeated simulation below.")

st.subheader("2 · Repeated simulation (Monte Carlo)")
st.caption("Generates fresh datasets with known truth and fits all three methods to each. Use moderate sizes: each "
           "replication runs three estimators.")
c1, c2, c3, c4 = st.columns(4)
reps = c1.number_input("Replications", 2, 200, 30)
n_mc = c2.number_input("Persons N", 50, 5000, 500, 50)
j_mc = c3.number_input("Items J", 5, 60, 20)
limit = c4.number_input("Time limit (seconds)", 10, 1800, 180, help="The run stops after this many seconds; finished replications are kept.")
c1, c2, c3, c4 = st.columns(4)
tm = c1.number_input("Ability mean", -2.0, 2.0, 0.0, 0.1)
ts = c2.number_input("Ability SD", 0.3, 2.0, 1.0, 0.1)
bsd = c3.number_input("Difficulty SD", 0.3, 2.0, 1.0, 0.1)
seed0 = c4.number_input("First seed", 0, 10**6, 1000)
raw_jmle = st.checkbox("Also fit JMLE without the (J−1)/J bias correction", True,
                       help="Shows the incidental-parameter bias that TAM's correction removes.")
st.caption("True item difficulties are drawn once (from the first seed) and held fixed; abilities and responses are "
           "redrawn in every replication, so the spread of an item's estimates measures estimation error.")
mc_settings = dict(reps=int(reps), n=int(n_mc), J=int(j_mc), tm=tm, ts=ts, bsd=bsd, seed0=int(seed0), raw=raw_jmle)
if st.button("Run Monte Carlo", type="primary"):
    bar = st.progress(0.0, text="Starting…")
    started = time.perf_counter()
    records = []
    done = 0
    b_fixed = np.random.default_rng(int(seed0)).normal(0.0, bsd, int(j_mc))
    for rep in range(int(reps)):
        if time.perf_counter() - started > limit:
            st.warning(f"Time limit reached after {done} replications; results below use those.")
            break
        sim = simulate(int(n_mc), int(j_mc), tm, ts, 0.0, bsd, 0.0, int(seed0) + 1 + rep, true_b=b_fixed)
        for key, run in run_all(sim.X, raw_jmle).items():
            r = run["result"]
            if r is None:
                records.append(dict(rep=rep, method=key, ok=False, seconds=run["seconds"]))
                continue
            aligned, _ = align_location(r.b, sim.true_b)
            for j in range(int(j_mc)):
                records.append(dict(rep=rep, method=key, ok=bool(r.converged), seconds=run["seconds"], item=j,
                                    est=aligned[j], true=sim.true_b[j], se=r.se_b[j]))
        done += 1
        bar.progress(done / reps, text=f"Replication {done} / {reps}")
    st.session_state.monte_carlo = dict(settings=mc_settings, records=pd.DataFrame(records), done=done)
mc = st.session_state.get("monte_carlo")
if mc:
    rec = mc["records"]
    if mc["settings"] != mc_settings:
        st.caption("Showing the last completed run (settings above have changed since).")
    rows = []
    for key in ("JMLE", "JMLE_raw", "CMLE", "MML"):
        r = rec[rec.method == key]
        if r.empty:
            continue
        fits = r.dropna(subset=["est"]) if "est" in r else r.iloc[0:0]
        if fits.empty:
            rows.append(dict(method=NAMES[key], replications=0))
            continue
        fits = fits.assign(err=fits.est - fits.true)
        per_rep = fits.groupby("rep")
        per_item = fits.groupby("item")
        slope = float(per_rep.apply(lambda z: np.polyfit(z.true, z.est, 1)[0], include_groups=False).mean())
        rows.append(dict(method=NAMES[key], replications=int(per_rep.ngroups), scale_slope=slope,
                         average_bias=float(per_rep.err.mean().mean()),
                         RMSE=float(np.sqrt((fits.err ** 2).mean())), MAD=float(fits.err.abs().mean()),
                         empirical_SD=float(per_item.est.std().mean()), mean_reported_SE=float(fits.se.mean()),
                         convergence_rate=float(r.drop_duplicates("rep").ok.mean()),
                         mean_seconds=float(r.drop_duplicates("rep").seconds.mean())))
    st.markdown(f"**Across {mc['done']} replications** (N = {mc['settings']['n']}, J = {mc['settings']['J']}; "
                "each replication's estimates aligned to its true mean)")
    table(rows)
    download(pd.DataFrame(rows), "monte_carlo_summary.csv")
    interpret("- **Empirical SD** is the standard deviation of each item's estimate across replications (averaged "
              "over items); **mean reported SE** is what each single run claimed. A single run's SE is never a "
              "substitute for the empirical SD.\n- Estimates are aligned to the true mean in every replication, which "
              "removes uncertainty about the scale's origin. CMLE's sum-zero SEs are on that footing; JMLE and MML-EM "
              "anchor the scale on the persons, so their reported SEs also carry origin uncertainty and run a little "
              "larger than the aligned empirical SD.\n- **scale_slope** is the average slope of estimates on true "
              "difficulties: about J/(J−1) for uncorrected JMLE (the incidental-parameter bias), close to 1 for the "
              "others. Average bias is zero by construction after alignment, so look at the slope and RMSE.")
