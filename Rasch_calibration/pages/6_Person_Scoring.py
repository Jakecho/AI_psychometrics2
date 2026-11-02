import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREY, ORANGE, PURPLE, current_reference, current_result, data, download, fig_layout,
                           fmt, interpret, labels, lines, page_setup, show, store_reference, table, teaching_tabs)
from r_bridge.bridge import run_reference
from rasch.metrics import align_location, recovery
from rasch.model import observed_values
from rasch.scoring import newton_trace, score_persons, score_table
from rasch.validation import compare, max_abs

page_setup("6 · Person Scoring", "Item calibration and person scoring are separate steps: score abilities with fixed difficulties.")
d = data()

st.subheader("A · Overview")
teaching_tabs([r"U(\theta)=\sum_{j\in O_i}\big(X_{ij}-P_j(\theta)\big)=0\quad(\text{MLE})",
               r"U_W(\theta)=U(\theta)+\frac{I'(\theta)}{2I(\theta)}=0,\quad I(\theta)=\sum_jP_j(1-P_j),\ "
               r"I'(\theta)=\sum_jP_j(1-P_j)(1-2P_j)\quad(\text{WLE})",
               r"SE(\hat\theta)=1/\sqrt{I(\hat\theta)}"],
              "With item difficulties held fixed, each person's ability solves *observed score = expected score*. "
              "The MLE is biased outward on short tests and is **infinite for zero and perfect scores**. Warm's "
              "weighted likelihood adds the term I′/(2I), which pulls estimates toward the items' centre by just "
              "enough to remove the first-order bias — and gives finite values for every raw score. WLE is a "
              "person-scoring method, not a fourth way to calibrate items.",
              "- $r_i$: raw score on observed items\n- $I$: test information; $I'$: its derivative in θ\n"
              "- Reference: `TAM::tam.wle()` (WLE; and `WLE = FALSE` for MLE, which TAM makes finite with adj = 0.3)")

options = {}
for m, label in (("JMLE", "JMLE"), ("CMLE", "CMLE"), ("MML", "MML-EM")):
    r = current_result(m)
    if r is not None:
        options[f"{label} calibration"] = (m, r.b)
if d.has_truth:
    options["True difficulties (simulation)"] = ("Truth", d.true_b)
if not options:
    st.info("Calibrate items first (JMLE, CMLE or MML-EM page) — or simulate data, which provides true difficulties.")
    st.stop()
choice = st.selectbox("Item difficulties to score with", list(options))
method_key, b = options[choice]
b = np.asarray(b, float)

st.subheader("B · Raw score → ability (complete test)")
labels("edu")
mle_t = score_table(b, "MLE")
wle_t = score_table(b, "WLE")
score_frame = pd.DataFrame(dict(raw_score=mle_t[:, 0].astype(int), MLE=mle_t[:, 1], SE_MLE=mle_t[:, 2], WLE=wle_t[:, 1],
                                SE_WLE=wle_t[:, 2], WLE_minus_MLE=wle_t[:, 1] - mle_t[:, 1]))
left, right = st.columns(2)
with left:
    fig = go.Figure()
    fin = np.isfinite(mle_t[:, 1])
    fig.add_trace(go.Scatter(x=mle_t[fin, 0], y=mle_t[fin, 1], mode="lines+markers", name="MLE", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=wle_t[:, 0], y=wle_t[:, 1], mode="lines+markers", name="WLE", line=dict(color=ORANGE)))
    show(fig_layout(fig, "Ability by raw score (MLE infinite at 0 and J — not plotted)", "Raw score", "θ estimate", 330))
with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mle_t[fin, 0], y=mle_t[fin, 2], mode="lines+markers", name="SE at MLE", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=wle_t[:, 0], y=wle_t[:, 2], mode="lines+markers", name="SE at WLE", line=dict(color=ORANGE)))
    show(fig_layout(fig, "Standard error 1/√I(θ̂)", "Raw score", "SE", 330))
table(score_frame.replace([np.inf, -np.inf], np.nan), height=240)
st.caption("MLE cells shown as None are ±∞: no finite θ makes the expected score equal 0 or J.")

st.subheader("C · Numerical walkthrough for one person")
labels("demo")
person = st.selectbox("Person", range(d.n_persons), format_func=lambda i: d.person_ids[i])
x, mask = observed_values(d.X)
obs = mask[person]
r_i = int(x[person, obs].sum())
st.markdown(f"Observed items: {int(obs.sum())} · raw score r = **{r_i}**")
left, right = st.columns(2)
for col, meth in ((left, "MLE"), (right, "WLE")):
    with col:
        st.markdown(f"**{meth} Newton iterations** (start θ = 0)")
        trace = pd.DataFrame(newton_trace(r_i, b[obs], meth))
        cols = ["step", "theta", "expected_score", "residual", "information"] + (["adjustment"] if meth == "WLE" else []) + ["update", "theta_next"]
        table(trace[cols], 5, height=230)
        if meth == "MLE" and r_i in (0, int(obs.sum())):
            st.warning("Extreme score: every step moves θ further out — the MLE does not exist (diverges to ±∞).")
        else:
            last = trace.iloc[-1]
            st.caption(f"Converged θ = {last.theta_next:.5f}; SE = {1 / np.sqrt(last.information):.4f}.")

st.subheader("D · All persons")
mle, mle_se = score_persons(d.X, b, "MLE")
wle, wle_se = score_persons(d.X, b, "WLE")
persons = pd.DataFrame(dict(person=d.person_ids, raw_score=x.sum(axis=1), observed_items=mask.sum(axis=1), MLE=mle,
                            SE_MLE=mle_se, WLE=wle, SE_WLE=wle_se))
ref_settings = dict(b=b.round(10).tolist())
if st.button("Run TAM::tam.wle reference with these difficulties"):
    with st.spinner("Running TAM in R…"):
        store_reference("WLE", run_reference("WLE", d.X, ref_settings, d.item_ids), ref_settings)
ref = current_reference("WLE", ref_settings)
if ref:
    if ref["status"] == "Executed":
        persons["WLE_TAM"], persons["SE_WLE_TAM"] = ref["wle"], ref["wle_se"]
        labels("ref")
        table([dict(check="WLE abilities", max_abs_difference=max_abs(wle, ref["wle"]), tolerance=1e-3,
                    status="Pass" if max_abs(wle, ref["wle"]) <= 1e-3 else "Fail"),
               dict(check="WLE standard errors", max_abs_difference=max_abs(wle_se, ref["wle_se"]), tolerance=1e-3,
                    status="Pass" if max_abs(wle_se, ref["wle_se"]) <= 1e-3 else "Fail")])
    else:
        st.warning(f"TAM reference {ref['status']}: {ref.get('reason', '')}")
table(persons.replace([np.inf, -np.inf], np.nan), height=260)
download(persons, f"person_scores_{d.identity}.csv")
extreme = (persons.raw_score == 0) | (persons.raw_score == persons.observed_items)
st.caption(f"{int(extreme.sum())} persons with extreme scores: MLE = ±∞ (None above), WLE finite.")

if d.has_truth:
    st.subheader("E · Ability recovery (simulation)")
    _, shift = align_location(b, d.true_b)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.true_theta, y=wle + shift, mode="markers", name="WLE", marker=dict(color=ORANGE, size=5, opacity=.6)))
        ok = np.isfinite(mle)
        fig.add_trace(go.Scatter(x=d.true_theta[ok], y=mle[ok] + shift, mode="markers", name="MLE (finite)", marker=dict(color=BLUE, size=5, opacity=.6)))
        lim = [-4, 4]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=GREY), name="identity"))
        show(fig_layout(fig, "True vs estimated ability (same location shift as the items)", "True θ", "Estimate", 360))
    with right:
        bins = np.quantile(d.true_theta, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(d.true_theta, bins[1:-1]), 0, 7)
        rows = []
        for k in range(8):
            sel = idx == k
            e_w = wle[sel] + shift - d.true_theta[sel]
            e_m = (mle[sel] + shift - d.true_theta[sel])
            e_m = e_m[np.isfinite(e_m)]
            rows.append(dict(bin=k + 1, mean_true=d.true_theta[sel].mean(), bias_WLE=e_w.mean(),
                             bias_MLE=e_m.mean() if e_m.size else np.nan))
        bf = pd.DataFrame(rows)
        show(lines(bf.mean_true, {"WLE bias": bf.bias_WLE, "MLE bias (finite only)": bf.bias_MLE},
                   "Ability bias by true-ability bin", "Mean true θ in bin", "Mean error",
                   colors={"WLE bias": ORANGE, "MLE bias (finite only)": BLUE}))
    rw = recovery(wle + shift, d.true_theta)
    rm = recovery((mle + shift)[np.isfinite(mle)], d.true_theta[np.isfinite(mle)])
    table([dict(estimator="WLE (all persons)", **rw), dict(estimator="MLE (finite only)", **rm)])
interpret("- MLE estimates fan out beyond WLE at both ends of the score range: that outward bias is what Warm's "
          "correction removes.\n- WLE gives finite abilities for zero and perfect scores; MLE cannot.\n"
          "- Ability estimates inherit the item scale: difficulties from different calibrations put θ on "
          "different origins, so the same location shift used for the items is applied before comparing with truth.")
