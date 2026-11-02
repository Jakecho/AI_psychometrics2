import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREY, ORANGE, PURPLE, current_reference, current_result, data, download, fig_layout,
                           fmt, interpret, labels, lines, page_setup, phase_cards, show, stale_result, stepper,
                           store_reference, store_result, table, teaching_tabs)
from r_bridge.bridge import run_reference
from rasch.jmle import JMLESettings, jmle, walkthrough
from rasch.metrics import align_location, recovery
from rasch.validation import compare

page_setup("3 · JMLE Lab", "Joint maximum likelihood: person abilities and item difficulties estimated together.")
d = data()

st.subheader("A · Overview")
teaching_tabs([r"L(\boldsymbol\theta,\mathbf b)=\prod_{i=1}^N\prod_{j\in O_i}P_{ij}^{X_{ij}}(1-P_{ij})^{1-X_{ij}}",
               r"U_{\theta_i}=\sum_{j\in O_i}(X_{ij}-P_{ij})=0,\qquad U_{b_j}=\sum_{i:\,j\in O_i}(P_{ij}-X_{ij})=0",
               r"\theta_i\leftarrow\theta_i+\frac{r_i-\sum_jP_{ij}}{\sum_jP_{ij}(1-P_{ij})},\qquad "
               r"b_j\leftarrow b_j+\frac{\sum_iP_{ij}-s_j}{\sum_iP_{ij}(1-P_{ij})}"],
              "JMLE treats **every person's ability as a parameter** and maximises one joint likelihood. It "
              "alternates: (1) update each θ with difficulties fixed, (2) re-anchor the scale, (3) update each b "
              "with abilities fixed, until nothing changes. The person equation says *observed score = expected "
              "score*; the item equation says the same for items. Each step is a Newton step: residual ÷ information.",
              "- $O_i$: items observed for person *i*; $r_i$: raw score; $s_j$: item score\n"
              "- $I_{\\theta_i}=\\sum_{j\\in O_i}P_{ij}(1-P_{ij})$, $I_{b_j}=\\sum_{i}P_{ij}(1-P_{ij})$",
              "Python: `rasch/jmle.py`. Reference: `TAM::tam.jml()` (same defaults: adj = 0.3, bias = TRUE, "
              "constraint = \"cases\", tolerance 1e-4).")

st.subheader("B · Estimation controls")
c1, c2, c3, c4 = st.columns(4)
adj = c1.number_input("Extreme-score adjustment", 0.0, 0.49, 0.3, 0.05,
                      help="Zero and perfect raw scores are moved inward by this amount (TAM: adj = 0.3). "
                           "With 0, persons with extreme scores make JMLE fail (infinite MLE).")
bias = c2.checkbox("Bias correction (J−1)/J", True, help="TAM: bias = TRUE. Multiplies final difficulties by (J−1)/J.")
constraint = c3.selectbox("Identification", ["cases", "items"],
                          format_func=lambda v: "Mean ability = 0 (cases)" if v == "cases" else "Mean difficulty = 0 (items)")
tol = c4.number_input("Convergence tolerance", 1e-8, 1e-2, 1e-4, format="%.0e")
c1, c2, c3, _ = st.columns(4)
max_iter = c1.number_input("Maximum iterations", 1, 5000, 1000)
inner = c2.number_input("Newton steps per update", 1, 50, 10, help="Inner Newton steps for each person/item block.")
max_step = c3.number_input("Step limit (logits)", 0.1, 5.0, 1.0, 0.1, help="Damping: each Newton step is capped at this size.")
settings = dict(adj=adj, bias=bias, constraint=constraint, tol=tol, max_iter=int(max_iter), inner_steps=int(inner),
                max_step=max_step)
ref_settings = dict(adj=adj, bias=bias, constraint=constraint, tol=tol, max_iter=int(max_iter))
b1, b2 = st.columns(2)
if b1.button("Run Python JMLE", type="primary"):
    try:
        with st.spinner("Running JMLE…"):
            store_result("JMLE", jmle(d.X, JMLESettings(**settings)), settings)
    except ValueError as e:
        st.error(str(e))
if b2.button("Run TAM::tam.jml reference"):
    with st.spinner("Running TAM in R…"):
        store_reference("JMLE", run_reference("JMLE", d.X, ref_settings, d.item_ids), ref_settings)
res = current_result("JMLE", settings)
ref = current_reference("JMLE", ref_settings)
if stale_result("JMLE", settings):
    st.warning("Settings changed since the last run. Press **Run Python JMLE** to update the results below.")
if res is None:
    st.info("Press **Run Python JMLE** to calibrate the current dataset.")
    st.stop()

st.subheader("C · Algorithm visualisation")
phase_cards(["Start values (logits of scores)", "Update persons θ", "Re-anchor scale", "Update items b",
             "Check max |change|"], 1)
H = res.history
its = np.array([h["iteration"] for h in H])
left, right = st.columns(2)
with left:
    show(lines(its, {"Joint log-likelihood": [h["loglik"] for h in H]}, "Joint log-likelihood", "Iteration", "log L",
               colors={"Joint log-likelihood": BLUE}))
with right:
    fig = lines(its, {"max |Δθ|": [h["max_theta_change"] for h in H], "max |Δb|": [h["max_b_change"] for h in H]},
                "Maximum change per iteration (log scale)", "Iteration", "Change", colors={"max |Δb|": ORANGE})
    fig.add_hline(y=tol, line_dash="dot", annotation_text="tolerance")
    show(fig.update_yaxes(type="log"))
sel_items = st.multiselect("Items to trace", d.item_ids, d.item_ids[:5])
sel_persons = st.multiselect("Persons to trace", d.person_ids, d.person_ids[:5])
left, right = st.columns(2)
with left:
    show(lines(np.r_[0, its], {i: np.r_[H[0]["b_in"][d.item_ids.index(i)], [h["b"][d.item_ids.index(i)] for h in H]]
                               for i in sel_items}, "Item-difficulty trajectories (before bias correction)",
               "Iteration (0 = start)", "b"))
with right:
    show(lines(np.r_[0, its], {p: np.r_[H[0]["theta_in"][d.person_ids.index(p)], [h["theta"][d.person_ids.index(p)] for h in H]]
                               for p in sel_persons}, "Ability trajectories", "Iteration (0 = start)", "θ"))

st.subheader("D · Numerical walkthrough")
labels("edu", "demo")
c1, c2 = st.columns(2)
person = c1.selectbox("Person", range(d.n_persons), format_func=lambda i: d.person_ids[i])
item = c2.selectbox("Item", range(d.n_items), format_func=lambda j: d.item_ids[j])
it = stepper("jmle_it", len(H), "Iteration")
w = walkthrough(d.X, res, it, person, item)
left, right = st.columns(2)
with left:
    st.markdown(f"**Person {d.person_ids[person]} · iteration {it}** (first Newton step)")
    st.latex(rf"r_i^{{*}}={w['person_score']:.3f},\quad \sum_jP_{{ij}}={w['expected_person']:.4f},\quad"
             rf"\ I_{{\theta_i}}={w['person_info']:.4f}")
    st.latex(rf"\theta^{{new}}={w['theta_in']:.4f}+\frac{{{w['person_residual']:.4f}}}{{{w['person_info']:.4f}}}"
             rf"={w['theta_first_step']:.4f}")
    st.caption(f"{w['observed_items']} observed items. After all inner steps and re-anchoring: θ = {w['theta_out']:.4f}.")
with right:
    st.markdown(f"**Item {d.item_ids[item]} · iteration {it}** (first Newton step, abilities just updated)")
    st.latex(rf"s_j^{{*}}={w['item_score']:.3f},\quad \sum_iP_{{ij}}={w['expected_item']:.4f},\quad"
             rf"\ I_{{b_j}}={w['item_info']:.4f}")
    st.latex(rf"b^{{new}}={w['b_in']:.4f}+\frac{{{w['item_residual']:.4f}}}{{{w['item_info']:.4f}}}={w['b_first_step']:.4f}")
    st.caption(f"After all inner steps: b = {w['b_out']:.4f}.")
rec = w["rec"]
st.markdown(f"**Stopping check, iteration {it}:** max |Δθ| = {rec['max_theta_change']:.2e}, max |Δb| = "
            f"{rec['max_b_change']:.2e}; tolerance {tol:.0e} → "
            + ("**stop**" if max(rec['max_theta_change'], rec['max_b_change']) < tol else "continue"))
table(pd.DataFrame([dict(iteration=h["iteration"], theta_selected=h["theta"][person], b_selected=h["b"][item],
                         max_theta_change=h["max_theta_change"], max_b_change=h["max_b_change"],
                         joint_loglik=h["loglik"], converged=max(h["max_theta_change"], h["max_b_change"]) < tol)
                    for h in H]), height=240)

st.subheader("E · Results")
labels("edu", "ref")
st.markdown(f"**Status:** {res.status} · {res.iterations} iterations · {res.n_extreme_persons} persons with extreme "
            f"scores (adjusted by {adj})")
items = pd.DataFrame(dict(item=d.item_ids, b_python=res.b, se_python=res.se_b, b_before_bias=res.b_uncorrected))
if ref and ref["status"] == "Executed":
    items["b_TAM"], items["se_TAM"] = ref["b"], ref["se_b"]
    items["difference"] = items.b_python - items.b_TAM
if d.has_truth:
    items["true_b"] = d.true_b
table(items, height=300)
download(items, f"jmle_items_{d.identity}.csv")
persons = pd.DataFrame(dict(person=d.person_ids, theta_python=res.theta, se_python=res.se_theta,
                            adjusted_raw_score=res.adjusted_person_scores))
if ref and ref["status"] == "Executed":
    persons["theta_TAM"] = ref["theta"]
with st.expander("Person estimates"):
    table(persons, height=300)
    download(persons, f"jmle_persons_{d.identity}.csv")
if ref:
    if ref["status"] == "Executed":
        st.markdown(f"**Reference comparison** · TAM {ref['version']}, {ref['iterations']} iterations")
        table(compare("JMLE", dict(b=res.b, se_b=res.se_b, theta=res.theta), ref))
    else:
        st.warning(f"TAM reference {ref['status']}: {ref.get('reason', '')}")

st.subheader("F · Diagnostics and interpretation")
if d.has_truth:
    aligned, shift = align_location(res.b, d.true_b)
    m = recovery(aligned, d.true_b)
    c = st.columns(5)
    c[0].metric("Bias", fmt(m["bias"]))
    c[1].metric("RMSE", fmt(m["rmse"]))
    c[2].metric("MAD", fmt(m["mad"]))
    c[3].metric("Correlation", fmt(m["corr"], 3))
    c[4].metric("Location shift applied", f"{shift:+.3f}")
    fig = go.Figure(go.Scatter(x=d.true_b, y=aligned, mode="markers", marker=dict(color=BLUE, size=8), text=d.item_ids))
    lim = [min(d.true_b.min(), aligned.min()) - .2, max(d.true_b.max(), aligned.max()) + .2]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=GREY), name="identity"))
    show(fig_layout(fig, "True vs estimated difficulty (location-aligned)", "True b", "Estimated b", 340, False))
interpret("- **Incidental parameters:** every person adds a parameter, so the number of parameters grows with N. "
          "With short tests JMLE difficulties are spread too wide; the (J−1)/J factor is a simple correction, not "
          "an exact one.\n- **Extreme scores:** all-correct or all-wrong persons have infinite unadjusted MLEs; the "
          f"adjustment ({adj}) makes them finite but is a convention, not an estimate.\n- **Convergence ≠ unbiased:** "
          "the algorithm can converge tightly to a biased answer. Compare with CMLE (consistent for items) on "
          "page 7.\n- Recovery statistics use a post-hoc shift so the mean estimate equals the mean true difficulty; "
          "the fitted parameters themselves are unchanged.")
