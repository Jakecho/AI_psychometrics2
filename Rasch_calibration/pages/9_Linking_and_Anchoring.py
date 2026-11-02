import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.ui import (BLUE, GREEN, GREY, NAVY, ORANGE, PURPLE, download, fig_layout, interpret, labels,
                           page_setup, phase_cards, show, signature, table, teaching_tabs)
from rasch.jmle import JMLESettings
from rasch.linking import (DESIGN_LABELS, DESIGNS, SHORT, anchor_stability, evaluate, monte_carlo, reference_rows,
                           run_all, simulate_forms)

page_setup("9 · Linking & Anchoring",
           "Put two test forms on one item-bank scale: separate calibration with linking, concurrent calibration, "
           "fixed-anchor calibration, and fixed anchor + concurrent calibration.")
st.caption("This page builds its own two-form dataset (section B); it does not use the shared single-form dataset.")

ROLE_COLORS = dict(anchor=PURPLE, common=GREEN, A=BLUE, B=ORANGE)
ROLE_NAMES = dict(anchor="Bank anchor", common="Internal common", A="Form A only", B="Form B only")
DESIGN_COLORS = dict(separate=BLUE, concurrent=ORANGE, fixed=GREEN, fixed_concurrent=PURPLE)
METHOD_NAMES = dict(JMLE="JMLE", CMLE="CMLE", MML="MML-EM")

# ---------------------------------------------------------------------------------------------
st.subheader("A · Overview")
teaching_tabs(
    [r"\text{Separate: }\ c_F=\overline{b}^{\,bank}_K-\overline{\hat b}^{(F)}_K,\qquad b^*_j=\hat b^{(F)}_j+c_F,\quad"
     r"\theta^*_i=\hat\theta^{(F)}_i+c_F",
     r"\text{Concurrent: maximise }L\big(\mathbf b\mid \mathbf X_{A}\cup\mathbf X_{B}\big)"
     r"\text{ over the stacked matrix (unadministered = missing), then one shift } c",
     r"\text{Fixed anchor: }\ b_k=b_k^{bank}\ (k\in K),\ \text{maximise over the other items (per form, or stacked)}",
     r"\text{MML with groups: }\ \theta_i\mid g\sim N(\mu_g,\sigma_g^2);\ \mu_A=0\text{ unless anchors are fixed}"],
    "Every Rasch calibration has to fix the origin of its scale (page 1), and each method does it its own way: "
    "JMLE sets the mean ability to 0, CMLE sets the mean difficulty to 0, and MML sets the population mean to 0. "
    "Calibrate Form B on its own and its "
    "scale starts at **group B's** mean, so if group B is abler, its items look easier than they are. **Linking** "
    "moves every form onto one common scale, here an existing item bank, using items whose bank difficulty is "
    "known (**anchors**).\n\n"
    "- **Separate + linking:** calibrate each form alone, then shift it so that its anchors line up with the bank "
    "on average (mean/mean; under the Rasch model the slope is 1, so only a shift is needed).\n"
    "- **Concurrent:** calibrate both forms in one run. Items a group never saw are missing by design; the "
    "shared items tie the forms together inside the likelihood. One shift then moves the whole calibration onto the bank scale.\n"
    "- **Fixed anchor:** calibrate each form with the anchors held at their bank values, so the result is on "
    "the bank scale directly, with no linking step.\n"
    "- **Fixed anchor + concurrent:** one stacked run with the anchors held at their bank values.",
    "- $K$: bank anchors (on both forms, with bank values); internal common items are on both forms but have no "
    "bank value\n- $c_F$: linking constant for form $F$\n- $\\overline{b}^{\\,bank}_K$: mean bank difficulty of "
    "the anchors used\n- $\\mu_g,\\sigma_g$: ability mean and SD of group $g$ (MML only)",
    "Python: `rasch/linking.py` (designs) uses the fixed-item options of `jmle()`, `cmle()` and `mml_em_groups()`. "
    "References: `TAM::tam.jml(xsi.fixed=)` and `TAM::tam.mml(group=, xsi.fixed=)`; `eRm::RM()` for the "
    "free CMLE runs (eRm cannot fix item parameters).")
table([dict(design=DESIGN_LABELS["separate"], runs="2 (one per form)", links_forms="Bank anchors, after calibration",
            scale_set_by="Mean/mean shift on anchors", watch_for="Ignores internal common items; drift moves the shift"),
       dict(design=DESIGN_LABELS["concurrent"], runs="1 (stacked)", links_forms="Every shared item, inside the likelihood",
            scale_set_by="One shift on anchors", watch_for="MML needs one distribution per group"),
       dict(design=DESIGN_LABELS["fixed"], runs="2 (one per form)", links_forms="Anchors held at bank values",
            scale_set_by="Fixed anchors", watch_for="A drifted anchor is forced to its old value"),
       dict(design=DESIGN_LABELS["fixed_concurrent"], runs="1 (stacked)", links_forms="Fixed anchors + common items",
            scale_set_by="Fixed anchors", watch_for="Same drift risk; MML group means must be free")])

# ---------------------------------------------------------------------------------------------
st.subheader("B · Two-form data with known truth")
labels("demo")
c = st.columns(4)
n_a = c[0].number_input("Group A persons (Form A)", 50, 5000, 500, 50)
n_b = c[0].number_input("Group B persons (Form B)", 50, 5000, 500, 50)
n_anchor = c[1].slider("Bank anchors (on both forms)", 1, 15, 6)
n_common = c[1].slider("Internal common items (both forms, no bank value)", 0, 10, 4)
n_ua = c[2].slider("Items only on Form A", 2, 30, 10)
n_ub = c[2].slider("Items only on Form B", 2, 30, 10)
mean_b = c[3].slider("Group B ability mean (group A = 0)", -1.5, 1.5, 0.5, 0.1)
sd_b = c[3].slider("Group B ability SD (group A = 1)", 0.5, 2.0, 1.0, 0.1)
c = st.columns(4)
drift_items = c[0].slider("Anchors that drift in Form B", 0, n_anchor, 0)
drift = c[1].slider("Drift size (logits, harder in Form B)", -1.5, 1.5, 0.5, 0.1, disabled=drift_items == 0)
bank_error = c[2].slider("Bank value error SD", 0.0, 0.3, 0.0, 0.01,
                         help="Bank values = true anchor difficulty + noise, standing for the error of the earlier calibration.")
seed = c[3].number_input("Seed", 1, 10**6, 2026)
try:
    ld = simulate_forms(int(n_a), int(n_b), n_anchor, n_common, n_ua, n_ub, mean_b, sd_b, 1.0, drift_items, drift,
                        bank_error, int(seed))
except ValueError as e:
    st.error(str(e))
    st.stop()
J = len(ld.item_ids)


def design_map():
    blocks = [("anchor", n_anchor), ("common", n_common), ("A", n_ua), ("B", n_ub)]
    fig = go.Figure()
    x0 = 0
    for role, k in blocks:
        if k == 0:
            continue
        for gi, (gname, rows) in enumerate([("Group A", n_a), ("Group B", n_b)]):
            on_form = role in ("anchor", "common") or (role == "A" and gi == 0) or (role == "B" and gi == 1)
            fig.add_shape(type="rect", x0=x0, x1=x0 + k, y0=1 - gi - 0.45, y1=1 - gi + 0.45, line=dict(color="white", width=2),
                          fillcolor=ROLE_COLORS[role] if on_form else "#eef1f6")
            text = f"{ROLE_NAMES[role]}<br>{k} items" if on_form else "not administered"
            fig.add_annotation(x=x0 + k / 2, y=1 - gi, text=text, showarrow=False,
                               font=dict(color="white" if on_form else GREY, size=11))
        x0 += k
    fig.update_xaxes(range=[0, x0], title="Items", showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.55, 1.55], tickvals=[1, 0], ticktext=[f"Group A · {n_a}", f"Group B · {n_b}"],
                     showgrid=False, zeroline=False)
    return fig_layout(fig, "Design: who answered which items", height=230, legend=False)


show(design_map())
c = st.columns(5)
c[0].metric("Items in the bank design", J)
c[1].metric("Persons", f"{len(ld.X):,}")
c[2].metric("True mean difference (B − A)", f"{ld.true_mean_difference:+.3f}")
c[3].metric("Drifting anchors", f"{drift_items}" + (f" × {drift:+.1f}" if drift_items else ""))
c[4].metric("Missing by design", f"{np.isnan(ld.X).mean():.1%}")
with st.expander("Item list with true and bank difficulties"):
    table(pd.DataFrame(dict(item=ld.item_ids, role=[ROLE_NAMES[r] for r in ld.roles], true_b=ld.true_b,
                            true_b_in_form_B=np.where(ld.roles == "A", np.nan, ld.true_b_form_b), bank=ld.bank)), height=280)
    st.caption("The bank column is blank for items without a bank value. Drift makes an anchor's Form B difficulty "
               "differ from its bank value.")

# ---------------------------------------------------------------------------------------------
st.subheader("C · Calibrate under all four designs")
labels("edu", "ref")
c = st.columns([1, 2, 1])
method = c[0].radio("Estimator", ["JMLE", "CMLE", "MML"], index=2, format_func=METHOD_NAMES.get, horizontal=True)
anchor_ids = [ld.item_ids[j] for j in ld.anchors]
if "link_anchors" not in st.session_state or not set(st.session_state.link_anchors) <= set(anchor_ids):
    st.session_state.link_anchors = anchor_ids
used_ids = c[1].multiselect("Anchors used for linking", anchor_ids, key="link_anchors",
                            help="An anchor left out no longer links the forms: the separate and fixed designs estimate it "
                                 "on each form, and the concurrent designs split it into a Form A item and a Form B item.")
bias = c[2].checkbox("JMLE bias correction (J − 1)/J", True, disabled=method != "JMLE")
single_pop = c[2].checkbox("Concurrent MML: one population (ignore groups)", False, disabled=method != "MML")
anchors = np.array([ld.item_ids.index(i) for i in used_ids], int)
jset = JMLESettings(bias=bias)
sig = signature(dict(data=ld.identity, method=method, anchors=used_ids, bias=bias, single=single_pop))
b1, b2 = st.columns(2)
if b1.button("Run the four designs", type="primary", disabled=len(anchors) == 0):
    try:
        with st.spinner(f"Calibrating with {METHOD_NAMES[method]} under four designs…"):
            res_all = run_all(method, ld, anchors, single_population=single_pop, jmle_settings=jset)
        st.session_state.linking = dict(sig=sig, results=res_all, refs=None)
    except ValueError as e:
        st.error(str(e))
if len(anchors) == 0:
    st.warning("Select at least one anchor item.")
entry = st.session_state.get("linking")
if entry and entry["sig"] != sig:
    st.warning("Data or settings changed since the last run. Press **Run the four designs** to update.")
    entry = None
if entry is None:
    st.info("Press **Run the four designs** to calibrate both forms under every design.")
    st.stop()
res = entry["results"]
if b2.button("Run TAM / eRm references for every calibration run"):
    with st.spinner("Running the R references (one per calibration run)…"):
        rows = []
        for r in res.values():
            rows += reference_rows(method, ld, r, jmle_settings=jset)
        entry["refs"] = rows
target = np.flatnonzero(ld.roles != "anchor")
evals = {d: evaluate(ld, res[d], anchors) for d in DESIGNS}

# ---------------------------------------------------------------------------------------------
st.subheader("D · Step through each design")
labels("edu", "demo")
tabs = st.tabs([DESIGN_LABELS[d] for d in DESIGNS])


def runs_table(r):
    rows = []
    for run in r.runs:
        fixed = 0 if run.fixed_b is None else int(np.isfinite(run.fixed_b).sum())
        g = ld.groups[run.rows]
        if len(run.group_means) == 2:          # MML with one distribution per group
            means = dict(zip((0, 1), run.group_means))
        elif run.groups is None and method == "MML" and len(set(g)) == 1:
            means = {int(g[0]): run.group_means[0]}
        else:                                  # person estimates averaged by group
            means = {k: float(run.theta[g == k].mean()) for k in sorted(set(g.tolist()))}
        rows.append(dict(run=run.name, persons=len(run.rows), items=run.X.shape[1], fixed_items=fixed,
                         iterations=run.iterations, converged="Yes" if run.converged else "No",
                         native_mean_A=means.get(0, np.nan), native_mean_B=means.get(1, np.nan),
                         linking_shift=r.shifts[run.name]))
    table(rows)


def item_table(r):
    frame = pd.DataFrame(dict(item=ld.item_ids, role=[ROLE_NAMES[x] for x in ld.roles], true_b=ld.true_b,
                              estimate=r.b, error=r.b - ld.true_b,
                              status=["Fixed at bank value" if not e else "Estimated" for e in r.estimated]))
    if r.design in ("separate", "fixed"):
        frame.insert(4, "form_A", r.by_form["A"])
        frame.insert(5, "form_B", r.by_form["B"])
    table(frame, height=260)
    download(frame, f"linking_{r.design}_{method}_{ld.identity}.csv")


def anchor_plot(r, before):
    """Anchors' estimates against bank values: native (before linking) and linked, per form."""
    fig = go.Figure()
    lim = [ld.bank[anchors].min() - 0.5, ld.bank[anchors].max() + 0.5]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dash"), name="Bank = estimate"))
    for form, color in (("A", BLUE), ("B", ORANGE)):
        if before:
            fig.add_trace(go.Scatter(x=ld.bank[anchors], y=r.unlinked[form][anchors], mode="markers", name=f"Form {form} before linking",
                                     marker=dict(color=color, size=9, symbol="circle-open", line=dict(width=2))))
        fig.add_trace(go.Scatter(x=ld.bank[anchors], y=r.by_form[form][anchors], mode="markers", name=f"Form {form} linked",
                                 marker=dict(color=color, size=9, line=dict(color="white", width=1)),
                                 text=[ld.item_ids[j] for j in anchors], hovertemplate="%{text}: bank %{x:.3f}, estimate %{y:.3f}"))
    return fig_layout(fig, "Anchors: estimate vs bank value", "Bank difficulty", "Estimated difficulty", 380).update_layout(
        legend=dict(y=-0.3))


def truth_plot(r, title):
    fig = go.Figure()
    lim = [ld.true_b.min() - 0.3, ld.true_b.max() + 0.3]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dash"), name="Estimate = truth"))
    for role in ("common", "A", "B"):
        idx = np.flatnonzero(ld.roles == role)
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scatter(x=ld.true_b[idx], y=r.b[idx], mode="markers", name=ROLE_NAMES[role],
                                 marker=dict(color=ROLE_COLORS[role], size=8, line=dict(color="white", width=1)),
                                 text=[ld.item_ids[j] for j in idx], hovertemplate="%{text}: true %{x:.3f}, estimate %{y:.3f}"))
    return fig_layout(fig, title, "True difficulty", "Estimate on the bank scale", 380).update_layout(legend=dict(y=-0.3))


with tabs[0]:
    r = res["separate"]
    phase_cards(["Calibrate Form A alone", "Calibrate Form B alone", "Shift each form: c = bank mean − anchor mean",
                 "Average items estimated on both forms"], 2)
    runs_table(r)
    cA, cB = r.shifts["Form A"], r.shifts["Form B"]
    left, right = st.columns(2)
    with left:
        show(anchor_plot(r, True), key="sep_anchor")
    with right:
        show(truth_plot(r, "All items after linking"), key="sep_truth")
    interpret(f"Each form was calibrated on its own, so each has its own zero point. The linking constants are "
              f"c_A = {cA:+.3f} and c_B = {cB:+.3f}. **Before linking**, Form B's anchors (hollow orange) sit "
              f"{cB - cA:+.3f} logits away from Form A's: that gap is group B's ability advantage, which the Form B "
              f"calibration absorbed into the item difficulties. After the shift both forms' anchors line up with "
              f"the bank on average. Linked group means: A = {r.group_means[0]:+.3f}, B = {r.group_means[1]:+.3f} "
              f"(true sample means {evals['separate']['true_mean_A']:+.3f}, {evals['separate']['true_mean_B']:+.3f}). "
              "Internal common items are estimated twice and averaged; they play no part in the shift.")
    item_table(r)

with tabs[1]:
    r = res["concurrent"]
    phase_cards(["Stack A and B (missing by design)", "One calibration", "Shift: c = bank mean − anchor mean",
                 "All items on the bank scale"], 1)
    runs_table(r)
    run = r.runs[0]
    if method == "MML" and not single_pop:
        st.markdown(f"**Group distributions (native scale):** A ~ N(0, {run.group_sds[0]:.3f}²) fixed mean, "
                    f"B ~ N({run.group_means[1]:+.3f}, {run.group_sds[1]:.3f}²)")
    left, right = st.columns(2)
    with left:
        show(truth_plot(r, "Concurrent calibration vs truth"), key="conc_truth")
    with right:
        frame = pd.DataFrame(dict(item=[ld.item_ids[j] for j in anchors], bank=ld.bank[anchors], estimate=r.b[anchors],
                                  estimate_minus_bank=r.anchor_displacement))
        table(frame)
    pop_note = ("**Concurrent MML assumes one population here.** Group B's different mean and SD are forced into one "
                "N(0, σ²), which biases the estimates. Compare the results with the box unticked."
                if method == "MML" and single_pop else
                "MML gives each group its own normal distribution (group A's mean fixed at 0)." if method == "MML" else
                f"{METHOD_NAMES[method]} needs no ability distribution, so stacking the forms needs nothing extra.")
    interpret("One run sees both groups. Every item on both forms, the bank anchors and the internal common items "
              f"alike, ties Form A's scale to Form B's inside the likelihood. A single shift c = {r.shifts['Forms A + B']:+.3f} "
              f"then moves the joint scale onto the bank. {pop_note} The anchor table shows how far each anchor's "
              "jointly estimated value lands from its bank value.")
    item_table(r)

with tabs[2]:
    r = res["fixed"]
    phase_cards(["Fix anchors at bank values", "Calibrate Form A (other items free)",
                 "Calibrate Form B (other items free)", "Check anchor displacement"], 3)
    runs_table(r)
    disp = pd.DataFrame(dict(item=[ld.item_ids[j] for j in anchors], bank=ld.bank[anchors],
                             displacement_form_A=r.runs[0].displacement[np.searchsorted(ld.form_items("A"), anchors)],
                             displacement_form_B=r.runs[1].displacement[np.searchsorted(ld.form_items("B"), anchors)]))
    left, right = st.columns(2)
    with left:
        show(truth_plot(r, "Fixed-anchor calibration vs truth"), key="fix_truth")
    with right:
        table(disp)
        st.caption("Displacement = the Newton step an anchor would take if it were freed: (expected − observed score) "
                   "÷ information at the fixed value. Positive = the anchor is harder in this form's data than its bank value.")
    worst = disp.iloc[np.argmax(np.abs(disp.displacement_form_B.to_numpy()))]
    mean_note = ("With the anchors fixed, each group's mean is estimated freely; holding it at 0 would distort the "
                 "free items." if method == "MML" else
                 "TAM's bias factor (J − 1)/J is applied to the free items only; the anchors are not rescaled."
                 if method == "JMLE" and bias else "")
    interpret(f"No linking step: the fixed anchors are the scale. Form B's largest displacement is "
              f"{worst.item} ({worst.displacement_form_B:+.3f}). An anchor whose difficulty has drifted cannot move, "
              f"so the free items and group B's mean absorb the misfit. Watch displacements larger than about "
              f"0.3–0.5 logits. {mean_note}")
    item_table(r)

with tabs[3]:
    r = res["fixed_concurrent"]
    phase_cards(["Stack A and B", "Fix anchors at bank values", "One calibration (common items free)",
                 "Check anchor displacement"], 2)
    runs_table(r)
    run = r.runs[0]
    left, right = st.columns(2)
    with left:
        show(truth_plot(r, "Fixed anchor + concurrent vs truth"), key="fc_truth")
    with right:
        table(pd.DataFrame(dict(item=[ld.item_ids[j] for j in anchors], bank=ld.bank[anchors],
                                displacement=r.anchor_displacement)))
    same = n_common == 0 and len(anchors) == n_anchor
    interpret("One stacked run with the anchors held at their bank values. Each internal common item is estimated "
              "once, from both groups' responses, instead of twice. "
              + ("**With no internal common items, this design gives the same answer as fixed-anchor calibration:** "
                 "the forms share only fixed items, so the stacked likelihood splits into one independent part per "
                 "form (exactly for JMLE and CMLE; for MML too, once each group has its own free mean)."
                 if same else "The common items share information across the two groups, which is why this design "
                              "usually gives the most precise group-mean difference. Compare with the Fixed anchor tab.")
              + (f" Both group means are free: A = {run.group_means[0]:+.3f}, B = {run.group_means[1]:+.3f}." if method == "MML" else ""))
    item_table(r)

# ---------------------------------------------------------------------------------------------
st.subheader("E · Compare the designs")
labels("edu", "ref")
cmp = pd.DataFrame([evals[d] for d in DESIGNS])
table(cmp[["design", "items_compared", "bias", "rmse", "max_abs", "mean_A", "mean_B", "mean_difference",
           "true_difference", "calibration_runs", "seconds"]])
st.caption(f"Recovery uses the {len(target)} items that are not bank anchors (estimated in every design, whichever "
           "anchors are used), "
           "compared with the truth on the bank scale. No post-hoc alignment is applied: placing the items on the "
           "bank scale is the design's own job. "
           + ("Group means are the MML population means." if method == "MML" else
              f"Group means are averages of the {METHOD_NAMES[method]} person estimates "
              + ("(JMLE θ)" if method == "JMLE" else "(WLE scored with the final difficulties)")
              + ", which are noisier than population estimates."))
fig = make_subplots(rows=1, cols=4, subplot_titles=[SHORT[d] for d in DESIGNS], shared_yaxes=True)
lim = [ld.true_b[target].min() - 0.3, ld.true_b[target].max() + 0.3]
for k, d in enumerate(DESIGNS):
    for role in ("common", "A", "B"):
        idx = target[ld.roles[target] == role]
        if len(idx) == 0:
            continue
        fig.add_trace(go.Scatter(x=ld.true_b[idx], y=res[d].b[idx], mode="markers", name=ROLE_NAMES[role],
                                 legendgroup=role, showlegend=k == 0,
                                 marker=dict(color=ROLE_COLORS[role], size=7, line=dict(color="white", width=1)),
                                 text=[ld.item_ids[j] for j in idx], hovertemplate="%{text}: true %{x:.3f}, estimate %{y:.3f}"),
                      row=1, col=k + 1)
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dash"), showlegend=False,
                             hoverinfo="skip"), row=1, col=k + 1)
fig = fig_layout(fig, "Estimated vs true difficulty (non-anchor items)", height=360)
fig.update_layout(margin=dict(t=85), title=dict(y=0.97))
fig.update_annotations(yshift=6)
show(fig, key="cmp_grid")
left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_hline(y=ld.true_mean_difference, line=dict(color=GREY, dash="dash"),
                  annotation_text=f"true {ld.true_mean_difference:+.3f}", annotation_position="top left")
    fig.add_trace(go.Scatter(x=[SHORT[d] for d in DESIGNS], y=cmp.mean_difference, mode="markers",
                             marker=dict(size=14, color=[DESIGN_COLORS[d] for d in DESIGNS], line=dict(color="white", width=2)),
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    show(fig_layout(fig, "Estimated group difference (B − A)", "", "Mean difference", 300, False), key="cmp_diff")
with right:
    fig = go.Figure(go.Bar(x=[SHORT[d] for d in DESIGNS], y=cmp.rmse, marker=dict(color=[DESIGN_COLORS[d] for d in DESIGNS]),
                           text=[f"{v:.3f}" for v in cmp.rmse], textposition="outside", hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    show(fig_layout(fig, "RMSE of non-anchor items", "", "RMSE (logits)", 300, False), key="cmp_rmse")
best = cmp.iloc[int(np.argmin(cmp.rmse))]
closest = cmp.iloc[int(np.argmin(np.abs(cmp.mean_difference - cmp.true_difference)))]
if best.design == closest.design:
    first = (f"In this sample **{best.design}** both recovers the item difficulties best (RMSE {best.rmse:.3f}) and "
             f"comes closest to the true group difference ({closest.mean_difference:+.3f} vs {closest.true_difference:+.3f}).")
else:
    first = (f"In this sample **{best.design}** recovers the item difficulties best (RMSE {best.rmse:.3f}), and "
             f"**{closest.design}** comes closest to the true group difference ({closest.mean_difference:+.3f} vs "
             f"{closest.true_difference:+.3f}).")
spread = float(cmp.mean_difference.max() - cmp.mean_difference.min())
interpret(first + f" The four designs' group differences lie within {spread:.3f} of each other, so most of any error "
          "they share comes from the sample itself (for example, the anchors happening to be easier or harder for "
          "these examinees), not from the design. One sample is noisy, so section F repeats the comparison over many "
          "samples. The RMSE gaps between designs are often smaller than their sampling error; the clearer "
          "differences are in how precisely the group difference is estimated, and in how each design reacts to "
          "anchor drift.")
if entry.get("refs"):
    st.markdown("**Reference checks for every calibration run** (native scale, before any linking shift)")
    refs = pd.DataFrame(entry["refs"])
    table(refs.drop(columns=[c for c in ("reason",) if c in refs]))
    n_pass = int((refs.status == "Pass").sum())
    n_checked = int(refs.status.isin(["Pass", "Fail"]).sum())
    st.caption(f"{n_pass} of {n_checked} executed checks passed. Differences come from the two programs' stopping "
               "rules; designs with fixed anchors and free group means converge more slowly under EM."
               + (" eRm cannot fix item parameters, so fixed-anchor CMLE runs have no R reference; the test suite "
                  "checks them by shifting the anchors and confirming that every free item moves by exactly the "
                  "same amount." if method == "CMLE" else ""))
    if "reason" in refs and refs.status.isin(["Unavailable", "Failed"]).any():
        st.warning("Reference unavailable or failed: " + "; ".join(sorted(set(refs.reason.dropna().astype(str)))))

# ---------------------------------------------------------------------------------------------
st.subheader("F · Anchor stability, drift and repeated samples")
labels("edu")
thr = st.slider("Flag an anchor when |B − A| exceeds (logits)", 0.1, 1.0, 0.5, 0.05)
stab = pd.DataFrame(anchor_stability(ld, res["separate"], thr))
left, right = st.columns(2)
with left:
    fig = go.Figure()
    lo, hi = min(stab.form_A.min(), stab.form_B.min()) - 0.4, max(stab.form_A.max(), stab.form_B.max()) + 0.4
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GREY, dash="dash"), name="Form A = Form B"))
    # Robust reference line: the median shift between the forms after linking.
    off = float(np.median(stab.form_B - stab.form_A))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo + off, hi + off], mode="lines", line=dict(color=GREEN, dash="dot"),
                             name=f"Median shift ({off:+.2f})"))
    for flag, color in (("Stable", NAVY), ("Unstable", ORANGE)):
        s = stab[stab.flag == flag]
        if len(s):
            fig.add_trace(go.Scatter(x=s.form_A, y=s.form_B, mode="markers+text", text=s["item"], textposition="top center",
                                     name=flag, marker=dict(color=color, size=11, line=dict(color="white", width=2))))
    show(fig_layout(fig, "Anchors: Form A vs Form B (separate calibrations, linked)", "Form A estimate", "Form B estimate", 340),
         key="stab")
with right:
    table(stab, 3)
    st.caption("From the separate free calibrations. B − A is each anchor's Form B minus Form A difficulty, centred on "
               "the **median** over all bank anchors, so that a few drifted anchors cannot make the stable ones look "
               "unstable (centring on the mean would). z = (B − A) / √(SE²_A + SE²_B). An anchor is flagged when "
               "|B − A| exceeds the threshold **and** |z| > 1.96. The form_A / form_B columns are the linked estimates.")
flagged = stab.loc[stab.flag == "Unstable", "item"].tolist()


def drop_flagged():
    st.session_state.link_anchors = [i for i in anchor_ids if i not in flagged]


still_used = [i for i in flagged if i in used_ids]
if still_used:
    st.button(f"Drop flagged anchors ({', '.join(still_used)}) from the anchor set", on_click=drop_flagged,
              disabled=len(flagged) == len(anchor_ids))
    st.caption("Then press **Run the four designs** again.")
elif flagged:
    st.caption(f"Flagged anchors ({', '.join(flagged)}) are already excluded from the anchor set.")
truly = [ld.item_ids[j] for j in range(drift_items)]
interpret("An anchor drifts when it behaves differently in the new form than when it was banked (it was exposed, "
          "taught to, or reworded, for example). " +
          (f"In this simulation **{', '.join(truly)}** drift by {drift:+.1f} logits in Form B. The separate and "
           "fixed-anchor designs pass that drift on to Form B's items and group mean. The concurrent designs dilute "
           "it somewhat, because each drifted anchor gets one compromise estimate from both groups, but they do not "
           "remove it. Dropping a drifted anchor from the anchor set removes the bias in every design: the separate "
           "and fixed designs then estimate it separately on each form, and the concurrent designs split it into a "
           "Form A item and a Form B item, so it no longer links the forms. The cost is a little precision (fewer "
           "links); compare the repeated-samples table before and after." if drift_items else
           "No drift is simulated, so any flags are sampling noise. Add drift in section B to see how each design reacts."))

st.markdown("**Repeated samples** · items, bank values and design held fixed; persons and responses redrawn")
c = st.columns([1, 1, 3])
reps = c[0].number_input("Replications", 5, 200, 30 if method != "CMLE" else 15, 5)
limit = c[1].number_input("Time limit (s)", 10, 600, 120, 10)
mc_sig = signature(dict(sig=sig, reps=int(reps)))
if c[2].button("Run repeated samples"):
    with st.spinner("Replicating every design…"):
        t0 = time.perf_counter()
        rows, done = monte_carlo(method, ld, int(reps), anchors, single_pop, float(limit), jmle_settings=jset)
        st.session_state.linking_mc = dict(sig=mc_sig, rows=rows, done=done, seconds=time.perf_counter() - t0)
mc = st.session_state.get("linking_mc")
if mc and mc["sig"] == mc_sig:
    f = pd.DataFrame(mc["rows"])
    summ = f.groupby("design", sort=False).agg(mean_rmse=("rmse", "mean"), item_bias=("bias", "mean"),
                                                difference_bias=("diff_error", "mean"), difference_sd=("diff_error", "std"),
                                                group_B_mean_bias=("mean_B_error", "mean")).reset_index()
    table(summ)
    st.caption(f"{mc['done']} of {int(reps)} replications in {mc['seconds']:.1f} s"
               + (" (time limit reached)" if mc["done"] < int(reps) else "")
               + ". difference_bias / difference_sd: error in the estimated B − A mean difference.")
    prec = summ.iloc[int(np.argmin(summ.difference_sd))]
    interpret(f"Over repeated samples, **{prec.design}** estimates the group difference most precisely (SD "
              f"{prec.difference_sd:.3f}). Designs that calibrate both forms in one run use every shared item to "
              "connect the groups, while the separate and fixed designs connect them only through the bank anchors. "
              "Bias shows up when anchors drift or when concurrent MML ignores the groups.")
elif mc:
    st.caption("Settings changed since the last repeated-samples run.")

interpret("- **Separate + linking**: simple and transparent, and each form can be checked on its own first. It links only "
          "through the anchors, so it is the least precise.\n"
          "- **Concurrent**: uses every shared item and estimates the group difference most precisely. With MML it needs "
          "one distribution per group, and all the data must be calibrated together.\n"
          "- **Fixed anchor**: puts new forms straight onto an existing bank, with no linking step. Its quality depends "
          "on the anchors being stable, so check displacement.\n"
          "- **Fixed anchor + concurrent**: one run on the bank scale that also shares information through the "
          "internal common items. It is the usual choice for adding several new forms to a bank at once.",
          "Choosing a design")
