import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREEN, GREY, NAVY, ORANGE, current_reference, current_result, data, download,
                           fig_layout, fmt, interpret, labels, lines, page_setup, phase_cards, show, stale_result,
                           stepper, store_reference, store_result, table, teaching_tabs)
from r_bridge.bridge import run_reference
from rasch.metrics import align_location, recovery
from rasch.mml_em import MMLSettings, estep_walkthrough, mml_em
from rasch.validation import compare

page_setup("5 · MML-EM Lab", "Marginal maximum likelihood: integrate ability out over a normal distribution, fitted by EM.")
d = data()

st.subheader("A · Overview")
teaching_tabs([r"L(\mathbf b,\sigma^2)=\prod_i\int P(\mathbf x_i\mid\theta,\mathbf b)\,g(\theta;0,\sigma^2)\,d\theta"
               r"\;\approx\;\prod_i\sum_{q=1}^{Q}w_q\,L_i(\theta_q\mid\mathbf b)",
               r"\tau_{iq}=\frac{w_qL_i(\theta_q\mid\mathbf b)}{\sum_{h}w_hL_i(\theta_h\mid\mathbf b)}\quad(\text{E-step})",
               r"\sum_q\big[R_{jq}-N_{jq}P_j(\theta_q;b_j)\big]=0,\quad N_{jq}=\sum_i m_{ij}\tau_{iq},\ "
               r"R_{jq}=\sum_i m_{ij}x_{ij}\tau_{iq}\quad(\text{M-step})",
               r"\sigma^2\leftarrow\frac1N\sum_i\sum_q\tau_{iq}\theta_q^2\quad(\text{mean fixed at }0)"],
              "Abilities are not estimated individually. Each person's likelihood is **averaged over an ability "
              "distribution**, approximated on a grid of quadrature nodes. EM alternates: the **E-step** turns each "
              "person's likelihood × prior into posterior weights over the nodes (which sum to 1); these spread each "
              "person's responses fractionally across the nodes as *expected counts*. The **M-step** fits each item "
              "curve to those expected counts and updates the ability variance. The marginal log-likelihood never "
              "decreases from one EM cycle to the next.",
              "- $\\theta_q$, $w_q$: quadrature node and prior weight ($\\sum_q w_q=1$)\n- $\\tau_{iq}$: posterior weight of "
              "person *i* at node *q*\n- $N_{jq}$, $R_{jq}$: expected respondents and expected correct answers\n"
              "- identification: ability mean fixed at 0 (TAM `constraint = \"cases\"`)",
              "Python: `rasch/mml_em.py` (log-sum-exp E-step, vectorised safeguarded Newton M-step). Reference: "
              "`TAM::tam.mml()` with the same nodes. Item SE: inverse observed information (main) and the "
              "complete-data information that TAM reports as `se.xsi`.")

st.subheader("B · Estimation controls")
c1, c2, c3, c4 = st.columns(4)
n_nodes = c1.number_input("Quadrature nodes Q", 5, 101, 21, help="TAM default: 21 nodes on [−6, 6].")
extent = c2.number_input("Grid half-width", 2.0, 10.0, 6.0, 0.5)
est_var = c3.checkbox("Estimate ability variance", True, help="Mean is fixed at 0 for identification.")
fixed_sd = c4.number_input("Fixed SD (if not estimated)", 0.2, 3.0, 1.0, 0.1, disabled=est_var)
c1, c2, _, _ = st.columns(4)
tol = c1.number_input("Convergence tolerance", 1e-8, 1e-2, 1e-4, format="%.0e")
max_iter = c2.number_input("Maximum EM cycles", 1, 5000, 1000)
settings = dict(n_nodes=int(n_nodes), extent=extent, estimate_variance=est_var, fixed_sd=fixed_sd, tol=tol,
                max_iter=int(max_iter))
b1, b2 = st.columns(2)
if b1.button("Run Python MML-EM", type="primary"):
    try:
        with st.spinner("Running EM…"):
            store_result("MML", mml_em(d.X, MMLSettings(**settings)), settings)
    except ValueError as e:
        st.error(str(e))
if b2.button("Run TAM::tam.mml reference"):
    with st.spinner("Running TAM in R…"):
        store_reference("MML", run_reference("MML", d.X, settings, d.item_ids), settings)
res = current_result("MML", settings)
ref = current_reference("MML", settings)
if stale_result("MML", settings):
    st.warning("Settings changed since the last run. Press **Run Python MML-EM** to update.")
if res is None:
    st.info("Press **Run Python MML-EM** to calibrate the current dataset.")
    st.stop()

st.subheader("C · One EM cycle, step by step")
labels("edu", "demo")
H = res.history
c1, c2, c3 = st.columns(3)
person = c1.selectbox("Person", range(d.n_persons), format_func=lambda i: d.person_ids[i])
item = c2.selectbox("Item", range(d.n_items), format_func=lambda j: d.item_ids[j])
phase = c3.radio("Phase", ["E-step", "M-step", "Likelihood & check"], horizontal=True)
it = stepper("mml_it", len(H), "EM cycle")
w = estep_walkthrough(d.X, res, it, person, item)
rec = w["rec"]
phase_cards(["Initialize / current state", "E-step: posterior weights", "Expected counts", "M-step: items & variance",
             "Likelihood & convergence check"], {"E-step": 1, "M-step": 3, "Likelihood & check": 4}[phase])
st.markdown(f"**Cycle {it} input state** ψ⁽ᵗ⁾: σ = {rec['sd_in']:.4f}, b({d.item_ids[item]}) = {rec['b_in'][item]:.4f}")
t = w["table"]
nodes = t["node"]
if phase == "E-step":
    resp = "".join("·" if np.isnan(v) else str(int(v)) for v in d.X[person])
    st.caption(f"{d.person_ids[person]} responses: {resp}  (· = missing, excluded from the likelihood)")
    cols = st.columns([5, 1, 5, 1, 5])
    with cols[0]:
        show(fig_layout(go.Figure(go.Scatter(x=nodes, y=t["likelihood"] / t["likelihood"].max(), mode="lines+markers",
                                             line=dict(color=NAVY))), "Response likelihood (relative)", "θ node", "", 260, False))
    cols[1].markdown("<div style='font-size:28px;text-align:center;margin-top:110px'>×</div>", unsafe_allow_html=True)
    with cols[2]:
        show(fig_layout(go.Figure(go.Bar(x=nodes, y=t["prior_weight"], marker_color=ORANGE)), "Population prior weight w_q",
                        "θ node", "", 260, False))
    cols[3].markdown("<div style='font-size:28px;text-align:center;margin-top:110px'>→</div>", unsafe_allow_html=True)
    with cols[4]:
        show(fig_layout(go.Figure(go.Bar(x=nodes, y=t["posterior"], marker_color=GREEN)), "Posterior weights τ_iq",
                        "θ node", "", 260, False))
    frame = pd.DataFrame(dict(node=nodes, prior_weight=t["prior_weight"], likelihood=t["likelihood"],
                              unnormalised=t["unnormalised"], posterior=t["posterior"]))
    table(frame, 6, height=260)
    st.markdown(f"Σ posterior weights = **{w['posterior_sum']:.10f}** · posterior mean (EAP) = {w['eap']:.4f}")
    interpret("The likelihood says which abilities explain this person's answers; the prior says how common each "
              "ability is in the population. Their product, normalised to sum to one, is the posterior. Its weights "
              "are how the person's responses are shared out across the nodes in the expected counts.")
elif phase == "M-step":
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=nodes, y=w["N_item"], name="N_jq expected respondents", marker_color="#b7d3f6"))
        fig.add_trace(go.Bar(x=nodes, y=w["R_item"], name="R_jq expected correct", marker_color=BLUE))
        show(fig_layout(fig.update_layout(barmode="overlay"), f"Expected counts for {d.item_ids[item]}", "θ node", "count", 300))
        st.caption(f"Σ N_jq = {w['N_item'].sum():.3f} = observed respondents to this item ({int(d.mask[:, item].sum())}).")
    with right:
        obs_p = np.divide(w["R_item"], w["N_item"], out=np.full_like(w["R_item"], np.nan), where=w["N_item"] > 1e-8)
        fig = go.Figure()
        keep = w["N_item"] > 0.5
        fig.add_trace(go.Scatter(x=nodes[keep], y=obs_p[keep], mode="markers", name="R/N at nodes (N > 0.5)",
                                 marker=dict(size=9, color=BLUE)))
        xx = np.linspace(nodes.min(), nodes.max(), 200)
        fig.add_trace(go.Scatter(x=xx, y=1 / (1 + np.exp(-(xx - rec["b_in"][item]))), name="curve before", line=dict(color=GREY, dash="dash")))
        fig.add_trace(go.Scatter(x=xx, y=1 / (1 + np.exp(-(xx - rec["b"][item]))), name="curve after", line=dict(color=ORANGE)))
        show(fig_layout(fig, "Item curve fitted to expected counts", "θ node", "P(correct)", 300))
    st.latex(r"b_j\leftarrow b_j+\frac{\sum_q\left(N_{jq}P_{jq}-R_{jq}\right)}{\sum_qN_{jq}P_{jq}(1-P_{jq})}")
    table(pd.DataFrame(w["m_trace"]), 6)
    st.markdown(f"Variance update: σ² = Σ τ θ² / N = **{rec['sd'] ** 2:.5f}** (σ = {rec['sd']:.4f})"
                + ("" if res.settings.estimate_variance else " — fixed, not estimated"))
else:
    st.markdown(f"Observed marginal log-likelihood: before cycle **{rec['loglik_in']:.4f}** → after **{rec['loglik']:.4f}** "
                f"(change {rec['loglik'] - rec['loglik_in']:+.5f}). Max |Δb| = {rec['max_b_change']:.2e}, "
                f"|Δσ| = {rec['sd_change']:.2e}; tolerance {tol:.0e} → "
                + ("**stop**" if max(rec['max_b_change'], rec['sd_change']) < tol else "continue"))

st.subheader("D · Convergence")
its = np.array([h["iteration"] for h in H])
left, right = st.columns(2)
with left:
    show(lines(its, {"Marginal log-likelihood": [h["loglik"] for h in H]}, "Observed marginal log-likelihood", "EM cycle",
               "log L", colors={"Marginal log-likelihood": GREEN}))
with right:
    fig = lines(its, {"max |Δb|": [h["max_b_change"] for h in H], "|Δσ|": [h["sd_change"] for h in H]},
                "Maximum change per cycle (log scale)", "EM cycle", "Change")
    fig.add_hline(y=tol, line_dash="dot", annotation_text="tolerance")
    show(fig.update_yaxes(type="log"))
sel = st.multiselect("Items to trace", d.item_ids, d.item_ids[:5])
show(lines(np.r_[0, its], {i: np.r_[H[0]["b_in"][d.item_ids.index(i)], [h["b"][d.item_ids.index(i)] for h in H]] for i in sel},
           "Item-difficulty trajectories", "EM cycle (0 = start)", "b"))

st.subheader("E · Results")
labels("edu", "ref")
st.markdown(f"**Status:** {res.status} · {res.iterations} EM cycles · σ² = {res.variance:.4f} "
            f"(SE {fmt(res.se_variance)}) · log L = {res.loglik:.3f} · {res.seconds:.2f} s")
items = pd.DataFrame(dict(item=d.item_ids, b_python=res.b, se_observed=res.se_b, se_complete_data=res.se_b_complete))
if ref and ref["status"] == "Executed":
    items["b_TAM"], items["se_TAM"] = ref["b"], ref["se_b"]
    items["difference"] = items.b_python - items.b_TAM
if d.has_truth:
    items["true_b"] = d.true_b
table(items, height=300)
download(items, f"mml_items_{d.identity}.csv")
st.caption(f"log L uses node weights renormalised to sum to 1 ({res.loglik:.4f}). TAM reports it with weights = "
           f"normal density × node spacing ({res.loglik_tam:.4f}); these differ only when the grid cuts off the tails "
           "of the ability distribution, and the estimates are identical either way. The reference check compares the "
           "TAM-convention value.")
st.caption("se_observed inverts the observed marginal information (accounts for not knowing each θ); "
           "se_complete_data = 1/√Σ_q N_jq P(1−P) treats expected counts as data and is smaller — TAM's se.xsi uses this.")
if ref:
    if ref["status"] == "Executed":
        st.markdown(f"**Reference comparison** · TAM {ref['version']}, σ² = {ref['variance']:.4f}")
        table(compare("MML", dict(b=res.b, se_b=res.se_b_complete, se_label=" (complete-data, as TAM)",
                                  variance=res.variance, loglik=res.loglik_tam), ref, d.n_persons))
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
    c[4].metric("True vs estimated σ", f"{np.std(d.true_theta):.3f} / {res.sd:.3f}")
interpret("- MML-EM uses **all** persons, including extreme scores, through the ability distribution; it does not "
          "estimate individual abilities.\n- Its validity depends on the assumed distribution (normal here). A badly "
          "misspecified distribution can bias item estimates.\n- More quadrature nodes approximate the integral "
          "better; 21 nodes on [−6, 6] matches TAM's default.\n- Convergence here means the parameters stopped "
          "changing; the log-likelihood trace should rise and level off.")
