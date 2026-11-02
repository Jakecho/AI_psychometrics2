import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import logsumexp

from components.explorer import reference_check
from components.ui import (BLUE, GREEN, GREY, LAB_TABS, NAVY, ORANGE, code_view, data, download, fig_layout, fmt, interpret,
                           labels, page_setup, pattern_text, person, phase_cards, settings, show, step_controls, table,
                           teaching_tabs)
from irt_models import item_table
from scoring import convergence_table, eap, mle, node_table, posterior, quadrature
from scoring.eap import posterior_quantiles

page_setup("4 · EAP Learning Lab", "Expected a posteriori: the mean of the posterior, computed by numerical integration over θ.")
d, s = data(), settings()
p = person()
y = d.Y[p]
items = d.items
st.markdown(f"Examinee **{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · prior "
            f"N({s.prior_mean:g}, {s.prior_sd:g}²)", unsafe_allow_html=True)
tabs = st.tabs(LAB_TABS)

with tabs[0]:
    st.markdown("""
**What is estimated?** The *mean* of the posterior distribution of θ given the responses: the average ability among
examinees from the prior population who would produce this pattern.

**What information does it use?** The full response pattern **and** a prior g(θ) for the population.
It is a **Bayesian** estimator.

**When is it useful?** Short tests, adaptive tests, and any pattern without a finite MLE: EAP is always finite and
needs no iterations. Averaged over the population it has the smallest mean squared error.

**What does it need?** A sensible prior and accurate numerical integration. EAP is **shrunk toward the prior mean**:
for a fixed true θ it is biased inward, more so on short tests.
""")
    interpret("Think of the posterior as a histogram of plausible abilities. Each quadrature node is a candidate θ; it "
              "gets a weight proportional to *how likely the responses are there* × *how common that ability is in the "
              "population*. EAP is the weighted average of the nodes; the posterior SD is their weighted spread.")

with tabs[1]:
    teaching_tabs(
        [r"\hat\theta_{EAP}=\frac{\int\theta\,L_i(\theta)g(\theta)\,d\theta}{\int L_i(\theta)g(\theta)\,d\theta}"
         r"\approx\frac{\sum_q\theta_q L_i(\theta_q)w_q}{\sum_q L_i(\theta_q)w_q}",
         r"\pi_{iq}=\frac{\exp\{\ell_i(\theta_q)+\log w_q-M_i\}}{\sum_r\exp\{\ell_i(\theta_r)+\log w_r-M_i\}},\quad "
         r"M_i=\max_q\{\ell_i(\theta_q)+\log w_q\}",
         r"PSD_i=\sqrt{\sum_q(\theta_q-\hat\theta_{EAP})^2\pi_{iq}}"],
        "**Gauss–Hermite** (irtQ, default): nodes θ_q = μ + √2·σ·x_q and weights w_q = h_q/√π from the Hermite rule. "
        "The normal prior is built into the weights, and n nodes integrate polynomials of degree 2n − 1 exactly. "
        "**Rectangular** (mirt): equally spaced nodes on [−L, L] with w_q ∝ φ((θ_q − μ)/σ), normalised. All products "
        "are formed as sums of logs and normalised with log-sum-exp, so long tests do not underflow.",
        "- θ_q, w_q: quadrature nodes and prior weights (Σ w_q = 1)\n- ℓ_i(θ_q): log-likelihood at node q\n"
        "- π_iq: normalised posterior weight\n- PSD: posterior standard deviation (Bayesian uncertainty, not a "
        "frequentist SE)",
        "**Why log-sum-exp.** L_i(θ) is a product of up to hundreds of probabilities and can be smaller than 10⁻³⁰⁸ "
        "(the smallest double). Subtracting the largest log term M_i before exponentiating keeps the largest term at "
        "exp(0) = 1 and every other term in (0, 1]; the normalised weights are unchanged because M_i cancels.")

with tabs[2]:
    labels("edu", "demo")
    st.caption("Explore the integration here; these node choices apply on this page only (the sidebar settings drive "
               "every other page).")
    c1, c2, c3 = st.columns(3)
    rule = c1.selectbox("Quadrature rule", ["gauss-hermite", "rectangular"], index=["gauss-hermite", "rectangular"].index(s.quad_rule))
    n_nodes = c2.slider("Number of nodes", 3, 121, int(s.n_nodes), 2)
    lim = c3.slider("Rectangular limit ±L", 2.0, 8.0, float(s.theta_lim), 0.5, disabled=rule != "rectangular")
    s_local = s.with_(quad_rule=rule, n_nodes=n_nodes, theta_lim=lim)
    tab, m, psd = node_table(items, y, s_local)
    nodes = tab["node"]
    stages = ["Prior & nodes", "Likelihood at nodes", "Unnormalised posterior", "Normalise (log-sum-exp)",
              "Posterior mean", "Posterior SD"]
    cards = st.empty()
    pos = step_controls("eap", len(stages))
    stage = 6 if pos is None else pos
    with cards.container():
        phase_cards(stages, stage - 1 if pos is not None else -1)
    if pos is None:
        st.caption("Showing the completed calculation. Press **Initialize** to build it one stage at a time.")
    view = [s.prior_mean - 5 * s.prior_sd, s.prior_mean + 5 * s.prior_sd]
    outside = int(((nodes < view[0]) | (nodes > view[1])).sum())
    lik_scaled = np.exp(tab["log_likelihood"] - tab["log_likelihood"].max())
    a, x1, b, x2, c = st.columns([10, 1, 10, 1, 10])
    with a:
        fig = go.Figure(go.Scatter(x=nodes, y=lik_scaled if stage >= 2 else np.full_like(nodes, np.nan), mode="lines+markers",
                                   line=dict(color=BLUE, width=2), marker=dict(size=5)))
        fig.update_xaxes(range=view)
        show(fig_layout(fig, "Response likelihood L(θ_q)", "θ", "L ÷ max L", 290, legend=False))
        st.caption("How well does each node explain the answers? (scaled so the largest is 1)")
    x1.markdown("<div class='op'>×</div>", unsafe_allow_html=True)
    with b:
        fig = go.Figure(go.Bar(x=nodes, y=tab["prior_weight"], marker=dict(color=ORANGE), width=(np.ptp(nodes) / len(nodes)) * 0.7))
        fig.update_xaxes(range=view)
        show(fig_layout(fig, "Prior weight w_q", "θ", "w_q", 290, legend=False))
        st.caption("How much population mass sits at each node?")
    x2.markdown("<div class='op'>→</div>", unsafe_allow_html=True)
    with c:
        post = tab["posterior_weight"] if stage >= 4 else (tab["unnormalised"] / tab["unnormalised"].max() if stage >= 3 else np.full_like(nodes, np.nan))
        fig = go.Figure(go.Bar(x=nodes, y=post, marker=dict(color=GREEN), width=(np.ptp(nodes) / len(nodes)) * 0.7))
        if stage >= 5:
            fig.add_vline(x=m, line=dict(color=NAVY, dash="dot"), annotation_text=f"EAP {m:.3f}")
        fig.update_xaxes(range=view)
        show(fig_layout(fig, "Posterior weight π_q" if stage >= 4 else "Unnormalised L·w (÷ max)", "θ", "weight", 290, legend=False))
        st.caption("Combine and normalise so the weights sum to one.")
    if outside:
        st.caption(f"Charts show μ ± 5σ; {outside} outer node(s) lie beyond it with negligible prior weight "
                   f"(largest {tab['prior_weight'][(nodes < view[0]) | (nodes > view[1])].max():.1e}). The table lists every node.")
    lse = logsumexp(tab["log_unnormalised"])
    msgs = {1: f"{len(nodes)} {rule} nodes from {nodes.min():.3f} to {nodes.max():.3f}; prior weights sum to "
               f"{tab['prior_weight'].sum():.6f}.",
            2: f"ℓ(θ_q) summed over {int((~np.isnan(y)).sum())} observed items; largest log-likelihood "
               f"{tab['log_likelihood'].max():.4f} at θ = {nodes[tab['log_likelihood'].argmax()]:.3f}.",
            3: f"log(L·w) = ℓ + log w; largest value M = {tab['log_unnormalised'].max():.4f}.",
            4: f"log Σ exp(ℓ + log w) = {lse:.6f}; π_q = exp(ℓ + log w − {lse:.4f}); Σ π_q = {tab['posterior_weight'].sum():.12f}.",
            5: f"EAP = Σ θ_q π_q = {m:.6f}.",
            6: f"PSD = √Σ (θ_q − EAP)² π_q = {psd:.6f}."}
    for k in range(1, stage + 1):
        st.markdown(f"**{k} · {stages[k - 1]}.** {msgs[k]}")
    cols = ["node", "prior_weight", "log_likelihood", "likelihood", "log_unnormalised", "unnormalised", "posterior_weight",
            "theta_x_weight", "sq_dev_x_weight"][: [2, 4, 6, 7, 8, 9][stage - 1]]
    frame = pd.DataFrame(tab)
    st.markdown("**Node table** (one row per quadrature node; columns appear as the stages advance)")
    table(frame[cols], 6, height=320)
    download(frame, "eap_quadrature_nodes.csv", "Export node table (CSV)", key="eap_export")
    st.markdown("**Inspect one node**")
    q = st.slider("Node", 1, len(nodes), int(np.argmax(tab["posterior_weight"])) + 1)
    th_q = nodes[q - 1]
    it = pd.DataFrame(item_table(items, y, th_q))[["item", "model", "response", "P_observed", "log_P"]]
    left, right = st.columns([3, 2])
    with left:
        table(it, 6, height=min(38 * (len(it) + 1) + 3, 300))
    with right:
        st.latex(rf"\theta_{{{q}}}={th_q:.4f}")
        st.latex(rf"\ell(\theta_{{{q}}})=\sum_j\log P_{{j,y_j}}={tab['log_likelihood'][q - 1]:.5f}")
        st.latex(rf"\log w_{{{q}}}={np.log(tab['prior_weight'][q - 1]):.5f}")
        st.latex(rf"\pi_{{{q}}}=\exp\big({tab['log_unnormalised'][q - 1]:.4f}-({lse:.4f})\big)={tab['posterior_weight'][q - 1]:.6f}")
    st.markdown("**Does the integral converge?** EAP and PSD for increasing numbers of nodes, against adaptive "
                "quadrature (scipy `integrate.quad`, relative tolerance 10⁻¹²) on the continuous prior.")
    conv, m_ref, s_ref = convergence_table(items, y, s)
    conv = pd.DataFrame(conv)
    fig = go.Figure()
    for r_, colr in (("gauss-hermite", BLUE), ("rectangular", ORANGE)):
        sub = conv[conv["rule"] == r_]
        fig.add_trace(go.Scatter(x=sub["nodes"], y=np.maximum(np.abs(sub["error_EAP"]), 1e-17), mode="lines+markers",
                                 name=r_, line=dict(color=colr)))
    fig.update_yaxes(type="log")
    show(fig_layout(fig, f"|EAP(n) − reference|; reference EAP = {m_ref:.8f}", "number of nodes", "absolute error", 300))
    table(conv, 8, height=260)
    st.caption(f"Rectangular nodes use the grid ±{s.theta_lim:g} from the sidebar; reference PSD = {s_ref:.8f}.")
    def err(rule_, n_):
        row = conv[(conv["rule"] == rule_) & (conv["nodes"] == n_)]
        return float(abs(row["error_EAP"].iloc[0])) if len(row) else np.nan
    n_obs = int((~np.isnan(y)).sum())
    interpret(f"With 41 nodes the EAP error is {err('gauss-hermite', 41):.1e} (Gauss–Hermite) and "
              f"{err('rectangular', 41):.1e} (rectangular). Gauss–Hermite is exact when L(θ) is a low-degree polynomial; "
              f"with {n_obs} observed items the likelihood is a narrow peak that only a few Gauss–Hermite nodes reach, so "
              "it needs more nodes than its reputation suggests, while an equally spaced grid covering the peak converges "
              "quickly. On short tests (Example A) both rules agree to many digits with few nodes. Check this table before "
              "trusting the last digits of any EAP.", title="What the convergence study shows")

with tabs[3]:
    labels("edu")
    r = eap(items, y[None, :], s)
    q = posterior_quantiles(items, y[None, :], s)[0]
    ml = mle(items, y[None, :], s).theta[0]
    c = st.columns(5)
    c[0].metric("EAP", fmt(r.theta[0]))
    c[1].metric("posterior SD", fmt(r.se[0]))
    c[2].metric("posterior median", fmt(q[1]))
    c[3].metric("95% posterior interval", f"[{q[0]:.3f}, {q[2]:.3f}]")
    c[4].metric("MLE (for contrast)", fmt(ml))
    st.markdown(f"**Integration:** {r.status[0]} (sidebar settings). No iterations: EAP is a ratio of two sums.")
    grid = np.linspace(s.prior_mean - 4 * s.prior_sd, s.prior_mean + 4 * s.prior_sd, 161)
    rows_ = []
    for sd_ in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        rr = eap(items, y[None, :], s.with_(prior_sd=sd_, quad_rule="gauss-hermite", n_nodes=61))
        rows_.append(dict(prior_sd=sd_, EAP=rr.theta[0], PSD=rr.se[0]))
    left, right = st.columns(2)
    with left:
        sens = pd.DataFrame(rows_)
        fig = go.Figure(go.Scatter(x=sens["prior_sd"], y=sens["EAP"], mode="lines+markers", name="EAP", line=dict(color=GREEN)))
        if np.isfinite(ml):
            fig.add_hline(y=ml, line=dict(color=BLUE, dash="dot"), annotation_text="MLE")
        show(fig_layout(fig, "Prior sensitivity: EAP as the prior SD grows", "prior SD σ", "EAP", 300, legend=False))
    with right:
        table(sens, 4)
    shrink = "toward" if np.isfinite(ml) and abs(r.theta[0] - s.prior_mean) < abs(ml - s.prior_mean) else "relative to"
    interpret(f"EAP = {r.theta[0]:.3f} with posterior SD {r.se[0]:.3f}. "
              + (f"It lies {shrink} the prior mean {s.prior_mean:g} compared with the MLE {ml:.3f}: the prior pulls the "
                 "estimate in, more strongly when the test is short or the prior narrow. As the prior SD grows the EAP "
                 "approaches the MLE (left chart)." if np.isfinite(ml) else
                 "This pattern has no finite MLE, yet EAP is finite: the prior keeps the posterior proper.")
              + " The posterior SD describes uncertainty about this examinee *given the prior*; it is not the sampling "
                "SE of a frequentist estimator.")
    with st.expander("Reference check against R for this examinee"):
        labels("ref")
        st.caption("irtQ EAP uses the same Gauss–Hermite nodes (or the rectangular grid passed as weights); mirt EAP is "
                   "compared on a rectangular grid of the same size.")
        reference_check("eap", "EAP", items, y, s, packages=("irtQ", "mirt"))

with tabs[4]:
    code_view([quadrature, posterior, eap, node_table], """
`quadrature` returns nodes and prior weights for either rule. `posterior` evaluates ℓ_i(θ_q) for every examinee and node
in one pass (`irt_models.loglik_grid`), adds log w_q and normalises with `scipy.special.logsumexp`. `eap` is then two
matrix products. `node_table` returns every intermediate column for one examinee — exactly the numbers in the table on
the *Numerical example* tab.
""")
