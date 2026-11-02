import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.explorer import reference_check, root_section
from components.ui import (BLUE, GREEN, GREY, LAB_TABS, NAVY, ORANGE, PURPLE, code_view, data, download, fig_layout, fmt,
                           interpret, labels, page_setup, pattern_text, person, settings, show, table, teaching_tabs)
from irt_models import RASCH_FAMILY, pattern_terms
from scoring import eap, eapsum, mle, posterior, quadrature, recursion_trace, score_distribution, score_table, tcc_inverse
from scoring.tcc import monotonicity, tcc_limits

page_setup("7 · TCC & EAPsum Lab", "Scoring from the summed score only: invert the test characteristic curve, or average the summed-score posterior.")
d, s = data(), settings()
p = person()
y = d.Y[p]
obs = ~np.isnan(y)
items_obs = [it for it, o in zip(d.items, obs) if o]
y_obs = y[obs]
r = int(np.nansum(y))
st.markdown(f"Examinee **{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · summed score "
            f"**r = {r}** on {int(obs.sum())} observed item(s)", unsafe_allow_html=True)
if not obs.all():
    st.caption("This examinee has missing responses: T(θ) and the score distribution use only the observed items.")
tabs = st.tabs(LAB_TABS)

with tabs[0]:
    st.markdown("""
**What is estimated?** Two summed-score estimates. **TCC inversion** finds the θ whose *expected* summed score equals
the observed one. **EAPsum** is the posterior mean of θ given only the summed score r.

**What information does it use?** The summed score r (not which items were answered correctly) and the item
parameters; EAPsum also uses a prior. TCC inversion is a **total-score, method-of-moments** estimator; EAPsum is
**Bayesian**.

**When is it useful?** Reporting a score-conversion table (every examinee with the same r gets the same θ), test
equating, and situations where scoring by pattern would be hard to explain to stakeholders.

**What does it need?** A monotone TCC (true when every slope a > 0). TCC inversion has no finite solution at r = 0,
at the maximum score, and — with 3PL/4PL items — for scores at or below Σc (or at or above Σd).
""")
    interpret("For Rasch-family models the summed score is a sufficient statistic: the MLE depends on the pattern only "
              "through r, and the MLE equation ℓ′ = 0 *is* T(θ) = r. So pattern MLE and TCC inversion coincide. For 2PL "
              "and other models the sufficient statistic is a weighted score, so two patterns with the same r get "
              "different MLEs but the same TCC estimate (Example B).")

with tabs[1]:
    teaching_tabs(
        [r"T(\theta)=\sum_{j\in O_i}\sum_k s_{jk}P_{jk}(\theta),\qquad T(\hat\theta_{TCC})=r_i",
         r"\theta^{(t+1)}=\theta^{(t)}+\frac{r_i-T(\theta^{(t)})}{T'(\theta^{(t)})},\qquad SE\approx\frac{\sqrt{\sum_j\mathrm{Var}_j(\hat\theta)}}{T'(\hat\theta)}",
         r"f_j(s\mid\theta)=\sum_k f_{j-1}(s-k\mid\theta)\,P_{jk}(\theta),\qquad f_0(0\mid\theta)=1",
         r"\hat\theta_{EAPsum}(r)=\frac{\sum_q\theta_q\,f_J(r\mid\theta_q)\,w_q}{\sum_q f_J(r\mid\theta_q)\,w_q}"],
        "T(θ) is strictly increasing (T′ = Σ_j Σ_k k P′_jk > 0 when every a > 0), so the root is unique when it exists; "
        "safeguarded Newton with a bisection fallback finds it. The delta-method SE propagates the conditional SD of the "
        "summed score through the slope of the TCC — an approximation, labelled as such. The Lord–Wingersky recursion "
        "builds P(R = r | θ) exactly by adding one item at a time, a probability convolution.",
        "- s_jk = k: points for category k · T(θ): expected summed score on the observed items\n"
        "- T′(θ): slope of the TCC · Var_j(θ): conditional variance of item j's score\n"
        "- f_j(s | θ): probability of summed score s on the first j items",
        "**Rasch MLE = TCC inversion.** For Rasch-family items d log P_jk/dθ = D(k − E_j(θ)), so "
        "ℓ′(θ) = D Σ_j (y_j − E_j(θ)) = D (r − T(θ)). Setting ℓ′ = 0 is exactly T(θ) = r. For the 2PL "
        "ℓ′ = D Σ_j a_j (y_j − P_j), whose root depends on Σ a_j y_j, not on r.")

with tabs[2]:
    labels("edu", "demo")
    st.markdown("### TCC inversion")
    lo, hi = tcc_limits(d.items, y[None, :])
    mono, min_slope = monotonicity(items_obs, np.linspace(-s.bound, s.bound, 401))
    st.caption(f"T(−∞) = {lo[0]:.3f}, T(+∞) = {hi[0]:.3f}; observed r = {r}. Monotonicity check on ±{s.bound:g}: "
               f"{'T′ > 0 everywhere' if mono else 'T′ ≤ 0 somewhere'} (smallest slope {min_slope:.2e}) ⇒ "
               f"{'at most one root' if mono else 'root may not be unique'}.")

    def curve_fn(grid):
        t = pattern_terms(d.items, np.tile(y, (len(grid), 1)), grid)
        return {"r − T(θ)": r - t["expected"]}

    res, hist, pos = root_section("tcc", d.items, y, s, "TCC", tcc_inverse, curve_fn, "r − T(θ)",
                                  extra_cols=("expected", "slope"))
    grid = np.linspace(-s.bound, s.bound, 401)
    T = pattern_terms(d.items, np.tile(y, (len(grid), 1)), grid)["expected"]
    fig = go.Figure(go.Scatter(x=grid, y=T, name="T(θ)", line=dict(color=NAVY, width=2)))
    fig.add_hline(y=r, line=dict(color=ORANGE, dash="dash"), annotation_text=f"r = {r}")
    if hist:
        shown = hist[: (pos or len(hist))]
        fig.add_trace(go.Scatter(x=[h["theta"] for h in shown], y=[h["expected"] for h in shown], mode="markers+text",
                                 text=[str(k + 1) for k in range(len(shown))], textposition="top center",
                                 name="iterates θ⁽ᵗ⁾", marker=dict(color=PURPLE, size=9)))
    show(fig_layout(fig, "Test characteristic curve and the observed summed score", "θ", "expected summed score", 330))
    th_t = tcc_inverse(d.items, y[None, :], s).theta[0]
    if np.isfinite(th_t):
        rows = [dict(item=it.id, model=it.model, observed=int(v), expected_score=float(it.expected(np.array([th_t]))[0]))
                for it, v in zip(items_obs, y_obs)]
        frame = pd.DataFrame(rows)
        st.markdown(f"**Item expected scores at θ̂_TCC = {th_t:.4f}** — they add up to the observed r, item by item they need not match.")
        table(frame, 4, height=min(38 * (len(frame) + 1) + 3, 300))
        st.caption(f"Σ expected = {frame['expected_score'].sum():.6f} · Σ observed = {frame['observed'].sum()}")

    st.markdown("### EAPsum: the Lord–Wingersky recursion")
    nodes, w = quadrature(s)
    q = st.slider("Quadrature node to inspect", 1, len(nodes), int(np.argmin(np.abs(nodes))) + 1, key="lw_node")
    trace = recursion_trace(items_obs, nodes[q - 1])
    max_r = len(trace[-1][2])
    tab = pd.DataFrame([dict(after=name, **{f"s={k}": (f[k] if k < len(f) else 0.0) for k in range(max_r)})
                        for name, _, f in trace])
    st.caption(f"Row j: P(summed score = s on the first j items | θ = {nodes[q - 1]:.4f}). Each row is the previous row "
               "convolved with the item's category probabilities; every row sums to 1. Scores not yet reachable have "
               "probability 0.")
    table(tab, 5, height=min(38 * (len(tab) + 1) + 3, 360))
    if len(trace) > 1:
        name, P, f = trace[1]
        st.caption(f"Example: after {name}, f₁(s) = P_{{{name},s}}: " + ", ".join(f"{v:.4f}" for v in P) + ".")
    tabsum, (nodes_, w_, F) = score_table(items_obs, s)
    post_r = F[:, r] * w_ / (F[:, r] * w_).sum()
    nodes_e, w_e, ll_e, post_e = posterior(d.items, y[None, :], s)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nodes_, y=post_r, mode="lines+markers", name=f"given r = {r} (EAPsum)", line=dict(color=GREEN)))
        fig.add_trace(go.Scatter(x=nodes_e, y=post_e[0], mode="lines+markers", name="given the pattern (EAP)", line=dict(color=BLUE, dash="dot")))
        fig.update_xaxes(range=[s.prior_mean - 5 * s.prior_sd, s.prior_mean + 5 * s.prior_sd])
        show(fig_layout(fig, "Posterior weights over the quadrature nodes (μ ± 5σ shown)", "θ", "posterior weight", 320))
    with right:
        sf = pd.DataFrame(tabsum)
        sf["this examinee"] = np.where(sf["score"] == r, "◀", "")
        table(sf, 4, height=320)
    download(pd.DataFrame(tabsum), "eapsum_score_table.csv", "Export EAPsum score table (CSV)", key="eapsum_export")

with tabs[3]:
    labels("edu")
    t_res = tcc_inverse(d.items, y[None, :], s)
    e_res = eapsum(d.items, y[None, :], s)
    m_res = mle(d.items, y[None, :], s)
    a_res = eap(d.items, y[None, :], s)
    c = st.columns(5)
    c[0].metric("θ̂ TCC", fmt(t_res.theta[0]))
    c[1].metric("SE (delta method)", fmt(t_res.se[0]))
    c[2].metric("θ̂ EAPsum", fmt(e_res.theta[0]))
    c[3].metric("PSD (EAPsum)", fmt(e_res.se[0]))
    c[4].metric("pattern MLE / EAP", f"{fmt(m_res.theta[0], 3)} / {fmt(a_res.theta[0], 3)}")
    st.markdown(f"**TCC status:** {t_res.status[0]}")
    Y = d.Y
    allm, allt = mle(d.items, Y, s), tcc_inverse(d.items, Y, s)
    alle, alls = eap(d.items, Y, s), eapsum(d.items, Y, s)
    raw = np.nansum(Y, 1)
    fin = np.isfinite(allm.theta) & np.isfinite(allt.theta)
    rasch = all(it.model in RASCH_FAMILY for it in d.items)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw[fin], y=allm.theta[fin], mode="markers", name="pattern MLE", marker=dict(color=BLUE, size=7, opacity=0.6)))
        fig.add_trace(go.Scatter(x=raw[fin], y=allt.theta[fin], mode="markers", name="TCC inversion",
                                 marker=dict(color=ORANGE, size=9, symbol="line-ew-open", line=dict(width=2))))
        show(fig_layout(fig, "MLE and TCC estimate by summed score", "summed score", "θ", 330))
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw, y=alle.theta, mode="markers", name="pattern EAP", marker=dict(color=BLUE, size=7, opacity=0.6)))
        fig.add_trace(go.Scatter(x=raw, y=alls.theta, mode="markers", name="EAPsum",
                                 marker=dict(color=GREEN, size=9, symbol="line-ew-open", line=dict(width=2))))
        show(fig_layout(fig, "EAP and EAPsum by summed score", "summed score", "θ", 330))
    gap = float(np.max(np.abs(allm.theta[fin] - allt.theta[fin]))) if fin.any() else np.nan
    same_obs = (~np.isnan(Y)).all(1)
    interpret(("All items are Rasch-family, so the pattern MLE and the TCC estimate should be identical: the largest "
               f"difference over {int(fin.sum())} examinees with finite estimates is {gap:.2e}. "
               if rasch else
               f"Items here are not all Rasch-family, so the pattern MLE spreads around the TCC estimate within a score "
               f"(largest difference {gap:.3f}): which items were answered correctly carries information beyond r. ")
              + "EAPsum is a single value per summed score (on complete data) while pattern EAP varies within a score; "
                "averaged over the patterns with that score, pattern EAP is close to EAPsum. "
              + ("" if same_obs.all() else "Examinees with missing responses are scored on their own observed items, so "
                                            "their summed scores are not on the same scale as complete records."))
    with st.expander("Reference check against R for this examinee"):
        labels("ref")
        st.caption("irtQ INV.TCC (bisection on a grid) and EAP.SUM (Gauss–Hermite nodes); mirt EAPsum on a rectangular "
                   "grid. irtQ does not score data with missing responses by summed score (it treats them as 0).")
        reference_check("tcc", "TCC", d.items, y, s, packages=("irtQ",))
        reference_check("eapsum", "EAPsum", d.items, y, s, packages=("irtQ", "mirt"))

with tabs[4]:
    code_view([tcc_inverse, score_distribution, score_table, eapsum], """
`tcc_inverse` solves r − T(θ) = 0 with the same bracketed Newton solver as WLE and reports scores outside
(T(−∞), T(+∞)) as having no finite solution. `score_distribution` is the Lord–Wingersky recursion vectorised over
nodes; `score_table` turns it into EAPsum and PSD for every summed score; `eapsum` scores each examinee on the items they
answered (grouping examinees with the same observed-item set).
""")
