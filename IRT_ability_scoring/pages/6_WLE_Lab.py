import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.explorer import reference_check, root_section
from components.ui import (BLUE, GREEN, GREY, LAB_TABS, NAVY, ORANGE, PURPLE, code_view, data, fig_layout, fmt, interpret,
                           labels, page_setup, pattern_text, person, settings, show, table, teaching_tabs)
from irt_models import J_EQUALS_IPRIME, pattern_terms
from scoring import mle, weighted_likelihood_mode, wle
from scoring.solvers import find_root
from scoring.wle import adjustment_curves, equation

page_setup("6 · WLE Learning Lab", "Warm's weighted likelihood: remove the MLE's first-order bias by adding J/(2I) to the score.")
d, s = data(), settings()
p = person()
y = d.Y[p]
items = d.items
exp_family = all(it.model in J_EQUALS_IPRIME for it in items)   # J = I′ for every item
st.markdown(f"Examinee **{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · items: "
            f"{', '.join(d.models)}", unsafe_allow_html=True)
tabs = st.tabs(LAB_TABS)

with tabs[0]:
    st.markdown("""
**What is estimated?** A corrected MLE whose bias, for a fixed true θ, is zero to first order (O(1/n) bias removed).

**What information does it use?** The full response pattern and the item parameters — no prior. It is a
**bias-adjusted likelihood** estimator.

**When is it useful?** Short and medium tests where examinees must be scored without a population prior (e.g. PISA
and many Rasch programmes report WLEs), and whenever extreme patterns need a finite, prior-free value.

**What does it need?** The adjustment term J(θ)/(2I(θ)), which depends on the model. For Rasch, 2PL, PCM, GPCM,
RSM and GRM items it equals I′/(2I); items with a guessing or slipping asymptote (3PL, 4PL) need the general J.
""")
    interpret("Why is the MLE biased? On a finite test the score equation is noisy, and the noise is not symmetric: the "
              "likelihood is flatter on the side of θ̂ where the test has less information, so random errors there are "
              "stretched further. On average θ̂ lands too far from the items' centre. Warm's term pushes the score function "
              "back by the expected amount of that stretch, B(θ)·I(θ) = J/(2I).")

with tabs[1]:
    teaching_tabs(
        [r"\text{Bias}(\hat\theta_{ML})\approx B(\theta)=\frac{J(\theta)}{2I(\theta)^2},\quad "
         r"I(\theta)=\sum_j\sum_k\frac{P_{jk}'^2}{P_{jk}},\quad J(\theta)=\sum_j\sum_k\frac{P'_{jk}P''_{jk}}{P_{jk}}",
         r"U_W(\theta)=\ell'(\theta)+\frac{J(\theta)}{2I(\theta)}=0",
         r"\text{Rasch: } I=\sum_jP_j(1-P_j),\quad J=I'=\sum_jP_j(1-P_j)(1-2P_j)",
         r"\text{Rasch, 2PL, PCM, GPCM, RSM, GRM: } J=I'\ \Rightarrow\ \hat\theta_W=\arg\max_\theta\ L(\theta)\sqrt{I(\theta)}",
         r"SE(\hat\theta_W)\approx 1/\sqrt{I(\hat\theta_W)}"],
        "U_W is solved by safeguarded Newton (the derivative of J/(2I) is taken by a central difference, h = 10⁻⁵) inside "
        "a bracket where U_W changes sign; a step that leaves the bracket is replaced by bisection. **Extreme patterns:** "
        "as θ → +∞ for an all-highest pattern, ℓ′ → 0⁺ while J/(2I) tends to a negative value for logistic items, so U_W "
        "crosses zero and the WLE is finite.\n\n**Two definitions in software.** irtQ (method = \"WL\") and TAM solve "
        "ℓ′ + J/(2I) = 0 (Warm). mirt (method = \"WLE\") maximises ℓ + ½ log I, i.e. solves ℓ′ + I′/(2I) = 0. They agree "
        "only when J = I′.",
        "- I(θ): test information · I′ = dI/dθ\n- J(θ): Warm's J (sum over items and categories)\n"
        "- B(θ): first-order bias of the MLE\n- P′, P″: first and second θ-derivatives of category probabilities",
        "**Where J = I′ comes from.** For divide-by-total items d log P_k/dθ = Da(k − E) and d² log P_k/dθ² = −(Da)²Var, "
        "the same for every category. Then I = (Da)²Var, I′ = (Da)³μ₃ and J = Σ_k P_k(Da)(k − E)·(Da)²[(k − E)² − Var] = "
        "(Da)³μ₃, so J = I′. In general I′ − J = Σ_k P_k·dlogP_k·d²logP_k. **GRM:** with w_k = P*_k(1 − P*_k), "
        "P_k·dlogP_k = Da(w_k − w_k+1) and d²logP_k = −(Da)²(w_k + w_k+1), so the sum is −(Da)³Σ_k(w_k² − w_k+1²), "
        "which telescopes to 0 (w₀ = w_m+1 = 0): J = I′ although the GRM is not an exponential family. **3PL/4PL:** "
        "the asymptotes make d log P/dθ = Da(P − c)(d − P)(y − P)/((d − c)P(1 − P)), and the sum is not zero.")

with tabs[2]:
    labels("edu", "demo")

    def curve_fn(grid):
        c = adjustment_curves(items, y, grid)
        out = {"U_W = ℓ′ + J/(2I)": c["score"] + c["warm"], "ℓ′ (MLE score)": c["score"]}
        if not exp_family:
            out["ℓ′ + I′/(2I) (mirt's WLE)"] = c["score"] + c["jeffreys"]
        return out

    res, hist, pos = root_section("wle", items, y, s, "WLE", wle, curve_fn, "Adjusted score U_W(θ)",
                                  extra_cols=("score", "adjustment", "info", "J"))
    if pos:
        h = hist[pos - 1]
        st.latex(rf"U_W(\theta^{{({pos - 1})}})=\ell'+\frac{{J}}{{2I}}={h['score']:.5f}+\frac{{{h['J']:.5f}}}{{2\times{h['info']:.5f}}}"
                 rf"={h['score']:.5f}+({h['adjustment']:.5f})={h['U']:.5f}")

with tabs[3]:
    labels("edu")
    r = wle(items, y[None, :], s)
    ml = mle(items, y[None, :], s)
    jm = weighted_likelihood_mode(items, y[None, :], s)
    c = st.columns(5)
    c[0].metric("θ̂ (WLE, Warm)", fmt(r.theta[0]))
    c[1].metric("SE 1/√I(θ̂)", fmt(r.se[0]))
    c[2].metric("MLE", fmt(ml.theta[0]))
    c[3].metric("L·√I mode (mirt)", fmt(jm.theta[0]))
    c[4].metric("Warm − L·√I", fmt(r.theta[0] - jm.theta[0], 6))
    st.markdown(f"**Status:** {r.status[0]} after {int(r.iterations[0])} iteration(s).")
    grid = np.linspace(-4, 4, 241)
    t = pattern_terms(items, np.zeros((len(grid), len(items))), grid)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=t["J"], name="J(θ)", line=dict(color=PURPLE, width=3)))
        fig.add_trace(go.Scatter(x=grid, y=t["Iprime"], name="I′(θ)", line=dict(color=ORANGE, dash="dash", width=2)))
        show(fig_layout(fig, "J(θ) and I′(θ) for the whole test", "θ", "value", 320))
        st.caption("Identical curves for Rasch, 2PL, PCM, GPCM, RSM and GRM items; only 3PL/4PL items separate them. "
                   + ("They coincide for this test." if exp_family else "They differ for this test (3PL/4PL items)."))
    with right:
        fig = go.Figure(go.Scatter(x=grid, y=t["J"] / (2 * t["info"] ** 2), line=dict(color=NAVY, width=2)))
        fig.add_hline(y=0, line=dict(color=GREY))
        show(fig_layout(fig, "First-order MLE bias B(θ) = J / (2I²)", "θ", "bias (logits)", 320, legend=False))
        st.caption("Positive above the test's centre, negative below: the MLE is pushed outward; WLE removes this term.")
    st.markdown("**Extreme patterns on this test** — no finite MLE, but a finite WLE")
    lo_pat = np.zeros(len(items))
    hi_pat = np.array([it.max_score for it in items], float)
    E = np.vstack([lo_pat, hi_pat])
    we, me, jme = wle(items, E, s), mle(items, E, s), weighted_likelihood_mode(items, E, s)
    table(pd.DataFrame(dict(pattern=["all lowest", "all highest"], MLE=["−∞", "+∞"], WLE=we.theta, SE_WLE=we.se,
                            L_sqrtI_mode=jme.theta, status=we.status)), 4)
    interpret(f"WLE = {r.theta[0]:.3f}" + (f" versus MLE {ml.theta[0]:.3f}: the correction moved the estimate "
                                           f"{abs(r.theta[0] - ml.theta[0]):.3f} toward the test's centre. "
                                           if np.isfinite(ml.theta[0]) else
                                           ": finite although this pattern has no MLE. ")
              + ("For this test J = I′, so Warm's WLE and mirt's L·√I mode are the same number."
                 if exp_family else
                 f"Because the test contains {', '.join(m for m in d.models if m not in J_EQUALS_IPRIME)} items, "
                 f"J ≠ I′ and the two definitions differ by {abs(r.theta[0] - jm.theta[0]):.4f} for this examinee. irtQ "
                 "and TAM implement Warm's J; mirt's 'WLE' is the L·√I mode."))
    with st.expander("Reference check against R for this examinee"):
        labels("ref")
        st.caption("irtQ WL and TAM tam.wle solve Warm's equation (TAM: Rasch-family items only). mirt's WLE is compared "
                   "with the L·√I mode; the Warm-vs-mirt row is shown for information.")
        reference_check("wle", "WLE", items, y, s)

with tabs[4]:
    code_view([equation, wle, weighted_likelihood_mode, find_root], """
`equation` returns U_W(θ) = ℓ′ + J/(2I) (or ℓ′ + I′/(2I) for the mirt definition) together with its numerical
derivative; `find_root` keeps a sign-change bracket and falls back to bisection. J and I′ come from
`irt_models.pattern_terms`, which sums P·dlogP·(d²logP + dlogP²) and P·dlogP·(dlogP² + 2·d²logP) over categories.
""")
