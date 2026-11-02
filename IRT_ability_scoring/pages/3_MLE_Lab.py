import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.explorer import newton_section, reference_check
from components.ui import (BLUE, GREEN, GREY, LAB_TABS, NAVY, ORANGE, code_view, data, fig_layout, fmt, interpret, labels,
                           page_setup, pattern_text, person, settings, show, table, teaching_tabs)
from irt_models import extreme_status
from scoring import mle, tcc_inverse
from scoring.likelihood import curves, nonconcave
from scoring.solvers import maximize

page_setup("3 · MLE Learning Lab", "Maximum likelihood: find the θ that makes the observed response pattern most probable.")
d, s = data(), settings()
p = person()
y = d.Y[p]
items = d.items
st.markdown(f"Examinee **{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · "
            f"{d.name}", unsafe_allow_html=True)
tabs = st.tabs(LAB_TABS)

with tabs[0]:
    st.markdown("""
**What is estimated?** The ability value θ̂ at which the examinee's *whole response pattern* is most likely, treating the
item parameters as known.

**What information does it use?** Only the responses and the item response functions — no prior and no population
assumption. It is a **likelihood-based, frequentist** estimator.

**When is it useful?** Long tests, when an estimate free of prior influence is wanted (e.g. research on item
functioning, or when examinees from very different populations must be scored the same way).

**What does it need?** A finite maximum. That fails for **extreme patterns** (all lowest or all highest categories): the
likelihood keeps rising as θ → ±∞. On short tests the finite MLE is **biased outward** (away from the items' centre);
WLE corrects this. For 3PL/4PL items the log-likelihood need not be concave, so a local maximum is possible.
""")
    interpret("Read the likelihood as a *plausibility score* for each candidate θ. Every item multiplies in the probability "
              "of the category actually observed; a correct answer to a hard item makes low θ implausible, a wrong answer "
              "to an easy item makes high θ implausible. The MLE is where these pressures balance: the score function "
              "ℓ′(θ) — the sum of each item's push — equals zero.")

with tabs[1]:
    teaching_tabs(
        [r"L_i(\theta)=\prod_{j\in O_i}P_{j,y_{ij}}(\theta),\qquad \ell_i(\theta)=\sum_{j\in O_i}\log P_{j,y_{ij}}(\theta)",
         r"\ell_i'(\theta)=\sum_{j\in O_i}\frac{P'_{j,y_{ij}}(\theta)}{P_{j,y_{ij}}(\theta)}=0",
         r"\theta^{(t+1)}=\theta^{(t)}-\frac{\ell_i'(\theta^{(t)})}{\ell_i''(\theta^{(t)})}",
         r"SE(\hat\theta)=\frac{1}{\sqrt{I(\hat\theta)}},\quad I(\theta)=\sum_{j}\sum_k\frac{P'_{jk}(\theta)^2}{P_{jk}(\theta)}"
         r"\qquad\Big(\text{observed: }1/\sqrt{-\ell''(\hat\theta)}\Big)"],
        "Newton–Raphson approximates ℓ by a parabola at θ⁽ᵗ⁾ and jumps to the parabola's peak. Near the solution the "
        "number of correct digits roughly doubles each iteration. **Safeguards** used here: if ℓ″ ≥ 0 (possible with "
        "3PL/4PL items) the step uses the expected information instead (Fisher scoring); steps are capped at "
        f"{s.max_step:g} logit; a step that lowers ℓ is halved; θ is kept inside ±{s.bound:g}. Convergence: "
        f"|Δθ| < {s.tol:g}.\n\n**Model-specific score contributions** d log P/dθ:\n"
        "- Rasch/2PL: D·a·(y − P)  ·  3PL/4PL: D·a·(P − c)(d − P)(y − P) / ((d − c)·P·(1 − P))\n"
        "- PCM/GPCM/RSM: D·a·(k − E_j(θ))  ·  GRM: D·a·(1 − P*_k − P*_k+1)",
        "- O_i: items examinee i answered (missing responses are skipped)\n- y_ij: observed category\n"
        "- ℓ′, ℓ″: first and second derivatives of the log-likelihood\n- I(θ): expected (Fisher) information; "
        "−ℓ″(θ̂): observed information (equal for Rasch, 2PL, PCM, GPCM, RSM)",
        "**Why extreme patterns have no finite MLE.** Every model here has a lowest-category probability that decreases "
        "and a highest-category probability that increases in θ. If all observed responses are highest-category, every "
        "term of ℓ is increasing, so ℓ′(θ) > 0 for all θ and no root exists; the supremum is approached as θ → +∞. "
        "The same argument with signs reversed holds for all-lowest patterns.\n\n**Fisher scoring step.** Replacing −ℓ″ "
        "by its expectation I(θ) gives θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ + ℓ′/I. For exponential-family items (Rasch, 2PL, PCM, GPCM, RSM) "
        "−ℓ″ = I exactly, so Newton and Fisher scoring coincide.")

with tabs[2]:
    labels("edu", "demo")
    res, hist, pos = newton_section("mle", items, y, s, "MLE", mle)
    ext = extreme_status(items, y[None, :])[0]
    if ext:
        st.warning("This is an **extreme pattern**: every Newton step pushes θ further out until the search bound "
                   f"±{s.bound:g} stops it. The bounded value is an artefact of the bound, **not an MLE**.")

with tabs[3]:
    labels("edu")
    r = mle(items, y[None, :], s)
    th, se = r.theta[0], r.se[0]
    c = st.columns(5)
    c[0].metric("θ̂ (MLE)", fmt(th))
    c[1].metric("SE 1/√I(θ̂)", fmt(se))
    c[2].metric("SE 1/√(−ℓ″(θ̂))", fmt(r.extra["se_observed"][0]))
    c[3].metric("iterations", int(r.iterations[0]))
    c[4].metric("ℓ(θ̂)", fmt(r.extra["loglik"][0]))
    st.markdown(f"**Status:** {r.status[0]}")
    if np.isfinite(th):
        st.markdown(f"Approximate 95% interval: θ̂ ± 1.96·SE = [{th - 1.96 * se:.3f}, {th + 1.96 * se:.3f}] "
                    "(Wald interval; its coverage is only approximate on short tests).")
    grid = np.linspace(-s.bound, s.bound, 801)
    cv = curves(items, y, grid)
    fig = go.Figure(go.Scatter(x=grid, y=cv["loglik"], name="ℓ(θ)", line=dict(color=BLUE, width=2)))
    if np.isfinite(th):
        fig.add_vline(x=th, line=dict(color=GREEN, dash="dot"), annotation_text=f"θ̂ = {th:.3f}")
    show(fig_layout(fig, f"Log-likelihood over the whole search range ±{s.bound:g}", "θ", "ℓ(θ)", 320, legend=False))
    if nonconcave(items):
        st.caption("3PL/4PL items present: ℓ can have more than one maximum, so the Newton solution is compared with an "
                   "801-point grid; if the grid is higher, Newton restarts from the grid maximum. "
                   + ("That happened here — the step-through tab shows the first run from θ⁽⁰⁾."
                      if r.extra["restarted"][0] else "Here the Newton solution is the grid maximum."))
    interpret("θ̂ is the peak of the curve above. Its SE comes from the curvature there: a sharply peaked likelihood "
              "(many informative items near θ̂) gives a small SE. "
              + ("Here the peak does not exist inside the range — the curve rises toward the bound — so no finite MLE "
                 "is reported. MAP, EAP and WLE (later pages) give finite values for this pattern."
                 if not np.isfinite(th) else
                 "The two SEs agree for Rasch-family and 2PL/GPCM items and can differ slightly for 3PL/4PL/GRM, where the "
                 "observed and expected information are not identical."))
    if d.name.startswith("Example B"):
        st.markdown("**Example B: the response pattern matters, not only the raw score**")
        both = np.array([[1, 0], [0, 1]], float)
        rb = mle(items, both, s)
        tb = tcc_inverse(items, both, s)
        table([dict(pattern="A = (1, 0)", raw_score=1, MLE=rb.theta[0], TCC=tb.theta[0]),
               dict(pattern="B = (0, 1)", raw_score=1, MLE=rb.theta[1], TCC=tb.theta[1])], 4)
        st.caption("Pattern B answers the more discriminating item (a = 2) correctly, which is stronger evidence of "
                   "ability than answering the a = 1 item; the 2PL MLE uses the weighted score Σ a_j y_j. TCC inversion "
                   "sees only the raw score and returns 0 for both.")
    with st.expander("Reference check against R for this examinee"):
        labels("ref")
        st.caption("irtQ ML: Fisher scoring, range ±bound, SE 1/√I. mirt ML: nlm, observed-information SE, ±∞ for "
                   "extreme patterns. TAM: Rasch-family items only.")
        reference_check("mle", "MLE", items, y, s)

with tabs[4]:
    code_view([mle, maximize], """
`scoring.mle.mle` builds the objective from `irt_models.pattern_terms` (ℓ, ℓ′, ℓ″ and I summed over the observed
items) and hands it to `scoring.solvers.maximize`, which is vectorised over examinees and records every iteration.
Extreme patterns are detected *before* interpretation (`extreme_status`) and reported as ±∞; the bounded optimiser
value is kept in `extra["theta_bounded"]` only so the path can be shown. For 3PL/4PL items `global_check` evaluates
ℓ on a grid to flag local maxima.
""")
