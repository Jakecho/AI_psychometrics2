import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.explorer import newton_section, reference_check
from components.ui import (BLUE, GREEN, GREY, LAB_TABS, NAVY, ORANGE, PURPLE, code_view, data, fig_layout, fmt, interpret,
                           labels, page_setup, pattern_text, person, settings, show, table, teaching_tabs)
from scoring import eap, map_estimate, mle
from scoring.likelihood import curves

page_setup("5 · MAP Learning Lab", "Maximum a posteriori: the mode of the posterior — the likelihood's peak pulled toward the prior.")
d, s = data(), settings()
p = person()
y = d.Y[p]
items = d.items
mu, sd = s.prior_mean, s.prior_sd
st.markdown(f"Examinee **{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · prior "
            f"N({mu:g}, {sd:g}²)", unsafe_allow_html=True)
tabs = st.tabs(LAB_TABS)

with tabs[0]:
    st.markdown("""
**What is estimated?** The *most probable* θ after combining the responses with a prior: the peak of the posterior.

**What information does it use?** The full response pattern and a prior g(θ). It is a **Bayesian** estimator, but
computed by optimisation (like MLE) instead of integration (like EAP).

**When is it useful?** When a finite estimate is needed for every pattern and integration is inconvenient — for
example inside adaptive testing algorithms. With a normal prior the log-posterior is the log-likelihood plus a
quadratic penalty, so Newton–Raphson applies directly.

**What does it need?** A prior. Like EAP, MAP is shrunk toward the prior mean; with a flat prior (σ → ∞) it becomes
the MLE.
""")
    interpret("The prior acts like an extra, always-answered item centred at μ whose pull grows with distance: its score "
              "−(θ − μ)/σ² is a straight line. MAP is where the likelihood's push ℓ′(θ) and the prior's pull cancel.")

with tabs[1]:
    teaching_tabs(
        [r"\hat\theta_{MAP}=\arg\max_\theta\big[\ell_i(\theta)+\log g(\theta)\big],\qquad "
         r"\log g(\theta)=-\frac{(\theta-\mu)^2}{2\sigma^2}+\text{const}",
         r"U_{MAP}(\theta)=U_{ML}(\theta)-\frac{\theta-\mu}{\sigma^2},\qquad "
         r"\frac{dU_{MAP}}{d\theta}=\ell''(\theta)-\frac{1}{\sigma^2}",
         r"\theta^{(t+1)}=\theta^{(t)}-\frac{U_{MAP}(\theta^{(t)})}{\ell''(\theta^{(t)})-1/\sigma^2}",
         r"SE_{MAP}\approx\Big[I(\hat\theta)+\frac1{\sigma^2}\Big]^{-1/2}\quad(\text{observed: }[-\ell''(\hat\theta)+1/\sigma^2]^{-1/2})"],
        "The prior adds −1/σ² to the curvature everywhere, which makes the log-posterior strictly concave for Rasch-family, "
        "2PL, GPCM and GRM items, so Newton converges from any start. With 3PL/4PL items ℓ″ can still be positive away "
        "from the peak; the same safeguards as for MLE apply (Fisher scoring, step cap, halving). The SE is the posterior "
        "curvature at the mode — a normal approximation to the posterior.",
        "- U_ML = ℓ′: likelihood score · −(θ − μ)/σ²: prior score\n- μ, σ: prior mean and SD\n"
        "- I(θ): test information (irtQ's SE uses it); −ℓ″: observed information (mirt's SE uses it)",
        "**From the posterior to the update.** log p(θ | y) = ℓ(θ) + log g(θ) − log p(y). The last term does not depend "
        "on θ, so maximising the posterior is maximising ℓ + log g. Differentiating twice gives U_MAP and its slope "
        "above; a Newton step solves the linearised equation U_MAP(θ⁽ᵗ⁾) + U′_MAP(θ⁽ᵗ⁾)(θ − θ⁽ᵗ⁾) = 0.")

with tabs[2]:
    labels("edu", "demo")
    res, hist, pos = newton_section("map", items, y, s, "MAP", map_estimate, prior=(mu, sd))
    if pos:
        h = hist[pos - 1]
        th = h["theta"]
        t_lik = h["grad"] + (th - mu) / sd ** 2
        st.markdown("**Score decomposition at θ⁽ᵗ⁾**")
        table([dict(likelihood_score=t_lik, prior_score=-(th - mu) / sd ** 2, posterior_score=h["grad"],
                    likelihood_curvature=h["hess"] + 1 / sd ** 2, prior_curvature=-1 / sd ** 2,
                    posterior_curvature=h["hess"])], 6)

with tabs[3]:
    labels("edu")
    r = map_estimate(items, y[None, :], s)
    ml = mle(items, y[None, :], s)
    ea = eap(items, y[None, :], s)
    c = st.columns(5)
    c[0].metric("θ̂ (MAP)", fmt(r.theta[0]))
    c[1].metric("SE [I + 1/σ²]^-½", fmt(r.se[0]))
    c[2].metric("SE observed", fmt(r.extra["se_observed"][0]))
    c[3].metric("MLE", fmt(ml.theta[0]))
    c[4].metric("EAP", fmt(ea.theta[0]))
    st.markdown(f"**Status:** {r.status[0]} after {int(r.iterations[0])} iteration(s).")
    grid = np.linspace(mu - 4 * sd - 1, mu + 4 * sd + 1, 401)
    cv = curves(items, y, grid)
    logprior = -(grid - mu) ** 2 / (2 * sd ** 2)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=cv["loglik"] - cv["loglik"].max(), name="ℓ(θ) (shifted)", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=grid, y=logprior, name="log g(θ)", line=dict(color=ORANGE)))
        lp = cv["loglik"] + logprior
        fig.add_trace(go.Scatter(x=grid, y=lp - lp.max(), name="log posterior (shifted)", line=dict(color=GREEN, width=3)))
        fig.add_vline(x=r.theta[0], line=dict(color=GREEN, dash="dot"), annotation_text="MAP",
                      annotation_position="top right" if r.theta[0] >= ml.theta[0] else "top left")
        if np.isfinite(ml.theta[0]):
            fig.add_vline(x=ml.theta[0], line=dict(color=BLUE, dash="dot"), annotation_text="MLE",
                          annotation_position="bottom left" if r.theta[0] >= ml.theta[0] else "bottom right")
        fig.update_yaxes(range=[-8, 0.5])
        show(fig_layout(fig, "Log-likelihood + log-prior = log-posterior", "θ", "log scale (each shifted to max 0)", 340))
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=cv["score"], name="likelihood score ℓ′", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=grid, y=-(grid - mu) / sd ** 2, name="prior score", line=dict(color=ORANGE)))
        fig.add_trace(go.Scatter(x=grid, y=cv["score"] - (grid - mu) / sd ** 2, name="posterior score", line=dict(color=GREEN, width=3)))
        fig.add_hline(y=0, line=dict(color=GREY))
        span = np.abs(cv["score"]).max() + 1
        fig.update_yaxes(range=[-span, span])
        show(fig_layout(fig, "Score functions: MAP is where the posterior score is 0", "θ", "score", 340))
    st.markdown("**Prior sensitivity** — change the prior here (this page only) and watch MAP move between the prior "
                "mean and the MLE.")
    c1, c2 = st.columns(2)
    mu_l = c1.slider("Prior mean", -3.0, 3.0, float(mu), 0.1, key="map_mu")
    sds = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])
    rows = []
    for v in sds:
        rr = map_estimate(items, y[None, :], s.with_(prior_mean=mu_l, prior_sd=float(v)))
        rows.append(dict(prior_sd=v, MAP=rr.theta[0], SE=rr.se[0]))
    sens = pd.DataFrame(rows)
    with c1:
        fig = go.Figure(go.Scatter(x=sens["prior_sd"], y=sens["MAP"], mode="lines+markers", name="MAP", line=dict(color=GREEN)))
        fig.add_hline(y=mu_l, line=dict(color=ORANGE, dash="dot"), annotation_text="prior mean")
        if np.isfinite(ml.theta[0]):
            fig.add_hline(y=ml.theta[0], line=dict(color=BLUE, dash="dot"), annotation_text="MLE")
        fig.update_xaxes(type="log", tickvals=list(sds), ticktext=[f"{v:g}" for v in sds])
        show(fig_layout(fig, "MAP as the prior SD grows (log axis)", "prior SD σ", "MAP", 300, legend=False))
    with c2:
        table(sens, 4)
    st.markdown("**MAP and MLE on the same response patterns** (first 25 examinees of this dataset)")
    Y = d.Y[:25]
    mm, ll = map_estimate(items, Y, s), mle(items, Y, s)
    comp = pd.DataFrame(dict(examinee=d.person_ids[:25], summed_score=np.nansum(Y, 1).astype(int),
                             MLE=ll.theta,
                             MAP=mm.theta, SE_MAP=mm.se))
    table(comp, 4, height=300)
    interpret(f"MAP = {r.theta[0]:.3f}. "
              + (f"The MLE is {ml.theta[0]:.3f}; MAP sits {abs(r.theta[0] - ml.theta[0]):.3f} closer to the prior mean "
                 f"{mu:g}. " if np.isfinite(ml.theta[0]) else "This pattern has no finite MLE; the prior gives MAP a finite peak. ")
              + "MAP and EAP are both shrunk; they differ because the mode and the mean of a skewed posterior differ. "
                "Extreme patterns show the largest shrinkage (table above).")
    with st.expander("Reference check against R for this examinee"):
        labels("ref")
        st.caption("irtQ MAP: Fisher scoring with SE [I + 1/σ²]^-½. mirt MAP: nlm with observed-curvature SE.")
        reference_check("map", "MAP", items, y, s, packages=("irtQ", "mirt"))

with tabs[4]:
    from scoring.map_estimate import objective
    code_view([map_estimate, objective], """
`objective` adds the normal log-prior, its score and its curvature to the likelihood quantities; `map_estimate` reuses
the same safeguarded maximiser as MLE. Because the prior keeps the objective bounded, every pattern — including
extreme ones — has a finite maximum.
""")
