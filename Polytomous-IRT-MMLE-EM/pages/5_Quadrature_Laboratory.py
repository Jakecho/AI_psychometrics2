import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import quad as integrate
from scipy.stats import norm

from components.ui import bars, data, four_views, guided, interpret, latest_fit, lines, page_setup, table
from core import models as M
from core.em import marginal_loglik, start_items
from core.quadrature import gauss_hermite, illustrative_distribution, make_grid, moments

page_setup("5 · Quadrature Laboratory", "Numerical integration over ability: Gauss–Hermite, equally spaced grids and "
                                        "discrete empirical distributions — and how many nodes are enough.")
guided(["Compare Gauss–Hermite and equally spaced nodes/weights for the same number of points.",
        "Watch the marginal category probabilities converge to the adaptive-integration value as Q grows.",
        "Compute the dataset's marginal log-likelihood under each rule.",
        "Note: irtQ's Quadrature = c(Q, bound) is an equally spaced grid, not Gauss–Hermite."],
       "With 3 Gauss–Hermite points for N(0,1), the nodes and weights are…",
       ["θ = (−√3, 0, √3), π = (1/6, 2/3, 1/6)", "θ = (−1, 0, 1), π = (1/3, 1/3, 1/3)",
        "θ = (−2, 0, 2), π = (0.25, 0.5, 0.25)", "θ = (−6, 0, 6), π normalised normal density"],
       "θ = (−√3, 0, √3), π = (1/6, 2/3, 1/6)",
       "The 3-point probabilists' Hermite rule integrates polynomials up to degree 5 exactly against N(0,1).", key="quad")

c1, c2, c3, c4 = st.columns(4)
Q = c1.slider("Nodes Q", 3, 81, 11, 2)
bound = c2.slider("Equal-grid bound ±", 2.0, 8.0, 4.0, 0.5)
mu = c3.number_input("Distribution mean μ", -2.0, 2.0, 0.0, 0.1)
sd = c4.number_input("Distribution SD σ", 0.3, 3.0, 1.0, 0.1)
xg, wg = gauss_hermite(Q, mu, sd)
eq = make_grid("equal", Q, bound, mu, sd)
shape = st.selectbox("Empirical (discrete) distribution on the equal grid", ["skewed", "bimodal", "uniform", "normal"])
we = illustrative_distribution(eq.nodes, shape, mu, sd)
c1, c2 = st.columns(2)
with c1:
    bars(np.round(xg, 3), {"Gauss–Hermite": wg}, f"Gauss–Hermite (nodes move with μ, σ)", key="gh")
with c2:
    bars(np.round(eq.nodes, 3), {"equally spaced (normal masses)": eq.base_weights, f"empirical ({shape})": we},
         f"Equally spaced on ±{bound:g}", key="eq")
table(pd.DataFrame([dict(rule="Gauss–Hermite", nodes=Q, mean=moments(xg, wg)[0], sd=moments(xg, wg)[1]),
                    dict(rule="Equally spaced, normal masses", nodes=Q, mean=moments(eq.nodes, eq.base_weights)[0],
                         sd=moments(eq.nodes, eq.base_weights)[1]),
                    dict(rule=f"Equally spaced, {shape}", nodes=Q, mean=moments(eq.nodes, we)[0], sd=moments(eq.nodes, we)[1])]),
      digits=6)

st.subheader("Accuracy: marginal category probabilities")
c1, c2, c3 = st.columns(3)
model = c1.selectbox("Model", M.MODELS, index=2)
a = c2.slider("a", 0.3, 4.0, 2.5, 0.1, disabled=model in M.RASCH)
loc = c3.slider("location", -3.0, 3.0, 1.0, 0.1)
d = tuple(loc + np.array([-1.0, 0.0, 1.0]))
item = M.make_item("q", model, 1.0 if model in M.RASCH else a, d)
exact = np.array([integrate(lambda t, k=k: M.probs(item, [t])[0, k] * norm.pdf(t, mu, sd), mu - 12 * sd, mu + 12 * sd,
                            epsabs=1e-13, limit=200)[0] for k in range(item.ncat)])
rows, err_gh, err_eq = [], [], []
Qs = list(range(3, 82, 2))
for q in Qs:
    x1, w1 = gauss_hermite(q, mu, sd)
    p1 = w1 @ M.probs(item, x1)
    g2 = make_grid("equal", q, bound, mu, sd)
    p2 = g2.base_weights @ M.probs(item, g2.nodes)
    err_gh.append(np.max(np.abs(p1 - exact)))
    err_eq.append(np.max(np.abs(p2 - exact)))
lines(np.array(Qs), {"Gauss–Hermite": np.log10(np.maximum(err_gh, 1e-17)),
                     f"Equally spaced ±{bound:g}": np.log10(np.maximum(err_eq, 1e-17))},
      "log₁₀ max |p̂_k − p_k| against adaptive integration", xl="nodes Q", yl="log10 error", key="acc")
x1, w1 = gauss_hermite(Q, mu, sd)
table(pd.DataFrame(dict(category=range(item.ncat), adaptive=exact, gauss_hermite=w1 @ M.probs(item, x1),
                        equally_spaced=eq.base_weights @ M.probs(item, eq.nodes))), digits=8)
st.caption("A narrow equal grid truncates the tails (error floor set by the bound); Gauss–Hermite converges fast for "
           "smooth integrands but steep items (large a) need more nodes. Tolerance used by adaptive integration: 1e-13.")

st.subheader("Marginal log-likelihood of the active dataset")
ds = data()
res = latest_fit()
items = res.items if res is not None else None
if items is None:
    st.caption("No calibration yet for this dataset — using starting values for the items.")
    try:
        from core.em import plan
        from components.ui import config
        cfg = config()
        items = start_items(ds, cfg.model, plan(ds, cfg)["fixed"])
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()
rows = []
for q in [3, 5, 7, 11, 15, 21, 31, 41, 61]:
    x1, w1 = gauss_hermite(q, mu, sd)
    g2 = make_grid("equal", q, bound, mu, sd)
    rows.append(dict(Q=q, gauss_hermite=marginal_loglik(ds, items, np.tile(x1, (ds.G, 1)), np.tile(w1, (ds.G, 1))),
                     equally_spaced=marginal_loglik(ds, items, np.tile(g2.nodes, (ds.G, 1)), np.tile(g2.base_weights, (ds.G, 1)))))
table(pd.DataFrame(rows), digits=4)
interpret("Low-order rules (e.g. 3 points) are teaching approximations: the marginal log-likelihood changes by several "
          "units until Q is around 15–21. Estimated and empirical priors need a **fixed** grid — Gauss–Hermite nodes "
          "move with μ and σ, so the item likelihood would have to be recomputed at new nodes inside the prior "
          "M-step. The lab therefore uses Gauss–Hermite only with a fixed normal prior.")

four_views(
    "Integration replaces ∫ L(θ) g(θ) dθ by Σ_q π_q L(θ_q). **Gauss–Hermite** chooses nodes and weights optimal for a "
    "normal g. An **equally spaced** grid uses normalised density values at evenly spaced points (irtQ, TAM, mirt "
    "default). A **fixed empirical** distribution supplies its own masses on a grid; an **estimated empirical** "
    "distribution estimates them.",
    ["$$\\int L(\\theta)\\,\\phi(\\theta;\\mu,\\sigma)\\,d\\theta\\approx\\sum_{q=1}^{Q}\\frac{w_q^{GH}}{\\sqrt{2\\pi}}\\,L(\\mu+\\sigma x_q)$$",
     "$$\\theta_q=-B+\\frac{2B(q-1)}{Q-1},\\qquad \\pi_q=\\frac{\\phi(\\theta_q;\\mu,\\sigma)}{\\sum_s\\phi(\\theta_s;\\mu,\\sigma)}$$"],
    None, [gauss_hermite, make_grid])
