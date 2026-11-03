"""Polytomous IRT MMLE–EM Learning Lab — page 1 · Overview.

Run:  streamlit run Polytomous_MMLE_EM.py      (or: streamlit run app.py)
"""
import streamlit as st

from components.ui import ROUTE, data, interpret, mode, page_setup, table
from core.em import DESIGN_NAMES
from r_reference.bridge import ROLE, check_r

page_setup("1 · Overview — Polytomous IRT MMLE–EM Learning Lab",
           "How PCM, RSM, GPCM and GRM item parameters are estimated by marginal maximum likelihood with the EM "
           "algorithm — under single-group, concurrent, fixed-item and concurrent fixed-item designs, with fixed or "
           "estimated, normal or empirical ability distributions. Every number comes from the working engine.")


@st.cache_data(ttl=600, show_spinner=False)
def r_status():
    return check_r()


st.subheader("Learning objective")
st.markdown("A learner should be able to **inspect, reproduce and explain every numerical calculation** used in "
            "polytomous MMLE–EM calibration: category probabilities → latent distribution → posterior → expected "
            "counts → constrained item optimisation → latent-distribution update → convergence.")

st.subheader("Route")
text = {
    "1 Overview": "Orientation, objectives, model families, reference status.",
    "2 Model Explorer": "Equations, category curves, adjacent odds vs cumulative boundaries, information, conversions.",
    "3 Prior Explorer": "Fixed / estimated, normal / empirical ability distributions; posterior and one-step update.",
    "4 Data Simulation": "Simulate any model × design × distribution, or upload responses and bank anchors.",
    "5 Quadrature Lab": "Gauss–Hermite vs equally spaced vs empirical grids; accuracy of marginal probabilities.",
    "6 EM Walkthrough": "N = 5, J = 3, Q = 3 — thirteen clickable steps with every intermediate matrix.",
    "7 Calibration Designs": "Single, concurrent, FIPC (OEM/MEM), MG-FIPC; identification; comparison laboratory.",
    "8 Results": "Item, group and prior estimates, convergence trajectories, inspect any iteration.",
    "9 Parameter Recovery": "Monte Carlo bias, RMSE, MAD, empirical SD, SE coverage, convergence rate, runtime.",
    "10 Model Fit": "Observed vs predicted category proportions, residuals, summed scores, AIC/BIC.",
    "11 Drift Analysis": "Anchor drift vs population change; free vs fixed-anchor calibration; displacement.",
    "12 Reference Validation": "Run TAM, irtQ and mirt on the same data and settings; differences reported.",
}
cols = st.columns(3)
for k, name in enumerate(ROUTE):
    with cols[k % 3]:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(text[name])

st.subheader("Model families")
table([dict(family="Rasch", model="Partial Credit (PCM)", discrimination="fixed at 1",
            parameters="item step difficulties δ_jh"),
       dict(family="Rasch", model="Rating Scale (RSM)", discrimination="fixed at 1",
            parameters="item location b_j + shared steps τ_h (Σ τ = 0), estimated jointly"),
       dict(family="General", model="Generalized Partial Credit (GPCM)", discrimination="estimated a_j > 0",
            parameters="a_j and step difficulties δ_jh (adjacent categories)"),
       dict(family="General", model="Graded Response (GRM)", discrimination="estimated a_j > 0",
            parameters="a_j and ordered thresholds b_j1 < … < b_jK (cumulative boundaries)")])

st.subheader("Calibration designs × ability priors")
table([dict(design=DESIGN_NAMES[k], data="one group" if k in ("single", "fipc") else "two or more groups/forms",
            fixed_items="none" if k in ("single", "concurrent") else "bank anchors",
            metric="latent distribution of the reference group" if k in ("single", "concurrent") else "anchor parameters")
       for k in DESIGN_NAMES])
st.caption("Ability priors: A · fixed normal, B · estimated normal, C · fixed empirical, D · estimated empirical. "
           "These are latent ability distributions in the marginal likelihood — not Bayesian priors on item "
           "parameters (this lab estimates items by pure MMLE, without item-parameter penalties).")

d = data()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active dataset", d.name.split(" · ")[0])
c2.metric("Examinees × items", f"{d.N:,} × {d.J}")
c3.metric("Groups / anchors", f"{d.G} / {len(d.anchors)}")
c4.metric("Mode", mode())

st.subheader("Reference software")
info = r_status()
rows = [dict(package=p, role=ROLE[p], version=info.get(p) or "not installed") for p in ("TAM", "irtQ", "mirt")]
table(rows)
if info["available"]:
    st.success(f"{info['R']} · {info['reason']}. Page 12 runs the reference packages on the current data.")
else:
    st.warning(f"Reference validation unavailable: {info['reason']} The Python engine still runs; reference "
               "results are reported as unverified.")

interpret("Use **Guided** mode (sidebar) for structured steps and knowledge checks, or **Exploration** mode for every "
          "control. Settings chosen in the sidebar — model, design, prior, quadrature, tolerances — apply to every "
          "page. Choosing a simulated dataset sets the model and design it was generated with; you can then change "
          "them to see how a misspecified choice behaves.")
