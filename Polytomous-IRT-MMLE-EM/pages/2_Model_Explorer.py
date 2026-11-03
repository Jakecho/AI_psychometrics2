import numpy as np
import pandas as pd
import streamlit as st

from components.ui import four_views, guided, interpret, lines, mode, page_setup, table
from core import models as M
from core.models import Item, make_item

page_setup("2 · Model Explorer", "Category response functions, adjacent-category odds, cumulative boundaries, expected "
                                 "scores and information for PCM, RSM, GPCM and GRM — with parameter conversions.")

guided(["Pick a model and move the step/threshold sliders: watch where adjacent category curves cross.",
        "For GPCM raise and lower a: curves sharpen or flatten, information rises or falls.",
        "Compare the GRM's cumulative boundaries with the GPCM's adjacent-category steps.",
        "Check the conversion table: every representation gives identical probabilities."],
       "In the PCM, where do the curves of categories k−1 and k cross?",
       ["At θ = δ_k", "At θ = the item mean location", "At θ = 0", "Where P(Y ≥ k) = 0.5"], "At θ = δ_k",
       "Adjacent-category odds are log(P_k/P_{k−1}) = θ − δ_k, which is zero exactly at θ = δ_k. (θ where "
       "P(Y ≥ k) = 0.5 is the GRM threshold b_k.)", key="model")

c0, c1, c2, c3 = st.columns(4)
model = c0.selectbox("Model", M.MODELS, index=2, format_func=lambda m: f"{m} · {M.MODEL_NAMES[m]}")
ncat = c1.slider("Categories", 2, 6, 4)
D = c2.radio("D", [1.0, 1.702], horizontal=True)
a = c3.slider("Discrimination a", 0.2, 3.0, 1.0, 0.05, disabled=model in M.RASCH)
K = ncat - 1
st.markdown("**Steps / thresholds**" + (" — RSM: item location b plus shared steps τ (Σ τ = 0)" if model == "RSM" else ""))
cols = st.columns(K + (1 if model == "RSM" else 0))
if model == "RSM":
    b = cols[0].slider("location b", -3.0, 3.0, 0.0, 0.05)
    raw = [cols[h + 1].slider(f"τ{h + 1}", -3.0, 3.0, float(np.linspace(-1, 1, K)[h]), 0.05) for h in range(K)]
    tau = np.array(raw) - np.mean(raw)
    if abs(np.mean(raw)) > 1e-9:
        st.caption(f"τ re-centred to sum to zero: {np.round(tau, 3).tolist()}")
    item = Item("demo", "RSM", 1.0, (), b=b, tau=tuple(tau))
else:
    d = [cols[h].slider(("b" if model == "GRM" else "δ") + str(h + 1), -4.0, 4.0, float(np.linspace(-1.5, 1.5, K)[h]),
                        0.05) for h in range(K)]
    if model == "GRM" and np.any(np.diff(d) <= 0):
        st.error("GRM thresholds must be strictly increasing (b₁ < b₂ < …). Adjust the sliders.")
        st.stop()
    item = make_item("demo", model, 1.0 if model in M.RASCH else a, tuple(d))
theta = np.linspace(-5, 5, 401)
P = M.probs(item, theta, D)
c1, c2 = st.columns(2)
with c1:
    lines(theta, {f"P(Y={k})": P[:, k] for k in range(ncat)}, "Category response functions", yl="probability", key="crf")
with c2:
    if model == "GRM":
        cum = M.cumulative_probs(item, theta, D)
        lines(theta, {f"P(Y≥{k + 1})": cum[:, k] for k in range(K)}, "GRM cumulative boundary curves",
              yl="probability", key="cum")
    else:
        odds = M.adjacent_odds(item, theta, D)
        lines(theta, {f"log P{k + 1}/P{k}": odds[:, k] for k in range(K)}, "Adjacent-category log-odds",
              yl="log-odds", key="odds")
c1, c2 = st.columns(2)
with c1:
    lines(theta, {"E[Y | θ]": M.expected_score(item, theta, D)}, "Expected score", yl="score", key="es")
with c2:
    lines(theta, {"I(θ)": M.information(item, theta, D)}, "Item information", yl="information", key="info")

st.subheader("Special cases and contrasts")
t1, t2 = st.tabs(["PCM = GPCM with a = 1", "GRM boundaries vs GPCM steps"])
with t1:
    if model in ("PCM", "GPCM", "RSM"):
        pc = make_item("pcm", "PCM", 1.0, item.d)
        gp1 = make_item("g1", "GPCM", 1.0, item.d)
        gpa = make_item("ga", "GPCM", a if model == "GPCM" else 1.6, item.d)
        diff = np.max(np.abs(M.probs(pc, theta, D) - M.probs(gp1, theta, D)))
        lines(theta, {**{f"PCM k={k}": M.probs(pc, theta, D)[:, k] for k in range(ncat)},
                      **{f"GPCM a={gpa.a:.2f} k={k}": M.probs(gpa, theta, D)[:, k] for k in range(ncat)}},
              "Same steps, slope 1 vs slope a", yl="probability",
              dash={f"GPCM a={gpa.a:.2f} k={k}": "dash" for k in range(ncat)}, key="pcm_gpcm")
        st.caption(f"Largest |P_PCM − P_GPCM(a = 1)| over the θ grid: {diff:.2e} (identical up to rounding).")
    else:
        st.info("Choose PCM, RSM or GPCM to see the slope special case.")
with t2:
    dd = np.sort(np.asarray(item.d))
    if np.all(np.diff(dd) > 0):
        g = make_item("grm", "GRM", item.a, tuple(dd))
        gp = make_item("gpcm", "GPCM", item.a, tuple(dd))
        lines(theta, {**{f"GRM k={k}": M.probs(g, theta, D)[:, k] for k in range(ncat)},
                      **{f"GPCM k={k}": M.probs(gp, theta, D)[:, k] for k in range(ncat)}},
              "Same numbers read as GRM thresholds (solid) vs GPCM steps (dashed)", yl="probability",
              dash={f"GPCM k={k}": "dash" for k in range(ncat)}, key="grm_gpcm")
        st.caption("GRM b_k is where P(Y ≥ k) = 0.5; GPCM δ_k is where P(Y = k) = P(Y = k−1). The same values give "
                   "different curves, so parameters are not interchangeable across the two families.")

st.subheader("Parameter conversions (TAM · irtQ · mirt)")
ic = M.to_intercepts(item, D)
back = M.from_intercepts("demo", "GRM" if model == "GRM" else ("PCM" if model in M.RASCH else "GPCM"),
                         ic["alpha"], ic["intercepts"], D)
err = np.max(np.abs(M.probs(back, theta, D) - P))
rows = [dict(representation="Canonical (this lab)", values=str(np.round(item.vector(), 4).tolist()))]
if model in M.RASCH:
    rows.append(dict(representation="TAM", values=str({k: np.round(v, 4).tolist() if not np.isscalar(v) else round(v, 4)
                                                     for k, v in M.to_tam(item).items()})))
if model != "RSM":
    rows.append(dict(representation="irtQ shape_df", values=str({k: (np.round(v, 4).tolist() if not isinstance(v, str)
                                                                    else v) for k, v in M.to_irtq(item).items()})))
rows.append(dict(representation="mirt (slope–intercept)", values=str({k: round(v, 4) for k, v in M.to_mirt(item, D).items()})))
rows.append(dict(representation="Back-converted from intercepts", values=str(np.round(back.vector(), 4).tolist())))
table(pd.DataFrame(rows))
st.caption(f"Probability reconstruction check: largest |P(original) − P(back-converted)| = {err:.2e}.")


def example():
    th = np.array([-1.0, 0.0, 1.0])
    lp = M.log_probs(item, th, D)
    rows = []
    for q, t in enumerate(th):
        if model == "GRM":
            z = D * item.a * (t - np.asarray(item.d))
            r = {"θ": t, **{f"P(Y≥{k + 1})": 1 / (1 + np.exp(-z[k])) for k in range(K)}}
        else:
            z = M.adjacent_logits([t], item.a, item.d, D)[0]
            r = {"θ": t, **{f"z_{k}": z[k] for k in range(ncat)}}
        r.update({f"P{k}": np.exp(lp[q, k]) for k in range(ncat)})
        rows.append(r)
    table(pd.DataFrame(rows), digits=5)
    st.caption("Adjacent models: z_k = D·a·(kθ − Σ_{h≤k} δ_h), P_k = exp(z_k)/Σ_c exp(z_c). GRM: P_k = P(Y≥k) − P(Y≥k+1).")


four_views(
    "Each polytomous model gives the probability of every score category as a function of ability θ. "
    "**Adjacent-category** models (PCM, RSM, GPCM) parameterise the odds of category k over k−1; **cumulative** "
    "models (GRM) parameterise the probability of scoring k or higher. Rasch-family models fix the slope at 1, "
    "so the raw score is sufficient and only location needs identification; GPCM and GRM estimate a slope per "
    "item, so both the location and the unit of θ must be fixed by the latent distribution or by anchors.",
    ["$$P_{jk}(\\theta)=\\frac{\\exp\\left[a_j\\sum_{h=1}^{k}(\\theta-\\delta_{jh})\\right]}{\\sum_{c=0}^{K_j}\\exp\\left[a_j\\sum_{h=1}^{c}(\\theta-\\delta_{jh})\\right]}$$",
     "PCM: a_j = 1. RSM: δ_jh = b_j + τ_h with Σ_h τ_h = 0. Adjacent log-odds: log(P_k/P_{k−1}) = D a (θ − δ_k).",
     "$$P(Y_j\\ge k\\mid\\theta)=\\operatorname{logit}^{-1}[a_j(\\theta-b_{jk})],\\quad P_{jk}=P(Y_j\\ge k)-P(Y_j\\ge k+1)$$",
     "Slope–intercept form (adjacent models): P_k ∝ exp(k·aθ + γ_k), γ_k = −a Σ_{h≤k} δ_h. GRM intercepts: c_k = −a b_k.",
     "$$I_j(\\theta)=\\sum_k\\frac{[\\partial P_{jk}/\\partial\\theta]^2}{P_{jk}}\\;=\\;D^2a_j^2\\operatorname{Var}(Y_j\\mid\\theta)\\ \\text{(adjacent models)}$$"],
    example, [M.log_probs_adjacent, M.log_probs_grm, M.information, M.to_intercepts],
    "Probabilities are computed on the log scale: log-sum-exp for adjacent models and a stable difference of "
    "logistic functions for the GRM (log σ(x_k) + log σ(−x_{k+1}) + log(1 − e^{x_{k+1}−x_k})).")

interpret("Increasing a sharpens every transition and concentrates information; shifting all steps by a constant "
          "moves the item along θ without changing its shape — the location indeterminacy that the latent mean "
          "(or anchors) must resolve. Reversed PCM/GPCM steps (δ₂ < δ₁) are allowed and mean a middle category is "
          "never the most likely; GRM thresholds cannot be reversed by construction.")
