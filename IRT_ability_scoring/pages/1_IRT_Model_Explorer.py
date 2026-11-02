import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREEN, GREY, NAVY, ORANGE, PURPLE, data, fig_layout, interpret, labels, page_setup, show,
                           table, teaching_tabs)
from irt_models import MODEL_NAMES, MODELS, make_item

page_setup("1 · IRT Model Explorer", "How each model turns ability θ into category probabilities, expected scores and information.")

PALETTE = [BLUE, ORANGE, GREEN, PURPLE, "#b8860b", "#0f8b8d", GREY]

EQUATIONS = {
    "Rasch": [r"P_j(\theta)=\frac{\exp(D(\theta-b_j))}{1+\exp(D(\theta-b_j))}"],
    "2PL": [r"P_j(\theta)=\frac{\exp(Da_j(\theta-b_j))}{1+\exp(Da_j(\theta-b_j))}"],
    "3PL": [r"P_j(\theta)=c_j+(1-c_j)\frac{\exp(Da_j(\theta-b_j))}{1+\exp(Da_j(\theta-b_j))}"],
    "4PL": [r"P_j(\theta)=c_j+(d_j-c_j)\frac{\exp(Da_j(\theta-b_j))}{1+\exp(Da_j(\theta-b_j))}"],
    "PCM": [r"P_{jk}(\theta)=\frac{\exp\sum_{v=1}^{k}D(\theta-\delta_{jv})}{\sum_{h=0}^{m_j}\exp\sum_{v=1}^{h}D(\theta-\delta_{jv})}"],
    "GPCM": [r"P_{jk}(\theta)=\frac{\exp\sum_{v=1}^{k}Da_j(\theta-\delta_{jv})}{\sum_{h=0}^{m_j}\exp\sum_{v=1}^{h}Da_j(\theta-\delta_{jv})}"],
    "RSM": [r"P_{jk}(\theta)=\frac{\exp\sum_{v=1}^{k}D(\theta-b_j-\tau_v)}{\sum_{h=0}^{m}\exp\sum_{v=1}^{h}D(\theta-b_j-\tau_v)}"],
    "GRM": [r"P^*_{jk}(\theta)=\frac{\exp(Da_j(\theta-b_{jk}))}{1+\exp(Da_j(\theta-b_{jk}))},\quad P^*_{j0}=1,\ P^*_{j,m+1}=0",
            r"P_{jk}(\theta)=P^*_{jk}(\theta)-P^*_{j,k+1}(\theta)"],
}
COMMON = [r"E_j(\theta)=\sum_k k\,P_{jk}(\theta),\qquad I_j(\theta)=\sum_k\frac{P'_{jk}(\theta)^2}{P_{jk}(\theta)}"]
EXPLAIN = {
    "Rasch": "One parameter per item: difficulty **b** (the θ where P = 0.5). All items are equally discriminating, so "
             "the summed score is a sufficient statistic for θ.",
    "2PL": "Adds discrimination **a > 0**: the slope of the curve at θ = b is D·a/4. Items with larger a separate "
           "nearby abilities more sharply and carry more information.",
    "3PL": "Adds a lower asymptote **c** (0 ≤ c < 1), the probability of a correct answer at very low ability "
           "(\"guessing\"). P(b) = (1 + c)/2, and the likelihood need not be concave in θ.",
    "4PL": "Adds an upper asymptote **d** (c < d ≤ 1): even very able examinees answer correctly with probability d "
           "(\"slipping\"). P(b) = (c + d)/2. Constraints: a > 0, 0 ≤ c < d ≤ 1.",
    "PCM": "Divide-by-total model for ordered categories 0..m. **δ_v** is the step difficulty where categories v−1 and v "
           "are equally likely. Steps need not be ordered (a reversed step shrinks the middle category). a = 1.",
    "GPCM": "The PCM with an item slope **a**. Like the PCM it is an exponential family in θ, so the weighted summed "
            "score Σ a_j y_j is sufficient for θ.",
    "RSM": "PCM in which every item shares the same threshold pattern **τ_v** (Σ τ_v = 0 here), shifted by an item "
           "location **b**: δ_jv = b_j + τ_v. Typical for Likert scales.",
    "GRM": "Difference model: the boundary curves P*_k = P(Y ≥ k) are 2PL curves with thresholds **b_1 < … < b_m** "
           "and a common slope a; category probabilities are differences of adjacent boundaries.",
}

st.subheader("A · Choose a model and its parameters")
c0, c1, c2 = st.columns([1.2, 1, 1])
model = c0.selectbox("Model", MODELS, format_func=lambda m: MODEL_NAMES[m], index=1)
D = c0.radio("Scaling constant D", [1.0, 1.702], horizontal=True, format_func=lambda v: f"{v:g}" + (" (≈ normal ogive)" if v != 1 else ""))
a = c1.slider("Discrimination a", 0.2, 3.0, 1.2, 0.05, disabled=model in ("Rasch", "PCM", "RSM"))
if model in ("Rasch", "2PL", "3PL", "4PL"):
    b = c1.slider("Difficulty b", -3.0, 3.0, 0.0, 0.1)
    c = c2.slider("Lower asymptote c", 0.0, 0.45, 0.2, 0.01, disabled=model not in ("3PL", "4PL"))
    dd = c2.slider("Upper asymptote d", 0.55, 1.0, 0.92, 0.01, disabled=model != "4PL")
    item = make_item("item", model, a=a, b=b, c=c, d=dd, D=D)
else:
    m = c1.slider("Number of categories", 2, 6, 4)
    if model == "RSM":
        b = c1.slider("Item location b", -3.0, 3.0, 0.0, 0.1)
        spread = c2.slider("Threshold spread (τ from −s to s)", 0.2, 3.0, 1.0, 0.1)
        steps = np.linspace(-spread, spread, m - 1) if m > 2 else np.zeros(1)
        steps = steps - steps.mean()
        item = make_item("item", "RSM", b=b, steps=steps, D=D)
    else:
        default = np.linspace(-1.5, 1.5, m - 1) if m > 2 else np.zeros(1)
        txt = c2.text_input("Step difficulties / thresholds (comma separated)", ", ".join(f"{v:.2f}" for v in default),
                            key=f"steps_{m}")
        try:
            steps = [float(v) for v in txt.split(",") if v.strip()]
        except ValueError:
            steps = list(default)
            c2.error("Could not read the numbers; using the default thresholds.")
        if len(steps) != m - 1:
            c2.warning(f"{model} with {m} categories needs {m - 1} values; using the default thresholds.")
            steps = list(default)
        item = make_item("item", model, a=a, steps=steps, D=D)
problems = item.problems()
if problems:
    st.error("Invalid parameters: " + "; ".join(problems))
    st.stop()
st.caption(item.describe())

teaching_tabs(EQUATIONS[model] + COMMON, EXPLAIN[model],
              "- θ: ability · D: scaling constant (1 here unless chosen)\n- a: discrimination · b: difficulty / location\n"
              "- c, d: lower / upper asymptotes (3PL, 4PL)\n- δ_v: step difficulties (PCM, GPCM) · τ_v: shared thresholds "
              "(RSM) · b_k: boundary thresholds (GRM)\n- P'_jk: derivative of a category probability with respect to θ")

st.subheader("B · Category probabilities, expected score and information")
labels("edu")
grid = np.linspace(-5, 5, 401)
P = item.probs(grid)
left, right = st.columns(2)
with left:
    fig = go.Figure()
    for k in range(item.n_cat):
        fig.add_trace(go.Scatter(x=grid, y=P[:, k], name=f"P(Y = {k})", line=dict(color=PALETTE[k % len(PALETTE)], width=2)))
    if model == "GRM":
        from scipy.special import expit
        for k, bk in enumerate(item.steps, start=1):
            fig.add_trace(go.Scatter(x=grid, y=expit(D * item.a * (grid - bk)), name=f"P*(Y ≥ {k})",
                                     line=dict(color=PALETTE[k % len(PALETTE)], dash="dot", width=1.5)))
    show(fig_layout(fig, "Category response curves" + (" (dotted: boundary curves P*)" if model == "GRM" else ""),
                    "θ", "probability", 360))
with right:
    show(fig_layout(go.Figure(go.Scatter(x=grid, y=item.expected(grid), line=dict(color=NAVY, width=2))),
                    "Expected item score E(θ)", "", "E(θ)", 180, legend=False))
    show(fig_layout(go.Figure(go.Scatter(x=grid, y=item.info(grid), line=dict(color=ORANGE, width=2))),
                    "Item information I(θ)", "θ", "I(θ)", 180, legend=False))

theta0 = st.slider("Evaluate at θ =", -4.0, 4.0, 0.0, 0.1)
Pd, dP, d2P = item.derivatives(np.array([theta0]))
rows = pd.DataFrame(dict(category=np.arange(item.n_cat), probability=Pd[0], derivative=dP[0], second_derivative=d2P[0],
                         info_contribution=dP[0] ** 2 / Pd[0]))
table(rows, 6)
st.caption(f"Σ probabilities = {Pd[0].sum():.12f} · E(θ) = {item.expected(np.array([theta0]))[0]:.6f} · "
           f"I(θ) = Σ P′²/P = {item.info(np.array([theta0]))[0]:.6f} · Σ P′ = {dP[0].sum():.1e} (always 0)")

with st.expander("Parameter interpretation and constraints", expanded=False):
    table([dict(parameter="a (discrimination)", models="2PL, 3PL, 4PL, GPCM, GRM", constraint="a > 0",
                meaning="Steepness; information is proportional to (D·a)²"),
           dict(parameter="b (difficulty / location)", models="Rasch–4PL, RSM", constraint="any real",
                meaning="Location on the θ scale"),
           dict(parameter="c (lower asymptote)", models="3PL, 4PL", constraint="0 ≤ c < d",
                meaning="P as θ → −∞ (guessing)"),
           dict(parameter="d (upper asymptote)", models="4PL", constraint="c < d ≤ 1", meaning="P as θ → +∞ (slipping)"),
           dict(parameter="δ_v (step difficulty)", models="PCM, GPCM", constraint="any real; may be unordered",
                meaning="θ where categories v−1 and v are equally likely"),
           dict(parameter="τ_v (RSM threshold)", models="RSM", constraint="shared by all RSM items",
                meaning="Category pattern common to the scale"),
           dict(parameter="b_k (GRM threshold)", models="GRM", constraint="strictly increasing",
                meaning="θ where P(Y ≥ k) = 0.5")])

st.subheader("C · Example D check: three-category GRM item, a = 1, thresholds (−1, 1)")
labels("demo")
g = make_item("G1", "GRM", a=1.0, steps=(-1.0, 1.0))
from scipy.special import expit  # noqa: E402
ps = expit(np.array([0 - (-1.0), 0 - 1.0]))
Pg = g.probs(np.array([0.0]))[0]
table([dict(quantity="P*₁(0) = Ψ(0 − (−1))", value=ps[0]), dict(quantity="P*₂(0) = Ψ(0 − 1)", value=ps[1]),
       dict(quantity="P₀ = 1 − P*₁", value=Pg[0]), dict(quantity="P₁ = P*₁ − P*₂", value=Pg[1]),
       dict(quantity="P₂ = P*₂", value=Pg[2])], 5)
st.caption(f"(P₀, P₁, P₂) = ({Pg[0]:.5f}, {Pg[1]:.5f}, {Pg[2]:.5f}) — the specification's (0.26894, 0.46212, 0.26894).")

st.subheader("D · The current test (mixed formats score on one θ scale)")
d = data()
grid2 = np.linspace(-4, 4, 241)
T = sum(it.expected(grid2) for it in d.items)
I = sum(it.info(grid2) for it in d.items)
left, right = st.columns(2)
with left:
    show(fig_layout(go.Figure(go.Scatter(x=grid2, y=T, line=dict(color=NAVY, width=2), name="T(θ)")),
                    f"Test characteristic curve (max score {sum(it.max_score for it in d.items)})", "θ", "expected summed score", 300,
                    legend=False))
with right:
    fig = go.Figure(go.Scatter(x=grid2, y=I, line=dict(color=ORANGE, width=2), name="I(θ)"))
    show(fig_layout(fig, "Test information I(θ)", "θ", "information", 300, legend=False))
interpret(f"The test **{d.name}** combines {', '.join(d.models)} items. Each item contributes its own category "
          "probabilities to one likelihood in θ; the TCC adds their expected scores and the test information adds their "
          f"information. Information peaks at θ ≈ {grid2[I.argmax()]:.2f} (I = {I.max():.2f}, SE ≈ {1 / np.sqrt(I.max()):.2f}): "
          "abilities there are measured most precisely.")
