import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREEN, NAVY, ORANGE, fig_layout, interpret, lines, page_setup, show, table,
                           teaching_tabs)
from rasch.model import information, probability

page_setup("1 · Rasch Model Explorer", "How ability θ and difficulty b determine the probability of a correct answer.")

with st.sidebar:
    st.markdown("**MODEL CONTROLS**")
    theta = st.slider("Person ability θ", -4.0, 4.0, 0.5, 0.1)
    b = st.slider("Item difficulty b", -4.0, 4.0, 0.0, 0.1)
    lo, hi = st.slider("Ability range shown", -6.0, 6.0, (-4.0, 4.0), 0.5)
    n_items = st.slider("Number of example items", 2, 9, 5)
    spread = st.slider("Item difficulty spread (± around b)", 0.0, 3.0, 1.5, 0.1)

p = float(probability([theta], [b])[0, 0])
st.subheader("1 · The model")
teaching_tabs(r"P(X_{ij}=1\mid\theta_i,b_j)=\frac{\exp(\theta_i-b_j)}{1+\exp(\theta_i-b_j)}",
              "The probability depends only on the **difference θ − b**. When ability equals difficulty the "
              "probability is exactly 0.5; each logit of advantage multiplies the odds of success by e ≈ 2.72. "
              "Every item has the same discrimination (slope fixed at 1), which is what makes the raw score "
              "sufficient for θ.",
              "- $\\theta_i$: ability of person *i* (logits)\n- $b_j$: difficulty of item *j* (logits)\n"
              "- $P_{ij}$: probability of a correct response\n- $I_j(\\theta)=P_j(1-P_j)$: item information")
c1, c2, c3, c4 = st.columns(4)
c1.metric("θ − b", f"{theta - b:+.2f}")
c2.metric("P(correct)", f"{p:.4f}")
c3.metric("Odds P/(1−P)", f"{p / (1 - p):.3f}")
c4.metric("Information P(1−P)", f"{p * (1 - p):.4f}")

x = np.linspace(lo, hi, 301)
left, right = st.columns(2)
with left:
    fig = lines(x, {"ICC": probability(x, [b])[:, 0]}, f"Item characteristic curve (b = {b:+.1f})", "Ability θ",
                "P(correct)", colors={"ICC": BLUE})
    fig.add_trace(go.Scatter(x=[theta], y=[p], mode="markers", marker=dict(size=12, color=ORANGE), name="Selected θ"))
    fig.add_vline(x=b, line_dash="dot", line_color="#999")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999")
    show(fig)
with right:
    d = np.linspace(-6, 6, 301)
    fig = lines(d, {"P(θ − b)": 1 / (1 + np.exp(-d))}, "Probability is a function of θ − b only", "θ − b (logits)",
                "P(correct)", colors={"P(θ − b)": NAVY})
    fig.add_trace(go.Scatter(x=[theta - b], y=[p], mode="markers", marker=dict(size=12, color=ORANGE), name="Selected"))
    show(fig)

st.subheader("2 · Several items and Fisher information")
bs = b + np.linspace(-spread, spread, n_items)
left, right = st.columns(2)
with left:
    show(lines(x, {f"b = {v:+.2f}": probability(x, [v])[:, 0] for v in bs}, "Item characteristic curves",
               "Ability θ", "P(correct)"))
with right:
    info = information(x, bs)
    series = {f"b = {v:+.2f}": info[:, k] for k, v in enumerate(bs)}
    series["Test information"] = info.sum(axis=1)
    show(lines(x, series, "Information  I_j(θ) = P_j(1 − P_j)", "Ability θ", "Information",
               dash={"Test information": "dash"}, colors={"Test information": NAVY}))
interpret("Each item is most informative (0.25) where θ = b. Test information is the sum over items, and the "
          "standard error of an ability estimate is about 1/√(test information): abilities far from the item "
          "difficulties are measured less precisely.")

st.subheader("3 · Why the scale needs an anchor (identification)")
shift = st.slider("Add the same constant c to every θ and b", -3.0, 3.0, 1.0, 0.25)
thetas = np.array([-1.0, 0.0, theta])
rows = []
for t in thetas:
    for v in bs[:3]:
        rows.append(dict(theta=t, b=v, P=float(probability([t], [v])[0, 0]), theta_plus_c=t + shift, b_plus_c=v + shift,
                         P_shifted=float(probability([t + shift], [v + shift])[0, 0])))
table(rows)
st.latex(r"P(\theta_i,b_j)=P(\theta_i+c,\,b_j+c)\quad\text{for every }c")
interpret("Adding the same constant to all abilities and difficulties leaves every probability unchanged "
          f"(here c = {shift:+.2f}), so the data cannot tell where zero is. Each estimator fixes it differently: "
          "JMLE (TAM default) sets the mean person ability to 0, CMLE sets the sum of difficulties to 0, and "
          "MML-EM fixes the mean of the ability distribution at 0.")

st.subheader("4 · Numerical example")
st.latex(rf"P=\frac{{\exp({theta:.2f}-({b:.2f}))}}{{1+\exp({theta:.2f}-({b:.2f}))}}"
         rf"=\frac{{{np.exp(theta - b):.4f}}}{{{1 + np.exp(theta - b):.4f}}}={p:.4f}")
