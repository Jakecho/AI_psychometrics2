import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, GREY, ORANGE, current_reference, current_result, data, download, fig_layout, fmt,
                           interpret, labels, lines, page_setup, phase_cards, show, stale_result, stepper,
                           store_reference, store_result, table, teaching_tabs)
from r_bridge.bridge import run_reference
from rasch.cmle import cmle, demonstration, log_esf
from rasch.metrics import align_location, recovery
from rasch.model import observed_values
from rasch.validation import compare

page_setup("4 · CMLE Lab", "Conditional maximum likelihood: condition on raw scores so that ability drops out.")
d = data()

st.subheader("A · Overview")
teaching_tabs([r"R_i=\sum_{j\in O_i}X_{ij}",
               r"P(\mathbf X_i=\mathbf x_i\mid R_i=r_i,\mathbf b)=\frac{\exp\!\left(-\sum_{j\in O_i}x_{ij}b_j\right)}"
               r"{\gamma_{r_i}(\mathbf b;O_i)},\qquad \gamma_r=\sum_{\mathbf x:\,\sum x=r}\ \prod_{j}\varepsilon_j^{x_j},\ \ "
               r"\varepsilon_j=e^{-b_j}",
               r"\gamma_r^{(k)}=\gamma_r^{(k-1)}+\varepsilon_k\,\gamma_{r-1}^{(k-1)}\qquad(\text{coefficients of }"
               r"\textstyle\prod_j(1+t\,\varepsilon_j))"],
              "Under the Rasch model the raw score is a **sufficient statistic** for θ: every response pattern with "
              "the same raw score has the same θ-dependence, so in the conditional probability of the pattern given "
              "its score, θ cancels. Maximising the product of these conditional probabilities estimates the "
              "difficulties **without estimating or assuming anything about abilities** — which makes CMLE "
              "consistent for items. Abilities are scored afterwards (page 6).",
              "- $\\gamma_r$: elementary symmetric function of order *r* over the person's observed items\n"
              "- $\\varepsilon_j=\\exp(-b_j)$: item easiness\n- identification: $\\sum_j b_j=0$ (eRm sum0)",
              "Python: `rasch/cmle.py` — ESFs by a log-space dynamic programme (no 2^J enumeration); missing items "
              "get ε = 0 for that person, i.e. conditioning on the observed subset. Gradient: "
              "$\\partial \\ell/\\partial b_j=\\sum_i[P(X_{ij}=1\\mid r_i)-x_{ij}]$ with "
              "$P(X_{ij}=1\\mid r)=\\varepsilon_j\\gamma^{(-j)}_{r-1}/\\gamma_r$. Reference: `eRm::RM()`.")

st.subheader("B · Estimation controls")
c1, c2, _ = st.columns(3)
tol = c1.number_input("Gradient tolerance", 1e-10, 1e-3, 1e-7, format="%.0e")
max_iter = c2.number_input("Maximum optimizer iterations", 10, 5000, 500)
settings = dict(tol=tol, max_iter=int(max_iter))
b1, b2 = st.columns(2)
if b1.button("Run Python CMLE", type="primary"):
    try:
        with st.spinner("Maximising the conditional likelihood…"):
            store_result("CMLE", cmle(d.X, tol=tol, max_iter=int(max_iter)), settings)
    except ValueError as e:
        st.error(str(e))
if b2.button("Run eRm::RM reference"):
    with st.spinner("Running eRm in R…"):
        store_reference("CMLE", run_reference("CMLE", d.X, {}, d.item_ids), {})
res = current_result("CMLE", settings)
ref = current_reference("CMLE", {})
if stale_result("CMLE", settings):
    st.warning("Settings changed since the last run. Press **Run Python CMLE** to update.")

st.subheader("C · Small example: why ability disappears")
labels("demo")
k = st.slider("Items in the example (first k items of the dataset)", 4, min(6, d.n_items), min(5, d.n_items))
source_b = res.b[:k] if res is not None else np.zeros(k)
st.caption("Difficulties used: " + ("current CMLE estimates" if res is not None else "starting values (all 0) — run CMLE to use estimates")
           + " · " + ", ".join(f"{d.item_ids[j]} = {source_b[j]:+.3f}" for j in range(k)))
r = st.slider("Raw score r", 0, k, min(2, k))
demo = demonstration(source_b, r)
left, right = st.columns([3, 2])
with left:
    rows = pd.DataFrame(demo["rows"])
    table(rows, 5, height=260)
    st.caption(f"Sum of conditional probabilities = {rows.conditional_probability.sum():.6f}")
with right:
    st.latex(r"\gamma_{%d}=\sum\text{numerators}=%.5f" % (r, demo["gamma"][r]))
    poly = pd.DataFrame(dict(r=range(k + 1), ESF_recursion=demo["gamma"], polynomial_coefficient=demo["polynomial"]))
    table(poly, 5)
    st.caption("ESFs from the recursion equal the coefficients of ∏(1 + t·εⱼ).")
theta_demo = st.slider("Try any ability θ for this pattern set", -3.0, 3.0, 0.0, 0.5)
eps = demo["eps"]
joint = []
for row in demo["rows"]:
    x = np.array([int(c) for c in row["pattern"]])
    p = 1 / (1 + np.exp(-(theta_demo - source_b)))
    joint.append(float(np.prod(np.where(x == 1, p, 1 - p))))
joint = np.array(joint)
st.markdown(f"Unconditional pattern probabilities at θ = {theta_demo:+.1f} sum to {joint.sum():.5f} (= P(R = {r})); "
            f"divided by that sum they give **{np.round(joint / joint.sum(), 5).tolist()}** — identical to the "
            "conditional probabilities above for every θ.")
interpret("The numerator depends only on which items were answered correctly; the denominator γ_r sums the same "
          "products over **all** patterns with that raw score. Because θ multiplies every pattern with score r by "
          "the same factor $e^{r\\theta}$, it cancels from the ratio — move the θ slider and the conditional "
          "probabilities do not change.")

if res is None:
    st.info("Press **Run Python CMLE** to calibrate the current dataset.")
    st.stop()

st.subheader("D · Iterations and a person's conditional likelihood")
labels("edu", "demo")
phase_cards(["Start: logits of p-values, sum 0", "ESFs γ_r for each pattern", "Conditional score per item",
             "L-BFGS-B update (sum b = 0)", "Check gradient"], 3)
H = res.history
its = np.array([h["iteration"] for h in H])
left, right = st.columns(2)
with left:
    show(lines(its, {"Conditional log-likelihood": [h["cond_loglik"] for h in H]}, "Conditional log-likelihood",
               "Optimizer iteration", "log L_c", colors={"Conditional log-likelihood": ORANGE}))
with right:
    fig = lines(its, {"max |score|": [h["grad_norm"] for h in H]}, "Max |conditional score| (log scale)",
                "Optimizer iteration", "|gradient|", colors={"max |score|": ORANGE})
    show(fig.update_yaxes(type="log"))
sel = st.multiselect("Items to trace", d.item_ids, d.item_ids[:5])
show(lines(its, {i: [h["b"][d.item_ids.index(i)] for h in H] for i in sel}, "Item-difficulty iterates (actual L-BFGS-B iterates)",
           "Optimizer iteration", "b"))
it = stepper("cmle_it", len(H), "Iterate", default=len(H))
b_it = H[it - 1]["b"]
person = st.selectbox("Person", range(d.n_persons), format_func=lambda i: d.person_ids[i])
x, mask = observed_values(d.X)
obs = mask[person]
r_i = int(x[person, obs].sum())
lg = log_esf(-b_it[obs])
numer_log = -float(x[person, obs] @ b_it[obs])
c = st.columns(4)
c[0].metric("Raw score r", f"{r_i} / {int(obs.sum())}")
c[1].metric("log numerator −Σ x b", f"{numer_log:.4f}")
c[2].metric("log γ_r", f"{lg[r_i]:.4f}")
c[3].metric("Conditional log-prob", "0 (extreme score)" if r_i in (0, int(obs.sum())) else f"{numer_log - lg[r_i]:.4f}")
st.caption(f"Computed with the iterate-{it - 1} difficulties over this person's {int(obs.sum())} observed items. "
           "Extreme scores have conditional probability 1, so they add nothing to the conditional likelihood.")

st.subheader("E · Results")
labels("edu", "ref")
st.markdown(f"**Status:** {res.status} · {res.iterations} optimizer iterations · {res.n_informative} informative "
            f"persons · {res.n_extreme} with extreme scores excluded · {res.seconds:.2f} s")
items = pd.DataFrame(dict(item=d.item_ids, b_python=res.b, se_python=res.se_b))
if ref and ref["status"] == "Executed":
    items["b_eRm"], items["se_eRm"] = ref["b"], ref["se_b"]
    items["difference"] = items.b_python - items.b_eRm
if d.has_truth:
    items["true_b"] = d.true_b
table(items, height=300)
download(items, f"cmle_items_{d.identity}.csv")
if ref:
    if ref["status"] == "Executed":
        st.markdown(f"**Reference comparison** · eRm {ref['version']}")
        table(compare("CMLE", dict(b=res.b, se_b=res.se_b, cond_loglik=res.cond_loglik), ref, d.n_persons))
    else:
        st.warning(f"eRm reference {ref['status']}: {ref.get('reason', '')}")

st.subheader("F · Diagnostics and interpretation")
if d.has_truth:
    aligned, shift = align_location(res.b, d.true_b)
    m = recovery(aligned, d.true_b)
    c = st.columns(5)
    c[0].metric("Bias", fmt(m["bias"]))
    c[1].metric("RMSE", fmt(m["rmse"]))
    c[2].metric("MAD", fmt(m["mad"]))
    c[3].metric("Correlation", fmt(m["corr"], 3))
    c[4].metric("Location shift applied", f"{shift:+.3f}")
    fig = go.Figure(go.Scatter(x=d.true_b, y=aligned, mode="markers", marker=dict(color=ORANGE, size=8), text=d.item_ids))
    lim = [min(d.true_b.min(), aligned.min()) - .2, max(d.true_b.max(), aligned.max()) + .2]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color=GREY)))
    show(fig_layout(fig, "True vs estimated difficulty (location-aligned)", "True b", "Estimated b", 340, False))
interpret("- CMLE estimates **items only**; abilities are not parameters of the conditional likelihood.\n"
          "- Standard errors come from the inverse of the observed conditional information (here a finite-difference "
          "Hessian of the exact gradient), propagated through the sum-zero constraint.\n"
          "- Extreme raw scores are uninformative and excluded; with missing data each person is conditioned on "
          "exactly the items they answered.\n- CMLE requires the Rasch model (equal discriminations): it is the "
          "sufficiency of the raw score that makes conditioning possible.")
