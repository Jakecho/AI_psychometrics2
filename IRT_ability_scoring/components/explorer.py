"""Step-through explorers shared by the MLE, MAP, WLE and TCC labs.

Every row shown comes from the recorded history of the actual solver run (scoring.solvers); the
item-level tables are recomputed at the recorded θ⁽ᵗ⁾ with the same functions the solver used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from irt_models import item_table, pattern_terms
from scoring import rows as history_rows
from scoring.likelihood import curves

from .ui import BLUE, GREEN, GREY, NAVY, ORANGE, PURPLE, RED, download, fig_layout, fmt, phase_cards, show, step_controls, table

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def theta_t(t):
    return f"θ⁽{str(t).translate(SUP)}⁾"


PHASES = ["Initialize", "Calculate", "Update", "Inspect", "Repeat", "Converge"]


def display_grid(s, *points, n=401):
    lo, hi = -min(s.bound, 5.0), min(s.bound, 5.0)
    pts = [p for p in points if np.isfinite(p)]
    if pts:
        lo, hi = min(lo, min(pts) - 0.5), max(hi, max(pts) + 0.5)
    return np.linspace(lo, hi, n)


def start_control(key, s):
    return st.number_input("Starting value θ⁽⁰⁾ (change to see a different path)", -float(s.bound), float(s.bound),
                           float(s.start), 0.25, key=f"{key}_start")


def _phase(pos, n, status):
    if pos is None:
        return 0
    if pos == n and status == "converged":
        return 5
    return 2 if pos == 1 else 4


# ---- Newton–Raphson on a log objective (MLE, MAP) ---------------------------------------------------

def newton_section(key, items, y, s, method, fn, prior=None):
    """method 'MLE' or 'MAP'; fn(items, Y, settings, start=, record=) -> ScoreResult."""
    theta0 = start_control(key, s)
    details = st.checkbox("Show calculation details (item by item)", value=True, key=f"{key}_details")
    res = fn(items, y[None, :], s, start=theta0, record=True)
    hist = history_rows(res.history, 0)
    n = len(hist)
    status = str(res.status[0])
    cards = st.empty()
    pos = step_controls(key, n)
    with cards.container():
        phase_cards(PHASES, _phase(pos, n, status))
    st.caption(f"The solver recorded {n} iteration(s) from θ⁽⁰⁾ = {theta0:g}; status: **{status}**. "
               "Initialize shows iteration 1; Next / Previous move through the recorded iterations.")
    obj_name = "ℓ(θ)" if method == "MLE" else "log posterior ℓ(θ) + log g(θ)"
    grid = display_grid(s, *(h["theta"] for h in hist), *(h["theta_new"] for h in hist))
    c = curves(items, y, grid)
    mu, sd = (prior if prior else (0.0, np.inf))
    obj = c["loglik"] - ((grid - mu) ** 2 / (2 * sd ** 2) if method == "MAP" else 0)
    grad = c["score"] - ((grid - mu) / sd ** 2 if method == "MAP" else 0)
    if pos is None:
        fig = go.Figure(go.Scatter(x=grid, y=obj, name=obj_name, line=dict(color=BLUE, width=2)))
        fig.add_vline(x=theta0, line=dict(color=GREY, dash="dot"), annotation_text="θ⁽⁰⁾")
        show(fig_layout(fig, f"{obj_name} for this response pattern", "θ", obj_name, 320))
        st.info("Press **Initialize** to evaluate the first Newton step at the starting value.")
        return res, hist, None
    h = hist[pos - 1]
    th, th_new = h["theta"], h["theta_new"]
    m = st.columns(6)
    m[0].metric("θ⁽ᵗ⁾", fmt(th))
    m[1].metric(obj_name.split(" ")[0] if method == "MLE" else "log posterior", fmt(h["obj"]))
    m[2].metric("score U(θ⁽ᵗ⁾)", fmt(h["grad"]))
    m[3].metric("second derivative", fmt(h["hess"]))
    m[4].metric("step", fmt(h["step"]))
    m[5].metric("θ⁽ᵗ⁺¹⁾", fmt(th_new))
    newton = h["hess"] < 0
    U = r"\ell'" if method == "MLE" else r"U_{MAP}"
    H = r"\ell''" if method == "MLE" else r"(\ell''-1/\sigma^2)"
    if newton:
        st.latex(rf"\theta^{{({pos})}}=\theta^{{({pos - 1})}}-\frac{{{U}}}{{{H}}}={th:.5f}-\frac{{{h['grad']:.5f}}}"
                 rf"{{{h['hess']:.5f}}}={th + h['raw_step']:.5f}")
    else:
        st.latex(rf"{H}\ge 0\Rightarrow\text{{Fisher scoring: }}\theta^{{({pos})}}=\theta^{{({pos - 1})}}+\frac{{{U}}}{{I}}"
                 rf"={th:.5f}+\frac{{{h['grad']:.5f}}}{{{h['curvature']:.5f}}}={th + h['raw_step']:.5f}")
    if abs(h["step"] - h["raw_step"]) > 1e-12:
        st.caption(f"Safeguard applied ({h['step_type']}): the step actually taken was {h['step']:.5f}, "
                   f"so {theta_t(pos)} = {th_new:.5f}.")
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=grid, y=obj, name=obj_name, line=dict(color=BLUE, width=2)))
        quad = h["obj"] + h["grad"] * (grid - th) + 0.5 * (h["hess"] if newton else -h["curvature"]) * (grid - th) ** 2
        keep = quad > obj.min() - 0.25 * (obj.max() - obj.min() + 1)
        fig.add_trace(go.Scatter(x=grid[keep], y=quad[keep], name="2nd-order approximation at θ⁽ᵗ⁾",
                                 line=dict(color=ORANGE, dash="dash")))
        fig.add_trace(go.Scatter(x=[th], y=[h["obj"]], mode="markers", name="θ⁽ᵗ⁾", marker=dict(color=NAVY, size=10)))
        fig.add_vline(x=th_new, line=dict(color=GREEN, dash="dot"), annotation_text="θ⁽ᵗ⁺¹⁾")
        fig.update_yaxes(range=[obj.min() - 0.05 * (obj.max() - obj.min() + 1), obj.max() + 0.15 * (obj.max() - obj.min() + 1)])
        show(fig_layout(fig, "Objective and its quadratic approximation", "θ", obj_name, 340))
    with right:
        fig = go.Figure(go.Scatter(x=grid, y=grad, name="score U(θ)", line=dict(color=PURPLE, width=2)))
        slope = h["hess"] if newton else -h["curvature"]
        tangent = h["grad"] + slope * (grid - th)
        fig.add_trace(go.Scatter(x=grid, y=tangent, name="tangent at θ⁽ᵗ⁾", line=dict(color=ORANGE, dash="dash")))
        fig.add_hline(y=0, line=dict(color=GREY))
        fig.add_trace(go.Scatter(x=[th], y=[h["grad"]], mode="markers", name="U(θ⁽ᵗ⁾)", marker=dict(color=NAVY, size=10)))
        fig.add_trace(go.Scatter(x=[th + h["raw_step"]], y=[0], mode="markers", name="tangent crosses 0",
                                 marker=dict(color=GREEN, size=10, symbol="x")))
        span = np.nanmax(np.abs(grad)) if np.isfinite(grad).any() else 1
        fig.update_yaxes(range=[-1.1 * span, 1.1 * span])
        show(fig_layout(fig, "Score function: Newton follows the tangent to zero", "θ", "U(θ)", 340))
    if details:
        st.markdown(f"**Calculation details at {theta_t(pos - 1)} = {th:.5f}** — one row per item; the column sums are "
                    "the quantities used above.")
        it = pd.DataFrame(item_table(items, y, th))
        table(it, 5, height=min(38 * (len(it) + 1) + 3, 360))
        t = pattern_terms(items, y[None, :], [th])
        sums = dict(log_likelihood=t["loglik"][0], score=t["score"][0], second_derivative=t["hessian"][0],
                    information=t["info"][0])
        if method == "MAP":
            sums.update(prior_score=-(th - mu) / sd ** 2, posterior_score=t["score"][0] - (th - mu) / sd ** 2,
                        posterior_curvature=t["hessian"][0] - 1 / sd ** 2)
        table(pd.DataFrame([sums]), 6)
    iteration_table(key, hist, pos, method)
    return res, hist, pos


def iteration_table(key, hist, pos, method):
    frame = pd.DataFrame(hist).rename(columns=dict(obj="objective", grad="score", hess="second_derivative",
                                                   theta="theta_t", theta_new="theta_next"))
    frame["abs_change"] = frame["step"].abs()
    cols = ["iteration", "theta_t", "objective", "score", "second_derivative", "curvature", "step_type", "raw_step",
            "step", "theta_next", "abs_change"]
    frame = frame[[c for c in cols if c in frame]]
    st.markdown(f"**Iteration history** (rows 1–{pos} of {len(frame)})")
    table(frame.iloc[:pos], 6, height=min(38 * (pos + 1) + 3, 300))
    download(frame, f"{method.lower()}_iterations.csv", "Export iteration table (CSV)", key=f"{key}_export")
    if len(frame) > 1:
        fig = go.Figure(go.Scatter(x=frame["iteration"], y=np.maximum(frame["abs_change"], 1e-16), mode="lines+markers",
                                   line=dict(color=BLUE)))
        fig.add_vline(x=pos, line=dict(color=ORANGE, dash="dot"))
        fig.update_yaxes(type="log")
        show(fig_layout(fig, "Convergence: |Δθ| per iteration (log scale)", "iteration", "|Δθ|", 260, legend=False))


# ---- safeguarded root finding (WLE, TCC) ----------------------------------------------------------

def root_section(key, items, y, s, label, fn, curve_fn, u_label, extra_cols=()):
    """fn(items, Y, settings, start=, record=) -> ScoreResult using scoring.solvers.find_root.
    curve_fn(grid) -> dict(name: array) whose first entry is U(θ)."""
    theta0 = start_control(key, s)
    details = st.checkbox("Show calculation details (item by item)", value=True, key=f"{key}_details")
    res = fn(items, y[None, :], s, start=theta0, record=True)
    hist = history_rows(res.history, 0)
    n = len(hist)
    status = str(res.status[0])
    grid = display_grid(s, *(h["theta"] for h in hist))
    cv = curve_fn(grid)
    names = list(cv)
    if n == 0:
        phase_cards(PHASES, 0)
        fig = go.Figure()
        for k, nm in enumerate(names):
            fig.add_trace(go.Scatter(x=grid, y=cv[nm], name=nm, line=dict(width=2, color=[PURPLE, GREY, ORANGE][k % 3],
                                                                          dash=None if k == 0 else "dot")))
        fig.add_hline(y=0, line=dict(color=GREY))
        show(fig_layout(fig, f"{u_label}: no sign change inside ±{s.bound:g}", "θ", u_label, 320))
        st.warning(f"No iteration was run: {status}. The equation has no root inside the search range, so there is no "
                   "finite solution there.")
        return res, hist, None
    cards = st.empty()
    pos = step_controls(key, n)
    with cards.container():
        phase_cards(PHASES, _phase(pos, n, status))
    st.caption(f"The solver recorded {n} iteration(s) from θ⁽⁰⁾ = {theta0:g}; status: **{status}**. A Newton step "
               "that would leave the current bracket is replaced by bisection.")
    if pos is None:
        fig = go.Figure(go.Scatter(x=grid, y=cv[names[0]], name=names[0], line=dict(color=PURPLE, width=2)))
        fig.add_hline(y=0, line=dict(color=GREY))
        fig.add_vline(x=theta0, line=dict(color=GREY, dash="dot"), annotation_text="θ⁽⁰⁾")
        show(fig_layout(fig, f"{u_label} for this response pattern", "θ", u_label, 320))
        st.info("Press **Initialize** to evaluate the first step at the starting value.")
        return res, hist, None
    h = hist[pos - 1]
    th, th_new = h["theta"], h["theta_new"]
    m = st.columns(6)
    m[0].metric("θ⁽ᵗ⁾", fmt(th))
    m[1].metric("U(θ⁽ᵗ⁾)", fmt(h["U"]))
    m[2].metric("U′(θ⁽ᵗ⁾)", fmt(h["dU"]))
    m[3].metric("bracket width", fmt(h["hi"] - h["lo"]), help="hi − lo of the sign-change bracket")
    m[4].metric("step", fmt(h["step"]))
    m[5].metric("θ⁽ᵗ⁺¹⁾", fmt(th_new))
    st.caption(f"Bracket before this step: [{h['lo']:.5f}, {h['hi']:.5f}] — U changes sign inside it, so the root is there.")
    if h["step_type"] == "Newton":
        st.latex(rf"\theta^{{({pos})}}=\theta^{{({pos - 1})}}-\frac{{U}}{{U'}}={th:.5f}-\frac{{{h['U']:.5f}}}{{{h['dU']:.5f}}}"
                 rf"={th_new:.5f}")
    else:
        st.latex(rf"\text{{bisection: }}\theta^{{({pos})}}=\tfrac12(\text{{lo}}+\text{{hi}})=\tfrac12({h['lo']:.4f}+{h['hi']:.4f})"
                 rf"={th_new:.5f}")
    fig = go.Figure()
    for k, nm in enumerate(names):
        fig.add_trace(go.Scatter(x=grid, y=cv[nm], name=nm, line=dict(width=2 if k == 0 else 1.5,
                                                                      color=[PURPLE, GREY, ORANGE, GREEN][k % 4],
                                                                      dash=None if k == 0 else "dot")))
    fig.add_hline(y=0, line=dict(color=GREY))
    fig.add_vrect(x0=h["lo"], x1=h["hi"], fillcolor="#dfe9fb", opacity=0.45, line_width=0, annotation_text="bracket")
    fig.add_trace(go.Scatter(x=grid, y=h["U"] + h["dU"] * (grid - th), name="tangent at θ⁽ᵗ⁾",
                             line=dict(color=ORANGE, dash="dash")))
    fig.add_trace(go.Scatter(x=[th], y=[h["U"]], mode="markers", name="U(θ⁽ᵗ⁾)", marker=dict(color=NAVY, size=10)))
    fig.add_trace(go.Scatter(x=[th_new], y=[0], mode="markers", name="θ⁽ᵗ⁺¹⁾", marker=dict(color=GREEN, size=11, symbol="x")))
    span = np.nanmax(np.abs(cv[names[0]]))
    fig.update_yaxes(range=[-1.1 * span, 1.1 * span])
    show(fig_layout(fig, f"{u_label} and the step from θ⁽ᵗ⁾", "θ", u_label, 360))
    if details:
        st.markdown(f"**Calculation details at {theta_t(pos - 1)} = {th:.5f}**")
        table(pd.DataFrame(item_table(items, y, th)), 5, height=min(38 * (len(items) + 1) + 3, 360))
        extra = {c: h[c] for c in extra_cols if c in h}
        if extra:
            table(pd.DataFrame([extra]), 6)
    frame = pd.DataFrame(hist).rename(columns=dict(theta="theta_t", theta_new="theta_next"))
    frame["abs_change"] = frame["step"].abs()
    cols = ["iteration", "theta_t", "U", "dU", *extra_cols, "lo", "hi", "step_type", "step", "theta_next", "abs_change"]
    frame = frame[[c for c in cols if c in frame]]
    st.markdown(f"**Iteration history** (rows 1–{pos} of {len(frame)})")
    table(frame.iloc[:pos], 6, height=min(38 * (pos + 1) + 3, 300))
    download(frame, f"{label.lower()}_iterations.csv", "Export iteration table (CSV)", key=f"{key}_export")
    if len(frame) > 1:
        fig = go.Figure(go.Scatter(x=frame["iteration"], y=np.maximum(frame["abs_change"], 1e-16), mode="lines+markers",
                                   line=dict(color=BLUE)))
        fig.add_vline(x=pos, line=dict(color=ORANGE, dash="dot"))
        fig.update_yaxes(type="log")
        show(fig_layout(fig, "Convergence: |Δθ| per iteration (log scale)", "iteration", "|Δθ|", 260, legend=False))
    return res, hist, pos


def reference_check(key, method, items, y, s, packages=("irtQ", "mirt", "TAM")):
    """Button: score this one pattern with the reference packages and show the agreement rows."""
    from r_bridge.compare import compare_package
    if st.button(f"Check this examinee's {method} against R (irtQ / mirt / TAM)", key=f"{key}_ref"):
        rows = []
        with st.spinner("Running Rscript…"):
            for p in packages:
                _, r = compare_package(p, items, y[None, :], s, methods=(method,))
                rows += r
        st.session_state[f"{key}_refrows"] = (rows, s.as_dict(), list(np.nan_to_num(y, nan=-1)))
    saved = st.session_state.get(f"{key}_refrows")
    if saved and saved[1] == s.as_dict() and saved[2] == list(np.nan_to_num(y, nan=-1)):
        frame = pd.DataFrame(saved[0])[["package", "method", "python", "compared", "max_abs_diff_theta",
                                        "max_abs_diff_se", "tolerance_theta", "result", "note"]]
        table(frame, 6)
    elif saved:
        st.caption("Settings or the response pattern changed since the last check; run it again.")
