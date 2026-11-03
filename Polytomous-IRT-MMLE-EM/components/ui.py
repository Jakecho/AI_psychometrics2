"""Shared layout, state and table/plot helpers for the Polytomous IRT MMLE–EM Learning Lab.

Visual system follows the MMLE-EM Learning Lab and its Rasch / Ability-Scoring successors: light blue-grey
sidebar headed "EXPERIMENT SETTINGS", navy titles, bordered metric cards, phase cards and "How to read this"
boxes. Every page shares (st.session_state):

  datasets {name: Dataset} · active (dataset name) · person (selected examinee)
  cfg_* calibration settings → FitConfig · mode ("Guided" | "Exploration")
  fits {"latest": FitResult, saved: {label: FitResult}}
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import walkthrough_dataset  # noqa: E402
from core.em import DESIGN_NAMES, DESIGNS, FitConfig  # noqa: E402
from core.models import MODEL_NAMES, MODELS  # noqa: E402
from core.priors import PRIOR_NAMES, PRIOR_TYPES  # noqa: E402
from core.quadrature import GRID_NAMES, illustrative_distribution, make_grid  # noqa: E402
from core.simulation import example_datasets  # noqa: E402

NAVY, BLUE, ORANGE, GREEN, GREY, PURPLE, RED, TEAL = ("#17273f", "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98",
                                                     "#6250d6", "#c8463d", "#0f8b8d")
SERIES = [BLUE, ORANGE, GREEN, PURPLE, RED, TEAL, "#b8860b", GREY]

CSS = """<style>
.block-container{padding-top:2rem;max-width:1500px}
h1,h2,h3{color:#17273f}
div[data-testid='stMetric']{border:1px solid #dce3ef;border-radius:8px;padding:10px 12px;background:#fff}
section[data-testid='stSidebar']{background:#edf1f8}
.phase-row{display:flex;gap:8px;margin:6px 0 14px 0;flex-wrap:wrap}
.phase{flex:1;min-width:96px;border:1px solid #dce3ef;border-radius:8px;padding:6px 8px;background:#fff;color:#17273f;font-size:0.82rem}
.phase b{display:block;color:#2a5db0;font-size:0.75rem}
.phase.active{background:#dfe9fb;border-color:#9fbbea}
.phase.done{background:#f3f6fb;color:#5b6575}
.label-ref{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.78rem;margin-right:6px}
.ref{background:#efe9fb;color:#4b3a9e}.edu{background:#e3f0fb;color:#1c5a9c}.demo{background:#fdf0e3;color:#9a4f12}
</style>"""

CFG_DEFAULTS = dict(model="GPCM", design="single", prior="fixed_normal", pooled=False, grid="equal", n_nodes=41,
                    bound=6.0, mean=0.0, sd=1.0, normal_update="exact", empirical_update="constrained",
                    fipc_method="MEM", D=1.0, max_cycles=500, tol=1e-4, prior_tol=1e-4, se=False,
                    emp_shape="skewed", emp_custom="")
ROUTE = ["1 Overview", "2 Model Explorer", "3 Prior Explorer", "4 Data Simulation", "5 Quadrature Lab",
         "6 EM Walkthrough", "7 Calibration Designs", "8 Results", "9 Parameter Recovery", "10 Model Fit",
         "11 Drift Analysis", "12 Reference Validation"]


# ---- state --------------------------------------------------------------------------------------

def ensure_state():
    ss = st.session_state
    if "datasets" not in ss:
        ss.datasets = {}
        for d in example_datasets():
            ss.datasets[d.name] = d
        w = walkthrough_dataset()
        ss.datasets[w.name] = w
        ss.active = next(iter(ss.datasets))
        ss.person = 0
    for k, v in CFG_DEFAULTS.items():
        ss.setdefault(f"cfg_{k}", v)
    ss.setdefault("mode", "Guided")
    ss.setdefault("fits", {})
    ss.setdefault("saved", {})


def data():
    ensure_state()
    return st.session_state.datasets[st.session_state.active]


def add_dataset(d, activate=True):
    ensure_state()
    st.session_state.datasets[d.name] = d
    if activate:
        activate_dataset(d.name)


def activate_dataset(name):
    ss = st.session_state
    ss.active = name
    ss.person = 0
    d = ss.datasets[name]
    if d.truth and "spec" in d.truth:            # match the calibration settings to how the data were built
        ss.cfg_model = d.truth["spec"]["model"]
        ss.cfg_design = d.truth["spec"]["design"]
    elif d.G > 1:
        ss.cfg_design = "mgfipc" if d.anchors else "concurrent"
    else:
        ss.cfg_design = "fipc" if d.anchors else "single"


def person():
    d = data()
    return int(min(max(int(st.session_state.get("person", 0)), 0), d.N - 1))


def mode():
    ensure_state()
    return st.session_state.mode


def grid_nodes():
    ss = st.session_state
    return make_grid(ss.cfg_grid if ss.cfg_grid != "custom" else "equal", int(ss.cfg_n_nodes), float(ss.cfg_bound)).nodes


def empirical_weights(nodes=None):
    """Fixed (or starting) empirical masses from the sidebar choice."""
    ss = st.session_state
    nodes = grid_nodes() if nodes is None else nodes
    if ss.cfg_emp_shape == "custom":
        try:
            w = np.asarray(json.loads(ss.cfg_emp_custom or "[]"), float)
        except (ValueError, TypeError):
            raise ValueError("Custom weights must be a JSON list of numbers, one per node.")
        if len(w) != len(nodes):
            raise ValueError(f"Custom weights need {len(nodes)} values (one per node); got {len(w)}.")
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Custom weights must be non-negative with positive sum.")
        return w / w.sum()
    return illustrative_distribution(nodes, ss.cfg_emp_shape, float(ss.cfg_mean), float(ss.cfg_sd))


def config(**override):
    ensure_state()
    ss = st.session_state
    kw = {k: ss[f"cfg_{k}"] for k in CFG_DEFAULTS if k not in ("emp_shape", "emp_custom")}
    kw.update(override)
    cfg = FitConfig(**kw)
    if cfg.prior == "fixed_empirical":
        cfg.fixed_weights = list(empirical_weights())
    if cfg.prior == "estimated_empirical" and ss.cfg_emp_shape == "custom":
        cfg.initial_weights = list(empirical_weights())
    return cfg


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def fit_key(d, cfg):
    return signature([d.identity, cfg.as_dict()])


def latest_fit():
    """The most recent calibration for the active dataset (or None)."""
    f = st.session_state.fits.get("latest")
    if f is not None and f.data_id == data().identity:
        return f
    return None


# ---- persistent widgets -----------------------------------------------------------------------------

def _sync(key):
    st.session_state[key] = st.session_state["w_" + key]


def keep(widget, label, key, **kw):
    st.session_state["w_" + key] = st.session_state[key]
    return widget(label, key="w_" + key, on_change=_sync, args=(key,), **kw)


# ---- page frame ---------------------------------------------------------------------------------

def page_setup(title, subtitle):
    st.set_page_config(page_title=f"{title} · Polytomous MMLE–EM Lab", page_icon="📐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_state()
    st.title(title)
    st.caption(subtitle)
    sidebar()


def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown("**EXPERIMENT SETTINGS**")
        keep(st.radio, "Mode", "mode", options=["Guided", "Exploration"], horizontal=True,
             help="Guided: structured steps, explanations and knowledge checks. Exploration: every control open.")
        names = list(ss.datasets)
        ss["w_active"] = ss.active
        st.selectbox("Dataset", names, key="w_active", on_change=lambda: activate_dataset(ss["w_active"]))
        d = data()
        st.caption(f"{d.N:,} examinees · {d.J} items · {d.G} group(s) · {len(d.anchors)} bank anchors")
        keep(st.selectbox, "Model", "cfg_model", options=list(MODELS), format_func=lambda m: f"{m} · {MODEL_NAMES[m]}")
        keep(st.selectbox, "Calibration design", "cfg_design", options=list(DESIGNS), format_func=DESIGN_NAMES.get)
        keep(st.selectbox, "Ability prior", "cfg_prior", options=list(PRIOR_TYPES), format_func=PRIOR_NAMES.get)
        if ss.cfg_design in ("concurrent", "mgfipc"):
            keep(st.checkbox, "One common (pooled) prior across groups", "cfg_pooled")
        if ss.cfg_design in ("fipc", "mgfipc"):
            keep(st.radio, "FIPC method", "cfg_fipc_method", options=["MEM", "OEM"], horizontal=True,
                 format_func=lambda m: "MWU–MEM" if m == "MEM" else "NWU–OEM")
        with st.expander("Quadrature & prior settings", expanded=ss.mode == "Exploration"):
            keep(st.selectbox, "Grid", "cfg_grid", options=["equal", "gauss_hermite"], format_func=GRID_NAMES.get,
                 help="Estimated and empirical priors need a fixed grid (equally spaced).")
            keep(st.number_input, "Nodes Q", "cfg_n_nodes", min_value=3, max_value=161, step=2)
            keep(st.number_input, "Grid bound ±", "cfg_bound", min_value=2.0, max_value=10.0, step=0.5)
            keep(st.number_input, "Reference / fixed mean μ", "cfg_mean", step=0.1, format="%.2f")
            keep(st.number_input, "Reference / fixed SD σ", "cfg_sd", min_value=0.1, step=0.1, format="%.2f")
            keep(st.selectbox, "Empirical masses (fixed or start)", "cfg_emp_shape",
                 options=["normal", "skewed", "bimodal", "uniform", "custom"])
            if ss.cfg_emp_shape == "custom":
                keep(st.text_area, "Custom masses (JSON list, one per node)", "cfg_emp_custom")
            keep(st.radio, "Normal update", "cfg_normal_update", options=["exact", "moments"], horizontal=True,
                 format_func=lambda v: "Exact M-step" if v == "exact" else "Posterior moments")
            keep(st.radio, "Constrained histogram", "cfg_empirical_update", options=["constrained", "woods"],
                 horizontal=True, format_func=lambda v: "Exact constrained" if v == "constrained" else "Woods (irtQ)")
        with st.expander("Convergence", expanded=False):
            keep(st.number_input, "Maximum cycles", "cfg_max_cycles", min_value=1, max_value=5000, step=50)
            keep(st.select_slider, "Item / group tolerance", "cfg_tol", options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                 format_func=lambda v: f"{v:.0e}")
            keep(st.select_slider, "Prior-mass tolerance", "cfg_prior_tol", options=[1e-3, 1e-4, 1e-5, 1e-6],
                 format_func=lambda v: f"{v:.0e}")
            keep(st.radio, "Logistic scale D", "cfg_D", options=[1.0, 1.702], horizontal=True)
            keep(st.checkbox, "Compute standard errors (slower)", "cfg_se")
            if st.button("Reset settings"):
                for k, v in CFG_DEFAULTS.items():
                    ss[f"cfg_{k}"] = v
                st.rerun()
        st.markdown("---")
        st.caption("Route: " + " → ".join(r.split(" ", 1)[1] for r in ROUTE))


def interpret(markdown_text, title="How to read this"):
    with st.container(border=True):
        st.markdown(f"📘 **{title}**\n\n" + markdown_text)


def labels(*kinds):
    names = dict(ref=("ref", "Reference result (R)"), edu=("edu", "Lab engine (Python)"),
                 demo=("demo", "Numerical teaching demonstration"))
    st.markdown("".join(f"<span class='label-ref {names[k][0]}'>{names[k][1]}</span>" for k in kinds),
                unsafe_allow_html=True)


def phase_cards(phases, active):
    html = "".join(f"<div class='phase{' active' if k == active else (' done' if k < active else '')}'>"
                   f"<b>{k + 1}</b>{p}</div>" for k, p in enumerate(phases))
    st.markdown(f"<div class='phase-row'>{html}</div>", unsafe_allow_html=True)


def four_views(concept, mathematics, example=None, functions=(), notes=""):
    """Concept / Mathematics / Numerical example / Implementation tabs used for every major concept."""
    t = st.tabs(["Concept", "Mathematics", "Numerical example", "Implementation"])
    with t[0]:
        st.markdown(concept)
    with t[1]:
        for part in mathematics:
            if part.startswith("$$"):
                st.latex(part.strip("$"))
            else:
                st.markdown(part)
    with t[2]:
        if example is None:
            st.caption("See the interactive panels on this page.")
        else:
            example()
    with t[3]:
        if notes:
            st.markdown(notes)
        for fn in functions:
            with st.expander(f"`{fn.__module__}.{fn.__name__}`"):
                st.code(inspect.getsource(fn), language="python")


def guided(steps, question=None, options=None, answer=None, explanation="", key="quiz"):
    """Guided-mode box: numbered learning steps and one knowledge check with feedback."""
    if mode() != "Guided":
        return
    with st.container(border=True):
        st.markdown("🧭 **Guided steps**\n\n" + "\n".join(f"{k + 1}. {s}" for k, s in enumerate(steps)))
        if question:
            choice = st.radio(f"Knowledge check: {question}", options, index=None, key=f"q_{key}")
            if choice is not None:
                if choice == answer:
                    st.success("Correct. " + explanation)
                else:
                    st.error("Not quite. " + explanation)


# ---- figures and tables -------------------------------------------------------------------------

def fig_layout(fig, title="", x="", y="", height=340, legend=True):
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=NAVY)), height=height, template="plotly_white",
                      margin=dict(l=10, r=10, t=45 if title else 15, b=10), xaxis_title=x, yaxis_title=y,
                      legend=dict(orientation="h", y=-0.25) if legend else None, showlegend=legend,
                      font=dict(size=12), hovermode="x unified")
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key)


def lines(x, series, title="", xl="θ", yl="", height=340, dash=None, key=None):
    fig = go.Figure()
    for k, (name, y) in enumerate(series.items()):
        fig.add_scatter(x=x, y=y, name=name, mode="lines", line=dict(color=SERIES[k % len(SERIES)],
                                                                     dash=(dash or {}).get(name)))
    show(fig_layout(fig, title, xl, yl, height), key)


def bars(x, series, title="", xl="θ", yl="mass", height=340, key=None, mode="group"):
    fig = go.Figure()
    for k, (name, y) in enumerate(series.items()):
        fig.add_bar(x=x, y=y, name=name, marker_color=SERIES[k % len(SERIES)])
    fig.update_layout(barmode=mode)
    show(fig_layout(fig, title, xl, yl, height), key)


def table(frame, digits=4, height=None, sci=()):
    frame = pd.DataFrame(frame).copy()
    floats = [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]
    kwargs = dict(hide_index=True, width="stretch",
                  column_config={c: st.column_config.NumberColumn(format="%.3e" if c in sci else f"%.{digits}f")
                                 for c in floats})
    if height:
        kwargs["height"] = height
    st.dataframe(frame, **kwargs)


def full_precision(frame, name, key):
    """Copyable full-precision values (CSV download + text area)."""
    frame = pd.DataFrame(frame)
    c1, c2 = st.columns([1, 3])
    c1.download_button("Download CSV", frame.to_csv(index=False, float_format="%.17g"), name, "text/csv", key=f"dl_{key}")
    with c2.expander("Full-precision values (copy)"):
        st.code(frame.to_csv(index=False, float_format="%.17g"), language="text")


def download(frame, name, label="Download CSV", key=None):
    st.download_button(label, pd.DataFrame(frame).to_csv(index=False), name, "text/csv", key=key)


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{d}f}"


def show_plan(pl, names):
    """Identification summary for a design check."""
    for e in pl["errors"]:
        st.error(e)
    for w in pl["warnings"]:
        st.warning(w)
    for n in pl["notes"]:
        st.info(n)
    if pl["units"]:
        table(pd.DataFrame([dict(distribution=", ".join(names[g] for g in u.groups),
                                 updated="yes" if u.estimated else "no",
                                 mean="free" if u.estimated and u.fixed_mean is None else
                                 (f"fixed {u.fixed_mean:g}" if u.fixed_mean is not None else "not updated"),
                                 SD="free" if u.estimated and u.fixed_sd is None else
                                 (f"fixed {u.fixed_sd:g}" if u.fixed_sd is not None else "not updated"),
                                 role=u.role) for u in pl["units"]]))
