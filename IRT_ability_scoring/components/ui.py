"""Shared layout, state and table/plot helpers.

Visual system follows the MMLE-EM Learning Lab reference (mmle-em-layout.png) and the Rasch
Calibration Lab built from it: light blue-grey sidebar headed "EXPERIMENT SETTINGS", navy section
titles, bordered metric cards, numbered phase cards, Equation / Explanation / Notation tabs and
"📘 How to read this" boxes.

State shared by every page (st.session_state):
  datasets  {name: Dataset}   active  name of the dataset in use
  person    selected examinee (row index)
  cfg_*     scoring settings (prior, quadrature, bounds, tolerance, start value)
"""
from __future__ import annotations

import hashlib
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

from scoring import ScoringSettings  # noqa: E402
from simulation import EXAMPLES, simulate  # noqa: E402

NAVY, BLUE, ORANGE, GREEN, GREY, PURPLE, RED = "#17273f", "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#6250d6", "#c8463d"
METHOD_COLORS = dict(MLE=BLUE, EAP=GREEN, MAP=ORANGE, WLE=PURPLE, TCC="#b8860b", EAPsum="#0f8b8d", Truth=GREY)

CSS = """<style>
.block-container{padding-top:2rem;max-width:1500px}
h1,h2,h3{color:#17273f}
div[data-testid='stMetric']{border:1px solid #dce3ef;border-radius:8px;padding:10px 12px;background:#fff}
section[data-testid='stSidebar']{background:#edf1f8}
.phase-row{display:flex;gap:10px;margin:6px 0 14px 0;flex-wrap:wrap}
.phase{flex:1;min-width:110px;border:1px solid #dce3ef;border-radius:8px;padding:8px 10px;background:#fff;color:#17273f;font-size:0.9rem}
.phase b{display:block;color:#2a5db0;font-size:0.8rem}
.phase.active{background:#dfe9fb;border-color:#9fbbea}
.label-ref{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.78rem;margin-right:6px}
.ref{background:#efe9fb;color:#4b3a9e}.edu{background:#e3f0fb;color:#1c5a9c}.demo{background:#fdf0e3;color:#9a4f12}
.pattern{font-family:ui-monospace,Menlo,monospace;background:#fff;border:1px solid #dce3ef;border-radius:6px;padding:2px 8px}
.op{font-size:1.8rem;color:#8a8f98;text-align:center;padding-top:140px}
</style>"""

DEFAULTS = ScoringSettings()
SETTING_KEYS = ("prior_mean", "prior_sd", "quad_rule", "n_nodes", "theta_lim", "bound", "tol", "max_iter", "start")


# ---- state -----------------------------------------------------------------------------------

def default_dataset():
    return simulate(300, seed=2026, n_dich=15, dich_model="2PL", n_poly=5, poly_model="GPCM", n_cat=4)


def ensure_state():
    ss = st.session_state
    if "datasets" not in ss:
        ss.datasets = {}
        d = default_dataset()
        ss.datasets[d.name] = d
        for name, fn in EXAMPLES.items():
            ex = fn()
            ss.datasets[ex.name] = ex
        ss.active = d.name
        ss.person = 0
    for k in SETTING_KEYS:
        ss.setdefault(f"cfg_{k}", getattr(DEFAULTS, k))
    ss.setdefault("cache", {})


def data():
    ensure_state()
    return st.session_state.datasets[st.session_state.active]


def add_dataset(d, activate=True, person=0):
    ensure_state()
    st.session_state.datasets[d.name] = d
    if activate:
        st.session_state.active = d.name
        st.session_state.person = person


def person():
    d = data()
    p = int(st.session_state.get("person", 0))
    return min(max(p, 0), d.n_persons - 1)


def settings():
    ensure_state()
    return ScoringSettings(**{k: st.session_state[f"cfg_{k}"] for k in SETTING_KEYS})


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached(name, fn, *parts):
    """Session cache keyed by dataset identity, settings and extra parts."""
    key = signature([name, data().identity, settings().as_dict(), parts])
    store = st.session_state.cache
    if key not in store:
        if len(store) > 60:
            store.clear()
        store[key] = fn()
    return store[key]


def pattern_text(y):
    return "(" + ", ".join("·" if np.isnan(v) else str(int(v)) for v in y) + ")"


# ---- persistent widgets (values survive page switches) ---------------------------------------

def _sync(key):
    st.session_state[key] = st.session_state["w_" + key]


def keep(widget, label, key, **kw):
    st.session_state["w_" + key] = st.session_state[key]
    return widget(label, key="w_" + key, on_change=_sync, args=(key,), **kw)


# ---- page frame ---------------------------------------------------------------------------------

def page_setup(title, subtitle):
    st.set_page_config(page_title=f"{title} · IRT Ability Scoring Lab", page_icon="🎯", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_state()
    st.title(title)
    st.caption(subtitle)
    sidebar()


def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown("**EXPERIMENT SETTINGS**")
        names = list(ss.datasets)
        ss["w_active"] = ss.active

        def _set_active():
            ss.active = ss["w_active"]
            ss.person = 0
        st.selectbox("Dataset", names, key="w_active", on_change=_set_active)
        d = data()
        st.caption(f"{d.n_persons:,} examinees · {d.n_items} items · models: {', '.join(d.models)}")
        ss["w_person"] = person()

        def _set_person():
            ss.person = ss["w_person"]
        st.selectbox("Selected examinee", range(d.n_persons), key="w_person", on_change=_set_person,
                     format_func=lambda i: f"{d.person_ids[i]} · {pattern_text(d.Y[i])}"[:60])
        y = d.Y[person()]
        st.markdown(f"Responses <span class='pattern'>{pattern_text(y)}</span>", unsafe_allow_html=True)
        st.caption(f"Summed score {int(np.nansum(y))} of {sum(it.max_score for it, v in zip(d.items, y) if not np.isnan(v))}"
                   + (f" · true θ = {d.theta_true[person()]:.3f}" if d.has_truth else ""))
        with st.expander("Scoring settings (all methods)", expanded=False):
            keep(st.number_input, "Prior mean μ", "cfg_prior_mean", step=0.1, format="%.2f")
            keep(st.number_input, "Prior SD σ", "cfg_prior_sd", min_value=0.1, step=0.1, format="%.2f")
            keep(st.selectbox, "Quadrature rule (EAP, EAPsum)", "cfg_quad_rule", options=["gauss-hermite", "rectangular"])
            keep(st.number_input, "Quadrature nodes", "cfg_n_nodes", min_value=3, max_value=201, step=2)
            keep(st.number_input, "Rectangular grid limit ±", "cfg_theta_lim", min_value=2.0, max_value=10.0, step=0.5)
            keep(st.number_input, "Search range ± (MLE, MAP, WLE, TCC)", "cfg_bound", min_value=2.0, max_value=20.0, step=0.5)
            keep(st.number_input, "Starting value θ⁽⁰⁾", "cfg_start", min_value=-5.0, max_value=5.0, step=0.25)
            keep(st.select_slider, "Convergence tolerance |Δθ|", "cfg_tol", options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10],
                 format_func=lambda v: f"{v:.0e}")
            keep(st.number_input, "Maximum iterations", "cfg_max_iter", min_value=5, max_value=500, step=5)
            if st.button("Reset to irtQ defaults"):
                for k in SETTING_KEYS:
                    ss[f"cfg_{k}"] = getattr(DEFAULTS, k)
                st.rerun()
        s = settings()
        st.caption(f"Prior N({s.prior_mean:g}, {s.prior_sd:g}²) · {s.n_nodes} {s.quad_rule} nodes · range ±{s.bound:g}")
        st.markdown("---")
        st.caption("Route: Models → Data → MLE → EAP → MAP → WLE → TCC & EAPsum → Compare → Monte Carlo → Theory")


def interpret(markdown_text, title="How to read this"):
    with st.container(border=True):
        st.markdown(f"📘 **{title}**\n\n" + markdown_text)


def labels(*kinds):
    names = dict(ref=("ref", "Reference result (R)"), edu=("edu", "Educational algorithm (Python)"),
                 demo=("demo", "Numerical teaching demonstration"))
    st.markdown("".join(f"<span class='label-ref {names[k][0]}'>{names[k][1]}</span>" for k in kinds),
                unsafe_allow_html=True)


def phase_cards(phases, active):
    html = "".join(f"<div class='phase{' active' if k == active else ''}'><b>{k + 1}</b>{p}</div>"
                   for k, p in enumerate(phases))
    st.markdown(f"<div class='phase-row'>{html}</div>", unsafe_allow_html=True)


def teaching_tabs(equation, explanation, notation, derivation=None):
    names = ["Equation", "Explanation", "Notation"] + (["Derivation"] if derivation else [])
    tabs = st.tabs(names)
    with tabs[0]:
        for eq in ([equation] if isinstance(equation, str) else equation):
            st.latex(eq)
    with tabs[1]:
        st.markdown(explanation)
    with tabs[2]:
        st.markdown(notation)
    if derivation:
        with tabs[3]:
            st.markdown(derivation)


# ---- step-through controls ----------------------------------------------------------------------

def step_controls(key, n_steps):
    """Initialize / Previous / Next / Run to convergence / Reset. Returns the step shown (None before Initialize).

    Buttons act through on_click callbacks, so the new position is in place before anything is drawn."""
    k = f"{key}_pos"
    ss = st.session_state
    ss.setdefault(k, None)
    if ss[k] is not None and ss[k] > n_steps:
        ss[k] = n_steps

    def go(value):
        ss[k] = value

    pos = ss[k]
    cols = st.columns(5)
    cols[0].button("Initialize", key=f"{key}_init", width="stretch", on_click=go, args=(1,))
    cols[1].button("◀ Previous step", key=f"{key}_prev", disabled=pos is None or pos <= 1, width="stretch",
                   on_click=go, args=((pos or 2) - 1,))
    cols[2].button("Next step ▶", key=f"{key}_next", disabled=pos is not None and pos >= n_steps, width="stretch",
                   on_click=go, args=(1 if pos is None else pos + 1,))
    cols[3].button("Run to convergence ⏭", key=f"{key}_run", width="stretch", on_click=go, args=(n_steps,))
    cols[4].button("Reset", key=f"{key}_reset", width="stretch", on_click=go, args=(None,))
    return ss[k]


# ---- figures and tables -------------------------------------------------------------------------

def fig_layout(fig, title="", x="", y="", height=330, legend=True):
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=NAVY)), height=height, template="plotly_white",
                      margin=dict(l=10, r=10, t=45 if title else 15, b=10), xaxis_title=x, yaxis_title=y,
                      legend=dict(orientation="h", y=-0.25) if legend else None, showlegend=legend,
                      font=dict(size=12), hovermode="x unified")
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key)


SCIENTIFIC = ("max_abs_diff_theta", "max_abs_diff_se", "tolerance_theta", "tolerance_se", "error_EAP", "error_PSD",
              "likelihood", "unnormalised", "prior_weight", "posterior_weight", "theta_x_weight", "sq_dev_x_weight", "U",
              "change", "abs_change", "probability_r")


def table(frame, digits=4, height=None, sci=()):
    frame = pd.DataFrame(frame).copy()
    for c in frame.columns:                      # mixed text/number columns (e.g. "+∞" next to numbers) → text
        if frame[c].dtype == object and frame[c].map(lambda v: isinstance(v, (float, int, np.floating, np.integer))).any():
            frame[c] = frame[c].map(lambda v: _cell(v, digits))
        elif pd.api.types.is_float_dtype(frame[c]) and np.isinf(frame[c]).any():
            frame[c] = frame[c].map(lambda v: _cell(v, digits))
    floats = [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]
    sci = set(sci) | {c for c in floats if c in SCIENTIFIC}
    kwargs = dict(hide_index=True, width="stretch",
                  column_config={c: st.column_config.NumberColumn(format="%.3e" if c in sci else f"%.{digits}f")
                                 for c in floats})
    if height:
        kwargs["height"] = height
    st.dataframe(frame, **kwargs)


def _cell(v, digits):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "—"
    if isinstance(v, (float, np.floating)) and np.isinf(v):
        return "+∞" if v > 0 else "−∞"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def download(frame, name, label="Download CSV", key=None):
    st.download_button(label, pd.DataFrame(frame).to_csv(index=False), name, "text/csv", key=key)


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (float, np.floating)) and np.isinf(v):
        return "+∞" if v > 0 else "−∞"
    return f"{v:.{d}f}"


def tidy(value):
    """±inf → text for display tables."""
    if isinstance(value, (float, np.floating)) and np.isinf(value):
        return "+∞" if value > 0 else "−∞"
    return value


def code_view(functions, notes):
    """Implementation view: the executed source of the given functions plus reading notes."""
    import inspect
    st.markdown(notes)
    for fn in functions:
        with st.expander(f"`{fn.__module__}.{fn.__name__}`", expanded=False):
            st.code(inspect.getsource(fn), language="python")


LAB_TABS = ["1 · Concept", "2 · Mathematics", "3 · Numerical example", "4 · Results & interpretation", "5 · Implementation"]
