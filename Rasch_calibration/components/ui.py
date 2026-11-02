"""Shared layout, state and chart helpers.

Visual system follows the MMLE-EM Learning Lab reference (mmle-em-layout.png):
light blue-grey sidebar with an "EXPERIMENT SETTINGS"-style header, navy numbered section
titles, bordered metric cards, numbered phase cards, and Equation / Explanation / Notation tabs.
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

from rasch.simulation import simulate  # noqa: E402

NAVY, BLUE, ORANGE, GREEN, GREY, PURPLE = "#17273f", "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#6250d6"
METHOD_COLORS = dict(JMLE=BLUE, JMLE_raw=PURPLE, CMLE=ORANGE, MML=GREEN, Truth=GREY, TAM=PURPLE, eRm=PURPLE)

CSS = """<style>
.block-container{padding-top:2rem;max-width:1500px}
h1,h2,h3{color:#17273f}
div[data-testid='stMetric']{border:1px solid #dce3ef;border-radius:8px;padding:10px 12px;background:#fff}
section[data-testid='stSidebar']{background:#edf1f8}
.phase-row{display:flex;gap:10px;margin:6px 0 14px 0;flex-wrap:wrap}
.phase{flex:1;min-width:120px;border:1px solid #dce3ef;border-radius:8px;padding:8px 10px;background:#fff;color:#17273f;font-size:0.9rem}
.phase b{display:block;color:#2a5db0;font-size:0.8rem}
.phase.active{background:#dfe9fb;border-color:#9fbbea}
.label-ref{display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.78rem;margin-right:6px}
.ref{background:#efe9fb;color:#4b3a9e}.edu{background:#e3f0fb;color:#1c5a9c}.demo{background:#fdf0e3;color:#9a4f12}
</style>"""


def page_setup(title, subtitle):
    st.set_page_config(page_title=f"{title} · Rasch Calibration Lab", page_icon="📐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_dataset()
    st.title(title)
    st.caption(subtitle)
    sidebar_status()


def ensure_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = simulate()
        st.session_state.results = {}
        st.session_state.references = {}


def data():
    ensure_dataset()
    return st.session_state.dataset


def sidebar_status():
    d = data()
    with st.sidebar:
        st.markdown("**EXPERIMENT SETTINGS**")
        st.caption("Shared dataset (pages 1–8; page 9 builds its own)")
        st.markdown(f"**{d.source}** · {d.n_persons:,} persons · {d.n_items} items")
        miss = 1 - d.mask.mean()
        st.caption(f"ID {d.identity} · missing {miss:.1%}" + (" · true parameters known" if d.has_truth else " · no known truth"))
        done = [m for m in ("JMLE", "CMLE", "MML") if current_result(m) is not None]
        st.caption("Calibrated: " + (", ".join(done) if done else "none yet"))
        st.markdown("---")
        st.caption("Suggested route: Explore model → Simulate data → Calibrate items (JMLE, CMLE, MML-EM) → "
                   "Score persons → Compare methods → Link forms.")


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def store_result(method, result, settings):
    st.session_state.results[method] = dict(result=result, data_id=data().identity, sig=signature(settings),
                                            settings=settings)


def current_result(method, settings=None):
    """Result only if it was computed on the current dataset (and, if given, these settings)."""
    entry = st.session_state.get("results", {}).get(method)
    if not entry or entry["data_id"] != data().identity:
        return None
    if settings is not None and entry["sig"] != signature(settings):
        return None
    return entry["result"]


def stale_result(method, settings):
    entry = st.session_state.get("results", {}).get(method)
    return entry is not None and entry["data_id"] == data().identity and entry["sig"] != signature(settings)


def store_reference(method, ref, settings):
    st.session_state.references[method] = dict(ref=ref, data_id=data().identity, sig=signature(settings))


def current_reference(method, settings):
    entry = st.session_state.get("references", {}).get(method)
    if not entry or entry["data_id"] != data().identity or entry["sig"] != signature(settings):
        return None
    return entry["ref"]


def interpret(markdown_text, title="How to read this"):
    with st.container(border=True):
        st.markdown(f"📘 **{title}**\n\n" + markdown_text)


def watch_box(changed, why, watch):
    st.info(f"**What changed:** {changed}  \n**Why:** {why}  \n**What to watch:** {watch}")


def labels(*kinds):
    names = dict(ref=("ref", "Reference result (R)"), edu=("edu", "Educational algorithm (Python)"),
                 demo=("demo", "Intermediate demonstration"))
    st.markdown("".join(f"<span class='label-ref {names[k][0]}'>{names[k][1]}</span>" for k in kinds),
                unsafe_allow_html=True)


def phase_cards(phases, active):
    html = "".join(f"<div class='phase{' active' if k == active else ''}'><b>{k + 1}</b>{p}</div>"
                   for k, p in enumerate(phases))
    st.markdown(f"<div class='phase-row'>{html}</div>", unsafe_allow_html=True)


def teaching_tabs(equation, explanation, notation, implementation=None):
    names = ["Equation", "Explanation", "Notation"] + (["Implementation"] if implementation else [])
    tabs = st.tabs(names)
    with tabs[0]:
        for eq in ([equation] if isinstance(equation, str) else equation):
            st.latex(eq)
    with tabs[1]:
        st.markdown(explanation)
    with tabs[2]:
        st.markdown(notation)
    if implementation:
        with tabs[3]:
            st.markdown(implementation)


def stepper(key, n, label="Iteration", default=None):
    """Slider plus Previous / Next buttons over 1..n, state kept in session."""
    if n < 1:
        return None
    k = f"{key}_value"
    if k not in st.session_state or st.session_state[k] > n:
        st.session_state[k] = default or 1
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("◀ Previous", key=f"{key}_prev", disabled=st.session_state[k] <= 1):
        st.session_state[k] -= 1
    if c2.button("Next ▶", key=f"{key}_next", disabled=st.session_state[k] >= n):
        st.session_state[k] += 1
    if n > 1:
        c3.slider(label, 1, n, key=k)
    else:
        c3.caption(f"{label} 1 of 1")
    return st.session_state[k]


def fig_layout(fig, title="", x="", y="", height=320, legend=True):
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=NAVY)), height=height, template="plotly_white",
                      margin=dict(l=10, r=10, t=45 if title else 15, b=10), xaxis_title=x, yaxis_title=y,
                      legend=dict(orientation="h", y=-0.22) if legend else None, showlegend=legend,
                      font=dict(size=12))
    return fig


def lines(x, series, title="", xlab="", ylab="", height=320, colors=None, dash=None):
    fig = go.Figure()
    for k, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                                 line=dict(color=(colors or {}).get(name), width=2, dash=(dash or {}).get(name))))
    return fig_layout(fig, title, xlab, ylab, height)


def show(fig, key=None):
    st.plotly_chart(fig, width="stretch", key=key)


# Columns whose values span many orders of magnitude are shown in scientific notation.
SCIENTIFIC = ("difference", "max_abs_difference", "likelihood", "unnormalised", "grad_norm")


def table(frame, digits=4, height=None, sci=()):
    frame = pd.DataFrame(frame)
    floats = [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]
    sci = set(sci) | {c for c in floats if c in SCIENTIFIC}
    kwargs = dict(hide_index=True, width="stretch",
                  column_config={c: st.column_config.NumberColumn(format="%.3e" if c in sci else f"%.{digits}f")
                                 for c in floats})
    if height:
        kwargs["height"] = height
    st.dataframe(frame, **kwargs)


def download(frame, name, label="Download CSV"):
    st.download_button(label, pd.DataFrame(frame).to_csv(index=False), name, "text/csv")


def fmt(v, d=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float) and np.isinf(v):
        return "+∞" if v > 0 else "−∞"
    return f"{v:.{d}f}"
