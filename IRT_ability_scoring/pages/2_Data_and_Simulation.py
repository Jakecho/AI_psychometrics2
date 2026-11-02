import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (BLUE, ORANGE, add_dataset, data, download, fig_layout, interpret, labels, page_setup, pattern_text,
                           person, show, table)
from irt_models import DICHOTOMOUS, POLYTOMOUS, extreme_status
from simulation import EXAMPLES, items_frame, items_from_frame, simulate

page_setup("2 · Data & Simulation", "Choose the items and responses every page will score: built-in examples or a simulated test.")
d = data()

st.subheader("A · Built-in numerical examples")
cols = st.columns(4)
for col, (name, fn) in zip(cols, EXAMPLES.items()):
    ex = fn()
    with col:
        with st.container(border=True):
            st.markdown(f"**{ex.name}**")
            st.caption(ex.description)
            if st.button("Use this example", key=f"use_{name}", width="stretch"):
                add_dataset(ex)
                st.rerun()
st.caption("Loading an example restores its original items and patterns (edits are discarded).")

st.subheader("B · Simulate a test (dichotomous, polytomous or mixed format)")
with st.form("simulate"):
    c1, c2, c3, c4 = st.columns(4)
    n = c1.number_input("Examinees N", 10, 20000, 300, 50)
    seed = c1.number_input("Random seed", 0, 10**6, 2026)
    dist = c2.selectbox("True ability distribution", ["normal", "uniform", "skewed"])
    tm = c2.number_input("Ability mean", -3.0, 3.0, 0.0, 0.1)
    tsd = c2.number_input("Ability SD", 0.2, 3.0, 1.0, 0.1)
    n_dich = c3.number_input("Dichotomous items", 0, 200, 15)
    dich_model = c3.selectbox("Dichotomous model", DICHOTOMOUS, index=1)
    n_poly = c3.number_input("Polytomous items", 0, 100, 5)
    poly_model = c4.selectbox("Polytomous model", POLYTOMOUS, index=1)
    n_cat = c4.number_input("Categories per polytomous item", 2, 7, 4)
    missing = c4.slider("Missing-response rate", 0.0, 0.5, 0.0, 0.05)
    with st.expander("Item-parameter distributions"):
        e1, e2, e3 = st.columns(3)
        b_mean = e1.number_input("Difficulty / location mean", -3.0, 3.0, 0.0, 0.1)
        b_sd = e1.number_input("Difficulty / location SD", 0.0, 3.0, 1.0, 0.1)
        a_sdlog = e2.number_input("log-discrimination SD (a = exp(N(0, s)))", 0.0, 1.0, 0.25, 0.05)
        spread = e2.number_input("Step / threshold spread (±)", 0.1, 3.0, 1.0, 0.1)
        c_lo, c_hi = e3.slider("Lower asymptote c range (3PL/4PL)", 0.0, 0.4, (0.1, 0.25), 0.01)
        d_lo, d_hi = e3.slider("Upper asymptote d range (4PL)", 0.6, 1.0, (0.9, 0.99), 0.01)
        D = e3.radio("Scaling constant D", [1.0, 1.702], horizontal=True)
    go_sim = st.form_submit_button("Generate responses", type="primary")
if go_sim:
    if n_dich + n_poly == 0:
        st.error("Choose at least one item.")
    else:
        new = simulate(int(n), seed=int(seed), theta_dist=dist, theta_mean=tm, theta_sd=tsd, missing_rate=missing,
                       n_dich=int(n_dich), dich_model=dich_model, n_poly=int(n_poly), poly_model=poly_model,
                       n_cat=int(n_cat), b_mean=b_mean, b_sd=b_sd, a_sdlog=a_sdlog, step_spread=spread,
                       c_range=(c_lo, c_hi), d_range=(d_lo, d_hi), D=D)
        add_dataset(new)
        st.rerun()

d = data()
st.subheader(f"C · Item parameters · {d.name}")
labels("edu")
st.caption("Edit any value and press **Apply item edits**. Steps are comma separated (PCM/GPCM step difficulties, "
           "GRM thresholds, RSM τ). Constraints are checked before the edit is accepted.")
frame = pd.DataFrame(items_frame(d.items))
edited = st.data_editor(frame, hide_index=True, width="stretch", key=f"items_{d.identity}",
                        disabled=["id", "model", "categories"],
                        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("a", "b", "c", "d", "D")})
if st.button("Apply item edits"):
    try:
        items = items_from_frame(edited)
        cats_changed = [it.id for it, old in zip(items, d.items) if it.n_cat != old.n_cat]
        if cats_changed:
            raise ValueError(f"the number of steps changes the category count of {', '.join(cats_changed)}; "
                             "simulate a new test to change categories")
        add_dataset(d.with_items(items), person=person())
        st.rerun()
    except ValueError as e:
        st.error(f"Not applied: {e}")

st.subheader("D · Selected examinee's response pattern")
p = person()
y = d.Y[p]
st.markdown(f"**{d.person_ids[p]}** · responses <span class='pattern'>{pattern_text(y)}</span> · summed score "
            f"{int(np.nansum(y))}", unsafe_allow_html=True)
row = pd.DataFrame([{it.id: (None if np.isnan(v) else int(v)) for it, v in zip(d.items, y)}])
edited_row = st.data_editor(row, hide_index=True, width="stretch", key=f"pattern_{d.identity}_{p}",
                            column_config={it.id: st.column_config.NumberColumn(min_value=0, max_value=it.max_score, step=1,
                                                                                help=f"0 … {it.max_score}; blank = missing")
                                           for it in d.items})
st.caption("Change a response (blank = missing) and apply; every page then scores the edited pattern.")
if st.button("Apply pattern edit"):
    new_y = np.array([np.nan if pd.isna(v) else float(v) for v in edited_row.iloc[0].tolist()])
    bad = [it.id for it, v in zip(d.items, new_y) if not np.isnan(v) and (v < 0 or v > it.max_score or v != int(v))]
    if bad:
        st.error(f"Not applied: responses out of range for {', '.join(bad)}.")
    elif np.isnan(new_y).all():
        st.error("Not applied: at least one response must be observed.")
    else:
        add_dataset(d.with_pattern(p, new_y), person=p)
        st.rerun()

st.subheader("E · Response summary")
Y = d.Y
max_scores = np.array([it.max_score for it in d.items])
raw = np.nansum(Y, 1)
ext = extreme_status(d.items, Y)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Examinees", f"{d.n_persons:,}")
c2.metric("Missing responses", f"{np.isnan(Y).mean():.1%}")
c3.metric("All-lowest patterns", int((ext == "min").sum()))
c4.metric("All-highest patterns", int((ext == "max").sum()))
left, right = st.columns(2)
with left:
    fig = go.Figure(go.Histogram(x=raw, marker=dict(color=BLUE, line=dict(color="white", width=2)), nbinsx=int(max_scores.sum()) + 1))
    show(fig_layout(fig, "Summed scores", "summed score (observed items)", "examinees", 300, legend=False))
with right:
    rel = np.nanmean(Y, 0) / max_scores
    fig = go.Figure(go.Bar(x=[it.id for it in d.items], y=rel, marker=dict(color=ORANGE, line=dict(color="white", width=2))))
    show(fig_layout(fig, "Mean item score ÷ maximum score", "item", "proportion of maximum", 300, legend=False))
preview = pd.DataFrame(Y, columns=[it.id for it in d.items])
preview.insert(0, "examinee", d.person_ids)
if d.has_truth:
    preview["true_theta"] = d.theta_true
table(preview.head(200), 0, height=280)
download(preview, "responses.csv", "Download responses (CSV)")
interpret(f"{int((ext != '').sum())} examinee(s) answered every observed item in its lowest or highest category. For "
          "them the likelihood keeps rising toward ±∞, so **MLE and TCC inversion have no finite estimate**; MAP, EAP, "
          "EAPsum and WLE still return finite values. The Monte Carlo page reports how often this happens. "
          + ("True abilities are known for this dataset, so recovery can be checked." if d.has_truth else
             "This dataset has no generating abilities: comparisons are between methods, not against truth."))
