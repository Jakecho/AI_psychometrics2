import numpy as np
import pandas as pd
import streamlit as st

from components.ui import add_dataset, bars, config, data, guided, interpret, page_setup, table
from core.data import anchors_from_frame, from_frame, validate
from core.em import DESIGN_NAMES
from core.models import MODEL_NAMES, MODELS
from core.simulation import SimSpec, simulate

page_setup("4 · Data Simulation", "Generate responses under any model, design and ability distribution — or load your own "
                                  "responses and bank anchors.")
guided(["Choose model, design and sizes; the generating distribution can differ by group (mixtures).",
        "Generate: the dataset becomes active everywhere and the sidebar switches to its model and design.",
        "Inspect category counts and the missing-by-design pattern below.",
        "Or upload a CSV of responses (blank = not administered) and, for anchored designs, a bank table."],
       "In a concurrent design, a blank cell for an item that a form did not contain is treated as…",
       ["missing by design — it does not enter the likelihood", "a score of 0", "the item's mean score",
        "an error"], "missing by design — it does not enter the likelihood",
       "Only observed responses contribute: the product in L_i(θ) runs over O_i, the items examinee i saw.", key="sim")

tab_sim, tab_up, tab_view = st.tabs(["Simulate", "Upload", "Inspect active dataset"])
with tab_sim:
    c1, c2, c3, c4 = st.columns(4)
    model = c1.selectbox("Generating model", MODELS, index=2, format_func=lambda m: f"{m} · {MODEL_NAMES[m]}")
    design = c2.selectbox("Design", list(DESIGN_NAMES), format_func=DESIGN_NAMES.get)
    n = c3.select_slider("Examinees per group", [100, 250, 500, 1000, 2000, 5000], 1000)
    ncat = c4.selectbox("Categories", [3, 4, 5], index=1)
    c1, c2, c3, c4 = st.columns(4)
    J = c1.select_slider("Items per form", [4, 6, 8, 10, 20, 40], 10)
    G = c2.number_input("Groups / forms", 2, 4, 2, disabled=design in ("single", "fipc"))
    n_common = c3.number_input("Common free items", 0, 20, 4, disabled=design in ("single", "fipc"))
    n_anchor = c4.number_input("Bank anchors", 1, 20, 5, disabled=design in ("single", "concurrent"))
    groups = 1 if design in ("single", "fipc") else int(G)
    st.markdown("**Generating ability distribution per group** (population moments; shapes are standardised)")
    defaults = pd.DataFrame(dict(group=[f"G{g + 1}" for g in range(groups)],
                                 mean=[0.0, 0.5, -0.3, 0.8][:groups] if design not in ("fipc",) else [0.4],
                                 sd=[1.0, 1.1, 0.9, 1.2][:groups] if design not in ("fipc",) else [1.1],
                                 shape=["normal"] * groups))
    gtab = st.data_editor(defaults, hide_index=True, disabled=["group"], key=f"gtab_{design}_{groups}",
                          column_config=dict(shape=st.column_config.SelectboxColumn(options=["normal", "skewed", "bimodal", "discrete"])))
    if design in ("single", "concurrent"):
        st.caption("Without anchors, group 1 is the reference: its population is generated as mean 0 (and SD 1 for "
                   "GPCM/GRM), matching the identification constraint so recovery is on the same metric.")
    c1, c2, c3, c4 = st.columns(4)
    drift = c1.number_input("Anchor drift (location shift)", -2.0, 2.0, 0.0, 0.1, disabled=design in ("single", "concurrent"))
    n_drift = c2.number_input("Drifting anchors", 0, 20, 0, disabled=design in ("single", "concurrent"))
    a_err = c3.number_input("Bank error SD", 0.0, 1.0, 0.0, 0.05, disabled=design in ("single", "concurrent"))
    seed = c4.number_input("Seed", 0, 10 ** 6, 2026)
    if st.button("Generate dataset", type="primary"):
        try:
            spec = SimSpec(model=model, design=design, n_per_group=int(n), n_items=int(J), ncat=int(ncat), groups=groups,
                           n_common=int(n_common), n_anchor=int(n_anchor), shape=gtab["shape"].iloc[0],
                           group_means=gtab["mean"].astype(float).tolist(), group_sds=gtab["sd"].astype(float).tolist(),
                           group_shapes=gtab["shape"].tolist(), anchor_drift=float(drift), n_drift=int(n_drift),
                           anchor_error=float(a_err), seed=int(seed))
            d = simulate(spec)
            add_dataset(d)
            st.success(f"Generated and activated: {d.name}")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

with tab_up:
    st.markdown("Responses CSV: one column per item (integer categories 0…K, blank = not administered), optional "
                "group column. Bank CSV (anchored designs): `item, model, a, d1, d2, …` (RSM may give `b, tau1, …`).")
    up = st.file_uploader("Responses CSV", type="csv", key="up_resp")
    bank = st.file_uploader("Bank anchors CSV (optional)", type="csv", key="up_bank")
    if up is not None:
        frame = pd.read_csv(up)
        st.dataframe(frame.head(10), hide_index=True)
        gcol = st.selectbox("Group column", ["(none)"] + list(frame.columns))
        name = st.text_input("Dataset name", up.name.rsplit(".", 1)[0])
        if st.button("Load dataset"):
            try:
                anchors = anchors_from_frame(pd.read_csv(bank)) if bank is not None else {}
                d = from_frame(frame, name=f"Uploaded · {name}", group_column=None if gcol == "(none)" else gcol,
                               anchors=anchors)
                errs, warns = validate(d, None, bool(anchors))
                for w in warns:
                    st.warning(w)
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    add_dataset(d)
                    st.success(f"Loaded {d.N} examinees × {d.J} items in {d.G} group(s), {len(anchors)} anchors.")
            except (ValueError, KeyError) as e:
                st.error(f"Could not read the files: {e}")

with tab_view:
    d = data()
    st.markdown(f"**{d.name}** — {d.description}")
    cfg = config()
    errs, warns = validate(d, cfg.model, cfg.anchored)
    for e in errs:
        st.error(e)
    for w in warns[:8]:
        st.warning(w)
    rows = []
    for j, s in enumerate(d.specs):
        c = d.counts(j)
        rows.append(dict(item=s.id, categories=s.ncat, n=int(c.sum()), anchor=s.id in d.anchors,
                         **{f"k={k}": int(c[k]) for k in range(max(d.ncats))[:s.ncat]}))
    table(pd.DataFrame(rows), digits=0)
    adm = d.administered()
    st.markdown("**Administration pattern** (✓ = item on the group's form)")
    table(pd.DataFrame([{**{"group": d.group_names[g], "N": int(d.group_sizes()[g])},
                         **{d.ids[j]: "✓" if adm[g, j] else "" for j in range(d.J)}} for g in range(d.G)]))
    if d.truth:
        th = d.truth["theta"]
        hist = [np.histogram(th[d.group == g], bins=np.linspace(-4, 4, 33))[0] / max((d.group == g).sum(), 1)
                for g in range(d.G)]
        bars(np.round(np.linspace(-4, 4, 33)[:-1] + 0.125, 3), {d.group_names[g]: hist[g] for g in range(d.G)},
             "Generated abilities (sample proportions)", key="theta_hist")
        with st.expander("Generating item parameters and bank values"):
            rows = []
            for k, it in d.truth["items"].items():
                r = it.as_row()
                if k in d.anchors:
                    r["bank location"] = d.anchors[k].location()
                    r["generating location (later groups)"] = d.truth["generating"][k].location()
                rows.append(r)
            table(pd.DataFrame(rows))
    st.download_button("Download responses CSV", d.to_csv(), f"{d.name[:40]}.csv", "text/csv")
    interpret("Sparse or empty categories make the adjacent step (or threshold) weakly identified: the engine keeps "
              "it finite with bounds and flags it. Items on only one form are linked to the other forms only "
              "through the common items (concurrent) or through anchors (FIPC/MG-FIPC).")
