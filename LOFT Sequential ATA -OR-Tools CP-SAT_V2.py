"""
LOFT Sequential ATA — OR-Tools CP-SAT (V2)
==========================================
Linear-on-the-Fly Testing with sequential form assembly using Google OR-Tools CP-SAT.
Run: streamlit run "LOFT Sequential ATA -OR-Tools CP-SAT_V2.py"

V2 changes from LOFT_squentialATA.py (left unchanged):
- Item bundles (bundle_id / testlet_id): all-or-nothing, bundle count limits, solved bundles first
  (with a look-ahead over the standalone fill), then standalone items; joint fallback.
- Faster models: one variable per bundle, constraints that cannot bind are dropped, parallel CP-SAT
  workers, and a small optimality gap for the two-stage stages.
- Target lines on the TIF/TCC plots; independent per-form constraint check (also exported).
- Sequential loop in loft_v2_engine.SequentialAssembler (shared by Run All and Step; tested directly);
  seeded sampling for reproducible runs; cached bank loading; throttled live charts for long runs.
- Built-in synthetic sample pool (loft_sample_item_pool.csv) covering every constraint.

Strategies from Cho (2025), Optimizing LOFT Test Assembly: Strategies for Exposure and Form Diversity:
(1) domain-specific exposure limits, (2) domain-stratified active pools, (3) auto difficulty bins.
"""

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from loft_v2_engine import (AssemblySettings, SequentialAssembler, auto_generate_difficulty_categories,
                            calculate_tcc, calculate_tif, curves)
from loft_v2_solver import bundle_members, validate_item_bank

SAMPLE_POOL = Path(__file__).resolve().parent / 'loft_sample_item_pool.csv'
HEATMAP_LIMIT = 60          # forms shown in the overlap matrix / heatmap

# ==================== Page Configuration ====================

st.set_page_config(page_title="LOFT Sequential ATA — OR-Tools CP-SAT (V2)", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #FAFAFA; }
    h1 { color: #6c3483; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stButton>button { background-color: #8e44ad; color: white; border-radius: 8px; font-weight: 600;
                       padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #7d3c98; }
</style>
""", unsafe_allow_html=True)
st.title("🧬 LOFT Sequential ATA — OR-Tools CP-SAT (V2)")
st.markdown("*A live dashboard for sequential Linear-on-the-Fly Test assembly using Google OR-Tools CP-SAT. "
            "All processing runs locally.*")
st.caption("📚 Based on: Cho (2025). _Optimizing LOFT Test Assembly: Strategies for Exposure and Form "
           "Diversity._ Credentialing Insights.")


@st.cache_data(show_spinner=False, max_entries=8)
def load_bank(data: bytes, name: str) -> pd.DataFrame:
    text_columns = {'item_id': str, 'bundle_id': str, 'testlet_id': str}
    if name.lower().endswith('.csv'):
        frame = pd.read_csv(io.BytesIO(data), dtype=text_columns)
    else:
        frame = pd.read_excel(io.BytesIO(data), dtype=text_columns)
    return validate_item_bank(frame)


# ==================== 1. File Upload ====================

st.sidebar.header("📁 1. Master Pool")
uploaded_file = st.sidebar.file_uploader(
    "Upload Item Bank (CSV or Excel)", type=['csv', 'xlsx', 'xls'],
    help="Required columns: item_id, domain, rasch_b, pvalue, point_biserial. "
         "Optional: raschb_cat, enemy_ids, has_image, has_audio, bundle_id (or testlet_id)")
use_sample = False
if not uploaded_file and SAMPLE_POOL.exists():
    use_sample = st.sidebar.checkbox(
        "Use built-in sample pool", value=False,
        help="504 synthetic items: 6 domains, 7 difficulty buckets, image/audio items, enemy pairs and "
             "24 item bundles. Every constraint in the app can be tried on it.")
    st.sidebar.download_button("📥 Download sample pool (CSV)", SAMPLE_POOL.read_bytes(),
                               file_name=SAMPLE_POOL.name, mime="text/csv")

if not uploaded_file and not use_sample:
    st.info("👋 Upload a CSV or Excel item bank to begin, or tick **Use built-in sample pool**.")
    with st.expander("📋 Example File Format"):
        example_df = pd.DataFrame({
            'item_id': ['NCX0001', 'NCX0002', 'NCX0003', 'NCX0004', 'NCX0005'],
            'domain': ['Cardiology', 'Cardiology', 'Pharmacology', 'Med-Surg', 'Med-Surg'],
            'rasch_b': [0.15, -0.32, 0.45, -0.10, 0.22],
            'pvalue': [0.62, 0.75, 0.55, 0.68, 0.60],
            'point_biserial': [0.35, 0.42, 0.38, 0.40, 0.36],
            'raschb_cat': ['6. hard', '3. easy', '6. hard', '4. moderately easy', '6. hard'],
            'enemy_ids': ['NCX0002', '', 'NCX0004', 'NCX0003', ''],
            'has_image': [1, 0, 1, 1, 0],
            'bundle_id': ['', '', 'CASE01', 'CASE01', ''],
        })
        st.dataframe(example_df)
        st.caption("Items sharing a bundle_id (e.g. a case study) are always selected together; leave it blank "
                   "for standalone items.")
        st.download_button("📥 Download Example CSV", data=example_df.to_csv(index=False),
                           file_name="item_pool_example.csv", mime="text/csv")
    st.stop()

try:
    if uploaded_file:
        bank_bytes, bank_name = uploaded_file.getvalue(), uploaded_file.name
    else:
        bank_bytes, bank_name = SAMPLE_POOL.read_bytes(), SAMPLE_POOL.name
    df = load_bank(bank_bytes, bank_name)
except Exception as exc:
    st.error(f"Unable to load item bank: {exc}")
    st.stop()

st.sidebar.success(f"✅ Bank Loaded: {len(df)} items" + (" (built-in sample)" if not uploaded_file else ""))

has_enemies = 'enemy_ids' in df.columns
has_image = 'has_image' in df.columns
has_audio = 'has_audio' in df.columns
has_rasch_cat = 'raschb_cat' in df.columns
has_domain = 'domain' in df.columns
bundles_in_bank = bundle_members(df)
has_bundles = bool(bundles_in_bank)

features = []
if has_domain:
    features.append(f"Domains ({df['domain'].nunique()})")
if has_enemies:
    features.append("Enemy Items")
if has_bundles:
    features.append(f"Item Bundles ({len(bundles_in_bank)}; {sum(len(r) for r in bundles_in_bank.values())} items)")
if has_image:
    features.append("Images")
if has_audio:
    features.append("Audio")
if has_rasch_cat:
    features.append(f"Difficulty Buckets ({df['raschb_cat'].nunique()})")
st.sidebar.caption("Features: " + " · ".join(features) if features else "No special columns detected.")
st.sidebar.divider()

# ==================== Master Pool Summary ====================

st.markdown("### 📊 Master Pool Summary")
sp1, sp2, sp3, sp4 = st.columns(4)
sp1.metric("Total Items", len(df))
sp2.metric("Mean Rasch B", f"{df['rasch_b'].mean():.2f}")
sp3.metric("SD Rasch B", f"{df['rasch_b'].std():.2f}")
sp4.metric("B Range", f"[{df['rasch_b'].min():.1f}, {df['rasch_b'].max():.1f}]")
if has_domain:
    st.markdown("**Domain Breakdown:**")
    domain_stats = df.groupby('domain')['rasch_b'].agg(['count', 'mean', 'std']).round(2)
    domain_stats.columns = ['Count', 'Mean B', 'SD B']
    st.dataframe(domain_stats, width='stretch')
if has_rasch_cat:
    st.markdown("**Difficulty Bucket Distribution:**")
    bucket_stats = df.groupby(df['raschb_cat'].astype(str))['rasch_b'].agg(['count', 'mean']).round(2)
    bucket_stats.columns = ['Count', 'Mean B']
    bucket_stats.index.name = 'Bucket'
    st.dataframe(bucket_stats, width='stretch')
if has_bundles:
    with st.expander(f"📦 Item Bundles ({len(bundles_in_bank)} bundles, all-or-nothing)"):
        bundle_rows = df[df['bundle_id'] != '']
        bundle_stats = pd.DataFrame({
            'Items': bundle_rows.groupby('bundle_id')['item_id'].count(),
            'Domain': bundle_rows.groupby('bundle_id')['domain'].agg(lambda d: ', '.join(sorted(set(d)))),
            'Mean B': bundle_rows.groupby('bundle_id')['rasch_b'].mean().round(2),
            'Images': bundle_rows.groupby('bundle_id')['has_image'].sum() if has_image else 0,
            'Audio': bundle_rows.groupby('bundle_id')['has_audio'].sum() if has_audio else 0})
        st.dataframe(bundle_stats, width='stretch')
st.divider()

# ==================== 2. Engine ====================

st.sidebar.header("🤖 2. Engine")
st.sidebar.caption("⚡ OR-Tools CP-SAT — constraint optimization, no LLM required.")
st.sidebar.divider()
with st.sidebar.expander("Solver Settings", expanded=False):
    solver_seconds = st.number_input("Time Limit per Solve (seconds)", 1, 120, 10)
    sampling_attempts = st.number_input("Maximum Sampling Attempts per Form", 1, 100, 10)
    num_workers = st.number_input("Solver Workers", 1, 32, min(8, os.cpu_count() or 1),
                                  help="Parallel CP-SAT search workers. 1 gives fully repeatable solver results.")
    random_seed = st.number_input("Random Seed", 0, 2**31 - 1, 0,
                                  help="Seeds the active-pool sampling. 0 = a new random run each time.")
    live_every = st.number_input("Live Chart Update Every N Forms", 0, 1000, 0,
                                 help="0 = automatic (about 30 updates per run). Charts are always drawn at the end.")

# ==================== 3. Constraints ====================

st.sidebar.header("⚙️ 3. Objectives & Constraints")
with st.sidebar.expander("Form Specifications", expanded=True):
    c1, c2 = st.columns(2)
    n_forms = c1.number_input("# Forms", 1, 10000, 3)
    test_len = c2.number_input("Test Length", 1, 500, 10)
    c3, c4 = st.columns(2)
    multiplier = c3.number_input("Pool Multiplier", 2, 10, 5)
    exposure_max = c4.number_input("Default Exposure Limit", 1, 1000, 2,
                                   help="Per-item reuse limit. Domain-specific limits override this default.")
    max_overlap = st.slider("Max Jaccard Overlap", 0.0, 1.0, 0.3, 0.05)
    apply_enemies = st.checkbox("Apply Enemy Item Constraints", value=has_enemies, disabled=not has_enemies)

with st.sidebar.expander("📌 Domain Exposure Limits (Strategy 1)", expanded=False):
    st.caption("Domains with fewer items may need higher exposure limits to avoid pool exhaustion. "
               "Enable auto-compute or set manually.")
    auto_domain_limits = st.checkbox("Auto-compute domain limits", value=True,
                                     help="Per-domain exposure limits from pool depth (Cho, 2025). "
                                          "Domains with fewer items get higher limits.")
    domain_exposure_payload = {}
    if not auto_domain_limits and has_domain:
        for dom in sorted(df['domain'].dropna().unique().tolist()):
            domain_exposure_payload[dom] = st.number_input(
                f"{dom} ({int((df['domain'] == dom).sum())} items)", 1, 1000, exposure_max, key=f"exp_{dom}")

with st.sidebar.expander("Targets & Tolerances", expanded=True):
    c5, c6 = st.columns(2)
    theta_target = c5.number_input("Theta Target (θ)", -3.0, 3.0, 0.0, 0.1)
    tif_target = c6.number_input("TIF Target", 0.0, 100.0, 2.0, 0.5)
    c7, c8 = st.columns(2)
    mean_diff = c7.number_input("Mean B Target", -3.0, 3.0, 0.0, 0.1)
    mean_tol = c8.number_input("Mean B Tol", 0.0, 3.0, 0.2, 0.1)
    c9, c10 = st.columns(2)
    tif_tol_val = c9.number_input("TIF Tol (+/-)", 0.0, 10.0, 0.2, 0.1)
    tcc_tol_val = c10.number_input("TCC Tol (+/-)", 0.0, 5.0, 0.5, 0.1)
    tcc_target_val = st.number_input("TCC Target (Expected Score)", min_value=0.0, value=0.0, step=1.0,
                                     help="0.0 = Disabled")

with st.sidebar.expander("Dynamic Domains", expanded=False):
    st.caption("A maximum of 0 leaves the upper limit unrestricted.")
    domain_payload = {}
    if has_domain:
        for dom in sorted(df['domain'].dropna().unique().tolist()):
            st.markdown(f"**{dom}** ({int((df['domain'] == dom).sum())} items)")
            d1, d2 = st.columns(2)
            d_min = d1.number_input("Min", 0, test_len, 0, key=f"dmin_{dom}")
            d_max = d2.number_input("Max", 0, test_len, 0, key=f"dmax_{dom}")
            if d_min > 0 or d_max > 0:
                domain_payload[dom] = {'min': d_min, 'max': d_max if d_max > 0 else test_len}
    else:
        st.info("No 'domain' column found in bank.")

with st.sidebar.expander("Media Constraints", expanded=False):
    st.caption("A maximum of 0 leaves the upper limit unrestricted.")
    c_items = st.text_input("Common Item IDs (comma-separated)", "")
    st.markdown("**Excluded Items**")
    excluded_str = st.text_input("Excluded Item IDs (comma-separated)", "", key="excluded_items_input")
    st.markdown("**Images**")
    im1, im2 = st.columns(2)
    img_min = im1.number_input("Min Image", 0, test_len, 0, disabled=not has_image)
    img_max = im2.number_input("Max Image", 0, test_len, 0, disabled=not has_image)
    st.markdown("**Audio**")
    au1, au2 = st.columns(2)
    audio_min = au1.number_input("Min Audio", 0, test_len, 0, disabled=not has_audio)
    audio_max = au2.number_input("Max Audio", 0, test_len, 0, disabled=not has_audio)

with st.sidebar.expander("📦 Item Bundles", expanded=False):
    bundle_constraints = {}
    bundle_strategy = 'two-stage'
    if has_bundles:
        st.caption("Items that share a bundle_id are selected all-or-nothing, and exposure, sampling and "
                   "exclusions apply to the whole bundle. A maximum of 0 is unrestricted.")
        bu1, bu2 = st.columns(2)
        min_bundles = bu1.number_input("Min Bundles", 0, len(bundles_in_bank), 0)
        max_bundles = bu2.number_input("Max Bundles", 0, len(bundles_in_bank), 0)
        bu3, bu4 = st.columns(2)
        min_bundle_items = bu3.number_input("Min Bundle Items", 0, test_len, 0)
        max_bundle_items = bu4.number_input("Max Bundle Items", 0, test_len, 0)
        if min_bundles or max_bundles or min_bundle_items or max_bundle_items:
            bundle_constraints = {'min_bundles': min_bundles, 'max_bundles': max_bundles,
                                  'min_bundle_items': min_bundle_items, 'max_bundle_items': max_bundle_items}
        strategy_label = st.radio(
            "Solve order", ["Bundles first, then standalone items", "Joint (one model)"],
            help="Bundles first: CP-SAT picks bundles (looking ahead at what standalone items can still add), "
                 "then fills the remaining slots with standalone items; a failed fill rules that bundle set out "
                 "and retries, then falls back to the joint model. Much faster; usually within 1% of the joint "
                 "objective. Joint: one model, proves the best objective.")
        bundle_strategy = 'two-stage' if strategy_label.startswith('Bundles') else 'joint'
    else:
        st.info("No 'bundle_id' (or 'testlet_id') column in bank.")

with st.sidebar.expander("📊 Difficulty Bins (Strategy 3)", expanded=False):
    st.caption("Auto-generate difficulty category constraints to maximize item pool utilization (Cho, 2025). "
               "Prevents the engine from always selecting items near the cut score.")
    auto_diff_bins = st.checkbox("Enable auto-difficulty bins", value=True,
                                 help="Divide items into bins based on standard normal density and impose "
                                      "min/max per bin.")
    n_diff_bins = st.number_input("Number of bins", 3, 11, 7, step=2, disabled=not auto_diff_bins,
                                  help="Odd number recommended (e.g. 5, 7, 9). "
                                       "7 bins centered at [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]")
    difficulty_categories = (auto_generate_difficulty_categories(df, test_len, n_categories=n_diff_bins)
                             if auto_diff_bins else {})
    if auto_diff_bins and difficulty_categories:
        st.dataframe(pd.DataFrame([{'Bin': label, 'Range': f"[{i['range'][0]:.1f}, {i['range'][1]:.1f})",
                                    'Pool': i['pool_count'], 'Target': i['target'], 'Min': i['min'], 'Max': i['max']}
                                   for label, i in difficulty_categories.items()]), width='stretch', hide_index=True)
    elif auto_diff_bins:
        st.info("No bins generated (check rasch_b column).")

with st.sidebar.expander("Difficulty Buckets (raschb_cat)", expanded=False):
    st.caption("A maximum of 0 leaves the upper limit unrestricted.")
    rasch_cats = {}
    if has_rasch_cat:
        for cat in sorted(df['raschb_cat'].astype(str).unique().tolist()):
            st.markdown(f"**{cat}** ({int((df['raschb_cat'].astype(str) == cat).sum())} items)")
            b1, b2 = st.columns(2)
            b_min = b1.number_input("Min", 0, test_len, 0, key=f"bmin_{cat}")
            b_max = b2.number_input("Max", 0, test_len, 0, key=f"bmax_{cat}")
            if b_min > 0 or b_max > 0:
                rasch_cats[cat] = {'min': b_min, 'max': b_max if b_max > 0 else test_len}
    else:
        st.info("No 'raschb_cat' column in bank.")

# ==================== Settings → solver configuration ====================

common_ids = [x.strip() for x in c_items.split(',') if x.strip()]
excluded_ids = [x.strip() for x in excluded_str.split(',') if x.strip()]
solver_config = {
    'test_length': int(test_len), 'time_limit': float(solver_seconds), 'num_workers': int(num_workers),
    'difficulty_categories': difficulty_categories,
    'domain_constraints': domain_payload, 'raschb_cat_constraints': rasch_cats, 'enemy_check': apply_enemies,
    'eval_points': {'theta_low': theta_target - 1, 'theta_mid': theta_target, 'theta_high': theta_target + 1,
                    'tif_low': max(tif_target - 0.5, 0), 'tif_mid': tif_target, 'tif_high': max(tif_target - 0.5, 0),
                    'tcc_mid': tcc_target_val if tcc_target_val > 0 else 0.0, 'tcc_enabled': tcc_target_val > 0},
    'tif_tolerance': {'tif': tif_tol_val, 'tcc': tcc_tol_val},
    'apply_mean_diff': True, 'mean_diff_target': mean_diff, 'mean_diff_tolerance': mean_tol,
    'bundle_constraints': bundle_constraints, 'bundle_strategy': bundle_strategy,
    'common_items': common_ids, 'excluded_items': excluded_ids,
}
if has_image and (img_min > 0 or img_max > 0):
    solver_config['image_constraint'] = {'min': img_min, 'max': img_max if img_max > 0 else test_len, 'enabled': True}
if has_audio and (audio_min > 0 or audio_max > 0):
    solver_config['audio_constraint'] = {'min': audio_min, 'max': audio_max if audio_max > 0 else test_len,
                                         'enabled': True}
pool_rules = {'test_length': int(test_len), 'multiplier': int(multiplier), 'domain_constraints': domain_payload,
              'common_items': common_ids, 'bundle_constraints': bundle_constraints}
settings = AssemblySettings(
    test_length=int(test_len), n_forms=int(n_forms), solver_config=solver_config, pool_rules=pool_rules,
    max_overlap=float(max_overlap), exposure_max=int(exposure_max), auto_domain_limits=auto_domain_limits,
    domain_exposure_limits=domain_exposure_payload or None, sampling_attempts=int(sampling_attempts),
    seed=int(random_seed) or None)

# A different bank or different settings start a new assembly session.
settings_fingerprint = hashlib.sha256(bank_bytes + json.dumps(
    [solver_config, pool_rules, settings.__dict__ | {'solver_config': None, 'pool_rules': None}],
    sort_keys=True, default=str).encode()).hexdigest()
if st.session_state.get('cpsat_settings') != settings_fingerprint:
    for state_key in list(st.session_state):
        if state_key.startswith('cpsat_'):
            del st.session_state[state_key]
    st.session_state['cpsat_settings'] = settings_fingerprint

# ==================== Plots with target lines ====================

THETA_GRID = np.linspace(min(-3.0, theta_target - 1.5), max(3.0, theta_target + 1.5), 121)
TIF_TARGETS = [(theta_target - 1, max(tif_target - 0.5, 0)), (theta_target, tif_target),
               (theta_target + 1, max(tif_target - 0.5, 0))]


def _target_figures():
    """Empty TIF/TCC figures carrying the assembly targets.

    TIF: dashed line at the θ-target information target, dotted line at θ, and the three constrained points
    (θ−1, θ, θ+1) with their ± tolerance. TCC: the same for the expected-score target when enabled (otherwise
    only the θ line). Target markers are named 'TIF target' / 'TCC target'.
    """
    fig_t, fig_c = go.Figure(), go.Figure()
    fig_t.update_layout(title="Test Information Functions (TIF)", xaxis_title="Theta (θ)", yaxis_title="Information",
                        height=400)
    fig_c.update_layout(title="Test Characteristic Curve (TCC)", xaxis_title="Theta (θ)",
                        yaxis_title="Expected Score", height=400)
    for fig in (fig_t, fig_c):
        fig.add_vline(x=theta_target, line=dict(color='#7f8c8d', dash='dot', width=1),
                      annotation_text=f"θ target = {theta_target:g}", annotation_position='bottom right')
    active = [(t, v) for t, v in TIF_TARGETS if v > 0]
    if tif_target > 0:
        fig_t.add_hline(y=tif_target, line=dict(color='#c0392b', dash='dash', width=1.5),
                        annotation_text=f"TIF target {tif_target:g} ± {tif_tol_val:g}", annotation_position='top left')
    if active:
        fig_t.add_trace(go.Scatter(
            x=[t for t, _ in active], y=[v for _, v in active], mode='markers', name='TIF target',
            marker=dict(symbol='diamond', size=11, color='#c0392b'),
            error_y=dict(type='constant', value=tif_tol_val, color='#c0392b', thickness=2, width=8),
            hovertemplate="θ=%{x:.2f}: target %{y:.2f} ± " + f"{tif_tol_val:g}<extra></extra>"))
    if tcc_target_val > 0:
        fig_c.add_hline(y=tcc_target_val, line=dict(color='#c0392b', dash='dash', width=1.5),
                        annotation_text=f"TCC target {tcc_target_val:g} ± {tcc_tol_val:g}",
                        annotation_position='top left')
        fig_c.add_trace(go.Scatter(
            x=[theta_target], y=[tcc_target_val], mode='markers', name='TCC target',
            marker=dict(symbol='diamond', size=11, color='#c0392b'),
            error_y=dict(type='constant', value=tcc_tol_val, color='#c0392b', thickness=2, width=8),
            hovertemplate="θ=%{x:.2f}: target %{y:.2f} ± " + f"{tcc_tol_val:g}<extra></extra>"))
    else:
        fig_c.add_annotation(x=theta_target, y=1, yref='paper', showarrow=False, yshift=-12,
                             text="TCC target off (0 = disabled)", font=dict(color='#7f8c8d', size=11))
    return fig_t, fig_c


def _add_form_curves(fig_t, fig_c, forms, start):
    """Add TIF/TCC curves for forms[start:] (each computed in one vectorized pass)."""
    b_of = df.set_index('item_id')['rasch_b']
    for k, form in enumerate(forms[start:], start + 1):
        tif, tcc = curves(THETA_GRID, b_of.loc[form['selected_items']].to_numpy())
        m = form['metrics']
        fig_t.add_trace(go.Scatter(x=THETA_GRID, y=tif, mode='lines', name=f"Form {k} (TIF:{m['primary_tif']:.2f})"))
        fig_c.add_trace(go.Scatter(x=THETA_GRID, y=tcc, mode='lines', name=f"Form {k} (TCC:{m['primary_tcc']:.2f})"))


def _usage_chart(used_ids):
    if not has_rasch_cat:
        return go.Figure()
    total = df['raschb_cat'].astype(str).value_counts().sort_index()
    used = df[df['item_id'].isin(used_ids)]['raschb_cat'].astype(str).value_counts().reindex(total.index, fill_value=0)
    fig = go.Figure([go.Bar(name='Bank Total', x=total.index, y=total.values, marker_color='#CBD5E1'),
                     go.Bar(name='Used', x=used.index, y=used.values, marker_color='#6c3483')])
    fig.update_layout(barmode='group', height=320, margin=dict(t=10, b=30), xaxis_title='Rasch B Category',
                      yaxis_title='Items')
    return fig


def _utilization_chart(used_ids):
    if not has_domain:
        return go.Figure()
    total = df['domain'].value_counts().sort_index()
    used = df[df['item_id'].isin(used_ids)]['domain'].value_counts().reindex(total.index, fill_value=0)
    fig = go.Figure([go.Bar(name='Used', x=total.index, y=used.values, marker_color='#1f77b4'),
                     go.Bar(name='Remaining', x=total.index, y=(total - used).values, marker_color='#E0E0E0')])
    fig.update_layout(barmode='stack', height=320, margin=dict(t=10, b=30), xaxis_title='Domain', yaxis_title='Items')
    return fig


def _overview_row(k, form):
    m = form['metrics']
    failed = sum(r['status'] != 'Pass' for r in form['audit'])
    row = {'Form': f'Form {k}', 'N Items': len(form['selected_items']), 'Solver Status': form['solver_status'],
           'Solve Order': form['strategy']}
    if has_bundles:
        row.update({'Bundles': len(form['bundles']),
                    'Bundle Items': int(df[df['item_id'].isin(form['selected_items'])]['bundle_id'].ne('').sum())})
    row.update({'Constraint Check': 'All pass' if not failed else f'{failed} failed',
                'Solve (s)': round(form['wall_time'], 3), 'Attempts': form['attempts'],
                'Mean B': round(m['mean_b'], 3), 'SD B': round(m['sd_b'], 3), 'TIF@θ': round(m['primary_tif'], 2),
                'TCC@θ': round(m['primary_tcc'], 2), 'Mean P': round(m['mean_pvalue'], 3),
                'Mean PBS': round(m['mean_pbs'], 3)})
    return row


# ==================== Real-Time Dashboard ====================

col_metrics, col_plot = st.columns([1, 2])
with col_metrics:
    st.markdown("### 🏃‍♂️ Engine Execution")
    exec_btn = st.button("🚀 Run All Forms", width='stretch', type="primary")
    step_col, reset_col = st.columns(2)
    with step_col:
        step_btn = st.button("🔬 Step: Next Form", width='stretch')
    with reset_col:
        reset_btn = st.button("🔄 Reset", width='stretch')
    status_text = st.empty()
    live_progress = st.progress(0)
    m1, m2, m3 = st.columns(3)
    metric_forms, metric_util, metric_exposure = m1.empty(), m2.empty(), m3.empty()
    metric_forms.metric("Forms Built", f"0/{n_forms}")
    metric_util.metric("Bank Utilized", "0%")
    metric_exposure.metric("Max Exposure", "0")
    pool_summary_placeholder = st.empty()
    domain_info_placeholder = st.empty()
    active_pool_stats_placeholder = st.empty()
with col_plot:
    st.markdown("### 📈 Live TIF Chart")
    plot_placeholder_tif = st.empty()
    st.markdown("### 📈 Live TCC (Expected Score)")
    plot_placeholder_tcc = st.empty()
    init_tif, init_tcc = _target_figures()
    plot_placeholder_tif.plotly_chart(init_tif, width='stretch', key='tif_init')
    plot_placeholder_tcc.plotly_chart(init_tcc, width='stretch', key='tcc_init')

st.markdown("### 📋 Form Overview (updates per iteration)")
form_stats_placeholder = st.empty()
live_chart_c1, live_chart_c2 = st.columns(2)
with live_chart_c1:
    st.markdown("### 📊 Item Usage Tracker by Rasch B Category")
    usage_chart_placeholder = st.empty()
with live_chart_c2:
    st.markdown("### 📊 Bank Utilization by Domain")
    utilization_chart_placeholder = st.empty()
st.divider()


def _new_session():
    assembler = SequentialAssembler(df, settings)
    fig_t, fig_c = _target_figures()
    st.session_state.update(cpsat_assembler=assembler, cpsat_forms=assembler.forms, cpsat_tracker=assembler.tracker,
                            cpsat_overview_rows=[], cpsat_cumulative_ids=assembler.used_ids,
                            cpsat_fig_tif=fig_t, cpsat_fig_tcc=fig_c, cpsat_drawn=0, cpsat_export=None)
    return assembler


def _draw(key, final=False):
    """Bring tables, metrics and charts up to date with the forms built so far."""
    assembler = st.session_state['cpsat_assembler']
    forms, rows = assembler.forms, st.session_state['cpsat_overview_rows']
    rows.extend(_overview_row(k, f) for k, f in enumerate(forms[len(rows):], len(rows) + 1))
    built = len(forms)
    live_progress.progress(min(built / n_forms, 1.0))
    metric_forms.metric("Forms Built", f"{built}/{n_forms}")
    stats = assembler.tracker.get_exposure_stats()
    metric_util.metric("Bank Utilized", f"{stats['utilization_pct']}%")
    metric_exposure.metric("Max Exposure", stats['max_exposure'])
    domain_info_placeholder.caption("🔒 Domain limits: " + " · ".join(
        f"{d}: ≤{lim}" for d, lim in sorted(assembler.tracker.domain_max_usage.items())))
    form_stats_placeholder.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    fig_t, fig_c = st.session_state['cpsat_fig_tif'], st.session_state['cpsat_fig_tcc']
    _add_form_curves(fig_t, fig_c, forms, st.session_state['cpsat_drawn'])
    st.session_state['cpsat_drawn'] = built
    plot_placeholder_tif.plotly_chart(fig_t, width='stretch', key=f'tif_{key}')
    plot_placeholder_tcc.plotly_chart(fig_c, width='stretch', key=f'tcc_{key}')
    usage_chart_placeholder.plotly_chart(_usage_chart(assembler.used_ids), width='stretch', key=f'usage_{key}')
    utilization_chart_placeholder.plotly_chart(_utilization_chart(assembler.used_ids), width='stretch',
                                               key=f'util_{key}')


def _on_event(kind, info):
    if kind == 'sampling':
        eligible, active = info['eligible'], info['active']
        pool_summary_placeholder.markdown(
            f"**Eligible Pool**: {len(eligible)} items (Mean B: {eligible['rasch_b'].mean():.2f}, "
            f"SD: {eligible['rasch_b'].std():.2f}) → **Active Pool**: {len(active)} items "
            f"(Mean B: {active['rasch_b'].mean():.2f}, SD: {active['rasch_b'].std():.2f})")
        status_text.info(f"🔄 Form {len(st.session_state['cpsat_forms']) + 1}/{n_forms}: sampling active pool "
                         f"({len(active)} items), attempt {info['attempt']}...")
        if 'domain' in active.columns:
            stats = active.groupby('domain')['rasch_b'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            stats.loc['**TOTAL**'] = [len(active), *np.round([active['rasch_b'].mean(), active['rasch_b'].std(),
                                                              active['rasch_b'].min(), active['rasch_b'].max()], 2)]
            stats.columns = ['Count', 'Mean B', 'SD B', 'Min B', 'Max B']
            active_pool_stats_placeholder.dataframe(stats.fillna(0.0).rename_axis('Domain').reset_index(),
                                                    width='stretch', hide_index=True)
    elif kind == 'retry':
        status_text.warning(f"⚠️ Form {len(st.session_state['cpsat_forms']) + 1}: CP-SAT status '{info['status']}', "
                            f"retrying ({info['attempt']}/{sampling_attempts})...")
    elif kind == 'form':
        form = info['form']
        status_text.success(f"✅ Form {info['index']}/{n_forms} assembled! CP-SAT: {form['solver_status']} "
                            f"({form['strategy']}" + (f", {len(form['bundles'])} bundle(s)" if has_bundles else "")
                            + ").")
        every = live_every or max(1, n_forms // 30)
        if info['index'] % every == 0 or info['index'] == n_forms:
            _draw(f"live_{info['index']}")


# ==================== Summary ====================

def _render_similarity_heatmap(forms):
    sets = [set(f['selected_items']) for f in forms]
    z = [[round(len(a & b) / len(a | b), 3) if a | b else 0 for b in sets] for a in sets]
    labels = [f"Form {i + 1}" for i in range(len(forms))]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale='Purples', zmin=0, zmax=1,
                               hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>", colorbar=dict(title="Jaccard")))
    fig.update_layout(title="Form Similarity Heatmap (Jaccard Index)", height=400, xaxis=dict(visible=False),
                      yaxis=dict(autorange='reversed', visible=False), margin=dict(t=60, b=20, l=20, r=20))
    return fig


def _bundle_order(item_ids):
    """Bank rows for a form, with each bundle's items kept together (bundles in first-appearance order)."""
    rows = df[df['item_id'].isin(item_ids)].copy()
    if 'bundle_id' in rows.columns:
        first = {}
        for pos, label in enumerate(rows['bundle_id']):
            first.setdefault(label or f'__{pos}', pos)
        rows['_order'] = [first[label] if label else pos for pos, label in enumerate(rows['bundle_id'])]
        rows = rows.sort_values('_order', kind='stable').drop(columns='_order')
    return rows


def _audit_frame(forms):
    audit = pd.DataFrame([dict(Form=f"Form {i + 1}", **r) for i, f in enumerate(forms) for r in f['audit']])
    if not audit.empty:
        audit['actual'] = audit['actual'].astype(str)   # mixed counts / text
    return audit


def _excel(forms, rows, audit):
    cached = st.session_state.get('cpsat_export')
    if cached and cached[0] == len(forms):
        return cached[1]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='Overview', index=False)
        for i, form in enumerate(forms):
            _bundle_order(form['selected_items']).to_excel(writer, sheet_name=f"Form {i + 1}", index=False)
        audit.to_excel(writer, sheet_name='Constraint Check', index=False)
    st.session_state['cpsat_export'] = (len(forms), buffer.getvalue())
    return buffer.getvalue()


def _render_summary():
    assembler = st.session_state.get('cpsat_assembler')
    if not assembler or not assembler.forms:
        return
    forms, rows, tracker = assembler.forms, st.session_state['cpsat_overview_rows'], assembler.tracker
    stats = tracker.get_exposure_stats()
    st.markdown("### 📊 Exposure Statistics")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Items Used", stats['total_used'])
    e2.metric("Pool Size", stats['total_pool'])
    e3.metric("Utilization", f"{stats['utilization_pct']}%")
    e4.metric("Mean Exposure", stats['mean_exposure'])
    if common_ids:
        st.caption(f"Common items (and the rest of their bundles) are exempt from exposure limits and appeared on "
                   f"{stats['common_exposure']} form(s); Max Exposure above counts limited items only.")
    if has_bundles:
        uses = pd.Series([b for f in forms for b in f['bundles']], dtype=str).value_counts()
        b1, b2, b3 = st.columns(3)
        b1.metric("Bundles Used", f"{len(uses)}/{len(bundles_in_bank)}")
        b2.metric("Max Bundle Exposure", int(uses.max()) if len(uses) else 0)
        b3.metric("Bundles per Form", f"{np.mean([len(f['bundles']) for f in forms]):.1f}")

    st.markdown("### ✅ Constraint Check (recomputed independently of the solver)")
    audit = _audit_frame(forms)
    failed = audit[audit['status'] != 'Pass']
    if failed.empty:
        st.success(f"All {len(audit)} checks pass across {len(forms)} form(s).")
    else:
        st.error(f"{len(failed)} check(s) failed.")
    pick = st.selectbox("Show checks for", ["Failed checks only", "All forms"] +
                        [f"Form {i + 1}" for i in range(len(forms))], key='audit_pick')
    shown = failed if pick.startswith("Failed") else audit if pick == "All forms" else audit[audit['Form'] == pick]
    st.dataframe(shown, width='stretch', hide_index=True)

    if len(forms) > 1:
        st.markdown("### 🔗 Form Overlap (Shared Item Count)")
        shown_forms = forms[:HEATMAP_LIMIT]
        if len(forms) > HEATMAP_LIMIT:
            st.caption(f"Showing the first {HEATMAP_LIMIT} of {len(forms)} forms. Every form's overlap with all "
                       "earlier forms is in the constraint check above.")
        sets = [set(f['selected_items']) for f in shown_forms]
        st.dataframe(pd.DataFrame([{f"Form {j + 1}": len(a & b) for j, b in enumerate(sets)} for a in sets],
                                  index=[f"Form {i + 1}" for i in range(len(sets))]))
        st.plotly_chart(_render_similarity_heatmap(shown_forms), width='stretch', key='heatmap_summary')

    st.markdown("### 📥 Download Forms")
    st.download_button(label=f"📥 Download {len(forms)} Form(s) as Excel", data=_excel(forms, rows, audit),
                       file_name="LOFT_CP_SAT_Forms.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                       width='stretch')


# ==================== Execute ====================

def _report(result):
    if not result['ok']:
        status_text.error(f"🛑 {result['message']}")


if reset_btn:
    _new_session()
    status_text.info("🔄 Reset complete. Ready to assemble forms.")
elif step_btn:
    assembler = st.session_state.get('cpsat_assembler') or _new_session()
    if assembler.done:
        status_text.success(f"🎉 All {n_forms} forms already assembled!")
    else:
        _report(assembler.next_form(_on_event))
        _draw('step')
elif exec_btn:
    assembler = _new_session()
    status_text.info("⚡ Initializing CP-SAT Sequential LOFT...")
    _report(assembler.run(_on_event))
    _draw('run')
elif st.session_state.get('cpsat_assembler') and st.session_state['cpsat_forms']:
    _draw('restore')
    built = len(st.session_state['cpsat_forms'])
    if built < n_forms:
        status_text.info(f"🔬 Step mode: {built}/{n_forms} forms built. Press 'Step: Next Form' to continue.")
    else:
        status_text.success(f"🎉 All {n_forms} forms assembled!")

_render_summary()
