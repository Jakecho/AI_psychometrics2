"""
LOFT_squentialATA — Sequential LOFT with CBC Solver
===================================================
Linear-on-the-Fly Testing with sequential form assembly using
CBC Solver (PuLP MIP optimization).

Implements three LOFT optimization strategies from Cho (2025):
  1. Domain-specific exposure thresholds
  2. Domain-stratified active pool sampling
  3. Auto-generated difficulty category constraints

Features:
- Sequential form assembly with item exposure tracking
- Active pool sub-sampling per form (LOFT randomization)
- Jaccard overlap rejection for form distinctness
- Real-time Streamlit dashboard with live TIF/TCC plots
- Step-by-step demo mode for one-form-at-a-time assembly
- All psychometric constraints (domain, raschb_cat, image, audio, enemy, common, excluded)
- IRT (Rasch) TIF/TCC targets and evaluation
- Form similarity heatmap and overlap matrix
- Excel export

Prerequisites:
  pip install streamlit pandas plotly openpyxl scipy pulp

Author: AI Assistant
Date: March 3, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from loft_sequential_engine import (
    rasch_probability,
    rasch_information,
    calculate_tif,
    calculate_tcc,
    auto_generate_difficulty_categories,
    generate_active_pool,
    ItemUsageTracker,
)
# CBC_ATA import is deferred to avoid module-level st.set_page_config collision
import plotly.graph_objects as go
import io

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="LOFT Sequential ATA — CBC Solver",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FAFAFA; }
    h1 { color: #6c3483; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stButton>button {
        background-color: #8e44ad; color: white; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #7d3c98; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 LOFT Sequential ATA — CBC Solver")
st.markdown(
    "*A live Dashboard for sequential Linear-on-the-Fly Test assembly "
    "using CBC Solver (PuLP MIP optimization). All processing runs locally.*"
)
st.caption(
    "📚 Based on: Cho (2025). _Optimizing LOFT Test Assembly: "
    "Strategies for Exposure and Form Diversity._ Credentialing Insights."
)

# ==================== 1. File Upload ====================

st.sidebar.header("📁 1. Master Pool")
uploaded_file = st.sidebar.file_uploader(
    "Upload Item Bank (CSV or Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Required columns: item_id, domain, rasch_b, pvalue, point_biserial. "
         "Optional: raschb_cat, enemy_ids, has_image, has_audio"
)

if not uploaded_file:
    st.info("👋 Upload a CSV or Excel item bank to begin.")

    with st.expander("📋 Example File Format"):
        example_df = pd.DataFrame({
            'item_id': ['NCX0001', 'NCX0002', 'NCX0003', 'NCX0004', 'NCX0005'],
            'domain': ['Cardiology', 'Cardiology', 'Pharmacology', 'Med-Surg', 'Med-Surg'],
            'rasch_b': [0.15, -0.32, 0.45, -0.10, 0.22],
            'pvalue': [0.62, 0.75, 0.55, 0.68, 0.60],
            'point_biserial': [0.35, 0.42, 0.38, 0.40, 0.36],
            'raschb_cat': ['6. hard', '3. easy', '6. hard', '4. moderately easy', '6. hard'],
            'enemy_ids': ['NCX0002', '', 'NCX0004', 'NCX0003', ''],
            'has_image': [1, 0, 1, 1, 0]
        })
        st.dataframe(example_df)
        csv_example = example_df.to_csv(index=False)
        st.download_button(
            "📥 Download Example CSV", data=csv_example,
            file_name="item_pool_example.csv", mime="text/csv"
        )

    st.stop()

# Load
if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(('.xlsx', '.xls')):
    df = pd.read_excel(uploaded_file)
else:
    st.error("Unsupported file format.")
    st.stop()

# Validate required columns
required_cols = ['item_id', 'domain', 'rasch_b', 'pvalue', 'point_biserial']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

df['item_id'] = df['item_id'].astype(str)
df['domain'] = df['domain'].fillna('Unspecified')
if 'raschb_cat' in df.columns:
    df['raschb_cat'] = df['raschb_cat'].fillna('Unspecified')

st.sidebar.success(f"✅ Bank Loaded: {len(df)} items")

# Detect features
has_enemies = 'enemy_ids' in df.columns
has_testlets = 'testlet_id' in df.columns
has_image = 'has_image' in df.columns
has_audio = 'has_audio' in df.columns
has_rasch_cat = 'raschb_cat' in df.columns
has_domain = 'domain' in df.columns

features = []
if has_domain:
    features.append(f"Domains ({df['domain'].nunique()})")
if has_enemies:
    features.append("Enemy Items")
if has_testlets:
    features.append("Testlets")
if has_image:
    features.append("Images")
if has_audio:
    features.append("Audio")
if has_rasch_cat:
    features.append(f"Difficulty Buckets ({df['raschb_cat'].nunique()})")
st.sidebar.caption(
    "Features: " + " · ".join(features) if features else "No special columns detected."
)

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
    st.dataframe(domain_stats, use_container_width=True)

if has_rasch_cat:
    st.markdown("**Difficulty Bucket Distribution:**")
    bucket_stats = df.groupby(df['raschb_cat'].astype(str))['rasch_b'].agg(['count', 'mean']).round(2)
    bucket_stats.columns = ['Count', 'Mean B']
    bucket_stats.index.name = 'Bucket'
    st.dataframe(bucket_stats, use_container_width=True)

st.divider()

# ==================== 2. Engine ====================

st.sidebar.header("🤖 2. Engine")
st.sidebar.caption("⚡ CBC Solver — pure MIP optimization, no LLM required.")
st.sidebar.divider()

# ==================== 3. Constraints ====================

st.sidebar.header("⚙️ 3. Objectives & Constraints")

with st.sidebar.expander("Form Specifications", expanded=True):
    c1, c2 = st.columns(2)
    n_forms = c1.number_input("# Forms", 1, 10000, 3)
    test_len = c2.number_input("Test Length", 1, 500, 10)

    c3, c4 = st.columns(2)
    multiplier = c3.number_input("Pool Multiplier", 2, 10, 5)
    exposure_max = c4.number_input("Global Exposure Limit", 1, 1000, 2)

    max_overlap = st.slider("Max Jaccard Overlap", 0.0, 1.0, 0.3, 0.05)
    apply_enemies = st.checkbox(
        "Apply Enemy Item Constraints",
        value=has_enemies, disabled=not has_enemies
    )

# ─── Strategy 1: Domain-Specific Exposure ───
with st.sidebar.expander("📌 Domain Exposure Limits (Strategy 1)", expanded=False):
    st.caption(
        "Domains with fewer items may need higher exposure limits "
        "to avoid pool exhaustion. Enable auto-compute or set manually."
    )
    auto_domain_limits = st.checkbox(
        "Auto-compute domain limits", value=True,
        help="Automatically set per-domain exposure limits based on pool depth "
             "(Cho, 2025). Domains with fewer items get higher limits."
    )

    domain_exposure_payload = {}
    if not auto_domain_limits and has_domain:
        domains_list = sorted(df['domain'].dropna().unique().tolist())
        for dom in domains_list:
            dom_count = len(df[df['domain'] == dom])
            exp_val = st.number_input(
                f"{dom} ({dom_count} items)", 1, 1000, exposure_max,
                key=f"exp_{dom}"
            )
            domain_exposure_payload[dom] = exp_val

with st.sidebar.expander("Targets & Tolerances", expanded=True):
    c5, c6 = st.columns(2)
    theta_target = c5.number_input("Theta Target (θ)", -3.0, 3.0, 0.0, 0.1)
    tif_target = c6.number_input("TIF Target", 0.0, 100.0, 2.0, 0.5)

    c7, c8 = st.columns(2)
    mean_diff = c7.number_input("Mean B Target", -3.0, 3.0, 0.0, 0.1)
    mean_tol = c8.number_input("Mean B Tol", 0.0, 3.0, 0.2, 0.1)

    c9, c10 = st.columns(2)
    tif_tol_val = c9.number_input("TIF Tol (+)", 0.0, 10.0, 0.2, 0.1)
    tcc_tol_val = c10.number_input("TCC Tol (+/-)", 0.0, 5.0, 0.5, 0.1)

    tcc_target_val = st.number_input(
        "TCC Target (Expected Score)", value=0.0, step=1.0,
        help="0.0 = Disabled"
    )

with st.sidebar.expander("Dynamic Domains", expanded=False):
    domain_payload = {}
    if has_domain:
        domains_list = sorted(df['domain'].dropna().unique().tolist())
        for dom in domains_list:
            st.markdown(f"**{dom}** ({len(df[df['domain'] == dom])} items)")
            d1, d2 = st.columns(2)
            d_min = d1.number_input("Min", 0, test_len, 0, key=f"dmin_{dom}")
            d_max = d2.number_input("Max", 0, test_len, 0, key=f"dmax_{dom}")
            if d_min > 0 or d_max > 0:
                domain_payload[dom] = {
                    'min': d_min,
                    'max': d_max if d_max > 0 else test_len
                }
    else:
        st.info("No 'domain' column found in bank.")

with st.sidebar.expander("Media Constraints", expanded=False):
    c_items = st.text_input("Common Item IDs (comma-separated)", "")

    st.markdown("**Excluded Items**")
    excluded_str = st.text_input("Excluded Item IDs (comma-separated)", "",
                                 key="excluded_items_input")

    st.markdown("**Images**")
    im1, im2 = st.columns(2)
    img_min = im1.number_input("Min Image", 0, test_len, 0)
    img_max = im2.number_input("Max Image", 0, test_len, 0)

    st.markdown("**Audio**")
    au1, au2 = st.columns(2)
    audio_min = au1.number_input("Min Audio", 0, test_len, 0)
    audio_max = au2.number_input("Max Audio", 0, test_len, 0)

# ─── Strategy 3: Auto Difficulty Bins ───
with st.sidebar.expander("📊 Difficulty Bins (Strategy 3)", expanded=False):
    st.caption(
        "Auto-generate difficulty category constraints to maximize "
        "item pool utilization (Cho, 2025). Prevents the engine from "
        "always selecting items near the cut score."
    )
    auto_diff_bins = st.checkbox(
        "Enable auto-difficulty bins", value=True,
        help="Divide items into 7 difficulty bins based on standard normal "
             "density and impose min/max per bin."
    )
    n_diff_bins = st.number_input(
        "Number of bins", 3, 11, 7, step=2,
        help="Odd number recommended (e.g. 5, 7, 9). "
             "7 bins centered at [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5]",
        disabled=not auto_diff_bins,
    )

    # Preview auto-generated bins
    if auto_diff_bins:
        preview_cats = auto_generate_difficulty_categories(
            df, test_len, n_categories=n_diff_bins
        )
        if preview_cats:
            preview_rows = []
            for label, info in preview_cats.items():
                lo, hi = info['range']
                preview_rows.append({
                    'Bin': label,
                    'Range': f"[{lo:.1f}, {hi:.1f})",
                    'Pool': info['pool_count'],
                    'Target': info['target'],
                    'Min': info['min'],
                    'Max': info['max'],
                })
            st.dataframe(
                pd.DataFrame(preview_rows),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No bins generated (check rasch_b column).")

with st.sidebar.expander("Difficulty Buckets (raschb_cat)", expanded=False):
    rasch_cats = {}
    if has_rasch_cat:
        cat_list = sorted(df['raschb_cat'].astype(str).unique().tolist())
        for cat in cat_list:
            st.markdown(f"**{cat}** ({len(df[df['raschb_cat'].astype(str) == cat])} items)")
            b1, b2 = st.columns(2)
            b_min = b1.number_input("Min", 0, test_len, 0, key=f"bmin_{cat}")
            b_max = b2.number_input("Max", 0, test_len, 0, key=f"bmax_{cat}")
            if b_min > 0 or b_max > 0:
                rasch_cats[cat] = {
                    'min': b_min,
                    'max': b_max if b_max > 0 else test_len
                }
    else:
        st.info("No 'raschb_cat' column in bank.")

# ==================== Build Rules Payload ====================

rules_payload = {
    'n_forms': n_forms,
    'test_length': test_len,
    'multiplier': multiplier,
    'exposure_global_max': exposure_max,
    'theta_targets': [theta_target - 1, theta_target, theta_target + 1],
    'min_tif_targets': [max(tif_target - 0.5, 0), tif_target, max(tif_target - 0.5, 0)],
    'mean_difficulty_target': mean_diff,
    'mean_difficulty_tolerance': mean_tol,
    'max_overlap_threshold': max_overlap,
    'apply_enemies': apply_enemies,
    # Strategy 1
    'auto_domain_limits': auto_domain_limits,
    # Strategy 3
    'auto_difficulty_bins': auto_diff_bins,
    'n_difficulty_bins': n_diff_bins,
}

# Domain-specific exposure limits (Strategy 1 - manual mode)
if not auto_domain_limits and domain_exposure_payload:
    rules_payload['domain_exposure_limits'] = domain_exposure_payload

if domain_payload:
    rules_payload['domain_constraints'] = domain_payload
if tcc_target_val > 0.0:
    rules_payload['tcc_targets'] = [None, tcc_target_val, None]
    rules_payload['tcc_tolerances'] = [100.0, tcc_tol_val, 100.0]
if tif_tol_val > 0.0:
    rules_payload['tif_tolerances'] = [100.0, tif_tol_val, 100.0]
if c_items:
    rules_payload['common_items'] = [x.strip() for x in c_items.split(',') if x.strip()]
if excluded_str:
    rules_payload['excluded_items'] = [x.strip() for x in excluded_str.split(',') if x.strip()]
if img_min > 0 or img_max > 0:
    rules_payload['image_constraint'] = {
        'min': img_min, 'max': img_max if img_max > 0 else test_len
    }
if audio_min > 0 or audio_max > 0:
    rules_payload['audio_constraint'] = {
        'min': audio_min, 'max': audio_max if audio_max > 0 else test_len
    }
if rasch_cats:
    rules_payload['raschb_cat_constraints'] = rasch_cats

# ==================== Real-Time Dashboard ====================

col_metrics, col_plot = st.columns([1, 2])

with col_metrics:
    st.markdown("### 🏃‍♂️ Engine Execution")
    exec_btn = st.button(
        "🚀 Run All Forms", use_container_width=True, type="primary"
    )
    # Step-by-step demo buttons
    step_col, reset_col = st.columns(2)
    with step_col:
        step_btn = st.button("🔬 Step: Next Form", use_container_width=True)
    with reset_col:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    status_text = st.empty()
    live_progress = st.progress(0)

    m1, m2, m3 = st.columns(3)
    metric_forms = m1.empty()
    metric_util = m2.empty()
    metric_exposure = m3.empty()

    metric_forms.metric("Forms Built", f"0/{n_forms}")
    metric_util.metric("Bank Utilized", "0%")
    metric_exposure.metric("Max Exposure", "0")

    # Active pool summary placeholder
    pool_summary_placeholder = st.empty()
    # Domain exposure info placeholder
    domain_info_placeholder = st.empty()
    # Active pool statistics placeholder
    active_pool_stats_placeholder = st.empty()

with col_plot:
    st.markdown("### 📈 Live TIF Chart")
    plot_placeholder_tif = st.empty()

    st.markdown("### 📈 Live TCC (Expected Score)")
    plot_placeholder_tcc = st.empty()

    fig_tif = go.Figure()
    fig_tif.update_layout(
        title="Test Information Functions (TIF)",
        xaxis_title="Theta (θ)", yaxis_title="Information", height=400
    )
    plot_placeholder_tif.plotly_chart(fig_tif, use_container_width=True, key='tif_init')

    fig_tcc = go.Figure()
    fig_tcc.update_layout(
        title="Test Characteristic Curve (TCC)",
        xaxis_title="Theta (θ)", yaxis_title="Expected Score", height=400
    )
    plot_placeholder_tcc.plotly_chart(fig_tcc, use_container_width=True, key='tcc_init')

# ---- Live Iteration Placeholders (updated per form) ----
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


# ==================== Shared Helpers ====================

def _render_similarity_heatmap(forms_list):
    """Build a Plotly annotated heatmap of Jaccard similarity between forms."""
    n = len(forms_list)
    labels = [f"Form {i + 1}" for i in range(n)]
    z = []
    for i, f1 in enumerate(forms_list):
        row = []
        set1 = set(str(x) for x in f1['selected_items'])
        for j, f2 in enumerate(forms_list):
            set2 = set(str(x) for x in f2['selected_items'])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            row.append(round(intersection / union, 3) if union > 0 else 0)
        z.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale='Purples', zmin=0, zmax=1,
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Jaccard"),
        showscale=True,
    ))
    fig.update_layout(
        title="Form Similarity Heatmap (Jaccard Index)",
        height=max(350, 50 * n + 150),
        xaxis=dict(visible=False),
        yaxis=dict(autorange='reversed', visible=False),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig

# ==================== Execute ====================

is_run_all = exec_btn
is_step = step_btn
is_reset = reset_btn

if is_run_all or is_step or is_reset:

    # ── Helper: build live usage/utilization charts ──
    def _render_usage_chart(cumulative_ids, item_bank):
        """Build grouped bar chart: bank total vs cumulative used, by raschb_cat."""
        if 'raschb_cat' not in item_bank.columns:
            return go.Figure()
        cat_total = item_bank['raschb_cat'].astype(str).value_counts().sort_index()
        df_used = item_bank[item_bank['item_id'].astype(str).isin(cumulative_ids)]
        cat_used = df_used['raschb_cat'].astype(str).value_counts().reindex(cat_total.index, fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Bank Total', x=cat_total.index, y=cat_total.values, marker_color='#CBD5E1'))
        fig.add_trace(go.Bar(name='Used', x=cat_used.index, y=cat_used.values, marker_color='#6c3483'))
        fig.update_layout(barmode='group', height=320, margin=dict(t=10, b=30), xaxis_title='Rasch B Category', yaxis_title='Items')
        return fig

    def _render_utilization_chart(cumulative_ids, item_bank):
        """Build stacked bar: used / remaining per domain."""
        if 'domain' not in item_bank.columns:
            return go.Figure()
        dom_total = item_bank['domain'].value_counts().sort_index()
        df_used = item_bank[item_bank['item_id'].astype(str).isin(cumulative_ids)]
        dom_used = df_used['domain'].value_counts().reindex(dom_total.index, fill_value=0)
        dom_remaining = dom_total - dom_used
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Used', x=dom_total.index, y=dom_used.values, marker_color='#1f77b4'))
        fig.add_trace(go.Bar(name='Remaining', x=dom_total.index, y=dom_remaining.values, marker_color='#E0E0E0'))
        fig.update_layout(barmode='stack', height=320, margin=dict(t=10, b=30), xaxis_title='Domain', yaxis_title='Items')
        return fig


    # ── CBC Solver Path (sequential: sample → solve 1 form → track → repeat) ──
    # Deferred import to avoid module-level Streamlit side effects in CBC_ATA.py
    from CBC_ATA import assemble_forms_with_cbc

    # Build base config dict for CBC solver
    cbc_config = {
        'test_length': test_len,
        'approach': 'IRT (Rasch)',
        'domain_constraints': domain_payload,
        'raschb_cat_constraints': rasch_cats,
        'enemy_check': apply_enemies,
        'eval_points': {
            'theta_low': theta_target - 1,
            'theta_mid': theta_target,
            'theta_high': theta_target + 1,
            'tif_low': max(tif_target - 0.5, 0),
            'tif_mid': tif_target,
            'tif_high': max(tif_target - 0.5, 0),
            'tcc_mid': tcc_target_val if tcc_target_val > 0 else 0.0,
            'tcc_enabled': tcc_target_val > 0,
            'mean_rasch_enabled': True,
            'mean_rasch_target': mean_diff,
            'mean_rasch_tolerance': mean_tol,
        },
        'tif_tolerance': {
            'tif': tif_tol_val,
            'tcc': tcc_tol_val,
        },
        'mean_difficulty_target': mean_diff,
        'difficulty_tolerance': mean_tol,
        'apply_mean_diff': True,
        'mean_diff_target': mean_diff,
        'mean_diff_tolerance': mean_tol,
    }
    # Optional constraints
    if c_items:
        cbc_config['common_items'] = [x.strip() for x in c_items.split(',') if x.strip()]
    if excluded_str:
        cbc_config['excluded_items'] = [x.strip() for x in excluded_str.split(',') if x.strip()]
    if img_min > 0 or img_max > 0:
        cbc_config['image_constraint'] = {
            'min': img_min, 'max': img_max if img_max > 0 else test_len, 'enabled': True
        }
    if audio_min > 0 or audio_max > 0:
        cbc_config['audio_constraint'] = {
            'min': audio_min, 'max': audio_max if audio_max > 0 else test_len, 'enabled': True
        }

    # ---- Helper: assemble ONE form and update state ----
    def _cbc_assemble_one_form(tracker, all_forms, overview_rows, cumulative_used_ids,
                               fig_tif, fig_tcc, x_vals, max_overlap, max_retries=100):
        """Run one iteration of the CBC sequential loop. Returns True on success."""
        form_idx = len(all_forms)
        failed_attempts = 0

        while failed_attempts < max_retries:
            eligible_pool = tracker.get_eligible_pool()
            if len(eligible_pool) < test_len:
                status_text.error(
                    f"🛑 Eligible pool exhausted! {len(eligible_pool)} items remain, need {test_len}."
                )
                return False

            active_pool = generate_active_pool(eligible_pool, rules_payload)
            pool_summary_placeholder.markdown(
                f"**Eligible Pool**: {len(eligible_pool)} items "
                f"(Mean B: {eligible_pool['rasch_b'].mean():.2f}, "
                f"SD: {eligible_pool['rasch_b'].std():.2f}) → "
                f"**Active Pool**: {len(active_pool)} items "
                f"(Mean B: {active_pool['rasch_b'].mean():.2f}, "
                f"SD: {active_pool['rasch_b'].std():.2f})"
            )

            # Build active pool statistics summary
            _ap_stats_rows = []
            if 'domain' in active_pool.columns:
                for dom in sorted(active_pool['domain'].unique()):
                    dom_slice = active_pool[active_pool['domain'] == dom]
                    _ap_stats_rows.append({
                        'Domain': dom,
                        'Count': len(dom_slice),
                        'Mean B': round(dom_slice['rasch_b'].mean(), 2),
                        'SD B': round(dom_slice['rasch_b'].std(), 2) if len(dom_slice) > 1 else 0.0,
                        'Min B': round(dom_slice['rasch_b'].min(), 2),
                        'Max B': round(dom_slice['rasch_b'].max(), 2),
                    })
                # Add total row
                _ap_stats_rows.append({
                    'Domain': '**TOTAL**',
                    'Count': len(active_pool),
                    'Mean B': round(active_pool['rasch_b'].mean(), 2),
                    'SD B': round(active_pool['rasch_b'].std(), 2),
                    'Min B': round(active_pool['rasch_b'].min(), 2),
                    'Max B': round(active_pool['rasch_b'].max(), 2),
                })
            active_pool_stats_placeholder.dataframe(
                pd.DataFrame(_ap_stats_rows),
                use_container_width=True, hide_index=True
            )
            status_text.info(f"🔄 Form {form_idx+1}/{n_forms}: Sampling active pool ({len(active_pool)} items)...")

            try:
                result = assemble_forms_with_cbc(active_pool, cbc_config, 1)
            except Exception as e:
                status_text.error(f"🛑 CBC Solver Error on Form {form_idx+1}: {e}")
                return False

            cbc_status = result.get('status', 'Unknown')
            if cbc_status != 'Optimal':
                failed_attempts += 1
                status_text.warning(
                    f"⚠️ Form {form_idx+1}: CBC status '{cbc_status}', retrying ({failed_attempts}/{max_retries})..."
                )
                continue

            selected_forms_result = result.get('selected_forms', [[]])
            form_ids = selected_forms_result[0] if selected_forms_result else []
            if not form_ids:
                failed_attempts += 1
                continue

            # Jaccard overlap check
            new_set = set(str(x) for x in form_ids)
            overlap_rejected = False
            for past_form in all_forms:
                past_set = set(str(x) for x in past_form['selected_items'])
                intersection = len(new_set & past_set)
                union = len(new_set | past_set)
                jaccard = intersection / union if union > 0 else 0
                if jaccard > max_overlap:
                    overlap_rejected = True
                    break

            if overlap_rejected:
                failed_attempts += 1
                if failed_attempts >= max_retries:
                    status_text.warning(f"⚠️ Form {form_idx+1}: accepting despite overlap (retries exhausted).")
                else:
                    status_text.warning(f"⚠️ Form {form_idx+1}: high overlap, retrying ({failed_attempts}/{max_retries})...")
                    continue

            # Record usage
            str_ids = [str(x) for x in form_ids]
            tracker.record_usage(str_ids)
            cumulative_used_ids.update(str_ids)

            # Compute metrics
            subset = df[df['item_id'].astype(str).isin(str_ids)]
            b_array = subset['rasch_b'].values
            primary_tif = calculate_tif(theta_target, b_array) if len(b_array) > 0 else 0
            primary_tcc = calculate_tcc(theta_target, b_array) if len(b_array) > 0 else 0
            form_dict = {
                'selected_items': form_ids,
                'metrics': {
                    'mean_b': float(subset['rasch_b'].mean()) if len(subset) > 0 else 0,
                    'sd_b': float(subset['rasch_b'].std()) if len(subset) > 0 else 0,
                    'primary_tif': float(primary_tif),
                    'primary_tcc': float(primary_tcc),
                    'mean_pvalue': float(subset['pvalue'].mean()) if 'pvalue' in subset.columns and len(subset) > 0 else 0,
                    'mean_pbs': float(subset['point_biserial'].mean()) if 'point_biserial' in subset.columns and len(subset) > 0 else 0,
                }
            }
            all_forms.append(form_dict)
            m = form_dict['metrics']
            fi = len(all_forms)  # 1-indexed form number

            # Update dashboard
            status_text.success(f"✅ Form {fi}/{n_forms} assembled!")
            live_progress.progress(min(fi / n_forms, 1.0))
            metric_forms.metric("Forms Built", f"{fi}/{n_forms}")
            exp_stats = tracker.get_exposure_stats()
            metric_util.metric("Bank Utilized", f"{exp_stats['utilization_pct']}%")
            metric_exposure.metric("Max Exposure", exp_stats['max_exposure'])

            overview_rows.append({
                'Form': f'Form {fi}', 'N Items': len(form_ids),
                'Mean B': round(m['mean_b'], 3), 'SD B': round(m['sd_b'], 3),
                'TIF@θ': round(m['primary_tif'], 2), 'TCC@θ': round(m['primary_tcc'], 2),
                'Mean P': round(m['mean_pvalue'], 3), 'Mean PBS': round(m['mean_pbs'], 3),
            })
            form_stats_placeholder.dataframe(
                pd.DataFrame(overview_rows), use_container_width=True, hide_index=True
            )

            y_tif = [sum(rasch_information(v, b) for b in b_array) for v in x_vals]
            y_tcc = [sum(rasch_probability(v, b) for b in b_array) for v in x_vals]
            fig_tif.add_trace(go.Scatter(x=x_vals, y=y_tif, mode='lines',
                                         name=f"Form {fi} (TIF:{m['primary_tif']:.1f})"))
            fig_tcc.add_trace(go.Scatter(x=x_vals, y=y_tcc, mode='lines',
                                         name=f"Form {fi} (Len:{len(b_array)})"))
            plot_placeholder_tif.plotly_chart(fig_tif, use_container_width=True, key=f'tif_live_{fi}')
            plot_placeholder_tcc.plotly_chart(fig_tcc, use_container_width=True, key=f'tcc_live_{fi}')

            usage_chart_placeholder.plotly_chart(
                _render_usage_chart(cumulative_used_ids, df), use_container_width=True, key=f'usage_live_{fi}'
            )
            utilization_chart_placeholder.plotly_chart(
                _render_utilization_chart(cumulative_used_ids, df), use_container_width=True, key=f'util_live_{fi}'
            )
            return True

        # All retries exhausted
        status_text.error(f"🛑 Failed to assemble Form {form_idx+1} after {max_retries} retries.")
        return False

    # ---- Helper: initialize or get session state ----
    def _init_cbc_state():
        """Create fresh CBC session state."""
        st.session_state['cbc_tracker'] = ItemUsageTracker(
            df,
            global_max=rules_payload.get('exposure_global_max', 2),
            domain_max=rules_payload.get('domain_exposure_limits', None),
            auto_domain_limits=rules_payload.get('auto_domain_limits', True),
            n_forms=n_forms,
            test_length=test_len,
        )
        st.session_state['cbc_forms'] = []
        st.session_state['cbc_overview_rows'] = []
        st.session_state['cbc_cumulative_ids'] = set()
        fig_t = go.Figure()
        fig_t.update_layout(title="Test Information Functions (TIF)",
                            xaxis_title="Theta (θ)", yaxis_title="Information", height=400)
        fig_c = go.Figure()
        fig_c.update_layout(title="Test Characteristic Curve (TCC)",
                            xaxis_title="Theta (θ)", yaxis_title="Expected Score", height=400)
        st.session_state['cbc_fig_tif'] = fig_t
        st.session_state['cbc_fig_tcc'] = fig_c


    # ==== Handle Reset ====
    if is_reset:
        _init_cbc_state()
        status_text.info("🔄 Reset complete. Ready to assemble forms.")

    # ==== Handle Step Button ====
    elif is_step:
        # Initialize state if first step
        if 'cbc_tracker' not in st.session_state:
            _init_cbc_state()

        tracker = st.session_state['cbc_tracker']
        all_forms = st.session_state['cbc_forms']
        overview_rows = st.session_state['cbc_overview_rows']
        cumulative_used_ids = st.session_state['cbc_cumulative_ids']
        fig_tif = st.session_state['cbc_fig_tif']
        fig_tcc = st.session_state['cbc_fig_tcc']
        x_vals = np.linspace(-3, 3, 100)
        max_overlap = rules_payload.get('max_overlap_threshold', 0.3)

        if len(all_forms) >= n_forms:
            status_text.success(f"🎉 All {n_forms} forms already assembled!")
        else:
            domain_info_placeholder.caption(
                f"🔒 Domain limits: "
                + " · ".join([f"{d}: ≤{l}" for d, l in sorted(tracker.domain_max_usage.items())])
            )
            _cbc_assemble_one_form(
                tracker, all_forms, overview_rows, cumulative_used_ids,
                fig_tif, fig_tcc, x_vals, max_overlap
            )

    # ==== Handle Run All ====
    elif is_run_all:
        _init_cbc_state()
        tracker = st.session_state['cbc_tracker']
        all_forms = st.session_state['cbc_forms']
        overview_rows = st.session_state['cbc_overview_rows']
        cumulative_used_ids = st.session_state['cbc_cumulative_ids']
        fig_tif = st.session_state['cbc_fig_tif']
        fig_tcc = st.session_state['cbc_fig_tcc']
        x_vals = np.linspace(-3, 3, 100)
        max_overlap = rules_payload.get('max_overlap_threshold', 0.3)

        status_text.info("⚡ Initializing CBC Sequential LOFT...")
        domain_info_placeholder.caption(
            f"🔒 Domain limits: "
            + " · ".join([f"{d}: ≤{l}" for d, l in sorted(tracker.domain_max_usage.items())])
        )

        for _ in range(n_forms):
            success = _cbc_assemble_one_form(
                tracker, all_forms, overview_rows, cumulative_used_ids,
                fig_tif, fig_tcc, x_vals, max_overlap
            )
            if not success:
                break

    # ── Final Summary (shown after Run All or final Step) ──
    _final_forms = st.session_state.get('cbc_forms', [])
    _final_trk = st.session_state.get('cbc_tracker')
    _final_rows = st.session_state.get('cbc_overview_rows', [])

    if len(_final_forms) >= 1:
        # Exposure Statistics
        if _final_trk:
            exp_stats = _final_trk.get_exposure_stats()
            st.markdown("### 📊 Exposure Statistics")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Items Used", exp_stats.get('total_used', 0))
            e2.metric("Pool Size", exp_stats.get('total_pool', 0))
            e3.metric("Utilization", f"{exp_stats.get('utilization_pct', 0)}%")
            e4.metric("Mean Exposure", exp_stats.get('mean_exposure', 0))

        # Overlap Table + Heatmap
        if len(_final_forms) > 1:
            st.markdown("### 🔗 Form Overlap (Shared Item Count)")
            matrix = []
            for i, f1 in enumerate(_final_forms):
                row = {}
                set1 = set(str(x) for x in f1['selected_items'])
                for j, f2 in enumerate(_final_forms):
                    set2 = set(str(x) for x in f2['selected_items'])
                    row[f"Form {j + 1}"] = len(set1 & set2)
                matrix.append(row)
            st.dataframe(pd.DataFrame(
                matrix,
                index=[f"Form {i + 1}" for i in range(len(_final_forms))]
            ))
            st.plotly_chart(
                _render_similarity_heatmap(_final_forms), use_container_width=True, key='heatmap_summary'
            )

        # Excel Download
        st.markdown("### 📥 Download Forms")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            pd.DataFrame(_final_rows).to_excel(writer, sheet_name='Overview', index=False)
            for i, f_dict in enumerate(_final_forms):
                f_items = df[df['item_id'].astype(str).isin(
                    [str(x) for x in f_dict['selected_items']]
                )]
                f_items.to_excel(writer, sheet_name=f"Form {i + 1}", index=False)
        st.download_button(
            label=f"📥 Download {len(_final_forms)} Form(s) as Excel",
            data=excel_buffer.getvalue(),
            file_name="LOFT_CBC_Forms.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )


# ==================== Restore CBC Step Mode State on Page Reload ====================
if not is_run_all and not is_step and not is_reset:
    if 'cbc_forms' in st.session_state and st.session_state['cbc_forms']:
        _forms = st.session_state['cbc_forms']
        _rows = st.session_state.get('cbc_overview_rows', [])
        _cids = st.session_state.get('cbc_cumulative_ids', set())
        _trk = st.session_state.get('cbc_tracker')
        fi = len(_forms)

        live_progress.progress(min(fi / n_forms, 1.0))
        metric_forms.metric("Forms Built", f"{fi}/{n_forms}")
        if _trk:
            es = _trk.get_exposure_stats()
            metric_util.metric("Bank Utilized", f"{es['utilization_pct']}%")
            metric_exposure.metric("Max Exposure", es['max_exposure'])
            domain_info_placeholder.caption(
                f"🔒 Domain limits: "
                + " · ".join([f"{d}: ≤{l}" for d, l in sorted(_trk.domain_max_usage.items())])
            )
        if _rows:
            form_stats_placeholder.dataframe(
                pd.DataFrame(_rows), use_container_width=True, hide_index=True
            )
        if fi < n_forms:
            status_text.info(f"🔬 Step mode: {fi}/{n_forms} forms built. Press 'Step: Next Form' to continue.")
        else:
            status_text.success(f"🎉 All {n_forms} forms assembled!")

        ftif = st.session_state.get('cbc_fig_tif')
        ftcc = st.session_state.get('cbc_fig_tcc')
        if ftif:
            plot_placeholder_tif.plotly_chart(ftif, use_container_width=True, key='tif_restore')
        if ftcc:
            plot_placeholder_tcc.plotly_chart(ftcc, use_container_width=True, key='tcc_restore')
        if _cids:
            # Need helpers defined - inline them here
            if 'raschb_cat' in df.columns:
                cat_total = df['raschb_cat'].astype(str).value_counts().sort_index()
                df_used = df[df['item_id'].astype(str).isin(_cids)]
                cat_used = df_used['raschb_cat'].astype(str).value_counts().reindex(cat_total.index, fill_value=0)
                cat_remaining = cat_total - cat_used
                fig_u = go.Figure()
                fig_u.add_trace(go.Bar(name='Bank Total', x=cat_total.index, y=cat_total.values, marker_color='#CBD5E1'))
                fig_u.add_trace(go.Bar(name='Used', x=cat_used.index, y=cat_used.values, marker_color='#6c3483'))
                fig_u.update_layout(barmode='group', height=320, margin=dict(t=10, b=30), xaxis_title='Rasch B Category', yaxis_title='Items')
                usage_chart_placeholder.plotly_chart(fig_u, use_container_width=True, key='usage_restore')
            if 'domain' in df.columns:
                dom_total = df['domain'].value_counts().sort_index()
                df_used_dom = df[df['item_id'].astype(str).isin(_cids)]
                dom_used = df_used_dom['domain'].value_counts().reindex(dom_total.index, fill_value=0)
                dom_remaining = dom_total - dom_used
                fig_ut = go.Figure()
                fig_ut.add_trace(go.Bar(name='Used', x=dom_total.index, y=dom_used.values, marker_color='#1f77b4'))
                fig_ut.add_trace(go.Bar(name='Remaining', x=dom_total.index, y=dom_remaining.values, marker_color='#E0E0E0'))
                fig_ut.update_layout(barmode='stack', height=320, margin=dict(t=10, b=30), xaxis_title='Domain', yaxis_title='Items')
                utilization_chart_placeholder.plotly_chart(fig_ut, use_container_width=True, key='util_restore')

        # Show summary (grows as forms are assembled)
        if fi >= 1:
            if _trk:
                exp_stats = _trk.get_exposure_stats()
                st.markdown("### 📊 Exposure Statistics")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Items Used", exp_stats.get('total_used', 0))
                e2.metric("Pool Size", exp_stats.get('total_pool', 0))
                e3.metric("Utilization", f"{exp_stats.get('utilization_pct', 0)}%")
                e4.metric("Mean Exposure", exp_stats.get('mean_exposure', 0))

            if len(_forms) > 1:
                st.markdown("### 🔗 Form Overlap (Shared Item Count)")
                matrix = []
                for i, f1 in enumerate(_forms):
                    row = {}
                    set1 = set(str(x) for x in f1['selected_items'])
                    for j, f2 in enumerate(_forms):
                        set2 = set(str(x) for x in f2['selected_items'])
                        row[f"Form {j + 1}"] = len(set1 & set2)
                    matrix.append(row)
                st.dataframe(pd.DataFrame(
                    matrix,
                    index=[f"Form {i + 1}" for i in range(len(_forms))]
                ))
                st.plotly_chart(
                    _render_similarity_heatmap(_forms), use_container_width=True, key='heatmap_restore'
                )

            st.markdown("### 📥 Download Forms")
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                pd.DataFrame(_rows).to_excel(writer, sheet_name='Overview', index=False)
                for i, f_dict in enumerate(_forms):
                    f_items = df[df['item_id'].astype(str).isin(
                        [str(x) for x in f_dict['selected_items']]
                    )]
                    f_items.to_excel(writer, sheet_name=f"Form {i + 1}", index=False)

            st.download_button(
                label=f"📥 Download {len(_forms)} Form(s) as Excel",
                data=excel_buffer.getvalue(),
                file_name="LOFT_CBC_Forms.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
