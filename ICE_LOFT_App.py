import streamlit as st
import pandas as pd
import json
from ai_ata_engine import sequential_loft_assembly, rasch_information, rasch_probability
import plotly.graph_objects as go
import numpy as np
import io

st.set_page_config(page_title="AI LOFT Monitor V2", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #FAFAFA; }
    h1 { color: #1E3A8A; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stButton>button { background-color: #3B82F6; color: white; border-radius: 8px; font-weight: 600; padding: 0.5rem 1rem;}
    .stButton>button:hover { background-color: #2563EB; }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI LOFT Real-Time Monitor")
st.markdown("*A live Dashboard tracking sequential Linear-on-the-fly execution with exhaustive psychometric constraints.*")

# 1. FILE UPLOAD
st.sidebar.header("📁 1. Master Pool")
uploaded_file = st.sidebar.file_uploader("Upload Item Bank (CSV)", type=['csv'])

if not uploaded_file:
    st.info("👋 Upload a CSV item bank to begin.")
    st.stop()
    
df = pd.read_csv(uploaded_file)
st.sidebar.success(f"Bank Loaded: {len(df)} items")

# Detect features
has_enemies = 'enemy_ids' in df.columns
has_testlets = 'testlet_id' in df.columns
has_image = 'has_image' in df.columns
has_audio = 'has_audio' in df.columns
has_rasch_cat = 'raschb_cat' in df.columns
has_domain = 'domain' in df.columns

features = []
if has_domain: features.append(f"Domains ({df['domain'].nunique()})")
if has_enemies: features.append("Enemy Items")
if has_testlets: features.append("Testlets")
if has_image: features.append("Images")
if has_audio: features.append("Audio")
if has_rasch_cat: features.append(f"Difficulty Buckets ({df['raschb_cat'].nunique()})")
st.sidebar.caption("Features: " + " · ".join(features) if features else "No special columns detected.")

st.sidebar.divider()

# ─── MASTER POOL SUMMARY ───
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

# ─── 2. AI PROVIDER ───
st.sidebar.header("🤖 2. AI Provider")
llm_provider = st.sidebar.selectbox("Provider", ['Mock (No Key)', 'Ollama (Local)'])
api_key = ''
nlp_prompt = st.sidebar.text_area(
    "NLP Prompt (optional context for AI)", 
    "Build me test forms targeting theta 0.5 with high information.",
    height=80
)

st.sidebar.divider()

# ─── 3. CONSTRAINTS ───
st.sidebar.header("⚙️ 3. Objectives & Constraints")

with st.sidebar.expander("Form Specifications", expanded=True):
    c1, c2 = st.columns(2)
    n_forms = c1.number_input("# Forms", 1, 50, 3)
    test_len = c2.number_input("Test Length", 1, 200, 10)
    
    c3, c4 = st.columns(2)
    multiplier = c3.number_input("Pool Multiplier", 2, 10, 5)
    exposure_max = c4.number_input("Exposure Limit", 1, 10, 2)
    
    max_overlap = st.slider("Max Jaccard Overlap", 0.0, 1.0, 0.3, 0.05)
    apply_enemies = st.checkbox("Apply Enemy Item Constraints", value=has_enemies, disabled=not has_enemies)

with st.sidebar.expander("Targets & Tolerances", expanded=True):
    c5, c6 = st.columns(2)
    theta_target = c5.number_input("Theta Target (θ)", -3.0, 3.0, 0.0, 0.1)
    tif_target = c6.number_input("TIF Target", 0.0, 20.0, 2.0, 0.5)
    
    c7, c8 = st.columns(2)
    mean_diff = c7.number_input("Mean B Target", -3.0, 3.0, 0.0, 0.1)
    mean_tol = c8.number_input("Mean B Tol", 0.0, 3.0, 0.2, 0.1)
    
    c9, c10 = st.columns(2)
    tif_tol_val = c9.number_input("TIF Tol (+)", 0.0, 10.0, 0.2, 0.1)
    tcc_tol_val = c10.number_input("TCC Tol (+/-)", 0.0, 5.0, 0.5, 0.1)
    
    tcc_target_val = st.number_input("TCC Target (Expected Score)", value=0.0, step=1.0, help="0.0 = Disabled")

with st.sidebar.expander("Dynamic Domains", expanded=False):
    domain_payload = {}
    if has_domain:
        domains_list = df['domain'].dropna().unique().tolist()
        for dom in domains_list:
            st.markdown(f"**{dom}** ({len(df[df['domain']==dom])} items)")
            d1, d2 = st.columns(2)
            d_min = d1.number_input("Min", 0, test_len, 0, key=f"dmin_{dom}")
            d_max = d2.number_input("Max", 0, test_len, 0, key=f"dmax_{dom}")
            if d_min > 0 or d_max > 0:
                domain_payload[dom] = {'min': d_min, 'max': d_max if d_max > 0 else test_len}
    else:
        st.info("No 'domain' column found in bank.")

with st.sidebar.expander("Media Constraints", expanded=False):
    c_items = st.text_input("Common Item IDs (comma-separated)", "")
    st.markdown("**Images**")
    im1, im2 = st.columns(2)
    img_min = im1.number_input("Min Image", 0, test_len, 0)
    img_max = im2.number_input("Max Image", 0, test_len, 0)
    
    st.markdown("**Audio**")
    au1, au2 = st.columns(2)
    audio_min = au1.number_input("Min Audio", 0, test_len, 0)
    audio_max = au2.number_input("Max Audio", 0, test_len, 0)
    
with st.sidebar.expander("Difficulty Buckets (1-7)", expanded=False):
    st.markdown("*(1=Extreme Easy ➡️ 7=Extreme Hard)*")
    rasch_cats = {}
    for i in range(1, 8):
        b1, b2 = st.columns(2)
        b_min = b1.number_input(f"B{i} Min", 0, test_len, 0, key=f"bmin_{i}")
        b_max = b2.number_input(f"B{i} Max", 0, test_len, 0, key=f"bmax_{i}")
        if b_min > 0 or b_max > 0:
            rasch_cats[str(i)] = {'min': b_min, 'max': b_max if b_max > 0 else test_len}

# Build advanced settings payload
advanced_payload = {
    'n_forms': n_forms,
    'test_length': test_len,
    'multiplier': multiplier,
    'exposure_global_max': exposure_max,
    'theta_targets': [theta_target - 1, theta_target, theta_target + 1],
    'min_tif_targets': [max(tif_target - 0.5, 0), tif_target, max(tif_target - 0.5, 0)],
    'mean_difficulty_target': mean_diff,
    'mean_difficulty_tolerance': mean_tol,
    'max_overlap_threshold': max_overlap,
    'apply_enemies': apply_enemies
}
if domain_payload:
    advanced_payload['domain_constraints'] = domain_payload
if tcc_target_val > 0.0:
    advanced_payload['tcc_targets'] = [None, tcc_target_val, None]
    advanced_payload['tcc_tolerances'] = [100.0, tcc_tol_val, 100.0]
if tif_tol_val > 0.0:
    advanced_payload['tif_tolerances'] = [100.0, tif_tol_val, 100.0]
if c_items:
    advanced_payload['common_items'] = [x.strip() for x in c_items.split(',') if x.strip()]
if img_min > 0 or img_max > 0: 
    advanced_payload['image_constraint'] = {'min': img_min, 'max': img_max if img_max > 0 else test_len}
if audio_min > 0 or audio_max > 0: 
    advanced_payload['audio_constraint'] = {'min': audio_min, 'max': audio_max if audio_max > 0 else test_len}
if rasch_cats: 
    advanced_payload['raschb_cat_constraints'] = rasch_cats

# ─── EXECUTE REAL-TIME DASHBOARD ───
col_metrics, col_plot = st.columns([1, 2])

with col_metrics:
    st.markdown("### 🏃‍♂️ Engine Execution")
    exec_btn = st.button("🚀 Run Live Generation", use_container_width=True, type="primary")
    
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

with col_plot:
    st.markdown("### 📈 Live TIF Chart")
    plot_placeholder_tif = st.empty()
    
    st.markdown("### 📈 Live TCC (Expected Score)")
    plot_placeholder_tcc = st.empty()
    
    fig_tif = go.Figure()
    fig_tif.update_layout(title="Test Information Functions (TIF)", xaxis_title="Theta (θ)", yaxis_title="Information", height=400)
    plot_placeholder_tif.plotly_chart(fig_tif, use_container_width=True)

    fig_tcc = go.Figure()
    fig_tcc.update_layout(title="Test Characteristic Curve (TCC)", xaxis_title="Theta (θ)", yaxis_title="Expected Score", height=400)
    plot_placeholder_tcc.plotly_chart(fig_tcc, use_container_width=True)


if exec_btn:
    provider_map = {'Mock (No Key)': 'Mock (No Key)', 'Ollama (Local)': 'ollama'}
    engine_provider = provider_map.get(llm_provider, 'Mock (No Key)')
    
    status_text.info("Initiating Generator...")
    gen = sequential_loft_assembly(
        item_bank=df,
        user_prompt=nlp_prompt,
        llm_provider=engine_provider,
        api_key=api_key,
        advanced_settings=advanced_payload
    )

    all_forms = []
    total_requested = 0
    fig_tif = go.Figure()
    fig_tif.update_layout(title="Test Information Functions (TIF)", xaxis_title="Theta (θ)", yaxis_title="Information", height=400)
    
    fig_tcc = go.Figure()
    fig_tcc.update_layout(title="Test Characteristic Curve (TCC)", xaxis_title="Theta (θ)", yaxis_title="Expected Score", height=400)
    x_vals = np.linspace(-3, 3, 100)
    
    for step in gen:
        if step['step'] == 'sampling':
            status_text.info(f"🔄 Sampling Active Pool for Form {step['form_idx']}...")
            pool_summary_placeholder.markdown(
                f"**Eligible Pool**: {step['eligible_pool_size']} items "
                f"(Mean B: {step['eligible_pool_mean_b']:.2f}, SD: {step['eligible_pool_sd_b']:.2f}) → "
                f"**Active Pool**: {step['active_pool_size']} items "
                f"(Mean B: {step['active_pool_mean_b']:.2f}, SD: {step['active_pool_sd_b']:.2f})"
            )
                
        elif step['step'] == 'warning':
            status_text.warning(f"⚠️ {step['message']}")
            
        elif step['step'] == 'diagnostic':
            status_text.warning(f"⚠️ {step['message']}")
            with st.expander("🔍 AI Diagnostic (read-only)", expanded=True):
                st.info(step['diagnosis'])
            
        elif step['step'] == 'error':
            status_text.error(f"🛑 {step['message']}")
            break
            
        elif step['step'] == 'form_complete':
            f_idx = step['form_idx']
            status_text.success(f"✅ Form {f_idx}/{n_forms} Generated!")
            all_forms = step['forms']
            usage = step['usage_stats']
            
            # Update Progress Bar
            live_progress.progress(min(f_idx / n_forms, 1.0))
            
            # Update Live Metrics
            used_items = {k: v for k, v in usage.items() if v > 0}
            metric_forms.metric("Forms Built", f"{f_idx}/{n_forms}")
            metric_util.metric("Bank Utilized", f"{int(len(used_items)/len(df)*100)}%")
            metric_exposure.metric("Max Exposure", max(used_items.values()) if used_items else 0)
            
            # Render Live Plotly (TIF)
            latest = step['latest_form']['selected_items']
            subset = df[df['item_id'].astype(str).isin([str(x) for x in latest])]
            b_array = subset['rasch_b'].values
            
            y_vals_tif = [sum(rasch_information(v, b) for b in b_array) for v in x_vals]
            fig_tif.add_trace(go.Scatter(x=x_vals, y=y_vals_tif, mode='lines', name=f"Form {f_idx} (TIF:{step['latest_form']['metrics']['primary_tif']:.1f})"))
            plot_placeholder_tif.plotly_chart(fig_tif, use_container_width=True)

            # Render Live Plotly (TCC)
            y_vals_tcc = [sum(rasch_probability(v, b) for b in b_array) for v in x_vals]
            fig_tcc.add_trace(go.Scatter(x=x_vals, y=y_vals_tcc, mode='lines', name=f"Form {f_idx} (Len: {len(b_array)})"))
            plot_placeholder_tcc.plotly_chart(fig_tcc, use_container_width=True)
            
        elif step['step'] == 'finished':
            status_text.success(f"🎉 Engine Finished! Built {len(step['forms'])} forms.")
            live_progress.progress(1.0)
            
            # AI Audit Report
            st.markdown("### 🧠 AI Quality Audit")
            st.markdown(step.get('audit_report', 'No audit available.'))
            
            # Bank Utilization Bar Charts
            all_used_ids = set()
            for f_dict in all_forms:
                all_used_ids.update([str(x) for x in f_dict['selected_items']])
            df_used = df[df['item_id'].astype(str).isin(all_used_ids)]
            
            util_c1, util_c2 = st.columns(2)
            
            with util_c1:
                st.markdown("### 📊 Utilization by Domain")
                if has_domain:
                    dom_total = df['domain'].value_counts().sort_index()
                    dom_used = df_used['domain'].value_counts().reindex(dom_total.index, fill_value=0)
                    fig_dom = go.Figure()
                    fig_dom.add_trace(go.Bar(name='Bank Total', x=dom_total.index, y=dom_total.values, marker_color='#CBD5E1'))
                    fig_dom.add_trace(go.Bar(name='Used', x=dom_used.index, y=dom_used.values, marker_color='#3B82F6'))
                    fig_dom.update_layout(barmode='group', height=350, margin=dict(t=30, b=30))
                    st.plotly_chart(fig_dom, use_container_width=True)
                else:
                    st.info("No 'domain' column in bank.")
            
            with util_c2:
                st.markdown("### 📊 Utilization by Difficulty Bucket")
                if has_rasch_cat:
                    buck_total = df['raschb_cat'].astype(str).value_counts().sort_index()
                    buck_used = df_used['raschb_cat'].astype(str).value_counts().reindex(buck_total.index, fill_value=0)
                    fig_buck = go.Figure()
                    fig_buck.add_trace(go.Bar(name='Bank Total', x=buck_total.index, y=buck_total.values, marker_color='#CBD5E1'))
                    fig_buck.add_trace(go.Bar(name='Used', x=buck_used.index, y=buck_used.values, marker_color='#10B981'))
                    fig_buck.update_layout(barmode='group', height=350, margin=dict(t=30, b=30), xaxis_title="Bucket")
                    st.plotly_chart(fig_buck, use_container_width=True)
                else:
                    st.info("No 'raschb_cat' column in bank.")
            
            # Overlap Matrix
            if len(all_forms) > 1:
                st.markdown("### Form Similarity (Jaccard Overlap)")
                matrix = []
                for i, f1 in enumerate(all_forms):
                    row = {}
                    set1 = set(f1['selected_items'])
                    for j, f2 in enumerate(all_forms):
                        set2 = set(f2['selected_items'])
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        row[f"Form {j+1}"] = round(intersection / union, 2) if union > 0 else 0
                    matrix.append(row)
                st.dataframe(pd.DataFrame(matrix, index=[f"Form {i+1}" for i in range(len(all_forms))]))
            
            # Excel Export
            st.markdown("### 📥 Download Forms")
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                for i, f_dict in enumerate(all_forms):
                    f_items = df[df['item_id'].astype(str).isin([str(x) for x in f_dict['selected_items']])]
                    f_items.to_excel(writer, sheet_name=f"Form {i+1}", index=False)
            
            st.download_button(
                label="Download Forms as Excel",
                data=excel_buffer.getvalue(),
                file_name="AI_LOFT_Forms.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
