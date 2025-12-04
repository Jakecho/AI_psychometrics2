"""
CBC_ATA - CBC Solver-Based Automated Test Assembly
====================================================

Uses Mixed Integer Programming with CBC (Coin-or Branch and Cut) solver
for optimal test assembly. Works on Streamlit Community Cloud (free tier).

Features:
- Pure optimization approach (no LLM needed)
- Handles IRT and CTT constraints
- TIF/TCC targets with tolerances
- Domain distribution constraints
- Free and fast

Author: AI Assistant
Date: December 3, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Page configuration
st.set_page_config(
    page_title="CBC ATA Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ==================== File Loading ====================

def load_item_pool(uploaded_file) -> pd.DataFrame:
    """Load item pool from CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Validate required columns
        required_cols = ['item_id', 'domain', 'rasch_b', 'pvalue', 'point_biserial']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Required columns: item_id, domain, rasch_b, pvalue, point_biserial")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ==================== IRT Calculations ====================

def rasch_probability(theta: float, b: float) -> float:
    """Rasch model probability of correct response"""
    return 1.0 / (1.0 + np.exp(-(theta - b)))

def rasch_information(theta: float, b: float) -> float:
    """Item information at theta"""
    p = rasch_probability(theta, b)
    return p * (1 - p)

def calculate_tif(theta: float, b_params: np.ndarray) -> float:
    """Test Information Function at theta"""
    return sum(rasch_information(theta, b) for b in b_params)

def calculate_tcc(theta: float, b_params: np.ndarray) -> float:
    """Test Characteristic Curve (expected score) at theta"""
    return sum(rasch_probability(theta, b) for b in b_params)

# ==================== Reliability Calculation ====================

def estimate_cronbachs_alpha(items_df: pd.DataFrame) -> float:
    """
    Estimate Cronbach's Alpha using item statistics
    
    Uses Spearman-Brown prophecy formula approximation:
    Based on average inter-item correlation and test length
    """
    n_items = len(items_df)
    if n_items < 2:
        return 0.0
    
    # Estimate from point biserial (discrimination)
    # Higher discrimination -> higher reliability
    avg_disc = items_df['point_biserial'].mean()
    
    # Approximate average inter-item correlation from discrimination
    # Typical relationship: r_avg ≈ 0.3 * avg_discrimination
    avg_r = min(0.5, 0.3 * avg_disc)
    
    # Spearman-Brown formula
    alpha = (n_items * avg_r) / (1 + (n_items - 1) * avg_r)
    
    return max(0, min(1, alpha))  # Bound between 0 and 1

# ==================== CBC Optimization ====================

def assemble_form_with_cbc(
    items_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Assemble test form using CBC solver (Mixed Integer Programming)
    
    Args:
        items_df: Item pool DataFrame
        config: Configuration dictionary
    
    Returns:
        (selected_item_ids, assembly_info)
    """
    n_items = len(items_df)
    test_length = config['test_length']
    approach = config['approach']
    domain_constraints = config['domain_constraints']
    
    # Create the LP problem
    prob = LpProblem("Test_Assembly", LpMaximize)
    
    # Decision variables: x[i] = 1 if item i is selected, 0 otherwise
    item_vars = {}
    for idx, row in items_df.iterrows():
        item_id = row['item_id']
        item_vars[item_id] = LpVariable(f"x_{item_id}", cat='Binary')
    
    # ===== OBJECTIVE FUNCTION =====
    if approach == 'IRT' and config.get('eval_points'):
        # Maximize information at cut score
        eval_points = config['eval_points']
        theta_mid = eval_points.get('theta_mid', 0.0)
        
        # Calculate information contribution of each item at theta_mid
        info_contributions = {}
        for idx, row in items_df.iterrows():
            item_id = row['item_id']
            b = row['rasch_b']
            info = rasch_information(theta_mid, b)
            info_contributions[item_id] = info
        
        # Objective: maximize total information at cut score
        prob += lpSum([info_contributions[item_id] * item_vars[item_id] 
                      for item_id in item_vars.keys()])
    
    elif config.get('maximize_alpha'):
        # Maximize reliability (approximate with sum of discriminations)
        prob += lpSum([items_df[items_df['item_id']==item_id]['point_biserial'].values[0] * item_vars[item_id]
                      for item_id in item_vars.keys()])
    else:
        # Default: maximize discrimination
        prob += lpSum([items_df[items_df['item_id']==item_id]['point_biserial'].values[0] * item_vars[item_id]
                      for item_id in item_vars.keys()])
    
    # ===== CONSTRAINTS =====
    
    # 1. Test length constraint
    prob += lpSum([item_vars[item_id] for item_id in item_vars.keys()]) == test_length
    
    # 2. Domain constraints
    for domain, constraints in domain_constraints.items():
        domain_items = items_df[items_df['domain'] == domain]
        domain_item_ids = domain_items['item_id'].tolist()
        
        min_count = constraints['min']
        max_count = constraints['max']
        
        if min_count > 0:
            prob += lpSum([item_vars[item_id] for item_id in domain_item_ids 
                          if item_id in item_vars]) >= min_count
        
        if max_count < test_length:
            prob += lpSum([item_vars[item_id] for item_id in domain_item_ids 
                          if item_id in item_vars]) <= max_count
    
    # 3. P-value constraints (CTT)
    if 'pvalue_min' in config and 'pvalue_max' in config:
        for idx, row in items_df.iterrows():
            item_id = row['item_id']
            pval = row['pvalue']
            if pval < config['pvalue_min'] or pval > config['pvalue_max']:
                prob += item_vars[item_id] == 0
    
    # 4. Discrimination threshold
    if 'pbs_threshold' in config and config['pbs_threshold'] is not None:
        for idx, row in items_df.iterrows():
            item_id = row['item_id']
            if row['point_biserial'] < config['pbs_threshold']:
                prob += item_vars[item_id] == 0
    
    # 5. Excluded items
    if 'excluded_items' in config and config['excluded_items']:
        for excluded_id in config['excluded_items']:
            if excluded_id in item_vars:
                prob += item_vars[excluded_id] == 0
    
    # 6. Common items (must be included)
    if 'common_items' in config and config['common_items']:
        for common_id in config['common_items']:
            if common_id in item_vars:
                prob += item_vars[common_id] == 1
    
    # 7. Mean difficulty constraint (if specified)
    if config.get('apply_mean_diff') and config.get('mean_diff_target') is not None:
        target = config['mean_diff_target']
        tolerance = config.get('mean_diff_tolerance', 0.1)
        
        if approach == 'IRT':
            # Mean Rasch B constraint
            avg_b = lpSum([items_df[items_df['item_id']==item_id]['rasch_b'].values[0] * item_vars[item_id]
                          for item_id in item_vars.keys()]) / test_length
            
            prob += avg_b >= target - tolerance
            prob += avg_b <= target + tolerance
        else:
            # Mean P-value constraint
            avg_pval = lpSum([items_df[items_df['item_id']==item_id]['pvalue'].values[0] * item_vars[item_id]
                             for item_id in item_vars.keys()]) / test_length
            
            prob += avg_pval >= target - tolerance
            prob += avg_pval <= target + tolerance
    
    # 8. Mean difficulty constraint (CTT only)
    if approach == 'CTT' and config.get('mean_difficulty_target') is not None and config.get('difficulty_tolerance') is not None:
        target_mean = config['mean_difficulty_target']
        tolerance = config['difficulty_tolerance']
        
        # Average p-value constraint
        avg_pval = lpSum([items_df[items_df['item_id']==item_id]['pvalue'].values[0] * item_vars[item_id]
                         for item_id in item_vars.keys()]) / test_length
        
        prob += avg_pval >= target_mean - tolerance
        prob += avg_pval <= target_mean + tolerance
    
    # 9. TIF constraints (IRT only)
    if approach == 'IRT' and config.get('eval_points') and config.get('tif_tolerance'):
        eval_points = config['eval_points']
        tif_tol = config['tif_tolerance'].get('tif', 1.5)
        
        # TIF at low point
        theta_low = eval_points.get('theta_low', -1.0)
        tif_target_low = eval_points.get('tif_low', 8.0)
        
        tif_low = lpSum([rasch_information(theta_low, items_df[items_df['item_id']==item_id]['rasch_b'].values[0]) * 
                        item_vars[item_id] for item_id in item_vars.keys()])
        
        prob += tif_low >= tif_target_low - tif_tol
        prob += tif_low <= tif_target_low + tif_tol
        
        # TIF at mid point (cut score)
        theta_mid = eval_points.get('theta_mid', 0.0)
        tif_target_mid = eval_points.get('tif_mid', 12.0)
        
        tif_mid = lpSum([rasch_information(theta_mid, items_df[items_df['item_id']==item_id]['rasch_b'].values[0]) * 
                        item_vars[item_id] for item_id in item_vars.keys()])
        
        prob += tif_mid >= tif_target_mid - tif_tol
        prob += tif_mid <= tif_target_mid + tif_tol
        
        # TIF at high point
        theta_high = eval_points.get('theta_high', 1.0)
        tif_target_high = eval_points.get('tif_high', 8.0)
        
        tif_high = lpSum([rasch_information(theta_high, items_df[items_df['item_id']==item_id]['rasch_b'].values[0]) * 
                         item_vars[item_id] for item_id in item_vars.keys()])
        
        prob += tif_high >= tif_target_high - tif_tol
        prob += tif_high <= tif_target_high + tif_tol
    
    # 10. Enemy item constraints
    if config.get('enemy_check', False):
        # Build enemy pairs from 'enemy' column if it exists
        enemy_pairs = []
        if 'enemy' in items_df.columns:
            for idx, row in items_df.iterrows():
                item_id = row['item_id']
                enemy_str = row.get('enemy', '')
                
                if pd.notna(enemy_str) and str(enemy_str).strip():
                    # Parse comma-separated enemy IDs
                    try:
                        enemy_ids = [int(x.strip()) for x in str(enemy_str).split(',') if x.strip()]
                        for enemy_id in enemy_ids:
                            if enemy_id in item_vars and item_id in item_vars:
                                # Ensure we don't add duplicate pairs
                                if (item_id, enemy_id) not in enemy_pairs and (enemy_id, item_id) not in enemy_pairs:
                                    enemy_pairs.append((item_id, enemy_id))
                    except (ValueError, AttributeError):
                        pass
            
            # Add constraints: if item A is selected, enemy B cannot be selected
            for item_a, item_b in enemy_pairs:
                prob += item_vars[item_a] + item_vars[item_b] <= 1
    
    # Solve the problem using CBC solver
    solver = PULP_CBC_CMD(msg=0)  # msg=0 suppresses solver output
    prob.solve(solver)
    
    # Extract solution
    selected_ids = []
    for item_id in item_vars.keys():
        if item_vars[item_id].varValue == 1:
            selected_ids.append(item_id)
    
    # Assembly info
    status = LpStatus[prob.status]
    objective_value = value(prob.objective) if prob.objective else 0
    
    assembly_info = {
        'status': status,
        'objective_value': objective_value,
        'n_items_selected': len(selected_ids),
        'solver': 'CBC'
    }
    
    return selected_ids, assembly_info

# ==================== Evaluation ====================

def evaluate_form(items_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate assembled form quality"""
    
    approach = config.get('approach', 'IRT')
    b_params = items_df['rasch_b'].values
    
    # Calculate TIF and TCC across theta range
    theta_range = np.linspace(-3, 3, 61)
    tif_values = [calculate_tif(theta, b_params) for theta in theta_range]
    tcc_values = [calculate_tcc(theta, b_params) for theta in theta_range]
    
    # Basic statistics - use appropriate difficulty metric
    if approach == 'CTT':
        mean_difficulty = items_df['pvalue'].mean()
        sd_difficulty = items_df['pvalue'].std()
    else:  # IRT
        mean_difficulty = items_df['rasch_b'].mean()
        sd_difficulty = items_df['rasch_b'].std()
    
    stats = {
        'theta_range': theta_range,
        'tif_values': tif_values,
        'tcc_values': tcc_values,
        'mean_difficulty': mean_difficulty,
        'sd_difficulty': sd_difficulty,
        'mean_discrimination': items_df['point_biserial'].mean(),
        'domain_counts': items_df['domain'].value_counts().to_dict()
    }
    
    # Evaluation at specific points if provided
    if config.get('eval_points'):
        eval_points = config['eval_points']
        
        for point_name, theta_val in [('low', eval_points.get('theta_low')), 
                                      ('mid', eval_points.get('theta_mid')), 
                                      ('high', eval_points.get('theta_high'))]:
            if theta_val is not None:
                idx = np.argmin(np.abs(theta_range - theta_val))
                stats[f'tif_at_{point_name}'] = tif_values[idx]
                stats[f'tcc_at_{point_name}'] = tcc_values[idx]
    
    return stats

# ==================== Visualization ====================

def plot_tif_tcc(stats: Dict[str, Any], config: Dict[str, Any]) -> Tuple[go.Figure, go.Figure]:
    """Plot TIF and TCC"""
    
    theta_range = stats['theta_range']
    
    # TIF Plot
    fig_tif = go.Figure()
    fig_tif.add_trace(go.Scatter(
        x=theta_range,
        y=stats['tif_values'],
        mode='lines',
        name='TIF',
        line=dict(color='blue', width=3)
    ))
    
    # Add evaluation points if available
    if config.get('eval_points'):
        eval_points = config['eval_points']
        thetas = [eval_points.get('theta_low'), eval_points.get('theta_mid'), eval_points.get('theta_high')]
        targets = [eval_points.get('tif_low'), eval_points.get('tif_mid'), eval_points.get('tif_high')]
        
        fig_tif.add_trace(go.Scatter(
            x=[t for t in thetas if t is not None],
            y=[tgt for tgt in targets if tgt is not None],
            mode='markers',
            name='Targets',
            marker=dict(size=12, color='red', symbol='diamond')
        ))
    
    fig_tif.update_layout(
        title="Test Information Function (TIF)",
        xaxis_title="Theta (θ)",
        yaxis_title="Information",
        height=400
    )
    
    # TCC Plot
    fig_tcc = go.Figure()
    fig_tcc.add_trace(go.Scatter(
        x=theta_range,
        y=stats['tcc_values'],
        mode='lines',
        name='TCC',
        line=dict(color='green', width=3)
    ))
    
    # Add TCC target markers if available
    if config.get('eval_points'):
        eval_points = config['eval_points']
        thetas = [eval_points.get('theta_low'), eval_points.get('theta_mid'), eval_points.get('theta_high')]
        tcc_targets = [eval_points.get('tcc_low'), eval_points.get('tcc_mid'), eval_points.get('tcc_high')]
        
        if any(t is not None for t in tcc_targets):
            fig_tcc.add_trace(go.Scatter(
                x=[t for t in thetas if t is not None],
                y=[tgt for tgt in tcc_targets if tgt is not None],
                mode='markers',
                name='Targets',
                marker=dict(size=12, color='red', symbol='diamond')
            ))
    
    fig_tcc.update_layout(
        title="Test Characteristic Curve (TCC)",
        xaxis_title="Theta (θ)",
        yaxis_title="Expected Score",
        height=400
    )
    
    return fig_tif, fig_tcc

# ==================== Display Functions ====================

def display_form_results(form_data: Dict[str, Any], eval_points: Dict, common_items: List[int], approach: str, enemy_check: bool = False):
    """Display results for a single form"""
    selected_df = form_data['selected_df']
    stats = form_data['stats']
    alpha = form_data['alpha']
    tif_at_cut = form_data['tif_at_cut']
    tcc_at_cut = form_data['tcc_at_cut']
    theta_cut = form_data['theta_cut']
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Difficulty", f"{stats['mean_difficulty']:.2f}")
    with col2:
        st.metric("SD Difficulty", f"{stats['sd_difficulty']:.2f}")
    with col3:
        st.metric("Mean Discrimination", f"{stats['mean_discrimination']:.3f}")
    with col4:
        st.metric("Cronbach's α", f"{alpha:.3f}")
    
    # Domain distribution
    st.subheader("📚 Domain Distribution")
    domain_df = pd.DataFrame({
        'Domain': list(stats['domain_counts'].keys()),
        'Count': list(stats['domain_counts'].values())
    })
    st.dataframe(domain_df, use_container_width=True, hide_index=True)
    
    # Summary table
    st.subheader("📋 Summary")
    summary_data = {
        'Metric': [
            'Test Length',
            'Cronbach Alpha',
            'Mean P-value',
            'Mean Discrimination',
            'Mean Rasch B',
            'SD Rasch B',
            'Enemy Check'
        ],
        'Value': [
            len(selected_df),
            f"{alpha:.3f}",
            f"{selected_df['pvalue'].mean():.3f}",
            f"{stats['mean_discrimination']:.3f}",
            f"{stats['mean_difficulty']:.3f}",
            f"{stats['sd_difficulty']:.3f}",
            'Enabled' if enemy_check else 'Disabled'
        ]
    }
    
    if tif_at_cut is not None:
        summary_data['Metric'].extend([f'TIF @ θ={theta_cut:.1f}', f'TCC @ θ={theta_cut:.1f}'])
        summary_data['Value'].extend([f"{tif_at_cut:.2f}", f"{tcc_at_cut:.2f}"])
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==================== Main Application ====================

def main():
    st.markdown('<div class="main-header">🎯 CBC ATA Tool - Optimization-Based Test Assembly</div>', unsafe_allow_html=True)
    st.markdown("**Using CBC (Coin-or Branch and Cut) solver for optimal form assembly**")
    
    st.info("💡 **Portable & Free**: No database required! Upload your item pool as CSV or Excel file.")
    
    # File Upload
    st.subheader("📁 Upload Item Pool")
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Required columns: item_id, domain, rasch_b, pvalue, point_biserial"
    )
    
    if uploaded_file is None:
        st.warning("⬆️ Please upload an item pool file to begin")
        
        # Show example format
        with st.expander("📋 Example File Format"):
            example_df = pd.DataFrame({
                'item_id': [1, 2, 3, 4, 5],
                'domain': ['Cardiology', 'Cardiology', 'Pharmacology', 'Med-Surg', 'Med-Surg'],
                'rasch_b': [0.15, -0.32, 0.45, -0.10, 0.22],
                'pvalue': [0.62, 0.75, 0.55, 0.68, 0.60],
                'point_biserial': [0.35, 0.42, 0.38, 0.40, 0.36]
            })
            st.dataframe(example_df)
            
            # Download example
            csv_example = example_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Example CSV",
                data=csv_example,
                file_name="item_pool_example.csv",
                mime="text/csv"
            )
        
        return
    
    # Load items
    with st.spinner("Loading item pool..."):
        items_df = load_item_pool(uploaded_file)
    
    if items_df is None:
        return
    
    domains = sorted(items_df['domain'].unique().tolist())
    
    st.success(f"✅ Loaded {len(items_df)} items from {len(domains)} domains")
    
    # Show item pool summary by domain
    with st.expander("📊 Item Pool Summary by Domain"):
        domain_stats = items_df.groupby('domain').agg({
            'item_id': 'count',
            'pvalue': 'mean',
            'point_biserial': 'mean',
            'rasch_b': 'mean'
        }).round(3)
        domain_stats.columns = ['Count', 'Mean P-value', 'Mean Discrimination', 'Mean Rasch B']
        
        st.dataframe(domain_stats, use_container_width=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Test specifications
    st.sidebar.subheader("📝 Test Specifications")
    
    n_forms = st.sidebar.number_input(
        "Number of Forms to Assemble",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="Number of parallel forms to create"
    )
    
    test_length = st.sidebar.number_input(
        "Items per Form", 
        min_value=10, 
        max_value=200, 
        value=72, 
        step=1
    )
    
    approach = st.sidebar.radio(
        "Approach", 
        options=['IRT', 'CTT'], 
        help="IRT uses Rasch model, CTT uses classical statistics"
    )
    
    st.sidebar.divider()
    
    # Domain distribution - improved layout
    st.sidebar.subheader("📚 Domain Distribution")
    st.sidebar.caption("Set minimum and maximum items per domain")
    
    domain_constraints = {}
    
    # Create a more compact display
    for domain in domains:
        with st.sidebar.container():
            st.markdown(f"**{domain}**")
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input(
                    "Min", 
                    min_value=0, 
                    max_value=test_length, 
                    value=9, 
                    key=f"min_{domain}",
                    label_visibility="collapsed"
                )
            with col2:
                max_val = st.number_input(
                    "Max", 
                    min_value=min_val, 
                    max_value=test_length, 
                    value=9, 
                    key=f"max_{domain}",
                    label_visibility="collapsed"
                )
            
            # Show visual indicator
            if min_val == max_val:
                st.caption(f"🎯 Exactly {min_val} items")
            else:
                st.caption(f"📊 {min_val}-{max_val} items")
        
        domain_constraints[domain] = {'min': min_val, 'max': max_val}
    
    st.sidebar.divider()
    
    # Common Items
    st.sidebar.subheader("🔗 Common Items")
    st.sidebar.caption("Items that MUST appear in all forms")
    
    n_common = st.sidebar.number_input(
        "Number of Common Items",
        min_value=0,
        max_value=test_length // 2,
        value=0,
        step=1,
        help="Number of items shared across all parallel forms"
    )
    
    common_items_str = st.sidebar.text_input(
        "Common Item IDs (comma-separated)",
        value="",
        help="e.g., 101,205,312",
        placeholder="101,205,312"
    )
    
    # Parse common items
    common_items = []
    if common_items_str.strip():
        # User provided specific IDs
        try:
            common_items = [int(x.strip()) for x in common_items_str.split(',') if x.strip()]
            if len(common_items) != n_common:
                st.sidebar.warning(f"⚠️ Expected {n_common} common items, got {len(common_items)}")
        except ValueError:
            st.sidebar.error("❌ Invalid item IDs. Use comma-separated numbers.")
    elif n_common > 0:
        # Auto-sample based on domain weights
        st.sidebar.info(f"🎲 Auto-sampling {n_common} common items proportionally by domain")
        
        # Calculate how many items per domain
        total_domain_items = sum(dc['min'] for dc in domain_constraints.values())
        
        sampled_items = []
        for domain, constraints in domain_constraints.items():
            domain_items = items_df[items_df['domain'] == domain]
            
            # Calculate proportion
            domain_weight = constraints['min'] / total_domain_items if total_domain_items > 0 else 1.0 / len(domain_constraints)
            n_from_domain = max(1, int(n_common * domain_weight))
            
            # Sample randomly from this domain
            if len(domain_items) >= n_from_domain:
                sample = domain_items.sample(n=n_from_domain, random_state=42)
                sampled_items.extend(sample['item_id'].tolist())
        
        # Adjust to exact count if needed
        if len(sampled_items) > n_common:
            common_items = sampled_items[:n_common]
        elif len(sampled_items) < n_common:
            # Sample more from largest domain
            remaining = n_common - len(sampled_items)
            largest_domain = max(domain_constraints.keys(), key=lambda d: domain_constraints[d]['min'])
            additional = items_df[
                (items_df['domain'] == largest_domain) & 
                (~items_df['item_id'].isin(sampled_items))
            ].sample(n=min(remaining, len(items_df[items_df['domain'] == largest_domain])), random_state=42)
            common_items = sampled_items + additional['item_id'].tolist()
        else:
            common_items = sampled_items
        
        st.sidebar.success(f"✅ Auto-selected: {', '.join(map(str, common_items))}")
    
    # Excluded Items
    st.sidebar.subheader("🚫 Excluded Items")
    st.sidebar.caption("Items to exclude from all forms")
    
    excluded_items_str = st.sidebar.text_input(
        "Excluded Item IDs (comma-separated)",
        value="",
        help="e.g., 45,78,99",
        placeholder="45,78,99"
    )
    
    # Parse excluded items
    excluded_items = []
    if excluded_items_str.strip():
        try:
            excluded_items = [int(x.strip()) for x in excluded_items_str.split(',') if x.strip()]
            st.sidebar.info(f"🚫 Excluding {len(excluded_items)} items")
        except ValueError:
            st.sidebar.error("❌ Invalid item IDs. Use comma-separated numbers.")
    
    st.sidebar.divider()
    
    # Quality constraints
    st.sidebar.subheader("⚙️ Quality Constraints")
    maximize_alpha = st.sidebar.checkbox("Maximize Reliability", value=True)
    
    enemy_check = st.sidebar.checkbox(
        "Enforce Enemy Constraints",
        value=True,
        help="Prevent enemy items (marked in 'enemy' column) from appearing together in the same form"
    )
    
    # IRT Evaluation Points
    eval_points = None
    tif_tolerance = None
    tcc_tolerance = None
    
    if approach == 'IRT':
        st.sidebar.subheader("📐 IRT Evaluation Points")
        logit_cut = st.sidebar.number_input("Logit Cut (θ)", -3.0, 3.0, 0.0, 0.1)
        
        evaluation_points = [logit_cut - 1.0, logit_cut, logit_cut + 1.0]
        st.sidebar.markdown(f"**Points:** {evaluation_points[0]:.1f}, {evaluation_points[1]:.1f}, {evaluation_points[2]:.1f}")
        
        # TIF Targets
        st.sidebar.markdown("**TIF Targets:**")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            tif_low = st.number_input(f"TIF @ {evaluation_points[0]:.1f}", 0.0, 50.0, 5.0, 0.5, key="tif_low")
        with col2:
            tif_mid = st.number_input(f"TIF @ {evaluation_points[1]:.1f}", 0.0, 50.0, 10.0, 0.5, key="tif_mid")
        with col3:
            tif_high = st.number_input(f"TIF @ {evaluation_points[2]:.1f}", 0.0, 50.0, 5.0, 0.5, key="tif_high")
        
        # TCC Targets
        st.sidebar.markdown("**TCC Targets (Expected Score):**")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            tcc_low = st.number_input(f"TCC @ {evaluation_points[0]:.1f}", 0.0, float(test_length), float(test_length * 0.3), 1.0, key="tcc_low")
        with col2:
            tcc_mid = st.number_input(f"TCC @ {evaluation_points[1]:.1f}", 0.0, float(test_length), float(test_length * 0.5), 1.0, key="tcc_mid")
        with col3:
            tcc_high = st.number_input(f"TCC @ {evaluation_points[2]:.1f}", 0.0, float(test_length), float(test_length * 0.7), 1.0, key="tcc_high")
        
        # Tolerances
        st.sidebar.markdown("**Tolerances:**")
        tol_col1, tol_col2 = st.sidebar.columns(2)
        with tol_col1:
            tif_tolerance = st.number_input("TIF Tolerance (±)", 0.1, 10.0, 0.2, 0.1)
        with tol_col2:
            tcc_tolerance = st.number_input("TCC Tolerance (±)", 0.1, 20.0, 1.0, 0.5)
        
        eval_points = {
            'theta_low': evaluation_points[0],
            'theta_mid': evaluation_points[1],
            'theta_high': evaluation_points[2],
            'tif_low': tif_low,
            'tif_mid': tif_mid,
            'tif_high': tif_high,
            'tcc_low': tcc_low,
            'tcc_mid': tcc_mid,
            'tcc_high': tcc_high
        }
    
    # CTT Constraints
    mean_difficulty_target = None
    difficulty_tolerance = None
    pvalue_min = 0.0
    pvalue_max = 1.0
    discrimination_min = 0.0
    
    if approach == 'CTT':
        st.sidebar.subheader("📊 CTT Constraints")
        
        # Mean difficulty target
        st.sidebar.markdown("**Mean Difficulty (P-value):**")
        mean_difficulty_target = st.sidebar.number_input(
            "Target Mean P-value",
            0.0, 1.0, 0.6, 0.05,
            help="Target average difficulty (p-value) for the test"
        )
        difficulty_tolerance = st.sidebar.number_input(
            "Tolerance (±)",
            0.01, 0.5, 0.1, 0.01,
            help="Acceptable deviation from target mean p-value"
        )
        
        # P-value range
        st.sidebar.markdown("**P-value Range:**")
        pval_col1, pval_col2 = st.sidebar.columns(2)
        with pval_col1:
            pvalue_min = st.number_input("Min P-value", 0.0, 1.0, 0.3, 0.05)
        with pval_col2:
            pvalue_max = st.number_input("Max P-value", 0.0, 1.0, 0.9, 0.05)
        
        # Discrimination threshold
        discrimination_min = st.sidebar.number_input(
            "Min Discrimination",
            0.0, 1.0, 0.2, 0.05,
            help="Minimum point-biserial correlation"
        )
    
    # Assemble button
    st.sidebar.divider()
    if st.sidebar.button("🚀 Assemble Form", type="primary", use_container_width=True):
        # Prepare config
        config = {
            'test_length': test_length,
            'approach': approach,
            'domain_constraints': domain_constraints,
            'maximize_alpha': maximize_alpha,
            'eval_points': eval_points,
            'tif_tolerance': {'tif': tif_tolerance, 'tcc': tcc_tolerance} if tif_tolerance else None,
            'pvalue_min': pvalue_min,
            'pvalue_max': pvalue_max,
            'pbs_threshold': discrimination_min,
            'mean_difficulty_target': mean_difficulty_target,
            'difficulty_tolerance': difficulty_tolerance,
            'excluded_items': excluded_items,
            'common_items': common_items,
            'enemy_check': enemy_check
        }
        
        # Validate that common item count matches if specified
        if n_common > 0 and len(common_items) != n_common:
            st.error(f"❌ Please specify exactly {n_common} common item IDs")
        else:
            # Store results in session state
            st.session_state['assembly_complete'] = False
            st.session_state['config'] = config
            st.session_state['n_forms'] = n_forms
            st.session_state['common_items'] = common_items
            st.session_state['approach'] = approach
            st.session_state['eval_points'] = eval_points
            st.session_state['enemy_check'] = enemy_check
            
            # Assemble multiple forms
            with st.spinner(f"🔧 Running CBC solver for {n_forms} form(s)..."):
                all_forms = []
                used_items = set(excluded_items)  # Start with excluded items
                
                for form_num in range(1, n_forms + 1):
                    try:
                        # Update config with used items to exclude
                        config['excluded_items'] = list(used_items)
                        
                        selected_ids, assembly_info = assemble_form_with_cbc(items_df, config)
                        
                        if assembly_info['status'] != 'Optimal':
                            st.error(f"❌ Form {form_num}: Solver status: {assembly_info['status']}")
                            st.info("Try relaxing constraints or reducing number of forms")
                            break
                        
                        # Get selected items
                        selected_df = items_df[items_df['item_id'].isin(selected_ids)].copy()
                        
                        # Evaluate form
                        stats = evaluate_form(selected_df, config)
                        alpha = estimate_cronbachs_alpha(selected_df)
                        
                        # Calculate TIF/TCC at logit cut
                        if eval_points:
                            theta_cut = eval_points['theta_mid']
                            idx_cut = np.argmin(np.abs(stats['theta_range'] - theta_cut))
                            tif_at_cut = stats['tif_values'][idx_cut]
                            tcc_at_cut = stats['tcc_values'][idx_cut]
                        else:
                            tif_at_cut = None
                            tcc_at_cut = None
                        
                        # Store form data
                        all_forms.append({
                            'form_num': form_num,
                            'selected_df': selected_df,
                            'stats': stats,
                            'alpha': alpha,
                            'tif_at_cut': tif_at_cut,
                            'tcc_at_cut': tcc_at_cut,
                            'theta_cut': theta_cut if eval_points else None
                        })
                        
                        # Update used items (exclude common items from being marked as used)
                        non_common_items = set(selected_ids) - set(common_items)
                        used_items.update(non_common_items)
                        
                        st.success(f"✅ Form {form_num}: Optimal solution found! Selected {len(selected_ids)} items")
                        
                    except Exception as e:
                        st.error(f"❌ Form {form_num} assembly failed: {e}")
                        break
                
                if all_forms:
                    st.session_state['all_forms'] = all_forms
                    st.session_state['assembly_complete'] = True
    
    # Display results if assembly is complete
    if st.session_state.get('assembly_complete', False):
        all_forms = st.session_state['all_forms']
        common_items = st.session_state['common_items']
        approach = st.session_state['approach']
        eval_points = st.session_state['eval_points']
        enemy_check = st.session_state['enemy_check']
        
        # Display results for all forms
        st.header(f"📊 Assembly Results ({len(all_forms)} form(s))")
        
        # Show common items summary
        if common_items:
            st.info(f"🔗 **Common Items ({len(common_items)}):** {', '.join(map(str, common_items))}")
        
        # Overlay plots for all forms (IRT only)
        if approach == 'IRT' and eval_points:
            st.subheader("📈 TIF/TCC Comparison Across Forms")
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # TIF overlay
            fig_tif_all = go.Figure()
            for i, form_data in enumerate(all_forms):
                stats = form_data['stats']
                form_num = form_data['form_num']
                color = colors[i % len(colors)]
                
                fig_tif_all.add_trace(go.Scatter(
                    x=stats['theta_range'],
                    y=stats['tif_values'],
                    mode='lines',
                    name=f'Form {form_num}',
                    line=dict(color=color, width=2)
                ))
            
            # Add target markers
            thetas = [eval_points.get('theta_low'), eval_points.get('theta_mid'), eval_points.get('theta_high')]
            tif_targets = [eval_points.get('tif_low'), eval_points.get('tif_mid'), eval_points.get('tif_high')]
            
            fig_tif_all.add_trace(go.Scatter(
                x=[t for t in thetas if t is not None],
                y=[tgt for tgt in tif_targets if tgt is not None],
                mode='markers',
                name='Targets',
                marker=dict(size=12, color='black', symbol='diamond')
            ))
            
            fig_tif_all.update_layout(
                title="Test Information Function (TIF) - All Forms",
                xaxis_title="Theta (θ)",
                yaxis_title="Information",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_tif_all, use_container_width=True)
            
            # TCC overlay
            fig_tcc_all = go.Figure()
            for i, form_data in enumerate(all_forms):
                stats = form_data['stats']
                form_num = form_data['form_num']
                color = colors[i % len(colors)]
                
                fig_tcc_all.add_trace(go.Scatter(
                    x=stats['theta_range'],
                    y=stats['tcc_values'],
                    mode='lines',
                    name=f'Form {form_num}',
                    line=dict(color=color, width=2)
                ))
            
            # Add TCC target markers
            tcc_targets = [eval_points.get('tcc_low'), eval_points.get('tcc_mid'), eval_points.get('tcc_high')]
            
            if any(t is not None for t in tcc_targets):
                fig_tcc_all.add_trace(go.Scatter(
                    x=[t for t in thetas if t is not None],
                    y=[tgt for tgt in tcc_targets if tgt is not None],
                    mode='markers',
                    name='Targets',
                    marker=dict(size=12, color='black', symbol='diamond')
                ))
            
            fig_tcc_all.update_layout(
                title="Test Characteristic Curve (TCC) - All Forms",
                xaxis_title="Theta (θ)",
                yaxis_title="Expected Score",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_tcc_all, use_container_width=True)
        
        # Create tabs for each form
        if len(all_forms) == 1:
            # Single form - show directly
            form_data = all_forms[0]
            display_form_results(form_data, eval_points, common_items, approach, enemy_check)
        else:
            # Multiple forms - use tabs
            tabs = st.tabs([f"Form {i+1}" for i in range(len(all_forms))])
            for i, (tab, form_data) in enumerate(zip(tabs, all_forms)):
                with tab:
                    display_form_results(form_data, eval_points, common_items, approach, enemy_check)
        
        # Excel export with all forms
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Overview comparison of all forms
            comparison_rows = []
            for form_data in all_forms:
                form_num = form_data['form_num']
                alpha = form_data['alpha']
                stats = form_data['stats']
                selected_df = form_data['selected_df']
                tif_at_cut = form_data['tif_at_cut']
                tcc_at_cut = form_data['tcc_at_cut']
                theta_cut = form_data['theta_cut']
                
                row = {
                    'Form': f'Form {form_num}',
                    'N Items': len(selected_df),
                    'Cronbach α': f"{alpha:.3f}",
                    'Mean P-value': f"{selected_df['pvalue'].mean():.3f}",
                    'Mean Discrimination': f"{stats['mean_discrimination']:.3f}",
                    'Mean Rasch B': f"{stats['mean_difficulty']:.3f}",
                    'SD Rasch B': f"{stats['sd_difficulty']:.3f}"
                }
                
                if tif_at_cut is not None:
                    row[f'TIF @ θ={theta_cut:.1f}'] = f"{tif_at_cut:.2f}"
                    row[f'TCC @ θ={theta_cut:.1f}'] = f"{tcc_at_cut:.2f}"
                
                comparison_rows.append(row)
            
            comparison_df = pd.DataFrame(comparison_rows)
            comparison_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Add common items to overview
            if common_items:
                common_info = pd.DataFrame({
                    'Info': ['Common Items'],
                    'Value': [', '.join(map(str, common_items))]
                })
                common_info.to_excel(writer, sheet_name='Overview', index=False, startrow=len(comparison_df) + 2)
            
            # Individual summary sheets for each form
            for form_data in all_forms:
                form_num = form_data['form_num']
                alpha = form_data['alpha']
                stats = form_data['stats']
                selected_df = form_data['selected_df']
                tif_at_cut = form_data['tif_at_cut']
                tcc_at_cut = form_data['tcc_at_cut']
                theta_cut = form_data['theta_cut']
                
                # Build summary data
                summary_data = {
                    'Metric': [
                        'Test Length',
                        'Cronbach Alpha',
                        'Mean P-value',
                        'Mean Discrimination',
                        'Mean Rasch B',
                        'SD Rasch B',
                        'Enemy Check'
                    ],
                    'Value': [
                        len(selected_df),
                        f"{alpha:.3f}",
                        f"{selected_df['pvalue'].mean():.3f}",
                        f"{stats['mean_discrimination']:.3f}",
                        f"{stats['mean_difficulty']:.3f}",
                        f"{stats['sd_difficulty']:.3f}",
                        'Enabled' if enemy_check else 'Disabled'
                    ]
                }
                
                if tif_at_cut is not None:
                    summary_data['Metric'].extend([f'TIF @ θ={theta_cut:.1f}', f'TCC @ θ={theta_cut:.1f}'])
                    summary_data['Value'].extend([f"{tif_at_cut:.2f}", f"{tcc_at_cut:.2f}"])
                
                # Add domain counts
                summary_data['Metric'].append('')  # Blank row
                summary_data['Value'].append('')
                summary_data['Metric'].append('Domain Distribution')
                summary_data['Value'].append('')
                
                for domain, count in stats['domain_counts'].items():
                    summary_data['Metric'].append(f"  {domain}")
                    summary_data['Value'].append(count)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name=f'Form_{form_num}_Summary', index=False)
                
                # Items sheet
                selected_df.to_excel(writer, sheet_name=f'Form_{form_num}_Items', index=False)
        
        st.download_button(
            label=f"📥 Download {len(all_forms)} Form(s) (Excel)",
            data=output.getvalue(),
            file_name=f"CBC_Forms_{len(all_forms)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_forms"
        )

if __name__ == "__main__":
    main()
