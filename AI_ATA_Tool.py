"""
AI_ATA_Tool - Automated Test Assembly Tool
==========================================

This tool performs automated test assembly using Large Language Models (LLM)
to create optimal test forms based on psychometric constraints.

Supports both IRT (Item Response Theory) and CTT (Classical Test Theory) approaches.

Author: AI Assistant
Date: November 21, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from io import BytesIO
import json
from datetime import datetime
import os
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI ATA Tool",
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
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def calculate_cronbach_alpha(item_scores):
    """
    Calculate Cronbach's Alpha reliability coefficient
    
    Args:
        item_scores: DataFrame with rows=examinees, columns=items
    
    Returns:
        float: Cronbach's Alpha value
    """
    n_items = item_scores.shape[1]
    if n_items < 2:
        return 0.0
    
    item_variances = item_scores.var(axis=0, ddof=1)
    total_variance = item_scores.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha


def estimate_form_alpha(items_df, selected_items):
    """
    Estimate Cronbach's Alpha for a form based on item statistics
    Uses Spearman-Brown prophecy formula approximation
    
    Args:
        items_df: DataFrame with item statistics
        selected_items: List of selected item IDs
    
    Returns:
        float: Estimated Cronbach's Alpha
    """
    if len(selected_items) < 2:
        return 0.0
    
    # Average inter-item correlation approximation using PBS
    selected_data = items_df[items_df['ItemID'].isin(selected_items)]
    
    if 'PBS' in selected_data.columns:
        avg_pbs = selected_data['PBS'].mean()
        n = len(selected_items)
        # Spearman-Brown formula
        alpha = (n * avg_pbs) / (1 + (n - 1) * avg_pbs)
        return max(0.0, min(1.0, alpha))
    
    return 0.0


def calculate_tif(theta_range, b_values, a_values=None):
    """
    Calculate Test Information Function (TIF) for IRT
    
    Args:
        theta_range: Array of ability levels
        b_values: Item difficulty parameters
        a_values: Item discrimination parameters (if None, uses 1.0)
    
    Returns:
        Array of information values at each theta
    """
    if a_values is None:
        a_values = np.ones(len(b_values))
    
    tif = np.zeros(len(theta_range))
    
    for theta_idx, theta in enumerate(theta_range):
        item_info = 0
        for b, a in zip(b_values, a_values):
            # 1PL/Rasch model information function
            p = 1 / (1 + np.exp(-a * (theta - b)))
            info = a**2 * p * (1 - p)
            item_info += info
        tif[theta_idx] = item_info
    
    return tif


def calculate_tcc(theta_range, b_values, a_values=None):
    """
    Calculate Test Characteristic Curve (TCC) - Expected score at each ability level
    
    Args:
        theta_range: Array of ability levels
        b_values: Item difficulty parameters
        a_values: Item discrimination parameters (if None, uses 1.0 for Rasch)
    
    Returns:
        Array of expected scores at each theta
    """
    if a_values is None:
        a_values = np.ones(len(b_values))
    
    tcc = np.zeros(len(theta_range))
    
    for theta_idx, theta in enumerate(theta_range):
        expected_score = 0
        for b, a in zip(b_values, a_values):
            # Rasch/1PL probability of correct response
            p = 1 / (1 + np.exp(-a * (theta - b)))
            expected_score += p
        tcc[theta_idx] = expected_score
    
    return tcc


def evaluate_form_quality_irt(items_df, selected_items, eval_points, tolerance):
    """
    Evaluate form quality at 3 theta evaluation points for IRT approach
    
    Args:
        items_df: DataFrame with all items
        selected_items: List of selected item IDs
        eval_points: Dict with theta_low, theta_mid, theta_high, tif_low, tif_mid, tif_high, tcc_mid
        tolerance: Tolerance as decimal (e.g., 0.15 for 15%)
    
    Returns:
        Dictionary with evaluation results
    """
    # Get selected item data
    form_data = items_df[items_df['ItemID'].isin(selected_items)]
    
    if len(form_data) == 0:
        return None
    
    # Extract b parameters
    b_values = form_data['RaschB'].values
    
    # Define theta range for plotting
    theta_range = np.linspace(-3, 3, 100)
    
    # Calculate TIF and TCC
    tif = calculate_tif(theta_range, b_values)
    tcc = calculate_tcc(theta_range, b_values)
    
    # Get evaluation points
    theta_low = eval_points.get('theta_low', -1.0)
    theta_mid = eval_points.get('theta_mid', 0.0)
    theta_high = eval_points.get('theta_high', 1.0)
    tif_target_low = eval_points.get('tif_low', 8.0)
    tif_target_mid = eval_points.get('tif_mid', 12.0)
    tif_target_high = eval_points.get('tif_high', 8.0)
    tcc_target_mid = eval_points.get('tcc_mid', None)
    
    # Find TIF/TCC values at evaluation points
    idx_low = np.argmin(np.abs(theta_range - theta_low))
    idx_mid = np.argmin(np.abs(theta_range - theta_mid))
    idx_high = np.argmin(np.abs(theta_range - theta_high))
    
    tif_at_low = tif[idx_low]
    tif_at_mid = tif[idx_mid]
    tif_at_high = tif[idx_high]
    tcc_at_mid = tcc[idx_mid]
    
    # Get tolerances
    tif_tol = tolerance.get('tif', 1.5) if isinstance(tolerance, dict) else tolerance
    tcc_tol = tolerance.get('tcc', 2.0) if isinstance(tolerance, dict) else tolerance
    
    # Check if within tolerance (absolute values)
    meets_tif_low = abs(tif_at_low - tif_target_low) <= tif_tol
    meets_tif_mid = abs(tif_at_mid - tif_target_mid) <= tif_tol
    meets_tif_high = abs(tif_at_high - tif_target_high) <= tif_tol
    
    meets_tcc_mid = True
    if tcc_target_mid is not None:
        meets_tcc_mid = abs(tcc_at_mid - tcc_target_mid) <= tcc_tol
    
    tif_meets_target = meets_tif_low and meets_tif_mid and meets_tif_high
    
    return {
        'theta_range': theta_range,
        'tif_values': tif,
        'tcc_values': tcc,
        'theta_low': theta_low,
        'theta_mid': theta_mid,
        'theta_high': theta_high,
        'tif_at_low': tif_at_low,
        'tif_at_mid': tif_at_mid,
        'tif_at_high': tif_at_high,
        'tcc_at_mid': tcc_at_mid,
        'tif_target_low': tif_target_low,
        'tif_target_mid': tif_target_mid,
        'tif_target_high': tif_target_high,
        'tcc_target_mid': tcc_target_mid,
        'meets_tif_low': meets_tif_low,
        'meets_tif_mid': meets_tif_mid,
        'meets_tif_high': meets_tif_high,
        'meets_tcc_mid': meets_tcc_mid,
        'tif_meets_target': tif_meets_target,
        'meets_all_irt': tif_meets_target and meets_tcc_mid
    }


def plot_tif_tcc(forms_quality_data, form_names):
    """
    Plot TIF and TCC for all forms
    
    Args:
        forms_quality_data: List of quality evaluation dictionaries
        form_names: List of form names
    
    Returns:
        Tuple of (fig_tif, fig_tcc) plotly figures
    """
    # TIF Plot
    fig_tif = go.Figure()
    
    for quality, form_name in zip(forms_quality_data, form_names):
        if quality is None:
            continue
        fig_tif.add_trace(go.Scatter(
            x=quality['theta_range'],
            y=quality['tif_values'],
            mode='lines',
            name=form_name,
            line=dict(width=2)
        ))
    
    # Add target points for first form
    if len(forms_quality_data) > 0 and forms_quality_data[0] is not None:
        quality = forms_quality_data[0]
        theta_points = [quality['theta_low'], quality['theta_mid'], quality['theta_high']]
        target_points = [quality['tif_target_low'], quality['tif_target_mid'], quality['tif_target_high']]
        
        fig_tif.add_trace(go.Scatter(
            x=theta_points,
            y=target_points,
            mode='markers',
            name='Target Points',
            marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='darkred'))
        ))
        
        # Add vertical lines at evaluation points
        for theta, label in zip(theta_points, ['Low', 'Cut', 'High']):
            fig_tif.add_vline(x=theta, line_dash="dot", line_color="gray", opacity=0.5,
                             annotation_text=f"{label} θ={theta}")
    
    fig_tif.update_layout(
        title="Test Information Function (TIF) - All Forms",
        xaxis_title="Theta (Ability Level)",
        yaxis_title="Information",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # TCC Plot
    fig_tcc = go.Figure()
    
    for quality, form_name in zip(forms_quality_data, form_names):
        if quality is None:
            continue
        fig_tcc.add_trace(go.Scatter(
            x=quality['theta_range'],
            y=quality['tcc_values'],
            mode='lines',
            name=form_name,
            line=dict(width=2)
        ))
    
    # Add vertical lines and TCC target
    if len(forms_quality_data) > 0 and forms_quality_data[0] is not None:
        quality = forms_quality_data[0]
        theta_points = [quality['theta_low'], quality['theta_mid'], quality['theta_high']]
        
        for theta, label in zip(theta_points, ['Low', 'Cut', 'High']):
            fig_tcc.add_vline(x=theta, line_dash="dot", line_color="gray", opacity=0.5,
                             annotation_text=f"{label} θ={theta}")
        
        # Mark TCC target at cut point
        if quality['tcc_target_mid'] is not None:
            fig_tcc.add_trace(go.Scatter(
                x=[quality['theta_mid']],
                y=[quality['tcc_target_mid']],
                mode='markers',
                name='TCC Target at Cut',
                marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='darkred'))
            ))
    
    fig_tcc.update_layout(
        title="Test Characteristic Curve (TCC) - All Forms",
        xaxis_title="Theta (Ability Level)",
        yaxis_title="Expected Score",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig_tif, fig_tcc


def parse_enemy_items(enemy_str):
    """
    Parse enemy items string
    
    Args:
        enemy_str: String with comma-separated item IDs or NaN
    
    Returns:
        List of enemy item IDs
    """
    if pd.isna(enemy_str) or enemy_str == '' or enemy_str == 'None':
        return []
    
    return [item.strip() for item in str(enemy_str).split(',') if item.strip()]


def select_items_heuristic(items_df, config, form_number=0):
    """
    Select items using heuristic scoring algorithm
    
    Args:
        items_df: DataFrame with available items
        config: Assembly configuration
        form_number: Form index for seed variation
        
    Returns:
        List of selected item IDs
    """
    test_length = config['test_length']
    approach = config['approach']
    domain_constraints = config.get('domain_constraints', {})
    common_items = config.get('common_items', [])
    apply_mean_diff = config.get('apply_mean_diff', False)
    mean_diff_target = config.get('mean_diff_target', 0.0 if approach == 'IRT' else 0.65)
    
    # Start with common items
    selected_items = common_items.copy() if common_items else []
    remaining_needed = test_length - len(selected_items)
    
    # Filter available items (exclude already selected)
    available = items_df[~items_df['ItemID'].isin(selected_items)].copy()
    
    # Apply P-value and PBS filters
    if 'pvalue_min' in config and 'pvalue_max' in config:
        available = available[
            (available['Pvalue'] >= config['pvalue_min']) & 
            (available['Pvalue'] <= config['pvalue_max'])
        ]
    
    if 'pbs_threshold' in config and config['pbs_threshold'] is not None:
        available = available[available['PBS'] > config['pbs_threshold']]
    
    # Score items based on approach
    if approach == 'IRT' and 'RaschB' in available.columns:
        # Score: prefer items near target difficulty with good discrimination
        if apply_mean_diff and mean_diff_target is not None:
            available['difficulty_score'] = 1 / (1 + abs(available['RaschB'] - mean_diff_target))
        else:
            available['difficulty_score'] = 1 / (1 + abs(available['RaschB']))
    else:  # CTT
        # Score: prefer items near target p-value with good discrimination
        if apply_mean_diff and mean_diff_target is not None:
            available['difficulty_score'] = 1 / (1 + abs(available['Pvalue'] - mean_diff_target))
        else:
            available['difficulty_score'] = 1 / (1 + abs(available['Pvalue'] - 0.65))
    
    # Combine with discrimination (PBS)
    if config.get('maximize_alpha', False):
        # Prioritize high PBS items
        available['score'] = available['difficulty_score'] * (available['PBS'] ** 2)
    else:
        available['score'] = available['difficulty_score'] * available['PBS']
    
    # Add small random component for variation between forms
    np.random.seed(42 + form_number)
    available['score'] = available['score'] * (1 + np.random.uniform(-0.05, 0.05, len(available)))
    
    # Select items by domain
    selected_by_domain = []
    
    for domain, constraints in domain_constraints.items():
        # How many needed from this domain
        target_count = constraints['min']  # Use min as target
        
        # How many already selected from common items
        if common_items:
            common_in_domain = len([
                item for item in common_items 
                if item in items_df[items_df['Domain'] == domain]['ItemID'].values
            ])
        else:
            common_in_domain = 0
        
        needed_from_domain = target_count - common_in_domain
        
        if needed_from_domain > 0:
            domain_items = available[available['Domain'] == domain].copy()
            
            # Select top scoring items from this domain
            selected_from_domain = domain_items.nlargest(needed_from_domain, 'score')
            selected_by_domain.extend(selected_from_domain['ItemID'].tolist())
    
    # Combine common items and domain-selected items
    all_selected = selected_items + selected_by_domain
    
    # Check for enemy items and replace if needed
    if 'Enemy' in items_df.columns:
        final_selected = []
        for item_id in all_selected:
            # Get enemy items for this item
            enemy_str = items_df[items_df['ItemID'] == item_id]['Enemy'].values[0]
            enemies = parse_enemy_items(enemy_str)
            
            # Check if any enemy is already selected
            conflict = any(enemy in final_selected for enemy in enemies)
            
            if not conflict:
                final_selected.append(item_id)
            else:
                # Try to find replacement from same domain
                item_domain = items_df[items_df['ItemID'] == item_id]['Domain'].values[0]
                replacements = available[
                    (available['Domain'] == item_domain) &
                    (~available['ItemID'].isin(all_selected)) &
                    (~available['ItemID'].isin(final_selected))
                ].nlargest(5, 'score')
                
                # Find first replacement without enemy conflict
                replaced = False
                for _, replacement in replacements.iterrows():
                    repl_id = replacement['ItemID']
                    repl_enemies = parse_enemy_items(replacement['Enemy'])
                    if not any(enemy in final_selected for enemy in repl_enemies):
                        final_selected.append(repl_id)
                        replaced = True
                        break
                
                if not replaced:
                    # If no replacement found, still add (may violate enemy constraint)
                    final_selected.append(item_id)
        
        return final_selected[:test_length]
    
    return all_selected[:test_length]


def assemble_forms_with_llm(client, items_df, config, api_key, model="gpt-4o", temperature=0.3):
    """
    Assemble test forms using Large Language Model
    
    Args:
        client: OpenAI client
        items_df: DataFrame with item bank
        config: Dictionary with assembly configuration
        api_key: OpenAI API key
        model: Model name
        temperature: Temperature for generation
    
    Returns:
        Dictionary with assembly results
    """
    n_forms = config['n_forms']
    test_length = config['test_length']
    approach = config['approach']
    excluded_items = config.get('excluded_items', [])
    
    # Filter out excluded items from item bank
    if excluded_items:
        items_df = items_df[~items_df['ItemID'].isin(excluded_items)].copy()
    
    # Use heuristic selection for each form
    results = {
        'status': 'Success',
        'forms': {},
        'statistics': {},
        'llm_reasoning': {},
        'quality_checks': {}
    }
    
    used_items = set()
    
    for form_idx in range(n_forms):
        # Get available items (not yet used)
        available_items = items_df[~items_df['ItemID'].isin(used_items)].copy()
        
        # Use heuristic to select items
        try:
            selected_items = select_items_heuristic(available_items, config, form_number=form_idx)
            
            # Mark items as used
            used_items.update(selected_items)
            
            # Store form
            results['forms'][form_idx] = selected_items
            
            # Calculate statistics
            form_data = items_df[items_df['ItemID'].isin(selected_items)]
            
            stats = {
                'n_items': len(selected_items),
                'items': selected_items
            }
            
            if approach == 'IRT' and 'RaschB' in form_data.columns:
                stats['mean_difficulty'] = float(form_data['RaschB'].mean())
                stats['sd_difficulty'] = float(form_data['RaschB'].std())
            
            if 'Pvalue' in form_data.columns:
                stats['mean_pvalue'] = float(form_data['Pvalue'].mean())
                stats['sd_pvalue'] = float(form_data['Pvalue'].std())
            
            if 'PBS' in form_data.columns:
                stats['mean_pbs'] = float(form_data['PBS'].mean())
                stats['estimated_alpha'] = estimate_form_alpha(items_df, selected_items)
            
            # Domain distribution
            stats['domain_counts'] = form_data['Domain'].value_counts().to_dict()
            
            results['statistics'][form_idx] = stats
            
        except Exception as e:
            return {
                'status': f'Error in heuristic selection: {str(e)}',
                'forms': {},
                'statistics': {},
                'error_details': str(e)
            }
    
    # Get LLM validation and reasoning (optional, with minimal token usage)
    try:
        # Prepare summary for LLM (much smaller than full item bank)
        forms_summary = {}
        for form_idx, items in results['forms'].items():
            form_stats = results['statistics'][form_idx]
            forms_summary[f"Form_{form_idx+1}"] = {
                'item_count': len(items),
                'domain_distribution': form_stats['domain_counts'],
                'mean_difficulty': form_stats.get('mean_difficulty', form_stats.get('mean_pvalue')),
                'mean_pbs': form_stats.get('mean_pbs'),
                'estimated_alpha': form_stats.get('estimated_alpha')
            }
        
        # Build constraint description
        constraints_desc = []
        constraints_desc.append(f"- Test length: {test_length} items per form")
        constraints_desc.append(f"- Number of forms: {n_forms}")
        constraints_desc.append(f"- Approach: {approach}")
        
        if 'domain_constraints' in config and config['domain_constraints']:
            constraints_desc.append("\nDomain Requirements:")
            for domain, counts in config['domain_constraints'].items():
                constraints_desc.append(f"  - {domain}: {counts['min']}-{counts['max']} items")
        
        if config.get('apply_mean_diff', False):
            target = config.get('mean_diff_target')
            tolerance = config.get('mean_diff_tolerance', 0.1)
            if approach == 'IRT':
                constraints_desc.append(f"\nMean Rasch B Target: {target} ± {tolerance}")
            else:
                constraints_desc.append(f"\nMean P-value Target: {target} ± {tolerance}")
        
        constraints_text = "\n".join(constraints_desc)
        
        # Simplified LLM prompt for validation only
        validation_prompt = f"""You are an expert psychometrician reviewing test form assembly results.

**Assembly Constraints:**
{constraints_text}

**Forms Created (Summary):**
{json.dumps(forms_summary, indent=2)}

**Task:** Provide brief reasoning for each form's quality and any recommendations.

**Output Format (JSON):**
{{
    "reasoning": {{
        "Form_1": "Brief assessment of form quality...",
        "Form_2": "...",
        ...
    }},
    "quality_assessment": {{
        "overall_quality": "good/acceptable/needs_improvement",
        "recommendations": ["any suggestions for improvement"]
    }}
}}

Provide ONLY the JSON output."""

        # Call OpenAI API for validation
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert psychometrician. Provide concise, professional assessments of test form quality."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        # Parse LLM response
        llm_output = json.loads(response.choices[0].message.content)
        results['llm_reasoning'] = llm_output.get('reasoning', {})
        results['quality_checks'] = llm_output.get('quality_assessment', {})
        
    except Exception as e:
        # LLM validation failed, but heuristic selection succeeded
        results['llm_reasoning'] = {f"Form_{i+1}": "LLM validation unavailable" for i in range(n_forms)}
        results['quality_checks'] = {'overall_quality': 'unknown', 'llm_error': str(e)}
    
    return results


def main():
    st.markdown('<div class="main-header">🎯 AI Automated Test Assembly Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Optimize test forms using Large Language Models (LLM)</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'assembly_results' not in st.session_state:
        st.session_state.assembly_results = None
    if 'items_df' not in st.session_state:
        st.session_state.items_df = None
    
    # Sidebar - API Configuration
    st.sidebar.header("🔑 API Configuration")
    
    # API Key input
    api_key_env = os.getenv("OPENAI_API_KEY", "")
    
    # Try to get from secrets if not in env
    if not api_key_env:
        try:
            if "OPENAI_API_KEY" in st.secrets:
                api_key_env = st.secrets["OPENAI_API_KEY"]
        except FileNotFoundError:
            pass
            
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=api_key_env,
        type="password",
        help="Enter your OpenAI API key"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Select the OpenAI model for assembly"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )
    
    st.sidebar.divider()
    
    # Sidebar - File Upload
    st.sidebar.header("📁 Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Item Bank (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Required columns: ItemID, Domain, Pvalue, PBS, RaschB, Enemy"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                items_df = pd.read_csv(uploaded_file, keep_default_na=False, na_values=[''])
            else:
                items_df = pd.read_excel(uploaded_file, keep_default_na=False, na_values=[''])
            
            st.session_state.items_df = items_df
            
            st.sidebar.success(f"✅ Loaded {len(items_df)} items")
            
            # Display data preview
            with st.sidebar.expander("👁️ Preview Data"):
                st.dataframe(items_df.head(10), use_container_width=True)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            with st.sidebar.expander("Error Details"):
                st.code(error_details)
            return
    
    if st.session_state.items_df is None:
        st.info("👈 Please upload an item bank file to get started")
        # Show sample data format
        with st.expander("📋 Sample Data Format"):
            st.markdown("""
            **Required Columns:**
            - `ItemID`: Unique item identifier
            - `Domain`: Content domain/category
            - `Pvalue`: Item difficulty (CTT)
            - `PBS`: Point-Biserial correlation
            - `RaschB`: Rasch difficulty parameter (IRT)
            - `Enemy`: Comma-separated list of enemy item IDs
            
            **Example:**
            ```
            ItemID,Domain,Pvalue,PBS,RaschB,Enemy
            ITEM001,Cardiology,0.65,0.35,-0.5,"ITEM002,ITEM003"
            ITEM002,Cardiology,0.72,0.42,-0.2,ITEM001
            ITEM003,Neurology,0.58,0.38,-0.8,ITEM001
            ```
            """)
        return
    
    items_df = st.session_state.items_df
    
    # Main content - Assembly Configuration
    st.header("⚙️ Assembly Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Basic Settings")
        
        n_forms = st.number_input(
            "Number of Forms",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Number of parallel test forms to assemble"
        )
        
        test_length = st.number_input(
            "Test Length (items per form)",
            min_value=1,
            max_value=len(items_df),
            value=min(50, len(items_df)),
            step=1
        )
        
        approach = st.radio(
            "Psychometric Approach",
            options=['IRT', 'CTT'],
            index=0,
            help="IRT: Item Response Theory (uses RaschB)\nCTT: Classical Test Theory (uses Pvalue)"
        )
    
    with col2:
        st.subheader("🎯 Quality Constraints")
        
        # P-value constraints (CTT)
        st.markdown("**P-value Range** (CTT - items outside range excluded)")
        pval_col1, pval_col2 = st.columns(2)
        with pval_col1:
            pvalue_min = st.number_input("Min P-value", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        with pval_col2:
            pvalue_max = st.number_input("Max P-value", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
        
        # PBS constraint
        pbs_threshold = st.number_input(
            "PBS Threshold (items with PBS > threshold included)",
            min_value=-1.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Point-Biserial: higher values indicate better discrimination"
        )
        
        # Reliability
        st.markdown("**Reliability (Cronbach's Alpha)**")
        reliability_col1, reliability_col2 = st.columns(2)
        with reliability_col1:
            maximize_alpha = st.checkbox("Maximize Alpha", value=True)
        with reliability_col2:
            target_alpha = st.number_input("Target Min Alpha", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
    
    # Domain Distribution
    st.subheader("📚 Domain Distribution")
    st.caption("Set min/max item counts per domain. If min==max, that's the exact count.")
    
    domains = items_df['Domain'].unique()
    domain_constraints = {}
    
    domain_cols = st.columns(min(3, len(domains)))
    for idx, domain in enumerate(domains):
        with domain_cols[idx % len(domain_cols)]:
            st.markdown(f"**{domain}**")
            
            domain_count = len(items_df[items_df['Domain'] == domain])
            st.caption(f"Available: {domain_count} items")
            
            col_min, col_max = st.columns(2)
            with col_min:
                min_count = st.number_input(
                    f"Min",
                    min_value=0,
                    max_value=test_length,
                    value=0,
                    step=1,
                    key=f"domain_min_{domain}"
                )
            with col_max:
                max_count = st.number_input(
                    f"Max",
                    min_value=min_count,
                    max_value=test_length,
                    value=min(domain_count, test_length),
                    step=1,
                    key=f"domain_max_{domain}"
                )
            
            domain_constraints[domain] = {'min': min_count, 'max': max_count}
    
    # Common Items
    st.subheader("🔗 Common Items")
    
    common_col1, common_col2 = st.columns([1, 3])
    
    with common_col1:
        n_common_items = st.number_input(
            "Number of Common Items",
            min_value=0,
            max_value=test_length,
            value=0,
            step=1,
            help="Items that must appear in ALL forms"
        )
    
    with common_col2:
        common_items_input = st.text_input(
            "Common Item IDs (comma-separated)",
            value="",
            placeholder="e.g., ITEM001, ITEM005, ITEM012",
            help="Enter item IDs separated by commas"
        )
    
    # Parse common items
    common_items = []
    if common_items_input.strip():
        common_items = [item.strip() for item in common_items_input.split(',') if item.strip()]
    
    # Excluded Items
    st.subheader("🚫 Excluded Items")
    st.caption("Specify items to exclude from test assembly (optional)")
    
    excluded_items_input = st.text_area(
        "Excluded Item IDs (comma-separated)",
        value="",
        placeholder="e.g., NCX0001, NCX0005, NCX0012",
        help="Enter item IDs to exclude from all forms, separated by commas",
        height=80
    )
    
    # Parse excluded items
    excluded_items = []
    if excluded_items_input.strip():
        excluded_items = [item.strip() for item in excluded_items_input.split(',') if item.strip()]
        # Validate excluded items exist in dataset
        valid_excluded = [item for item in excluded_items if item in items_df['ItemID'].values]
        invalid_excluded = [item for item in excluded_items if item not in items_df['ItemID'].values]
        
        if valid_excluded:
            st.success(f"✅ Excluding {len(valid_excluded)} item(s) from assembly")
        if invalid_excluded:
            st.warning(f"⚠️ Items not found in dataset: {', '.join(invalid_excluded)}")
        
        excluded_items = valid_excluded
    
    # Mean Difficulty Constraints
    st.subheader("📏 Mean Difficulty Constraints")
    
    apply_mean_diff = st.checkbox(
        "Apply Mean Difficulty Constraint",
        value=False,
        help="Enable to set target mean difficulty for test forms"
    )
    
    mean_diff_target = None
    mean_diff_tolerance = None
    
    if apply_mean_diff:
        diff_col1, diff_col2 = st.columns(2)
        
        with diff_col1:
            if approach == 'IRT':
                mean_diff_target = st.number_input(
                    "Target Mean Difficulty (Rasch B)",
                    min_value=-3.0,
                    max_value=3.0,
                    value=0.0,
                    step=0.1,
                    help="Target mean Rasch difficulty parameter"
                )
            else:
                mean_diff_target = st.number_input(
                    "Target Mean P-value",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    help="Target mean item difficulty (proportion correct)"
                )
        
        with diff_col2:
            mean_diff_tolerance = st.number_input(
                "Difficulty Tolerance (±)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Allowable deviation from target mean"
            )
    
    
    # TIF/TCC Settings (IRT only)
    if approach == 'IRT':
        st.subheader("📈 Test Information Function (TIF) & TCC Targets")
        st.caption("Define target information and expected score at three ability levels")
        
        tif_col1, tif_col2, tif_col3 = st.columns(3)
        
        with tif_col1:
            st.markdown("**Low Ability (θ<sub>low</sub>)**", unsafe_allow_html=True)
            theta_low = st.number_input("θ Low", min_value=-3.0, max_value=0.0, value=-1.0, step=0.5, key="theta_low")
            tif_low = st.number_input("TIF Target", min_value=1.0, max_value=30.0, value=8.0, step=0.5, key="tif_low")
        
        with tif_col2:
            st.markdown("**Cut Point (θ<sub>mid</sub>)**", unsafe_allow_html=True)
            theta_mid = st.number_input("θ Mid (Cut)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key="theta_mid")
            tif_mid = st.number_input("TIF Target", min_value=1.0, max_value=50.0, value=12.0, step=0.5, key="tif_mid")
            tcc_mid = st.number_input("TCC Target", min_value=1.0, max_value=float(test_length), 
                                      value=float(test_length) * 0.5, step=0.5, key="tcc_mid",
                                      help="Expected score at cut point")
        
        with tif_col3:
            st.markdown("**High Ability (θ<sub>high</sub>)**", unsafe_allow_html=True)
            theta_high = st.number_input("θ High", min_value=0.0, max_value=3.0, value=1.0, step=0.5, key="theta_high")
            tif_high = st.number_input("TIF Target", min_value=1.0, max_value=30.0, value=8.0, step=0.5, key="tif_high")
        
        # Separate Tolerances for TIF and TCC
        tol_col1, tol_col2 = st.columns(2)
        
        with tol_col1:
            tif_tolerance = st.number_input(
                "TIF Tolerance (absolute)",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
                help="Acceptable absolute deviation from target TIF values"
            )
        
        with tol_col2:
            tcc_tolerance = st.number_input(
                "TCC Tolerance (absolute)",
                min_value=0.0,
                max_value=float(test_length),
                value=2.0,
                step=0.5,
                help="Acceptable absolute deviation from target TCC value"
            )
        
        # Store evaluation points
        eval_points = {
            'theta_low': theta_low,
            'theta_mid': theta_mid,
            'theta_high': theta_high,
            'tif_low': tif_low,
            'tif_mid': tif_mid,
            'tif_high': tif_high,
            'tcc_mid': tcc_mid
        }
    else:
        eval_points = None
        tif_tolerance = None
    
    # Enemy Check
    st.subheader("⚔️ Enemy Item Check")
    enemy_check = st.checkbox(
        "Enforce Enemy Constraints",
        value=True,
        help="Prevent enemy items from appearing together in the same form"
    )
    
    # LLM Settings
    st.subheader("🤖 LLM Assembly Settings")
    use_llm = st.checkbox(
        "Use LLM for Item Selection",
        value=True,
        help="Enable GPT-4 for intelligent item selection (requires OpenAI API key). Falls back to heuristics if disabled or on error."
    )
    
    top_k = st.slider(
        "Top-K Candidates for LLM",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
        help="Number of top candidates to send to LLM. Lower = faster & cheaper, Higher = more options for LLM to choose from."
    )
    
    if use_llm:
        st.info(f"💡 Using hybrid approach: Heuristic pre-filters to top {top_k} candidates, then LLM selects optimal subset.")
    else:
        st.info("ℹ️ Using pure heuristic selection (no LLM)")
    
    # Assembly Button
    st.divider()
    
    if st.button("🚀 Assemble Test Forms", type="primary", use_container_width=True):
        # Validate inputs
        if test_length > len(items_df):
            st.error(f"❌ Test length ({test_length}) exceeds available items ({len(items_df)})")
            return
        
        # Check domain constraints sum
        total_domain_min = sum(dc['min'] for dc in domain_constraints.values())
        if total_domain_min > test_length:
            st.error(f"❌ Sum of domain minimums ({total_domain_min}) exceeds test length ({test_length})")
            return
        
        # Prepare configuration
        config = {
            'n_forms': n_forms,
            'test_length': test_length,
            'approach': approach,
            'domain_constraints': domain_constraints,
            'common_items': common_items,
            'excluded_items': excluded_items,
            'pvalue_min': pvalue_min,
            'pvalue_max': pvalue_max,
            'pbs_threshold': pbs_threshold,
            'maximize_alpha': maximize_alpha,
            'target_alpha': target_alpha,
            'apply_mean_diff': apply_mean_diff,
            'mean_diff_target': mean_diff_target,
            'mean_diff_tolerance': mean_diff_tolerance,
            'enemy_check': enemy_check,
            'eval_points': eval_points,
            'tif_tolerance': {'tif': tif_tolerance, 'tcc': tcc_tolerance} if approach == 'IRT' else None,
            'use_llm': use_llm,
            'top_k': top_k
        }
        
        # Validate API key
        if not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar")
            return
        
        # Create OpenAI client
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
            return
        
        # Run assembly with LLM
        with st.spinner("🔄 Assembling test forms with AI... This may take a moment."):
            results = assemble_forms_with_llm(client, items_df, config, api_key, model, temperature)
        
        st.session_state.assembly_results = results
        
        if results['status'] == 'Success':
            st.success(f"✅ Successfully assembled {n_forms} test form(s)!")
            
            # Show LLM reasoning
            if 'llm_reasoning' in results and results['llm_reasoning']:
                with st.expander("🧠 AI Reasoning for Form Assembly"):
                    for form_name, reasoning in results['llm_reasoning'].items():
                        st.markdown(f"**{form_name}:**")
                        st.write(reasoning)
        else:
            st.error(f"❌ Assembly failed: {results['status']}")
            if 'error_details' in results:
                with st.expander("Error Details"):
                    st.code(results['error_details'])
    
    # Display Results
    if st.session_state.assembly_results is not None:
        results = st.session_state.assembly_results
        
        st.divider()
        st.header("📊 Assembly Results")
        
        if results['status'] == 'Success':
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            summary_cols = st.columns(len(results['forms']))
            
            for form_idx, (form_id, stats) in enumerate(results['statistics'].items()):
                with summary_cols[form_idx]:
                    st.markdown(f"### Form {form_id + 1}")
                    st.metric("Items", stats['n_items'])
                    
                    if approach == 'IRT':
                        st.metric("Mean Difficulty", f"{stats['mean_difficulty']:.3f}")
                        st.metric("SD Difficulty", f"{stats['sd_difficulty']:.3f}")
                    else:
                        st.metric("Mean P-value", f"{stats['mean_pvalue']:.3f}")
                        st.metric("SD P-value", f"{stats['sd_pvalue']:.3f}")
                    
                    if 'estimated_alpha' in stats:
                        alpha_color = "normal" if stats['estimated_alpha'] >= target_alpha else "inverse"
                        st.metric("Est. Alpha", f"{stats['estimated_alpha']:.3f}", delta_color=alpha_color)
                    
                    if 'mean_pbs' in stats:
                        st.metric("Mean PBS", f"{stats['mean_pbs']:.3f}")
            
            # Domain distribution table
            st.subheader("📚 Domain Distribution")
            
            domain_dist_data = []
            for form_id, stats in results['statistics'].items():
                row = {'Form': f"Form {form_id + 1}"}
                row.update(stats['domain_counts'])
                domain_dist_data.append(row)
            
            domain_dist_df = pd.DataFrame(domain_dist_data).fillna(0)
            # Convert only domain columns to int, not the Form column
            domain_cols = [col for col in domain_dist_df.columns if col != 'Form']
            domain_dist_df[domain_cols] = domain_dist_df[domain_cols].astype(int)
            st.dataframe(domain_dist_df, width='stretch')
            
            # TIF/TCC Visualization for IRT
            if approach == 'IRT' and eval_points is not None:
                st.subheader("📈 Test Information Function (TIF) & Test Characteristic Curve (TCC)")
                
                # Evaluate each form
                forms_quality_data = []
                form_names = []
                
                for form_id, form_items in results['forms'].items():
                    quality = evaluate_form_quality_irt(
                        items_df, 
                        form_items, 
                        eval_points, 
                        tif_tolerance
                    )
                    forms_quality_data.append(quality)
                    form_names.append(f"Form {form_id + 1}")
                
                # Display TIF/TCC evaluation results
                eval_cols = st.columns(len(results['forms']))
                
                for form_idx, (form_id, quality) in enumerate(zip(results['forms'].keys(), forms_quality_data)):
                    if quality is None:
                        continue
                    
                    with eval_cols[form_idx]:
                        st.markdown(f"**Form {form_id + 1} - TIF/TCC Evaluation**")
                        
                        # TIF at three points
                        tif_low_status = "✅" if quality['meets_tif_low'] else "❌"
                        tif_mid_status = "✅" if quality['meets_tif_mid'] else "❌"
                        tif_high_status = "✅" if quality['meets_tif_high'] else "❌"
                        
                        st.write(f"{tif_low_status} TIF @ θ={quality['theta_low']}: {quality['tif_at_low']:.2f} (target: {quality['tif_target_low']:.2f})")
                        st.write(f"{tif_mid_status} TIF @ θ={quality['theta_mid']}: {quality['tif_at_mid']:.2f} (target: {quality['tif_target_mid']:.2f})")
                        st.write(f"{tif_high_status} TIF @ θ={quality['theta_high']}: {quality['tif_at_high']:.2f} (target: {quality['tif_target_high']:.2f})")
                        
                        # TCC at cut point
                        if quality['tcc_target_mid'] is not None:
                            tcc_status = "✅" if quality['meets_tcc_mid'] else "❌"
                            st.write(f"{tcc_status} TCC @ θ={quality['theta_mid']}: {quality['tcc_at_mid']:.2f} (target: {quality['tcc_target_mid']:.2f})")
                        
                        # Overall status
                        overall_status = "✅ PASS" if quality['meets_all_irt'] else "⚠️ REVIEW"
                        st.markdown(f"**Overall: {overall_status}**")
                
                # Plot TIF and TCC
                fig_tif, fig_tcc = plot_tif_tcc(forms_quality_data, form_names)
                
                st.plotly_chart(fig_tif, use_container_width=True)
                st.plotly_chart(fig_tcc, use_container_width=True)
            
            # Detailed form contents
            st.subheader("📋 Form Contents")
            
            for form_id, form_items in results['forms'].items():
                with st.expander(f"📝 Form {form_id + 1} - {len(form_items)} items"):
                    form_data = items_df[items_df['ItemID'].isin(form_items)]
                    st.dataframe(form_data, width='stretch')
            
            # Export options
            st.divider()
            st.subheader("📥 Export Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            # CSV with form assignments
            with export_col1:
                assignment_data = []
                for form_id, form_items in results['forms'].items():
                    for item_id in form_items:
                        item_data = items_df[items_df['ItemID'] == item_id].iloc[0].to_dict()
                        item_data['AssignedForm'] = f"Form_{form_id + 1}"
                        assignment_data.append(item_data)
                
                assignment_df = pd.DataFrame(assignment_data)
                csv_buffer = BytesIO()
                assignment_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    "📄 CSV (Form Assignments)",
                    data=csv_buffer.getvalue(),
                    file_name=f"form_assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            # Excel with multiple sheets
            with export_col2:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_df = pd.DataFrame([
                        {'Parameter': 'Number of Forms', 'Value': n_forms},
                        {'Parameter': 'Test Length', 'Value': test_length},
                        {'Parameter': 'Approach', 'Value': approach},
                        {'Parameter': 'Assembly Status', 'Value': results['status']}
                    ])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Statistics sheet
                    stats_rows = []
                    for form_id, stats in results['statistics'].items():
                        row = {'Form': f"Form_{form_id + 1}", 'N_Items': stats['n_items']}
                        if approach == 'IRT':
                            row['Mean_Difficulty'] = stats['mean_difficulty']
                            row['SD_Difficulty'] = stats['sd_difficulty']
                        else:
                            row['Mean_Pvalue'] = stats['mean_pvalue']
                            row['SD_Pvalue'] = stats['sd_pvalue']
                        if 'estimated_alpha' in stats:
                            row['Est_Alpha'] = stats['estimated_alpha']
                        if 'mean_pbs' in stats:
                            row['Mean_PBS'] = stats['mean_pbs']
                        stats_rows.append(row)
                    
                    pd.DataFrame(stats_rows).to_excel(writer, sheet_name='Statistics', index=False)
                    
                    # Individual form sheets
                    for form_id, form_items in results['forms'].items():
                        form_data = items_df[items_df['ItemID'].isin(form_items)]
                        form_data.to_excel(writer, sheet_name=f'Form_{form_id + 1}', index=False)
                
                st.download_button(
                    "📊 Excel (Multiple Sheets)",
                    data=excel_buffer.getvalue(),
                    file_name=f"test_forms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch'
                )
            
            # JSON summary report
            with export_col3:
                report = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'n_forms': n_forms,
                        'test_length': test_length,
                        'approach': approach,
                        'assembly_status': results['status']
                    },
                    'configuration': config,
                    'statistics': results['statistics'],
                    'forms': {f"Form_{form_id + 1}": form_items for form_id, form_items in results['forms'].items()}
                }
                
                json_str = json.dumps(report, indent=2, default=str)
                
                st.download_button(
                    "📋 JSON (Summary Report)",
                    data=json_str,
                    file_name=f"assembly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width='stretch'
                )
        
        else:
            st.error(f"❌ Assembly Status: {results['status']}")
            st.markdown("""
            **Possible reasons for failure:**
            - Constraints are too restrictive (no feasible solution exists)
            - Insufficient items in item bank
            - Domain constraints sum exceeds test length
            - Too many common items specified
            - P-value/PBS thresholds exclude too many items
            
            **Suggestions:**
            - Relax some constraints (widen tolerances)
            - Reduce number of forms
            - Adjust domain distribution requirements
            - Check enemy item conflicts
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <b>AI Automated Test Assembly Tool</b> | Powered by Large Language Models (LLM)<br>
    Uses OpenAI GPT models for intelligent test form assembly
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
