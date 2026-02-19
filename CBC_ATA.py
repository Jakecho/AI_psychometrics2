"""
CBC_ATA - CBC Solver-Based Automated Test Assembly (Simultaneous)
====================================================
Sample bank: item_bank_hosted2.csv
Description:
Uses Mixed Integer Programming with CBC (Coin-or Branch and Cut) solver
for simultaneous multi-form test assembly. Works on Streamlit Community Cloud (free tier).

Features:
- Pure optimization approach (no LLM needed)
- Simultaneous assembly: All forms optimized in a single MIP problem
- Base Form Optimal Under CTT / Rasch: Max test information at logit cut
- Handles IRT (Rasch) (full TIF/TCC) and CTT constraints
- Domain distribution constraints
- Common items supported across forms
- Enemy item constraints enforced
- Free and fast

Author: AI Assistant
Date: February 9, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

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
        optional_cols = ['raschb_cat', 'enemy_ids', 'has_image']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info("Required columns: item_id, domain, rasch_b, pvalue, point_biserial")
            st.info("Optional columns: raschb_cat, enemy_ids, has_image")
            return None
        
        # Normalize ID types for reliable matching
        df['item_id'] = df['item_id'].astype(str)

        # Fill missing categories for consistent grouping
        df['domain'] = df['domain'].fillna('Unspecified')
        if 'raschb_cat' in df.columns:
            df['raschb_cat'] = df['raschb_cat'].fillna('Unspecified')

        # Notify about optional columns
        missing_optional = [col for col in optional_cols if col not in df.columns]
        if missing_optional:
            st.info(f"Optional columns not found: {missing_optional}. Those features will be disabled.")

        return df
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ==================== IRT Calculations ====================

# Scaling constant for Rasch model (D = 1.0 for logistic metric)
D = 1.0

def rasch_probability(theta: float, b: float) -> float:
    """Rasch model probability of correct response"""
    return 1.0 / (1.0 + np.exp(-D * (theta - b)))

def rasch_information(theta: float, b: float) -> float:
    """Item information at theta"""
    p = rasch_probability(theta, b)
    return (D ** 2) * p * (1 - p)

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

def assemble_forms_with_cbc(
    items_df: pd.DataFrame,
    config: Dict[str, Any],
    n_forms: int
) -> Dict[str, Any]:
    """Assemble one or more test forms simultaneously using the CBC solver."""

    if n_forms < 1:
        raise ValueError("n_forms must be at least 1")

    df = items_df.copy().reset_index(drop=True)
    if 'item_id' not in df.columns:
        raise ValueError("Item pool must include 'item_id'")

    df['item_id'] = df['item_id'].astype(str)
    n_items = len(df)

    if n_items == 0:
        return {
            'status': 'Infeasible',
            'objective_value': 0,
            'form_objectives': [],
            'selected_forms': [[] for _ in range(n_forms)],
            'solver': 'CBC'
        }

    test_length = config['test_length']
    approach = config.get('approach', 'IRT (Rasch)')
    use_ctt_mode = config.get('use_ctt_mode', False)
    domain_constraints = config.get('domain_constraints', {})
    raschb_cat_constraints = config.get('raschb_cat_constraints', {})
    image_constraint = config.get('image_constraint', {'min': 0, 'max': 0, 'enabled': False})

    excluded_items = {str(x) for x in config.get('excluded_items', [])}
    common_items = {str(x) for x in config.get('common_items', [])}

    # Filter common items to only those that exist in the dataframe
    valid_common_items = common_items & set(df['item_id'].astype(str))
    invalid_common_items = common_items - valid_common_items
    
    if invalid_common_items:
        error_msg = f"❌ Common items not found in item bank: {', '.join(sorted(invalid_common_items))}"
        return {
            'status': 'Infeasible',
            'objective_value': 0,
            'form_objectives': [],
            'selected_forms': [[] for _ in range(n_forms)],
            'validation_errors': [error_msg],
            'error_message': error_msg,
            'solver': 'CBC'
        }
    
    common_items = valid_common_items

    if excluded_items & common_items:
        raise ValueError("Common items cannot be in the excluded list")

    eval_points = config.get('eval_points', {}) or {}
    tif_tolerance_cfg = config.get('tif_tolerance') or {}
    pvalue_min = config.get('pvalue_min')
    pvalue_max = config.get('pvalue_max')
    pbs_threshold = config.get('pbs_threshold')
    enemy_check = config.get('enemy_check', False)

    mean_diff_target = config.get('mean_diff_target')
    mean_diff_tolerance = config.get('mean_diff_tolerance', 0.1)
    apply_mean_diff = config.get('apply_mean_diff', False)
    mean_difficulty_target = config.get('mean_difficulty_target')
    difficulty_tolerance = config.get('difficulty_tolerance')

    # Helper function for safe type conversion
    def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    # Pre-solve validation to catch infeasibility early
    validation_issues = []
    
    # Check 1: Domain constraint totals
    domain_min_sum = sum(constraints.get('min', 0) for constraints in domain_constraints.values())
    domain_max_sum = sum(
        constraints.get('max', 0) for constraints in domain_constraints.values() 
        if constraints.get('max', 0) > 0  # Only count non-zero max constraints
    )
    
    if domain_min_sum > test_length:
        validation_issues.append(
            f"Domain minimums sum ({domain_min_sum}) exceeds test length ({test_length})"
        )
    
    # Only validate max if constraints are specified
    if domain_max_sum > 0 and domain_max_sum < test_length:
        validation_issues.append(
            f"Domain maximums sum ({domain_max_sum}) less than test length ({test_length})"
        )
    
    # Check 2: Rasch B category constraint totals
    if raschb_cat_constraints:
        cat_min_sum = sum(constraints.get('min', 0) for constraints in raschb_cat_constraints.values())
        cat_max_sum = sum(
            constraints.get('max', 0) for constraints in raschb_cat_constraints.values()
            if constraints.get('max', 0) > 0  # Only count non-zero max constraints
        )
        
        if cat_min_sum > test_length:
            validation_issues.append(
                f"Rasch B category minimums sum ({cat_min_sum}) exceeds test length ({test_length})"
            )
        
        # Only validate max if constraints are specified
        if cat_max_sum > 0 and cat_max_sum < test_length:
            validation_issues.append(
                f"Rasch B category maximums sum ({cat_max_sum}) less than test length ({test_length})"
            )
    
    # Check 2b: Image constraint validation
    if 'has_image' in df.columns:
        image_min = image_constraint.get('min', 0)
        image_max = image_constraint.get('max', 0)
        items_with_images = df['has_image'].astype(int).sum()
        
        # Error: min set but max = 0
        if image_min > 0 and image_max == 0:
            validation_issues.append(
                f"Image minimum set ({image_min}) but maximum is 0. Set max > 0 to enable constraint."
            )
        
        # Validate if constraint is enabled (max > 0)
        if image_max > 0:
            if image_min > test_length:
                validation_issues.append(
                    f"Image minimum ({image_min}) exceeds test length ({test_length})"
                )
            
            if image_min > items_with_images:
                validation_issues.append(
                    f"Image minimum ({image_min}) exceeds available items with images ({items_with_images})"
                )
            
            if image_max < image_min:
                validation_issues.append(
                    f"Image maximum ({image_max}) less than minimum ({image_min})"
                )
    
    # Check 3: Common items validation
    if common_items:
        if len(common_items) > test_length:
            validation_issues.append(
                f"Common items count ({len(common_items)}) exceeds test length ({test_length})"
            )
        
        # Check if common items are in excluded or fail quality filters
        for common_id in common_items:
            if common_id in excluded_items:
                validation_issues.append(f"Common item {common_id} is also in excluded items")
            
            # Find the item in dataframe
            item_mask = df['item_id'] == common_id
            if not item_mask.any():
                validation_issues.append(f"Common item {common_id} NOT FOUND in item bank")
            elif item_mask.any():
                item_row = df[item_mask].iloc[0]
                
                # Only check CTT filters for CTT approach
                if use_ctt_mode or approach == 'CTT':
                    pval = _safe_float(item_row.get('pvalue'), None)
                    pbis = _safe_float(item_row.get('point_biserial'), None)
                    
                    if pvalue_min is not None and pvalue_max is not None:
                        if pval is None or pval < pvalue_min or pval > pvalue_max:
                            validation_issues.append(
                                f"Common item {common_id} fails p-value filter ({pval:.3f} not in [{pvalue_min:.3f}, {pvalue_max:.3f}])"
                            )
                    
                    if pbs_threshold is not None:
                        if pbis is None or pbis < pbs_threshold:
                            validation_issues.append(
                                f"Common item {common_id} fails discrimination threshold ({pbis:.3f} < {pbs_threshold:.3f})"
                            )
    
    # Check 4: Eligible item count
    eligible_mask = pd.Series([True] * n_items, index=range(n_items))
    
    # Only apply CTT filters for CTT approach
    if use_ctt_mode or approach == 'CTT':
        if pvalue_min is not None and pvalue_max is not None:
            eligible_mask &= (df['pvalue'] >= pvalue_min) & (df['pvalue'] <= pvalue_max)
        
        if pbs_threshold is not None:
            eligible_mask &= (df['point_biserial'] >= pbs_threshold)
    
    if excluded_items:
        eligible_mask &= ~df['item_id'].isin(excluded_items)
    
    eligible_count = eligible_mask.sum()
    required_count = test_length * n_forms
    
    # Account for common items that can be reused
    if common_items:
        # Check if common items are in eligible set
        common_eligible = 0
        for common_id in common_items:
            item_mask = (df['item_id'] == common_id) & eligible_mask
            if item_mask.any():
                common_eligible += 1
            else:
                validation_issues.append(
                    f"Common item {common_id} is filtered out and not eligible for assembly"
                )
        
        # Calculate unique slots needed (non-common items)
        unique_slots_available = eligible_count - common_eligible
        unique_slots_needed = required_count - (common_eligible * n_forms)
        
        if unique_slots_available < unique_slots_needed:
            validation_issues.append(
                f"Not enough unique eligible items: {unique_slots_available} available, {unique_slots_needed} needed "
                f"({common_eligible} common items in {n_forms} forms + {unique_slots_needed} unique items)"
            )
    else:
        if eligible_count < required_count:
            validation_issues.append(
                f"Not enough eligible items: {eligible_count} available, {required_count} needed "
                f"({test_length} per form × {n_forms} forms)"
            )
    
    # Report validation issues
    if validation_issues:
        error_msg = "⚠️ **FEASIBILITY ISSUES DETECTED:**\n\n"
        for i, issue in enumerate(validation_issues, 1):
            error_msg += f"{i}. {issue}\n"
        error_msg += "\n**Possible solutions:**\n"
        error_msg += "- Relax domain/category constraints\n"
        error_msg += "- Lower discrimination threshold\n"
        error_msg += "- Widen p-value range\n"
        error_msg += "- Remove excluded items\n"
        error_msg += "- Reduce number of forms or test length\n"
        
        return {
            'status': 'Infeasible',
            'objective_value': 0,
            'form_objectives': [],
            'selected_forms': [[] for _ in range(n_forms)],
            'solver': 'CBC',
            'validation_errors': validation_issues,
            'error_message': error_msg
        }
    
    # Variables declaration (moved here after validation)
    prob = LpProblem("Test_Assembly_Simultaneous", LpMaximize)

    item_vars: Dict[Tuple[int, int], LpVariable] = {}
    for item_idx in range(n_items):
        for form_idx in range(n_forms):
            var_name = f"x_item{item_idx}_form{form_idx}"
            item_vars[(item_idx, form_idx)] = LpVariable(var_name, cat='Binary')

    # Objective coefficients per item
    weights: List[float] = []
    if use_ctt_mode or approach == 'CTT':
        for _, row in df.iterrows():
            weights.append(_safe_float(row.get('point_biserial'), 0.0))
    elif approach == 'Base Form Optimal Under CTT / Rasch' and eval_points:
        theta_mid = eval_points.get('theta_mid', 0.0)
        for _, row in df.iterrows():
            b_param = _safe_float(row.get('rasch_b'), 0.0)
            weights.append(rasch_information(theta_mid, b_param))
    elif approach == 'IRT (Rasch)' and eval_points:
        theta_low = eval_points.get('theta_low', -1.0)
        theta_mid = eval_points.get('theta_mid', 0.0)
        theta_high = eval_points.get('theta_high', 1.0)
        for _, row in df.iterrows():
            b_param = _safe_float(row.get('rasch_b'), 0.0)
            info_low = rasch_information(theta_low, b_param) * 0.1
            info_mid = rasch_information(theta_mid, b_param) * 0.8
            info_high = rasch_information(theta_high, b_param) * 0.1
            weights.append(info_low + info_mid + info_high)
    else:
        for _, row in df.iterrows():
            weights.append(_safe_float(row.get('point_biserial'), 0.0))

    prob += lpSum(
        weights[item_idx] * item_vars[(item_idx, form_idx)]
        for item_idx in range(n_items)
        for form_idx in range(n_forms)
    )

    # Precompute index lists for faster constraint construction
    domain_item_indices = {
        domain: df.index[df['domain'] == domain].tolist()
        for domain in domain_constraints.keys()
    }

    raschb_item_indices = {}
    if raschb_cat_constraints and 'raschb_cat' in df.columns:
        raschb_item_indices = {
            category: df.index[df['raschb_cat'] == category].tolist()
            for category in raschb_cat_constraints.keys()
        }

    # 1. Test length for each form
    for form_idx in range(n_forms):
        prob += lpSum(item_vars[(item_idx, form_idx)] for item_idx in range(n_items)) == test_length

    # 2. Domain constraints per form
    for form_idx in range(n_forms):
        for domain, constraints in domain_constraints.items():
            indices = domain_item_indices.get(domain, [])
            min_count = constraints.get('min', 0)
            max_count = constraints.get('max', 0)

            if min_count > 0:
                prob += lpSum(item_vars[(idx, form_idx)] for idx in indices) >= min_count

            if max_count > 0 and max_count < test_length:
                prob += lpSum(item_vars[(idx, form_idx)] for idx in indices) <= max_count

    # 2b. Image constraints per form (third factor)
    if image_constraint.get('enabled', False) and 'has_image' in df.columns:
        image_indices = df.index[df['has_image'].astype(int) == 1].tolist()
        image_min = image_constraint.get('min', 0)
        image_max = image_constraint.get('max', 0)
        
        for form_idx in range(n_forms):
            if image_min > 0:
                prob += lpSum(item_vars[(idx, form_idx)] for idx in image_indices) >= image_min
            
            if image_max > 0 and image_max < test_length:
                prob += lpSum(item_vars[(idx, form_idx)] for idx in image_indices) <= image_max

    # 3. Rasch B category constraints per form
    if raschb_item_indices:
        for form_idx in range(n_forms):
            for category, constraints in raschb_cat_constraints.items():
                indices = raschb_item_indices.get(category, [])
                min_count = constraints.get('min', 0)
                max_count = constraints.get('max', 0)

                if min_count > 0:
                    prob += lpSum(item_vars[(idx, form_idx)] for idx in indices) >= min_count

                if max_count > 0 and max_count < test_length:
                    prob += lpSum(item_vars[(idx, form_idx)] for idx in indices) <= max_count

    # 4. P-value eligibility (CTT only)
    if use_ctt_mode or approach == 'CTT':
        if pvalue_min is not None and pvalue_max is not None:
            for item_idx in range(n_items):
                pval = _safe_float(df.at[item_idx, 'pvalue'], None)
                if pval is None or pval < pvalue_min or pval > pvalue_max:
                    for form_idx in range(n_forms):
                        prob += item_vars[(item_idx, form_idx)] == 0

    # 5. Discrimination threshold (CTT only)
    if use_ctt_mode or approach == 'CTT':
        if pbs_threshold is not None:
            for item_idx in range(n_items):
                pbis = _safe_float(df.at[item_idx, 'point_biserial'], None)
                if pbis is None or pbis < pbs_threshold:
                    for form_idx in range(n_forms):
                        prob += item_vars[(item_idx, form_idx)] == 0

    # 6. Explicit exclusions
    if excluded_items:
        for item_idx in range(n_items):
            item_id = df.at[item_idx, 'item_id']
            if item_id in excluded_items:
                for form_idx in range(n_forms):
                    prob += item_vars[(item_idx, form_idx)] == 0

    # 7. Common items present in every form
    if common_items:
        for item_idx in range(n_items):
            item_id = df.at[item_idx, 'item_id']
            if item_id in common_items:
                for form_idx in range(n_forms):
                    prob += item_vars[(item_idx, form_idx)] == 1

    # 8. Prevent reuse of non-common items across forms
    if n_forms > 1:
        for item_idx in range(n_items):
            item_id = df.at[item_idx, 'item_id']
            if item_id not in common_items:
                prob += lpSum(item_vars[(item_idx, form_idx)] for form_idx in range(n_forms)) <= 1

    # 9. Mean difficulty controls (optional)
    if apply_mean_diff and mean_diff_target is not None:
        for form_idx in range(n_forms):
            if use_ctt_mode or approach == 'CTT':
                avg_pval = (
                    lpSum(
                        _safe_float(df.at[item_idx, 'pvalue']) * item_vars[(item_idx, form_idx)]
                        for item_idx in range(n_items)
                    ) / test_length
                )
                prob += avg_pval >= mean_diff_target - mean_diff_tolerance
                prob += avg_pval <= mean_diff_target + mean_diff_tolerance
            else:
                avg_b = (
                    lpSum(
                        _safe_float(df.at[item_idx, 'rasch_b']) * item_vars[(item_idx, form_idx)]
                        for item_idx in range(n_items)
                    ) / test_length
                )
                prob += avg_b >= mean_diff_target - mean_diff_tolerance
                prob += avg_b <= mean_diff_target + mean_diff_tolerance

    if (use_ctt_mode or approach == 'CTT') and mean_difficulty_target is not None and difficulty_tolerance is not None:
        for form_idx in range(n_forms):
            avg_pval = (
                lpSum(
                    _safe_float(df.at[item_idx, 'pvalue']) * item_vars[(item_idx, form_idx)]
                    for item_idx in range(n_items)
                ) / test_length
            )
            prob += avg_pval >= mean_difficulty_target - difficulty_tolerance
            prob += avg_pval <= mean_difficulty_target + difficulty_tolerance

    # 10. Base form TIF minimums around logit cut
    if not use_ctt_mode and approach == 'Base Form Optimal Under CTT / Rasch' and eval_points:
        tolerance = eval_points.get('tolerance', 0.0)
        if tolerance > 0:
            theta_mid = eval_points.get('theta_mid', 0.0)
            theta_low_tol = theta_mid - tolerance
            theta_high_tol = theta_mid + tolerance
            min_tif = test_length * 0.20

            for form_idx in range(n_forms):
                tif_mid = lpSum(
                    rasch_information(theta_mid, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                    for item_idx in range(n_items)
                )
                tif_low = lpSum(
                    rasch_information(theta_low_tol, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                    for item_idx in range(n_items)
                )
                tif_high = lpSum(
                    rasch_information(theta_high_tol, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                    for item_idx in range(n_items)
                )

                prob += tif_mid >= min_tif
                prob += tif_low >= min_tif * 0.5
                prob += tif_high >= min_tif * 0.5

                if eval_points.get('tcc_enabled'):
                    tcc_mid_target = eval_points.get('tcc_mid', 0.0)
                    tcc_tol = eval_points.get('tcc_tolerance', 0.5)

                    if tcc_mid_target is not None and tcc_mid_target > 0:
                        tcc_mid_val = lpSum(
                            rasch_probability(theta_mid, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                            for item_idx in range(n_items)
                        )
                        prob += tcc_mid_val >= tcc_mid_target - tcc_tol
                        prob += tcc_mid_val <= tcc_mid_target + tcc_tol

                if eval_points.get('mean_rasch_enabled'):
                    mean_rasch_target = eval_points.get('mean_rasch_target')
                    mean_rasch_tolerance = eval_points.get('mean_rasch_tolerance', 0.2)

                    if mean_rasch_target is not None:
                        avg_b = (
                            lpSum(
                                _safe_float(df.at[item_idx, 'rasch_b']) * item_vars[(item_idx, form_idx)]
                                for item_idx in range(n_items)
                            ) / test_length
                        )
                        prob += avg_b >= mean_rasch_target - mean_rasch_tolerance
                        prob += avg_b <= mean_rasch_target + mean_rasch_tolerance

    # 11. IRT TIF/TCC targets per form
    if not use_ctt_mode and approach == 'IRT (Rasch)' and eval_points and tif_tolerance_cfg:
        tif_tol = tif_tolerance_cfg.get('tif', 1.5)
        tcc_tol = tif_tolerance_cfg.get('tcc', 1.0)
        tolerance = eval_points.get('tolerance', 0.0)

        theta_low = eval_points.get('theta_low', -1.0)
        theta_mid = eval_points.get('theta_mid', 0.0)
        theta_high = eval_points.get('theta_high', 1.0)

        tif_target_low = eval_points.get('tif_low', 0.0)
        tif_target_mid = eval_points.get('tif_mid', 0.0)
        tif_target_high = eval_points.get('tif_high', 0.0)

        # Only apply TIF constraints if targets are > 0
        if tif_target_low > 0 or tif_target_mid > 0 or tif_target_high > 0:
            for form_idx in range(n_forms):
                if tif_target_low > 0:
                    tif_low_val = lpSum(
                        rasch_information(theta_low, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                        for item_idx in range(n_items)
                    )
                    prob += tif_low_val >= tif_target_low - tif_tol
                    prob += tif_low_val <= tif_target_low + tif_tol
                
                if tif_target_mid > 0:
                    tif_mid_val = lpSum(
                        rasch_information(theta_mid, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                        for item_idx in range(n_items)
                    )
                    prob += tif_mid_val >= tif_target_mid - tif_tol
                    prob += tif_mid_val <= tif_target_mid + tif_tol
                
                if tif_target_high > 0:
                    tif_high_val = lpSum(
                        rasch_information(theta_high, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                        for item_idx in range(n_items)
                    )
                    prob += tif_high_val >= tif_target_high - tif_tol
                    prob += tif_high_val <= tif_target_high + tif_tol

            if tolerance > 0:
                min_tif = test_length * 0.20
                theta_low_tol = theta_mid - tolerance
                theta_high_tol = theta_mid + tolerance

                tif_low_tol = lpSum(
                    rasch_information(theta_low_tol, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                    for item_idx in range(n_items)
                )
                tif_high_tol = lpSum(
                    rasch_information(theta_high_tol, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                    for item_idx in range(n_items)
                )

                prob += tif_low_tol >= min_tif * 0.5
                prob += tif_high_tol >= min_tif * 0.5

            if tcc_tol is not None:
                # Only apply TCC constraint at mid point (logit cut) if target > 0
                tcc_mid_target = eval_points.get('tcc_mid', 0.0)

                if tcc_mid_target > 0:
                    tcc_mid_val = lpSum(
                        rasch_probability(theta_mid, _safe_float(df.at[item_idx, 'rasch_b'])) * item_vars[(item_idx, form_idx)]
                        for item_idx in range(n_items)
                    )

                    prob += tcc_mid_val >= tcc_mid_target - tcc_tol
                    prob += tcc_mid_val <= tcc_mid_target + tcc_tol

            # Apply mean Rasch difficulty constraint if enabled
            if eval_points.get('mean_rasch_enabled'):
                mean_rasch_target = eval_points.get('mean_rasch_target')
                mean_rasch_tolerance = eval_points.get('mean_rasch_tolerance', 0.2)

                if mean_rasch_target is not None:
                    avg_b = (
                        lpSum(
                            _safe_float(df.at[item_idx, 'rasch_b']) * item_vars[(item_idx, form_idx)]
                            for item_idx in range(n_items)
                        ) / test_length
                    )
                    prob += avg_b >= mean_rasch_target - mean_rasch_tolerance
                    prob += avg_b <= mean_rasch_target + mean_rasch_tolerance

    # 12. Enemy item constraints per form
    if enemy_check:
        enemy_col = None
        if 'enemy_ids' in df.columns:
            enemy_col = 'enemy_ids'
        elif 'enemy' in df.columns:
            enemy_col = 'enemy'

        if enemy_col:
            item_id_to_index = {df.at[idx, 'item_id']: idx for idx in range(n_items)}
            enemy_pairs = set()

            for item_idx in range(n_items):
                enemy_str = df.at[item_idx, enemy_col] if enemy_col in df.columns else ''
                if pd.notna(enemy_str) and str(enemy_str).strip():
                    for enemy_id in str(enemy_str).split(','):
                        enemy_id = enemy_id.strip()
                        if not enemy_id:
                            continue
                        if enemy_id in item_id_to_index and enemy_id != df.at[item_idx, 'item_id']:
                            other_idx = item_id_to_index[enemy_id]
                            pair = tuple(sorted((item_idx, other_idx)))
                            enemy_pairs.add(pair)

            for item_a, item_b in enemy_pairs:
                for form_idx in range(n_forms):
                    prob += item_vars[(item_a, form_idx)] + item_vars[(item_b, form_idx)] <= 1

    solver = PULP_CBC_CMD(
        msg=0,           # Silent mode (errors only)
        timeLimit=360     # 6-minute limit
    )
    
    # Debug: Check problem size
    prob_vars = len(prob.variables())
    prob_cons = len(prob.constraints)
    
    try:
        prob.solve(solver)
    except Exception as e:
        error_info = (
            f"CBC Solver Failed:\n"
            f"  Problem Size: {prob_vars} variables, {prob_cons} constraints\n"
            f"  Items: {n_items}, Forms: {n_forms}\n"
            f"  Common items: {len(common_items)}, Unique items needed per form: {test_length - len(common_items)}\n"
            f"  Error: {str(e)}\n\n"
            f"Possible causes:\n"
            f"  - Constraints are too tight (infeasible)\n"
            f"  - Item pool too small for test length and form count\n"
            f"  - Domain/category constraints conflict\n"
        )
        raise Exception(error_info)

    status = LpStatus.get(prob.status, 'Unknown')
    objective_value = value(prob.objective) if prob.objective else 0
    
    # If infeasible, provide detailed diagnostics
    if status == 'Infeasible':
        diagnostic_msg = (
            f"⚠️ **SOLVER INFEASIBILITY DIAGNOSTICS:**\n\n"
            f"**Problem Configuration:**\n"
            f"- Forms: {n_forms}\n"
            f"- Items per form: {test_length}\n"
            f"- Common items: {len(common_items)}\n"
            f"- Unique items needed: {test_length - len(common_items)} per form\n"
            f"- Total pool: {n_items} items\n\n"
            f"**Problem Size:**\n"
            f"- Variables: {prob_vars}\n"
            f"- Constraints: {prob_cons}\n\n"
            f"**Possible Issues:**\n"
            f"1. Domain/Rasch B constraints too restrictive\n"
            f"2. Item pool lacks sufficient items in constrained domains\n"
            f"3. Excluded items overlaps with required items\n"
            f"4. TIF/TCC targets conflicting with form structure\n"
        )
        return {
            'status': 'Infeasible',
            'objective_value': 0,
            'form_objectives': [],
            'selected_forms': [[] for _ in range(n_forms)],
            'validation_errors': [diagnostic_msg],
            'error_message': diagnostic_msg,
            'solver': 'CBC'
        }
    
    selected_forms: List[List[str]] = []
    for form_idx in range(n_forms):
        form_selection: List[str] = []
        for item_idx in range(n_items):
            var_value = item_vars[(item_idx, form_idx)].varValue
            if var_value is not None and var_value > 0.5:
                form_selection.append(df.at[item_idx, 'item_id'])
        selected_forms.append(form_selection)

    form_objectives: List[float] = []
    for form_idx in range(n_forms):
        form_value = 0.0
        for item_idx in range(n_items):
            var_value = item_vars[(item_idx, form_idx)].varValue
            if var_value is not None and var_value > 0.5:
                form_value += weights[item_idx]
        form_objectives.append(form_value)

    return {
        'status': status,
        'objective_value': objective_value,
        'form_objectives': form_objectives,
        'selected_forms': selected_forms,
        'solver': 'CBC'
    }


def assemble_form_with_cbc(
    items_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[List[str], Dict[str, Any]]:
    """Backward-compatible single-form wrapper around simultaneous solver."""

    result = assemble_forms_with_cbc(items_df, config, 1)
    selected_forms = result.get('selected_forms', [])
    selected_ids = selected_forms[0] if selected_forms else []

    assembly_info = {
        'status': result.get('status', 'Unknown'),
        'objective_value': result.get('objective_value', 0),
        'n_items_selected': len(selected_ids),
        'solver': result.get('solver', 'CBC')
    }

    return selected_ids, assembly_info

# ==================== Evaluation ====================

def evaluate_form(items_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate assembled form quality"""
    
    approach = config.get('approach', 'IRT (Rasch)')
    b_params = items_df['rasch_b'].values
    
    # Calculate TIF and TCC across theta range
    theta_range = np.linspace(-3, 3, 61)
    tif_values = [calculate_tif(theta, b_params) for theta in theta_range]
    tcc_values = [calculate_tcc(theta, b_params) for theta in theta_range]
    
    # Basic statistics - use appropriate difficulty metric
    if approach == 'CTT':
        mean_difficulty = items_df['pvalue'].mean()
        sd_difficulty = items_df['pvalue'].std()
    else:  # IRT or Base Form (both use Rasch)
        mean_difficulty = items_df['rasch_b'].mean()
        sd_difficulty = items_df['rasch_b'].std()
    
    stats = {
        'theta_range': theta_range,
        'tif_values': tif_values,
        'tcc_values': tcc_values,
        'mean_difficulty': mean_difficulty,
        'sd_difficulty': sd_difficulty,
        'mean_discrimination': items_df['point_biserial'].mean(),
        'domain_counts': items_df['domain'].value_counts().sort_index().to_dict()
    }

    if 'raschb_cat' in items_df.columns:
        stats['raschb_cat_counts'] = items_df['raschb_cat'].value_counts().sort_index().to_dict()
    
    # Evaluation at specific points if provided - use EXACT theta values
    if config.get('eval_points'):
        eval_points = config['eval_points']
        
        for point_name, theta_val in [('low', eval_points.get('theta_low')), 
                                      ('mid', eval_points.get('theta_mid')), 
                                      ('high', eval_points.get('theta_high'))]:
            if theta_val is not None:
                # Calculate at exact theta value instead of nearest discrete point
                stats[f'tif_at_{point_name}'] = calculate_tif(theta_val, b_params)
                stats[f'tcc_at_{point_name}'] = calculate_tcc(theta_val, b_params)
    
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

def display_form_results(form_data: Dict[str, Any], eval_points: Dict, common_items: List[str], approach: str, enemy_check: bool = False, use_ctt_mode: bool = False):
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
        'Domain': sorted(stats['domain_counts'].keys()),
        'Count': [stats['domain_counts'][k] for k in sorted(stats['domain_counts'].keys())]
    })
    st.dataframe(domain_df, use_container_width=True, hide_index=True)

    if stats.get('raschb_cat_counts'):
        st.subheader("🏷️ Rasch B Category Distribution")
        raschb_df = pd.DataFrame({
            'Rasch B Category': sorted(stats['raschb_cat_counts'].keys()),
            'Count': [stats['raschb_cat_counts'][k] for k in sorted(stats['raschb_cat_counts'].keys())]
        })
        st.dataframe(raschb_df, use_container_width=True, hide_index=True)
    
    # Summary metrics and table
    st.subheader("📋 Summary")
    
    # Display as key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Length", len(selected_df))
        st.metric("Mean P-value", f"{selected_df['pvalue'].mean():.3f}")
        st.metric("Mean Rasch B", f"{stats['mean_difficulty']:.3f}")
    with col2:
        st.metric("Cronbach Alpha", f"{alpha:.3f}")
        st.metric("Mean Discrimination", f"{stats['mean_discrimination']:.3f}")
        st.metric("SD Rasch B", f"{stats['sd_difficulty']:.3f}")
    with col3:
        st.metric("Enemy Check", "✅ Enabled" if enemy_check else "⭕ Disabled")
        if not use_ctt_mode and tif_at_cut is not None:
            st.metric(f"TIF @ θ={theta_cut:.3f}", f"{tif_at_cut:.2f}")
            st.metric(f"TCC @ θ={theta_cut:.3f}", f"{tcc_at_cut:.2f}")
    
    # Additional TIF/TCC values at ±1 (skip in CTT mode)
    if not use_ctt_mode and tif_at_cut is not None:
        if form_data.get('tif_at_low') is not None or form_data.get('tif_at_high') is not None:
            st.subheader("📊 Additional TIF/TCC Values")
            add_col1, add_col2 = st.columns(2)
            with add_col1:
                if form_data.get('tif_at_low') is not None:
                    st.metric(f"TIF @ θ={theta_cut-1:.3f}", f"{form_data['tif_at_low']:.2f}")
                    st.metric(f"TCC @ θ={theta_cut-1:.3f}", f"{form_data['tcc_at_low']:.2f}")
            with add_col2:
                if form_data.get('tif_at_high') is not None:
                    st.metric(f"TIF @ θ={theta_cut+1:.3f}", f"{form_data['tif_at_high']:.2f}")
                    st.metric(f"TCC @ θ={theta_cut+1:.3f}", f"{form_data['tcc_at_high']:.2f}")

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
        help="Required columns: item_id, domain, rasch_b, pvalue, point_biserial. Optional: raschb_cat, enemy_ids, has_image"
    )
    
    if uploaded_file is None:
        st.warning("⬆️ Please upload an item pool file to begin")
        
        # Show example format
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
        min_value=0, 
        max_value=200, 
        value=0, 
        step=1,
        help="Number of items in each test form (0 = not set)"
    )
    
    approach = st.sidebar.radio(
        "Approach", 
        options=['Base Form Optimal Under CTT / Rasch', 'IRT (Rasch)', 'CTT'],
        index=0,
        help="Base Form: Max info at logit cut | IRT (Rasch): Rasch with TIF/TCC targets | CTT: Classical statistics"
    )
    
    st.sidebar.divider()
    
    # Domain distribution - tidy layout
    st.sidebar.subheader("📚 Domain Distribution")
    st.sidebar.caption("Set minimum and maximum items per domain (0 = no constraint)")

    domain_constraints = {}

    # Default: no domain constraints (all set to 0)
    n_domains = len(domains)
    default_min = 0  # No minimum constraint
    default_max = 0  # No maximum constraint (0 = unconstrained)

    domain_constraints_df = pd.DataFrame({
        'Domain': domains,
        'Min': [default_min] * n_domains,
        'Max': [default_max] * n_domains
    })

    domain_constraints_df = st.sidebar.data_editor(
        domain_constraints_df,
        use_container_width=True,
        hide_index=True,
        key="domain_constraints_editor",
        column_config={
            'Domain': st.column_config.TextColumn('Domain', disabled=True),
            'Min': st.column_config.NumberColumn('Min', min_value=0, step=1),
            'Max': st.column_config.NumberColumn('Max', min_value=0, step=1)
        }
    )

    for _, row in domain_constraints_df.iterrows():
        domain = row['Domain']
        min_val = int(row['Min']) if pd.notna(row['Min']) else 0
        max_val_raw = int(row['Max']) if pd.notna(row['Max']) else min_val
        max_val = max(min_val, min(max_val_raw, test_length))
        domain_constraints[domain] = {'min': min_val, 'max': max_val}

    st.sidebar.divider()
    
    # Image constraint (third factor)
    st.sidebar.subheader("🖼️ Image Constraint")
    st.sidebar.caption("Control items with images/figures (has_image column: 0 or 1). Set max to enable.")
    
    image_constraint = {'min': 0, 'max': 0, 'enabled': False}
    if 'has_image' in items_df.columns:
        # Count items with images
        items_with_image = items_df['has_image'].astype(int).sum()
        st.sidebar.info(f"📊 Items with images: {items_with_image} / {len(items_df)}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            image_min = st.number_input(
                "Min Items with Images",
                min_value=0,
                max_value=test_length if test_length > 0 else 100,
                value=0,
                step=1,
                help="Minimum number of items with images (requires max to be set)"
            )
        with col2:
            image_max = st.number_input(
                "Max Items with Images",
                min_value=0,
                max_value=test_length if test_length > 0 else 100,
                value=0,
                step=1,
                help="Maximum number of items with images (0 = no constraint)"
            )
        
        image_constraint = {
            'min': image_min,
            'max': image_max,
            'enabled': image_max > 0
        }
    else:
        st.sidebar.warning("⚠️ Column 'has_image' not found. Image constraints disabled.")
    
    st.sidebar.divider()

    # Rasch B category distribution - tidy layout
    raschb_cat_constraints = {}
    if 'raschb_cat' in items_df.columns:
        raschb_cats = sorted(items_df['raschb_cat'].unique().tolist())
        st.sidebar.subheader("🏷️ Rasch B Category Distribution")
        st.sidebar.caption("Set minimum and maximum items per Rasch B category (0 = no constraint)")

        n_cats = len(raschb_cats)
        default_cat_min = 0  # No minimum constraint
        default_cat_max = 0  # No maximum constraint (0 = unconstrained)

        raschb_constraints_df = pd.DataFrame({
            'Rasch B Category': raschb_cats,
            'Min': [default_cat_min] * n_cats,
            'Max': [default_cat_max] * n_cats
        })

        raschb_constraints_df = st.sidebar.data_editor(
            raschb_constraints_df,
            use_container_width=True,
            hide_index=True,
            key="raschb_cat_constraints_editor",
            column_config={
                'Rasch B Category': st.column_config.TextColumn('Rasch B Category', disabled=True),
                'Min': st.column_config.NumberColumn('Min', min_value=0, step=1),
                'Max': st.column_config.NumberColumn('Max', min_value=0, step=1)
            }
        )

        for _, row in raschb_constraints_df.iterrows():
            cat = row['Rasch B Category']
            min_val = int(row['Min']) if pd.notna(row['Min']) else 0
            max_val_raw = int(row['Max']) if pd.notna(row['Max']) else min_val
            max_val = max(min_val, min(max_val_raw, test_length))
            raschb_cat_constraints[cat] = {'min': min_val, 'max': max_val}
    else:
        st.sidebar.info("Rasch B category column (raschb_cat) not found. Category constraints are disabled.")
    
    st.sidebar.divider()
    
    # Common Items
    st.sidebar.subheader("🔗 Common Items")
    st.sidebar.caption("Items that MUST appear in all forms")
    
    common_items_str = st.sidebar.text_input(
        "Common Item IDs (comma-separated)",
        value="",
        help="e.g., NCX0214,NCX0215. These items will appear in ALL forms.",
        placeholder="NCX0214,NCX0215"
    )
    
    # Parse common items
    common_items = []
    if common_items_str.strip():
        try:
            common_items = [x.strip() for x in common_items_str.split(',') if x.strip()]
            st.sidebar.info(f"🔗 {len(common_items)} common item(s) specified")
        except ValueError:
            st.sidebar.error("❌ Invalid item IDs. Use comma-separated IDs.")
    
    # Excluded Items
    st.sidebar.subheader("🚫 Excluded Items")
    st.sidebar.caption("Items to exclude from all forms")
    
    excluded_items_str = st.sidebar.text_input(
        "Excluded Item IDs (comma-separated)",
        value="",
        help="e.g., NCX0450,NCX0782",
        placeholder="NCX0450,NCX0782"
    )
    
    # Parse excluded items
    excluded_items = []
    if excluded_items_str.strip():
        try:
            excluded_items = [x.strip() for x in excluded_items_str.split(',') if x.strip()]
            st.sidebar.info(f"🚫 Excluding {len(excluded_items)} items")
        except ValueError:
            st.sidebar.error("❌ Invalid item IDs. Use comma-separated IDs.")
    
    st.sidebar.divider()
    
    # Quality constraints
    st.sidebar.subheader("⚙️ Quality Constraints")
    use_ctt_mode = st.sidebar.checkbox(
        "CTT (w/ Max Reliability)",
        value=False,
        help="Use CTT statistics (p-value, point-biserial) and maximize reliability. Ignores Rasch parameters."
    )
    
    enemy_check = st.sidebar.checkbox(
        "Enforce Enemy Constraints",
        value=True,
        help="Prevent enemy items (marked in 'enemy_ids' column) from appearing together in the same form"
    )
    
    # Evaluation Points and Constraints
    eval_points = None
    tif_tolerance = None
    tcc_tolerance = None
    logit_cut = 0.0
    mean_difficulty_target = None
    difficulty_tolerance = None
    pvalue_min = 0.0
    pvalue_max = 1.0
    discrimination_min = 0.0

    # CTT Mode: Show mean difficulty constraints instead of logit cut
    if use_ctt_mode:
        st.sidebar.subheader("📊 Mean Difficulty Target")
        st.sidebar.markdown("**Mean Difficulty (P-value):**")
        mean_difficulty_target = st.sidebar.number_input(
            "Target Mean P-value",
            0.0, 1.0, 0.6, 0.05,
            help="Target average difficulty (p-value) for the test",
            key="ctt_mean_target"
        )
        difficulty_tolerance = st.sidebar.number_input(
            "Tolerance (±)",
            0.01, 0.5, 0.1, 0.01,
            help="Acceptable deviation from target mean p-value",
            key="ctt_tolerance"
        )
    # Base Form Optimal Under CTT / Rasch
    elif approach == 'Base Form Optimal Under CTT / Rasch':
        st.sidebar.subheader("📐 Logit Cut")
        logit_cut = st.sidebar.number_input(
            "Logit Cut (θ)",
            -3.0, 3.0, 0.0, 0.001,
            help="Objective: Maximize test information at this θ value",
            format="%.3f"
        )

        tolerance = st.sidebar.number_input(
            "Tolerance (±)",
            0.0, 10.0, 0.2, 0.1,
            help="Apply the minimum TIF requirement around the logit cut (θ ± tolerance)",
            key="base_form_tolerance"
        )

        st.sidebar.markdown("**Optional Constraints**")
        
        enable_tcc_cut = st.sidebar.checkbox(
            "Enable TCC at Logit Cut",
            value=False,
            help="Add a TCC constraint at the logit cut"
        )

        tcc_mid = None
        tcc_tolerance = None
        if enable_tcc_cut:
            tcc_mid = st.sidebar.number_input(
                f"TCC @ {logit_cut:.3f}",
                0.0, float(test_length), 0.0, 1.0,
                help="Target expected score at the logit cut"
            )
            tcc_tolerance = st.sidebar.number_input(
                "TCC Tolerance (±)",
                0.1, 20.0, 0.5, 0.1,
                help="Allowed deviation from the TCC target"
            )

        enable_mean_rasch = st.sidebar.checkbox(
            "Enable Mean Rasch B",
            value=False,
            help="Add a mean Rasch difficulty constraint"
        )
        mean_rasch_target = None
        mean_rasch_tolerance = None
        if enable_mean_rasch:
            mean_rasch_target = st.sidebar.number_input(
                "Mean Rasch B Target",
                -3.0, 3.0, 0.0, 0.01,
                help="Target mean Rasch difficulty for the form"
            )
            mean_rasch_tolerance = st.sidebar.number_input(
                "Mean Rasch B Tolerance (±)",
                0.0, 2.0, 0.2, 0.05,
                help="Allowed deviation from the mean Rasch target"
            )

        st.sidebar.info("🎯 Objective: Max test information at logit cut")

        # Store in eval_points for consistency, including -1 and +1 points
        eval_points = {
            'theta_low': logit_cut - 1.0,
            'theta_mid': logit_cut,
            'theta_high': logit_cut + 1.0,
            'tolerance': tolerance,
            'tcc_enabled': enable_tcc_cut,
            'tcc_mid': tcc_mid,
            'tcc_tolerance': tcc_tolerance,
            'mean_rasch_enabled': enable_mean_rasch,
            'mean_rasch_target': mean_rasch_target,
            'mean_rasch_tolerance': mean_rasch_tolerance
        }

    # IRT (Rasch) with full evaluation points
    elif approach == 'IRT (Rasch)':
        st.sidebar.subheader("📐 IRT (Rasch) Evaluation Points")
        logit_cut = st.sidebar.number_input("Logit Cut (θ)", -3.0, 3.0, 0.0, 0.001, format="%.3f")
        
        tolerance = st.sidebar.number_input(
            "Tolerance (±)",
            0.0, 10.0, 0.2, 0.1,
            help="Minimum TIF acceptable at logit cut ± this range",
            key="irt_tolerance"
        )
        
        evaluation_points = [logit_cut - 1.0, logit_cut, logit_cut + 1.0]
        st.sidebar.markdown(f"**Points:** {evaluation_points[0]:.3f}, {evaluation_points[1]:.3f}, {evaluation_points[2]:.3f}")
        
        # TIF Targets
        st.sidebar.markdown("**TIF Targets:**")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            tif_low = st.number_input(f"TIF @ {evaluation_points[0]:.3f}", 0.0, 50.0, 0.0, 0.5, key="tif_low", help="(0 = no constraint)")
        with col2:
            tif_mid = st.number_input(f"TIF @ {evaluation_points[1]:.3f}", 0.0, 50.0, 0.0, 0.5, key="tif_mid", help="(0 = no constraint)")
        with col3:
            tif_high = st.number_input(f"TIF @ {evaluation_points[2]:.3f}", 0.0, 50.0, 0.0, 0.5, key="tif_high", help="(0 = no constraint)")
        
        # TCC Targets
        st.sidebar.markdown("**TCC Targets (Expected Score):**")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            tcc_low = st.number_input(f"TCC @ {evaluation_points[0]:.3f}", 0.0, float(test_length), 0.0, 1.0, key="tcc_low", help="(0 = no constraint)")
        with col2:
            tcc_mid = st.number_input(f"TCC @ {evaluation_points[1]:.3f}", 0.0, float(test_length), 0.0, 1.0, key="tcc_mid", help="(0 = no constraint)")
        with col3:
            tcc_high = st.number_input(f"TCC @ {evaluation_points[2]:.3f}", 0.0, float(test_length), 0.0, 1.0, key="tcc_high", help="(0 = no constraint)")
        
        # Tolerances
        st.sidebar.markdown("**Tolerances:**")
        tol_col1, tol_col2 = st.sidebar.columns(2)
        with tol_col1:
            tif_tolerance = st.number_input("TIF Tolerance (±)", 0.1, 10.0, 0.2, 0.1)
        with tol_col2:
            tcc_tolerance = st.number_input("TCC Tolerance (±)", 0.1, 20.0, 1.0, 0.5)
        
        # Mean Rasch Difficulty constraint
        st.sidebar.divider()
        st.sidebar.markdown("**Optional Constraint**")
        
        enable_mean_rasch_irt = st.sidebar.checkbox(
            "Enable Mean Rasch Difficulty",
            value=False,
            help="Add a mean Rasch difficulty constraint"
        )
        
        mean_rasch_irt_target = None
        mean_rasch_irt_tolerance = None
        if enable_mean_rasch_irt:
            mean_rasch_irt_target = st.sidebar.number_input(
                "Mean Rasch B Target",
                -3.0, 3.0, 0.0, 0.01,
                help="Target mean Rasch difficulty for the form"
            )
            mean_rasch_irt_tolerance = st.sidebar.number_input(
                "Mean Rasch B Tolerance (±)",
                0.0, 2.0, 0.2, 0.05,
                help="Allowed deviation from the mean Rasch target"
            )
        
        eval_points = {
            'theta_low': evaluation_points[0],
            'theta_mid': evaluation_points[1],
            'theta_high': evaluation_points[2],
            'tif_low': tif_low,
            'tif_mid': tif_mid,
            'tif_high': tif_high,
            'tcc_low': tcc_low,
            'tcc_mid': tcc_mid,
            'tcc_high': tcc_high,
            'tolerance': tolerance,
            'mean_rasch_enabled': enable_mean_rasch_irt,
            'mean_rasch_target': mean_rasch_irt_target,
            'mean_rasch_tolerance': mean_rasch_irt_tolerance
        }

    # CTT Approach (when not using CTT mode checkbox)
    elif approach == 'CTT' and not use_ctt_mode:
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
            pvalue_min = st.number_input("Min P-value", 0.0, 1.0, 0.25, 0.05)
        with pval_col2:
            pvalue_max = st.number_input("Max P-value", 0.0, 1.0, 0.95, 0.05)
        
        # Discrimination threshold
        discrimination_min = st.sidebar.number_input(
            "Min Discrimination",
            0.0, 1.0, 0.15, 0.05,
            help="Minimum point-biserial correlation"
        )
    
    # Assemble button
    st.sidebar.divider()
    if st.sidebar.button("🚀 Assemble Form", type="primary", use_container_width=True):
        # Validate constraints before assembly
        validation_errors = []
        
        # Check test length is set
        if test_length == 0:
            validation_errors.append(f"❌ Items per Form must be > 0. Set a value between 10-200")
        
        # Check number of forms is set
        if n_forms == 0:
            validation_errors.append(f"❌ Number of Forms must be > 0. Set a value >= 1")
        
        # Check domain constraints sum
        total_min = sum(dc['min'] for dc in domain_constraints.values())
        total_max = sum(dc['max'] for dc in domain_constraints.values())
        
        if total_min > test_length:
            validation_errors.append(f"❌ Domain minimums sum to {total_min}, exceeds test length {test_length}")
        
        if total_max < test_length:
            validation_errors.append(f"❌ Domain maximums sum to {total_max}, less than test length {test_length}")

        # Check Rasch B category constraints sum
        if raschb_cat_constraints:
            total_cat_min = sum(rc['min'] for rc in raschb_cat_constraints.values())
            total_cat_max = sum(rc['max'] for rc in raschb_cat_constraints.values())

            if total_cat_min > test_length:
                validation_errors.append(f"❌ Rasch B category minimums sum to {total_cat_min}, exceeds test length {test_length}")

            if total_cat_max < test_length:
                validation_errors.append(f"❌ Rasch B category maximums sum to {total_cat_max}, less than test length {test_length}")
        
        # Check image constraint
        if 'has_image' in items_df.columns:
            # Error: min set but max not set
            if image_constraint['min'] > 0 and image_constraint['max'] == 0:
                validation_errors.append(f"❌ Image minimum is set ({image_constraint['min']}) but maximum is 0. Set max > 0 to enable constraint.")
            
            # Validate if constraint is enabled
            if image_constraint['enabled']:
                items_with_images = items_df['has_image'].astype(int).sum()
                if image_constraint['min'] > items_with_images:
                    validation_errors.append(f"❌ Image minimum ({image_constraint['min']}) exceeds available items with images ({items_with_images})")
                if image_constraint['min'] > test_length:
                    validation_errors.append(f"❌ Image minimum ({image_constraint['min']}) exceeds test length ({test_length})")
                if image_constraint['max'] < image_constraint['min']:
                    validation_errors.append(f"❌ Image maximum ({image_constraint['max']}) less than minimum ({image_constraint['min']})")
        
        # Check if enough items meet quality thresholds (CTT only)
        if use_ctt_mode or approach == 'CTT':
            if discrimination_min > 0:
                eligible_items = items_df[items_df['point_biserial'] >= discrimination_min]
                if len(eligible_items) < test_length:
                    validation_errors.append(f"❌ Only {len(eligible_items)} items meet discrimination threshold {discrimination_min:.2f}, need {test_length}")
            
            if pvalue_min > 0 or pvalue_max < 1.0:
                eligible_items = items_df[(items_df['pvalue'] >= pvalue_min) & (items_df['pvalue'] <= pvalue_max)]
                if len(eligible_items) < test_length:
                    validation_errors.append(f"❌ Only {len(eligible_items)} items in p-value range [{pvalue_min:.2f}, {pvalue_max:.2f}], need {test_length}")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.info("💡 Try: Relax domain constraints, lower discrimination threshold, or widen p-value range")
        else:
            # Prepare config
            config = {
                'test_length': test_length,
                'approach': approach,
                'domain_constraints': domain_constraints,
                'raschb_cat_constraints': raschb_cat_constraints,
                'image_constraint': image_constraint,
                'use_ctt_mode': use_ctt_mode,
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
            
            # Store results in session state
            st.session_state['assembly_complete'] = False
            st.session_state['config'] = config
            st.session_state['n_forms'] = n_forms
            st.session_state['common_items'] = common_items
            st.session_state['approach'] = approach
            st.session_state['eval_points'] = eval_points
            st.session_state['enemy_check'] = enemy_check
            st.session_state['use_ctt_mode'] = use_ctt_mode
            
            # Assemble multiple forms simultaneously
            with st.spinner(f"🔧 Running simultaneous CBC solver for {n_forms} form(s)..."):
                try:
                    result = assemble_forms_with_cbc(items_df, config, n_forms)
                    
                    # Check for validation errors first
                    if 'validation_errors' in result and result['validation_errors']:
                        st.error(result.get('error_message', 'Validation failed'))
                        for error in result['validation_errors']:
                            st.warning(f"• {error}")
                    elif result['status'] != 'Optimal':
                        st.error(f"❌ Solver status: {result['status']}")
                        st.info("Try relaxing constraints or reducing number of forms")
                    else:
                        all_forms = []
                        selected_forms = result['selected_forms']
                        
                        for form_num, selected_ids in enumerate(selected_forms, start=1):
                            if not selected_ids:
                                st.warning(f"⚠️ Form {form_num}: No items selected")
                                continue
                            
                            # Get selected items
                            selected_df = items_df[items_df['item_id'].isin(selected_ids)].copy()
                            
                            # Evaluate form (now computes exact TIF/TCC at eval points)
                            stats = evaluate_form(selected_df, config)
                            alpha = estimate_cronbachs_alpha(selected_df)
                            
                            # Extract exact TIF/TCC values computed by evaluate_form
                            if eval_points:
                                theta_cut = eval_points['theta_mid']
                                # Use exact values computed at precise theta points (not approximated)
                                tif_at_cut = stats.get('tif_at_mid')
                                tcc_at_cut = stats.get('tcc_at_mid')
                                tif_at_low = stats.get('tif_at_low')
                                tcc_at_low = stats.get('tcc_at_low')
                                tif_at_high = stats.get('tif_at_high')
                                tcc_at_high = stats.get('tcc_at_high')
                            else:
                                tif_at_cut = None
                                tcc_at_cut = None
                                tif_at_low = None
                                tcc_at_low = None
                                tif_at_high = None
                                tcc_at_high = None
                            
                            # Store form data
                            all_forms.append({
                                'form_num': form_num,
                                'selected_df': selected_df,
                                'stats': stats,
                                'alpha': alpha,
                                'tif_at_cut': tif_at_cut,
                                'tcc_at_cut': tcc_at_cut,
                                'tif_at_low': tif_at_low,
                                'tcc_at_low': tcc_at_low,
                                'tif_at_high': tif_at_high,
                                'tcc_at_high': tcc_at_high,
                                'theta_cut': theta_cut if eval_points else None
                            })
                            
                            st.success(f"✅ Form {form_num}: Optimal solution found! Selected {len(selected_ids)} items")
                        
                        if all_forms:
                            st.session_state['all_forms'] = all_forms
                            st.session_state['assembly_complete'] = True
                
                except Exception as e:
                    st.error(f"❌ Assembly failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Display results if assembly is complete
    if st.session_state.get('assembly_complete', False):
        all_forms = st.session_state['all_forms']
        common_items = st.session_state['common_items']
        approach = st.session_state['approach']
        eval_points = st.session_state['eval_points']
        enemy_check = st.session_state['enemy_check']
        use_ctt_mode = st.session_state.get('use_ctt_mode', False)
        
        # Display results for all forms
        st.header(f"📊 Assembly Results ({len(all_forms)} form(s))")
        
        # Show mode indicator
        if use_ctt_mode:
            st.info("ℹ️ **CTT Mode Active:** Results are based on classical test theory statistics (p-value, point-biserial). TIF/TCC plots are not applicable.")
        
        # Show common items summary
        if common_items:
            st.info(f"🔗 **Common Items ({len(common_items)}):** {', '.join(map(str, common_items))}")
        
        # Overlay plots for all forms (IRT and Base Form, but not in CTT mode)
        if not use_ctt_mode and (approach == 'IRT (Rasch)' or approach == 'Base Form Optimal Under CTT / Rasch') and eval_points:
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
            
            # Add target markers (for IRT with full targets)
            thetas = [eval_points.get('theta_low'), eval_points.get('theta_mid'), eval_points.get('theta_high')]
            tif_targets = [eval_points.get('tif_low'), eval_points.get('tif_mid'), eval_points.get('tif_high')]
            
            # Only show target markers if we have all three points (IRT mode)
            if all(t is not None for t in tif_targets):
                fig_tif_all.add_trace(go.Scatter(
                    x=[t for t in thetas if t is not None],
                    y=[tgt for tgt in tif_targets if tgt is not None],
                    mode='markers',
                    name='Targets',
                    marker=dict(size=12, color='black', symbol='diamond')
                ))
            
            # Add vertical line at logit cut (for Base Form or IRT)
            if eval_points.get('theta_mid') is not None:
                # Calculate average TIF at logit cut across all forms
                tif_values_at_cut = []
                for form_data in all_forms:
                    if form_data.get('tif_at_cut') is not None:
                        tif_values_at_cut.append(form_data['tif_at_cut'])
                
                avg_tif = sum(tif_values_at_cut) / len(tif_values_at_cut) if tif_values_at_cut else None
                
                # Format annotation with TIF value
                if avg_tif is not None:
                    annotation_text = f"Logit Cut ({eval_points.get('theta_mid'):.3f}, TIF: {avg_tif:.2f})"
                else:
                    annotation_text = f"Logit Cut ({eval_points.get('theta_mid'):.3f})"
                
                fig_tif_all.add_vline(
                    x=eval_points.get('theta_mid'),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=annotation_text,
                    annotation_position="top right"
                )
            
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
            
            # Add TCC target marker ONLY at logit cut (theta_mid)
            tcc_mid_target = eval_points.get('tcc_mid')
            if tcc_mid_target is not None and tcc_mid_target > 0:
                fig_tcc_all.add_trace(go.Scatter(
                    x=[eval_points.get('theta_mid')],
                    y=[tcc_mid_target],
                    mode='markers',
                    name='Targets',
                    marker=dict(size=12, color='black', symbol='diamond')
                ))
            
            # Add vertical line at logit cut with TCC value
            if eval_points.get('theta_mid') is not None:
                # Calculate average TCC at logit cut across all forms
                tcc_values_at_cut = []
                for form_data in all_forms:
                    if form_data.get('tcc_at_cut') is not None:
                        tcc_values_at_cut.append(form_data['tcc_at_cut'])
                
                avg_tcc = sum(tcc_values_at_cut) / len(tcc_values_at_cut) if tcc_values_at_cut else None
                
                # Format annotation with TCC value
                if avg_tcc is not None:
                    annotation_text = f"Logit Cut ({eval_points.get('theta_mid'):.3f}, TCC: {avg_tcc:.2f})"
                else:
                    annotation_text = f"Logit Cut ({eval_points.get('theta_mid'):.3f})"
                
                fig_tcc_all.add_vline(
                    x=eval_points.get('theta_mid'),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=annotation_text,
                    annotation_position="top right"
                )
            
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
            display_form_results(form_data, eval_points, common_items, approach, enemy_check, use_ctt_mode)
        else:
            # Multiple forms - use tabs
            tabs = st.tabs([f"Form {i+1}" for i in range(len(all_forms))])
            for i, (tab, form_data) in enumerate(zip(tabs, all_forms)):
                with tab:
                    display_form_results(form_data, eval_points, common_items, approach, enemy_check, use_ctt_mode)
        
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
                
                if not use_ctt_mode and tif_at_cut is not None:
                    row[f'TIF @ θ={theta_cut:.3f}'] = f"{tif_at_cut:.2f}"
                    row[f'TCC @ θ={theta_cut:.3f}'] = f"{tcc_at_cut:.2f}"
                    
                    # Add -1 and +1 values if available (Base Form)
                    if form_data.get('tif_at_low') is not None:
                        row[f'TIF @ θ={theta_cut-1:.3f}'] = f"{form_data['tif_at_low']:.2f}"
                        row[f'TCC @ θ={theta_cut-1:.3f}'] = f"{form_data['tcc_at_low']:.2f}"
                    
                    if form_data.get('tif_at_high') is not None:
                        row[f'TIF @ θ={theta_cut+1:.3f}'] = f"{form_data['tif_at_high']:.2f}"
                        row[f'TCC @ θ={theta_cut+1:.3f}'] = f"{form_data['tcc_at_high']:.2f}"
                
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
                        str(len(selected_df)),
                        f"{alpha:.3f}",
                        f"{selected_df['pvalue'].mean():.3f}",
                        f"{stats['mean_discrimination']:.3f}",
                        f"{stats['mean_difficulty']:.3f}",
                        f"{stats['sd_difficulty']:.3f}",
                        'Enabled' if enemy_check else 'Disabled'
                    ]
                }
                
                if not use_ctt_mode and tif_at_cut is not None:
                    summary_data['Metric'].extend([f'TIF @ θ={theta_cut:.3f}', f'TCC @ θ={theta_cut:.3f}'])
                    summary_data['Value'].extend([f"{tif_at_cut:.2f}", f"{tcc_at_cut:.2f}"])
                    
                    # Add -1 and +1 values if available (Base Form)
                    if form_data.get('tif_at_low') is not None:
                        summary_data['Metric'].extend([f'TIF @ θ={theta_cut-1:.3f}', f'TCC @ θ={theta_cut-1:.3f}'])
                        summary_data['Value'].extend([f"{form_data['tif_at_low']:.2f}", f"{form_data['tcc_at_low']:.2f}"])
                    
                    if form_data.get('tif_at_high') is not None:
                        summary_data['Metric'].extend([f'TIF @ θ={theta_cut+1:.3f}', f'TCC @ θ={theta_cut+1:.3f}'])
                        summary_data['Value'].extend([f"{form_data['tif_at_high']:.2f}", f"{form_data['tcc_at_high']:.2f}"])
                
                # Add domain counts
                summary_data['Metric'].append('')  # Blank row
                summary_data['Value'].append('')
                summary_data['Metric'].append('Domain Distribution')
                summary_data['Value'].append('')
                
                for domain in sorted(stats['domain_counts'].keys()):
                    summary_data['Metric'].append(f"  {domain}")
                    summary_data['Value'].append(str(stats['domain_counts'][domain]))

                if stats.get('raschb_cat_counts'):
                    summary_data['Metric'].append('')  # Blank row
                    summary_data['Value'].append('')
                    summary_data['Metric'].append('Rasch B Category Distribution')
                    summary_data['Value'].append('')

                    for cat in sorted(stats['raschb_cat_counts'].keys()):
                        summary_data['Metric'].append(f"  {cat}")
                        summary_data['Value'].append(str(stats['raschb_cat_counts'][cat]))
                
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
