"""
AI_LOFT - Linear-on-the-Fly Testing
====================================

Scalable parallel form assembly for large-scale testing programs.
Generates 10-100+ parallel forms using stratified sequential assembly.

Features:
- PostgreSQL database integration
- Stratified sequential assembly algorithm
- Enemy item management with fast lookup
- Parallelism metrics and quality checks
- Optimized for LOFT scenarios

Author: AI Assistant
Date: December 3, 2025
"""

import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from openai import OpenAI
import json
from typing import List, Dict, Any, Tuple, Set
import plotly.graph_objects as go
from io import BytesIO
import os
from sqlalchemy import create_engine
from datetime import datetime
import time

# ==================== Configuration ====================
DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

# SQLAlchemy connection string
DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

# Difficulty strata for stratified sampling
DIFFICULTY_STRATA_IRT = {
    'very_easy': (-3.0, -1.0),
    'easy': (-1.0, -0.5),
    'medium': (-0.5, 0.5),
    'hard': (0.5, 1.0),
    'very_hard': (1.0, 3.0)
}

DIFFICULTY_STRATA_CTT = {
    'very_easy': (0.85, 1.0),
    'easy': (0.70, 0.85),
    'medium': (0.50, 0.70),
    'hard': (0.30, 0.50),
    'very_hard': (0.0, 0.30)
}

# Default difficulty distribution
DEFAULT_DIFFICULTY_DISTRIBUTION = {
    'very_easy': 0.10,
    'easy': 0.20,
    'medium': 0.40,
    'hard': 0.20,
    'very_hard': 0.10
}

# Page configuration
st.set_page_config(
    page_title="AI LOFT",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Database Agent ====================

class ItemPoolAgent:
    """Agent to access and query the item pool from PostgreSQL"""
    
    def __init__(self, db_config, db_url):
        self.db_config = db_config
        self.db_url = db_url
        self.conn = None
        self.engine = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.engine = create_engine(self.db_url)
            return True
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_all_items(self) -> pd.DataFrame:
        """Retrieve all items from the pool"""
        query = """
        SELECT item_id, domain, topic, rasch_b, pvalue, point_biserial
        FROM itembank
        WHERE embedding IS NOT NULL
        ORDER BY item_id
        """
        return pd.read_sql_query(query, self.engine)
    
    def get_domains(self) -> List[str]:
        """Get list of unique domains"""
        query = "SELECT DISTINCT domain FROM itembank WHERE domain IS NOT NULL ORDER BY domain"
        df = pd.read_sql_query(query, self.engine)
        return df['domain'].tolist()
    
    def get_item_stats(self) -> Dict[str, Any]:
        """Get item pool statistics"""
        query = """
        SELECT 
            COUNT(*) as total_items,
            COUNT(DISTINCT domain) as total_domains,
            AVG(rasch_b) as avg_difficulty,
            MIN(rasch_b) as min_difficulty,
            MAX(rasch_b) as max_difficulty,
            AVG(pvalue) as avg_pvalue,
            AVG(point_biserial) as avg_discrimination
        FROM itembank
        WHERE embedding IS NOT NULL
        """
        df = pd.read_sql_query(query, self.engine)
        return df.iloc[0].to_dict()
    
    def get_enemy_items(self, item_ids: List[int]) -> List[Tuple[int, int]]:
        """Get enemy item relationships from database"""
        query = """
        SELECT item_id, enemy
        FROM itembank
        WHERE item_id = ANY(%s) AND enemy IS NOT NULL
        """
        df = pd.read_sql_query(query, self.engine, params=(item_ids,))
        
        enemy_pairs = []
        for _, row in df.iterrows():
            item_id = row['item_id']
            # Enemy column might contain comma-separated IDs or a single ID
            enemy_str = str(row['enemy'])
            if enemy_str and enemy_str != 'None':
                # Parse enemy IDs (handle both single and comma-separated)
                enemy_ids = [int(e.strip()) for e in enemy_str.split(',') if e.strip().isdigit()]
                for enemy_id in enemy_ids:
                    if enemy_id in item_ids:
                        enemy_pairs.append((item_id, enemy_id))
        
        return enemy_pairs

# ==================== Enemy Item Management ====================

def build_enemy_index(items_df: pd.DataFrame, agent: ItemPoolAgent) -> Dict[int, Set[int]]:
    """
    Build fast lookup index for enemy relationships from database
    
    Args:
        items_df: DataFrame with items
        agent: Database agent
    
    Returns:
        dict: {item_id: set(enemy_ids)}
    """
    item_ids = items_df['item_id'].tolist()
    enemy_pairs = agent.get_enemy_items(item_ids)
    
    enemy_index = {item_id: set() for item_id in item_ids}
    
    for item1, item2 in enemy_pairs:
        enemy_index[item1].add(item2)
        enemy_index[item2].add(item1)  # Bidirectional
    
    return enemy_index

def has_enemy_in_form(item_id: int, form_items: List[int], enemy_index: Dict[int, Set[int]]) -> bool:
    """Check if item has enemy in form (O(1) average case)"""
    return bool(enemy_index.get(item_id, set()) & set(form_items))

# ==================== Stratified Assembly Functions ====================

def stratify_items(items_df: pd.DataFrame, approach: str, difficulty_col: str) -> Dict[str, pd.DataFrame]:
    """
    Stratify items by difficulty level
    
    Args:
        items_df: DataFrame with items
        approach: 'IRT' or 'CTT'
        difficulty_col: Column name for difficulty
    
    Returns:
        dict: {stratum_name: DataFrame of items in that stratum}
    """
    strata_def = DIFFICULTY_STRATA_IRT if approach == 'IRT' else DIFFICULTY_STRATA_CTT
    strata = {}
    
    for stratum_name, (min_diff, max_diff) in strata_def.items():
        strata[stratum_name] = items_df[
            (items_df[difficulty_col] >= min_diff) & 
            (items_df[difficulty_col] < max_diff)
        ].copy()
    
    return strata

def calculate_stratum_targets(test_length: int, domain_constraints: Dict[str, Dict], 
                              difficulty_distribution: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    """
    Calculate how many items needed from each stratum for each domain
    
    Returns:
        dict: {domain: {stratum: count}}
    """
    targets = {}
    
    for domain, constraints in domain_constraints.items():
        domain_count = constraints['min']  # Use min as target
        targets[domain] = {}
        
        for stratum_name, proportion in difficulty_distribution.items():
            targets[domain][stratum_name] = int(round(domain_count * proportion))
    
    return targets

def get_form_target_difficulty(base_target: float, form_index: int, tolerance: float, seed: int = 42) -> float:
    """
    Generate form-specific difficulty target with controlled variation
    
    Args:
        base_target: User-specified mean difficulty target
        form_index: Index of current form (0-based)
        tolerance: Allowable deviation
        seed: Random seed for reproducibility
    
    Returns:
        Form-specific target difficulty
    """
    np.random.seed(seed + form_index)
    variation = np.random.uniform(-tolerance/2, +tolerance/2)
    return base_target + variation

def score_items_with_jitter(items_df: pd.DataFrame, base_score_col: str, form_index: int, jitter: float = 0.05) -> pd.DataFrame:
    """
    Add small random jitter to item scores for variation between forms
    
    Args:
        items_df: DataFrame of items with base_score column
        form_index: Form index for seed variation
        jitter: Max percentage jitter (default 5%)
    
    Returns:
        Items with jittered scores
    """
    np.random.seed(42 + form_index * 17)  # Different seed per form
    items_df = items_df.copy()
    items_df['final_score'] = items_df[base_score_col] * (
        1 + np.random.uniform(-jitter, jitter, len(items_df))
    )
    return items_df

# ==================== Sequential Assembly Algorithm ====================

def assemble_form_stratified(
    items_df: pd.DataFrame,
    config: Dict[str, Any],
    form_index: int,
    used_items: Set[int],
    enemy_index: Dict[int, Set[int]]
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Assemble a single form using stratified sequential selection
    
    Args:
        items_df: Full item pool
        config: Assembly configuration
        form_index: Index of current form
        used_items: Set of already used item IDs
        enemy_index: Fast lookup for enemy items
    
    Returns:
        (selected_item_ids, statistics)
    """
    approach = config['approach']
    test_length = config['test_length']
    domain_constraints = config['domain_constraints']
    difficulty_distribution = config.get('difficulty_distribution', DEFAULT_DIFFICULTY_DISTRIBUTION)
    base_target = config.get('mean_diff_target', 0.0 if approach == 'IRT' else 0.65)
    tolerance = config.get('mean_diff_tolerance', 0.2)
    maximize_alpha = config.get('maximize_alpha', False)
    
    # Get form-specific target with controlled variation
    form_target = get_form_target_difficulty(base_target, form_index, tolerance)
    
    # Filter available items
    available = items_df[~items_df['item_id'].isin(used_items)].copy()
    
    # Apply P-value and PBS filters if specified
    if 'pvalue_min' in config and 'pvalue_max' in config:
        available = available[
            (available['pvalue'] >= config['pvalue_min']) & 
            (available['pvalue'] <= config['pvalue_max'])
        ]
    
    if 'pbs_threshold' in config and config['pbs_threshold'] is not None:
        available = available[available['point_biserial'] > config['pbs_threshold']]
    
    # Determine difficulty column
    difficulty_col = 'rasch_b' if approach == 'IRT' else 'pvalue'
    
    # Stratify available items
    strata = stratify_items(available, approach, difficulty_col)
    
    # Calculate targets per stratum per domain
    stratum_targets = calculate_stratum_targets(test_length, domain_constraints, difficulty_distribution)
    
    # Select items
    selected_items = []
    
    for domain, domain_strata_targets in stratum_targets.items():
        for stratum_name, needed_count in domain_strata_targets.items():
            if needed_count == 0:
                continue
            
            # Get items from this stratum and domain
            stratum_items = strata.get(stratum_name, pd.DataFrame())
            domain_stratum_items = stratum_items[stratum_items['domain'] == domain].copy()
            
            if len(domain_stratum_items) == 0:
                continue
            
            # Score items
            domain_stratum_items['difficulty_score'] = 1 / (
                1 + abs(domain_stratum_items[difficulty_col] - form_target)
            )
            
            if maximize_alpha:
                domain_stratum_items['quality_score'] = (
                    domain_stratum_items['difficulty_score'] * 
                    (domain_stratum_items['point_biserial'] ** 2)
                )
            else:
                domain_stratum_items['quality_score'] = (
                    domain_stratum_items['difficulty_score'] * 
                    domain_stratum_items['point_biserial']
                )
            
            # Add jitter for variation
            domain_stratum_items = score_items_with_jitter(
                domain_stratum_items, 'quality_score', form_index
            )
            
            # Select top items
            candidates = domain_stratum_items.nlargest(min(needed_count * 3, len(domain_stratum_items)), 'final_score')
            
            # Select items while avoiding enemy conflicts
            for _, item in candidates.iterrows():
                if len([i for i in selected_items if items_df[items_df['item_id']==i].iloc[0]['domain'] == domain]) >= needed_count:
                    break
                
                item_id = item['item_id']
                
                # Check for enemy conflict
                if not has_enemy_in_form(item_id, selected_items, enemy_index):
                    selected_items.append(item_id)
    
    # Calculate statistics
    form_data = items_df[items_df['item_id'].isin(selected_items)]
    
    stats = {
        'n_items': len(selected_items),
        'items': selected_items,
        'mean_difficulty': float(form_data[difficulty_col].mean()),
        'sd_difficulty': float(form_data[difficulty_col].std()),
        'mean_pbs': float(form_data['point_biserial'].mean()),
        'domain_counts': form_data['domain'].value_counts().to_dict()
    }
    
    return selected_items, stats

# ==================== Quality Metrics ====================

def calculate_parallelism_metrics(all_forms_stats: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Assess parallelism across all assembled forms
    
    Returns:
        {
            'difficulty_variance': float,
            'pbs_variance': float,
            'parallelism_index': float
        }
    """
    difficulties = [f['mean_difficulty'] for f in all_forms_stats]
    pbs_values = [f['mean_pbs'] for f in all_forms_stats]
    
    diff_var = float(np.std(difficulties))
    pbs_var = float(np.std(pbs_values))
    
    # Parallelism index (0=poor, 1=perfect)
    parallelism = 1 / (1 + diff_var + pbs_var)
    
    return {
        'difficulty_variance': diff_var,
        'pbs_variance': pbs_var,
        'parallelism_index': parallelism
    }

# ==================== Export Functions ====================

def export_forms_to_excel(forms_data: List[Dict], items_df: pd.DataFrame, parallelism_metrics: Dict) -> BytesIO:
    """Export all forms to Excel with statistics"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for i, form_stats in enumerate(forms_data):
            summary_data.append({
                'Form': f'Form_{i+1}',
                'N_Items': form_stats['n_items'],
                'Mean_Difficulty': round(form_stats['mean_difficulty'], 3),
                'SD_Difficulty': round(form_stats['sd_difficulty'], 3),
                'Mean_PBS': round(form_stats['mean_pbs'], 3)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Parallelism metrics
        parallelism_df = pd.DataFrame([parallelism_metrics])
        parallelism_df.to_excel(writer, sheet_name='Parallelism', index=False)
        
        # Individual forms
        for i, form_stats in enumerate(forms_data):
            form_items = items_df[items_df['item_id'].isin(form_stats['items'])]
            form_items.to_excel(writer, sheet_name=f'Form_{i+1}', index=False)
    
    output.seek(0)
    return output

# ==================== Main Application ====================

def main():
    st.markdown('<div class="main-header">🎯 AI LOFT - Linear-on-the-Fly Testing</div>', unsafe_allow_html=True)
    st.markdown("**Scalable parallel form assembly for large-scale testing programs**")
    
    # Initialize session state
    if 'forms_assembled' not in st.session_state:
        st.session_state.forms_assembled = False
        st.session_state.forms_data = []
        st.session_state.parallelism_metrics = {}
    
    # Connect to database
    agent = ItemPoolAgent(DB_CONFIG, DB_URL)
    
    with st.spinner("Connecting to database..."):
        if not agent.connect():
            st.error("❌ Could not connect to database. Please check your configuration.")
            return
    
    # Get item pool stats
    pool_stats = agent.get_item_stats()
    domains = agent.get_domains()
    
    # Sidebar - Configuration
    st.sidebar.header("🔧 LOFT Configuration")
    
    st.sidebar.subheader("📊 Item Pool")
    st.sidebar.metric("Total Items", int(pool_stats['total_items']))
    st.sidebar.metric("Domains", int(pool_stats['total_domains']))
    
    st.sidebar.divider()
    
    # Form specifications
    st.sidebar.subheader("📝 Form Specifications")
    
    n_forms = st.sidebar.slider(
        "Number of Forms",
        min_value=1,
        max_value=100,
        value=10,
        help="Number of parallel forms to generate"
    )
    
    test_length = st.sidebar.number_input(
        "Items per Form",
        min_value=10,
        max_value=min(200, int(pool_stats['total_items']) // n_forms),
        value=72,
        step=5
    )
    
    approach = st.sidebar.radio(
        "Approach",
        options=['IRT', 'CTT'],
        help="IRT: Uses Rasch B\nCTT: Uses P-value"
    )
    
    st.sidebar.divider()
    
    # Domain distribution
    st.sidebar.subheader("🎯 Domain Distribution")
    domain_constraints = {}
    
    for domain in domains:
        cols = st.sidebar.columns(2)
        with cols[0]:
            min_val = st.number_input(f"{domain} Min", min_value=0, max_value=test_length, value=9, key=f"min_{domain}")
        with cols[1]:
            max_val = st.number_input(f"{domain} Max", min_value=min_val, max_value=test_length, value=9, key=f"max_{domain}")
        
        domain_constraints[domain] = {'min': min_val, 'max': max_val}
    
    st.sidebar.divider()
    
    # IRT Evaluation Points (only for IRT approach)
    evaluation_points = None
    tif_targets = None
    tcc_targets = None
    tif_tolerance = None
    tcc_tolerance = None
    
    if approach == 'IRT':
        st.sidebar.subheader("📐 IRT Evaluation Points")
        
        logit_cut = st.sidebar.number_input(
            "Logit Cut (θ)",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Primary evaluation point (cut score)"
        )
        
        # Three evaluation points: cut-1, cut, cut+1
        evaluation_points = [logit_cut - 1.0, logit_cut, logit_cut + 1.0]
        
        st.sidebar.markdown(f"**Evaluation Points:** {evaluation_points[0]:.1f}, {evaluation_points[1]:.1f}, {evaluation_points[2]:.1f}")
        
        # TIF targets for each point
        st.sidebar.markdown("**TIF Targets:**")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            tif_low = st.number_input(f"TIF @ {evaluation_points[0]:.1f}", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="tif_low")
        with col2:
            tif_mid = st.number_input(f"TIF @ {evaluation_points[1]:.1f}", min_value=0.0, max_value=50.0, value=10.0, step=0.5, key="tif_mid")
        with col3:
            tif_high = st.number_input(f"TIF @ {evaluation_points[2]:.1f}", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="tif_high")
        
        tif_targets = [tif_low, tif_mid, tif_high]
        
        # TCC targets for each point
        st.sidebar.markdown("**TCC Targets (Expected Score):**")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            tcc_low = st.number_input(f"TCC @ {evaluation_points[0]:.1f}", min_value=0.0, max_value=float(test_length), value=float(test_length * 0.3), step=1.0, key="tcc_low")
        with col2:
            tcc_mid = st.number_input(f"TCC @ {evaluation_points[1]:.1f}", min_value=0.0, max_value=float(test_length), value=float(test_length * 0.5), step=1.0, key="tcc_mid")
        with col3:
            tcc_high = st.number_input(f"TCC @ {evaluation_points[2]:.1f}", min_value=0.0, max_value=float(test_length), value=float(test_length * 0.7), step=1.0, key="tcc_high")
        
        tcc_targets = [tcc_low, tcc_mid, tcc_high]
        
        # Tolerances only at logit cut (middle point)
        st.sidebar.markdown(f"**Tolerances @ {evaluation_points[1]:.1f} (Logit Cut):**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            tif_tolerance = st.number_input("TIF Tolerance (±)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="tif_tol")
        with col2:
            tcc_tolerance = st.number_input("TCC Tolerance (±)", min_value=0.1, max_value=10.0, value=2.0, step=0.5, key="tcc_tol")
    
    st.sidebar.divider()
    
    # Enemy item detection
    use_enemy_detection = st.sidebar.checkbox("Enemy Item Detection", value=True, help="Avoid items marked as enemies in the database")
    
    # Assembly button
    st.sidebar.divider()
    
    if st.sidebar.button("🚀 Assemble Forms", type="primary", use_container_width=True):
        # Validate constraints
        total_min = sum(dc['min'] for dc in domain_constraints.values())
        if total_min > test_length:
            st.error(f"❌ Sum of domain minimums ({total_min}) exceeds test length ({test_length})")
            return
        
        # Check if enough items
        if test_length * n_forms > pool_stats['total_items']:
            st.warning(f"⚠️ Requested {test_length * n_forms} items, but pool has {int(pool_stats['total_items'])}")
        
        # Get all items
        with st.spinner("Loading item pool..."):
            items_df = agent.get_all_items()
        
        # Build enemy index if enabled
        enemy_index = {}
        if use_enemy_detection:
            with st.spinner("Building enemy item index..."):
                enemy_index = build_enemy_index(items_df, agent)
                n_enemy_pairs = sum(len(enemies) for enemies in enemy_index.values()) // 2
                if n_enemy_pairs > 0:
                    st.info(f"📊 Detected {n_enemy_pairs} enemy item pairs from database")
        
        # Prepare config
        config = {
            'n_forms': n_forms,
            'test_length': test_length,
            'approach': approach,
            'domain_constraints': domain_constraints,
            'mean_diff_target': mean_target if apply_mean_diff else None,
            'mean_diff_tolerance': tolerance,
            'maximize_alpha': maximize_alpha,
            'difficulty_distribution': DEFAULT_DIFFICULTY_DISTRIBUTION
        }
        
        # Assemble forms
        forms_data = []
        used_items = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for i in range(n_forms):
            status_text.text(f"Assembling Form {i+1}/{n_forms}...")
            
            selected_items, stats = assemble_form_stratified(
                items_df, config, i, used_items, enemy_index
            )
            
            used_items.update(selected_items)
            forms_data.append(stats)
            
            progress_bar.progress((i + 1) / n_forms)
        
        elapsed_time = time.time() - start_time
        
        status_text.empty()
        progress_bar.empty()
        
        # Calculate parallelism metrics
        parallelism_metrics = calculate_parallelism_metrics(forms_data)
        
        # Store in session state
        st.session_state.forms_assembled = True
        st.session_state.forms_data = forms_data
        st.session_state.parallelism_metrics = parallelism_metrics
        st.session_state.items_df = items_df
        
        st.success(f"✅ Successfully assembled {n_forms} forms in {elapsed_time:.2f}s!")
    
    # Display results
    if st.session_state.forms_assembled:
        st.divider()
        
        # Parallelism metrics
        st.header("📊 Parallelism Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Difficulty Variance",
                f"{st.session_state.parallelism_metrics['difficulty_variance']:.3f}",
                help="Lower is better (more parallel)"
            )
        
        with col2:
            st.metric(
                "PBS Variance",
                f"{st.session_state.parallelism_metrics['pbs_variance']:.3f}",
                help="Lower is better (more parallel)"
            )
        
        with col3:
            parallelism_index = st.session_state.parallelism_metrics['parallelism_index']
            st.metric(
                "Parallelism Index",
                f"{parallelism_index:.3f}",
                help="Higher is better (0-1 scale)"
            )
        
        # Form statistics table
        st.subheader("📋 Form Statistics")
        
        summary_data = []
        for i, form_stats in enumerate(st.session_state.forms_data):
            summary_data.append({
                'Form': f'Form {i+1}',
                'N Items': form_stats['n_items'],
                'Mean Difficulty': round(form_stats['mean_difficulty'], 3),
                'SD Difficulty': round(form_stats['sd_difficulty'], 3),
                'Mean PBS': round(form_stats['mean_pbs'], 3)
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Export button
        excel_file = export_forms_to_excel(
            st.session_state.forms_data,
            st.session_state.items_df,
            st.session_state.parallelism_metrics
        )
        
        st.download_button(
            label="📥 Download All Forms (Excel)",
            data=excel_file,
            file_name=f"LOFT_Forms_{n_forms}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    agent.disconnect()

if __name__ == "__main__":
    main()
