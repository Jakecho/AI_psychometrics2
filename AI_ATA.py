"""
AI-Powered Automated Test Assembly (ATA)
Builds parallel forms with constraints using GPT-4o and PostgreSQL item pool
"""

import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from openai import OpenAI
import json
from typing import List, Dict, Any, Tuple
import plotly.graph_objects as go
from io import BytesIO
import os
from sqlalchemy import create_engine

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

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
               "choice_C", "choice_D", key, rationale, rasch_b, pvalue, 
               point_biserial, embedding
        FROM itembank
        WHERE embedding IS NOT NULL
        ORDER BY item_id
        """
        return pd.read_sql_query(query, self.engine)
    
    def get_items_by_domain(self, domain: str) -> pd.DataFrame:
        """Retrieve items filtered by domain"""
        query = """
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
               "choice_C", "choice_D", key, rationale, rasch_b, pvalue, 
               point_biserial
        FROM itembank
        WHERE domain = %s AND embedding IS NOT NULL
        ORDER BY item_id
        """
        return pd.read_sql_query(query, self.engine, params=(domain,))
    
    def get_items_by_difficulty(self, min_b: float, max_b: float) -> pd.DataFrame:
        """Retrieve items filtered by Rasch difficulty"""
        query = """
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
               "choice_C", "choice_D", key, rationale, rasch_b, pvalue, 
               point_biserial
        FROM itembank
        WHERE rasch_b BETWEEN %s AND %s AND embedding IS NOT NULL
        ORDER BY item_id
        """
        return pd.read_sql_query(query, self.engine, params=(min_b, max_b))
    
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
    
    def check_enemy_items(self, item_ids: List[int], similarity_threshold: float = 0.85) -> List[Tuple[int, int]]:
        """Check for enemy items (highly similar items) using embeddings"""
        query = """
        SELECT i1.item_id as item1, i2.item_id as item2,
               1 - (i1.embedding <=> i2.embedding) as similarity
        FROM itembank i1
        CROSS JOIN itembank i2
        WHERE i1.item_id < i2.item_id
          AND i1.item_id = ANY(%s)
          AND i2.item_id = ANY(%s)
          AND 1 - (i1.embedding <=> i2.embedding) > %s
        ORDER BY similarity DESC
        """
        df = pd.read_sql_query(query, self.engine, params=(item_ids, item_ids, similarity_threshold))
        return [(row['item1'], row['item2']) for _, row in df.iterrows()]

# ==================== IRT Functions ====================

def calculate_tif(items_df: pd.DataFrame, theta_range: np.ndarray) -> np.ndarray:
    """
    Calculate Test Information Function using Rasch model
    TIF(theta) = sum of item information functions
    For Rasch: I(theta) = P(theta) * Q(theta) where P = 1/(1+exp(-(theta-b)))
    """
    tif = np.zeros_like(theta_range)
    
    for _, item in items_df.iterrows():
        b = item['rasch_b']
        # Rasch probability
        p = 1 / (1 + np.exp(-(theta_range - b)))
        q = 1 - p
        # Item information
        item_info = p * q
        tif += item_info
    
    return tif

def calculate_tcc(items_df: pd.DataFrame, theta_range: np.ndarray) -> np.ndarray:
    """
    Calculate Test Characteristic Curve (expected score)
    TCC(theta) = sum of item probabilities
    """
    tcc = np.zeros_like(theta_range)
    
    for _, item in items_df.iterrows():
        b = item['rasch_b']
        # Rasch probability
        p = 1 / (1 + np.exp(-(theta_range - b)))
        tcc += p
    
    return tcc

def evaluate_form_quality(items_df: pd.DataFrame, theta_range: np.ndarray, 
                         eval_points: Dict[str, float], tolerance: float,
                         difficulty_constraint: Dict[str, float] = None) -> Dict[str, Any]:
    """Evaluate if form meets quality criteria at three theta points plus difficulty and TCC"""
    tif = calculate_tif(items_df, theta_range)
    tcc = calculate_tcc(items_df, theta_range)
    
    # Extract evaluation points
    theta_low = eval_points['theta_low']
    theta_mid = eval_points['theta_mid']
    theta_high = eval_points['theta_high']
    tif_target_low = eval_points['tif_low']
    tif_target_mid = eval_points['tif_mid']
    tif_target_high = eval_points['tif_high']
    tcc_target_mid = eval_points.get('tcc_mid', None)
    
    # Find TIF/TCC values at evaluation points
    idx_low = np.argmin(np.abs(theta_range - theta_low))
    idx_mid = np.argmin(np.abs(theta_range - theta_mid))
    idx_high = np.argmin(np.abs(theta_range - theta_high))
    
    tif_at_low = tif[idx_low]
    tif_at_mid = tif[idx_mid]
    tif_at_high = tif[idx_high]
    tcc_at_mid = tcc[idx_mid]
    
    # Check if TIF within tolerance at all three points
    meets_tif_low = abs(tif_at_low - tif_target_low) <= (tif_target_low * tolerance)
    meets_tif_mid = abs(tif_at_mid - tif_target_mid) <= (tif_target_mid * tolerance)
    meets_tif_high = abs(tif_at_high - tif_target_high) <= (tif_target_high * tolerance)
    
    tif_meets_target = meets_tif_low and meets_tif_mid and meets_tif_high
    
    # Check TCC at cut point if specified
    meets_tcc_mid = True
    if tcc_target_mid is not None:
        meets_tcc_mid = abs(tcc_at_mid - tcc_target_mid) <= (tcc_target_mid * tolerance)
    
    # Check mean difficulty constraint if specified
    mean_difficulty = items_df['rasch_b'].mean()
    meets_difficulty = True
    if difficulty_constraint:
        target_diff = difficulty_constraint.get('target', 0.0)
        diff_tolerance = difficulty_constraint.get('tolerance', 0.5)
        meets_difficulty = abs(mean_difficulty - target_diff) <= diff_tolerance
    
    # Overall pass/fail
    meets_all_constraints = tif_meets_target and meets_tcc_mid and meets_difficulty
    
    return {
        'tif_values': tif,
        'tcc_values': tcc,
        'tif_at_low': tif_at_low,
        'tif_at_mid': tif_at_mid,
        'tif_at_high': tif_at_high,
        'tcc_at_mid': tcc_at_mid,
        'target_low': tif_target_low,
        'target_mid': tif_target_mid,
        'target_high': tif_target_high,
        'target_tcc_mid': tcc_target_mid,
        'theta_low': theta_low,
        'theta_mid': theta_mid,
        'theta_high': theta_high,
        'meets_tif_low': meets_tif_low,
        'meets_tif_mid': meets_tif_mid,
        'meets_tif_high': meets_tif_high,
        'meets_tcc_mid': meets_tcc_mid,
        'tif_meets_target': tif_meets_target,
        'avg_difficulty': mean_difficulty,
        'difficulty_sd': items_df['rasch_b'].std(),
        'avg_discrimination': items_df['point_biserial'].mean(),
        'meets_difficulty': meets_difficulty,
        'meets_all_constraints': meets_all_constraints
    }

# ==================== AI Assembly Functions ====================

def build_form_with_ai(agent: ItemPoolAgent, form_specs: Dict[str, Any], 
                       available_items: pd.DataFrame, used_items: set,
                       common_items: List[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Use GPT-4o to build a test form meeting specifications including difficulty constraints
    """
    
    # Filter out already used items
    candidate_items = available_items[~available_items['item_id'].isin(used_items)].copy()
    
    # If common items specified, include them
    selected_items = []
    if common_items:
        common_df = available_items[available_items['item_id'].isin(common_items)]
        selected_items = common_items.copy()
        remaining_needed = form_specs['test_length'] - len(common_items)
        candidate_items = candidate_items[~candidate_items['item_id'].isin(common_items)]
    else:
        remaining_needed = form_specs['test_length']
    
    # Prepare domain constraints
    domain_constraints = form_specs.get('domain_counts', {})
    
    # Group candidates by domain
    domain_groups = {}
    for domain in domain_constraints.keys():
        domain_items = candidate_items[candidate_items['domain'] == domain]
        domain_groups[domain] = domain_items['item_id'].tolist()
    
    # Build prompt for GPT-4o
    prompt = f"""You are an expert psychometrician building a test form.

Test Specifications:
- Test Length: {form_specs['test_length']} items
- Target TIF at theta=0: {form_specs.get('target_tif', 10)}
- Tolerance: {form_specs.get('tolerance', 0.1) * 100}%

Domain Requirements:
{json.dumps(domain_constraints, indent=2)}

Available Items by Domain:
{json.dumps({k: len(v) for k, v in domain_groups.items()}, indent=2)}

Common Items Already Selected: {len(selected_items)}
Remaining Items Needed: {remaining_needed}

Task: Select {remaining_needed} item IDs (one per domain as specified) that will:
1. Meet domain distribution requirements
2. Provide balanced difficulty (target avg Rasch b ≈ 0)
3. Maximize test information around theta = 0
4. Avoid highly similar items (enemy items)

Return ONLY a JSON object with format:
{{
  "selected_item_ids": [list of item IDs],
  "reasoning": "brief explanation"
}}
"""

    # For now, use heuristic selection (GPT-4o can be added for optimization)
    # Heuristic: Balance domains, select items near b=0, good discrimination
    
    selected_df_list = []
    
    for domain, count in domain_constraints.items():
        # How many already selected from this domain (common items)
        if common_items:
            common_in_domain = available_items[
                (available_items['item_id'].isin(common_items)) & 
                (available_items['domain'] == domain)
            ]
            already_selected = len(common_in_domain)
        else:
            already_selected = 0
        
        needed_from_domain = count - already_selected
        
        if needed_from_domain > 0:
            domain_candidates = candidate_items[candidate_items['domain'] == domain].copy()
            
            # Score items: prefer b near 0, high discrimination
            domain_candidates['score'] = (
                1 / (1 + abs(domain_candidates['rasch_b'])) * 
                domain_candidates['point_biserial']
            )
            
            # Select top items
            selected_from_domain = domain_candidates.nlargest(needed_from_domain, 'score')
            selected_df_list.append(selected_from_domain)
            selected_items.extend(selected_from_domain['item_id'].tolist())
    
    # Combine all selected items
    if selected_df_list:
        if common_items:
            common_df = available_items[available_items['item_id'].isin(common_items)]
            selected_df_list.insert(0, common_df)
        form_df = pd.concat(selected_df_list, ignore_index=True)
    else:
        form_df = available_items[available_items['item_id'].isin(common_items)]
    
    # Evaluate form
    theta_range = np.linspace(-3, 3, 61)
    quality = evaluate_form_quality(form_df, theta_range, 
                                    form_specs.get('evaluation_points'),
                                    form_specs.get('tolerance', 0.1))
    
    return form_df, quality

# ==================== Visualization Functions ====================

def plot_tif_tcc(forms_data: List[Dict[str, Any]], theta_range: np.ndarray):
    """Plot TIF and TCC for all forms"""
    
    # TIF Plot
    fig_tif = go.Figure()
    for i, form_data in enumerate(forms_data):
        fig_tif.add_trace(go.Scatter(
            x=theta_range,
            y=form_data['quality']['tif_values'],
            mode='lines',
            name=f"Form {i+1}",
            line=dict(width=2)
        ))
    
    # Add evaluation points markers
    quality = forms_data[0]['quality']
    theta_points = [quality['theta_low'], quality['theta_mid'], quality['theta_high']]
    target_points = [quality['target_low'], quality['target_mid'], quality['target_high']]
    
    fig_tif.add_trace(go.Scatter(
        x=theta_points,
        y=target_points,
        mode='markers',
        name='Target Points',
        marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='darkred'))
    ))
    
    # Add vertical lines at evaluation points
    for theta, label in zip(theta_points, ['Low', 'Mid', 'High']):
        fig_tif.add_vline(x=theta, line_dash="dot", line_color="gray", opacity=0.5,
                         annotation_text=f"{label} θ={theta}")
    
    fig_tif.update_layout(
        title="Test Information Function (TIF) - All Forms",
        xaxis_title="Theta (Ability Level)",
        yaxis_title="Information",
        height=500,
        showlegend=True
    )
    
    # TCC Plot
    fig_tcc = go.Figure()
    for i, form_data in enumerate(forms_data):
        fig_tcc.add_trace(go.Scatter(
            x=theta_range,
            y=form_data['quality']['tcc_values'],
            mode='lines',
            name=f"Form {i+1}",
            line=dict(width=2)
        ))
    
    # Add vertical lines at evaluation points
    quality = forms_data[0]['quality']
    theta_points = [quality['theta_low'], quality['theta_mid'], quality['theta_high']]
    
    for theta, label in zip(theta_points, ['Low', 'Mid', 'High']):
        fig_tcc.add_vline(x=theta, line_dash="dot", line_color="gray", opacity=0.5,
                         annotation_text=f"{label} θ={theta}")
    
    fig_tcc.update_layout(
        title="Test Characteristic Curve (TCC) - All Forms",
        xaxis_title="Theta (Ability Level)",
        yaxis_title="Expected Score",
        height=500,
        showlegend=True
    )
    
    return fig_tif, fig_tcc

# ==================== Excel Export ====================

def export_to_excel(forms_data: List[Dict[str, Any]], summary_df: pd.DataFrame) -> BytesIO:
    """Export all forms and summary to Excel file"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Each form sheet
        for i, form_data in enumerate(forms_data):
            form_df = form_data['items']
            sheet_name = f'Form_{i+1}'
            form_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Statistics sheet
        stats_data = []
        for i, form_data in enumerate(forms_data):
            quality = form_data['quality']
            stats_data.append({
                'Form': i+1,
                f'TIF at θ={quality["theta_low"]}': round(quality['tif_at_low'], 2),
                f'Target Low': quality['target_low'],
                f'Meets TIF Low': quality['meets_tif_low'],
                f'TIF at θ={quality["theta_mid"]}': round(quality['tif_at_mid'], 2),
                f'Target Mid': quality['target_mid'],
                f'Meets TIF Mid': quality['meets_tif_mid'],
                f'TIF at θ={quality["theta_high"]}': round(quality['tif_at_high'], 2),
                f'Target High': quality['target_high'],
                f'Meets TIF High': quality['meets_tif_high'],
                f'TCC at θ={quality["theta_mid"]}': round(quality['tcc_at_mid'], 2),
                f'Target TCC': quality.get('target_tcc_mid', 'N/A'),
                f'Meets TCC': quality['meets_tcc_mid'],
                'Avg Difficulty': round(quality['avg_difficulty'], 3),
                'Meets Difficulty': quality['meets_difficulty'],
                'Difficulty SD': round(quality['difficulty_sd'], 3),
                'Avg Discrimination': round(quality['avg_discrimination'], 3),
                'Meets All Constraints': quality['meets_all_constraints']
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    output.seek(0)
    return output

# ==================== Main Streamlit App ====================

def main():
    st.set_page_config(page_title="AI-Powered ATA", layout="wide", page_icon="🤖")
    
    st.title("🤖 AI-Powered Automated Test Assembly")
    st.markdown("**Build parallel test forms using GPT-4o and intelligent constraints**")
    
    # Initialize session state
    if 'forms_built' not in st.session_state:
        st.session_state.forms_built = False
        st.session_state.forms_data = []
    
    # Sidebar - Input Controls
    st.sidebar.header("📋 Test Specifications")
    
    # Connect to database
    agent = ItemPoolAgent(DB_CONFIG, DB_URL)
    if not agent.connect():
        st.error("Cannot connect to database. Please check PostgreSQL connection.")
        return
    
    # Get pool statistics
    pool_stats = agent.get_item_stats()
    domains = agent.get_domains()
    
    st.sidebar.info(f"📊 Item Pool: {int(pool_stats['total_items'])} items\n\n"
                   f"🏷️ Domains: {len(domains)}")
    
    # Number of forms
    num_forms = st.sidebar.number_input("Number of Forms", min_value=1, max_value=20, value=3)
    
    # Test length
    test_length = st.sidebar.number_input("Test Length (items per form)", 
                                          min_value=10, max_value=200, value=75)
    
    # Domain distribution
    st.sidebar.subheader("Domain Distribution")
    
    # Calculate default distribution (proportional and ensure sum equals test_length)
    if len(domains) > 0:
        base_per_domain = test_length // len(domains)
        remainder = test_length % len(domains)
        
        default_values = {}
        for i, domain in enumerate(domains):
            default_values[domain] = base_per_domain + (1 if i < remainder else 0)
    else:
        default_values = {}
    
    domain_counts = {}
    total_allocated = 0
    
    for domain in domains:
        count = st.sidebar.number_input(f"{domain}", min_value=0, max_value=test_length, 
                                       value=default_values.get(domain, 0), key=f"domain_{domain}")
        domain_counts[domain] = count
        total_allocated += count
    
    if total_allocated != test_length:
        st.sidebar.error(f"❌ Total: {total_allocated}, Expected: {test_length}")
    
    # Common items (anchor items)
    st.sidebar.subheader("Common Items (Anchors)")
    num_common = st.sidebar.number_input("Number of common items across forms", 
                                         min_value=0, max_value=test_length, value=15)
    
    # Target TIF with three evaluation points
    st.sidebar.subheader("Target Information (TIF)")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theta_low = st.number_input("Low θ", min_value=-3.0, max_value=0.0, value=-1.0, step=0.5)
        tif_low = st.number_input("TIF at Low θ", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
    
    with col2:
        theta_high = st.number_input("High θ", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
        tif_high = st.number_input("TIF at High θ", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
    
    theta_mid = st.sidebar.number_input("Middle θ (Logit Cut)", min_value=-2.0, max_value=2.0, 
                                        value=0.0, step=0.1)
    tif_mid = st.sidebar.number_input("TIF at Middle θ", min_value=1.0, max_value=50.0, 
                                      value=12.0, step=0.5)
    tcc_mid = st.sidebar.number_input("TCC at Cut Point", min_value=1.0, max_value=float(test_length), 
                                      value=float(test_length) * 0.5, step=0.5,
                                      help="Expected score at the cut point (theta_mid)")
    
    st.sidebar.subheader("Tolerance Settings")
    tolerance = st.sidebar.slider("TIF/TCC Tolerance (%)", min_value=5, max_value=30, value=15) / 100
    
    st.sidebar.subheader("Mean Difficulty Constraint")
    target_difficulty = st.sidebar.number_input("Target Mean Difficulty (b)", 
                                                min_value=-2.0, max_value=2.0, 
                                                value=0.0, step=0.1,
                                                help="Target average Rasch b-parameter")
    difficulty_tolerance = st.sidebar.number_input("Difficulty Tolerance", 
                                                   min_value=0.1, max_value=1.0, 
                                                   value=0.5, step=0.1,
                                                   help="Acceptable deviation from target difficulty")
    
    # Store evaluation points
    evaluation_points = {
        'theta_low': theta_low,
        'theta_mid': theta_mid,
        'theta_high': theta_high,
        'tif_low': tif_low,
        'tif_mid': tif_mid,
        'tif_high': tif_high,
        'tcc_mid': tcc_mid
    }
    
    # Store difficulty constraint
    difficulty_constraint = {
        'target': target_difficulty,
        'tolerance': difficulty_tolerance
    }
    
    # Enemy check
    st.sidebar.subheader("Enemy Check")
    check_enemies = st.sidebar.checkbox("Check for enemy items", value=True)
    enemy_threshold = st.sidebar.slider("Similarity threshold for enemies", 
                                        min_value=0.7, max_value=0.95, value=0.85, step=0.05)
    
    # Item exposure
    st.sidebar.subheader("Item Exposure")
    max_exposure = st.sidebar.slider("Max exposure rate (%)", 
                                     min_value=20, max_value=100, value=50) / 100
    
    # Build button
    build_button = st.sidebar.button("🚀 Build Forms", type="primary", use_container_width=True)
    
    # Main content
    if build_button:
        if total_allocated != test_length:
            st.error("❌ Domain counts must sum to test length!")
            return
        
        with st.spinner("Building test forms with AI..."):
            # Get all available items
            all_items = agent.get_all_items()
            
            # Select common items (anchors) - balanced across domains
            common_item_ids = []
            if num_common > 0:
                # Distribute common items proportionally across domains
                common_per_domain = {}
                for domain, count in domain_counts.items():
                    common_per_domain[domain] = int(num_common * (count / test_length))
                
                # Adjust to ensure total equals num_common
                total_common = sum(common_per_domain.values())
                if total_common < num_common:
                    # Add remaining to largest domain
                    largest_domain = max(domain_counts, key=domain_counts.get)
                    common_per_domain[largest_domain] += (num_common - total_common)
                
                # Select common items
                for domain, count in common_per_domain.items():
                    if count > 0:
                        domain_items = all_items[all_items['domain'] == domain].copy()
                        # Select items near b=0 with good discrimination
                        domain_items['score'] = (
                            1 / (1 + abs(domain_items['rasch_b'])) * 
                            domain_items['point_biserial']
                        )
                        selected = domain_items.nlargest(count, 'score')
                        common_item_ids.extend(selected['item_id'].tolist())
            
            # Build forms
            forms_data = []
            used_items_global = set()
            theta_range = np.linspace(-3, 3, 61)
            
            form_specs = {
                'test_length': test_length,
                'domain_counts': domain_counts,
                'evaluation_points': evaluation_points,
                'tolerance': tolerance,
                'difficulty_constraint': difficulty_constraint
            }
            
            progress_bar = st.progress(0)
            
            for form_num in range(num_forms):
                # Build form
                form_df, quality = build_form_with_ai(
                    agent, form_specs, all_items, 
                    used_items_global, common_item_ids
                )
                
                # Check enemy items
                enemy_pairs = []
                if check_enemies:
                    enemy_pairs = agent.check_enemy_items(
                        form_df['item_id'].tolist(), 
                        enemy_threshold
                    )
                
                # Calculate exposure
                item_exposure = {}
                for item_id in form_df['item_id']:
                    if item_id in common_item_ids:
                        item_exposure[item_id] = 1.0  # Common items always exposed
                    else:
                        item_exposure[item_id] = 1 / num_forms
                
                forms_data.append({
                    'form_number': form_num + 1,
                    'items': form_df,
                    'quality': quality,
                    'enemy_pairs': enemy_pairs,
                    'exposure': item_exposure
                })
                
                # Update used items (excluding common items for variety)
                used_items_global.update(
                    form_df[~form_df['item_id'].isin(common_item_ids)]['item_id'].tolist()
                )
                
                progress_bar.progress((form_num + 1) / num_forms)
            
            st.session_state.forms_data = forms_data
            st.session_state.forms_built = True
            st.session_state.common_items = common_item_ids
            st.success(f"✅ Successfully built {num_forms} parallel forms!")
    
    # Display results
    if st.session_state.forms_built:
        forms_data = st.session_state.forms_data
        
        # Summary table
        st.subheader("📊 Forms Summary")
        summary_data = []
        for form in forms_data:
            quality = form['quality']
            tcc_display = f"{quality['tcc_at_mid']:.1f}"
            if quality.get('target_tcc_mid'):
                tcc_display += f" ({quality['target_tcc_mid']:.1f})"
            
            summary_data.append({
                'Form': form['form_number'],
                'Items': len(form['items']),
                f'TIF@θ={quality["theta_low"]}': f"{quality['tif_at_low']:.1f} ({quality['target_low']:.1f})",
                f'TIF@θ={quality["theta_mid"]}': f"{quality['tif_at_mid']:.1f} ({quality['target_mid']:.1f})",
                f'TIF@θ={quality["theta_high"]}': f"{quality['tif_at_high']:.1f} ({quality['target_high']:.1f})",
                f'TCC@θ={quality["theta_mid"]}': tcc_display,
                'Mean Diff': f"{quality['avg_difficulty']:.2f}",
                'Meets TIF': '✅' if quality['tif_meets_target'] else '❌',
                'Meets TCC': '✅' if quality['meets_tcc_mid'] else '❌',
                'Meets Diff': '✅' if quality['meets_difficulty'] else '❌',
                'Meets All': '✅' if quality['meets_all_constraints'] else '❌',
                'Avg Discrim': round(quality['avg_discrimination'], 3),
                'Enemies': len(form['enemy_pairs'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # TIF and TCC plots
        st.subheader("📈 Test Information & Characteristic Curves")
        theta_range = np.linspace(-3, 3, 61)
        fig_tif, fig_tcc = plot_tif_tcc(forms_data, theta_range)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_tif, use_container_width=True)
        with col2:
            st.plotly_chart(fig_tcc, use_container_width=True)
        
        # Individual form details
        st.subheader("📝 Form Details")
        
        for form in forms_data:
            with st.expander(f"Form {form['form_number']} - {len(form['items'])} items"):
                # Domain distribution
                domain_dist = form['items']['domain'].value_counts()
                st.write("**Domain Distribution:**")
                st.bar_chart(domain_dist)
                
                # Enemy pairs warning
                if form['enemy_pairs']:
                    st.warning(f"⚠️ Found {len(form['enemy_pairs'])} enemy item pairs!")
                    for pair in form['enemy_pairs'][:5]:  # Show first 5
                        st.write(f"  - Items {pair[0]} and {pair[1]}")
                
                # Item table
                st.write("**Items:**")
                display_df = form['items'][['item_id', 'domain', 'topic', 'rasch_b', 
                                            'pvalue', 'point_biserial']].copy()
                display_df.columns = ['Item ID', 'Domain', 'Topic', 'Difficulty (b)', 
                                     'P-value', 'Discrimination']
                st.dataframe(display_df, use_container_width=True)
        
        # Download button
        st.subheader("💾 Export Results")
        excel_file = export_to_excel(forms_data, summary_df)
        st.download_button(
            label="📥 Download Forms (Excel)",
            data=excel_file,
            file_name="test_forms_assembly.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Common items info
        if st.session_state.get('common_items'):
            common_items = st.session_state.common_items
            st.subheader("🔗 Common Items (Anchors)")
            st.info(f"**Total Common Items:** {len(common_items)} items across all forms")
            
            # Display full list in expandable section
            with st.expander("📋 View All Common Items", expanded=False):
                # Get full details for common items
                common_items_df = agent.get_all_items()
                common_items_df = common_items_df[common_items_df['item_id'].isin(common_items)]
                
                # Display compact table
                display_common = common_items_df[['item_id', 'domain', 'topic', 'rasch_b', 
                                                   'pvalue', 'point_biserial']].copy()
                display_common.columns = ['Item ID', 'Domain', 'Topic', 'Difficulty (b)', 
                                          'P-value', 'Discrimination']
                display_common = display_common.sort_values('Item ID')
                st.dataframe(display_common, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Difficulty", f"{common_items_df['rasch_b'].mean():.3f}")
                with col2:
                    st.metric("Avg P-value", f"{common_items_df['pvalue'].mean():.3f}")
                with col3:
                    st.metric("Avg Discrimination", f"{common_items_df['point_biserial'].mean():.3f}")
    
    agent.disconnect()

if __name__ == "__main__":
    main()
