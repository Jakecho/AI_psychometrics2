"""
CAT - Computerized Adaptive Testing
========================================

Advanced CAT implementation with Rasch IRT model.

Features:
- Rasch 1PL IRT model
- MLE with fence scoring (Chris Han's approach)
- Maximum information item selection
- Content-balanced selection with weighted random domain selection
- Exposure control (Sympson-Hetter method)
- Multiple stopping rules (max length, max time, SEM threshold)
- PostgreSQL database integration

Author: AI Assistant
Date: December 3, 2025
"""

import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
import plotly.graph_objects as go
from io import BytesIO
import os
from sqlalchemy import create_engine
from datetime import datetime
import time
from scipy.optimize import minimize_scalar

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

# Page configuration
st.set_page_config(
    page_title="AI CAT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cat-metric {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        margin: 0.5rem 0;
    }
    .item-display {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #ff9800;
        margin: 1rem 0;
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
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
               "choice_C", "choice_D", key, rasch_b, point_biserial
        FROM itembank
        WHERE embedding IS NOT NULL AND rasch_b IS NOT NULL
        ORDER BY item_id
        """
        return pd.read_sql_query(query, self.engine)
    
    def get_domains(self) -> List[str]:
        """Get list of unique domains"""
        query = "SELECT DISTINCT domain FROM itembank WHERE domain IS NOT NULL ORDER BY domain"
        df = pd.read_sql_query(query, self.engine)
        return df['domain'].tolist()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get item pool statistics"""
        query = """
        SELECT 
            COUNT(*) as total_items,
            COUNT(DISTINCT domain) as total_domains,
            AVG(rasch_b) as avg_difficulty,
            MIN(rasch_b) as min_difficulty,
            MAX(rasch_b) as max_difficulty,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rasch_b) as median_difficulty
        FROM itembank
        WHERE embedding IS NOT NULL AND rasch_b IS NOT NULL
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

# ==================== Rasch IRT Functions ====================

def rasch_probability(theta: float, b: float) -> float:
    """
    Calculate probability of correct response (Rasch 1PL)
    
    P(θ, b) = 1 / (1 + exp(-(θ - b)))
    
    Args:
        theta: Ability level
        b: Item difficulty
    
    Returns:
        Probability of correct response (0-1)
    """
    return 1 / (1 + np.exp(-(theta - b)))

def rasch_information(theta: float, b: float) -> float:
    """
    Calculate Fisher information for Rasch model
    
    I(θ, b) = P(θ, b) × Q(θ, b)
    
    Args:
        theta: Ability level
        b: Item difficulty
    
    Returns:
        Information value
    """
    p = rasch_probability(theta, b)
    q = 1 - p
    return p * q

# ==================== MLE with Fence Scoring ====================

def mle_with_fence(responses: List[Tuple[float, int]], 
                   theta_min: float = -4.0, 
                   theta_max: float = 4.0) -> Tuple[float, float]:
    """
    MLE estimation with fence constraints (Chris Han's approach)
    
    Args:
        responses: List of (b_value, score) tuples
        theta_min: Lower fence
        theta_max: Upper fence
    
    Returns:
        (theta_estimate, sem)
    """
    if len(responses) == 0:
        return 0.0, 999.0
    
    scores = [score for _, score in responses]
    
    # Check for perfect scores
    if sum(scores) == 0:
        # All incorrect: return lower fence
        theta = theta_min
    elif sum(scores) == len(scores):
        # All correct: return upper fence
        theta = theta_max
    else:
        # Use Newton-Raphson within fence
        theta = newton_raphson_mle(responses, theta_min, theta_max)
    
    # Calculate SEM
    total_info = sum(rasch_information(theta, b) for b, _ in responses)
    sem = 1 / np.sqrt(total_info) if total_info > 0 else 999.0
    
    return theta, sem

def newton_raphson_mle(responses: List[Tuple[float, int]], 
                       theta_min: float, theta_max: float,
                       max_iter: int = 50, tol: float = 0.001) -> float:
    """
    Newton-Raphson method for MLE with fencing
    """
    theta = 0.0  # Start at neutral ability
    
    for iteration in range(max_iter):
        # First derivative (score)
        first_deriv = 0.0
        second_deriv = 0.0
        
        for b, u in responses:
            p = rasch_probability(theta, b)
            first_deriv += (u - p)
            second_deriv += -p * (1 - p)
        
        # Newton step
        if abs(second_deriv) < 1e-10:
            break
        
        step = -first_deriv / second_deriv
        theta_new = theta + step
        
        # Apply fence
        theta_new = max(theta_min, min(theta_max, theta_new))
        
        # Check convergence
        if abs(theta_new - theta) < tol:
            break
        
        theta = theta_new
    
    return theta

# ==================== Content Balancing ====================

class ContentBalancer:
    """Manages content balancing for CAT using weighted random selection"""
    
    def __init__(self, domain_targets: Dict[str, int]):
        """
        Initialize with target domain counts
        
        Args:
            domain_targets: {domain: target_count}
        """
        self.domain_target_counts = domain_targets
        total_target = sum(domain_targets.values())
        
        # Convert counts to proportions for weighting
        if total_target > 0:
            self.domain_targets = {
                domain: count / total_target 
                for domain, count in domain_targets.items()
            }
        else:
            # Equal proportions if no targets specified
            n_domains = len(domain_targets)
            self.domain_targets = {
                domain: 1.0 / n_domains 
                for domain in domain_targets
            }
        
        self.domain_selected = {domain: 0 for domain in domain_targets}
        self.total_items = 0
    
    def get_domain_weights(self) -> Dict[str, float]:
        """
        Calculate current domain weights for random selection
        
        Weight = max(0.01, Target - Current_proportion + 0.1)
        """
        weights = {}
        
        for domain, target in self.domain_targets.items():
            current_prop = (self.domain_selected[domain] / self.total_items 
                           if self.total_items > 0 else 0)
            # Weight higher if under-represented
            weights[domain] = max(0.01, target - current_prop + 0.1)
        
        # Normalize
        total_weight = sum(weights.values())
        return {d: w/total_weight for d, w in weights.items()}
    
    def select_domain(self) -> str:
        """Randomly select domain based on weights"""
        weights = self.get_domain_weights()
        domains = list(weights.keys())
        probabilities = [weights[d] for d in domains]
        
        return np.random.choice(domains, p=probabilities)
    
    def update(self, domain: str):
        """Update counts after selecting item from domain"""
        self.domain_selected[domain] += 1
        self.total_items += 1
    
    def get_coverage_status(self) -> Dict[str, Dict]:
        """Get current vs target coverage"""
        status = {}
        for domain, target in self.domain_targets.items():
            actual = (self.domain_selected[domain] / self.total_items 
                     if self.total_items > 0 else 0)
            target_count = self.domain_target_counts[domain]
            actual_count = self.domain_selected[domain]
            
            status[domain] = {
                'target': target,
                'actual': actual,
                'count': actual_count,
                'target_count': target_count,
                'difference': actual - target
            }
        return status

# ==================== Exposure Control ====================

class ExposureControl:
    """Manages item exposure control using Sympson-Hetter method"""
    
    def __init__(self, max_exposure_rate: float = 0.25):
        """
        Args:
            max_exposure_rate: Maximum proportion of tests an item can appear in
        """
        self.max_exposure_rate = max_exposure_rate
        self.item_usage = {}  # {item_id: usage_count}
        self.total_tests = 0
    
    def filter_available(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out over-exposed items"""
        if self.total_tests == 0:
            return items_df
        
        max_usage = self.max_exposure_rate * self.total_tests
        
        available = items_df[
            items_df['item_id'].apply(
                lambda iid: self.item_usage.get(iid, 0) < max_usage
            )
        ]
        
        return available
    
    def get_least_exposed(self, items_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Get k least exposed items"""
        items_df = items_df.copy()
        items_df['exposure'] = items_df['item_id'].apply(
            lambda iid: self.item_usage.get(iid, 0)
        )
        return items_df.nsmallest(k, 'exposure')
    
    def record_usage(self, item_id: int):
        """Record that item was used"""
        self.item_usage[item_id] = self.item_usage.get(item_id, 0) + 1
    
    def start_new_test(self):
        """Signal start of new test"""
        self.total_tests += 1
    
    def get_exposure_rate(self, item_id: int) -> float:
        """Get current exposure rate for an item"""
        if self.total_tests == 0:
            return 0.0
        return self.item_usage.get(item_id, 0) / self.total_tests
    
    def get_pool_utilization_metrics(self, total_pool_size: int) -> Dict[str, Any]:
        """
        Calculate pool utilization and exposure balance metrics
        
        Returns:
            Dict with utilization statistics
        """
        if self.total_tests == 0:
            return {
                'utilization_rate': 0.0,
                'items_used': 0,
                'items_available': total_pool_size,
                'avg_usage': 0.0,
                'max_usage': 0,
                'min_usage': 0,
                'balance_index': 1.0
            }
        
        used_items = len(self.item_usage)
        usage_counts = list(self.item_usage.values())
        
        utilization_rate = used_items / total_pool_size if total_pool_size > 0 else 0
        avg_usage = np.mean(usage_counts) if usage_counts else 0
        std_usage = np.std(usage_counts) if len(usage_counts) > 1 else 0
        
        # Balance index: 1.0 = perfect balance, 0.0 = highly unbalanced
        balance_index = 1 - (std_usage / avg_usage) if avg_usage > 0 else 1.0
        balance_index = max(0.0, min(1.0, balance_index))  # Clamp to [0, 1]
        
        return {
            'utilization_rate': utilization_rate,
            'items_used': used_items,
            'items_available': total_pool_size,
            'avg_usage': float(avg_usage),
            'max_usage': int(max(usage_counts)) if usage_counts else 0,
            'min_usage': int(min(usage_counts)) if usage_counts else 0,
            'balance_index': float(balance_index)
        }

# ==================== Stopping Rules ====================

class StoppingRules:
    """Manages CAT stopping criteria"""
    
    def __init__(self, 
                 max_items: int = 75,
                 max_time_minutes: float = 60.0,
                 sem_threshold: float = 0.3,
                 cut_score: float = None,
                 min_items_for_cut_score: int = 10):
        """
        Args:
            max_items: Maximum number of items
            max_time_minutes: Maximum time allowed
            sem_threshold: Stop when SEM below this value
            cut_score: Cut score for mastery decision (optional)
            min_items_for_cut_score: Minimum items before cut score rule applies
        """
        self.max_items = max_items
        self.max_time_seconds = max_time_minutes * 60
        self.sem_threshold = sem_threshold
        self.cut_score = cut_score
        self.min_items_for_cut_score = min_items_for_cut_score
    
    def check_stopping(self, 
                      n_items: int,
                      current_sem: float,
                      elapsed_time: float = None,
                      theta: float = None) -> Tuple[bool, str]:
        """
        Check if any stopping criterion is met
        
        Args:
            n_items: Number of items
            current_sem: Current SEM
            elapsed_time: Elapsed time in seconds
            theta: Current ability estimate (for cut score check)
        
        Returns:
            (should_stop, reason)
        """
        # Check cut score rule first (highest priority) - only after minimum items
        if self.cut_score is not None and theta is not None and n_items >= self.min_items_for_cut_score:
            # Lower bound of 68% CI: θ - 1.0 * SEM
            lower_bound_68 = theta - 1.0 * current_sem
            if lower_bound_68 > self.cut_score:
                return True, f"Cut score criterion met (68% CI lower bound: {lower_bound_68:.2f} > {self.cut_score:.2f})"
        
        # Max items
        if n_items >= self.max_items:
            return True, f"Maximum items reached ({self.max_items})"
        
        # SEM threshold
        if current_sem <= self.sem_threshold:
            return True, f"SEM threshold met (SEM={current_sem:.3f} ≤ {self.sem_threshold})"
        
        # Max time
        if elapsed_time and elapsed_time >= self.max_time_seconds:
            return True, f"Maximum time exceeded ({elapsed_time/60:.1f} min)"
        
        return False, ""

# ==================== Item Selection ====================

def select_starting_item(item_pool: pd.DataFrame) -> pd.Series:
    """
    Select starting item at median difficulty
    
    Args:
        item_pool: DataFrame with items
    
    Returns:
        Series with selected item
    """
    median_b = item_pool['rasch_b'].median()
    closest_idx = (item_pool['rasch_b'] - median_b).abs().idxmin()
    return item_pool.loc[closest_idx]

def select_next_item(
    theta_current: float,
    available_items: pd.DataFrame,
    administered_items: List[int],
    content_balancer: ContentBalancer,
    exposure_control: ExposureControl,
    enemy_index: Dict[int, Set[int]] = None
) -> pd.Series:
    """
    Select next item using max information with content constraints
    
    Procedure:
    1. Randomly select domain (content balancing)
    2. Filter items in that domain
    3. Apply exposure control
    4. Filter enemy items (avoid conflicts with administered items)
    5. Select item with maximum information
    
    Returns:
        Series with selected item
    """
    # Step 1: Select domain
    domain = content_balancer.select_domain()
    
    # Step 2: Filter domain items
    domain_items = available_items[available_items['domain'] == domain].copy()
    
    if len(domain_items) == 0:
        # Fallback: use all available items if domain empty
        domain_items = available_items.copy()
    
    # Step 3: Apply exposure control
    unexposed_items = exposure_control.filter_available(domain_items)
    
    if len(unexposed_items) == 0:
        # All items over-exposed, fall back to least exposed
        unexposed_items = exposure_control.get_least_exposed(domain_items, k=10)
    
    # Step 4: Filter enemy items
    if enemy_index:
        # Get items without enemy conflicts
        non_enemy_items = []
        for idx, row in unexposed_items.iterrows():
            item_id = row['item_id']
            enemies = enemy_index.get(item_id, set())
            # Check if any enemy is in administered items
            has_conflict = bool(enemies & set(administered_items))
            if not has_conflict:
                non_enemy_items.append(idx)
        
        if non_enemy_items:
            unexposed_items = unexposed_items.loc[non_enemy_items]
        # If all items have conflicts, proceed anyway (prioritize other constraints)
    
    # Step 5: Calculate information for each item
    unexposed_items['information'] = unexposed_items['rasch_b'].apply(
        lambda b: rasch_information(theta_current, b)
    )
    
    # Select item with max information
    max_info_idx = unexposed_items['information'].idxmax()
    return unexposed_items.loc[max_info_idx]

# ==================== CAT Simulation ====================

def simulate_response(true_theta: float, item_b: float) -> int:
    """Simulate examinee response given true theta"""
    prob = rasch_probability(true_theta, item_b)
    return 1 if np.random.random() < prob else 0

def run_cat_session(
    item_pool: pd.DataFrame,
    domain_targets: Dict[str, float],
    stopping_rules: StoppingRules,
    exposure_control: ExposureControl,
    true_theta: float = 0.0,
    simulation_mode: bool = True,
    enemy_index: Dict[int, Set[int]] = None
) -> Dict[str, Any]:
    """
    Run a complete CAT session
    
    Args:
        item_pool: DataFrame with items
        domain_targets: Content proportions
        stopping_rules: Stopping criteria
        exposure_control: Exposure control manager
        true_theta: True ability (for simulation)
        simulation_mode: If True, simulate responses
        enemy_index: Optional enemy item index for conflict detection
    
    Returns:
        Dict with CAT results
    """
    # Initialize
    content_balancer = ContentBalancer(domain_targets)
    responses = []
    administered_items = []
    theta_history = [0.0]
    sem_history = [999.0]
    information_history = []
    
    # Start exposure tracking
    exposure_control.start_new_test()
    
    # Select starting item
    current_item = select_starting_item(item_pool)
    administered_items.append(current_item['item_id'])
    
    # Start timer
    start_time = time.time()
    
    # Initial theta estimate
    theta_current = 0.0
    
    # CAT session data for display
    session_data = {
        'items': [],
        'responses': [],
        'thetas': [theta_current],
        'sems': [999.0]
    }
    
    iteration = 0
    max_iterations = stopping_rules.max_items
    
    while iteration < max_iterations:
        # Simulate or get response
        if simulation_mode:
            response = simulate_response(true_theta, current_item['rasch_b'])
        else:
            # In real CAT, would present item and get response
            response = 1  # Placeholder
        
        # Record response
        responses.append((current_item['rasch_b'], response))
        
        # Update trackers
        exposure_control.record_usage(current_item['item_id'])
        content_balancer.update(current_item['domain'])
        
        # Store item info
        session_data['items'].append({
            'item_id': current_item['item_id'],
            'domain': current_item['domain'],
            'difficulty': current_item['rasch_b'],
            'response': response
        })
        
        # Estimate theta
        theta_current, sem_current = mle_with_fence(responses)
        theta_history.append(theta_current)
        sem_history.append(sem_current)
        
        session_data['thetas'].append(theta_current)
        session_data['sems'].append(sem_current)
        session_data['responses'].append(response)
        
        # Calculate information
        current_info = rasch_information(theta_current, current_item['rasch_b'])
        information_history.append(current_info)
        
        # Check stopping rules (now includes theta for cut score check)
        elapsed_time = time.time() - start_time
        should_stop, reason = stopping_rules.check_stopping(
            len(administered_items), sem_current, elapsed_time, theta_current
        )
        
        if should_stop:
            break
        
        # Select next item
        available = item_pool[~item_pool['item_id'].isin(administered_items)]
        
        if len(available) == 0:
            reason = "No more items available"
            break
        
        current_item = select_next_item(
            theta_current, available, administered_items, content_balancer, exposure_control, enemy_index
        )
        
        administered_items.append(current_item['item_id'])
        iteration += 1
    
    # Compile final results
    return {
        'final_theta': theta_current,
        'final_sem': sem_current,
        'true_theta': true_theta,
        'n_items': len(administered_items),
        'administered_items': administered_items,
        'responses': responses,
        'theta_history': theta_history,
        'sem_history': sem_history,
        'information_history': information_history,
        'stopping_reason': reason,
        'elapsed_time': elapsed_time,
        'domain_distribution': content_balancer.get_coverage_status(),
        'session_data': session_data,
        'exposure_control': exposure_control,  # For metrics calculation
        'pool_size': len(item_pool)  # Total pool size
    }

# ==================== Visualization ====================

def plot_theta_convergence(theta_history: List[float], true_theta: float = None):
    """Plot theta estimates over items"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(theta_history))),
        y=theta_history,
        mode='lines+markers',
        name='θ Estimate',
        line=dict(color='#2e7d32', width=2)
    ))
    
    if true_theta is not None:
        fig.add_hline(
            y=true_theta,
            line_dash="dash",
            line_color="red",
            annotation_text=f"True θ = {true_theta:.2f}"
        )
    
    fig.update_layout(
        title="Ability Estimate Convergence",
        xaxis_title="Number of Items",
        yaxis_title="θ (Ability)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_sem_reduction(sem_history: List[float], sem_threshold: float):
    """Plot SEM reduction over items with adaptive range"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(sem_history))),
        y=sem_history,
        mode='lines+markers',
        name='SEM',
        line=dict(color='#ff9800', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 152, 0, 0.1)'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=sem_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Threshold = {sem_threshold}",
        annotation_position="right"
    )
    
    # Intelligent y-axis range
    max_sem = max(sem_history) if sem_history else 3.0
    
    # Ensure threshold is visible
    max_display = max(max_sem, sem_threshold * 1.2)
    
    # Use fixed range [0, 3] if data fits, otherwise adapt
    if max_display <= 3.0:
        yaxis_range = [0, 3]
    else:
        # Cap at 10 for extreme early values
        yaxis_range = [0, min(max_display * 1.1, 10)]
    
    fig.update_layout(
        title="Standard Error of Measurement (SEM) Convergence",
        xaxis_title="Number of Items",
        yaxis_title="SEM",
        yaxis_range=yaxis_range,
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def plot_content_coverage(coverage_status: Dict[str, Dict]) -> go.Figure:
    """Plot content coverage progress"""
    domains = list(coverage_status.keys())
    target_counts = [coverage_status[d]['target_count'] for d in domains]
    actual_counts = [coverage_status[d]['count'] for d in domains]
    
    fig = go.Figure()
    
    # Target bars (lighter)
    fig.add_trace(go.Bar(
        name='Target',
        x=domains,
        y=target_counts,
        marker_color='lightblue',
        opacity=0.5
    ))
    
    # Actual bars (darker)
    fig.add_trace(go.Bar(
        name='Actual',
        x=domains,
        y=actual_counts,
        marker_color='#2e7d32'
    ))
    
    fig.update_layout(
        title="Content Coverage: Actual vs Target",
        xaxis_title="Domain",
        yaxis_title="Item Count",
        barmode='overlay',
        height=350,
        hovermode='x unified'
    )
    
    return fig

# ==================== Main Application ====================

def main():
    st.markdown('<div class="main-header">🎓 CAT Demo - Computerized Adaptive Testing</div>', unsafe_allow_html=True)
    st.markdown("**Rasch IRT Model with Content Balancing and Exposure Control**")
    
    # Initialize session state
    if 'cat_results' not in st.session_state:
        st.session_state.cat_results = None
        st.session_state.cat_running = False
    
    # Connect to database
    agent = ItemPoolAgent(DB_CONFIG, DB_URL)
    
    with st.spinner("Connecting to database..."):
        if not agent.connect():
            st.error("❌ Could not connect to database. Please check your configuration.")
            return
    
    # Get pool stats
    pool_stats = agent.get_pool_stats()
    domains = agent.get_domains()
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ CAT Configuration")
    
    st.sidebar.subheader("📊 Item Pool")
    st.sidebar.metric("Total Items", int(pool_stats['total_items']))
    st.sidebar.metric("Median Difficulty", f"{pool_stats['median_difficulty']:.2f}")
    
    st.sidebar.divider()
    
    # Stopping rules
    st.sidebar.subheader("🛑 Stopping Rules")
    
    st.sidebar.divider()
    
    # Content balancing - MOVED UP to calculate max_items from domain counts
    st.sidebar.subheader("🎯 Content Targets (Item Counts)")
    domain_targets = {}
    
    for domain in domains:
        count = st.sidebar.number_input(
            f"{domain}",
            min_value=0,
            max_value=150,
            value=max(1, 75 // len(domains)),  # Default: evenly distributed
            step=1,
            key=f"target_{domain}"
        )
        domain_targets[domain] = count
    
    # Calculate max_items from domain counts
    max_items = sum(domain_targets.values())
    
    # Show calculated total
    st.sidebar.metric("📊 Total Items", max_items)
    
    # Calculate total and show warning if needed
    if max_items == 0:
        st.sidebar.warning("⚠️ Total items is 0. Please set domain targets.")
        max_items = 75  # Fallback
    
    st.sidebar.divider()
    
    # Other stopping rules
    st.sidebar.markdown("**Time & Precision Limits:**")
    max_time = st.sidebar.slider("Max Time (minutes)", 10, 120, 60)
    sem_threshold = st.sidebar.slider("SEM Threshold", 0.1, 0.5, 0.3, 0.05)
    
    # Cut score for mastery decision
    use_cut_score = st.sidebar.checkbox("Use Cut Score", value=False, help="Stop when 68% CI lower bound exceeds cut score")
    cut_score = None
    min_items_for_cut_score = 10
    if use_cut_score:
        cut_score = st.sidebar.slider("Cut Score (θ)", -3.0, 3.0, 0.0, 0.1, 
                                       help="Stop when lower bound of 68% CI is above this value")
        min_items_for_cut_score = st.sidebar.slider("Min Items for Cut Score", 5, 50, 10, 5,
                                                     help="Minimum items before cut score rule applies")
    
    st.sidebar.divider()
    
    # Exposure control
    st.sidebar.subheader("🔒 Exposure Control")
    max_exposure = st.sidebar.slider("Max Exposure Rate", 0.1, 0.5, 0.25, 0.05)
    
    # Enemy item detection
    use_enemy_detection = st.sidebar.checkbox("Enemy Item Detection", value=True, help="Avoid items marked as enemies in the database")
    
    # Simulation settings
    st.sidebar.divider()
    st.sidebar.subheader("🎲 Testing Mode")
    
    # Mode selection
    test_mode = st.sidebar.radio(
        "Select Mode",
        ["Simulation", "Live Testing"],
        help="Simulation: Auto-generate responses | Live Testing: Answer items yourself"
    )
    
    simulation_mode = (test_mode == "Simulation")
    
    if simulation_mode:
        true_theta = st.sidebar.slider("True Ability (θ)", -3.0, 3.0, 0.0, 0.1)
    else:
        true_theta = 0.0  # Not used in live mode
        st.sidebar.info("💡 You will answer items interactively")
    
    # Run CAT button
    st.sidebar.divider()
    
    if simulation_mode:
        # Simulation mode - run entire test at once
        if st.sidebar.button("▶️ Start CAT", type="primary", use_container_width=True):
            # Load item pool
            with st.spinner("Loading item pool..."):
                item_pool = agent.get_all_items()
            
            # Build enemy index if enabled
            enemy_index = None
            if use_enemy_detection:
                with st.spinner("Building enemy item index..."):
                    enemy_index = build_enemy_index(item_pool, agent)
                    n_enemy_pairs = sum(len(enemies) for enemies in enemy_index.values()) // 2
                    if n_enemy_pairs > 0:
                        st.info(f"📊 Detected {n_enemy_pairs} enemy item pairs from database")
            
            # Initialize components
            stopping_rules = StoppingRules(max_items, max_time, sem_threshold, cut_score, min_items_for_cut_score)
            exposure_control = ExposureControl(max_exposure)
            
            # Run CAT
            with st.spinner("Running CAT simulation..."):
                results = run_cat_session(
                    item_pool,
                    domain_targets,
                    stopping_rules,
                    exposure_control,
                    true_theta,
                    simulation_mode=True,
                    enemy_index=enemy_index
                )
            
            st.session_state.cat_results = results
    else:
        # Live testing mode - interactive
        if 'live_cat_active' not in st.session_state:
            st.session_state.live_cat_active = False
        
        if not st.session_state.live_cat_active:
            if st.sidebar.button("▶️ Start Live CAT", type="primary", use_container_width=True):
                # Initialize live CAT session
                with st.spinner("Loading item pool..."):
                    item_pool = agent.get_all_items()
                
                # Build enemy index if enabled
                enemy_index = None
                if use_enemy_detection:
                    with st.spinner("Building enemy item index..."):
                        enemy_index = build_enemy_index(item_pool, agent)
                
                # Initialize components
                stopping_rules = StoppingRules(max_items, max_time, sem_threshold, cut_score, min_items_for_cut_score)
                exposure_control = ExposureControl(max_exposure)
                content_balancer = ContentBalancer(domain_targets)
                
                # Initialize session state for live CAT
                st.session_state.live_cat_active = True
                st.session_state.live_item_pool = item_pool
                st.session_state.live_enemy_index = enemy_index
                st.session_state.live_stopping_rules = stopping_rules
                st.session_state.live_exposure_control = exposure_control
                st.session_state.live_content_balancer = content_balancer
                st.session_state.live_responses = []
                st.session_state.live_administered_items = []
                st.session_state.live_theta_history = [0.0]
                st.session_state.live_sem_history = [999.0]
                st.session_state.live_theta_current = 0.0
                st.session_state.live_session_data = {'items': [], 'responses': [], 'thetas': [0.0], 'sems': [999.0]}
                st.session_state.live_start_time = time.time()
                
                # Select starting item
                exposure_control.start_new_test()
                starting_item = select_starting_item(item_pool)
                st.session_state.live_current_item = starting_item
                st.session_state.live_administered_items.append(starting_item['item_id'])
                
                st.rerun()
        else:
            if st.sidebar.button("❌ End Test", type="secondary", use_container_width=True):
                # End live CAT and show results
                st.session_state.live_cat_active = False
                # Compile results
                elapsed_time = time.time() - st.session_state.live_start_time
                results = {
                    'final_theta': st.session_state.live_theta_current,
                    'final_sem': st.session_state.live_sem_history[-1],
                    'true_theta': 0.0,
                    'n_items': len(st.session_state.live_administered_items),
                    'administered_items': st.session_state.live_administered_items,
                    'responses': st.session_state.live_responses,
                    'theta_history': st.session_state.live_theta_history,
                    'sem_history': st.session_state.live_sem_history,
                    'information_history': [],
                    'stopping_reason': 'User ended test',
                    'elapsed_time': elapsed_time,
                    'domain_distribution': st.session_state.live_content_balancer.get_coverage_status(),
                    'session_data': st.session_state.live_session_data,
                    'exposure_control': st.session_state.live_exposure_control,
                    'pool_size': len(st.session_state.live_item_pool)
                }
                st.session_state.cat_results = results
                st.rerun()
    
    
    # Live Testing Interface
    if st.session_state.get('live_cat_active', False):
        st.header("🎯 Live CAT Session")
        
        # Calculate elapsed time
        elapsed_time = time.time() - st.session_state.live_start_time
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        
        # Progress info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Items Completed", len(st.session_state.live_administered_items) - 1)
        with col2:
            st.metric("Elapsed Time", f"{elapsed_minutes}:{elapsed_seconds:02d}")
        with col3:
            st.metric("Current θ Estimate", f"{st.session_state.live_theta_current:.2f}")
        with col4:
            st.metric("Current SEM", f"{st.session_state.live_sem_history[-1]:.3f}")
        
        st.divider()
        
        # MLE Theta Estimate with Confidence Interval
        if len(st.session_state.live_theta_history) > 1:
            with st.expander("📈 Ability Estimate (θ) with 95% Confidence Interval", expanded=True):
                fig_theta = go.Figure()
                
                # Calculate 95% CI bounds (θ ± 1.96 * SEM)
                theta_values = st.session_state.live_theta_history
                sem_values = st.session_state.live_sem_history
                
                # Upper and lower bounds, capped to prevent excessive width
                upper_bound = [min(theta + 1.96 * sem, theta + 3.0) for theta, sem in zip(theta_values, sem_values)]
                lower_bound = [max(theta - 1.96 * sem, theta - 3.0) for theta, sem in zip(theta_values, sem_values)]
                
                x_values = list(range(len(theta_values)))
                
                # Add CI band (filled area)
                fig_theta.add_trace(go.Scatter(
                    x=x_values + x_values[::-1],  # x, then x reversed
                    y=upper_bound + lower_bound[::-1],  # upper, then lower reversed
                    fill='toself',
                    fillcolor='rgba(46, 125, 50, 0.2)',  # Light green with transparency
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=True,
                    name='95% CI',
                    hoverinfo='skip'
                ))
                
                # Theta line (on top of CI)
                fig_theta.add_trace(go.Scatter(
                    x=x_values,
                    y=theta_values,
                    mode='lines+markers',
                    name='θ Estimate',
                    line=dict(color='#2e7d32', width=3),
                    marker=dict(size=6)
                ))
                
                # Add cut score line if enabled
                if use_cut_score and cut_score is not None:
                    fig_theta.add_hline(
                        y=cut_score,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Cut Score = {cut_score:.2f}",
                        annotation_position="right"
                    )
                
                fig_theta.update_layout(
                    title="Real-time Ability Estimate with Confidence Interval",
                    xaxis_title="Number of Items Answered",
                    yaxis_title="θ (Ability Estimate)",
                    height=350,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_theta, use_container_width=True)
        
        # Content Coverage Chart
        with st.expander("📊 Content Coverage: Actual vs Target", expanded=True):
            coverage_status = st.session_state.live_content_balancer.get_coverage_status()
            
            # Chart visualization only
            if len(st.session_state.live_administered_items) > 1:
                fig_coverage = plot_content_coverage(coverage_status)
                st.plotly_chart(fig_coverage, use_container_width=True)
        
        st.divider()
        
        # Display current item
        current_item = st.session_state.live_current_item
        
        st.markdown(f"### Item #{len(st.session_state.live_administered_items)} &nbsp;&nbsp; 🔑 **Answer: {current_item['key']}** &nbsp; | &nbsp; 📊 **b = {current_item['rasch_b']:.2f}** *(Demo Mode)*")
        st.markdown(f"**Domain:** {current_item['domain']}")
        
        # Display stem
        st.markdown("#### Question")
        st.markdown(f"<div class='item-display'>{current_item['stem']}</div>", unsafe_allow_html=True)
        
        # Display choices
        st.markdown("#### Select your answer:")
        
        choices = {
            'A': current_item['choice_A'],
            'B': current_item['choice_B'],
            'C': current_item['choice_C'],
            'D': current_item['choice_D']
        }
        
        # Use radio buttons for answer selection
        selected_answer = st.radio(
            "Choose one:",
            options=['A', 'B', 'C', 'D'],
            format_func=lambda x: f"**{x}.** {choices[x]}",
            key=f"answer_{len(st.session_state.live_administered_items)}"
        )
        
        # Submit button
        if st.button("Submit Answer ➡️", type="primary"):
            # Check if correct
            is_correct = (selected_answer == current_item['key'])
            response = 1 if is_correct else 0
            
            # Show immediate feedback
            if is_correct:
                st.success(f"✅ **Correct!** The answer is **{current_item['key']}**")
            else:
                st.error(f"❌ **Incorrect.** You selected **{selected_answer}**, but the correct answer is **{current_item['key']}**")
            
            # Small delay to show feedback (time module already imported at top)
            time.sleep(1.5)
            
            # Record response
            st.session_state.live_responses.append((current_item['rasch_b'], response))
            
            # Update trackers
            st.session_state.live_exposure_control.record_usage(current_item['item_id'])
            st.session_state.live_content_balancer.update(current_item['domain'])
            
            # Store item info
            st.session_state.live_session_data['items'].append({
                'item_id': current_item['item_id'],
                'domain': current_item['domain'],
                'difficulty': current_item['rasch_b'],
                'response': response
            })
            
            # Estimate theta
            theta_current, sem_current = mle_with_fence(st.session_state.live_responses)
            st.session_state.live_theta_current = theta_current
            st.session_state.live_theta_history.append(theta_current)
            st.session_state.live_sem_history.append(sem_current)
            
            st.session_state.live_session_data['thetas'].append(theta_current)
            st.session_state.live_session_data['sems'].append(sem_current)
            st.session_state.live_session_data['responses'].append(response)
            
            # Check stopping rules
            elapsed_time = time.time() - st.session_state.live_start_time
            should_stop, reason = st.session_state.live_stopping_rules.check_stopping(
                len(st.session_state.live_administered_items), sem_current, elapsed_time, theta_current
            )
            
            if should_stop:
                # End test automatically
                st.session_state.live_cat_active = False
                results = {
                    'final_theta': theta_current,
                    'final_sem': sem_current,
                    'true_theta': 0.0,
                    'n_items': len(st.session_state.live_administered_items),
                    'administered_items': st.session_state.live_administered_items,
                    'responses': st.session_state.live_responses,
                    'theta_history': st.session_state.live_theta_history,
                    'sem_history': st.session_state.live_sem_history,
                    'information_history': [],
                    'stopping_reason': reason,
                    'elapsed_time': elapsed_time,
                    'domain_distribution': st.session_state.live_content_balancer.get_coverage_status(),
                    'session_data': st.session_state.live_session_data,
                    'exposure_control': st.session_state.live_exposure_control,
                    'pool_size': len(st.session_state.live_item_pool)
                }
                st.session_state.cat_results = results
                st.success(f"✅ Test completed! {reason}")
                st.rerun()
            else:
                # Select next item
                available = st.session_state.live_item_pool[
                    ~st.session_state.live_item_pool['item_id'].isin(st.session_state.live_administered_items)
                ]
                
                if len(available) == 0:
                    st.error("No more items available!")
                    st.session_state.live_cat_active = False
                else:
                    next_item = select_next_item(
                        theta_current, 
                        available, 
                        st.session_state.live_administered_items,
                        st.session_state.live_content_balancer, 
                        st.session_state.live_exposure_control,
                        st.session_state.live_enemy_index
                    )
                    st.session_state.live_current_item = next_item
                    st.session_state.live_administered_items.append(next_item['item_id'])
                    st.rerun()
        
        st.divider()
    
    # Display results (simulation or completed live test)
    if st.session_state.cat_results and not st.session_state.get('live_cat_active', False):
        results = st.session_state.cat_results
        
        st.divider()
        
        # Summary metrics
        st.header("📊 CAT Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final θ Estimate", f"{results['final_theta']:.3f}")
        
        with col2:
            st.metric("Final SEM", f"{results['final_sem']:.3f}")
        
        with col3:
            st.metric("Items Administered", results['n_items'])
        
        with col4:
            error = abs(results['final_theta'] - results['true_theta'])
            st.metric("Estimation Error", f"{error:.3f}")
        
        st.info(f"**Stopping Reason:** {results['stopping_reason']}")
        
        # Convergence plots
        st.subheader("📈 Convergence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_theta = plot_theta_convergence(results['theta_history'], results['true_theta'])
            st.plotly_chart(fig_theta, use_container_width=True)
        
        with col2:
            fig_sem = plot_sem_reduction(results['sem_history'], sem_threshold)
            st.plotly_chart(fig_sem, use_container_width=True)
        
        # Content coverage
        st.subheader("🎯 Content Coverage")
        
        coverage_data = []
        for domain, stats in results['domain_distribution'].items():
            coverage_data.append({
                'Domain': domain,
                'Target Count': stats['target_count'],
                'Actual Count': stats['count'],
                'Target %': f"{stats['target']:.1%}",
                'Actual %': f"{stats['actual']:.1%}",
                'Difference': f"{stats['difference']:+.1%}"
            })
        
        coverage_df = pd.DataFrame(coverage_data)
        st.dataframe(coverage_df, use_container_width=True)
        
        # Pool utilization metrics
        if 'exposure_control' in results and 'pool_size' in results:
            st.subheader("🔄 Pool Utilization & Exposure Balance")
            
            exposure_ctrl = results['exposure_control']
            pool_metrics = exposure_ctrl.get_pool_utilization_metrics(results['pool_size'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Pool Utilization",
                    f"{pool_metrics['utilization_rate']:.1%}",
                    help="Percentage of item pool used across all tests"
                )
            
            with col2:
                st.metric(
                    "Items Used",
                    f"{pool_metrics['items_used']}/{pool_metrics['items_available']}",
                    help="Number of unique items used"
                )
            
            with col3:
                st.metric(
                    "Avg Usage",
                    f"{pool_metrics['avg_usage']:.1f}",
                    help="Average times each item has been used"
                )
            
            with col4:
                st.metric(
                    "Balance Index",
                    f"{pool_metrics['balance_index']:.2f}",
                    help="1.0 = perfect balance, 0.0 = highly unbalanced"
                )
            
            # Usage range
            st.caption(
                f"Usage range: {pool_metrics['min_usage']} - {pool_metrics['max_usage']} times | "
                f"Total tests administered: {exposure_ctrl.total_tests}"
            )
        
        # Item-by-item detail
        with st.expander("📋 Item-by-Item Details"):
            item_details = []
            for i, item_data in enumerate(results['session_data']['items'][1:], 1):
                item_details.append({
                    'Item': i,
                    'Item ID': item_data['item_id'],
                    'Domain': item_data['domain'],
                    'Difficulty (b)': f"{item_data['difficulty']:.2f}",
                    'Response': '✓' if item_data['response'] == 1 else '✗',
                    'θ After': f"{results['theta_history'][i]:.3f}",
                    'SEM After': f"{results['sem_history'][i]:.3f}"
                })
            
            detail_df = pd.DataFrame(item_details)
            st.dataframe(detail_df, use_container_width=True)
    
    agent.disconnect()

if __name__ == "__main__":
    main()
