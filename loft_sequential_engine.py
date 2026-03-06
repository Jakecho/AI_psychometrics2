"""
loft_sequential_engine — Sequential LOFT Engine
===================================================================
Sequential Linear-on-the-Fly Testing using a local Ollama LLM (qwen3:8b).

Instead of building all forms simultaneously, this engine assembles forms
one at a time with:
  - Item exposure tracking (global + domain-specific thresholds)
  - Domain-stratified active pool sub-sampling
  - Auto-generated difficulty category constraints (7-bin normal allocation)
  - Jaccard overlap rejection (ensures form distinctness)
  - Real-time generator yield for live Streamlit dashboard

Based on optimization strategies from:
  Cho, J. (2025). Optimizing LOFT Test Assembly: Strategies for
  Exposure and Form Diversity. Credentialing Insights.

Prerequisites:
  pip install ollama pandas numpy
  ollama pull qwen3:8b

Author: AI Assistant
Date: March 3, 2026
"""

import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional, Generator, Set, Tuple
from scipy import stats as scipy_stats

# ==================== IRT Helpers ====================

D = 1.0

def rasch_probability(theta: float, b: float) -> float:
    """Rasch model probability of correct response."""
    return 1.0 / (1.0 + np.exp(-D * (theta - b)))

def rasch_information(theta: float, b: float) -> float:
    """Item information at theta under Rasch model."""
    p = rasch_probability(theta, b)
    return (D ** 2) * p * (1 - p)

def calculate_tif(theta: float, b_params: np.ndarray) -> float:
    """Test Information Function at theta."""
    return sum(rasch_information(theta, b) for b in b_params)

def calculate_tcc(theta: float, b_params: np.ndarray) -> float:
    """Test Characteristic Curve (expected score) at theta."""
    return sum(rasch_probability(theta, b) for b in b_params)


# ==================== Item Usage Tracker ====================

OLLAMA_MODEL = "qwen3:8b"


class ItemUsageTracker:
    """
    Tracks item exposure globally and by domain.

    Strategy 1 (Cho, 2025): Domain-specific thresholds are critical
    for domains with fewer items — they need higher reuse limits to
    avoid premature pool exhaustion, while large domains can use
    tighter limits to spread exposure more evenly.
    """

    def __init__(self, item_pool: pd.DataFrame, global_max: int = 2,
                 domain_max: Dict[str, int] = None,
                 auto_domain_limits: bool = True,
                 n_forms: int = 1, test_length: int = 10):
        self.item_pool = item_pool
        self.usage_count = {str(item_id): 0 for item_id in item_pool['item_id']}
        self.global_max_usage = global_max

        # Auto-compute domain-specific limits if not provided
        if domain_max:
            self.domain_max_usage = domain_max
        elif auto_domain_limits and 'domain' in item_pool.columns:
            self.domain_max_usage = self._auto_compute_domain_limits(
                n_forms, test_length
            )
        else:
            self.domain_max_usage = {}

    def _auto_compute_domain_limits(self, n_forms: int, test_length: int) -> Dict[str, int]:
        """
        Auto-compute domain-specific exposure limits based on pool depth.

        Logic: For each domain, estimate how many items are needed across
        all forms (n_forms × domain_proportion × test_length). If the
        domain has fewer items than needed, raise the exposure limit.
        Domains with deeper pools keep the global limit.
        """
        domain_counts = self.item_pool['domain'].value_counts().to_dict()
        total_items = len(self.item_pool)
        domain_limits = {}

        for domain, count in domain_counts.items():
            # Estimate demand: proportional share of test length × n_forms
            proportion = count / total_items
            estimated_demand = int(np.ceil(proportion * test_length * n_forms))

            if count == 0:
                domain_limits[domain] = self.global_max_usage
            else:
                # Minimum exposure needed = demand / supply
                min_exposure = max(1, int(np.ceil(estimated_demand / count)))
                # Use the higher of global limit or computed minimum
                domain_limits[domain] = max(self.global_max_usage, min_exposure)

        return domain_limits

    def record_usage(self, selected_items: List[str]):
        """Increment usage for items placed on a valid form."""
        for item in selected_items:
            key = str(item)
            if key in self.usage_count:
                self.usage_count[key] += 1

    def get_eligible_pool(self) -> pd.DataFrame:
        """Returns a dataframe of items that have not exceeded exposure limits."""
        eligible_items = []
        for idx, row in self.item_pool.iterrows():
            item_id = str(row['item_id'])
            domain = str(row.get('domain', 'Unspecified'))

            # Check global threshold
            if self.usage_count.get(item_id, 0) >= self.global_max_usage:
                continue

            # Check domain threshold (if specified, otherwise fallback to global)
            domain_limit = self.domain_max_usage.get(domain, self.global_max_usage)
            if self.usage_count.get(item_id, 0) >= domain_limit:
                continue

            eligible_items.append(row)

        return pd.DataFrame(eligible_items)

    def get_exposure_stats(self) -> Dict[str, Any]:
        """Return exposure statistics for reporting."""
        used = {k: v for k, v in self.usage_count.items() if v > 0}
        return {
            'total_used': len(used),
            'total_pool': len(self.usage_count),
            'utilization_pct': round(len(used) / len(self.usage_count) * 100, 1) if self.usage_count else 0,
            'max_exposure': max(self.usage_count.values()) if self.usage_count else 0,
            'mean_exposure': round(np.mean([v for v in self.usage_count.values() if v > 0]), 2) if used else 0,
            'domain_limits': dict(self.domain_max_usage),
        }


# ==================== Difficulty Category Auto-Generation ====================

# Default 7-bin centers (standard normal quantile-based)
DEFAULT_DIFFICULTY_CENTERS = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]

def auto_generate_difficulty_categories(
    item_pool: pd.DataFrame,
    test_length: int,
    n_categories: int = 7,
    b_col: str = 'rasch_b',
) -> Dict[str, Dict[str, Any]]:
    """
    Strategy 3 (Cho, 2025): Auto-generate difficulty bin constraints to
    maximize item pool utilization.

    Divides the item difficulty range into n_categories bins centered on
    standard normal density quantiles. Item allocation per bin is based
    on the normalized standard normal density at each center.

    For a 100-item test with 7 categories centered at
    [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], yields approximately:
    1-3 items from ±2.5, 7-9 from ±1.5, 23-25 from ±0.5, 27-29 from 0.

    Returns:
        Dict mapping category label to:
            {'center': float, 'range': (lo, hi), 'min': int, 'max': int,
             'pool_count': int, 'density_weight': float}
    """
    b_values = item_pool[b_col].dropna().values
    if len(b_values) == 0:
        return {}

    b_min, b_max = float(np.min(b_values)), float(np.max(b_values))

    # Define bin centers based on standard normal quantiles
    if n_categories == 7:
        centers = DEFAULT_DIFFICULTY_CENTERS
    else:
        # Evenly spaced from -2.5 to 2.5
        centers = np.linspace(-2.5, 2.5, n_categories).tolist()

    # Compute bin edges (midpoints between centers, plus extremes)
    edges = []
    edges.append(min(b_min - 0.5, centers[0] - 0.5))
    for i in range(len(centers) - 1):
        edges.append((centers[i] + centers[i + 1]) / 2.0)
    edges.append(max(b_max + 0.5, centers[-1] + 0.5))

    # Compute allocation weights from standard normal density at centers
    densities = [float(scipy_stats.norm.pdf(c)) for c in centers]
    total_density = sum(densities)
    weights = [d / total_density for d in densities]

    categories = {}
    for i, (center, weight) in enumerate(zip(centers, weights)):
        lo, hi = edges[i], edges[i + 1]

        # Count items in this bin
        pool_count = int(np.sum((b_values >= lo) & (b_values < hi)))
        # Handle last bin inclusively
        if i == len(centers) - 1:
            pool_count = int(np.sum((b_values >= lo) & (b_values <= hi)))

        # Target allocation
        target = weight * test_length
        # Use a range: floor-1 to ceil+1, but at least 0
        cat_min = max(0, int(np.floor(target) - 1))
        cat_max = min(pool_count, int(np.ceil(target) + 1))

        # Don't impose constraint if pool has 0 items in this range
        if pool_count == 0:
            cat_min = 0
            cat_max = 0

        label = f"B{i+1} [{lo:.1f}, {hi:.1f})"
        categories[label] = {
            'center': center,
            'range': (lo, hi),
            'min': cat_min,
            'max': cat_max,
            'target': round(target, 1),
            'pool_count': pool_count,
            'density_weight': round(weight, 4),
        }

    return categories


def compute_difficulty_bin(b_value: float, categories: Dict) -> Optional[str]:
    """Find which difficulty category a b-value belongs to."""
    for label, cat_info in categories.items():
        lo, hi = cat_info['range']
        if lo <= b_value < hi:
            return label
        # Last bin is inclusive on the right
    # Check last bin inclusively
    last_label = list(categories.keys())[-1]
    last_info = categories[last_label]
    if b_value >= last_info['range'][0]:
        return last_label
    return None


# ==================== Domain-Stratified Active Pool Generation ====================

def generate_active_pool(eligible_pool: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
    """
    Strategy 2 (Cho, 2025): Domain-stratified active pool sampling.

    Instead of purely random sampling, sample proportionally from each domain
    so that domain constraints remain feasible in the active pool. The multiplier
    is applied per-domain to determine how many items to draw from each.

    Steps:
      1. Force common items into the pool
      2. For each domain, sample min(available, multiplier × domain_target) items
      3. Fill remaining slots randomly from leftover items
    """
    test_length = rules.get('test_length', 10)
    multiplier = rules.get('multiplier', 5)
    target_size = test_length * multiplier

    # 1. Force common items into the pool
    common_items = rules.get('common_items', [])
    forced_items_df = eligible_pool[
        eligible_pool['item_id'].astype(str).isin([str(x) for x in common_items])
    ]
    forced_ids = set(forced_items_df['item_id'].astype(str).tolist())

    remaining_pool = eligible_pool[
        ~eligible_pool['item_id'].astype(str).isin(forced_ids)
    ]

    # 2. Domain-stratified sampling
    domain_constraints = rules.get('domain_constraints', {})
    sampled_parts = []

    if 'domain' in remaining_pool.columns and domain_constraints:
        domains_sampled = set()

        for domain, limits in domain_constraints.items():
            domain_items = remaining_pool[remaining_pool['domain'] == domain]
            # Target: multiplier × max(domain_min, domain_max) items
            domain_target = limits.get('max', limits.get('min', 0))
            if domain_target == 0:
                domain_target = limits.get('min', 0)
            if domain_target == 0:
                # No explicit constraint — use proportional allocation
                total_pool = len(remaining_pool)
                if total_pool > 0:
                    prop = len(domain_items) / total_pool
                    domain_target = max(1, int(np.ceil(prop * test_length)))

            n_sample = min(len(domain_items), multiplier * domain_target)
            if n_sample > 0 and len(domain_items) > 0:
                sampled = domain_items.sample(n=min(n_sample, len(domain_items)))
                sampled_parts.append(sampled)
                domains_sampled.update(sampled['item_id'].astype(str).tolist())

        # 3. Sample remaining items from unconstrained domains
        unconstrained = remaining_pool[
            ~remaining_pool['item_id'].astype(str).isin(domains_sampled)
        ]
        remaining_slots = target_size - len(forced_ids) - sum(len(p) for p in sampled_parts)
        if remaining_slots > 0 and len(unconstrained) > 0:
            n_extra = min(len(unconstrained), remaining_slots)
            sampled_parts.append(unconstrained.sample(n=n_extra))

    else:
        # No domain constraints — fall back to simple random sampling
        needed_size = target_size - len(forced_ids)
        if len(remaining_pool) <= needed_size or needed_size <= 0:
            sampled_parts.append(remaining_pool)
        else:
            sampled_parts.append(remaining_pool.sample(n=needed_size))

    # Combine
    all_parts = [forced_items_df] + sampled_parts
    final_pool = pd.concat(all_parts).drop_duplicates(subset='item_id').reset_index(drop=True)
    return final_pool


# ==================== Prompt Builder ====================

def build_loft_prompt(
    active_pool: pd.DataFrame,
    rules: Dict[str, Any],
    form_index: int = 1,
    difficulty_categories: Dict = None,
) -> str:
    """
    Build a structured prompt for the Ollama LLM to perform item selection
    from the active sub-pool for a single LOFT form.

    Includes difficulty category constraints (Strategy 3) when provided.
    """
    test_length = rules['test_length']
    domain_constraints = rules.get('domain_constraints', {})
    raschb_cat_constraints = rules.get('raschb_cat_constraints', {})
    image_constraint = rules.get('image_constraint', {})
    audio_constraint = rules.get('audio_constraint', {})
    excluded_items = rules.get('excluded_items', [])
    common_items = rules.get('common_items', [])
    apply_enemies = rules.get('apply_enemies', False)

    # IRT targets
    theta_targets = rules.get('theta_targets', [])
    min_tif_targets = rules.get('min_tif_targets', [])
    tcc_targets = rules.get('tcc_targets', [])
    tif_tolerances = rules.get('tif_tolerances', [])
    tcc_tolerances = rules.get('tcc_tolerances', [])

    mean_diff_target = rules.get('mean_difficulty_target', None)
    mean_diff_tol = rules.get('mean_difficulty_tolerance', 0.2)

    # Build constraint description
    constraint_lines = []
    constraint_lines.append(f"1. Select EXACTLY {test_length} items (item_ids).")

    # Domain constraints
    if domain_constraints:
        dc_parts = []
        for dom, limits in domain_constraints.items():
            mn = limits.get('min', 0)
            mx = limits.get('max', 0)
            if mn > 0 or mx > 0:
                dc_parts.append(f"  - {dom}: min={mn}, max={mx}")
        if dc_parts:
            constraint_lines.append("2. Domain distribution constraints:")
            constraint_lines.extend(dc_parts)

    # Rasch B category constraints (user-specified)
    if raschb_cat_constraints:
        rc_parts = []
        for cat, limits in raschb_cat_constraints.items():
            mn = limits.get('min', 0)
            mx = limits.get('max', 0)
            if mn > 0 or mx > 0:
                rc_parts.append(f"  - {cat}: min={mn}, max={mx}")
        if rc_parts:
            constraint_lines.append("3. Rasch B category constraints:")
            constraint_lines.extend(rc_parts)

    # Auto-generated difficulty category constraints (Strategy 3)
    if difficulty_categories:
        dc_parts = []
        for label, info in difficulty_categories.items():
            if info['pool_count'] > 0 and (info['min'] > 0 or info['max'] > 0):
                lo, hi = info['range']
                dc_parts.append(
                    f"  - Difficulty bin {label}: select {info['min']}-{info['max']} "
                    f"items with rasch_b in [{lo:.2f}, {hi:.2f})"
                )
        if dc_parts:
            constraint_lines.append(
                "3b. Difficulty distribution (auto-generated to maximize pool utilization):"
            )
            constraint_lines.extend(dc_parts)
            constraint_lines.append(
                "     NOTE: These ranges ensure items are drawn from across the "
                "difficulty spectrum, not just near the evaluation point."
            )

    # Image constraint
    if image_constraint and 'has_image' in active_pool.columns:
        img_min = image_constraint.get('min', 0)
        img_max = image_constraint.get('max', 0)
        if img_min > 0 or img_max > 0:
            constraint_lines.append(f"4. Image items: min={img_min}, max={img_max}")

    # Audio constraint
    if audio_constraint and 'has_audio' in active_pool.columns:
        aud_min = audio_constraint.get('min', 0)
        aud_max = audio_constraint.get('max', 0)
        if aud_min > 0 or aud_max > 0:
            constraint_lines.append(f"4b. Audio items: min={aud_min}, max={aud_max}")

    # Common items
    if common_items:
        constraint_lines.append(f"5. MUST include these common items: {common_items}")

    # Excluded items
    if excluded_items:
        constraint_lines.append(f"6. MUST NOT include these excluded items: {excluded_items}")

    # Enemy items
    if apply_enemies and 'enemy_ids' in active_pool.columns:
        constraint_lines.append(
            "7. Enemy constraint: No two items that are enemies of each other "
            "may appear together. The 'enemy_ids' column lists each item's enemies "
            "(comma-separated)."
        )

    # IRT targets
    if theta_targets and min_tif_targets:
        irt_lines = []
        for i, (theta_val, tif_t) in enumerate(zip(theta_targets, min_tif_targets)):
            if theta_val is not None and tif_t is not None and tif_t > 0:
                parts = [f"θ={theta_val:.3f}", f"TIF target≈{tif_t:.2f}"]
                # Add TCC target if available
                if tcc_targets and i < len(tcc_targets) and tcc_targets[i] is not None and tcc_targets[i] > 0:
                    parts.append(f"TCC target≈{tcc_targets[i]:.2f}")
                label = ["Low", "Mid (logit cut)", "High"][i] if i < 3 else f"Point {i+1}"
                irt_lines.append(f"  - {label}: {', '.join(parts)}")
        if irt_lines:
            constraint_lines.append("8. IRT targets (try to match closely):")
            constraint_lines.extend(irt_lines)

    # Mean difficulty target
    if mean_diff_target is not None:
        constraint_lines.append(
            f"9. Mean Rasch B difficulty target: {mean_diff_target} ± {mean_diff_tol}"
        )

    constraints_text = "\n".join(constraint_lines)

    # Determine which columns to include (keep it compact)
    cols_to_include = ['item_id', 'domain', 'rasch_b', 'pvalue', 'point_biserial']
    if 'raschb_cat' in active_pool.columns and raschb_cat_constraints:
        cols_to_include.append('raschb_cat')
    if 'has_image' in active_pool.columns and image_constraint:
        cols_to_include.append('has_image')
    if 'has_audio' in active_pool.columns and audio_constraint:
        cols_to_include.append('has_audio')
    if 'enemy_ids' in active_pool.columns and apply_enemies:
        cols_to_include.append('enemy_ids')

    available_cols = [c for c in cols_to_include if c in active_pool.columns]
    bank_csv = active_pool[available_cols].to_csv(index=False)

    prompt = f"""You are an expert psychometrician performing Automated Test Assembly (ATA) for Linear-on-the-Fly Testing (LOFT).
Your task is to select an optimal set of items from the ACTIVE POOL below to build Form {form_index}.

CONSTRAINTS:
{constraints_text}

OPTIMIZATION GOAL:
- Maximize test information across the theta range, especially near the logit cut point.
- Satisfy ALL hard constraints above (test length, domain counts, enemy items, etc.).
- If IRT targets are given, select items whose combined Test Information Function closely matches those targets.
- IMPORTANT: Select items from DIVERSE difficulty levels across the full range, not just items near the cut score.

RESPONSE FORMAT:
Respond with ONLY a valid JSON array of exactly {test_length} item_id strings.
No markdown formatting, no explanation, no code fences.
Example: ["NCX0001", "NCX0015", "NCX0042", ...]

ACTIVE POOL ({len(active_pool)} items available):
{bank_csv}"""

    return prompt


# ==================== Validation ====================

def validate_llm_selection(
    selected_ids: List[str],
    items_df: pd.DataFrame,
    rules: Dict[str, Any],
    difficulty_categories: Dict = None,
) -> Dict[str, Any]:
    """
    Validate the LLM's item selection against all constraints.
    Includes difficulty category validation (Strategy 3).
    Returns a dict with 'valid' (bool) and 'issues' (list of strings).
    """
    issues = []
    test_length = rules['test_length']

    # 1. Count check
    if len(selected_ids) != test_length:
        issues.append(f"Expected {test_length} items, got {len(selected_ids)}")

    # 2. Existence check
    valid_ids = set(items_df['item_id'].astype(str))
    missing = [sid for sid in selected_ids if str(sid) not in valid_ids]
    if missing:
        issues.append(f"Item IDs not found in bank: {missing[:10]}")

    # Filter to valid selections for further checks
    selected_ids_valid = [sid for sid in selected_ids if str(sid) in valid_ids]
    selected_df = items_df[items_df['item_id'].astype(str).isin(
        [str(s) for s in selected_ids_valid]
    )]

    # 3. Excluded items check
    excluded = set(str(x) for x in rules.get('excluded_items', []))
    violations = [sid for sid in selected_ids_valid if str(sid) in excluded]
    if violations:
        issues.append(f"Excluded items selected: {violations}")

    # 4. Common items check
    common = set(str(x) for x in rules.get('common_items', []))
    missing_common = [cid for cid in common if cid not in [str(s) for s in selected_ids_valid]]
    if missing_common:
        issues.append(f"Required common items missing: {missing_common}")

    # 5. Domain constraints
    domain_constraints = rules.get('domain_constraints', {})
    if domain_constraints and 'domain' in selected_df.columns:
        domain_counts = selected_df['domain'].value_counts().to_dict()
        for dom, limits in domain_constraints.items():
            count = domain_counts.get(dom, 0)
            mn = limits.get('min', 0)
            mx = limits.get('max', 0)
            if mn > 0 and count < mn:
                issues.append(f"Domain '{dom}': got {count}, need min {mn}")
            if mx > 0 and count > mx:
                issues.append(f"Domain '{dom}': got {count}, exceeds max {mx}")

    # 6. Enemy check
    if rules.get('apply_enemies', False) and 'enemy_ids' in items_df.columns:
        selected_set = set(str(s) for s in selected_ids_valid)
        for _, row in selected_df.iterrows():
            enemy_str = str(row.get('enemy_ids', ''))
            if enemy_str and enemy_str.lower() != 'nan':
                for eid in enemy_str.split(','):
                    eid = eid.strip()
                    if eid and eid in selected_set and eid != str(row['item_id']):
                        issues.append(f"Enemy pair: {row['item_id']} ↔ {eid}")

    # 7. Difficulty category validation (Strategy 3)
    difficulty_dist = {}
    if difficulty_categories and len(selected_df) > 0 and 'rasch_b' in selected_df.columns:
        for label, cat_info in difficulty_categories.items():
            lo, hi = cat_info['range']
            # Count items in this bin
            in_bin = selected_df[
                (selected_df['rasch_b'] >= lo) & (selected_df['rasch_b'] < hi)
            ]
            # Last bin inclusive
            if label == list(difficulty_categories.keys())[-1]:
                in_bin = selected_df[selected_df['rasch_b'] >= lo]
            count = len(in_bin)
            difficulty_dist[label] = count

            if cat_info['min'] > 0 and count < cat_info['min']:
                issues.append(
                    f"Difficulty bin '{label}': got {count}, need min {cat_info['min']}"
                )
            if cat_info['max'] > 0 and count > cat_info['max']:
                issues.append(
                    f"Difficulty bin '{label}': got {count}, exceeds max {cat_info['max']}"
                )

    # Compute form metrics
    metrics = {}
    if len(selected_df) > 0:
        b_params = selected_df['rasch_b'].values
        theta_targets = rules.get('theta_targets', [0.0])
        primary_theta = theta_targets[1] if len(theta_targets) > 1 else (theta_targets[0] if theta_targets else 0.0)

        metrics = {
            'mean_b': float(selected_df['rasch_b'].mean()),
            'sd_b': float(selected_df['rasch_b'].std()),
            'primary_tif': float(calculate_tif(primary_theta, b_params)),
            'primary_tcc': float(calculate_tcc(primary_theta, b_params)),
            'mean_pvalue': float(selected_df['pvalue'].mean()) if 'pvalue' in selected_df.columns else None,
            'mean_pbs': float(selected_df['point_biserial'].mean()) if 'point_biserial' in selected_df.columns else None,
            'difficulty_distribution': difficulty_dist,
        }

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'selected_df': selected_df,
        'metrics': metrics,
    }


# ==================== Single Form Assembly via Ollama ====================

def assemble_single_form_ollama(
    active_pool: pd.DataFrame,
    rules: Dict[str, Any],
    form_index: int = 1,
    model: str = OLLAMA_MODEL,
    difficulty_categories: Dict = None,
) -> Dict[str, Any]:
    """
    Assemble a single test form from the active pool using Ollama LLM.

    Returns dict with:
        'status': 'Optimal' | 'Failed'
        'selected_items': List[str]
        'validation': validation result dict
        'metrics': dict of form quality metrics
        'raw_response': str
    """
    try:
        import ollama as _ollama
    except ImportError:
        return {
            'status': 'Failed',
            'error': 'ollama package not installed. Run: pip install ollama',
            'selected_items': [],
            'metrics': {},
        }

    # Build prompt from the active pool (with difficulty categories)
    prompt = build_loft_prompt(active_pool, rules, form_index, difficulty_categories)

    print(f"\n🤖 Calling Ollama ({model}) for LOFT Form {form_index}...")
    try:
        response = _ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 4096},
        )
        # Handle both dict and object API
        msg = response.message if hasattr(response, "message") else response["message"]
        content = (msg.content if hasattr(msg, "content") else msg["content"]).strip()
    except Exception as e:
        return {
            'status': 'Failed',
            'error': f'Ollama call failed: {e}',
            'selected_items': [],
            'metrics': {},
        }

    # --- Parse response ---
    raw_response = content

    # Strip thinking tags if present (qwen3 sometimes wraps in <think>...</think>)
    if '<think>' in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Strip markdown fences
    for fence in ("```json", "```"):
        if content.startswith(fence):
            content = content[len(fence):]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Try to find a JSON array in the content
    try:
        selected_ids = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                selected_ids = json.loads(match.group())
            except json.JSONDecodeError:
                return {
                    'status': 'Failed',
                    'error': 'Could not parse JSON from response',
                    'raw_response': raw_response,
                    'selected_items': [],
                    'metrics': {},
                }
        else:
            return {
                'status': 'Failed',
                'error': 'No JSON array found in response',
                'raw_response': raw_response,
                'selected_items': [],
                'metrics': {},
            }

    # Ensure all IDs are strings
    selected_ids = [str(x) for x in selected_ids]

    # Validate against the active pool
    validation = validate_llm_selection(
        selected_ids, active_pool, rules, difficulty_categories
    )

    print(f"  ✅ Ollama returned {len(selected_ids)} items. Valid: {validation['valid']}")
    if not validation['valid']:
        for issue in validation['issues']:
            print(f"  ⚠️  {issue}")

    status = 'Optimal' if validation['valid'] else 'Partial'

    return {
        'status': status,
        'selected_items': selected_ids,
        'validation': validation,
        'metrics': validation.get('metrics', {}),
        'raw_response': raw_response,
    }


# ==================== Audit Report ====================

def audit_assembly_ollama(
    forms: list,
    rules: Dict,
    usage_stats: Dict,
    bank_size: int,
    model: str = OLLAMA_MODEL,
    domain_limits: Dict = None,
) -> str:
    """
    Generate a post-assembly quality report.
    Includes domain-level exposure analysis and difficulty distribution metrics.
    Uses Ollama LLM if available, otherwise falls back to rule-based report.
    """
    summary = {
        'forms_built': len(forms),
        'forms_requested': rules.get('n_forms', '?'),
        'test_length': rules.get('test_length', '?'),
        'form_metrics': [f.get('metrics', {}) for f in forms],
        'unique_items_used': len([k for k, v in usage_stats.items() if v > 0]),
        'bank_size': bank_size,
        'max_exposure': max(usage_stats.values()) if usage_stats else 0,
        'bank_utilization_pct': round(
            len([k for k, v in usage_stats.items() if v > 0]) / bank_size * 100, 1
        ) if bank_size else 0
    }

    # Rule-based audit (fast, no extra LLM call)
    lines = [f"**Assembly Complete**: {summary['forms_built']}/{summary['forms_requested']} forms built."]
    lines.append(f"**Bank Utilization**: {summary['bank_utilization_pct']}% ({summary['unique_items_used']}/{summary['bank_size']} items drawn).")
    lines.append(f"**Max Exposure**: {summary['max_exposure']} (limit: {rules.get('exposure_global_max', '?')}).")

    # Domain-specific exposure info
    if domain_limits:
        dl_parts = [f"{d}: max={l}" for d, l in domain_limits.items()]
        lines.append(f"**Domain Exposure Limits**: {', '.join(dl_parts)}")

    mean_bs = [m.get('mean_b', 0) for m in summary['form_metrics'] if m]
    if mean_bs:
        spread = max(mean_bs) - min(mean_bs)
        lines.append(f"**Mean B Range**: {min(mean_bs):.2f} to {max(mean_bs):.2f} (spread: {spread:.2f}).")
        if spread > 0.5:
            lines.append("⚠️ Forms vary substantially in average difficulty. Consider tightening mean difficulty tolerance.")

    tifs = [m.get('primary_tif', 0) for m in summary['form_metrics'] if m]
    if tifs:
        lines.append(f"**TIF Range**: {min(tifs):.2f} to {max(tifs):.2f}.")

    # Difficulty distribution across forms
    diffs = [m.get('difficulty_distribution', {}) for m in summary['form_metrics'] if m]
    if diffs and any(diffs):
        lines.append("**Difficulty Bin Coverage**: ✅ Active across forms.")

    # Try LLM-based audit
    try:
        import ollama as _ollama
        prompt = f"""You are a psychometric quality auditor. Review this test assembly output and write a concise quality report (5-8 lines max, use markdown).

Assembly summary: {json.dumps(summary, default=str)}
Constraints used: {json.dumps(rules, default=str)}
Domain exposure limits: {json.dumps(domain_limits, default=str) if domain_limits else 'global only'}

Comment on: form parallelism (are forms similar in difficulty/information?), bank utilization efficiency, exposure control effectiveness, difficulty distribution balance, any concerns, and overall quality rating."""

        response = _ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 1024},
        )
        msg = response.message if hasattr(response, "message") else response["message"]
        llm_audit = (msg.content if hasattr(msg, "content") else msg["content"]).strip()

        # Strip thinking tags
        if '<think>' in llm_audit:
            llm_audit = re.sub(r'<think>.*?</think>', '', llm_audit, flags=re.DOTALL).strip()

        return llm_audit
    except Exception:
        return "\n".join(lines)


# ==================== Sequential LOFT Assembly (Generator) ====================

def sequential_loft_assembly(
    item_bank: pd.DataFrame,
    rules: Dict[str, Any],
    model: str = OLLAMA_MODEL,
) -> Generator[Dict[str, Any], None, None]:
    """
    Main Driver: Executes the sequential LOFT loop using Ollama LLM.

    Implements all three strategies from Cho (2025):
      1. Domain-specific exposure thresholds
      2. Domain-stratified active pool sampling
      3. Auto-generated difficulty category constraints

    Yields step dicts for real-time UI tracking:
      - {'step': 'sampling', 'form_idx': int, ...}
      - {'step': 'warning', 'message': str}
      - {'step': 'form_complete', 'form_idx': int, ...}
      - {'step': 'diagnostic', 'message': str, 'diagnosis': str}
      - {'step': 'error', 'message': str}
      - {'step': 'finished', 'forms': list, ...}
    """
    n_forms = rules.get('n_forms', 1)
    test_length = rules.get('test_length', 10)
    max_overlap = rules.get('max_overlap_threshold', 0.3)

    # Strategy 1: Initialize Tracker with domain-specific exposure limits
    domain_exposure_limits = rules.get('domain_exposure_limits', None)
    tracker = ItemUsageTracker(
        item_bank,
        global_max=rules.get('exposure_global_max', 2),
        domain_max=domain_exposure_limits,
        auto_domain_limits=rules.get('auto_domain_limits', True),
        n_forms=n_forms,
        test_length=test_length,
    )

    # Strategy 3: Auto-generate difficulty categories
    difficulty_categories = None
    if rules.get('auto_difficulty_bins', True):
        difficulty_categories = auto_generate_difficulty_categories(
            item_bank, test_length,
            n_categories=rules.get('n_difficulty_bins', 7),
        )
        if difficulty_categories:
            print(f"📊 Auto-generated {len(difficulty_categories)} difficulty bins:")
            for label, info in difficulty_categories.items():
                print(f"   {label}: target={info['target']}, "
                      f"range=[{info['min']}, {info['max']}], "
                      f"pool={info['pool_count']}")

    forms = []
    failed_attempts = 0
    max_retries = 3

    print(f"Starting Sequential LOFT Assembly for {n_forms} forms via Ollama ({model})...")
    print(f"  Domain exposure limits: {tracker.domain_max_usage}")

    # Sequential Form Loop
    form_idx = 0
    while form_idx < n_forms:
        # A. Filter exposed items
        eligible_pool = tracker.get_eligible_pool()
        if len(eligible_pool) < test_length:
            yield {
                'step': 'error',
                'message': f"Eligible pool exhausted! Only {len(eligible_pool)} items remain, need {test_length}."
            }
            break

        # B. Strategy 2: Domain-stratified active pool generation
        active_pool = generate_active_pool(eligible_pool, rules)

        # Compute domain breakdown for live stats
        domain_breakdown = {}
        if 'domain' in active_pool.columns:
            domain_breakdown = active_pool['domain'].value_counts().to_dict()

        # Yield pre-assembly progress with pool stats
        yield {
            'step': 'sampling',
            'form_idx': form_idx + 1,
            'forms': forms,
            'usage_stats': tracker.usage_count,
            'eligible_pool_size': len(eligible_pool),
            'eligible_pool_mean_b': float(eligible_pool['rasch_b'].mean()),
            'eligible_pool_sd_b': float(eligible_pool['rasch_b'].std()),
            'active_pool_size': len(active_pool),
            'active_pool_mean_b': float(active_pool['rasch_b'].mean()),
            'active_pool_sd_b': float(active_pool['rasch_b'].std()),
            'active_pool_domains': domain_breakdown,
            'domain_exposure_limits': dict(tracker.domain_max_usage),
            'difficulty_categories': difficulty_categories,
        }

        # C. Assemble via Ollama LLM (with difficulty constraints)
        result = assemble_single_form_ollama(
            active_pool=active_pool,
            rules=rules,
            form_index=form_idx + 1,
            model=model,
            difficulty_categories=difficulty_categories,
        )

        # D. Overlap Jaccard Check / Commit
        if result['status'] in ('Optimal', 'Partial'):
            new_set = set(result['selected_items'])

            # Form Similarity Calculation
            overlap_rejected = False
            for past_form in forms:
                past_set = set(past_form['selected_items'])
                intersection = len(new_set.intersection(past_set))
                union = len(new_set.union(past_set))
                jaccard = intersection / union if union > 0 else 0

                if jaccard > max_overlap:
                    overlap_rejected = True
                    break

            if overlap_rejected:
                failed_attempts += 1
                yield {
                    'step': 'warning',
                    'message': f"Form {form_idx + 1} rejected due to high similarity (Jaccard > {max_overlap:.2f}). Retrying..."
                }
                if failed_attempts >= max_retries:
                    yield {
                        'step': 'error',
                        'message': "Critical Failure: Unable to build form after max retries due to overlap."
                    }
                    break
                continue

            # Form passes all validation!
            tracker.record_usage(result['selected_items'])
            forms.append(result)
            form_idx += 1
            failed_attempts = 0

            # Yield post-assembly success
            yield {
                'step': 'form_complete',
                'form_idx': form_idx,
                'latest_form': result,
                'forms': forms,
                'usage_stats': tracker.usage_count,
                'eligible_pool_size': len(eligible_pool),
                'exposure_stats': tracker.get_exposure_stats(),
            }
        else:
            failed_attempts += 1
            # Yield diagnostic
            yield {
                'step': 'diagnostic',
                'message': f"Assembly Failed: {result.get('error', 'Unknown')}",
                'diagnosis': f"Ollama returned status '{result['status']}'. Error: {result.get('error', 'N/A')}. "
                             f"Active pool had {len(active_pool)} items. "
                             f"Try adjusting constraints or checking Ollama availability."
            }
            if failed_attempts >= max_retries:
                yield {
                    'step': 'error',
                    'message': "Critical Failure: Unable to build form after max retries."
                }
                break

    # Post-Assembly Audit
    audit_report = audit_assembly_ollama(
        forms, rules, tracker.usage_count, len(item_bank),
        model=model,
        domain_limits=tracker.domain_max_usage,
    )

    yield {
        'step': 'finished',
        'forms': forms,
        'usage_stats': dict(tracker.usage_count),
        'audit_report': audit_report,
        'exposure_stats': tracker.get_exposure_stats(),
        'difficulty_categories': difficulty_categories,
    }
