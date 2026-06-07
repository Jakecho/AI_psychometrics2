import pandas as pd
import numpy as np
from pulp import *
import random
from typing import List, Dict, Any, Tuple

# IRT Helper Functions
D = 1.0

def rasch_probability(theta: float, b: float) -> float:
    return 1.0 / (1.0 + np.exp(-D * (theta - b)))

def rasch_information(theta: float, b: float) -> float:
    p = rasch_probability(theta, b)
    return (D ** 2) * p * (1 - p)


class ItemUsageTracker:
    """Tracks item exposure globally and by domain."""
    def __init__(self, item_pool: pd.DataFrame, global_max: int = 2, domain_max: Dict[str, int] = None):
        self.item_pool = item_pool
        self.usage_count = {str(item_id): 0 for item_id in item_pool['item_id']}
        self.global_max_usage = global_max
        self.domain_max_usage = domain_max or {}
        
    def record_usage(self, selected_items: List[str]):
        """Increment usage for items placed on a valid form."""
        for item in selected_items:
            self.usage_count[str(item)] += 1
            
    def get_eligible_pool(self) -> pd.DataFrame:
        """Returns a dataframe of items that have not exceeded exposure limits."""
        eligible_items = []
        for idx, row in self.item_pool.iterrows():
            item_id = str(row['item_id'])
            domain = str(row.get('domain', 'Unspecified'))
            
            # Check global threshold
            if self.usage_count[item_id] >= self.global_max_usage:
                continue
                
            # Check domain threshold (if specified, otherwise fallback to global)
            domain_limit = self.domain_max_usage.get(domain, self.global_max_usage)
            if self.usage_count[item_id] >= domain_limit:
                continue
                
            eligible_items.append(row)
            
        return pd.DataFrame(eligible_items)


def generate_active_pool(eligible_pool: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
    """Randomly subsamples the eligible pool to minimize between-form similarity."""
    test_length = rules.get('test_length', 10)
    multiplier = rules.get('multiplier', 4)
    target_size = test_length * multiplier
    
    # 1. Force common items into the pool
    common_items = rules.get('common_items', [])
    forced_items_df = eligible_pool[eligible_pool['item_id'].astype(str).isin([str(x) for x in common_items])]
    forced_items = forced_items_df['item_id'].tolist()
    
    remaining_pool = eligible_pool[~eligible_pool['item_id'].astype(str).isin([str(x) for x in forced_items])]
    
    needed_size = target_size - len(forced_items)
    if len(remaining_pool) <= needed_size or needed_size <= 0:
        sampled_remaining = remaining_pool
    else:
        sampled_remaining = remaining_pool.sample(n=needed_size)
        
    final_pool = pd.concat([forced_items_df, sampled_remaining]).reset_index(drop=True)
    return final_pool


def assemble_single_form_mip(
    active_pool: pd.DataFrame, 
    rules: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assembles a single form using CBC optimization from the active pool.
    Incorporates advanced constraints (Testlets, Enemies, Mean Difficulty).
    """
    test_length = rules.get('test_length', 10)
    domain_constraints = rules.get('domain_constraints', {})
    theta_targets = rules.get('theta_targets', [0.0])
    min_tif_targets = rules.get('min_tif_targets', [2.0] * len(theta_targets))
    mean_diff_target = rules.get('mean_difficulty_target', 0.0)
    mean_diff_tol = rules.get('mean_difficulty_tolerance', 0.5)
    
    n_items = len(active_pool)
    if n_items < test_length:
        return {'status': 'Infeasible', 'error': 'Active pool too small'}

    prob = LpProblem("Single_Form_Assembly", LpMaximize)
    
    # Variables
    item_vars = [LpVariable(f"x_{i}", cat='Binary') for i in range(n_items)]
    
    # Objective: Maximize Information at the primary/middle theta target
    primary_theta = theta_targets[len(theta_targets)//2] if theta_targets else 0.0
    weights_primary = [rasch_information(primary_theta, row['rasch_b']) for _, row in active_pool.iterrows()]
    prob += lpSum(weights_primary[i] * item_vars[i] for i in range(n_items))
    
    # 1. Test Length
    prob += lpSum(item_vars) == test_length
    
    # 2. Domain Constraints
    for domain, limits in domain_constraints.items():
        indices = active_pool.index[active_pool['domain'] == domain].tolist()
        min_val = limits.get('min', 0)
        max_val = limits.get('max', test_length)
        if min_val > 0:
            prob += lpSum(item_vars[i] for i in indices) >= min_val
        if max_val < test_length:
            prob += lpSum(item_vars[i] for i in indices) <= max_val
            
    # 3. Multiple TIF Constraints + Tolerances
    tif_tolerances = rules.get('tif_tolerances', [])
    for idx, (t_val, tif_req) in enumerate(zip(theta_targets, min_tif_targets)):
        weights_t = [rasch_information(t_val, row['rasch_b']) for _, row in active_pool.iterrows()]
        prob += lpSum(weights_t[i] * item_vars[i] for i in range(n_items)) >= tif_req
        if tif_tolerances and idx < len(tif_tolerances) and tif_tolerances[idx] > 0:
            prob += lpSum(weights_t[i] * item_vars[i] for i in range(n_items)) <= tif_req + tif_tolerances[idx]

    # 4. TCC (Expected Score) Constraints
    tcc_targets = rules.get('tcc_targets', [])
    tcc_tolerances = rules.get('tcc_tolerances', [])
    for idx, t_val in enumerate(theta_targets):
        if idx < len(tcc_targets) and tcc_targets[idx] is not None and tcc_targets[idx] > 0:
            req = tcc_targets[idx]
            tol = tcc_tolerances[idx] if idx < len(tcc_tolerances) else 1.0
            p_vals = [rasch_probability(t_val, row['rasch_b']) for _, row in active_pool.iterrows()]
            prob += lpSum(p_vals[i] * item_vars[i] for i in range(n_items)) >= req - tol
            prob += lpSum(p_vals[i] * item_vars[i] for i in range(n_items)) <= req + tol

    # 5. Mean Difficulty Constraint
    b_values = active_pool['rasch_b'].values
    prob += lpSum(b_values[i] * item_vars[i] for i in range(n_items)) >= (mean_diff_target - mean_diff_tol) * test_length
    prob += lpSum(b_values[i] * item_vars[i] for i in range(n_items)) <= (mean_diff_target + mean_diff_tol) * test_length

    # 6. Formatting Constraints (Image / Audio)
    image_constraint = rules.get('image_constraint', {})
    if image_constraint and 'has_image' in active_pool.columns:
        indices = active_pool.index[active_pool['has_image'] == 1].tolist()
        if 'min' in image_constraint and image_constraint['min'] > 0:
            prob += lpSum(item_vars[i] for i in indices) >= image_constraint['min']
        if 'max' in image_constraint:
            prob += lpSum(item_vars[i] for i in indices) <= image_constraint['max']

    audio_constraint = rules.get('audio_constraint', {})
    if audio_constraint and 'has_audio' in active_pool.columns:
        indices = active_pool.index[active_pool['has_audio'] == 1].tolist()
        if 'min' in audio_constraint and audio_constraint['min'] > 0:
            prob += lpSum(item_vars[i] for i in indices) >= audio_constraint['min']
        if 'max' in audio_constraint:
            prob += lpSum(item_vars[i] for i in indices) <= audio_constraint['max']

    # 7. Rasch B Category Constraints
    raschb_cat_constraints = rules.get('raschb_cat_constraints', {})
    if raschb_cat_constraints and 'raschb_cat' in active_pool.columns:
        for cat, limits in raschb_cat_constraints.items():
            indices = active_pool.index[active_pool['raschb_cat'].astype(str) == str(cat)].tolist()
            if 'min' in limits and limits['min'] > 0:
                prob += lpSum(item_vars[i] for i in indices) >= limits['min']
            if 'max' in limits:
                prob += lpSum(item_vars[i] for i in indices) <= limits['max']

    # 8. Common Items (Force Inclusion)
    common_items = [str(x) for x in rules.get('common_items', [])]
    if common_items:
        for i, row in active_pool.iterrows():
            if str(row['item_id']) in common_items:
                prob += item_vars[i] == 1

    # 9. Enemy Items Logic (gated by checkbox)
    if rules.get('apply_enemies', True) and 'enemy_ids' in active_pool.columns:
        id_to_idx = {str(row['item_id']): i for i, row in active_pool.iterrows()}
        for i, row in active_pool.iterrows():
            enemies_str = str(row.get('enemy_ids', ''))
            if enemies_str and enemies_str.lower() != 'nan':
                enemies = [e.strip() for e in enemies_str.split(',')]
                for e_id in enemies:
                    if e_id in id_to_idx:
                        j = id_to_idx[e_id]
                        if i < j:
                            prob += item_vars[i] + item_vars[j] <= 1

    # 10. Testlet Constraint
    if 'testlet_id' in active_pool.columns:
        testlets = active_pool['testlet_id'].dropna().unique()
        for t_id in testlets:
            if str(t_id).lower() == 'nan': continue
            t_indices = active_pool.index[active_pool['testlet_id'] == t_id].tolist()
            if len(t_indices) > 0:
                y_t = LpVariable(f"y_testlet_{t_id}", cat='Binary')
                t_size = len(t_indices)
                prob += lpSum(item_vars[i] for i in t_indices) == t_size * y_t

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=60)
    prob.solve(solver)
    
    status = LpStatus.get(prob.status, 'Unknown')
    if status != 'Optimal':
        return {'status': status, 'error': 'Could not satisfy constraints on this active pool.'}
        
    selected_indices = [i for i in range(n_items) if value(item_vars[i]) == 1]
    selected_df = active_pool.iloc[selected_indices]
    
    # Validation Check
    actual_length = len(selected_df)
    primary_tif = sum(rasch_information(primary_theta, b) for b in selected_df['rasch_b'])
    
    return {
        'status': 'Optimal',
        'selected_items': selected_df['item_id'].tolist(),
        'metrics': {
            'length': actual_length,
            'primary_tif': primary_tif,
            'mean_b': selected_df['rasch_b'].mean()
        }
    }


from pydantic import BaseModel, Field
import json

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

class DomainConstraint(BaseModel):
    min: int = Field(description="Minimum number of items from this domain")
    max: int = Field(description="Maximum number of items from this domain")

class FormConstraints(BaseModel):
    n_forms: int = Field(description="Number of forms to assemble")
    test_length: int = Field(description="Number of items per form")
    multiplier: int = Field(default=4, description="Active pool size multiplier (e.g. 3, 4, 5)")
    domain_constraints: Dict[str, DomainConstraint] = Field(description="Dictionary of domain limits")
    theta_targets: List[float] = Field(description="List of theta evaluation points (e.g. [-1.0, 0.0, 1.0])")
    min_tif_targets: List[float] = Field(description="Minimum TIF desired at each corresponding theta point")
    tcc_targets: List[float] = Field(default=[], description="Expected Score TCC tagets at theta")
    tcc_tolerances: List[float] = Field(default=[], description="Tolerance around TCC targets")
    tif_tolerances: List[float] = Field(default=[], description="Tolerance bound above min_tif")
    mean_difficulty_target: float = Field(default=0.0, description="Target mean Rasch B difficulty for the form")
    mean_difficulty_tolerance: float = Field(default=0.5, description="Allowed deviation from the mean difficulty target")
    max_overlap_threshold: float = Field(default=0.3, description="Maximum Jaccard similarity fraction allowed between forms")
    exposure_global_max: int = Field(description="Maximum times any single item can be used globally")
    image_constraint: Dict[str, int] = Field(default={}, description="Dict with min/max for items with images")
    audio_constraint: Dict[str, int] = Field(default={}, description="Dict with min/max for items with audio")
    raschb_cat_constraints: Dict[str, Dict[str, int]] = Field(default={}, description="Dict mapping B categories to min/max")
    common_items: List[str] = Field(default=[], description="Exact Item IDs to force include in every form")

def semantic_ai_constraint_builder(
    user_prompt: str, 
    available_domains: List[str],
    llm_provider: str = 'gemini',
    api_key: str = None
) -> Dict[str, Any]:
    """
    Intelligent NLP-to-Math Translator.
    Supports either 'gemini' (google.generativeai) or 'openai' (Langchain).
    """
    default_rules = {
        "n_forms": 3,
        "test_length": 10,
        "multiplier": 4,
        "domain_constraints": {
            "Health Promotion & Maintenance": {"min": 2, "max": 4},
            "Management of Care": {"min": 2, "max": 4}
        },
        "theta_targets": [-1.0, 0.0, 1.0],
        "min_tif_targets": [1.5, 2.0, 1.5],
        "mean_difficulty_target": 0.5,
        "mean_difficulty_tolerance": 0.3,
        "max_overlap_threshold": 0.25,
        "exposure_global_max": 2
    }
    
    if not api_key:
        print("⚠️ No API key provided! Falling back to default psychometric JSON constraints.")
        return default_rules
        
    system_instruction = f"""
    You are a psychometric engine constraint builder. 
    Translate the user's request into a strict JSON payload.
    Available item block domains for constraints are: {', '.join(available_domains[:10])}
    
    For `theta_targets` make sure to establish at least 3 anchor points (usually the cutoff logit, one below, and one above).
    Your output must strictly match the expected Schema format.
    """
    
    try:
        if llm_provider.lower() == 'ollama':
            if ChatOllama is None:
                raise ImportError("langchain_ollama is not installed")
            llm = ChatOllama(model="qwen2.5:7b", temperature=0.0, format="json")
            prompt = f"{system_instruction}\\nOutput pure JSON matching the FormConstraints schema.\\nUSER REQUEST: {user_prompt}"
            resp = llm.invoke(prompt)
            parsed_rules = json.loads(resp.content)
            
        else:
             print(f"⚠️ Unknown provider '{llm_provider}'. Falling back to defaults.")
             return default_rules

        # Simple validation
        if "n_forms" in parsed_rules and "domain_constraints" in parsed_rules:
            return parsed_rules
            
    except Exception as e:
        print(f"⚠️ NLP Parser failed for {llm_provider}. Falling back to defaults. Error: {e}")
        
    return default_rules


def diagnose_infeasibility(rules: Dict, pool_stats: Dict, llm_provider: str, api_key: str) -> str:
    """LLM diagnoses why the MIP solver returned Infeasible. Diagnostic only — user decides."""
    if not api_key or llm_provider == 'Mock (No Key)':
        # Build a rule-based diagnostic instead
        msgs = []
        pool_size = pool_stats.get('pool_size', 0)
        test_len = rules.get('test_length', 10)
        if pool_size < test_len * 2:
            msgs.append(f"Active pool ({pool_size}) is very small relative to test length ({test_len}). Consider increasing the multiplier.")
        for dom, lim in rules.get('domain_constraints', {}).items():
            avail = pool_stats.get('domain_counts', {}).get(dom, 0)
            if lim.get('min', 0) > avail:
                msgs.append(f"Domain '{dom}' requires min {lim['min']} but only {avail} available in active pool.")
        if not msgs:
            msgs.append("Constraints may be collectively too tight for this pool sample. Try relaxing mean difficulty tolerance or domain minimums.")
        return " | ".join(msgs)

    prompt = f"""You are a psychometric test assembly diagnostic engine.
The MIP solver returned INFEASIBLE for this active pool.

Constraints requested: {json.dumps(rules, default=str)}
Active pool stats: {json.dumps(pool_stats, default=str)}

Analyze which constraint(s) are most likely causing infeasibility and suggest specific relaxations.
Be concise (2-3 sentences max). This is diagnostic only — the user will decide whether to act."""

    try:
        if llm_provider.lower() == 'ollama':
            if ChatOllama is None:
                raise ImportError("langchain_ollama is not installed")
            llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)
            resp = llm.invoke(prompt)
            return resp.content.strip()
    except Exception as e:
        return f"Diagnostic unavailable: {e}"
    return "Unknown provider."


def audit_assembly(forms: list, rules: Dict, usage_stats: Dict, bank_size: int, llm_provider: str, api_key: str) -> str:
    """LLM writes a post-assembly quality report summarizing the generated forms."""
    summary = {
        'forms_built': len(forms),
        'forms_requested': rules.get('n_forms', '?'),
        'test_length': rules.get('test_length', '?'),
        'form_metrics': [f.get('metrics', {}) for f in forms],
        'unique_items_used': len([k for k, v in usage_stats.items() if v > 0]),
        'bank_size': bank_size,
        'max_exposure': max(usage_stats.values()) if usage_stats else 0,
        'bank_utilization_pct': round(len([k for k, v in usage_stats.items() if v > 0]) / bank_size * 100, 1) if bank_size else 0
    }

    if not api_key or llm_provider == 'Mock (No Key)':
        # Rule-based audit
        lines = [f"**Assembly Complete**: {summary['forms_built']}/{summary['forms_requested']} forms built."]
        lines.append(f"**Bank Utilization**: {summary['bank_utilization_pct']}% ({summary['unique_items_used']}/{summary['bank_size']} items drawn).")
        lines.append(f"**Max Exposure**: {summary['max_exposure']} (limit: {rules.get('exposure_global_max', '?')}).")
        mean_bs = [m.get('mean_b', 0) for m in summary['form_metrics']]
        if mean_bs:
            spread = max(mean_bs) - min(mean_bs)
            lines.append(f"**Mean B Range**: {min(mean_bs):.2f} to {max(mean_bs):.2f} (spread: {spread:.2f}).")
            if spread > 0.5:
                lines.append("⚠️ Forms vary substantially in average difficulty. Consider tightening mean difficulty tolerance.")
        tifs = [m.get('primary_tif', 0) for m in summary['form_metrics']]
        if tifs:
            lines.append(f"**TIF Range**: {min(tifs):.2f} to {max(tifs):.2f}.")
        return "\n".join(lines)

    prompt = f"""You are a psychometric quality auditor. Review this test assembly output and write a concise quality report (5-8 lines max, use markdown).

Assembly summary: {json.dumps(summary, default=str)}
Constraints used: {json.dumps(rules, default=str)}

Comment on: form parallelism (are forms similar in difficulty/information?), bank utilization efficiency, any concerns, and overall quality rating."""

    try:
        if llm_provider.lower() == 'ollama':
            if ChatOllama is None:
                raise ImportError("langchain_ollama is not installed")
            llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
            resp = llm.invoke(prompt)
            return resp.content.strip()
    except Exception as e:
        return f"Audit unavailable: {e}"
    return "Unknown provider."


from typing import Generator

def sequential_loft_assembly(
    item_bank: pd.DataFrame, 
    user_prompt: str,
    llm_provider: str = 'gemini',
    api_key: str = None,
    advanced_settings: Dict[str, Any] = None
) -> Generator[Dict[str, Any], None, None]:
    """Main Driver: Executes the AI -> Tracker -> Active Pool -> MIP Loop as a Generator for real-time UI tracking."""
    
    available_domains = item_bank['domain'].unique().tolist()
    
    # 1. AI parses natural language into strict bounds
    rules = semantic_ai_constraint_builder(user_prompt, available_domains, llm_provider, api_key)
    if advanced_settings:
        rules.update(advanced_settings)
        
    print(f"\\n🧠 AI Extracted Constraints: {rules}")
    
    # 2. Initialize Tracker
    tracker = ItemUsageTracker(item_bank, global_max=rules['exposure_global_max'])
    
    forms = []
    failed_attempts = 0
    max_retries = 3
    
    print(f"Starting Sequential Assembly for {rules['n_forms']} forms...")
    
    # 3. Sequential Form Loop
    form_idx = 0
    while form_idx < rules['n_forms']:
        # A. Filter exposed items
        eligible_pool = tracker.get_eligible_pool()
        if len(eligible_pool) < rules['test_length']:
            yield {'step': 'error', 'message': "Eligible pool exhausted!"}
            break
            
        # B. Generate Active Randomized Sub-pool (LOFT logic)
        active_pool = generate_active_pool(eligible_pool, rules)
        
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
        }
        
        # C. Assemble via enhanced MIP Engine
        result = assemble_single_form_mip(
            active_pool=active_pool,
            rules=rules
        )
        
        # D. Overlap Jaccard Check / Commit
        if result['status'] == 'Optimal':
            new_set = set(result['selected_items'])
            
            # Form Similarity Calculation
            overlap_rejected = False
            for past_form in forms:
                past_set = set(past_form['selected_items'])
                intersection = len(new_set.intersection(past_set))
                union = len(new_set.union(past_set))
                jaccard = intersection / union if union > 0 else 0
                
                if jaccard > rules.get('max_overlap_threshold', 0.3):
                    overlap_rejected = True
                    break
            
            if overlap_rejected:
                failed_attempts += 1
                yield {'step': 'warning', 'message': f"Form rejected due to high similarity Jaccard index."}
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
                'eligible_pool_size': len(eligible_pool)
            }
        else:
            failed_attempts += 1
            # AI Diagnostic on infeasibility
            pool_stats = {
                'pool_size': len(active_pool),
                'domain_counts': active_pool['domain'].value_counts().to_dict() if 'domain' in active_pool.columns else {}
            }
            diagnosis = diagnose_infeasibility(rules, pool_stats, llm_provider, api_key)
            yield {
                'step': 'diagnostic',
                'message': f"Assembly Failed: {result.get('error', 'Unknown')}",
                'diagnosis': diagnosis
            }
            if failed_attempts >= max_retries:
                yield {'step': 'error', 'message': "Critical Failure: Unable to build form after max retries."}
                break
    
    # Post-Assembly AI Audit
    audit_report = audit_assembly(forms, rules, tracker.usage_count, len(item_bank), llm_provider, api_key)
    
    yield {
        'step': 'finished',
        'forms': forms,
        'usage_stats': dict(tracker.usage_count),
        'audit_report': audit_report
    }

