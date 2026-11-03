"""Engine for LOFT Sequential ATA V2 (CP-SAT): Rasch helpers, exposure, sampling and the sequential loop.

Lean replacement for the parts of loft_sequential_engine.py that the CP-SAT app uses (the Ollama/LLM code is
not included). Differences from V1:
- vectorized exposure filtering (no row loop) and Rasch functions;
- item bundles (bundle_id) are sampled, limited and excluded as whole units;
- a seeded random generator, so a run can be reproduced exactly;
- the sequential loop lives in SequentialAssembler, outside Streamlit, so it can be tested directly.

Strategies from Cho (2025), Optimizing LOFT Test Assembly: (1) domain-specific exposure limits,
(2) domain-stratified active pools, (3) auto-generated difficulty bins.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

D = 1.0


# ==================== Rasch helpers (vectorized) ====================

def rasch_probability(theta, b):
    return 1.0 / (1.0 + np.exp(-D * (np.asarray(theta, float) - np.asarray(b, float))))


def rasch_information(theta, b):
    p = rasch_probability(theta, b)
    return (D ** 2) * p * (1 - p)


def calculate_tif(theta, b_params):
    return float(np.sum(rasch_information(theta, np.asarray(b_params, float))))


def calculate_tcc(theta, b_params):
    return float(np.sum(rasch_probability(theta, np.asarray(b_params, float))))


def curves(theta_grid, b_params):
    """TIF and TCC over a θ grid in one pass (grid × items)."""
    p = rasch_probability(np.asarray(theta_grid, float)[:, None], np.asarray(b_params, float)[None, :])
    return (D ** 2 * p * (1 - p)).sum(1), p.sum(1)


# ==================== Item bundles ====================

def bundle_labels(pool: pd.DataFrame) -> pd.Series:
    """Bundle id per row ('' = standalone). Accepts bundle_id or the testlet_id alias."""
    column = 'bundle_id' if 'bundle_id' in pool.columns else ('testlet_id' if 'testlet_id' in pool.columns else None)
    if column is None:
        return pd.Series('', index=pool.index)
    labels = pool[column].astype(object).where(pool[column].notna(), '')
    return labels.astype(str).str.strip().replace({'nan': '', 'None': ''}).str.replace(r'^(\d+)\.0$', r'\1', regex=True)


def complete_bundles_only(pool: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Drop every bundle that is not fully present in `pool` (relative to the master bank)."""
    labels = bundle_labels(pool)
    if not labels.ne('').any():
        return pool
    sizes = bundle_labels(master).value_counts()
    present = labels[labels != ''].value_counts()
    incomplete = present.index[present.to_numpy() < sizes.reindex(present.index).to_numpy()]
    return pool[~labels.isin(incomplete)]


# ==================== Exposure ====================

class ItemUsageTracker:
    """Item exposure with domain-specific limits (Strategy 1); bundles leave the pool as a unit.

    `exempt` item ids (common/anchor items, which every form must contain) have no exposure limit; the rest
    of a bundle that contains an exempt item is exempt too.
    """

    def __init__(self, item_pool: pd.DataFrame, global_max: int = 2, domain_max: Dict[str, int] = None,
                 auto_domain_limits: bool = True, n_forms: int = 1, test_length: int = 10, exempt=()):
        self.item_pool = item_pool
        self.ids = item_pool['item_id'].astype(str).to_numpy()
        self.counts = np.zeros(len(item_pool), dtype=np.int64)
        self.position = {item_id: i for i, item_id in enumerate(self.ids)}
        self.global_max_usage = global_max
        if domain_max:
            self.domain_max_usage = dict(domain_max)
        elif auto_domain_limits and 'domain' in item_pool.columns:
            self.domain_max_usage = self._auto_compute_domain_limits(n_forms, test_length)
        else:
            self.domain_max_usage = {}
        domains = item_pool['domain'].astype(str) if 'domain' in item_pool.columns else pd.Series('', index=item_pool.index)
        self.limits = domains.map(lambda d: self.domain_max_usage.get(d, global_max)).to_numpy()
        labels = bundle_labels(item_pool)
        exempt_ids = {str(x) for x in exempt}
        exempt_bundles = set(labels[pd.Series(self.ids, index=item_pool.index).isin(exempt_ids) & labels.ne('')])
        self.exempt = np.isin(self.ids, list(exempt_ids)) | labels.isin(exempt_bundles).to_numpy()

    def _auto_compute_domain_limits(self, n_forms, test_length):
        counts = self.item_pool['domain'].value_counts()
        total = len(self.item_pool)
        out = {}
        for domain, count in counts.items():
            demand = int(np.ceil(count / total * test_length * n_forms))
            out[domain] = max(self.global_max_usage, max(1, int(np.ceil(demand / count))))
        return out

    @property
    def usage_count(self):
        return dict(zip(self.ids, self.counts.tolist()))

    def record_usage(self, selected_items: List[str]):
        for item in selected_items:
            i = self.position.get(str(item))
            if i is not None:
                self.counts[i] += 1

    def get_eligible_pool(self) -> pd.DataFrame:
        eligible = self.item_pool[(self.counts < self.limits) | self.exempt]
        return complete_bundles_only(eligible, self.item_pool)

    def get_exposure_stats(self) -> Dict[str, Any]:
        used = self.counts[self.counts > 0]
        limited = self.counts[~self.exempt]
        return {'total_used': int(len(used)), 'total_pool': int(len(self.counts)),
                'utilization_pct': round(len(used) / len(self.counts) * 100, 1) if len(self.counts) else 0,
                'max_exposure': int(limited.max()) if len(limited) else 0,
                'common_exposure': int(self.counts[self.exempt].max()) if self.exempt.any() else 0,
                'mean_exposure': round(float(used.mean()), 2) if len(used) else 0,
                'domain_limits': dict(self.domain_max_usage)}


# ==================== Difficulty bins (Strategy 3) ====================

DEFAULT_DIFFICULTY_CENTERS = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]


def auto_generate_difficulty_categories(item_pool: pd.DataFrame, test_length: int, n_categories: int = 7,
                                        b_col: str = 'rasch_b') -> Dict[str, Dict[str, Any]]:
    """Bins centred on standard-normal quantiles; per-bin min/max from the normal density × test length."""
    b_values = item_pool[b_col].dropna().to_numpy()
    if len(b_values) == 0:
        return {}
    centers = DEFAULT_DIFFICULTY_CENTERS if n_categories == 7 else np.linspace(-2.5, 2.5, n_categories).tolist()
    edges = ([min(b_values.min() - 0.5, centers[0] - 0.5)] + [(a + b) / 2 for a, b in zip(centers, centers[1:])]
             + [max(b_values.max() + 0.5, centers[-1] + 0.5)])
    densities = scipy_stats.norm.pdf(centers)
    weights = densities / densities.sum()
    categories = {}
    for i, (center, weight) in enumerate(zip(centers, weights)):
        lo, hi = edges[i], edges[i + 1]
        inside = (b_values >= lo) & ((b_values <= hi) if i == len(centers) - 1 else (b_values < hi))
        pool_count = int(inside.sum())
        target = weight * test_length
        cat_min, cat_max = max(0, int(np.floor(target) - 1)), min(pool_count, int(np.ceil(target) + 1))
        if pool_count == 0:
            cat_min = cat_max = 0
        categories[f"B{i + 1} [{lo:.1f}, {hi:.1f})"] = dict(center=center, range=(lo, hi), min=cat_min, max=cat_max,
                                                          target=round(target, 1), pool_count=pool_count,
                                                          density_weight=round(float(weight), 4))
    return categories


# ==================== Active pool (Strategy 2) ====================

def _standalone_pool(eligible: pd.DataFrame, rules: Dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Domain-stratified sample of standalone items: multiplier × domain target per constrained domain."""
    test_length, multiplier = rules.get('test_length', 10), rules.get('multiplier', 5)
    target_size = test_length * multiplier
    common = {str(x) for x in rules.get('common_items', [])}
    ids = eligible['item_id'].astype(str)
    forced = eligible[ids.isin(common)]
    rest = eligible[~ids.isin(common)]

    def take(frame, n):
        if n >= len(frame):
            return frame
        return frame.iloc[np.sort(rng.choice(len(frame), size=n, replace=False))]

    constraints = rules.get('domain_constraints', {})
    if not constraints or 'domain' not in rest.columns:
        return pd.concat([forced, take(rest, max(target_size - len(forced), 0))])
    parts, used = [], np.zeros(len(rest), bool)
    domains = rest['domain'].to_numpy()
    for domain, limits in constraints.items():
        mask = domains == domain
        domain_target = limits.get('max', 0) or limits.get('min', 0)
        if not domain_target and len(rest):
            domain_target = max(1, int(np.ceil(mask.sum() / len(rest) * test_length)))
        picked = take(rest[mask], multiplier * domain_target)
        parts.append(picked)
        used |= rest.index.isin(picked.index)
    remaining = target_size - len(forced) - sum(len(p) for p in parts)
    if remaining > 0:
        parts.append(take(rest[~used], remaining))
    return pd.concat([forced] + parts)


def generate_active_pool(eligible: pd.DataFrame, rules: Dict[str, Any], rng: Optional[np.random.Generator] = None):
    """Active pool for one form. Bundles are sampled whole: multiplier × (max bundles per form, else 2)."""
    rng = rng or np.random.default_rng()
    labels = bundle_labels(eligible)
    if not labels.ne('').any():
        return _standalone_pool(eligible, rules, rng).drop_duplicates('item_id').reset_index(drop=True)
    common = {str(x) for x in rules.get('common_items', [])}
    ids = eligible['item_id'].astype(str)
    forced = sorted(set(labels[ids.isin(common) & labels.ne('')]))
    limits = rules.get('bundle_constraints') or {}
    per_form = limits.get('max_bundles') or max(limits.get('min_bundles', 0) or 0, 2)
    others = sorted(set(labels[labels != '']) - set(forced))
    n_more = max(int(rules.get('multiplier', 5)) * int(per_form) - len(forced), 0)
    if len(others) > n_more:
        others = [others[i] for i in np.sort(rng.choice(len(others), size=n_more, replace=False))]
    bundles = eligible[labels.isin(forced + others)]
    standalone = _standalone_pool(eligible[labels == ''], rules, rng)
    return pd.concat([bundles, standalone]).drop_duplicates('item_id').reset_index(drop=True)


# ==================== Sequential assembly ====================

@dataclass
class AssemblySettings:
    test_length: int
    n_forms: int
    solver_config: Dict[str, Any]          # passed to loft_v2_solver.assemble_forms_with_cpsat
    pool_rules: Dict[str, Any]             # multiplier, domain_constraints, common_items, bundle_constraints
    max_overlap: float = 0.3
    exposure_max: int = 2
    auto_domain_limits: bool = True
    domain_exposure_limits: Optional[Dict[str, int]] = None
    sampling_attempts: int = 10
    seed: Optional[int] = None


@dataclass
class SequentialAssembler:
    """Sample → solve one form → record exposure → repeat. Keeps its state between calls (Step mode)."""
    bank: pd.DataFrame
    settings: AssemblySettings
    forms: List[Dict[str, Any]] = field(default_factory=list)
    used_ids: set = field(default_factory=set)

    def __post_init__(self):
        s = self.settings
        self.tracker = ItemUsageTracker(self.bank, global_max=s.exposure_max, domain_max=s.domain_exposure_limits,
                                        auto_domain_limits=s.auto_domain_limits, n_forms=s.n_forms,
                                        test_length=s.test_length,
                                        exempt=s.solver_config.get('common_items', []))
        self.rng = np.random.default_rng(s.seed)
        self.common = set(s.solver_config.get('common_items', []))
        self.excluded = set(s.solver_config.get('excluded_items', []))

    @property
    def done(self):
        return len(self.forms) >= self.settings.n_forms

    def next_form(self, on_event: Callable[[str, Dict[str, Any]], None] = lambda *_: None) -> Dict[str, Any]:
        """Build one form. Returns {'ok': bool, 'message': str, 'form': dict | None}."""
        from loft_v2_solver import assemble_forms_with_cpsat, audit_form   # local import keeps the engine light
        s, bank = self.settings, self.bank
        if self.common - set(bank['item_id']) or self.common & self.excluded or len(self.common) > s.test_length:
            return dict(ok=False, message="Invalid common items: check IDs, exclusions, and test length.", form=None)
        last_error = ''
        for attempt in range(1, s.sampling_attempts + 1):
            eligible = self.tracker.get_eligible_pool()
            eligible = complete_bundles_only(eligible[~eligible['item_id'].isin(self.excluded)], bank)
            if not self.common <= set(eligible['item_id']):
                return dict(ok=False, form=None, message="A required common item (or another item in its bundle) has "
                            "reached its exposure limit. Increase its domain/default limit or reduce the form count.")
            if len(eligible) < s.test_length:
                return dict(ok=False, form=None,
                            message=f"Eligible pool exhausted! {len(eligible)} items remain, need {s.test_length}.")
            active = generate_active_pool(eligible, s.pool_rules, self.rng)
            on_event('sampling', dict(eligible=eligible, active=active, attempt=attempt))
            config = dict(s.solver_config, previous_forms=self.forms, max_overlap=s.max_overlap)
            result = assemble_forms_with_cpsat(active, config, 1)
            status = result['status']
            if status not in ('Optimal', 'Feasible') or not result['selected_forms'][0]:
                last_error = result.get('error_message', status)
                if status == 'ModelInvalid':
                    return dict(ok=False, message=last_error, form=None)
                on_event('retry', dict(status=status, attempt=attempt, message=last_error))
                continue
            ids = [str(x) for x in result['selected_forms'][0]]
            if len(ids) != s.test_length or len(set(ids)) != s.test_length:
                return dict(ok=False, message="Solver returned an invalid form length; no usage was recorded.", form=None)
            self.tracker.record_usage(ids)
            self.used_ids.update(ids)
            subset = bank[bank['item_id'].isin(ids)]
            b = subset['rasch_b'].to_numpy()
            theta = s.solver_config.get('eval_points', {}).get('theta_mid', 0.0)
            form = dict(selected_items=ids, solver_status=status, strategy=result.get('strategy', 'joint'),
                        bundles=(result.get('selected_bundles') or [[]])[0], attempts=attempt,
                        wall_time=float(result.get('wall_time', 0.0)), audit=audit_form(bank, ids, config),
                        metrics=dict(mean_b=float(b.mean()), sd_b=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
                                     primary_tif=calculate_tif(theta, b), primary_tcc=calculate_tcc(theta, b),
                                     mean_pvalue=float(subset['pvalue'].mean()),
                                     mean_pbs=float(subset['point_biserial'].mean())))
            self.forms.append(form)
            on_event('form', dict(form=form, index=len(self.forms)))
            return dict(ok=True, message='', form=form)
        return dict(ok=False, form=None, message=f"Failed to assemble Form {len(self.forms) + 1} after "
                    f"{s.sampling_attempts} sampling attempts. {last_error} Relax constraints or increase the pool "
                    "multiplier/time limit.")

    def run(self, on_event=lambda *_: None):
        """Build forms until done or a form fails. Returns the last next_form() result."""
        out = dict(ok=True, message='', form=None)
        while not self.done:
            out = self.next_form(on_event)
            if not out['ok']:
                break
        return out
