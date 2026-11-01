"""Streamlit-independent CP-SAT backend for sequential LOFT assembly.

Rasch coefficients use conservative integer bounds at 1e-7 precision: lower
bounds use floored coefficients and upper bounds use ceiled coefficients.
Thus integer rounding cannot relax a requested psychometric constraint.
"""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from loft_sequential_engine import rasch_information, rasch_probability

SCALE = 10_000_000
REQUIRED_COLUMNS = ('item_id', 'domain', 'rasch_b', 'pvalue', 'point_biserial')


def validate_item_bank(items):
    """Normalize bank data; reject invalid inputs before constructing a model."""
    df = items.copy().reset_index(drop=True)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError('The item bank is empty.')
    if df.item_id.isna().any():
        raise ValueError('Item IDs cannot be missing.')
    df['item_id'] = df.item_id.astype(str).str.strip()
    if df.item_id.eq('').any() or df.item_id.duplicated().any():
        raise ValueError('Item IDs must be nonempty and unique.')
    for column in ('rasch_b', 'pvalue', 'point_biserial'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if not np.isfinite(df[column]).all():
            raise ValueError(f'{column} must contain finite numeric values for every item.')
    if not df.pvalue.between(0, 1).all():
        raise ValueError('pvalue must be between 0 and 1.')
    if not df.point_biserial.between(-1, 1).all():
        raise ValueError('point_biserial must be between -1 and 1.')
    for column in ('domain', 'raschb_cat'):
        if column in df:
            df[column] = df[column].fillna('Unspecified').astype(str).str.strip().replace('', 'Unspecified')
    for column in ('has_image', 'has_audio'):
        if column in df:
            values = pd.to_numeric(df[column], errors='coerce')
            if not values.isin([0, 1]).all():
                raise ValueError(f'{column} must contain only 0 or 1.')
            df[column] = values.astype(int)
    return df


def _scaled(value, rounding):
    return int((Decimal(str(value)) * SCALE).to_integral_value(rounding=rounding))


def assemble_forms_with_cpsat(items_df, config, n_forms=1):
    """Assemble valid Rasch forms, returning selections only for feasible statuses.

    The LOFT dashboard calls this with one form and resamples between calls.
    Multi-form callers get disjoint non-common items. A time-limited feasible
    solution is usable and is explicitly distinguished from a proven optimum.
    """
    df = validate_item_bank(items_df)
    length = config['test_length']
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError('Test length must be a positive integer.')
    if not isinstance(n_forms, (int, np.integer)) or n_forms < 1:
        raise ValueError('Number of forms must be a positive integer.')

    def failure(status, message):
        return dict(status=status, selected_forms=[[] for _ in range(n_forms)],
                    objective_value=0.0, form_objectives=[], solver='CP-SAT',
                    error_message=message, validation_errors=[message])

    ids = df.item_id.tolist()
    index = {item_id: i for i, item_id in enumerate(ids)}
    common = {str(x).strip() for x in config.get('common_items', [])}
    excluded = {str(x).strip() for x in config.get('excluded_items', [])}
    if common - index.keys():
        return failure('Infeasible', f"Common items not found: {', '.join(sorted(common - index.keys()))}")
    if common & excluded:
        return failure('Infeasible', 'Common items cannot also be excluded.')
    if len(common) > length:
        return failure('Infeasible', 'Common item count exceeds test length.')
    if len(df) - len(excluded & index.keys()) < len(common) + (length - len(common)) * n_forms:
        return failure('Infeasible', 'Not enough eligible items for the requested form length and count.')

    model = cp_model.CpModel()
    x = [[model.new_bool_var(f'item_{i}_form_{f}') for i in range(len(df))]
         for f in range(n_forms)]
    for variables in x:
        model.add(sum(variables) == length)
        for item_id in common:
            model.add(variables[index[item_id]] == 1)
        for item_id in excluded & index.keys():
            model.add(variables[index[item_id]] == 0)
    if n_forms > 1:
        for i, item_id in enumerate(ids):
            if item_id not in common:
                model.add(sum(variables[i] for variables in x) <= 1)

    def count_bounds(indices, limits, label, zero_is_disabled=True):
        lo, hi = int(limits.get('min', 0)), limits.get('max')
        if hi is None or (zero_is_disabled and hi == 0):
            hi = length
        hi = int(hi)
        if lo < 0 or hi < lo or lo > length:
            raise ValueError(f'Invalid {label} limits: minimum {lo}, maximum {hi}.')
        for variables in x:
            expr = sum(variables[i] for i in indices)
            model.add(expr >= lo)
            model.add(expr <= hi)

    for column, key in (('domain', 'domain_constraints'), ('raschb_cat', 'raschb_cat_constraints')):
        constraints = config.get(key, {})
        if constraints and column not in df:
            return failure('Infeasible', f'{column} is required by the selected constraints.')
        for category, limits in constraints.items():
            indices = df.index[df[column] == str(category)].tolist()
            count_bounds(indices, limits, f'{column} {category}')

    for column, key in (('has_image', 'image_constraint'), ('has_audio', 'audio_constraint')):
        limits = config.get(key, {})
        if limits.get('enabled', bool(limits)):
            if column not in df:
                return failure('Infeasible', f'{column} is required by the selected media constraint.')
            count_bounds(df.index[df[column] == 1].tolist(), limits, column)

    bins = config.get('difficulty_categories', {})
    for position, (label, limits) in enumerate(bins.items()):
        lo, hi = limits['range']
        in_bin = (df.rasch_b >= lo) & (df.rasch_b <= hi if position == len(bins) - 1 else df.rasch_b < hi)
        count_bounds(df.index[in_bin].tolist(), limits, f'difficulty bin {label}', zero_is_disabled=False)

    if config.get('enemy_check'):
        enemy_column = 'enemy_ids' if 'enemy_ids' in df else 'enemy'
        pairs = set()
        if enemy_column in df:
            for i, text in enumerate(df[enemy_column].fillna('')):
                for enemy_id in str(text).split(','):
                    j = index.get(enemy_id.strip())
                    if j is not None and j != i:
                        pairs.add(tuple(sorted((i, j))))
        for i, j in pairs:
            for variables in x:
                model.add(variables[i] + variables[j] <= 1)

    overlap = float(config.get('max_overlap', 1.0))
    if not 0 <= overlap <= 1:
        raise ValueError('Max Jaccard overlap must be between 0 and 1.')
    for past_form in config.get('previous_forms', []):
        past_ids = set(str(i) for i in past_form['selected_items'])
        ratio = Decimal(str(overlap))
        cap = int((ratio * (length + len(past_ids)) / (1 + ratio)).to_integral_value(rounding=ROUND_FLOOR))
        for variables in x:
            model.add(sum(variables[index[i]] for i in past_ids & index.keys()) <= cap)

    # Preserve hard floating-point constraints using conservative integer bounds.
    def numeric_bounds(coefficients, lower=None, upper=None):
        for bound in (lower, upper):
            if bound is not None and not math.isfinite(bound):
                raise ValueError('Psychometric targets and tolerances must be finite.')
        if lower is not None:
            coefficients_lo = [_scaled(v, ROUND_FLOOR) for v in coefficients]
            for variables in x:
                model.add(sum(c * v for c, v in zip(coefficients_lo, variables)) >= _scaled(lower, ROUND_CEILING))
        if upper is not None:
            coefficients_hi = [_scaled(v, ROUND_CEILING) for v in coefficients]
            for variables in x:
                model.add(sum(c * v for c, v in zip(coefficients_hi, variables)) <= _scaled(upper, ROUND_FLOOR))

    points = config.get('eval_points', {})
    tolerances = config.get('tif_tolerance', {})
    theta_mid = points.get('theta_mid', 0.0)
    b = df.rasch_b.to_numpy()
    weights = sum(weight * rasch_information(theta, b) for weight, theta in (
        (0.1, points.get('theta_low', theta_mid - 1)),
        (0.8, theta_mid), (0.1, points.get('theta_high', theta_mid + 1))))
    for key in ('low', 'mid', 'high'):
        target = points.get(f'tif_{key}', 0)
        if target > 0:
            tol = tolerances.get('tif', 0.2)
            if tol < 0:
                raise ValueError('TIF tolerance cannot be negative.')
            theta = points.get(f'theta_{key}', theta_mid)
            numeric_bounds(rasch_information(theta, b), target - tol, target + tol)
    tcc_target = points.get('tcc_mid', 0)
    if points.get('tcc_enabled', tcc_target > 0) and tcc_target > 0:
        tol = tolerances.get('tcc', 0.5)
        if tol < 0:
            raise ValueError('TCC tolerance cannot be negative.')
        numeric_bounds(rasch_probability(theta_mid, b), tcc_target - tol, tcc_target + tol)
    if config.get('apply_mean_diff'):
        target = config.get('mean_diff_target')
        tol = config.get('mean_diff_tolerance', 0.2)
    elif points.get('mean_rasch_enabled'):
        target = points.get('mean_rasch_target')
        tol = points.get('mean_rasch_tolerance', 0.2)
    else:
        target = None
    if target is not None:
        if tol < 0:
            raise ValueError('Mean difficulty tolerance cannot be negative.')
        numeric_bounds(b, (target - tol) * length, (target + tol) * length)

    objective = [int(round(float(w) * SCALE)) for w in weights]
    model.maximize(sum(c * v for variables in x for c, v in zip(objective, variables)))
    invalid = model.validate()
    if invalid:
        return failure('ModelInvalid', invalid)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.get('time_limit', 10.0))
    # A single worker also avoids native parallel-search stalls on macOS.
    solver.parameters.num_search_workers = int(config.get('num_workers', 1))
    solver.parameters.random_seed = int(config.get('random_seed', 0))
    status_code = solver.solve(model)
    status = {cp_model.OPTIMAL: 'Optimal', cp_model.FEASIBLE: 'Feasible',
              cp_model.INFEASIBLE: 'Infeasible', cp_model.MODEL_INVALID: 'ModelInvalid',
              cp_model.UNKNOWN: 'Unknown'}.get(status_code, 'Unknown')
    if status not in ('Optimal', 'Feasible'):
        message = ('No form satisfies all selected constraints in this active pool.'
                   if status == 'Infeasible' else f'CP-SAT returned {status}; no valid form is available. Try a longer time limit.')
        return failure(status, message)
    selected_indices = [[i for i, v in enumerate(variables) if solver.value(v)] for variables in x]
    form_objectives = [float(sum(weights[i] for i in selection)) for selection in selected_indices]
    return dict(status=status, selected_forms=[[ids[i] for i in selection] for selection in selected_indices],
                objective_value=sum(form_objectives), form_objectives=form_objectives,
                solver='CP-SAT', wall_time=solver.wall_time, optimal=status == 'Optimal')
