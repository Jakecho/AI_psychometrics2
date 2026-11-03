"""CP-SAT backend for LOFT Sequential ATA V2.

The model is built on *units*: a standalone item, or a whole item bundle (``bundle_id``; ``testlet_id`` is an
alias). Each unit is one Boolean per form, and every constraint uses the unit's summed coefficients, so a bundle
is all-or-nothing by construction and the model has one variable per bundle instead of one per item.

Rasch coefficients use conservative integer bounds at 1e-7 precision: lower bounds use floored member
coefficients and upper bounds ceiled ones, so integer rounding can never relax a requested constraint.

Bundles are solved first (``bundle_strategy='two-stage'``, the default):

1. **Bundle stage**: only bundle variables. Common units are fixed in, and the constraints are the ones every
   complete form must meet (bundle requirements, exclusions, enemies and upper limits, which standalone items
   can only add to). If this stage is infeasible, no form exists.
2. **Fill stage**: the chosen bundles become constants, and only standalone items are variables, under every
   constraint.

If a fill is infeasible, that bundle combination is cut off and stage 1 is solved again. After
``bundle_stage_attempts`` tries, or if a stage times out, one joint model solves everything together.
Each stage stops once its objective is within ``stage_gap`` (default 1 %) of the proven bound; the joint
model proves optimality unless ``joint_gap`` is set. Constraints that cannot bind are dropped before solving (an overlap cap above the reachable overlap, or a count
bound looser than the number of eligible items).
"""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from loft_v2_engine import rasch_information, rasch_probability

SCALE = 10_000_000
REQUIRED_COLUMNS = ('item_id', 'domain', 'rasch_b', 'pvalue', 'point_biserial')
FEASIBLE = ('Optimal', 'Feasible')


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
    if 'bundle_id' not in df and 'testlet_id' in df:
        df = df.rename(columns={'testlet_id': 'bundle_id'})
    if 'bundle_id' in df:
        ids = df.bundle_id.astype(object).where(df.bundle_id.notna(), '')
        df['bundle_id'] = (ids.astype(str).str.strip().replace({'nan': '', 'None': ''})
                           .str.replace(r'^(\d+)\.0$', r'\1', regex=True))
    df.attrs['loft_validated'] = True
    return df


def bundle_members(df):
    """Map bundle id -> list of row indices (standalone items excluded)."""
    if 'bundle_id' not in df:
        return {}
    labelled = df[df.bundle_id != '']
    return {b: rows.index.tolist() for b, rows in labelled.groupby('bundle_id', sort=True)}


def _scaled(value, rounding):
    return int((Decimal(str(value)) * SCALE).to_integral_value(rounding=rounding))


def _scaled_array(values, rounding):
    """_scaled for an array, using exact decimal arithmetic so rounding is always conservative."""
    return np.array([_scaled(v, rounding) for v in np.asarray(values, float)], np.int64)


def _bundle_limits(config):
    """Normalized per-form bundle requirements; a maximum of 0/None is unrestricted."""
    raw = config.get('bundle_constraints') or {}
    out = {}
    for key in ('bundles', 'bundle_items'):
        lo = int(raw.get(f'min_{key}', 0) or 0)
        hi = raw.get(f'max_{key}')
        hi = None if hi in (None, 0) else int(hi)
        if lo < 0 or (hi is not None and hi < lo):
            raise ValueError(f'Invalid {key.replace("_", " ")} limits: minimum {lo}, maximum {hi}.')
        out[key] = (lo, hi)
    return out


def _mean_target(config):
    points = config.get('eval_points', {})
    if config.get('apply_mean_diff'):
        return config.get('mean_diff_target'), config.get('mean_diff_tolerance', 0.2)
    if points.get('mean_rasch_enabled'):
        return points.get('mean_rasch_target'), points.get('mean_rasch_tolerance', 0.2)
    return None, None


def _item_weights(df, config):
    points = config.get('eval_points', {})
    theta_mid = points.get('theta_mid', 0.0)
    b = df.rasch_b.to_numpy()
    return sum(weight * rasch_information(theta, b) for weight, theta in (
        (0.1, points.get('theta_low', theta_mid - 1)), (0.8, theta_mid),
        (0.1, points.get('theta_high', theta_mid + 1))))


def _constraints(ctx):
    """Item-level constraint rows: (name, item coefficient array, lower, upper, kind).

    kind: 'count' (integer counts), 'numeric' (float; scaled conservatively), 'mean' (numeric, sign-free).
    Upper bounds of 'count' and nonnegative 'numeric' rows are necessary for partial (bundle-only) forms.
    """
    df, config, length = ctx['df'], ctx['config'], ctx['length']
    rows = []

    def count(name, mask, limits, zero_is_disabled=True):
        lo, hi = int(limits.get('min', 0)), limits.get('max')
        if hi is None or (zero_is_disabled and hi == 0):
            hi = length
        hi = int(hi)
        if lo < 0 or hi < lo or lo > length:
            raise ValueError(f'Invalid {name} limits: minimum {lo}, maximum {hi}.')
        rows.append((name, np.asarray(mask, np.int64), lo, hi, 'count'))

    for column, key in (('domain', 'domain_constraints'), ('raschb_cat', 'raschb_cat_constraints')):
        for category, limits in config.get(key, {}).items():
            count(f'{column} {category}', df[column].to_numpy() == str(category), limits)
    for column, key in (('has_image', 'image_constraint'), ('has_audio', 'audio_constraint')):
        limits = config.get(key, {})
        if limits.get('enabled', bool(limits)):
            count(column, df[column].to_numpy() == 1, limits)
    bins = config.get('difficulty_categories', {})
    b = df.rasch_b.to_numpy()
    for position, (label, limits) in enumerate(bins.items()):
        lo, hi = limits['range']
        inside = (b >= lo) & ((b <= hi) if position == len(bins) - 1 else (b < hi))
        count(f'difficulty bin {label}', inside, limits, zero_is_disabled=False)

    def numeric(name, coefficients, lower, upper, kind='numeric'):
        for bound in (lower, upper):
            if not math.isfinite(bound):
                raise ValueError('Psychometric targets and tolerances must be finite.')
        rows.append((name, np.asarray(coefficients, float), lower, upper, kind))

    points, tolerances = config.get('eval_points', {}), config.get('tif_tolerance', {})
    theta_mid = points.get('theta_mid', 0.0)
    for key in ('low', 'mid', 'high'):
        target = points.get(f'tif_{key}', 0)
        if target > 0:
            tol = tolerances.get('tif', 0.2)
            if tol < 0:
                raise ValueError('TIF tolerance cannot be negative.')
            numeric(f'TIF {key}', rasch_information(points.get(f'theta_{key}', theta_mid), b), target - tol, target + tol)
    tcc_target = points.get('tcc_mid', 0)
    if points.get('tcc_enabled', tcc_target > 0) and tcc_target > 0:
        tol = tolerances.get('tcc', 0.5)
        if tol < 0:
            raise ValueError('TCC tolerance cannot be negative.')
        numeric('TCC', rasch_probability(theta_mid, b), tcc_target - tol, tcc_target + tol)
    target, tol = _mean_target(config)
    if target is not None:
        if tol < 0:
            raise ValueError('Mean difficulty tolerance cannot be negative.')
        numeric('mean b', b, (target - tol) * length, (target + tol) * length, kind='mean')
    return rows


def _solve(ctx, stage, fixed_bundles=None, cuts=()):
    """Build and solve one stage: 'joint', 'bundles' or 'fill'."""
    units, n_forms, length, config = ctx['units'], ctx['n_forms'], ctx['length'], ctx['config']
    bundle_stage = stage == 'bundles'
    model = cp_model.CpModel()
    infeasible = []

    # Which units are variables, which are constants (1) per form, which are absent.
    variable, constant = [], []
    for f in range(n_forms):
        var_f, const_f = [], []
        for u, unit in enumerate(units):
            if unit['excluded']:
                continue
            if unit['common']:
                const_f.append(u)
            elif stage == 'fill' and unit['bundle']:
                if unit['bundle'] in fixed_bundles[f]:
                    const_f.append(u)
            elif bundle_stage and not unit['bundle']:
                continue
            else:
                var_f.append(u)
        variable.append(var_f)
        constant.append(const_f)
    v = [{u: model.new_bool_var(f'u{u}_f{f}') for u in variable[f]} for f in range(n_forms)]

    def bound(f, coef, lo=None, hi=None):
        """lo <= Σ coef[u]·x_u (constants folded in) <= hi; skips bounds that cannot bind."""
        const = int(sum(coef[u] for u in constant[f]))
        terms = [(coef[u], v[f][u]) for u in variable[f] if coef[u]]
        low_reach = const + sum(c for c, _ in terms if c < 0)
        high_reach = const + sum(c for c, _ in terms if c > 0)
        if lo is not None and high_reach < lo:
            infeasible.append(True)
        if hi is not None and low_reach > hi:
            infeasible.append(True)
        if not terms:
            return
        expr = cp_model.LinearExpr.weighted_sum([x for _, x in terms], [int(c) for c, _ in terms])
        if lo is not None and low_reach < lo:
            model.add(expr >= lo - const)
        if hi is not None and high_reach > hi:
            model.add(expr <= hi - const)

    size = np.array([unit['size'] for unit in units], np.int64)
    for f in range(n_forms):
        if bundle_stage:
            bound(f, size, hi=length)
        else:
            bound(f, size, lo=length, hi=length)

    for name, kind, lo, hi, lo_coef, hi_coef in ctx['unit_constraints']:
        if kind == 'count':
            for f in range(n_forms):
                bound(f, lo_coef, lo=None if bundle_stage else lo, hi=hi)
            continue
        if bundle_stage and kind == 'mean':
            continue
        for f in range(n_forms):
            if not bundle_stage:
                bound(f, lo_coef, lo=_scaled(lo, ROUND_CEILING))
            bound(f, hi_coef, hi=_scaled(hi, ROUND_FLOOR))   # coefficients ≥ 0 for TIF/TCC rows

    for a, b in ctx['enemy_units']:
        for f in range(n_forms):
            coef = np.zeros(len(units), np.int64)
            coef[[a, b]] = 1
            bound(f, coef, hi=1)

    for overlap, cap in ctx['overlap_caps']:
        for f in range(n_forms):
            bound(f, overlap, hi=cap)

    lim = ctx['bundle_limits']
    is_bundle = np.array([1 if unit['bundle'] else 0 for unit in units], np.int64)
    for f in range(n_forms):
        for (lo, hi), coef in ((lim['bundles'], is_bundle), (lim['bundle_items'], size * is_bundle)):
            bound(f, coef, lo=lo or None, hi=hi)

    if n_forms > 1:      # forms share only common units
        for u in range(len(units)):
            holders = [v[f][u] for f in range(n_forms) if u in v[f]]
            if len(holders) > 1:
                model.add_at_most_one(holders)

    for chosen in cuts:  # bundle combinations whose fill stage was proven infeasible
        flips = [(1 - x) if units[u]['bundle'] in chosen[f] else x for f in range(n_forms) for u, x in v[f].items()]
        model.add(sum(flips) >= 1)

    if infeasible:
        return dict(status='Infeasible', wall_time=0.0)
    objective = ctx['objective']
    extra_objective = []
    if bundle_stage:
        extra_objective = _look_ahead(model, ctx, v, constant, size)
    model.maximize(cp_model.LinearExpr.weighted_sum(
        [x for f in range(n_forms) for x in v[f].values()] + extra_objective,
        [int(objective[u]) for f in range(n_forms) for u in v[f]] + [1] * len(extra_objective)))
    invalid = model.validate()
    if invalid:
        return dict(status='ModelInvalid', message=invalid, wall_time=0.0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.get('time_limit', 10.0))
    # A single worker also avoids native parallel-search stalls on macOS.
    solver.parameters.num_search_workers = int(config.get('num_workers', 1))
    solver.parameters.random_seed = int(config.get('random_seed', 0))
    # Two-stage forms are not claimed optimal, so their stages may stop within a small gap of the bound.
    gap = config.get('stage_gap', 0.01) if stage in ('bundles', 'fill') else config.get('joint_gap', 0.0)
    solver.parameters.relative_gap_limit = float(gap)
    code = solver.solve(model)
    status = {cp_model.OPTIMAL: 'Optimal', cp_model.FEASIBLE: 'Feasible', cp_model.INFEASIBLE: 'Infeasible',
              cp_model.MODEL_INVALID: 'ModelInvalid', cp_model.UNKNOWN: 'Unknown'}.get(code, 'Unknown')
    try:
        wall = solver.wall_time
    except RuntimeError:          # no search was run (e.g. a stubbed solver)
        wall = 0.0
    out = dict(status=status, wall_time=wall)
    if status in FEASIBLE:
        chosen = [constant[f] + [u for u, x in v[f].items() if solver.value(x)] for f in range(n_forms)]
        out['units'] = chosen
        out['bundles'] = [{units[u]['bundle'] for u in c if units[u]['bundle']} for c in chosen]
    return out


def _look_ahead(model, ctx, v, constant, size):
    """Bundle-stage relaxation of the standalone fill (necessary conditions only).

    With r open slots, r standalone items can add at most the sum of the r largest coefficients and at least
    the sum of the r smallest. Each lower bound must be reachable with the largest fill and each upper bound with
    the smallest, using the conservative coefficient for each side. The returned variables add the best
    possible fill objective, so stage 1 ranks bundle sets by the best whole form they allow.
    """
    units, length = ctx['units'], ctx['length']
    fill = [u for u, unit in enumerate(units) if not unit['bundle'] and not unit['excluded'] and not unit['common']]
    max_r = min(length, len(fill))
    extra = []

    def prefix(values, descending):
        ordered = np.sort(np.asarray(values, np.int64))
        if descending:
            ordered = ordered[::-1]
        return [int(x) for x in np.r_[0, np.cumsum(ordered)][:max_r + 1]]

    def table(r, values, name):
        t = model.new_int_var(min(values), max(values), name)
        model.add_element(r, values, t)
        return t

    for f, vf in enumerate(v):
        const_size = int(sum(size[u] for u in constant[f]))
        r = model.new_int_var(0, max_r, f'open_slots_{f}')
        model.add(r + cp_model.LinearExpr.weighted_sum(list(vf.values()), [int(size[u]) for u in vf])
                  == length - const_size)

        def side(coef, rhs, fill_values, sense, name):
            const = int(sum(coef[u] for u in constant[f]))
            t = table(r, prefix(fill_values, descending=sense == '>='), name)
            expr = cp_model.LinearExpr.weighted_sum(list(vf.values()) + [t], [int(coef[u]) for u in vf] + [1])
            model.add(expr >= rhs - const) if sense == '>=' else model.add(expr <= rhs - const)

        for k, (name, kind, lo, hi, lo_coef, hi_coef) in enumerate(ctx['unit_constraints']):
            if kind == 'count':
                if lo:
                    side(lo_coef, lo, lo_coef[fill], '>=', f'fill_max_{k}_{f}')
                side(hi_coef, hi, hi_coef[fill], '<=', f'fill_min_{k}_{f}')
            else:
                side(lo_coef, _scaled(lo, ROUND_CEILING), hi_coef[fill], '>=', f'fill_max_{k}_{f}')
                side(hi_coef, _scaled(hi, ROUND_FLOOR), lo_coef[fill], '<=', f'fill_min_{k}_{f}')
        extra.append(table(r, prefix(ctx['objective'][fill], descending=True), f'fill_objective_{f}'))
    return extra


def assemble_forms_with_cpsat(items_df, config, n_forms=1):
    """Assemble valid Rasch forms, returning selections only for feasible statuses.

    The LOFT dashboard calls this with one form and resamples between calls. Multi-form callers get disjoint
    non-common items. A time-limited feasible solution is usable and is distinguished from a proven optimum.
    ``config['bundle_strategy']`` is 'two-stage' (default) or 'joint'.
    """
    df = validate_item_bank(items_df)
    length = config['test_length']
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError('Test length must be a positive integer.')
    if not isinstance(n_forms, (int, np.integer)) or n_forms < 1:
        raise ValueError('Number of forms must be a positive integer.')

    def failure(status, message, **extra):
        return dict(status=status, selected_forms=[[] for _ in range(n_forms)], selected_bundles=[[] for _ in range(n_forms)],
                    objective_value=0.0, form_objectives=[], solver='CP-SAT', error_message=message,
                    validation_errors=[message], **extra)

    ids = df.item_id.tolist()
    index = {item_id: i for i, item_id in enumerate(ids)}
    members = bundle_members(df)
    bundle_of = {i: b for b, rows in members.items() for i in rows}
    common = {str(x).strip() for x in config.get('common_items', [])}
    excluded = {str(x).strip() for x in config.get('excluded_items', [])} & index.keys()
    if common - index.keys():
        return failure('Infeasible', f"Common items not found: {', '.join(sorted(common - index.keys()))}")
    common |= {ids[i] for c in list(common) if index[c] in bundle_of for i in members[bundle_of[index[c]]]}
    if common & excluded:
        return failure('Infeasible', 'Common items (including the rest of their bundles) cannot also be excluded.')
    if len(common) > length:
        return failure('Infeasible', 'Common item count (including the rest of their bundles) exceeds test length.')
    if len(df) - len(excluded) < len(common) + (length - len(common)) * n_forms:
        return failure('Infeasible', 'Not enough eligible items for the requested form length and count.')
    for column, key in (('domain', 'domain_constraints'), ('raschb_cat', 'raschb_cat_constraints')):
        if config.get(key) and column not in df:
            return failure('Infeasible', f'{column} is required by the selected constraints.')
    for column, key in (('has_image', 'image_constraint'), ('has_audio', 'audio_constraint')):
        limits = config.get(key, {})
        if limits.get('enabled', bool(limits)) and column not in df:
            return failure('Infeasible', f'{column} is required by the selected media constraint.')
    try:
        limits = _bundle_limits(config)
    except ValueError as exc:
        return failure('ModelInvalid', str(exc))
    if not members and (limits['bundles'][0] or limits['bundle_items'][0]):
        return failure('Infeasible', 'Bundle minimums require a bundle_id column with bundled items.')
    overlap = float(config.get('max_overlap', 1.0))
    if not 0 <= overlap <= 1:
        raise ValueError('Max Jaccard overlap must be between 0 and 1.')

    # Units: one per standalone item, one per bundle.
    units = [dict(rows=np.array(rows), bundle=b, size=len(rows)) for b, rows in members.items()]
    units += [dict(rows=np.array([i]), bundle='', size=1) for i in range(len(df)) if i not in bundle_of]
    unit_of = {int(i): u for u, unit in enumerate(units) for i in unit['rows']}
    for unit in units:
        unit_ids = {ids[i] for i in unit['rows']}
        unit['common'] = bool(unit_ids & common)
        unit['excluded'] = bool(unit_ids & excluded)

    enemy_units = set()
    if config.get('enemy_check'):
        column = 'enemy_ids' if 'enemy_ids' in df else 'enemy'
        if column in df:
            for i, text in enumerate(df[column].fillna('')):
                for enemy in str(text).split(','):
                    j = index.get(enemy.strip())
                    if j is None or j == i:
                        continue
                    a, b = sorted((unit_of[i], unit_of[j]))
                    if a == b:
                        units[a]['excluded'] = True       # an enemy pair inside one bundle: never usable
                        if units[a]['common']:
                            return failure('Infeasible', f"Common bundle {units[a]['bundle']} contains an enemy pair.")
                    else:
                        enemy_units.add((a, b))

    caps = []
    for past_form in config.get('previous_forms', []):
        past = set(str(i) for i in past_form['selected_items'])
        ratio = Decimal(str(overlap))
        cap = int((ratio * (length + len(past)) / (1 + ratio)).to_integral_value(rounding=ROUND_FLOOR))
        coef = np.array([sum(ids[i] in past for i in unit['rows']) for unit in units], np.int64)
        if coef.sum() > cap:                      # otherwise the cap cannot bind
            caps.append((coef, cap))

    weights = _item_weights(df, config)
    objective = np.array([int(sum(round(float(weights[i]) * SCALE) for i in unit['rows'])) for unit in units])
    ctx = dict(df=df, config=config, n_forms=n_forms, length=length, units=units, enemy_units=enemy_units,
               overlap_caps=caps, bundle_limits=limits, objective=objective)
    # Per-unit coefficients, computed once and shared by every stage and retry.
    unit_rows = np.concatenate([unit['rows'] for unit in units])

    unit_starts = np.r_[0, np.cumsum([unit['size'] for unit in units])[:-1]]

    def per_unit(item_values):
        return np.add.reduceat(np.asarray(item_values, np.int64)[unit_rows], unit_starts)

    ctx['unit_constraints'] = []
    for name, coefficients, lo, hi, kind in _constraints(ctx):
        if kind == 'count':
            coef = per_unit(coefficients)
            ctx['unit_constraints'].append((name, kind, lo, hi, coef, coef))
        else:
            ctx['unit_constraints'].append((name, kind, lo, hi, per_unit(_scaled_array(coefficients, ROUND_FLOOR)),
                                            per_unit(_scaled_array(coefficients, ROUND_CEILING))))

    def finish(result, strategy, attempts, wall):
        rows = [sorted(int(i) for u in chosen for i in units[u]['rows']) for chosen in result['units']]
        form_objectives = [float(sum(weights[i] for i in r)) for r in rows]
        status = result['status'] if strategy.startswith('joint') else 'Feasible'
        return dict(status=status, stage_status=result['status'], selected_forms=[[ids[i] for i in r] for r in rows],
                    selected_bundles=[sorted(b) for b in result['bundles']], objective_value=sum(form_objectives),
                    form_objectives=form_objectives, solver='CP-SAT', wall_time=wall, optimal=status == 'Optimal',
                    strategy=strategy, bundle_stage_attempts=attempts)

    strategy = config.get('bundle_strategy', 'two-stage')
    wall, attempts, cuts = 0.0, 0, []
    if members and strategy == 'two-stage':
        for attempts in range(1, int(config.get('bundle_stage_attempts', 5)) + 1):
            first = _solve(ctx, 'bundles', cuts=cuts)
            wall += first['wall_time']
            if first['status'] == 'ModelInvalid':
                return failure('ModelInvalid', first['message'])
            if first['status'] == 'Infeasible':
                message = ('No combination of bundles meets the bundle requirements and limits in this active pool.'
                           if not cuts else 'Every bundle combination that meets the bundle requirements leaves '
                                            'no valid standalone fill in this active pool.')
                return failure('Infeasible', message, strategy='two-stage', bundle_stage_attempts=attempts)
            if first['status'] not in FEASIBLE:
                break
            second = _solve(ctx, 'fill', fixed_bundles=first['bundles'])
            wall += second['wall_time']
            if second['status'] in FEASIBLE:
                return finish(second, 'two-stage', attempts, wall)
            if second['status'] == 'ModelInvalid':
                return failure('ModelInvalid', second['message'])
            if second['status'] != 'Infeasible':
                break
            cuts.append(first['bundles'])
        strategy = 'joint (fallback)'
    else:
        strategy = 'joint'
    result = _solve(ctx, 'joint')
    wall += result['wall_time']
    if result['status'] == 'ModelInvalid':
        return failure('ModelInvalid', result['message'])
    if result['status'] not in FEASIBLE:
        message = ('No form satisfies all selected constraints in this active pool.' if result['status'] == 'Infeasible'
                   else f"CP-SAT returned {result['status']}; no valid form is available. Try a longer time limit.")
        return failure(result['status'], message, strategy=strategy, bundle_stage_attempts=attempts)
    return finish(result, strategy, attempts, wall)


def audit_form(items_df, selected_ids, config):
    """Recheck one assembled form against every configured constraint.

    Independent of the CP-SAT model: counts and sums are recomputed from the bank item by item.
    Returns rows of (constraint, requirement, actual, status).
    """
    df = items_df if items_df.attrs.get('loft_validated') else validate_item_bank(items_df)
    ids = [str(i) for i in selected_ids]
    id_set = set(ids)
    s = df[df.item_id.isin(id_set)]
    length = config['test_length']
    rows = []

    def add(name, requirement, actual, ok):
        rows.append(dict(constraint=name, requirement=requirement, actual=actual, status='Pass' if ok else 'Fail'))

    def count(name, n, limits, zero_is_disabled=True):
        lo, hi = int(limits.get('min', 0)), limits.get('max')
        if hi is None or (zero_is_disabled and hi == 0):
            hi = length
        add(name, f'{lo}–{hi}', int(n), lo <= n <= hi)

    add('Test length', str(length), len(ids), len(ids) == length == len(id_set) == len(s))
    common = {str(x) for x in config.get('common_items', [])}
    if common:
        add('Common items included', ', '.join(sorted(common)), len(common & id_set), common <= id_set)
    excluded = {str(x) for x in config.get('excluded_items', [])}
    if excluded:
        add('Excluded items absent', ', '.join(sorted(excluded)), len(excluded & id_set), not excluded & id_set)
    for column, key in (('domain', 'domain_constraints'), ('raschb_cat', 'raschb_cat_constraints')):
        for category, limits in config.get(key, {}).items():
            count(f'{column} = {category}', (s[column] == str(category)).sum(), limits)
    for column, key, label in (('has_image', 'image_constraint', 'Image items'),
                               ('has_audio', 'audio_constraint', 'Audio items')):
        limits = config.get(key, {})
        if limits.get('enabled', bool(limits)):
            count(label, s[column].sum(), limits)
    bins = config.get('difficulty_categories', {})
    for position, (label, limits) in enumerate(bins.items()):
        lo, hi = limits['range']
        inside = (s.rasch_b >= lo) & (s.rasch_b <= hi if position == len(bins) - 1 else s.rasch_b < hi)
        count(f'Difficulty bin {label}', inside.sum(), limits, zero_is_disabled=False)
    if config.get('enemy_check') and 'enemy_ids' in df:
        clashes = sorted({tuple(sorted((r.item_id, e.strip()))) for r in s.itertuples()
                          for e in str(r.enemy_ids).split(',') if e.strip() in id_set - {r.item_id}})
        add('Enemy pairs', 'none', len(clashes), not clashes)
    members = bundle_members(df)
    if members:
        chosen = {b for b, r in members.items() if df.loc[r, 'item_id'].isin(id_set).any()}
        partial = [b for b in chosen if not df.loc[members[b], 'item_id'].isin(id_set).all()]
        add('Bundles complete (all-or-nothing)', 'every bundle whole', f'{len(chosen) - len(partial)}/{len(chosen)} whole',
            not partial)
        limits = _bundle_limits(config)
        n_items = int(s.bundle_id.ne('').sum())
        for key, label, n in (('bundles', 'Bundles', len(chosen)), ('bundle_items', 'Bundle items', n_items)):
            lo, hi = limits[key]
            if lo or hi is not None:
                add(label, f'{lo}–{hi if hi is not None else "∞"}', n, lo <= n and (hi is None or n <= hi))
    points, tolerances = config.get('eval_points', {}), config.get('tif_tolerance', {})
    theta_mid = points.get('theta_mid', 0.0)
    b = s.rasch_b.to_numpy()
    for key in ('low', 'mid', 'high'):
        target = points.get(f'tif_{key}', 0)
        if target > 0:
            theta, tol = points.get(f'theta_{key}', theta_mid), tolerances.get('tif', 0.2)
            value = float(rasch_information(theta, b).sum())
            add(f'TIF at θ = {theta:g}', f'{target:g} ± {tol:g}', round(value, 4), abs(value - target) <= tol + 1e-9)
    tcc_target = points.get('tcc_mid', 0)
    if points.get('tcc_enabled', tcc_target > 0) and tcc_target > 0:
        tol = tolerances.get('tcc', 0.5)
        value = float(rasch_probability(theta_mid, b).sum())
        add(f'TCC at θ = {theta_mid:g}', f'{tcc_target:g} ± {tol:g}', round(value, 4),
            abs(value - tcc_target) <= tol + 1e-9)
    target, tol = _mean_target(config)
    if target is not None:
        value = float(b.mean()) if len(b) else float('nan')
        add('Mean Rasch b', f'{target:g} ± {tol:g}', round(value, 4), abs(value - target) <= tol + 1e-9)
    max_overlap = float(config.get('max_overlap', 1.0))
    worst = 0.0
    for past in config.get('previous_forms', []):
        p = set(str(i) for i in past['selected_items'])
        worst = max(worst, len(id_set & p) / len(id_set | p) if id_set | p else 0.0)
    if config.get('previous_forms'):
        add(f"Max Jaccard overlap with {len(config['previous_forms'])} earlier form(s)", f'≤ {max_overlap:g}',
            round(worst, 4), worst <= max_overlap + 1e-12)
    return rows
