"""Constraint and dashboard regressions. Run with: python -m pytest test_loft_cpsat.py.

Set LOFT_SAMPLE_BANK to exercise an external bank without adding it to Git.
"""

import copy
import io
import itertools
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from ortools.sat.python import cp_model
from streamlit.testing.v1 import AppTest

from loft_cpsat_solver import assemble_forms_with_cpsat, validate_item_bank
from loft_sequential_engine import (
    ItemUsageTracker, auto_generate_difficulty_categories, generate_active_pool,
    calculate_tif, calculate_tcc,
)


@pytest.fixture
def bank():
    return pd.DataFrame({
        'item_id': [f'I{i}' for i in range(16)],
        'domain': ['A', 'B'] * 8,
        'rasch_b': [-1., -.7, -.5, -.2, 0., .2, .5, .7] * 2,
        'pvalue': [.7] * 16,
        'point_biserial': [.3] * 16,
        'raschb_cat': ['easy'] * 8 + ['hard'] * 8,
        'has_image': [0, 1, 0, 1] * 4,
        'has_audio': [1, 1, 0, 0] * 4,
        'enemy_ids': ['I1'] + [''] * 15,
    })


def solve(bank, **config):
    return assemble_forms_with_cpsat(bank, dict(test_length=4, time_limit=2, **config))


def selection(bank, result, i=0):
    assert result['status'] in ('Optimal', 'Feasible'), result
    return bank[bank.item_id.isin(result['selected_forms'][i])]


def test_objective_matches_exhaustive_search(bank):
    pool = bank.iloc[:8]
    result = solve(pool)
    expected = max(.1 * calculate_tif(-1, pool.iloc[list(c)].rasch_b)
                   + .8 * calculate_tif(0, pool.iloc[list(c)].rasch_b)
                   + .1 * calculate_tif(1, pool.iloc[list(c)].rasch_b)
                   for c in itertools.combinations(range(8), 4))
    assert result['status'] == 'Optimal'
    assert result['objective_value'] == pytest.approx(expected)
    assert len(selection(pool, result)) == 4


def test_domains_categories_media_enemies_and_ids(bank):
    r = solve(bank, domain_constraints={'A': {'min': 2, 'max': 2}, 'B': {'min': 2, 'max': 2}},
              raschb_cat_constraints={'easy': {'min': 2, 'max': 2}, 'hard': {'min': 2, 'max': 2}},
              image_constraint={'min': 2, 'max': 2}, audio_constraint={'min': 2, 'max': 2},
              enemy_check=True, common_items=['I0'], excluded_items=['I4'])
    s = selection(bank, r)
    assert s.domain.value_counts().to_dict() == {'A': 2, 'B': 2}
    assert s.raschb_cat.value_counts().to_dict() == {'easy': 2, 'hard': 2}
    assert s.has_image.sum() == s.has_audio.sum() == 2
    assert 'I0' in s.item_id.values and 'I1' not in s.item_id.values and 'I4' not in s.item_id.values


def test_partial_domain_constraints_do_not_limit_other_domains(bank):
    s = selection(bank, solve(bank, domain_constraints={'A': {'min': 1, 'max': 1}}))
    assert sum(s.domain == 'A') == 1
    assert sum(s.domain == 'B') == 3


def test_bins_are_independent_of_existing_categories(bank):
    bins = {'negative': {'range': (-2, 0), 'min': 1, 'max': 1},
            'positive': {'range': (0, 2), 'min': 3, 'max': 3}}
    s = selection(bank, solve(bank, difficulty_categories=bins,
                             raschb_cat_constraints={'hard': {'min': 2, 'max': 2}}))
    assert sum(s.rasch_b < 0) == 1
    assert sum(s.rasch_b >= 0) == 3
    assert sum(s.raschb_cat == 'hard') == 2


def test_numeric_rasch_constraints(bank):
    target = bank.iloc[[3, 4, 5, 6]]
    points = {f'theta_{k}': t for k, t in zip(('low', 'mid', 'high'), (-1., 0., 1.))}
    points.update({f'tif_{k}': calculate_tif(t, target.rasch_b) for k, t in zip(('low', 'mid', 'high'), (-1., 0., 1.))})
    points.update(tcc_mid=calculate_tcc(0, target.rasch_b), tcc_enabled=True)
    s = selection(bank, solve(bank, eval_points=points, tif_tolerance={'tif': .02, 'tcc': .03},
                             apply_mean_diff=True, mean_diff_target=target.rasch_b.mean(), mean_diff_tolerance=.02))
    for k in ('low', 'mid', 'high'):
        assert abs(calculate_tif(points[f'theta_{k}'], s.rasch_b) - points[f'tif_{k}']) <= .02
    assert abs(calculate_tcc(0, s.rasch_b) - points['tcc_mid']) <= .03
    assert abs(s.rasch_b.mean() - target.rasch_b.mean()) <= .02


def test_exact_mean_and_conservative_rounding(bank):
    d = bank.iloc[:2].copy()
    d['rasch_b'] = [-.1, .1]
    assert assemble_forms_with_cpsat(d, {'test_length': 2, 'apply_mean_diff': True,
        'mean_diff_target': 0, 'mean_diff_tolerance': 0})['status'] == 'Optimal'
    # Rounding to the nearest scaled integer must not admit an out-of-range mean.
    d['rasch_b'] = [0.000000049, 0.000000049]
    assert assemble_forms_with_cpsat(d, {'test_length': 2, 'apply_mean_diff': True,
        'mean_diff_target': 0, 'mean_diff_tolerance': 0})['status'] == 'Infeasible'


def test_overlap_is_hard_even_when_only_same_form_available(bank):
    first = solve(bank)['selected_forms'][0]
    r = solve(bank, previous_forms=[{'selected_items': first}], max_overlap=0)
    assert set(r['selected_forms'][0]).isdisjoint(first)
    r = solve(bank[bank.item_id.isin(first)], previous_forms=[{'selected_items': first}], max_overlap=.3)
    assert r['status'] == 'Infeasible'
    assert r['selected_forms'] == [[]]


def test_multiple_forms_only_share_common_items(bank):
    r = assemble_forms_with_cpsat(bank, {'test_length': 4, 'common_items': ['I0']}, 3)
    assert r['status'] == 'Optimal'
    for a, b in itertools.combinations(r['selected_forms'], 2):
        assert set(a) & set(b) == {'I0'}


@pytest.mark.parametrize('config', [
    {'common_items': ['missing']}, {'common_items': ['I0'], 'excluded_items': ['I0']},
    {'domain_constraints': {'missing': {'min': 1, 'max': 1}}},
    {'eval_points': {'tif_mid': 100}},
    {'image_constraint': {'min': 4, 'max': 4}, 'audio_constraint': {'min': 4, 'max': 4}, 'common_items': ['I0']},
])
def test_infeasible_has_no_partial_selection(bank, config):
    r = solve(bank, **config)
    assert r['status'] == 'Infeasible'
    assert r['selected_forms'] == [[]]
    assert r['error_message']


def test_missing_media_is_an_error(bank):
    r = solve(bank.drop(columns=['has_audio']), audio_constraint={'min': 1, 'max': 2})
    assert r['status'] == 'Infeasible' and 'has_audio' in r['error_message']


@pytest.mark.parametrize('mutation', ['duplicate', 'blank', 'missing', 'nan', 'infinite', 'pvalue', 'media'])
def test_bad_banks_rejected(bank, mutation):
    d = bank.copy()
    if mutation == 'duplicate': d.loc[1, 'item_id'] = d.loc[0, 'item_id']
    if mutation == 'blank': d.loc[0, 'item_id'] = ' '
    if mutation == 'missing': d = d.drop(columns=['rasch_b'])
    if mutation == 'nan': d.loc[0, 'rasch_b'] = np.nan
    if mutation == 'infinite': d.loc[0, 'rasch_b'] = np.inf
    if mutation == 'pvalue': d.loc[0, 'pvalue'] = 1.2
    if mutation == 'media': d.loc[0, 'has_image'] = 2
    with pytest.raises(ValueError): validate_item_bank(d)


def test_time_limited_statuses(bank, monkeypatch):
    original = cp_model.CpSolver.solve
    def feasible(self, model):
        original(self, model)
        return cp_model.FEASIBLE
    monkeypatch.setattr(cp_model.CpSolver, 'solve', feasible)
    r = solve(bank)
    assert r['status'] == 'Feasible' and len(r['selected_forms'][0]) == 4
    assert not r['optimal']
    monkeypatch.setattr(cp_model.CpSolver, 'solve', lambda self, model: cp_model.UNKNOWN)
    assert solve(bank)['selected_forms'] == [[]]


def test_exposure_domain_override_and_empty_pool(bank):
    tracker = ItemUsageTracker(bank.iloc[:2], global_max=1, domain_max={'A': 2, 'B': 1})
    tracker.record_usage(['I0', 'I1'])
    assert tracker.get_eligible_pool().item_id.tolist() == ['I0']
    tracker.record_usage(['I0'])
    assert tracker.get_eligible_pool().empty
    assert list(tracker.get_eligible_pool().columns) == list(bank.columns)


def test_active_pool_stratification_and_common(bank):
    rules = {'test_length': 2, 'multiplier': 2, 'common_items': ['I0'],
             'domain_constraints': {'B': {'min': 1, 'max': 1}}}
    active = generate_active_pool(bank, rules)
    assert 'I0' in active.item_id.values
    assert sum(active.domain == 'B') >= 2
    assert not active.item_id.duplicated().any()


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def uploaded(data, name='bank.csv'):
    file = io.BytesIO(data)
    file.name = name
    return file


@pytest.fixture
def sample_bank():
    path = os.environ.get('LOFT_SAMPLE_BANK')
    if not path:
        pytest.skip('Set LOFT_SAMPLE_BANK to test with an external sample CSV.')
    return validate_item_bank(pd.read_csv(path, dtype={'item_id': str}))


def start_app(data, name='bank.csv'):
    # AppTest does not yet implement file-uploader input; all later interactions
    # use real Streamlit widget and session-state APIs.
    return patch.object(st.sidebar, 'file_uploader', side_effect=lambda *a, **k: uploaded(data, name))


def assert_forms(bank, forms, length=10, max_overlap=.3):
    for f in forms:
        ids = f['selected_items']
        assert len(ids) == len(set(ids)) == length
        s = bank[bank.item_id.isin(ids)]
        assert abs(s.rasch_b.mean()) <= .2
        for theta, target in [(-1., 1.5), (0., 2.), (1., 1.5)]:
            assert abs(calculate_tif(theta, s.rasch_b) - target) <= .2
        for row in s.itertuples():
            assert not (set(str(row.enemy_ids).split(',')) & (set(ids) - {row.item_id}))
        bins = auto_generate_difficulty_categories(bank, length)
        for info in bins.values():
            lo, hi = info['range']
            count = sum(s.rasch_b.between(lo, hi, inclusive='left'))
            assert info['min'] <= count <= info['max']
    for f1, f2 in itertools.combinations(forms, 2):
        a, b = set(f1['selected_items']), set(f2['selected_items'])
        assert len(a & b) / len(a | b) <= max_overlap


def test_sample_dashboard_run_step_reset_rerun_export(sample_bank):
    data = sample_bank.to_csv(index=False).encode()
    with start_app(data):
        app = AppTest.from_file('LOFT_squentialATA.py').run(timeout=30)
        assert not app.exception
        widget(app, 'number_input', 'Time Limit per Solve (seconds)').set_value(2).run()
        widget(app, 'button', '🚀 Run All Forms').click().run(timeout=60)
        assert not app.exception and not app.error
        forms = copy.deepcopy(app.session_state['cpsat_forms'])
        assert len(forms) == 3
        assert_forms(sample_bank, forms)
        assert len(app.session_state['cpsat_fig_tif'].data) == 3
        assert len(app.session_state['cpsat_fig_tcc'].data) == 3
        assert len(app.get('plotly_chart')) >= 5
        tracker = app.session_state['cpsat_tracker']
        assert tracker.get_exposure_stats()['max_exposure'] <= 2
        # Capture and verify the bytes passed to the real Excel download control.
        with patch('streamlit.download_button') as download:
            app.run()
        assert app.session_state['cpsat_forms'] == forms
        payload = download.call_args.kwargs['data']
        excel = pd.ExcelFile(io.BytesIO(payload))
        assert excel.sheet_names == ['Overview', 'Form 1', 'Form 2', 'Form 3']
        assert len(pd.read_excel(excel, 'Overview')) == 3
        for i, form in enumerate(forms, 1):
            assert set(pd.read_excel(excel, f'Form {i}').item_id) == set(form['selected_items'])
        widget(app, 'button', '🔄 Reset').click().run()
        assert app.session_state['cpsat_forms'] == []
        assert app.session_state['cpsat_tracker'].get_exposure_stats()['total_used'] == 0
        for count in (1, 2, 3):
            widget(app, 'button', '🔬 Step: Next Form').click().run(timeout=60)
            assert not app.exception and not app.error
            assert len(app.session_state['cpsat_forms']) == count
        widget(app, 'button', '🔬 Step: Next Form').click().run()
        assert len(app.session_state['cpsat_forms']) == 3
        # Changing a real configuration control clears stale results.
        widget(app, 'number_input', 'Mean B Target').set_value(.1).run()
        assert 'cpsat_forms' not in app.session_state


def test_sample_dashboard_combined_user_constraints(sample_bank):
    data = sample_bank.to_csv(index=False).encode()
    with start_app(data):
        app = AppTest.from_file('LOFT_squentialATA.py').run(timeout=30)
        widget(app, 'number_input', 'Pool Multiplier').set_value(10)
        widget(app, 'number_input', 'Default Exposure Limit').set_value(3)
        widget(app, 'number_input', 'TCC Target (Expected Score)').set_value(5.)
        widget(app, 'number_input', 'TCC Tol (+/-)').set_value(.2)
        widget(app, 'number_input', 'Min Image').set_value(1)
        widget(app, 'number_input', 'Max Image').set_value(1)
        widget(app, 'number_input', 'Time Limit per Solve (seconds)').set_value(2)
        common, excluded = sample_bank.item_id.iloc[:2]
        widget(app, 'text_input', 'Common Item IDs (comma-separated)').set_value(common)
        widget(app, 'text_input', 'Excluded Item IDs (comma-separated)').set_value(excluded)
        for domain in sample_bank.domain.unique():
            app.number_input(key=f'dmin_{domain}').set_value(1)
            app.number_input(key=f'dmax_{domain}').set_value(3)
        app.number_input(key='bmin_3. easy').set_value(1)
        app.number_input(key='bmin_6. hard').set_value(1)
        app.run()
        widget(app, 'button', '🚀 Run All Forms').click().run(timeout=60)
        assert not app.exception and not app.error
        forms = app.session_state['cpsat_forms']
        assert len(forms) == 3
        assert_forms(sample_bank, forms)
        for form in forms:
            s = sample_bank[sample_bank.item_id.isin(form['selected_items'])]
            assert common in s.item_id.values and excluded not in s.item_id.values
            assert s.has_image.sum() == 1
            assert abs(calculate_tcc(0, s.rasch_b) - 5.) <= .2
            assert all(s.domain.value_counts().reindex(sample_bank.domain.unique(), fill_value=0).between(1, 3))
            assert {'3. easy', '6. hard'}.issubset(set(s.raschb_cat))


@pytest.mark.parametrize('extension', ['csv', 'xlsx', 'xls'])
def test_dashboard_input_formats(bank, extension):
    if extension == 'csv': data = bank.to_csv(index=False).encode()
    elif extension == 'xlsx':
        buf = io.BytesIO()
        bank.to_excel(buf, index=False)
        data = buf.getvalue()
    else:
        xlwt = pytest.importorskip('xlwt')
        book = xlwt.Workbook()
        sheet = book.add_sheet('Bank')
        for r, row in enumerate([bank.columns.tolist()] + bank.values.tolist()):
            for c, value in enumerate(row): sheet.write(r, c, value)
        buf = io.BytesIO()
        book.save(buf)
        data = buf.getvalue()
    with start_app(data, f'BANK.{extension.upper()}'):
        app = AppTest.from_file('LOFT_squentialATA.py').run(timeout=30)
        assert not app.exception and not app.error
        assert any('Bank Loaded: 16 items' in s.value for s in app.success)


def test_dashboard_invalid_upload_is_friendly():
    with start_app(b'wrong_column\n1\n'):
        app = AppTest.from_file('LOFT_squentialATA.py').run(timeout=30)
        assert not app.exception
        assert any('Missing required columns' in e.value for e in app.error)


def test_dashboard_impossible_overlap_does_not_record_usage(bank):
    # Four common items fill each form. A zero-overlap second form is impossible.
    with start_app(bank.to_csv(index=False).encode()):
        app = AppTest.from_file('LOFT_squentialATA.py').run(timeout=30)
        widget(app, 'number_input', 'Test Length').set_value(4)
        widget(app, 'number_input', '# Forms').set_value(2)
        widget(app, 'checkbox', 'Enable auto-difficulty bins').uncheck()
        widget(app, 'checkbox', 'Apply Enemy Item Constraints').uncheck()
        widget(app, 'number_input', 'TIF Target').set_value(0.)
        widget(app, 'number_input', 'Mean B Tol').set_value(3.)
        widget(app, 'number_input', 'Maximum Sampling Attempts per Form').set_value(1)
        widget(app, 'slider', 'Max Jaccard Overlap').set_value(0.)
        widget(app, 'text_input', 'Common Item IDs (comma-separated)').set_value('I0,I1,I2,I3')
        app.run()
        widget(app, 'button', '🚀 Run All Forms').click().run(timeout=30)
        assert not app.exception
        assert len(app.session_state['cpsat_forms']) == 1
        assert app.session_state['cpsat_tracker'].get_exposure_stats()['max_exposure'] == 1
        assert app.error
