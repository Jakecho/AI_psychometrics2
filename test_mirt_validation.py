"""R comparison calculations, subprocess validation, and dashboard integration."""

from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mhrm_engine import marginal_loglikelihood, simulate_data
from mirt_validation import check_mirt, compare_estimates, fit_mirt


def test_agreement_excludes_fixed_zeros_and_uses_common_likelihood():
    y = np.array([[1, 0], [0, 1]])
    q = np.eye(2)
    a, d = np.diag([1., 2.]), np.array([.1, -.2])
    reference = dict(a=np.diag([1.1, 1.8]), d=np.array([.2, -.1]), native_loglik=-999999.)
    result = compare_estimates(y, q, a, d, reference, a, d)
    slopes = result['summary'].iloc[0]
    assert slopes.Count == 2
    assert slopes.RMSE == pytest.approx(np.sqrt((.1**2+.2**2)/2))
    assert slopes.MAE == pytest.approx(.15)
    assert slopes.Mean_difference == pytest.approx(.05)
    assert len(result['per_item']) == 6
    assert sum(result['per_item'].Active) == 4
    expected_r, _ = marginal_loglikelihood(y, reference['a'], reference['d'])
    assert result['likelihood'].iloc[1].Common_integrated_loglik == expected_r
    assert result['likelihood'].iloc[1].Common_integrated_loglik != reference['native_loglik']
    assert result['recovery'].query("Estimator == 'Python'").RMSE.eq(0).all()


def test_optional_factor_labels_rename_comparison_rows():
    y = np.array([[1, 0], [0, 1]])
    a, d = np.diag([1., 2.]), np.array([.1, -.2])
    result = compare_estimates(y, np.eye(2), a, d, dict(a=a, d=d), factor_labels=['G', 'S_1'])
    assert set(result['per_item'].Parameter) == {'a(G)', 'a(S_1)', 'd'}
    assert list(result['summary'].Parameter[2:]) == ['Active a(G)', 'Active a(S_1)']
    default = compare_estimates(y, np.eye(2), a, d, dict(a=a, d=d))
    assert list(default['summary'].Parameter[2:]) == ['Active a1', 'Active a2']
    with pytest.raises(ValueError, match='one factor label'):
        compare_estimates(y, np.eye(2), a, d, dict(a=a, d=d), factor_labels=['G'])


def test_constant_coefficients_have_undefined_correlation():
    y = np.array([[1, 0], [0, 1]])
    a, d = np.ones((2, 1)), np.zeros(2)
    result = compare_estimates(y, np.ones((2, 1)), a, d, dict(a=a, d=d))
    assert result['summary'].Correlation.isna().all()


@pytest.mark.parametrize('kwargs', [{'cycles': 0}, {'mh_steps': 0}, {'decay': .5},
                                   {'gain_scale': 0}, {'seed': -1}, {'tolerance': float('nan')}])
def test_invalid_settings_do_not_launch_r(kwargs):
    with pytest.raises(ValueError): fit_mirt(np.eye(2), np.eye(2), **kwargs)


def test_missing_r_is_friendly(monkeypatch):
    monkeypatch.setattr('mirt_validation.shutil.which', lambda _: None)
    ready, message = check_mirt()
    assert not ready and 'not installed' in message
    with pytest.raises(RuntimeError, match='Rscript'):
        fit_mirt(np.eye(2), np.eye(2))


def test_timeout_is_friendly(monkeypatch):
    monkeypatch.setattr('mirt_validation.shutil.which', lambda _: '/fake/Rscript')
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr('mirt_validation.subprocess.run', timeout)
    with pytest.raises(RuntimeError, match='exceeded'):
        fit_mirt(np.eye(2), np.eye(2), timeout=1)


@pytest.mark.parametrize('defect', [None, 'latent', 'mask', 'nan', 'bad_flag'])
def test_reference_output_is_validated_and_temporary_files_removed(monkeypatch, defect):
    monkeypatch.setattr('mirt_validation.shutil.which', lambda _: '/fake/Rscript')
    roots = []
    def run(command, **kwargs):
        root = Path(command[3]); roots.append(root)
        assert kwargs.get('shell', False) is False
        a = np.eye(2)
        if defect == 'mask': a[0, 1] = .1
        if defect == 'nan': a[0, 0] = np.nan
        pd.DataFrame(dict(a1=a[:, 0], a2=a[:, 1], d=[0., 0.])).to_csv(root/'estimates.csv', index=False)
        pd.DataFrame({'mean': [0., .2 if defect == 'latent' else 0.]}).to_csv(root/'means.csv', index=False)
        pd.DataFrame(np.eye(2)).to_csv(root/'covariance.csv', index=False)
        pd.DataFrame([dict(converged='maybe' if defect == 'bad_flag' else 'FALSE',
            stage3_iterations=500, native_loglik='NA', mirt_version='1.46.1', r_version='4.4.3')]).to_csv(root/'metadata.csv', index=False)
        (root/'model.txt').write_text('F1=1\nF2=2\n')
        return subprocess.CompletedProcess(command, 0, '', 'Reference reached its cycle limit.')
    monkeypatch.setattr('mirt_validation.subprocess.run', run)
    if defect:
        with pytest.raises(RuntimeError): fit_mirt(np.eye(2), np.eye(2))
    else:
        result = fit_mirt(np.eye(2), np.eye(2))
        assert not result['converged']
        assert np.isnan(result['native_loglik'])
        assert result['messages']
    assert roots and not any(root.exists() for root in roots)


@pytest.fixture(scope='module')
def r_installed():
    ready, version = check_mirt()
    if not ready: pytest.skip(version)
    return version


@pytest.mark.parametrize('design', ['simple_1d', 'anchored_cross', 'bifactor'])
def test_real_r_reference_supports_all_designs(r_installed, design):
    if design == 'simple_1d':
        q = np.ones((9, 1))
    elif design == 'anchored_cross':
        q = np.r_[np.repeat(np.eye(2), 3, axis=0), np.ones((6, 2))]
    else:
        q = np.column_stack([np.ones(15), np.repeat(np.eye(3), 5, axis=0)])
    y, _, _, _ = simulate_data(700, len(q), q.shape[1], q, seed=123)
    result = fit_mirt(y, q, cycles=80, burn_in=30, sem_cycles=30, seed=124,
                      gain_scale=.2, decay=.8, mh_steps=4, tolerance=.002)
    assert result['mirt_version'] == r_installed
    assert isinstance(result['converged'], bool)
    assert result['a'].shape == q.shape
    assert np.all(result['a'][q == 0] == 0)
    assert result['settings']['gain_scale'] == .2
    assert result['settings']['decay'] == .8


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def quick_app(seed):
    app = AppTest.from_file('MHRM_2D.py').run(timeout=30)
    widget(app, 'slider', 'Sample Size (N)').set_value(500)
    widget(app, 'slider', 'Number of Items (J)').set_value(10)
    widget(app, 'slider', 'Maximum RM Cycles (Stage 3)').set_value(50)
    widget(app, 'slider', 'Burn-in Cycles (Stage 1)').set_value(10)
    widget(app, 'slider', 'SEM Averaging Cycles (Stage 2)').set_value(20)
    widget(app, 'number_input', 'Random Seed').set_value(seed)
    app.run(timeout=30)
    return app


def test_dashboard_real_outputs_csv_and_persistence(r_installed):
    app = quick_app(877)
    assert widget(app, 'checkbox', 'Cross-check with R mirt').value
    app.button[0].click().run(timeout=120)
    assert not app.exception and not app.error
    assert any(s.value == 'Step 7 · Cross-check against R mirt output' for s in app.subheader)
    assert any(m.label == 'Integrated log-likelihood difference (Python − R)' for m in app.metric)
    assert any('provisional' in w.value for w in app.warning)
    saved = app.session_state['mhrm_saved_run']['model'].est_a.copy()
    with patch('streamlit.download_button') as download:
        app.run(timeout=30)
    assert not app.exception and not app.error
    np.testing.assert_array_equal(saved, app.session_state['mhrm_saved_run']['model'].est_a)
    assert download.call_count == 3
    payload = next(c.args[1] for c in download.call_args_list if c.args[2] == 'mhrm_python_r_items.csv')
    import io
    exported = pd.read_csv(io.StringIO(payload))
    assert {'Item', 'Parameter', 'Active', 'Python', 'R_mirt', 'Python_minus_R', 'True'} <= set(exported.columns)
    assert len(exported) == 30
    widget(app, 'checkbox', 'Cross-check with R mirt').uncheck().run()
    np.testing.assert_array_equal(saved, app.session_state['mhrm_saved_run']['model'].est_a)
    assert not any('Cross-check against R mirt' in s.value for s in app.subheader)


def test_reference_failure_preserves_python_results(r_installed):
    app = quick_app(889)
    with patch('mirt_validation.fit_mirt', side_effect=RuntimeError('Deliberate R failure')):
        app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any('Deliberate R failure' in e.value for e in app.error)
    assert app.session_state['mhrm_saved_run']['model'].rm_iterations_ == 50
    assert any(s.value == 'Step 4 · Parameter recovery' for s in app.subheader)
