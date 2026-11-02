"""Mathematical, simulation, optional R reference, and Streamlit regressions."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from streamlit.testing.v1 import AppTest

from mhrm_engine import (MHRM_MIRT, identification_issues, log_likelihood,
                         marginal_loglikelihood, prob_m2pl, simulate_data)


@pytest.fixture
def small_model():
    q = np.r_[np.repeat(np.eye(2), 3, axis=0), [[1, 1]]]
    y = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0],
                  [1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
    model = MHRM_MIRT(y, 2, q)
    model.current_theta = np.array([[.8, -.2], [-.6, .7], [.4, .3], [-.1, -.9]])
    return model


def test_full_scores_and_curvature_match_numerical_derivatives(small_model):
    m = small_model
    score, info = m.complete_data_derivatives()
    eps = 1e-5
    base = m._parameters()
    for item in range(m.J):
        for k in range(len(base[item])):
            plus, minus = [x.copy() for x in base], [x.copy() for x in base]
            plus[item][k] += eps
            minus[item][k] -= eps
            m._set_parameters(plus)
            upper = log_likelihood(m.Y, m.current_theta, m.est_a, m.est_d) / m.N
            g_plus = m.complete_data_derivatives()[0][item]
            m._set_parameters(minus)
            lower = log_likelihood(m.Y, m.current_theta, m.est_a, m.est_d) / m.N
            g_minus = m.complete_data_derivatives()[0][item]
            assert (upper-lower)/(2*eps) == pytest.approx(score[item][k], abs=1e-9)
            np.testing.assert_allclose(-(g_plus-g_minus)/(2*eps), info[item][:, k], atol=1e-9)
            m._set_parameters(base)
    assert abs(info[-1][0, 1]) > 1e-4  # slope--slope term is retained
    assert abs(info[-1][0, 2]) > 1e-4  # slope--intercept term is retained


def test_hand_calculated_update_and_posterior_ratio():
    m = MHRM_MIRT(np.array([[1, 0], [0, 1]]), 2, np.eye(2), check_identification=False)
    m.est_a, m.est_d = np.eye(2), np.zeros(2)
    proposal = np.array([[.5, -.5], [-.5, .5]])
    ratio = np.exp(m._row_log_posterior(proposal) - m._row_log_posterior(np.zeros((2, 2))))
    np.testing.assert_allclose(ratio, 1.20700296, rtol=1e-7)
    m.current_theta = proposal
    m.rm_step(.1)
    np.testing.assert_allclose(np.diag(m.est_a), 1.32130613, rtol=1e-7)
    np.testing.assert_allclose(m.est_d, 0, atol=1e-14)
    assert m.est_a[0, 1] == m.est_a[1, 0] == 0


def test_smoothed_curvature_is_used(small_model):
    m = small_model
    m.curvature_ = [2*np.eye(len(v)) for v in m._parameters()]
    scores = [np.full(len(v), .2) for v in m._parameters()]
    information = [4*np.eye(len(v)) for v in m._parameters()]
    before = m._parameters()
    m.rm_step(.25, scores, information)
    for old, new, block in zip(before, m._parameters(), m.curvature_):
        np.testing.assert_allclose(block, 2.5*np.eye(len(old)))
        np.testing.assert_allclose(new-old, .02)  # .25 * (.2 / 2.5)


def test_full_newton_update_matches_conditional_logistic_mle():
    q = np.r_[np.repeat(np.eye(2), 3, axis=0), [[1, 1]]]
    y, theta, _, _ = simulate_data(800, 7, 2, q, seed=23)
    m = MHRM_MIRT(y, 2, q)
    m.current_theta = theta
    for _ in range(30):
        m.rm_step(1.)
    for j, active in enumerate(m.active):
        design = np.column_stack([theta[:, active], np.ones(len(theta))])
        def objective(beta):
            eta = design @ beta
            return np.mean(np.logaddexp(0., eta)-y[:, j]*eta)
        oracle = minimize(objective, np.r_[np.ones(len(active)), 0.], method='L-BFGS-B',
                          bounds=[(0., 10.)]*len(active)+[(None, None)],
                          options={'ftol': 1e-14, 'gtol': 1e-9})
        assert oracle.success
        np.testing.assert_allclose(m._parameters()[j], oracle.x, atol=2e-5)


def test_stage_lengths_averaging_gain_and_frozen_proposal(small_model):
    m = small_model
    stage2 = []
    def callback(model, diagnostic):
        if diagnostic['iteration'] == 5:
            stage2.extend(v.copy() for v in model._parameters())
    m.fit(max_iter=4, burn_in=2, sem_cycles=3, callback=callback)
    assert [r['stage'] for r in m.diagnostics] == ['Burn-in']*2 + ['SEM']*3 + ['RM']*4
    for got, expected in zip(stage2, m.sem_parameter_mean_):
        np.testing.assert_allclose(got, expected)
    gains = [r['gain'] for r in m.diagnostics if r['stage'] == 'RM']
    np.testing.assert_allclose(gains, (.1/np.arange(1, 5))**.75)
    assert len(set(r['proposal_sd'] for r in m.diagnostics[1:])) == 1
    assert m.n_iter_ == 9 and not m.converged
    assert m.rm_iterations_ == 4 and m.sem_curvature_mean_ is not None


def test_small_updates_alone_do_not_claim_convergence(small_model, monkeypatch):
    m = small_model
    def derivatives():
        return ([np.full(len(v), .2) for v in m._parameters()],
                [np.eye(len(v))*1e8 for v in m._parameters()])
    monkeypatch.setattr(m, 'complete_data_derivatives', derivatives)
    m.fit(max_iter=55, burn_in=2, sem_cycles=2)
    assert m.diagnostics[-1]['max_change'] < .001
    assert m.diagnostics[-1]['mean_projected_score'] > .005
    assert not m.converged


def test_mh_posterior_moment_matches_quadrature():
    m = MHRM_MIRT(np.array([[1, 0], [0, 1]]), 2, np.eye(2), check_identification=False, seed=91)
    m.est_a, m.est_d = np.eye(2), np.zeros(2)
    m.mh_step(500, candidate_sd=1.)
    samples = []
    for _ in range(6000):
        m.mh_step(2, candidate_sd=1.)
        samples.append(m.current_theta[0].copy())
    expected = 2 * quad(lambda t: t * expit(t) * norm.pdf(t), -10, 10, epsabs=1e-11)[0]
    np.testing.assert_allclose(np.mean(samples, axis=0), [expected, -expected], atol=.07)


def test_integrated_likelihood_matches_independent_quadrature():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    a, d = np.array([[.8], [1.1], [1.4]]), np.array([-.3, .1, .7])
    expected = 0.
    for pattern in y:
        def integrand(t):
            p = expit(a[:, 0]*t+d)
            return np.prod(np.where(pattern, p, 1-p)) * norm.pdf(t)
        expected += np.log(quad(integrand, -12, 12, epsabs=1e-12)[0])
    actual, method = marginal_loglikelihood(y, a, d, points_per_dim=61)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert 'Gauss-Hermite' in method


def test_qmc_likelihood_reproducible_and_finite():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    result = marginal_loglikelihood(y, np.eye(3), np.zeros(3))
    assert result == marginal_loglikelihood(y, np.eye(3), np.zeros(3))
    assert result[0] == pytest.approx(6*np.log(.5), abs=.003)
    assert 'Sobol QMC approximation' in result[1]


def test_local_random_streams_and_fixed_zeros():
    q = np.repeat(np.eye(2), 3, axis=0)
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)
    one = simulate_data(100, 6, 2, q)
    two = simulate_data(100, 6, 2, q)
    assert np.random.random() == expected
    for a, b in zip(one, two): np.testing.assert_array_equal(a, b)
    m = MHRM_MIRT(one[0], 2, q, seed=8).fit(max_iter=10, burn_in=2, sem_cycles=3)
    n = MHRM_MIRT(one[0], 2, q, seed=8).fit(max_iter=10, burn_in=2, sem_cycles=3)
    np.testing.assert_array_equal(m.est_a, n.est_a)
    assert np.all(m.est_a[q == 0] == 0)
    assert np.all((m.est_a >= 0) & (m.est_a <= 10))


@pytest.mark.parametrize('kwargs', [{'decay': .5}, {'decay': 1.1}, {'burn_in': 0},
    {'sem_cycles': 0}, {'max_iter': 0}, {'gain_scale': 0}, {'tolerance': float('nan')}])
def test_invalid_schedule_rejected(small_model, kwargs):
    with pytest.raises(ValueError): small_model.fit(**kwargs)


def test_unidentified_and_invalid_data_rejected():
    assert any('identical' in s for s in identification_issues(np.ones((6, 2))))
    y = np.tile([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], (5, 1))
    with pytest.raises(ValueError, match='identical'): MHRM_MIRT(y, 2, np.ones((6, 2)))
    with pytest.raises(ValueError, match='constant'): MHRM_MIRT(np.ones((10, 6)), 2, np.repeat(np.eye(2), 3, axis=0))
    y = y.astype(float); y[0, 0] = np.nan
    with pytest.raises(ValueError, match='complete binary'): MHRM_MIRT(y, 2, np.repeat(np.eye(2), 3, axis=0))


def _r_env():
    env = os.environ.copy()
    local = Path.home()/'.cache/ai_psychometrics2/R-library'
    if local.is_dir(): env['MIRT_R_LIBRARY'] = str(local)
    return env


@pytest.fixture(scope='module')
def benchmark():
    q = np.repeat(np.eye(2), 8, axis=0)
    y, _, a, d = simulate_data(1200, 16, 2, q, seed=2026)
    m = MHRM_MIRT(y, 2, q, seed=2027).fit(max_iter=600)
    return y, q, a, d, m


def test_parameter_recovery(benchmark):
    _, q, a, d, m = benchmark
    assert m.converged
    assert np.sqrt(np.mean((m.est_a[q == 1]-a[q == 1])**2)) < .25
    assert np.sqrt(np.mean((m.est_d-d)**2)) < .15


def test_against_real_r_mirt(benchmark, tmp_path):
    executable = shutil.which('Rscript')
    if not executable: pytest.skip('Rscript is not installed.')
    probe = "p<-Sys.getenv('MIRT_R_LIBRARY');if(nzchar(p)).libPaths(c(p,.libPaths()));quit(status=if(requireNamespace('mirt',quietly=TRUE))0 else 2)"
    ready = subprocess.run([executable, '--vanilla', '-e', probe], env=_r_env(), capture_output=True, timeout=30)
    if ready.returncode: pytest.skip('R mirt is not installed.')
    y, q, _, _, m = benchmark
    pd.DataFrame(y).to_csv(tmp_path/'responses.csv', index=False)
    pd.DataFrame(q).to_csv(tmp_path/'q.csv', index=False)
    script = Path(__file__).parent/'tests/mhrm_reference.R'
    result = subprocess.run([executable, '--vanilla', str(script), str(tmp_path), '600', '150', '100', '2027'],
                            env=_r_env(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    estimates = pd.read_csv(tmp_path/'estimates.csv')
    metadata = pd.read_csv(tmp_path/'metadata.csv').iloc[0]
    assert metadata.converged
    a_r, d_r = estimates[['a1', 'a2']].to_numpy(), estimates.d.to_numpy()
    slope_difference = float(np.sqrt(np.mean((m.est_a[q == 1]-a_r[q == 1])**2)))
    intercept_difference = float(np.sqrt(np.mean((m.est_d-d_r)**2)))
    ll_python, _ = marginal_loglikelihood(y, m.est_a, m.est_d, points_per_dim=41)
    ll_r, _ = marginal_loglikelihood(y, a_r, d_r, points_per_dim=41)
    assert slope_difference < .1
    assert intercept_difference < .05
    assert abs(ll_python-ll_r) < 2.
    report = dict(mirt_version=str(metadata.mirt_version), N=len(y), J=y.shape[1],
        python_converged=m.converged, python_rm_cycles=m.rm_iterations_,
        slope_rmse_between_estimators=slope_difference,
        intercept_rmse_between_estimators=intercept_difference,
        python_integrated_loglik=ll_python, r_integrated_loglik=ll_r)
    print('\nR reference comparison:', json.dumps(report))


def _widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


@pytest.mark.parametrize('structure, dims', [('Simple', 2), ('Cross-load', 1), ('Cross-load', 2), ('Bifactor', 3)])
def test_dashboard_calibrates_supported_structures(structure, dims):
    app = AppTest.from_file('MHRM_2D.py').run(timeout=30)
    _widget(app, 'slider', 'Sample Size (N)').set_value(500)
    _widget(app, 'slider', 'Number of Items (J)').set_value(15)
    _widget(app, 'slider', 'Dimensions (Specific Factors)').set_value(dims)
    _widget(app, 'slider', 'Maximum RM Cycles (Stage 3)').set_value(50)
    _widget(app, 'slider', 'Burn-in Cycles (Stage 1)').set_value(10)
    _widget(app, 'slider', 'SEM Averaging Cycles (Stage 2)').set_value(20)
    app.sidebar.radio[0].set_value(structure).run()
    app.button[0].click().run(timeout=120)
    assert not app.exception and not app.error
    assert any('marginal log-likelihood' in m.label.lower() for m in app.metric)
    assert any(s.value == 'Step 6 · Model fit: log-likelihood' for s in app.subheader)
    assert [s.value.split(' · ')[0] for s in app.subheader if s.value.startswith('Step')][:6] == [f'Step {k}' for k in range(1, 7)]
    diagnostic = app.dataframe[0].value
    assert set(diagnostic.stage) == {'Burn-in', 'SEM', 'RM'}
    assert len(diagnostic) == 80
    assert any('provisional' in w.value for w in app.warning)


def test_dashboard_bifactor_slider_starts_at_supported_design():
    # Bifactor needs >= 3 specific factors, so its slider starts (and bottoms out) at 3.
    app = AppTest.from_file('MHRM_2D.py').run(timeout=30)
    assert _widget(app, 'slider', 'Dimensions (Specific Factors)').value == 2
    app.sidebar.radio[0].set_value('Bifactor').run()
    slider = _widget(app, 'slider', 'Dimensions (Specific Factors)')
    assert slider.value == 3 and slider.min == 3
    _widget(app, 'slider', 'Sample Size (N)').set_value(500)
    _widget(app, 'slider', 'Maximum RM Cycles (Stage 3)').set_value(50)
    _widget(app, 'slider', 'Burn-in Cycles (Stage 1)').set_value(10)
    _widget(app, 'slider', 'SEM Averaging Cycles (Stage 2)').set_value(20).run()
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert not any('at least three specific factors' in e.value for e in app.error)
    summary = next(df.value for df in app.dataframe if 'Dimension' in df.value.columns)
    assert list(summary.Dimension) == ['General (G)', 'Specific (S_1)', 'Specific (S_2)', 'Specific (S_3)', '–']
