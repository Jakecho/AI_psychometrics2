"""Numerical tests for the Rasch model, JMLE, CMLE, MML-EM and person scoring."""
from itertools import product

import numpy as np
import pytest
from scipy.special import expit, logsumexp

from conftest import requires_erm, requires_tam
from r_bridge.bridge import run_reference
from rasch.cmle import _cll_and_grad, _Data, cmle, demonstration, log_esf, log_esf_leave_one_out
from rasch.jmle import JMLESettings, jmle, walkthrough
from rasch.model import information, joint_loglik, probability
from rasch.model import test_information as total_information
from rasch.mml_em import MMLSettings, e_step, estep_walkthrough, m_step_items, marginal_loglik, mml_em, prior_weights
from rasch.scoring import _parts, mle_score, newton_trace, score_persons, score_table, wle_score
from rasch.simulation import connectivity_issues, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(400, 12, seed=7)


@pytest.fixture(scope="module")
def sim_missing():
    return simulate(400, 12, missing=0.15, seed=8)


# ---------------- Rasch model ----------------
def test_probability_information_and_shift_invariance():
    assert probability([0.0], [0.0])[0, 0] == 0.5
    np.testing.assert_allclose(probability([1.0], [0.0]), [[np.e / (1 + np.e)]])
    th, b = np.array([-1.2, 0.3, 2.0]), np.array([-0.5, 0.1, 1.4, 2.2])
    np.testing.assert_allclose(probability(th, b), probability(th + 1.7, b + 1.7))
    p = probability(th, b)
    np.testing.assert_allclose(information(th, b), p * (1 - p))
    np.testing.assert_allclose(total_information(th, b), (p * (1 - p)).sum(axis=1))
    # expected score is the row sum of probabilities
    np.testing.assert_allclose(p.sum(axis=1), [expit(t - b).sum() for t in th])


def test_joint_loglik_skips_missing():
    X = np.array([[1, np.nan], [0, 1.0]])
    th, b = np.array([0.2, -0.3]), np.array([0.1, -0.4])
    p = probability(th, b)
    expected = np.log(p[0, 0]) + np.log(1 - p[1, 0]) + np.log(p[1, 1])
    assert joint_loglik(X, th, b) == pytest.approx(expected)


def test_simulation_reproducible_and_missing(sim):
    a, b = simulate(100, 8, seed=3, missing=.1), simulate(100, 8, seed=3, missing=.1)
    np.testing.assert_array_equal(np.nan_to_num(a.X, nan=-1), np.nan_to_num(b.X, nan=-1))
    assert a.identity == b.identity and simulate(100, 8, seed=4).identity != a.identity
    assert 0.05 < np.isnan(a.X).mean() < 0.15 and not np.isnan(a.X).all(axis=1).any()
    assert connectivity_issues(sim) == []


# ---------------- JMLE ----------------
def test_jmle_score_equations_hold_at_solution(sim):
    # Without extreme scores the adjustment is inactive and both sets of equations hold exactly.
    x = np.nan_to_num(sim.X)
    keep = (x.sum(axis=1) > 0) & (x.sum(axis=1) < sim.n_items)
    X = sim.X[keep]
    r = jmle(X, JMLESettings(bias=False, tol=1e-10))
    p = probability(r.theta, r.b)
    np.testing.assert_allclose(p.sum(axis=1), r.adjusted_person_scores, atol=1e-6)
    np.testing.assert_allclose(p.sum(axis=0), r.adjusted_item_scores, atol=1e-6)
    assert abs(r.theta.mean()) < 1e-10 and r.converged


def test_jmle_adjusted_totals_explain_residuals(sim):
    # With adjusted extreme persons, person and item totals differ, so no point satisfies every
    # equation exactly; the item equations hold and the person residuals absorb the difference.
    r = jmle(sim.X, JMLESettings(bias=False, tol=1e-10))
    p = probability(r.theta, r.b)
    np.testing.assert_allclose(p.sum(axis=0), r.adjusted_item_scores, atol=1e-6)
    gap = r.adjusted_person_scores.sum() - r.adjusted_item_scores.sum()
    assert (r.adjusted_person_scores - p.sum(axis=1)).sum() == pytest.approx(gap, abs=1e-5)


def test_jmle_identification_and_bias_factor(sim):
    cases = jmle(sim.X, JMLESettings(bias=False, constraint="cases", tol=1e-10))
    items = jmle(sim.X, JMLESettings(bias=False, constraint="items", tol=1e-10))
    # Same solution up to a common shift.
    np.testing.assert_allclose(cases.b - cases.b.mean(), items.b, atol=1e-6)
    np.testing.assert_allclose(cases.theta - cases.b.mean(), items.theta, atol=1e-6)
    corrected = jmle(sim.X, JMLESettings(bias=True, tol=1e-10))
    np.testing.assert_allclose(corrected.b, cases.b * (sim.n_items - 1) / sim.n_items, atol=1e-9)


def test_jmle_update_direction_and_walkthrough(sim):
    r = jmle(sim.X)
    w = walkthrough(sim.X, r, 1, 0, 0)
    # Person step moves toward the observed score; item step toward the item score.
    assert np.sign(w["person_step"]) == np.sign(w["person_residual"])
    assert np.sign(w["item_step"]) == np.sign(w["item_residual"])
    assert w["theta_first_step"] == pytest.approx(w["theta_in"] + w["person_residual"] / w["person_info"])


def test_jmle_extreme_scores_need_adjustment():
    d = simulate(200, 6, theta_sd=2.5, seed=1)
    with pytest.raises(ValueError, match="infinite"):
        jmle(d.X, JMLESettings(adj=0.0))
    r = jmle(d.X)
    assert np.isfinite(r.theta).all() and r.n_extreme_persons > 0


# ---------------- CMLE ----------------
def test_esf_matches_brute_force_and_polynomial():
    b = np.array([-1.1, -0.2, 0.4, 0.9, 1.6])
    eps = np.exp(-b)
    gamma = np.exp(log_esf(-b))
    brute = np.zeros(6)
    for pat in product([0, 1], repeat=5):
        brute[sum(pat)] += np.prod(eps ** np.array(pat))
    np.testing.assert_allclose(gamma, brute)
    np.testing.assert_allclose(demonstration(b, 2)["polynomial"], brute)
    loo = np.exp(log_esf_leave_one_out((-b)[None, :]))[0]
    for j in range(5):
        np.testing.assert_allclose(loo[j, :5], np.exp(log_esf(np.delete(-b, j))))


def test_conditional_probabilities_normalise():
    for r in range(0, 6):
        rows = demonstration(np.array([-1.0, -.3, .2, .8, 1.5]), r)["rows"]
        assert sum(row["conditional_probability"] for row in rows) == pytest.approx(1.0)


def test_cmle_gradient_matches_finite_difference(sim_missing):
    d = _Data(sim_missing.X)
    b = np.linspace(-1, 1, d.J)
    ll, g = _cll_and_grad(b, d)
    h = 1e-6
    num = np.array([(_cll_and_grad(b + h * e, d)[0] - _cll_and_grad(b - h * e, d)[0]) / (2 * h) for e in np.eye(d.J)])
    np.testing.assert_allclose(g, num, atol=1e-4)


def test_cmle_missing_data_matches_brute_force_conditional_likelihood():
    d = simulate(60, 5, missing=.2, seed=11)
    b = np.array([-0.8, -0.1, 0.3, 0.6, 1.2])
    expected = 0.0
    for i in range(d.n_persons):
        obs = ~np.isnan(d.X[i])
        x = d.X[i, obs]
        r = x.sum()
        if r in (0, obs.sum()):
            continue
        bo = b[obs]
        denom = sum(np.exp(-np.dot(p, bo)) for p in product([0, 1], repeat=len(bo)) if sum(p) == r)
        expected += -x @ bo - np.log(denom)
    assert _cll_and_grad(b, _Data(d.X))[0] == pytest.approx(expected, rel=1e-10)


def test_cmle_sum_zero_extremes_and_convergence(sim):
    r = cmle(sim.X)
    assert abs(r.b.sum()) < 1e-10 and r.converged and np.isfinite(r.se_b).all()
    assert r.n_informative + r.n_extreme == sim.n_persons
    assert r.history[-1]["cond_loglik"] >= r.history[0]["cond_loglik"]


# ---------------- MML-EM ----------------
def test_estep_posteriors_and_weights(sim_missing):
    from rasch.model import observed_values
    x, mask = observed_values(sim_missing.X)
    nodes = np.linspace(-6, 6, 21)
    w = prior_weights(nodes, 1.3)
    assert w.sum() == pytest.approx(1.0)
    b = np.linspace(-1, 1, sim_missing.n_items)
    tau, ll, node_ll, w2 = e_step(x, mask.astype(float), b, nodes, 1.3)
    np.testing.assert_allclose(tau.sum(axis=1), 1.0)
    # Direct posterior for person 0 (observed items only).
    i = 0
    o = mask[i]
    lik = np.array([np.prod(np.where(x[i, o] == 1, expit(q - b[o]), 1 - expit(q - b[o]))) for q in nodes])
    np.testing.assert_allclose(tau[i], lik * w / (lik * w).sum(), rtol=1e-10)
    assert ll == pytest.approx(float(np.sum(logsumexp(node_ll + np.log(w), axis=1))))
    # Expected counts conserve exposure.
    N = mask.astype(float).T @ tau
    np.testing.assert_allclose(N.sum(axis=1), mask.sum(axis=0))


def test_mstep_increases_expected_loglik_and_em_monotone(sim):
    r = mml_em(sim.X)
    lls = [h["loglik"] for h in r.history]
    assert all(b >= a - 1e-8 for a, b in zip(lls, lls[1:]))
    assert r.converged
    w = estep_walkthrough(sim.X, r, 1, 0, 0)
    N, R = w["N_item"], w["R_item"]
    nodes = r.settings.nodes

    def q(bj):
        p = expit(nodes - bj)
        return float((R * np.log(p) + (N - R) * np.log(1 - p)).sum())
    first = w["m_trace"][0]["b"]
    assert q(r.history[0]["b"][0]) >= q(first) - 1e-12
    # The M-step root satisfies its score equation.
    b_new, _ = m_step_items(N[None, :], R[None, :], np.array([first]), nodes, 50)
    p = expit(nodes - b_new[0])
    assert abs((N * p - R).sum()) < 1e-8


def test_mml_fixed_variance_option(sim):
    r = mml_em(sim.X, MMLSettings(estimate_variance=False, fixed_sd=1.0))
    assert r.sd == 1.0 and np.isnan(r.se_variance)


# ---------------- Person scoring ----------------
def test_wle_solves_adjusted_equation_and_is_finite_for_extremes():
    b = np.linspace(-1.5, 1.5, 8)
    table = score_table(b, "WLE")
    assert np.isfinite(table[:, 1]).all()
    for r, t, _ in table:
        assert abs(wle_score(t, r, b)) < 1e-8
    mle = score_table(b, "MLE")
    assert np.isinf(mle[0, 1]) and np.isinf(mle[-1, 1])
    for r, t, _ in mle[1:-1]:
        assert abs(mle_score(t, r, b)) < 1e-8
    # WLE is pulled toward the centre relative to MLE.
    inner = slice(1, -1)
    assert np.all(np.abs(table[inner, 1]) <= np.abs(mle[inner, 1]) + 1e-12)


def test_newton_trace_converges_to_root():
    b = np.linspace(-1, 1, 6)
    rows = newton_trace(4, b, "WLE")
    assert abs(rows[-1]["update"]) < 1e-8
    assert abs(wle_score(rows[-1]["theta_next"], 4, b)) < 1e-8


def test_wle_sign_of_adjustment():
    b = np.zeros(5)
    _, info, dinfo = _parts(-1.0, b)
    assert dinfo / (2 * info) > 0          # below the items the correction pushes θ up
    _, info, dinfo = _parts(1.0, b)
    assert dinfo / (2 * info) < 0


# ---------------- Reference agreement (R) ----------------
@requires_tam
def test_jmle_matches_tam(sim_missing):
    r = jmle(sim_missing.X)
    ref = run_reference("JMLE", sim_missing.X, dict(adj=.3, bias=True, constraint="cases", tol=1e-4, max_iter=1000))
    assert ref["status"] == "Executed", ref
    np.testing.assert_allclose(r.b, ref["b"], atol=1e-3)
    np.testing.assert_allclose(r.theta, ref["theta"], atol=1e-3)
    np.testing.assert_allclose(r.se_b, ref["se_b"], atol=1e-3)


@requires_erm
def test_cmle_matches_erm(sim_missing):
    r = cmle(sim_missing.X)
    ref = run_reference("CMLE", sim_missing.X, {})
    assert ref["status"] == "Executed", ref
    np.testing.assert_allclose(r.b, ref["b"], atol=1e-3)
    np.testing.assert_allclose(r.se_b, ref["se_b"], atol=1e-3)
    assert r.cond_loglik == pytest.approx(ref["cond_loglik"], abs=1e-4)


@requires_tam
def test_mml_and_wle_match_tam(sim_missing):
    r = mml_em(sim_missing.X)
    ref = run_reference("MML", sim_missing.X, dict(n_nodes=21, extent=6.0, estimate_variance=True, fixed_sd=1.0, tol=1e-4, max_iter=1000))
    assert ref["status"] == "Executed", ref
    np.testing.assert_allclose(r.b, ref["b"], atol=1e-3)
    np.testing.assert_allclose(r.se_b_complete, ref["se_b"], atol=1e-3)
    assert r.variance == pytest.approx(ref["variance"], abs=1e-3)
    assert r.loglik_tam == pytest.approx(ref["loglik"], abs=1e-3)
    wle, wse = score_persons(sim_missing.X, ref["b"], "WLE")
    np.testing.assert_allclose(wle, ref["wle"], atol=1e-3)
    np.testing.assert_allclose(wse, ref["wle_se"], atol=1e-3)
    wref = run_reference("WLE", sim_missing.X, dict(b=ref["b"].tolist()))
    np.testing.assert_allclose(wle, wref["wle"], atol=1e-3)


@requires_tam
def test_mml_loglik_convention_with_truncated_grid():
    # A wide ability distribution is cut off by the [-6, 6] grid: TAM's density x spacing weights then sum
    # to < 1, and its log-likelihood differs from the renormalised one by N log(sum of weights).
    d = simulate(300, 8, theta_sd=2.0, seed=31)
    r = mml_em(d.X)
    ref = run_reference("MML", d.X, dict(n_nodes=21, extent=6.0, estimate_variance=True, fixed_sd=1.0, tol=1e-4, max_iter=1000))
    np.testing.assert_allclose(r.b, ref["b"], atol=1e-3)
    assert r.loglik_tam == pytest.approx(ref["loglik"], abs=1e-3)
    nodes = r.settings.nodes
    mass = (np.exp(-0.5 * (nodes / r.sd) ** 2) / (r.sd * np.sqrt(2 * np.pi))).sum() * (nodes[1] - nodes[0])
    assert r.loglik - r.loglik_tam == pytest.approx(-d.n_persons * np.log(mass), rel=1e-6)
