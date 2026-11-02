"""Scoring methods: verified examples, independent brute-force checks, safeguards, missing data."""
import itertools

import numpy as np
import pytest
from scipy import integrate, optimize
from scipy.stats import norm

from irt_models import make_item, pattern_terms
from scoring import (ScoringSettings, eap, eapsum, map_estimate, mle, node_table, reference_integral, rows, score_all,
                     score_distribution, tcc_inverse, weighted_likelihood_mode, wle)
from scoring.extras import adjusted_mle, plausible_values, posterior_median, prox
from simulation import example_a, example_b, example_c, example_d, simulate

S = ScoringSettings()
MIXED = simulate(60, seed=11, n_dich=8, dich_model="3PL", n_poly=4, poly_model="GRM", n_cat=4)


def brute_loglik(items, y, th):
    return sum(np.log(it.probs(np.array([th]))[0, int(v)]) for it, v in zip(items, y) if not np.isnan(v))


def test_example_a_rasch_mle_matches_direct_optimisation():
    a = example_a()
    r = mle(a.items, a.Y, S)
    y = a.Y[0]
    direct = optimize.minimize_scalar(lambda t: -brute_loglik(a.items, y, t), bounds=(-5, 5), method="bounded",
                                      options=dict(xatol=1e-10)).x
    assert abs(r.theta[0] - direct) < 1e-6
    # Same raw score ⇒ same Rasch MLE; 000 and 111 have none.
    raw = np.nansum(a.Y, 1)
    for k in (1, 2):
        assert np.ptp(r.theta[raw == k]) < 1e-9
    assert np.isneginf(r.theta[raw == 0]).all() and np.isposinf(r.theta[raw == 3]).all()


def test_example_b_pattern_vs_raw_score():
    b = example_b()
    r = mle(b.items, b.Y, S)
    assert np.allclose(r.theta[:2], [-0.4196, 0.4196], atol=5e-5)
    t = tcc_inverse(b.items, b.Y, S)
    assert np.allclose(t.theta[:2], 0, atol=1e-10)


def test_newton_history_is_real_and_converges():
    a = example_a()
    r = mle(a.items, a.Y[:1], S, record=True)
    h = rows(r.history, 0)
    assert h[0]["theta"] == S.start
    for k in range(1, len(h)):
        assert h[k]["theta"] == h[k - 1]["theta_new"]
    t = pattern_terms(a.items, a.Y[:1], [h[0]["theta"]])
    assert np.isclose(h[0]["grad"], t["score"][0]) and np.isclose(h[0]["hess"], t["hessian"][0])
    assert abs(h[-1]["step"]) < S.tol
    assert abs(pattern_terms(a.items, a.Y[:1], r.theta)["score"][0]) < 1e-8


def test_eap_equals_direct_sum_and_converges_to_adaptive_integral():
    c = example_c()
    y = c.Y[0]
    tab, m, psd = node_table(c.items, y, S)
    w = tab["prior_weight"] * np.exp(tab["log_likelihood"])
    assert np.isclose(m, (tab["node"] * w).sum() / w.sum(), atol=1e-14)
    assert np.isclose(tab["posterior_weight"].sum(), 1, atol=1e-14)
    ref_m, ref_s = reference_integral(c.items, y, S)
    big = eap(c.items, y[None, :], S.with_(n_nodes=81))
    assert abs(big.theta[0] - ref_m) < 1e-9 and abs(big.se[0] - ref_s) < 1e-9
    errs = [abs(eap(c.items, y[None, :], S.with_(n_nodes=n)).theta[0] - ref_m) for n in (3, 7, 15, 41)]
    assert errs[-1] < errs[0]


def test_eap_with_independent_integration_for_3pl_grm():
    y = MIXED.Y[3]
    f = lambda t, k: t ** k * np.exp(brute_loglik(MIXED.items, y, t)) * norm.pdf(t)  # noqa: E731
    z = integrate.quad(f, -10, 10, args=(0,), epsabs=0, epsrel=1e-12, limit=200)[0]
    m1 = integrate.quad(f, -10, 10, args=(1,), epsabs=0, epsrel=1e-12, limit=200)[0] / z
    assert abs(eap(MIXED.items, y[None, :], S.with_(n_nodes=101)).theta[0] - m1) < 1e-7


def test_log_domain_survives_long_tests():
    items = [make_item(f"i{j}", "2PL", a=1.5, b=0) for j in range(1500)]
    y = np.ones((1, 1500))
    y[0, ::3] = 0
    r = eap(items, y, S)
    assert np.isfinite(r.theta[0]) and np.isfinite(r.se[0])


def test_map_prior_influence():
    d = example_d()
    y = d.Y[:1]
    ml = mle(d.items, y, S).theta[0]
    wide = map_estimate(d.items, y, S.with_(prior_sd=1000)).theta[0]
    assert abs(wide - ml) < 1e-4
    narrow = map_estimate(d.items, y, S.with_(prior_mean=1.5, prior_sd=0.01)).theta[0]
    assert abs(narrow - 1.5) < 0.01
    r = map_estimate(d.items, y, S)
    t = pattern_terms(d.items, y, r.theta)
    assert abs(t["score"][0] - r.theta[0]) < 1e-7   # ℓ'(θ̂) = (θ̂ − 0)/1²


def test_map_finite_for_extreme_patterns():
    a = example_a()
    r = map_estimate(a.items, a.Y, S)
    assert np.isfinite(r.theta).all()


def test_wle_root_and_definitions():
    for d in (example_a(), example_c(), MIXED):
        r = wle(d.items, d.Y, S)
        t = pattern_terms(d.items, d.Y, r.theta)
        ok = r.status == "converged"
        assert ok.mean() > 0.9
        assert np.max(np.abs((t["score"] + t["J"] / (2 * t["info"]))[ok])) < 1e-8
    a = example_a()
    assert np.allclose(wle(a.items, a.Y, S).theta, weighted_likelihood_mode(a.items, a.Y, S).theta, atol=1e-8)
    diff = np.abs(wle(MIXED.items, MIXED.Y, S).theta - weighted_likelihood_mode(MIXED.items, MIXED.Y, S).theta)
    assert diff.max() > 1e-3


def test_wle_shrinks_mle_and_is_finite_for_extremes():
    a = example_a()
    w, m = wle(a.items, a.Y, S), mle(a.items, a.Y, S)
    fin = np.isfinite(m.theta)
    assert (np.abs(w.theta[fin]) <= np.abs(m.theta[fin]) + 1e-12).all()
    assert np.isfinite(w.theta).all()


def test_tcc_inversion_and_rasch_equivalence():
    d = simulate(80, seed=4, n_dich=10, dich_model="Rasch", n_poly=3, poly_model="PCM")
    t, m = tcc_inverse(d.items, d.Y, S), mle(d.items, d.Y, S)
    fin = np.isfinite(t.theta)
    T = pattern_terms(d.items, d.Y, np.where(fin, t.theta, 0))["expected"]
    assert np.allclose(T[fin], np.nansum(d.Y, 1)[fin], atol=1e-9)
    assert np.allclose(t.theta[fin], m.theta[fin], atol=1e-7)
    t3 = tcc_inverse(MIXED.items, MIXED.Y, S)
    lo = sum(it.c for it in MIXED.items)
    assert np.isneginf(t3.theta[np.nansum(MIXED.Y, 1) <= lo]).all()


def test_eapsum_recursion_matches_enumeration():
    items = example_c().items
    nodes = np.array([-1.3, 0.2, 2.0])
    f = score_distribution(items, nodes)
    brute = np.zeros_like(f)
    for pat in itertools.product(*[range(it.n_cat) for it in items]):
        p = np.prod([it.probs(nodes)[:, k] for it, k in zip(items, pat)], axis=0)
        brute[:, sum(pat)] += p
    assert np.allclose(f, brute, atol=1e-14)
    assert np.allclose(f.sum(1), 1)


def test_eapsum_same_score_same_estimate():
    d = example_d()
    r = eapsum(d.items, d.Y, S)
    raw = np.nansum(d.Y, 1)
    for k in np.unique(raw):
        assert np.ptp(r.theta[raw == k]) < 1e-12


def test_missing_responses_equal_scoring_the_observed_items():
    d = MIXED
    y = d.Y[5].copy()
    y[[1, 4, 9]] = np.nan
    keep = ~np.isnan(y)
    sub = [it for it, k in zip(d.items, keep) if k]
    full = score_all(d.items, y[None, :], S)
    red = score_all(sub, y[keep][None, :], S)
    for m in full:
        a, b = full[m].theta[0], red[m].theta[0]
        assert (np.isinf(a) and a == b) or abs(a - b) < 1e-10, m


def test_examinee_without_responses_is_reported():
    d = example_b()
    Y = np.vstack([d.Y[:1], [np.nan, np.nan]])
    res = score_all(d.items, Y, S)
    for m, r in res.items():
        assert np.isnan(r.theta[1]) and r.status[1] == "no observed responses", m


def test_3pl_global_maximum_and_restart():
    from irt_models import loglik_grid
    spec = [(1.7, -0.57, 0.2), (2.81, -0.41, 0.16), (2.89, 0.61, 0.17), (1.67, -0.17, 0.16), (2.45, 0.58, 0.15),
            (2.71, 1.43, 0.29), (2.51, 0.96, 0.17)]
    items = [make_item(f"i{j}", "3PL", a=a, b=b, c=c) for j, (a, b, c) in enumerate(spec)]
    y = np.array([[0, 1, 0, 0, 0, 0, 1]], float)
    grid = np.linspace(-S.bound, S.bound, 20001)
    best = grid[loglik_grid(items, y, grid)[0].argmax()]
    r = mle(items, y, S, start=-2.0, record=True)        # this start runs off to the lower bound first
    assert rows(r.history, 0)[-1]["theta_new"] == -S.bound
    assert "restart" in r.status[0] and abs(r.theta[0] - best) < 1e-3
    assert abs(pattern_terms(items, y, r.theta)["score"][0]) < 1e-7
    # Misses the easy items and answers the hard ones: ℓ is highest as θ → −∞, so there is no finite MLE.
    items2 = [make_item(f"k{j}", "3PL", a=2.0, b=b, c=0.25) for j, b in enumerate((-2, -1.5, 1.5, 2))]
    r2 = mle(items2, np.array([[0, 0, 1, 1]], float), S)
    assert np.isneginf(r2.theta[0])


def test_extras():
    a = example_a()
    pm = posterior_median(a.items, a.Y, S)
    assert np.isfinite(pm.theta).all()
    pv = plausible_values(a.items, a.Y, S, draws=4, seed=1)
    assert pv.shape == (8, 4) and np.array_equal(pv, plausible_values(a.items, a.Y, S, draws=4, seed=1))
    p = prox(a.items, a.Y)
    assert np.isinf(p[[5, 6]]).all() and np.isfinite(p[:5]).all()
    adj = adjusted_mle(a.items, a.Y, S)
    assert np.isfinite(adj.theta).all()
    with pytest.raises(ValueError):
        prox(example_b().items, example_b().Y)


def test_simulation_is_reproducible():
    a = simulate(50, seed=9, n_dich=5, dich_model="4PL", n_poly=2, poly_model="RSM", missing_rate=0.2)
    b = simulate(50, seed=9, n_dich=5, dich_model="4PL", n_poly=2, poly_model="RSM", missing_rate=0.2)
    assert a.identity == b.identity and np.allclose(a.theta_true, b.theta_true)
    assert np.isfinite(a.Y).any(1).all()
    for j, it in enumerate(a.items):
        v = a.Y[:, j][~np.isnan(a.Y[:, j])]
        assert set(np.unique(v)) <= set(range(it.n_cat))


def test_monte_carlo_runs_and_is_reproducible():
    from simulation import monte_carlo
    kw = dict(n_persons=100, replications=3, seed=5, n_dich=10, dich_model="2PL", n_poly=2, poly_model="GPCM")
    a = monte_carlo.run(["MLE", "EAP", "WLE"], S, **kw)
    b = monte_carlo.run(["MLE", "EAP", "WLE"], S, **kw)
    assert a["completed"] == 3
    assert np.allclose(a["summary"]["rmse"], b["summary"]["rmse"])
    s = a["summary"].set_index("method")
    assert s.loc["EAP", "rmse"] < s.loc["MLE", "rmse"]          # shrinkage pays off on a short test
    assert s.loc["MLE", "nonfinite_rate"] > 0 and s.loc["EAP", "nonfinite_rate"] == 0
