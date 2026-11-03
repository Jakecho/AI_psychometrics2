"""Model tests: probabilities, special cases, constraints, gradients, conversions (acceptance items 1–4, 15)."""
import numpy as np
import pytest
from scipy.optimize import approx_fprime

from core import models as M
from core.models import Item, make_item

THETA = np.linspace(-8, 8, 161)
ITEMS = [make_item("p", "PCM", 1, (-1.0, 0.3, 1.2)), Item("r", "RSM", 1, (), b=0.4, tau=(-0.8, -0.1, 0.9)),
         make_item("g", "GPCM", 1.7, (0.5, -0.4, 1.1)), make_item("q", "GRM", 0.8, (-1.5, 0.0, 2.0)),
         make_item("x", "GPCM", 3.0, (9.0, -9.0)), make_item("y", "GRM", 6.0, (-9.5, 9.5))]


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
@pytest.mark.parametrize("D", [1.0, 1.702])
def test_category_probabilities_sum_to_one_and_are_finite(item, D):
    P = M.probs(item, np.r_[THETA, -40, 40], D)
    assert np.all(np.isfinite(P)) and np.all(P >= 0)
    np.testing.assert_allclose(P.sum(axis=1), 1, atol=1e-12)


def test_pcm_equals_gpcm_with_unit_slope():
    d = (-1.2, 0.1, 0.8, 2.0)
    np.testing.assert_allclose(M.probs(make_item("a", "PCM", 1, d), THETA), M.probs(make_item("b", "GPCM", 1.0, d), THETA),
                               atol=1e-15)


def test_rsm_steps_are_location_plus_shared_tau():
    it = Item("r", "RSM", 1, (), b=0.4, tau=(-0.8, -0.1, 0.9))
    np.testing.assert_allclose(it.d, 0.4 + np.array(it.tau))
    assert abs(sum(it.tau)) < 1e-12
    with pytest.raises(ValueError):
        Item("bad", "RSM", 1, (), b=0.0, tau=(0.5, 0.5))
    np.testing.assert_allclose(M.probs(it, THETA), M.probs(make_item("p", "PCM", 1, it.d), THETA), atol=1e-15)


def test_grm_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        make_item("g", "GRM", 1.0, (0.5, 0.2))
    it = make_item("g", "GRM", 1.3, (-1.0, 0.5, 1.5))
    cum = M.cumulative_probs(it, THETA)
    assert np.all(np.diff(cum, axis=1) <= 1e-15)              # P(Y≥1) ≥ P(Y≥2) ≥ …
    # P(Y ≥ k) = logistic(a(θ − b_k))
    np.testing.assert_allclose(cum, 1 / (1 + np.exp(-1.3 * (THETA[:, None] - np.array(it.d)))), atol=1e-12)


def test_adjacent_log_odds_equal_slope_times_distance():
    it = make_item("g", "GPCM", 1.4, (-0.5, 0.7))
    np.testing.assert_allclose(M.adjacent_odds(it, THETA, 1.0), 1.4 * (THETA[:, None] - np.array(it.d)), atol=1e-10)


@pytest.mark.parametrize("item", ITEMS[:4], ids=lambda i: i.id)
def test_information_matches_numerical_derivative(item):
    th = np.linspace(-3, 3, 13)
    h = 1e-5
    dP = (M.probs(item, th + h) - M.probs(item, th - h)) / (2 * h)
    num = np.sum(dP ** 2 / M.probs(item, th), axis=1)
    np.testing.assert_allclose(M.information(item, th), num, rtol=1e-6)


def _counts(item, seed=0):
    rng = np.random.default_rng(seed)
    th = np.linspace(-4, 4, 21)
    return th, rng.gamma(2.0, 3.0, (len(th), item.ncat))


def test_adjacent_gradient_matches_finite_differences():
    it = make_item("g", "GPCM", 1.3, (-0.6, 0.2, 1.1))
    th, C = _counts(it)

    def f(x):
        return M.adjacent_objective(th, C, np.exp(x[0]), x[1:], 1.7)[0]
    x0 = np.r_[np.log(it.a), it.d]
    _, ga, gd = M.adjacent_objective(th, C, it.a, it.d, 1.7)
    np.testing.assert_allclose(np.r_[ga, gd], approx_fprime(x0, f, 1e-7), rtol=1e-5, atol=1e-4)


def test_grm_gradient_matches_finite_differences():
    it = make_item("q", "GRM", 0.9, (-1.0, 0.4, 1.3))
    th, C = _counts(it, 1)

    def f(x):
        return M.grm_objective(th, C, x[0], x[1:], 1.0)[0]
    x0 = np.r_[it.a, it.d]
    _, ga, gb = M.grm_objective(th, C, it.a, it.d, 1.0)
    np.testing.assert_allclose(np.r_[ga, gb], approx_fprime(x0, f, 1e-7), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
@pytest.mark.parametrize("D", [1.0, 1.702])
def test_intercept_conversion_round_trip_preserves_probabilities(item, D):
    ic = M.to_intercepts(item, D)
    back = M.from_intercepts(item.id, item.model if item.model != "RSM" else "PCM", ic["alpha"], ic["intercepts"], D)
    np.testing.assert_allclose(M.probs(back, THETA, D), M.probs(item, THETA, D), atol=1e-12)
    # slope–intercept form reproduces the probabilities directly
    if item.model != "GRM":
        k = np.arange(item.ncat)
        z = ic["alpha"] * THETA[:, None] * k + np.array(ic["intercepts"])
        P = np.exp(z - z.max(axis=1, keepdims=True))
        np.testing.assert_allclose(P / P.sum(axis=1, keepdims=True), M.probs(item, THETA, D), atol=1e-12)


def test_mirt_and_tam_representations():
    it = make_item("g", "GPCM", 1.2, (-0.5, 0.5))
    m = M.to_mirt(it)
    assert m["a1"] == pytest.approx(1.2) and m["d0"] == 0
    assert m["d1"] == pytest.approx(0.6) and m["d2"] == pytest.approx(0.0)
    r = Item("r", "RSM", 1, (), b=0.3, tau=(-0.4, 0.4))
    assert M.to_tam(r) == dict(xsi_item=0.3, tau=(-0.4, 0.4))
    with pytest.raises(ValueError):
        M.to_irtq(r)


def test_sampling_matches_probabilities():
    it = make_item("g", "GPCM", 1.0, (-0.5, 0.5))
    rng = np.random.default_rng(3)
    y = M.sample_responses([it], np.zeros(200000), rng)[:, 0]
    np.testing.assert_allclose(np.bincount(y) / len(y), M.probs(it, [0.0])[0], atol=4e-3)
