"""Quadrature and latent-distribution tests (acceptance items 5, 8, 9, 13)."""
import numpy as np
import pytest
from scipy.optimize import minimize

from core import priors as P
from core.quadrature import gauss_hermite, illustrative_distribution, make_grid, moments, validate_weights


def test_three_point_gauss_hermite():
    x, w = gauss_hermite(3)
    np.testing.assert_allclose(x, [-np.sqrt(3), 0, np.sqrt(3)], atol=1e-14)
    np.testing.assert_allclose(w, [1 / 6, 2 / 3, 1 / 6], atol=1e-14)


@pytest.mark.parametrize("kind", ["gauss_hermite", "equal"])
def test_grid_weights_valid(kind):
    g = make_grid(kind, 21, 5.0, 0.3, 1.2)
    assert np.all(g.base_weights >= 0) and g.base_weights.sum() == pytest.approx(1)
    m, s = moments(g.nodes, g.base_weights)
    assert m == pytest.approx(0.3, abs=1e-3) and s == pytest.approx(1.2, abs=1e-2)   # discretised on a finite grid


def test_weight_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_weights([0.5, -0.1, 0.6])
    with pytest.raises(ValueError):
        validate_weights([0.2, 0.8], Q=3)
    np.testing.assert_allclose(validate_weights([1, 1, 2]), [0.25, 0.25, 0.5])


def test_unconstrained_update_is_fractional_count_average():
    rng = np.random.default_rng(1)
    post = rng.dirichlet(np.ones(11), size=40)            # w_iq for 40 examinees
    w = P.empirical_unconstrained(post.sum(axis=0))
    np.testing.assert_allclose(w, post.mean(axis=0), atol=1e-15)


@pytest.mark.parametrize("sd", [None, 1.0])
def test_constrained_update_satisfies_moments_and_is_optimal(sd):
    x = np.linspace(-4, 4, 17)
    rng = np.random.default_rng(2)
    n = rng.gamma(2, 5, 17) * np.exp(-0.2 * (x - 1) ** 2)
    w, info = P.empirical_constrained(n, x, 0.0, sd)
    m, s = moments(x, w)
    assert w.sum() == pytest.approx(1, abs=1e-12) and np.all(w >= 0)
    assert m == pytest.approx(0, abs=1e-9)
    if sd is not None:
        assert s == pytest.approx(1, abs=1e-9)
    # compare with a general-purpose constrained optimiser
    cons = [dict(type="eq", fun=lambda v: v.sum() - 1), dict(type="eq", fun=lambda v: v @ x)]
    if sd is not None:
        cons.append(dict(type="eq", fun=lambda v: v @ x ** 2 - 1))
    ref = minimize(lambda v: -np.sum(n * np.log(np.maximum(v, 1e-300))), np.full(17, 1 / 17), constraints=cons,
                   bounds=[(1e-12, 1)] * 17, method="SLSQP", options=dict(ftol=1e-14, maxiter=1000))
    assert P.objective(n, w) >= -ref.fun - 1e-6


def test_constrained_three_nodes_both_moments_is_unique():
    x = np.array([-np.sqrt(3), 0, np.sqrt(3)])
    w, _ = P.empirical_constrained([5.0, 1.0, 2.0], x, 0.0, 1.0)
    np.testing.assert_allclose(w, [1 / 6, 2 / 3, 1 / 6], atol=1e-10)


def test_infeasible_moments_raise():
    with pytest.raises(ValueError):
        P.empirical_constrained([1, 2, 3], [-1.0, 0.0, 1.0], 5.0)


def test_normal_exact_is_maximiser_with_positive_sd():
    x = np.linspace(-5, 5, 31)
    n = 1000 * illustrative_distribution(x, "skewed", 0.4, 1.3)
    w, mu, sd, info = P.normal_exact(n, x)
    assert sd > 0 and info["success"]
    for dm, ds in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]:
        from core.quadrature import normal_weights
        assert P.objective(n, w) >= P.objective(n, normal_weights(x, mu + dm, sd + ds)) - 1e-9
    w2, mu2, sd2, _ = P.normal_exact(n, x, mean=0.0)        # mean held fixed
    assert mu2 == 0.0 and sd2 > 0


def test_moment_update_returns_posterior_moments():
    x = np.linspace(-5, 5, 41)
    n = 500 * illustrative_distribution(x, "normal", 0.5, 0.8)
    _, mu, sd, _ = P.normal_moments(n, x)
    m, s = moments(x, n / n.sum())
    assert mu == pytest.approx(m) and sd == pytest.approx(s)


def test_woods_rescaling_hits_target_moments_approximately():
    x = np.linspace(-6, 6, 49)
    w = illustrative_distribution(x, "skewed", 0.6, 1.4)
    out = P.woods_rescale(x, w, 0.0, 1.0)
    m, s = moments(x, out)
    assert out.sum() == pytest.approx(1) and abs(m) < 0.02 and abs(s - 1) < 0.02
