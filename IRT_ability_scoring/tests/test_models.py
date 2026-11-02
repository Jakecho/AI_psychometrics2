"""Item models: probabilities, derivatives, information, J, special cases."""
import numpy as np
import pytest

from irt_models import J_EQUALS_IPRIME, make_item, pattern_terms
from simulation import example_d

GRID = np.linspace(-6, 6, 25)
ITEMS = [make_item("r", "Rasch", b=0.4), make_item("2", "2PL", a=1.7, b=-0.3), make_item("3", "3PL", a=1.2, b=0.5, c=0.2),
         make_item("4", "4PL", a=0.9, b=-1, c=0.1, d=0.88), make_item("p", "PCM", steps=(-1, 0.6, 0.2)),
         make_item("g", "GPCM", a=1.4, steps=(0.5, -0.4)), make_item("s", "RSM", b=0.3, steps=(-1, 0, 1)),
         make_item("m", "GRM", a=1.3, steps=(-1.2, 0.1, 1.5)), make_item("d", "2PL", a=1.0, b=0.0, D=1.702)]


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.model + i.id)
def test_probabilities_sum_to_one_and_are_positive(item):
    P = item.probs(np.linspace(-12, 12, 97))
    assert np.allclose(P.sum(1), 1, atol=1e-12)
    assert (P > 0).all()


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.model + i.id)
def test_derivatives_match_finite_differences(item):
    h = 1e-5
    P, dP, d2P = item.derivatives(GRID)
    Pp, Pm = item.probs(GRID + h), item.probs(GRID - h)
    assert np.allclose(dP, (Pp - Pm) / (2 * h), atol=1e-7)
    assert np.allclose(d2P, (Pp - 2 * P + Pm) / h ** 2, atol=1e-4)
    assert np.allclose(dP.sum(1), 0, atol=1e-12) and np.allclose(d2P.sum(1), 0, atol=1e-12)


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.model + i.id)
def test_information_is_expected_negative_hessian_and_J_identity(item):
    P, dl, d2l = item.log_derivs(GRID)
    assert np.allclose(item.info(GRID), -(P * d2l).sum(1), atol=1e-10)
    h = 1e-5
    Iprime = (item.info(GRID + h) - item.info(GRID - h)) / (2 * h)
    if item.model in J_EQUALS_IPRIME:
        assert np.allclose(item.warm_j(GRID), Iprime, atol=1e-7)
    else:
        assert np.max(np.abs(item.warm_j(GRID) - Iprime)) > 1e-3


def test_pattern_terms_match_numerical_derivatives_mixed_format():
    y = np.array([[0, 1, 1, 0, 2, 1, 3, 2, 1]], float)
    for th in (-2.3, 0.1, 1.9):
        t = pattern_terms(ITEMS, y, [th])
        tp, tm = pattern_terms(ITEMS, y, [th + 1e-5]), pattern_terms(ITEMS, y, [th - 1e-5])
        assert np.isclose(t["score"], (tp["loglik"] - tm["loglik"]) / 2e-5, atol=1e-7)
        assert np.isclose(t["hessian"], (tp["score"] - tm["score"]) / 2e-5, atol=1e-6)
        assert np.isclose(t["slope"], (tp["expected"] - tm["expected"]) / 2e-5, atol=1e-7)
        assert np.isclose(t["Iprime"], (tp["info"] - tm["info"]) / 2e-5, atol=1e-6)


def test_example_d_grm_probabilities():
    P = example_d().items[0].probs(np.array([0.0]))[0]
    assert np.allclose(P, [0.26894, 0.46212, 0.26894], atol=5e-6)


def test_special_cases_reduce_correctly():
    th = np.linspace(-4, 4, 33)
    two = make_item("a", "2PL", a=1.3, b=0.2)
    assert np.allclose(make_item("b", "GPCM", a=1.3, steps=(0.2,)).probs(th), two.probs(th))
    assert np.allclose(make_item("c", "GRM", a=1.3, steps=(0.2,)).probs(th), two.probs(th))
    assert np.allclose(make_item("d", "4PL", a=1.3, b=0.2, c=0, d=1).probs(th), two.probs(th))
    rsm = make_item("e", "RSM", b=0.5, steps=(-1, 1))
    assert np.allclose(rsm.probs(th), make_item("f", "PCM", steps=(-0.5, 1.5)).probs(th))


def test_extreme_theta_stays_finite():
    for it in ITEMS:
        lp = it.log_probs(np.array([-40.0, 40.0]))
        assert np.isfinite(lp).all()


def test_constraints_reported():
    assert make_item("x", "GRM", steps=(1, 0)).problems()
    assert make_item("x", "4PL", c=0.5, d=0.4).problems()
    assert not make_item("x", "3PL", a=1, c=0.2).problems()


def test_grm_satisfies_J_equals_Iprime_but_not_observed_equals_expected():
    """For the GRM, Σ_k P_k dlogP_k d²logP_k telescopes to 0 (so Warm's WLE = L·√I mode) even though
    −d²logP_k varies by category (so observed and expected information differ)."""
    it = make_item("m", "GRM", a=1.6, steps=(-1.5, -0.2, 0.4, 2.0))
    P, dl, d2l = it.log_derivs(GRID)
    assert np.allclose((P * dl * d2l).sum(1), 0, atol=1e-12)
    assert np.ptp(d2l, axis=1).max() > 0.1
