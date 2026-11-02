"""Agreement with irtQ (primary), mirt and TAM under matched settings."""
import numpy as np
import pytest

from conftest import requires
from r_bridge.compare import compare_package
from scoring import ScoringSettings
from simulation import example_a, example_b, example_c, example_d, simulate

S = ScoringSettings()
DATA = {"A": example_a, "B": example_b, "C": example_c, "D": example_d,
        "3PL+GRM": lambda: simulate(100, seed=3, n_dich=12, dich_model="3PL", n_poly=4, poly_model="GRM"),
        "2PL+GPCM": lambda: simulate(100, seed=8, n_dich=10, dich_model="2PL", n_poly=4, poly_model="GPCM", n_cat=5)}


def _check(rows):
    bad = [r for r in rows if r["result"] not in ("Pass", "Information", "Explained difference")]
    assert not bad, bad
    assert any(r["result"] == "Pass" for r in rows)


@requires("irtQ")
@pytest.mark.parametrize("name", list(DATA))
def test_irtq_agreement(name):
    d = DATA[name]()
    _, rows = compare_package("irtQ", d.items, d.Y, S)
    _check(rows)
    for r in rows:
        if r["result"] == "Pass":
            assert r["max_abs_diff_theta"] < 1e-6, r


@requires("irtQ")
def test_irtq_rectangular_nodes_and_prior():
    d = example_d()
    _, rows = compare_package("irtQ", d.items, d.Y, S.with_(quad_rule="rectangular", n_nodes=31, prior_mean=0.4, prior_sd=1.3))
    _check(rows)


@requires("mirt")
@pytest.mark.parametrize("name", ["A", "D", "3PL+GRM"])
def test_mirt_agreement(name):
    d = DATA[name]()
    _, rows = compare_package("mirt", d.items, d.Y, S)
    _check(rows)


@requires("mirt")
def test_mirt_4pl():
    d = simulate(80, seed=4, n_dich=10, dich_model="4PL", n_poly=3, poly_model="RSM")
    _, rows = compare_package("mirt", d.items, d.Y, S)
    _check(rows)
    warm = next(r for r in rows if r["python"].startswith("Warm"))
    assert warm["max_abs_diff_theta"] > 1e-3           # Warm's WLE ≠ mirt's L·√I mode for 4PL


@requires("TAM")
def test_tam_wle_rasch_family():
    d = simulate(100, seed=5, n_dich=10, dich_model="Rasch", n_poly=4, poly_model="RSM", missing_rate=0.1)
    _, rows = compare_package("TAM", d.items, d.Y, S)
    _check(rows)


@requires("irtQ")
def test_unsupported_model_reports_unavailable():
    d = simulate(30, seed=4, n_dich=4, dich_model="4PL", n_poly=0)
    ref, rows = compare_package("irtQ", d.items, d.Y, S)
    assert ref["status"] == "Unavailable" and "4PL" in ref["reason"]
