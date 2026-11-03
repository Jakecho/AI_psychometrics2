"""Executed comparisons with TAM, irtQ and mirt (skipped when R or the package is unavailable)."""
from dataclasses import replace

import pytest

from conftest import _R
from core.em import FitConfig, fit
from core.simulation import SimSpec, simulate
from r_reference.bridge import compare, run_reference, scope

from r_reference.cases import CASES, TIGHT


@pytest.mark.parametrize("pkg,spec,cfg,tol", CASES,
                         ids=[f"{c[0]}-{c[2].model}-{c[2].design}-{c[2].prior}-{c[2].fipc_method}" for c in CASES])
def test_lab_agrees_with_reference(pkg, spec, cfg, tol):
    if not (_R["available"] and _R.get(pkg)):
        pytest.skip(f"R/{pkg} not available")
    d = simulate(spec)
    cfg = replace(cfg, **TIGHT)
    res = fit(d, cfg)
    ref = run_reference(pkg, d, cfg)
    assert ref["status"] == "Executed", ref.get("reason")
    items, groups, sm = compare(res, ref)
    assert sm["max_item_diff"] < tol
    assert sm["max_moment_diff"] < max(tol, 1e-4)
    assert abs(sm["loglik_diff"]) < (0.05 if cfg.prior == "estimated_empirical" else 1e-3)


def test_out_of_scope_combinations_are_reported_not_faked():
    assert "rating scale" in scope("irtQ", FitConfig(model="RSM"))
    assert scope("TAM", FitConfig(model="GRM"))
    assert not scope("TAM", FitConfig(model="GPCM"))
    assert "fixed grid" not in scope("mirt", FitConfig(model="GPCM"))
    assert scope("irtQ", FitConfig(model="GPCM", grid="gauss_hermite"))
    d = simulate(SimSpec(model="RSM", n_per_group=100, n_items=3, ncat=3, seed=1))
    out = run_reference("irtQ", d, FitConfig(model="RSM"))
    assert out["status"] in ("Not applicable", "Unavailable")
