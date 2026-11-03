"""Streamlit application tests: every page loads, controls change results, calculation navigation works."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, _R
from core.simulation import SimSpec, simulate

PAGES = sorted((ROOT / "pages").glob("*.py"), key=lambda p: int(p.name.split("_")[0]))


def app(path, timeout=300, **state):
    at = AppTest.from_file(str(path), default_timeout=timeout)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def ok(at):
    assert not at.exception, [e.message for e in at.exception]


def button(at, label):
    return next(b for b in at.button if b.label == label)


def small():
    d = simulate(SimSpec(model="GPCM", n_per_group=300, n_items=5, ncat=3, seed=3), "Small GPCM")
    return {"datasets": {d.name: d}, "active": d.name, "person": 0}


@pytest.mark.parametrize("path", [ROOT / "Polytomous_MMLE_EM.py", ROOT / "app.py"] + PAGES, ids=lambda p: Path(p).name)
def test_every_page_loads(path):
    ok(app(path))


def test_calibration_results_fit_pages_share_one_run():
    at = app(ROOT / "Polytomous_MMLE_EM.py", **small())
    at.switch_page("pages/7_Calibration_Designs.py").run()
    ok(at)
    button(at, "Run MMLE–EM calibration").click().run()
    ok(at)
    res = at.session_state["fits"]["latest"]
    assert res.converged and res.config.model == "GPCM"
    at.switch_page("pages/8_Results.py").run()
    ok(at)
    assert any(m.label == "Cycles" and str(res.cycles) == m.value for m in at.metric)
    slider = next(s for s in at.slider if s.label == "Cycle")
    slider.set_value(min(3, res.cycles)).run()
    ok(at)
    assert any("Cycle 3" in m.value for m in at.markdown if m.value.startswith("**Cycle"))
    at.switch_page("pages/10_Model_Fit.py").run()
    ok(at)
    assert any(m.label == "RMSE (proportions)" for m in at.metric)


def test_sidebar_prior_changes_the_calibration():
    st = small()
    at = app(ROOT / "pages/7_Calibration_Designs.py", **st)
    button(at, "Run MMLE–EM calibration").click().run()
    ll_fixed = at.session_state["fits"]["latest"].loglik
    at.selectbox(key="w_cfg_prior").set_value("estimated_empirical").run()
    ok(at)
    button(at, "Run MMLE–EM calibration").click().run()
    ok(at)
    r = at.session_state["fits"]["latest"]
    assert r.config.prior == "estimated_empirical" and r.loglik > ll_fixed


def test_invalid_design_is_explained_and_blocked():
    at = app(ROOT / "pages/7_Calibration_Designs.py", **small())
    at.selectbox(key="w_cfg_design").set_value("fipc").run()
    ok(at)
    assert any("anchor" in e.value for e in at.error)
    assert button(at, "Run MMLE–EM calibration").disabled


def test_walkthrough_navigation_and_next_cycle():
    at = app(ROOT / "pages/6_EM_Walkthrough.py")
    ok(at)
    for _ in range(12):
        button(at, "Next ▶").click().run()
        ok(at)
    assert at.session_state["wt_step"] == 13
    button(at, "Start the next cycle ▶▶").click().run()
    ok(at)
    assert at.session_state["wt_step"] == 4 and len(at.session_state["wt_hist"]) == 1
    at.session_state["wt_step"] = 13
    at.run()
    button(at, "Run 50 more cycles").click().run()
    ok(at)
    h = at.session_state["wt_hist"]
    assert len(h) == 51 and all(b["ll_after"] >= b["ll_before"] - 1e-9 for b in h)
    for model in ("RSM", "GRM"):
        for prior in ("estimated_normal", "estimated_empirical", "fixed_empirical"):
            at.selectbox(key="wt_model").set_value(model).run()
            at.selectbox(key="wt_prior").set_value(prior).run()
            at.session_state["wt_step"] = 10
            at.run()
            ok(at)


def test_simulation_page_generates_and_activates():
    at = app(ROOT / "pages/4_Data_Simulation.py")
    sel = next(s for s in at.selectbox if s.label == "Design")
    sel.set_value("mgfipc").run()
    button(at, "Generate dataset").click().run()
    ok(at)
    d = at.session_state["datasets"][at.session_state["active"]]
    assert d.G == 2 and len(d.anchors) == 5 and at.session_state["cfg_design"] == "mgfipc"


def test_prior_explorer_and_quadrature_controls():
    at = app(ROOT / "pages/3_Prior_Distribution_Explorer.py", **small())
    at.selectbox(key="w_cfg_prior").set_value("fixed_empirical").run()
    ok(at)
    button(at, "Run the three-prior comparison").click().run()
    ok(at)
    assert len(at.session_state["prior_comparison"][0]) == 3
    at = app(ROOT / "pages/5_Quadrature_Laboratory.py")
    next(s for s in at.slider if s.label == "Nodes Q").set_value(5).run()
    ok(at)


def test_recovery_and_drift_pages_run():
    at = app(ROOT / "pages/9_Parameter_Recovery.py", timeout=600)
    next(s for s in at.select_slider if s.label == "N per group").set_value(250).run()
    next(s for s in at.select_slider if s.label == "Items per form").set_value(5).run()
    next(n for n in at.number_input if n.label == "Replications").set_value(2).run()
    button(at, "Run Monte Carlo").click().run()
    ok(at)
    res, _, _ = at.session_state["recovery"]
    assert set(res) == {"fixed_normal", "estimated_empirical"}
    assert res["fixed_normal"]["runs"]["ok"].all()
    at = app(ROOT / "pages/11_Drift_Analysis.py", timeout=600)
    next(s for s in at.select_slider if s.label == "N per year").set_value(500).run()
    next(s for s in at.selectbox if s.label == "Scenario").set_value("Anchors become harder (+0.5 for 2 anchors)").run()
    button(at, "Simulate and analyse").click().run()
    ok(at)
    summary, disp, *_ = at.session_state["drift"]
    assert len(summary) == 3
    drifted = disp[disp["drifted"]]["estimated_drift"]
    stable = disp[~disp["drifted"]]["estimated_drift"]
    assert drifted.mean() > stable.mean() + 0.2


@pytest.mark.skipif(not (_R["available"] and _R.get("irtQ")), reason="R/irtQ not available")
def test_reference_page_runs_irtq():
    at = app(ROOT / "pages/12_Reference_Validation.py", timeout=600, **small())
    button(at, "Run lab engine and references").click().run()
    ok(at)
    res, out, *_ = at.session_state["validation"]
    assert out["irtQ"]["status"] == "Executed"
