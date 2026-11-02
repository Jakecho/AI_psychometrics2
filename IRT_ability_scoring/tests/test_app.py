"""Streamlit application tests: every page loads on several datasets, controls work, state is shared."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, requires
from simulation import example_a, example_b, simulate

PAGES = sorted((ROOT / "pages").glob("*.py"), key=lambda p: int(p.name.split("_")[0]))
DATASETS = {"default": None, "exampleB": example_b,
            "3PL+GRM+missing": lambda: simulate(60, seed=3, n_dich=8, dich_model="3PL", n_poly=3, poly_model="GRM",
                                                missing_rate=0.15),
            "4PL+RSM": lambda: simulate(60, seed=4, n_dich=6, dich_model="4PL", n_poly=3, poly_model="RSM")}


def app(path, dataset=None, person=0, timeout=240):
    at = AppTest.from_file(str(path), default_timeout=timeout)
    if dataset is not None:
        at.session_state["datasets"] = {dataset.name: dataset}
        at.session_state["active"] = dataset.name
        at.session_state["person"] = person
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def no_errors(at):
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("path", [ROOT / "IRT_ability_scoring.py", ROOT / "app.py"] + PAGES, ids=lambda p: Path(p).name)
@pytest.mark.parametrize("name", list(DATASETS))
def test_every_page_loads(path, name):
    fn = DATASETS[name]
    no_errors(app(path, fn() if fn else None))


@pytest.mark.parametrize("page,key", [("3_MLE_Lab.py", "mle"), ("5_MAP_Lab.py", "map"), ("6_WLE_Lab.py", "wle"),
                                      ("7_TCC_and_EAPsum_Lab.py", "tcc"), ("4_EAP_Lab.py", "eap")])
def test_step_controls(page, key):
    at = app(ROOT / "pages" / page, example_a())
    button(at, "Initialize").click().run()
    no_errors(at)
    assert at.session_state[f"{key}_pos"] == 1
    button(at, "Next step ▶").click().run()
    assert at.session_state[f"{key}_pos"] == 2
    button(at, "◀ Previous step").click().run()
    assert at.session_state[f"{key}_pos"] == 1
    button(at, "Run to convergence ⏭").click().run()
    last = at.session_state[f"{key}_pos"]
    assert last >= 2
    button(at, "Reset").click().run()
    assert at.session_state[f"{key}_pos"] is None
    no_errors(at)


def test_mle_iteration_values_are_the_recorded_ones():
    from scoring import ScoringSettings, mle, rows
    d = example_a()
    at = app(ROOT / "pages/3_MLE_Lab.py", d)
    button(at, "Run to convergence ⏭").click().run()
    r = mle(d.items, d.Y[:1], ScoringSettings(), record=True)
    last = rows(r.history, 0)[-1]
    shown = [m.value for m in at.metric]
    assert f"{last['theta_new']:.4f}" in shown
    assert any(f"{r.theta[0]:.4f}" == v for v in shown)


def test_extreme_pattern_warning_and_shared_examinee():
    d = example_a()
    at = app(ROOT / "IRT_ability_scoring.py", d, person=6)    # pattern 111
    at.switch_page("pages/3_MLE_Lab.py").run()
    no_errors(at)
    assert any("extreme pattern" in w.value for w in at.warning)
    at.switch_page("pages/6_WLE_Lab.py").run()
    no_errors(at)
    assert at.session_state["person"] == 6
    assert any(m.label == "θ̂ (WLE, Warm)" and m.value not in ("—", "+∞") for m in at.metric)


def test_sidebar_settings_change_results_on_every_page():
    at = app(ROOT / "IRT_ability_scoring.py", example_a())
    at.switch_page("pages/4_EAP_Lab.py").run()
    before = next(m.value for m in at.metric if m.label == "EAP")
    at.session_state["cfg_prior_mean"] = 1.0
    at.run()
    after = next(m.value for m in at.metric if m.label == "EAP")
    assert before != after
    at.switch_page("pages/5_MAP_Lab.py").run()
    assert "N(1, 1²)" in "".join(m.value for m in at.markdown)


def test_simulate_and_edit_pattern():
    at = app(ROOT / "pages/2_Data_and_Simulation.py")
    button(at, "Generate responses").click().run()
    no_errors(at)
    d = at.session_state["datasets"][at.session_state["active"]]
    assert d.n_persons == 300 and d.has_truth
    at = app(ROOT / "pages/2_Data_and_Simulation.py", example_b())
    button(at, "Use this example").click().run()       # first example card (A)
    no_errors(at)
    assert at.session_state["active"].startswith("Example A")


def test_monte_carlo_page_runs_and_marks_stale():
    at = app(ROOT / "pages/9_Monte_Carlo_Simulation.py")
    button(at, "Run simulation").click().run()
    no_errors(at)
    assert at.session_state["mc_result"]["out"]["completed"] == 10
    assert any(d.label == "Summary (CSV)" for d in at.get("download_button"))
    # Form inputs apply only on submit; a changed sidebar setting makes the stored results stale.
    at.session_state["cfg_tol"] = 1e-4
    at.run()
    assert any("changed since the results" in w.value for w in at.warning)


@requires("irtQ")
def test_comparison_page_reference_run():
    at = app(ROOT / "pages/8_Estimator_Comparison.py", example_b())
    button(at, "Run the reference comparison").click().run()
    no_errors(at)
    rows = at.session_state["val_rows"][0]
    assert not [r for r in rows if r["result"] == "Fail"]
    assert any(r["package"] == "irtQ" and r["result"] == "Pass" for r in rows)


@requires("irtQ")
def test_lab_reference_button():
    at = app(ROOT / "pages/6_WLE_Lab.py", example_a())
    button(at, "Check this examinee's WLE against R (irtQ / mirt / TAM)").click().run()
    no_errors(at)
    rows = at.session_state["wle_refrows"][0]
    assert {r["package"] for r in rows} == {"irtQ", "mirt", "TAM"}
    assert all(r["result"] in ("Pass", "Information") for r in rows)
