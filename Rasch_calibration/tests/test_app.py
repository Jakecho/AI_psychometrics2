"""Streamlit application tests (pages load, shared state, controls change results, uploads validate)."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT
from rasch.simulation import load_csv, simulate

PAGES = sorted((ROOT / "pages").glob("*.py"))


def app(path, dataset=None, timeout=120):
    at = AppTest.from_file(str(path), default_timeout=timeout)
    if dataset is not None:
        at.session_state["dataset"] = dataset
        at.session_state["results"] = {}
        at.session_state["references"] = {}
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def no_errors(at):
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("path", [ROOT / "Rasch_calibration.py", ROOT / "app.py"] + PAGES, ids=lambda p: Path(p).name)
def test_every_page_loads(path):
    no_errors(app(path, simulate(150, 10, seed=5)))


def test_shared_dataset_persists_and_results_are_tied_to_it():
    d = simulate(200, 10, seed=9)
    at = app(ROOT / "Rasch_calibration.py", d)
    at.switch_page("pages/3_JMLE_Lab.py").run()
    button(at, "Run Python JMLE").click().run()
    no_errors(at)
    entry = at.session_state["results"]["JMLE"]
    assert entry["data_id"] == d.identity
    assert any("Converged" in m.value for m in at.markdown)
    # Another page in the same session sees the same dataset and the stored calibration.
    at.switch_page("pages/6_Person_Scoring.py").run()
    no_errors(at)
    assert any(t.value == "6 · Person Scoring" for t in at.title)
    assert at.session_state["dataset"].identity == d.identity
    choices = next(s for s in at.selectbox if s.label == "Item difficulties to score with").options
    assert "JMLE calibration" in choices
    # A new dataset invalidates the old calibration everywhere.
    at.session_state["dataset"] = simulate(200, 10, seed=10)
    at.run()
    choices = next(s for s in at.selectbox if s.label == "Item difficulties to score with").options
    assert "JMLE calibration" not in choices


def test_changed_settings_mark_result_stale():
    at = app(ROOT / "pages/3_JMLE_Lab.py", simulate(200, 10, seed=9))
    button(at, "Run Python JMLE").click().run()
    next(c for c in at.checkbox if c.label.startswith("Bias correction")).uncheck().run()
    no_errors(at)
    assert any("Settings changed" in w.value for w in at.warning)


@pytest.mark.parametrize("page,label,key", [("pages/4_CMLE_Lab.py", "Run Python CMLE", "CMLE"),
                                            ("pages/5_MML_EM_Lab.py", "Run Python MML-EM", "MML")])
def test_cmle_and_mml_pages_compute_results(page, label, key):
    d = simulate(200, 8, seed=12, missing=.05)
    at = app(ROOT / page, d)
    button(at, label).click().run()
    no_errors(at)
    res = at.session_state["results"][key]["result"]
    assert res.converged and np.isfinite(res.b).all()
    assert any(dl for dl in at.get("download_button"))


def test_mml_phases_render():
    at = app(ROOT / "pages/5_MML_EM_Lab.py", simulate(150, 8, seed=2))
    button(at, "Run Python MML-EM").click().run()
    for phase in ["E-step", "M-step", "Likelihood & check"]:
        next(r for r in at.radio if r.label == "Phase").set_value(phase).run()
        no_errors(at)
    assert any("Σ posterior weights" in m.value or "Observed marginal" in m.value for m in at.markdown)


def test_simulation_form_regenerates_reproducibly():
    at = app(ROOT / "pages/2_Data_and_Simulation.py")
    first = at.session_state["dataset"].identity
    next(n for n in at.number_input if n.label == "Random seed").set_value(77)
    next(b for b in at.button if b.label == "Generate new dataset").click().run()
    no_errors(at)
    new = at.session_state["dataset"]
    assert new.identity != first and new.identity == simulate(500, 20, seed=77).identity


def test_person_scoring_and_comparison_pages_run():
    d = simulate(200, 10, seed=4)
    at = app(ROOT / "pages/6_Person_Scoring.py", d)
    no_errors(at)  # true difficulties are available for simulated data
    at = app(ROOT / "pages/7_Estimator_Comparison.py", d)
    button(at, "Run all three estimators").click().run()
    no_errors(at)
    runs = at.session_state["comparison"]["runs"]
    assert all(runs[k]["result"] is not None for k in ("JMLE", "CMLE", "MML"))


def test_upload_validation():
    good = b"pid,i1,i2,i3\n" + b"".join(f"p{k},{k%2},1,{(k//2)%2}\n".encode() for k in range(12))
    d = load_csv(good, "pid")
    assert d.item_ids == ["i1", "i2", "i3"] and d.n_persons == 12 and not d.has_truth
    with pytest.raises(ValueError, match="Non-numeric"):
        load_csv(good)                      # identifier not declared → refused, never used as an item
    with pytest.raises(ValueError, match="0, 1, or blank"):
        load_csv(b"a,b,c\n" + b"2,0,1\n" * 12)
    with pytest.raises(ValueError, match="no observed"):
        load_csv(b"a,b,c\n" + b",,\n" + b"1,0,1\n" * 11)
    with pytest.raises(ValueError, match="Person identifiers must be unique"):
        load_csv(b"pid,b,c,d\n" + b"x,1,0,1\n" * 12, "pid")


def test_monte_carlo_holds_items_fixed_and_reports_empirical_sd():
    at = app(ROOT / "pages/7_Estimator_Comparison.py", simulate(100, 8, seed=1))
    next(n for n in at.number_input if n.label == "Replications").set_value(3)
    next(n for n in at.number_input if n.label == "Persons N").set_value(150)
    next(n for n in at.number_input if n.label == "Items J").set_value(8)
    next(b for b in at.button if b.label == "Run Monte Carlo").click().run()
    no_errors(at)
    rec = at.session_state["monte_carlo"]["records"]
    assert at.session_state["monte_carlo"]["done"] == 3
    # The same true difficulty for each item in every replication.
    assert (rec.dropna(subset=["true"]).groupby("item").true.nunique() == 1).all()
    assert set(rec.method) == {"JMLE", "JMLE_raw", "CMLE", "MML"}


def test_theory_page_shows_recorded_validation():
    """Where R is missing (e.g. Streamlit Cloud), page 8 still shows the recorded local run, labelled as such."""
    at = app(ROOT / "pages/8_Theory_and_Algorithms.py")
    no_errors(at)
    assert any("Recorded, not live" in i.value for i in at.info)


def test_recorded_validation_covers_all_datasets_and_passes():
    import json

    from r_bridge.record_validation import DATASETS, RECORD

    rec = json.loads(RECORD.read_text())
    assert [d["name"] for d in rec["datasets"]] == [n for n, _ in DATASETS]
    for name, _ in DATASETS:
        methods = {r["method"] for r in rec["rows"] if r["dataset"] == name}
        assert len(methods) == 4, (name, methods)          # JMLE, CMLE, MML-EM, WLE
    assert all(r["status"] == "Pass" for r in rec["rows"]), [r for r in rec["rows"] if r["status"] != "Pass"]
