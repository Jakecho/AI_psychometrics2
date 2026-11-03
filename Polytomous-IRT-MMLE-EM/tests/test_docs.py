"""Documented numerical examples must equal what the engine produces now (acceptance item 16)."""
import importlib.util

from conftest import ROOT


def test_numerical_examples_document_matches_engine():
    spec = importlib.util.spec_from_file_location("mk", ROOT / "docs" / "make_numerical_examples.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    assert (ROOT / "docs" / "NUMERICAL_EXAMPLES.md").read_text() == mk.build()


def test_recorded_validation_matches_cases_and_report():
    """docs/validation_record.json (shown on page 12) covers every case, and agreement rows meet their tolerance."""
    import json
    from r_reference.cases import CASES, DIFFERENCES

    rec = json.loads((ROOT / "docs" / "validation_record.json").read_text())
    agree = [r for r in rec["rows"] if r["kind"] == "agreement"]
    assert len(agree) == len(CASES) and len(rec["rows"]) == len(CASES) + len(DIFFERENCES)
    for r, (pkg, _, _, tol) in zip(agree, CASES):
        assert r["package"] == pkg and r["status"] == "Executed"
        assert r["max_item_diff"] < tol and r["max_moment_diff"] < max(tol, 1e-4)
    assert f"({rec['date']})" in (ROOT / "docs" / "VALIDATION_RESULTS.md").read_text().splitlines()[0]
