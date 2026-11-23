from src import decomposition as dc
from src import distractors as ds
from src import feature_frequency as ff
from tests.conftest import make_item, records_from_rows

SIBS = ["S1", "S2", "S3", "S4", "S5"]


def store_with(features):
    """features: list of (type, label, supporting items, option refs)."""
    store = dc.new_store()
    proposals = []
    for ftype, label, items, refs in features:
        spans = [{"item_id": i, "field": "stem", "option_id": None, "excerpt": "x", "evidence_type": "explicit_text"} for i in items]
        proposals.append({"canonical_label": label, "definition": "", "feature_type": ftype, "task_relevance": "",
                          "dominance": "dominant" if ftype == "procedural_key" else None, "procedure_group": None,
                          "sequence_order": None, "supporting_items": list(dict.fromkeys(items)), "source_spans": spans,
                          "option_refs": [{"item_id": i, "option_id": o} for i, o in refs], "evidence_types": ["explicit_text"],
                          "inference_rationale": "", "proposed_misconception": None, "llm_uncertainty_note": "",
                          "validation_issues": [], "raw_feature_ids": [label]})
    dc.integrate(store, proposals, {"run_id": "r1"}, "sme")
    for fid in store["order"]:
        dc.update_feature(store, fid, {"review_status": "accepted"}, "sme")
    return store


def row(table, label):
    return table[table["canonical_label"] == label].iloc[0]


def test_unique_item_counting_and_target_exclusion():
    store = store_with([
        ("declarative_key", "In all five", SIBS + ["TARGET"], []),
        ("declarative_key", "In three", ["S1", "S2", "S3"], []),
        ("procedural_key", "Repeated in one item", ["S4", "S4", "S4"], []),
        ("instance_context", "Target only", ["TARGET"], []),
    ])
    t = ff.frequency_table(store, SIBS + ["TARGET", "S1"], "TARGET")
    r = row(t, "In all five")
    assert (r["item_frequency"], r["denominator_N"], r["coverage_pct"], r["coverage"]) == (5, 5, 100.0, "5/5 = 100%")
    assert bool(r["in_target"])
    r = row(t, "In three")
    assert (r["item_frequency"], r["coverage_pct"], r["coverage"]) == (3, 60.0, "3/5 = 60%")
    assert row(t, "Repeated in one item")["item_frequency"] == 1
    r = row(t, "Target only")
    assert r["item_frequency"] == 0 and bool(r["in_target"]) and r["denominator_N"] == 5


def test_empty_sibling_set_reports_not_applicable():
    store = store_with([("declarative_key", "Any", ["TARGET"], [])])
    r = ff.frequency_table(store, [], "TARGET").iloc[0]
    assert r["denominator_N"] == 0 and r["coverage_pct"] is None and r["coverage"].startswith("N/A")


def test_option_count_differs_from_item_frequency():
    store = store_with([("distractor_feature", "One misconception", ["S1"], [("S1", "B"), ("S1", "C"), ("S1", "D")])])
    r = ff.frequency_table(store, SIBS, "TARGET").iloc[0]
    assert r["item_frequency"] == 1 and r["distractor_option_count"] == 3


def test_sme_and_raw_incidence_are_both_available():
    store = store_with([("declarative_key", "Edited", ["S1", "S2"], []), ("declarative_key", "Rejected", ["S3"], [])])
    fid_e, fid_r = store["order"]
    dc.update_feature(store, fid_e, {"supporting_items": ["S1"]}, "sme")
    dc.update_feature(store, fid_r, {"review_status": "rejected"}, "sme")
    sme = ff.frequency_table(store, SIBS, "TARGET", "sme")
    raw = ff.frequency_table(store, SIBS, "TARGET", "raw")
    assert list(sme["canonical_label"]) == ["Edited"] and sme.iloc[0]["item_frequency"] == 1
    assert sme.iloc[0]["review_status"] == "edited"
    assert row(raw, "Edited")["item_frequency"] == 2 and row(raw, "Rejected")["item_frequency"] == 1
    matrix = ff.incidence_matrix(store, ["TARGET"] + SIBS, "raw")
    assert matrix.loc["S2", fid_e] == 1 and matrix.loc["TARGET"].sum() == 0
    long = ff.long_incidence(store, "TARGET", SIBS, "sme")
    assert set(long.columns) == {"item_id", "role", "feature_id", "feature_type", "value"}
    assert long[(long["item_id"] == "S1") & (long["feature_id"] == fid_e)]["value"].item() == 1


def test_distractor_review_shows_all_options_and_pending_keys():
    rows = [make_item("A-1", "Q?", ("k", "d1", "d2", "d3", "d4", "d5"), key="A"),
            make_item("A-2", "Q?", ("x", "y"), key="")]
    records, _, _ = records_from_rows(rows)
    by_id = {r["item_id"]: r for r in records}
    table = ds.option_review(by_id, "A-1", ["A-2"])
    assert len(table) == 8
    a1 = table[table["item_id"] == "A-1"]
    assert a1["option_role"].tolist() == ["key"] + ["distractor"] * 5
    a2 = table[table["item_id"] == "A-2"]
    assert (a2["option_role"] == "unknown").all() and a2["analysis_status"].str.contains("pending").all()
