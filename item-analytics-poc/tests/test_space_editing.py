"""Adding/removing elements of the key-feature, instance and distractor spaces, and the space library search."""

import pytest

from src import decomposition as dc
from src import distractors as ds
from src import feature_frequency as ff
from src import space_library as sl
from src import task_alignment as ta
from tests.test_decomposition import FakeLLM, bank, feature, run, span  # noqa: F401 - bank is a fixture


def test_add_elements_to_each_space_with_validated_evidence(bank):
    by_id, review = bank
    store = dc.new_store()
    d = dc.add_feature(store, by_id, "declarative_key", "Tachycardia threshold", "sme",
                       supporting_items=["K-0", "K-1"], excerpt="Which action is first?", importance=5)
    p = dc.add_feature(store, by_id, "procedural_key", "Stabilize before diagnostics", "sme",
                       supporting_items=["K-2"], rationale="Low pressure requires immediate support.",
                       dominance="alternative", procedure_group="stabilize", sequence_order=2)
    i = dc.add_feature(store, by_id, "instance_context", "Specific heart rate", "sme", supporting_items=["K-1"],
                       excerpt="125 bpm")
    x = dc.add_feature(store, by_id, "distractor_feature", "Delays treatment", "sme",
                       option_refs=[("K-1", "C"), ("K-2", "B")], misconception="May underestimate urgency.")
    assert [d[0], p[0], i[0], x[0]] == ["D", "P", "I", "X"]
    fd, fp, fx = store["features"][d], store["features"][p], store["features"][x]
    assert fd["origin"] == "sme" and fd["review_status"] == "accepted" and fd["sme_importance"] == 5
    assert fd["evidence_types"] == ["explicit_text"] and fd["supporting_items"] == ["K-0", "K-1"]
    assert fp["evidence_types"] == ["clinically_inferred"] and (fp["dominance"], fp["sequence_order"]) == ("alternative", 2)
    assert fx["option_refs"] == [{"item_id": "K-1", "option_id": "C"}, {"item_id": "K-2", "option_id": "B"}]
    sme = ff.frequency_table(store, ["K-1", "K-2", "K-3"], "K-0", "sme").set_index("feature_id")
    assert sme.loc[d, "item_frequency"] == 1 and bool(sme.loc[d, "in_target"])
    assert sme.loc[x, "item_frequency"] == 2 and sme.loc[x, "distractor_option_count"] == 2
    assert ff.frequency_table(store, ["K-1"], "K-0", "raw").empty  # SME additions are not AI proposals
    assert [h["action"] for h in store["history"]] == ["add_feature"] * 4


def test_invalid_additions_are_refused(bank):
    by_id, _ = bank
    store = dc.new_store()
    with pytest.raises(ValueError, match="not in K-2"):
        dc.add_feature(store, by_id, "declarative_key", "x", "sme", supporting_items=["K-1", "K-2"], excerpt="125 bpm")
    with pytest.raises(ValueError, match="rationale"):
        dc.add_feature(store, by_id, "instance_context", "x", "sme", supporting_items=["K-1"])
    with pytest.raises(ValueError, match="label"):
        dc.add_feature(store, by_id, "declarative_key", " ", "sme", supporting_items=["K-1"], rationale="r")
    with pytest.raises(ValueError, match="is the key"):
        dc.add_feature(store, by_id, "distractor_feature", "x", "sme", option_refs=[("K-1", "A")])
    with pytest.raises(ValueError, match="no usable key"):
        dc.add_feature(store, by_id, "distractor_feature", "x", "sme", option_refs=[("K-3", "B")])
    with pytest.raises(ValueError, match="does not exist"):
        dc.add_feature(store, by_id, "distractor_feature", "x", "sme", option_refs=[("K-1", "Z")])
    assert store["features"] == {}


def test_remove_and_restore_are_reversible_and_survive_reruns(bank):
    by_id, review = bank
    store = dc.new_store()
    out = [feature("a", "Waiting is unsafe", "distractor_feature", [span("K-1", option="C", excerpt="Wait")]),
           feature("b", "Tachycardia", "declarative_key", [span("K-1", excerpt="Heart rate is 125 bpm")])]
    run(by_id, review, store, FakeLLM(out))
    xa = next(k for k, f in store["features"].items() if f["feature_type"] == "distractor_feature")
    db = next(k for k, f in store["features"].items() if f["feature_type"] == "declarative_key")
    dc.update_feature(store, xa, {"review_status": "accepted"}, "sme")
    dc.update_feature(store, db, {"review_status": "accepted"}, "sme")
    assert dc.remove_features(store, [xa], "sme", "not plausible") == [xa]
    assert store["features"][xa]["review_status"] == "removed"
    sib = ta.accepted_siblings(review)
    assert xa not in set(ff.frequency_table(store, sib, "K-0", "sme")["feature_id"])
    options = ds.option_review(by_id, "K-0", sib, store)
    assert (options["linked_distractor_features"] == "").all()
    _, result = run(by_id, review, store, FakeLLM(out))  # the AI proposes the removed element again
    assert xa in result["pending"] and store["features"][xa]["review_status"] == "removed"
    from src import storage
    from src.ingest import ColumnMapping
    bundle = storage.export_bundle(list(by_id.values()), ColumnMapping(text_columns=["stem"]), None, {"store": store})
    restored = storage.import_bundle(bundle)["review"]["store"]["features"][xa]
    assert (restored["review_status"], restored["removed_from_status"], restored["removal_reason"]) == ("removed", "accepted", "not plausible")
    assert dc.restore_features(store, [xa], "sme") == [xa]
    assert store["features"][xa]["review_status"] == "accepted"
    assert xa in set(ff.frequency_table(store, sib, "K-0", "sme")["feature_id"])
    actions = [h["action"] for h in store["history"]]
    assert "remove_features" in actions and "restore_features" in actions
    assert dc.editor_changes([{"feature_id": db, "review_status": "accepted"}],
                             [{"feature_id": db, "review_status": "removed"}]) == {}


def test_space_library_filters_by_task_domain_type_and_keyword(bank):
    by_id, review = bank
    current = dc.new_store()
    dc.add_feature(current, by_id, "declarative_key", "Perfusion signs", "sme", supporting_items=["K-1"], rationale="r")
    dc.add_feature(current, by_id, "distractor_feature", "Delays treatment", "sme", option_refs=[("K-1", "C")])
    gone = dc.add_feature(current, by_id, "instance_context", "Heart rate value", "sme", supporting_items=["K-1"], excerpt="125 bpm")
    dc.remove_features(current, [gone], "sme")
    other_review = ta.new_review("K-2", ["K-1"])
    other_review["task"] = {"name": "Interpret findings", "decision_or_action": "", "domain": "Renal", "rationale": ""}
    ta.set_task(other_review, "sme", "accepted")
    ta.set_decision(other_review, "sme", "K-1", "same_task")
    other = dc.new_store()
    dc.add_feature(other, by_id, "instance_context", "Pressure finding", "sme", supporting_items=["K-1"], rationale="r")
    spaces = sl.collect_spaces(review, current, [{"task_review": other_review, "store": other}], by_id)
    summary = sl.space_summary(spaces)
    assert summary["task"].tolist() == ["Prioritize", "Interpret findings"]
    assert summary["domain"].tolist() == ["Acute", "Renal"]
    assert summary.loc[0, "Declarative"] == 1 and summary.loc[0, "Instance"] == 0 and summary.loc[0, "removed"] == 1
    table = sl.space_features(spaces)
    assert len(table) == 4
    assert sl.filter_features(table, tasks=["Interpret findings"])["label"].tolist() == ["Pressure finding"]
    assert set(sl.filter_features(table, domains=["Acute"])["label"]) == {"Perfusion signs", "Delays treatment", "Heart rate value"}
    assert sl.filter_features(table, space_types=["Distractor"])["label"].tolist() == ["Delays treatment"]
    assert sl.filter_features(table, text="perfusion")["space"].tolist() == ["current"]
    assert sl.filter_features(table, statuses=["removed"])["label"].tolist() == ["Heart rate value"]
    row = table[table["label"] == "Perfusion signs"].iloc[0]
    assert row["coverage (SME-approved)"] == "1/3 = 33%" and row["origin"] == "SME-added"
