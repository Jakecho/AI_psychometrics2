import pytest

from src import decomposition as dc
from src import storage
from src import task_alignment as ta
from src.ingest import ColumnMapping
from tests.conftest import make_item, records_from_rows


def feature(fid, label, ftype, spans, items=None, **extra):
    base = {"feature_id": fid, "canonical_label": label, "definition": f"def {label}", "feature_type": ftype,
            "task_relevance": "", "dominance": None, "procedure_group": None, "sequence_order": None,
            "supporting_items": items if items is not None else sorted({s["item_id"] for s in spans}),
            "source_spans": spans, "inference_rationale": "", "proposed_misconception": None, "llm_uncertainty_note": ""}
    base.update(extra)
    return base


def span(item, field="stem", excerpt="", kind="explicit_text", option=None):
    return {"item_id": item, "field": "option" if option else field, "option_id": option, "excerpt": excerpt, "evidence_type": kind}


class FakeLLM:
    model = "fake"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.payloads = []

    def complete(self, system, payload, schema, name, sleep=None):
        self.payloads.append(payload)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return {"schema_version": "1.0", "target_item_id": payload["target_item_id"], "features": out}, {"input_tokens": 1, "output_tokens": 1}


@pytest.fixture
def bank():
    rows = [make_item("K-0", "Heart rate is 130 bpm. Which action is first?", ("Give fluids", "Wait", "Walk", "Eat"), key="A"),
            make_item("K-1", "Heart rate is 125 bpm. Which action is first?", ("Give fluids", "Rest", "Wait", "Read"), key="A"),
            make_item("K-2", "Pressure is low. Which action is first?", ("Give fluids", "Wait", "Sleep"), key="A"),
            make_item("K-3", "No key here. Which action is first?", ("One", "Two"), key="")]
    records, _, _ = records_from_rows(rows)
    by_id = {r["item_id"]: r for r in records}
    review = ta.new_review("K-0", ["K-1", "K-2", "K-3"])
    for c in ("K-1", "K-2", "K-3"):
        ta.set_decision(review, "sme", c, "same_task")
    review["task"] = {"name": "Prioritize", "decision_or_action": "first action", "domain": "Acute", "rationale": "r"}
    ta.set_task(review, "sme", "accepted")
    return by_id, review


def run(by_id, review, store, llm, **kw):
    return dc.decompose(llm, by_id["K-0"], [by_id[i] for i in ta.accepted_siblings(review)], review, store, "sme", **kw)


def test_decomposition_requires_confirmed_task(bank):
    by_id, review = bank
    review["task_status"] = "unreviewed"
    with pytest.raises(dc.DecompositionGateError):
        run(by_id, review, dc.new_store(), FakeLLM([]))


def test_evidence_types_are_preserved_and_fabrications_rejected(bank):
    by_id, review = bank
    out = [
        feature("a", "Tachycardia signals compromise", "declarative_key", [span("K-1", excerpt="Heart rate is 125 bpm")]),
        feature("b", "Fluid resuscitation restores perfusion", "declarative_key",
                [span("K-2", kind="clinically_inferred")], inference_rationale="Low pressure implies hypoperfusion."),
        feature("c", "Fabricated quote", "declarative_key", [span("K-1", excerpt="The patient is septic")]),
        feature("d", "Inference without reasoning", "declarative_key", [span("K-2", kind="clinically_inferred")]),
        feature("e", "Ghost support", "procedural_key", [span("K-1", excerpt="Which action is first?")], items=["K-1", "K-99"]),
        feature("f", "Waiting is unsafe", "distractor_feature", [span("K-1", option="C", excerpt="Wait")]),
        feature("g", "Key is not a distractor", "distractor_feature", [span("K-1", option="A", excerpt="Give fluids")]),
        feature("h", "No key distractor", "distractor_feature", [span("K-3", option="B", excerpt="Two")]),
        feature("i", "Bad option", "distractor_feature", [span("K-1", option="Z", excerpt="x")]),
    ]
    store = dc.new_store()
    run(by_id, review, store, FakeLLM(out))
    f = {x["ai"]["raw_feature_ids"][0]: x for x in store["features"].values()}
    assert f["a"]["evidence_types"] == ["explicit_text"] and f["a"]["review_status"] == "pending"
    assert f["b"]["evidence_types"] == ["clinically_inferred"] and f["b"]["supporting_items"] == ["K-2"]
    assert f["c"]["supporting_items"] == [] and f["c"]["review_status"] == "needs_adjudication"
    assert any("verbatim" in i for i in f["c"]["validation_issues"])
    assert f["d"]["review_status"] == "needs_adjudication" and any("rationale" in i for i in f["d"]["validation_issues"])
    assert f["e"]["supporting_items"] == ["K-1"] and any("K-99" in i for i in f["e"]["validation_issues"])
    assert f["e"]["dominance"] == "unknown"
    assert f["f"]["option_refs"] == [{"item_id": "K-1", "option_id": "C"}] and f["f"]["review_status"] == "pending"
    assert f["g"]["supporting_items"] == [] and any("is the key" in i for i in f["g"]["validation_issues"])
    assert f["h"]["supporting_items"] == [] and any("pending" in i for i in f["h"]["validation_issues"])
    assert f["i"]["supporting_items"] == [] and any("unknown option" in i for i in f["i"]["validation_issues"])


def test_arbitrary_counts_dedupe_within_type_and_distinct_procedures(bank):
    by_id, review = bank
    q = "Which action is first?"
    out = [feature(f"d{n}", f"Declarative concept {n}", "declarative_key", [span("K-1", excerpt="Heart rate")]) for n in range(7)]
    out += [
        feature("p1", "Assess then act", "procedural_key", [span("K-1", excerpt=q)], dominance="dominant", sequence_order=1),
        feature("p2", "Assess then act", "procedural_key", [span("K-2", excerpt=q)], dominance="dominant"),
        feature("p3", "Act immediately", "procedural_key", [span("K-2", excerpt=q)], dominance="alternative"),
        feature("p4", "Assess then act", "procedural_key", [span("K-3", excerpt=q)], dominance="contingent"),
        feature("x1", "Assess then act", "declarative_key", [span("K-1", excerpt=q)]),
        feature("dup", "declarative   CONCEPT 0!", "declarative_key", [span("K-2", excerpt="Pressure is low")]),
    ]
    store = dc.new_store()
    run(by_id, review, store, FakeLLM(out))
    feats = list(store["features"].values())
    decl = [f for f in feats if f["feature_type"] == "declarative_key"]
    proc = [f for f in feats if f["feature_type"] == "procedural_key"]
    assert len(decl) == 8  # 7 concepts + the cross-type "Assess then act" label, which is not merged across types
    merged = next(f for f in decl if f["canonical_label"] == "Declarative concept 0")
    assert merged["supporting_items"] == ["K-1", "K-2"] and merged["ai"]["raw_feature_ids"] == ["d0", "dup"]
    assert len(proc) == 3  # same label merges only with the same dominance
    dominant = next(f for f in proc if f["dominance"] == "dominant")
    assert dominant["supporting_items"] == ["K-1", "K-2"]
    assert {f["dominance"] for f in proc} == {"dominant", "alternative", "contingent"}
    assert all(fid[0] in "DPIX" for fid in store["features"])


def test_sme_edits_survive_reruns_and_pending_updates_need_confirmation(bank):
    by_id, review = bank
    first = [feature("a", "Tachycardia signals compromise", "declarative_key", [span("K-1", excerpt="Heart rate is 125 bpm")]),
             feature("b", "Untouched", "declarative_key", [span("K-1", excerpt="Heart rate")])]
    store = dc.new_store()
    run(by_id, review, store, FakeLLM(first))
    fid_a = next(k for k, f in store["features"].items() if f["canonical_label"].startswith("Tachy"))
    fid_b = next(k for k, f in store["features"].items() if f["canonical_label"] == "Untouched")
    dc.update_feature(store, fid_a, {"review_status": "accepted", "sme_importance": 5, "sme_note": "core"}, "sme")
    with pytest.raises(ValueError):
        dc.update_feature(store, fid_a, {"sme_importance": 9}, "sme")
    second = [feature("a2", "Tachycardia signals compromise", "declarative_key",
                      [span("K-1", excerpt="Heart rate is 125 bpm"), span("K-2", excerpt="Pressure is low")]),
              feature("b2", "Untouched", "declarative_key", [span("K-2", excerpt="Pressure")])]
    _, result = run(by_id, review, store, FakeLLM(second))
    a = store["features"][fid_a]
    assert (a["review_status"], a["sme_importance"], a["supporting_items"]) == ("accepted", 5, ["K-1"])
    assert result["pending"] == [fid_a] and result["refreshed"] == [fid_b]
    assert store["features"][fid_b]["supporting_items"] == ["K-2"]
    dc.accept_pending_update(store, fid_a, "sme")
    assert a["supporting_items"] == ["K-1", "K-2"] and a["sme_importance"] == 5
    assert [h["action"] for h in store["history"]].count("accept_pending_update") == 1


def test_merge_split_and_procedural_order_persist_through_bundle(bank):
    by_id, review = bank
    q = "Which action is first?"
    out = [feature("p", "Check pulse first", "procedural_key", [span("K-1", excerpt=q), span("K-2", excerpt=q)], dominance="dominant"),
           feature("p2", "Pulse check", "procedural_key", [span("K-3", excerpt=q)], dominance="dominant"),
           feature("d", "A fact", "declarative_key", [span("K-1", excerpt="Heart rate")])]
    store = dc.new_store()
    run(by_id, review, store, FakeLLM(out))
    p1, p2 = [k for k, f in store["features"].items() if f["feature_type"] == "procedural_key"]
    d1 = next(k for k, f in store["features"].items() if f["feature_type"] == "declarative_key")
    with pytest.raises(ValueError):
        dc.merge_features(store, p1, [d1], "sme")
    dc.merge_features(store, p1, [p2], "sme")
    assert store["features"][p1]["supporting_items"] == ["K-1", "K-2", "K-3"]
    assert store["features"][p2]["review_status"] == "merged" and store["features"][p2]["merged_into"] == p1
    new = dc.split_feature(store, p1, ["K-3"], "Pulse check (variant)", "sme")
    assert store["features"][p1]["supporting_items"] == ["K-1", "K-2"] and store["features"][new]["supporting_items"] == ["K-3"]
    dc.update_feature(store, new, {"dominance": "alternative", "sequence_order": 2, "review_status": "accepted"}, "sme")
    records = list(by_id.values())
    bundle = storage.export_bundle(records, ColumnMapping(text_columns=["stem"]), None, {"store": store, "task_review": review})
    restored = storage.import_bundle(bundle)["review"]
    rs = restored["store"]
    assert rs["features"][new]["dominance"] == "alternative" and rs["features"][new]["sequence_order"] == 2
    assert rs["features"][p2]["merged_into"] == p1 and rs["order"] == store["order"]
    assert [h["action"] for h in rs["history"]] == [h["action"] for h in store["history"]]
    assert restored["task_review"]["decisions"] == review["decisions"]


def test_chunking_keeps_target_and_partial_results(bank):
    by_id, review = bank
    sibs = [by_id[i] for i in ta.accepted_siblings(review)]
    payloads = dc.build_payloads(by_id["K-0"], sibs, review["task"], max_chars=1200)
    assert len(payloads) == 3 and all(p["target"]["item_id"] == "K-0" for p in payloads)
    assert sorted(i for p in payloads for i in p["accepted_sibling_ids"]) == ["K-1", "K-2", "K-3"]
    ok = [feature("a", "Kept from chunk one", "declarative_key", [span("K-1", excerpt="Heart rate")])]
    store = dc.new_store()
    llm = FakeLLM(ok, RuntimeError("boom"), [])
    run_record, result = run(by_id, review, store, llm, max_chars=1200)
    assert len(result["added"]) == 1 and run_record["errors"] and "Chunk 2" in run_record["errors"][0]
    assert len(run_record["raw_outputs"]) == 2


def test_review_table_edits_match_by_feature_id_not_position():
    before = [{"feature_id": "P001", "canonical_label": "a", "definition": "", "review_status": "pending", "sme_note": "",
               "procedure_group": "", "importance": "unrated", "dominance": "dominant", "sequence_order": float("nan")},
              {"feature_id": "P002", "canonical_label": "b", "definition": "", "review_status": "pending", "sme_note": "",
               "procedure_group": "", "importance": "unrated", "dominance": "alternative", "sequence_order": 2.0}]
    # rows come back in a different order with edits to P002 only
    after = [dict(before[1], importance="4", review_status="accepted", dominance="contingent", sequence_order=3.0,
                  procedure_group=""),
             dict(before[0])]
    assert dc.editor_changes(before, after) == {
        "P002": {"review_status": "accepted", "sme_importance": 4, "dominance": "contingent", "sequence_order": 3}}
    merged_attempt = [dict(before[0], review_status="merged")]
    assert dc.editor_changes(before, merged_attempt) == {}
    assert dc.editor_changes(before, [dict(before[0], importance="unrated", procedure_group="g1")]) == {"P001": {"procedure_group": "g1"}}
