"""Offline acceptance checks for the feature and distractor authoring views."""

from copy import deepcopy
import json

import pytest
from streamlit.testing.v1 import AppTest

import authoring_spaces as spaces
import core
import decision_model as dm
import specialist_workflow as sw
import storage
from test_specialist_workflow import evidence_payload, fit_case, score_model, task_case


def positive_fit():
    case, item, model = fit_case()
    case = sw.save_model(case, item["id"], model, "Fixture SME")
    return sw.accept_scenario(case, item["id"], "fit-positive", "Fixture SME", "Fictional FIT exercise.")


def review_fit(case, action, *, cid=None, decision="Selected"):
    clinical = {
        "colonoscopy": "The valid positive report calls for diagnostic follow-up.",
        "annual": "The positive report requires follow-up before the routine screening interval.",
        "repeat": "The sample already yielded a valid positive report.",
    }
    appeal = {"annual": "Apply the routine negative screening interval.",
              "repeat": "Repeat the test to confirm the positive report."}
    inactive = {"annual": "fit-negative-rule", "repeat": "fit-rejected-rule"}
    return spaces.review_candidate(case, case["items"][0]["id"], cid or "modeled-" + action, {
        "action_id": action, "clinical_rationale": clinical[action], "plausibility_rationale": appeal.get(action),
        "attractive_fact_ids": ["fact-fit_result"], "violated_rule_ids": [inactive[action]] if action in inactive else [],
        "misconception_hypothesis": appeal.get(action),
    }, decision, "Fixture SME", "fit-positive")


def rows(case, scenario="fit-positive", changes=None):
    item = case["items"][0]
    ctx = spaces.space_context(case, item, scenario, changes)
    return ctx, dm.by_id(spaces.candidate_rows(case, item, ctx), "candidate_id")


def test_coupled_fit_combinations_visibility_and_confirmation():
    case = positive_fit()
    original = deepcopy(case)
    item = case["items"][0]
    ctx, _ = rows(case)
    detail = spaces.feature_rows(case, item, ctx)["fit_result"]
    assert detail["Coupled features"] == "lab_status"
    assert detail["Feasibility constraints"] == "final-report"
    assert detail["Value"] == "positive" and detail["Disclosure"] == case["stages"][1]["title"]
    assert "fit-positive-rule" in detail["Rules"] and "fact-fit_result" in detail["Linked facts"]
    assert {sid: r["keyed_action_ids"] for sid, r in ctx["rows"].items()} == {
        "fit-positive": ["colonoscopy"], "fit-negative": ["annual"], "fit-rejected": ["repeat"]}
    combinations = {}
    for lab in ("processed", "rejected"):
        for report in ("positive", "negative", "no_reportable_result"):
            trial, _ = rows(case, changes={"lab_status": lab, "fit_result": report})
            combinations[lab, report] = trial["row"]["outcome"]
    assert list(combinations.values()).count("infeasible") == 3
    assert combinations["processed", "negative"] == combinations["rejected", "no_reportable_result"] == "key_changed"
    with pytest.raises(ValueError, match="feasible"):
        spaces.confirm_trial(case, item["id"], "fit-positive", {"lab_status": "rejected"}, "SME", "Check report")
    new = spaces.confirm_trial(case, item["id"], "fit-positive",
                               {"lab_status": "rejected", "fit_result": "no_reportable_result"}, "SME", "Check report")
    assert new["items"][0]["decision_workflow"]["selection"]["keyed_action_ids"] == ["repeat"]
    assert [f["value"] for f in new["facts"]] == ["rejected", "no_reportable_result"]
    assert case == original
    earlier = deepcopy(case)
    earlier["items"][0]["stage_id"] = earlier["stages"][0]["id"]
    hidden, _ = rows(earlier)
    assert hidden["row"]["outcome"] == "undetermined" and hidden["visible"] == set()
    assert hidden["row"]["missing_features"] == ["fit_result", "lab_status"]
    assert spaces.feature_rows(earlier, earlier["items"][0], hidden)["fit_result"]["Value"] == "Withheld at this stage"
    with pytest.raises(ValueError, match="disclosed"):
        rows(earlier, changes={"fit_result": "negative"})


def test_candidate_dispositions_shortage_switch_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("ITEMDEV_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ITEMDEV_HOSTED", raising=False)
    case = storage.save_case(positive_fit())
    item_id = case["items"][0]["id"]
    manual_id = case["items"][0]["options"][0]["id"]
    original_options = deepcopy(case["items"][0]["options"])
    for action in ("colonoscopy", "annual", "repeat"):
        case = storage.save_case(review_fit(case, action), "SME candidate decision", expected_revision=case["version"])
    ctx, candidates = rows(case)
    assert candidates["modeled-annual"]["neighbors"] == ["fit-negative"]
    assert candidates["modeled-repeat"]["neighbors"] == ["fit-rejected"]
    for action in ("annual", "repeat"):
        row = candidates["modeled-" + action]
        assert row["local_judgment"] == "eligible"
        assert row["clinical_rationale"] and row["plausibility_rationale"]
        assert row["evidence_status"] == "author hypothesis"
        assert {r["rule_id"] for r in row["exclusions"]} == set(row["violated_rule_ids"])
    assert candidates["modeled-colonoscopy"]["local_judgment"] == "keyed"
    assert candidates[manual_id]["local_judgment"] == "review"
    with pytest.raises(ValueError, match="Duplicate"):
        review_fit(case, "annual", cid=manual_id)
    held = review_fit(case, "annual", decision="Hold")
    assert rows(held)[1]["modeled-annual"]["local_judgment"] == "review"
    assert sw.target_count(case["items"][0]) - sum(r["local_judgment"] != "review" for r in candidates.values()) == 1
    assembled = storage.save_case(spaces.assemble_reviewed(case, item_id, "fit-positive", "SME"), "Assemble", expected_revision=case["version"])
    choices = assembled["items"][0]["options"]
    assert len(choices) == 3 and next(o for o in choices if o["correct"])["modeled_action_id"] == "colonoscopy"
    assert assembled["items"][0]["decision_workflow"]["options"]["scenario_id"] == "fit-positive"
    assert set(r["decision"] for r in rows(assembled)[1].values() if r["action_id"]) == {"Selected"}
    assert core.normalize_import(json.loads(json.dumps(assembled)))["items"][0]["decision_workflow"]["candidate_reviews"]
    trial, stale = rows(assembled, changes={"fit_result": "negative"})
    assert trial["row"]["keyed_action_ids"] == ["annual"] and not trial["confirmed"]
    assert all(stale["modeled-" + a]["decision"] == "Reassess" for a in ("annual", "colonoscopy", "repeat"))
    with pytest.raises(ValueError, match="Confirm"):
        spaces.assemble_reviewed(assembled, item_id, "fit-negative", "SME")
    negative = spaces.confirm_trial(assembled, item_id, "fit-positive", {"fit_result": "negative"}, "SME", "Changed report")
    assert rows(negative, "fit-negative")[0]["row"]["keyed_action_ids"] == ["annual"]
    assert all(r["local_judgment"] == "review" for r in rows(negative, "fit-negative")[1].values())
    assert negative["items"][0]["options"] == choices
    with pytest.raises(ValueError, match="Select the keyed action"):
        spaces.assemble_reviewed(negative, item_id, "fit-negative", "SME")
    restored = storage.recover(case["id"])
    assert restored["items"][0]["options"] == original_options
    assert set(restored["items"][0]["decision_workflow"]["candidate_reviews"]) == {"modeled-annual", "modeled-colonoscopy", "modeled-repeat"}


def test_changes_in_the_item_editor_are_shown_before_reassembly():
    case = positive_fit()
    item_id = case["items"][0]["id"]
    for action in ("colonoscopy", "annual", "repeat"):
        case = review_fit(case, action)
    case = spaces.assemble_reviewed(case, item_id, "fit-positive", "Fixture SME")
    annual = next(o for o in case["items"][0]["options"] if o["modeled_action_id"] == "annual")
    annual["text"] = "Changed in the item editor"
    row = rows(case)[1]["modeled-annual"]
    assert row["decision"] == "Reassess" and row["text"] == annual["text"]
    revised = spaces.review_candidate(case, item_id, "modeled-annual", {"text": "SME revision of that edit"},
                                      "Selected", "Fixture SME", "fit-positive")
    assert rows(revised)[1]["modeled-annual"]["text"] == "SME revision of that edit"


def test_feature_candidate_sources_and_stable_existing_option_id():
    case, item = task_case()
    item_id = item["id"]
    option_ids = [o["id"] for o in item["options"]]
    case = sw.accept_evidence(case, item_id, evidence_payload(), "Fixture SME")
    case = sw.save_model(case, item_id, score_model(case), "Fixture SME")
    case = spaces.edit_feature(case, item_id, "score", {"label": "Reviewed score", "unit": "points"}, ["evidence-1"], "SME")
    model = case["items"][0]["decision_workflow"]["model"]
    assert model["feature_sources"] == {"score": ["evidence-1"]}
    assert model["source_links"]["evidence-1"]["locator"] == "Rule 1"
    with pytest.raises(ValueError, match="source"):
        spaces.edit_feature(case, item_id, "score", {}, ["outside-item-source"], "SME")
    with pytest.raises(ValueError, match="range"):
        spaces.edit_feature(case, item_id, "score", {"allowed_values": [9]}, [], "SME")
    case = sw.accept_scenario(case, item_id, "below", "SME", "Reviewed synthetic threshold")
    for oid, action in zip(option_ids, ("A", "B")):
        case = spaces.review_candidate(case, item_id, oid, {
            "action_id": action, "clinical_rationale": "Reviewed synthetic rule", "evidence_claim_ids": ["claim-1"],
            "bank_item_ids": ["bank-1"], "plausibility_rationale": None if action == "A" else "Interpret 9 as 10",
            "violated_rule_ids": [] if action == "A" else ["high"]}, "Selected", "SME", "below")
    case = spaces.assemble_reviewed(case, item_id, "below", "SME")
    assert {o["id"] for o in case["items"][0]["options"]} == set(option_ids)
    references = case["items"][0]["decision_workflow"]["candidate_reviews"][option_ids[1]]["source_links"]
    assert case["items"][0]["decision_workflow"]["candidate_reviews"][option_ids[1]]["claim_sources"] == {"claim-1": "evidence-1"}
    assert references["evidence-1"]["url"] == "https://example.test/fixture" and references["bank-1"]["locator"] == "Summary 1"
    case["specialist_sources"].clear()
    assert sw.reviewed_source(case, case["items"][0], "bank-1") is None
    assert references["bank-1"]["source_date"] == "Fixture v2"
    assert all(r["local_judgment"] == "review" for r in rows(case, "below")[1].values())


def test_trial_respects_fixed_scope_and_scenario_specific_rules():
    case, item = task_case()
    model = score_model(case, evidence=False)
    model["features"][1]["variation"] = "fixed"
    model["scenarios"] = model["scenarios"][:1]
    model["decision_rules"][1]["condition"]["scenario"] = "below"
    case = sw.save_model(case, item["id"], model, "SME")
    item = case["items"][0]
    with pytest.raises(ValueError, match="variable feature"):
        spaces.space_context(case, item, "below", {"exception": True})
    trial = spaces.space_context(case, item, "below", {"score": 10})
    assert trial["trial"] and trial["row"]["outcome"] == "undetermined"
    with pytest.raises(ValueError, match="determinate"):
        spaces.confirm_trial(case, item["id"], "below", {"score": 10}, "SME", "Synthetic boundary")
    # A changed fixed observation stays outside review scope when varying a different feature.
    model["decision_rules"][1]["condition"].pop("scenario")
    case = sw.save_model(case, item["id"], model, "SME")
    case["facts"][1]["value"] = True
    row = spaces.space_context(case, case["items"][0], "below", {"score": 10})["row"]
    assert row["outcome"] == "undetermined" and any("fixed decision scope" in r for r in row["reasons"])


def test_review_ui_renders_and_commits_locally():
    at = AppTest.from_string("""
import streamlit as st
import specialist_workflow as sw
from test_authoring_spaces import positive_fit
from ui_authoring_spaces import render_authoring_spaces
case = st.session_state.setdefault("fixture_case", positive_fit())
item = case["items"][0]
def commit(candidate, reason, specialist):
    candidate["version"] += 1
    st.session_state["fixture_case"] = candidate
render_authoring_spaces(case, item, sw.model_status(case, item), "SME", "Synthetic exercise", "test:", commit)
""").run()
    assert not at.exception
    assert {t.label for t in at.tabs} == {"Feature space", "Distractor space"}
    assert any("Arrange diagnostic colonoscopy" in block.value for block in at.markdown)
    next(s for s in at.selectbox if s.label == "Candidate to review").select("modeled-annual").run()
    next(t for t in at.text_area if t.label == "Why would an examinee choose it?").set_value("Recall the routine interval.")
    next(t for t in at.text_area if t.label == "Why is it keyed, inferior, or unresolved here?").set_value("The positive report requires follow-up.")
    next(s for s in at.multiselect if s.label == "Rules explaining exclusion").set_value(["fit-negative-rule"])
    next(s for s in at.selectbox if s.label == "SME disposition").select("Selected")
    next(b for b in at.button if b.label == "Save candidate decision").click().run()
    assert not at.exception and not at.error
    assert at.session_state["fixture_case"]["items"][0]["decision_workflow"]["candidate_reviews"]["modeled-annual"]["decision"] == "Selected"
    selector = next(s for s in at.selectbox if s.label == "fit_result")
    selector.select(1).run()
    assert not at.exception
    assert any("Repeat screening FIT in one year" in block.value for block in at.markdown)
    next(b for b in at.button if b.label == "Confirm trial scenario and key").click().run()
    assert not at.exception and not at.error
    state = at.session_state["fixture_case"]["items"][0]["decision_workflow"]
    assert state["selection"]["scenario_id"] == "fit-negative" and state["selection"]["actor"] == "SME"
    feature = next(t for t in at.text_input if t.label == "Feature label")
    feature.set_value("Laboratory status")
    next(b for b in at.button if b.label == "Save reviewed feature").click().run()
    assert not at.exception and not at.error
    assert at.session_state["fixture_case"]["items"][0]["decision_workflow"]["model"]["payload"]["features"][0]["label"] == "Laboratory status"
