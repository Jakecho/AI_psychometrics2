"""Synthetic checks for explicit evidence review and the permission boundary at handoff."""

from copy import deepcopy
import json

import pytest

import ai_adapter as ai
import authoring_spaces as spaces
import core
import specialist_workflow as sw
import storage
from test_specialist_workflow import SETTINGS, evidence_payload, option, score_model, task_case, transport


def selected_case():
    case, item = task_case()
    passage = case["specialist_sources"].pop(0) | {
        "providers": ["anthropic"], "citation": "Synthetic evidence report, edition 1",
        "locator": "decisions.md#Threshold", "population": "Fictional adult clinic cohort",
    }
    case = sw.replace_retrieved_sources(case, item["id"], [passage], "Fixture reviewer")
    payload = evidence_payload()
    payload["claims"][0]["locator"] = passage["locator"]
    return case, item["id"], payload


def test_only_applicable_claims_support_rules_and_keep_the_passage_locator():
    case, iid, claims = selected_case()
    reviewed = sw.accept_evidence(case, iid, claims, "Fixture SME")
    assert not sw._claims(reviewed, reviewed["items"][0])
    with pytest.raises(ValueError):
        sw.save_model(reviewed, iid, score_model(reviewed), "Fixture SME")
    with pytest.raises(ValueError, match="retrieved clinical"):
        sw.accept_evidence(case, iid, claims, "Fixture SME", applicable_claim_ids=["unknown"])
    missing_scope = deepcopy(claims)
    missing_scope["claims"][0]["applicability"] = " "
    with pytest.raises(ValueError, match="applies"):
        sw.accept_evidence(case, iid, missing_scope, "Fixture SME", applicable_claim_ids=["claim-1"])

    reviewed = sw.accept_evidence(case, iid, claims, "Fixture SME", applicable_claim_ids=["claim-1"])
    item = reviewed["items"][0]
    assert sw.build_request(reviewed, item, "key_feature_modeling", "anthropic")["role_input"]["evidence_claims"] == claims["claims"]
    assert not sw.build_request(reviewed, item, "key_feature_modeling", "openai")["role_input"]["evidence_claims"]
    modeled = sw.save_model(reviewed, iid, score_model(reviewed), "Fixture SME")
    modeled = sw.accept_scenario(modeled, iid, "below", "Fixture SME", "Synthetic reviewed rule.")
    modeled = sw.record_human_decision(modeled, iid, "Fixture SME", "Reviewed wording.", "drafting")
    candidates = [option("candidate-a", "A", "Action A", keyed=True),
                  option("candidate-b", "B", "Action B", rule_id="high", bank_item_ids=["bank-1"])]
    candidates[1]["evidence_claim_ids"] = ["claim-1"]
    assembled = sw.assemble_candidate_options(modeled, iid, {"scenario_id": "below", "options": candidates,
        "rejected_candidates": [], "shortage_count": 0}, "Fixture SME")
    result = sw.replace_retrieved_sources(assembled, iid, [], "Fixture reviewer")
    state = result["items"][0]["decision_workflow"]
    for artifact in ("model", "options", "evidence"):
        assert state[artifact]["claim_sources"] == {"claim-1": "evidence-1"}
        assert state[artifact]["source_links"]["evidence-1"]["locator"] == "decisions.md#Threshold"
        assert state[artifact]["source_links"]["evidence-1"]["citation"].startswith("Synthetic evidence")
    assert state["options"]["source_links"]["bank-1"]["locator"] == "Summary 1"
    assert not sw._claims(result, result["items"][0])
    assert core.normalize_import(result)["items"][0]["decision_workflow"]["model"] == state["model"]


def test_bindings_and_provider_restrictions_hold_on_import_and_projection():
    case, iid, _ = selected_case()
    item = case["items"][0]
    selected = item["decision_workflow"]["retrieved_sources"][0]
    assert selected["item_ids"] == [iid] and selected["stage_ids"] == [item["stage_id"]]
    assert "decisions.md" not in json.dumps(sw.build_request(case, item, "evidence_review", "openai"))
    assert "decisions.md" in json.dumps(sw.build_request(case, item, "evidence_review", "anthropic"))
    manual = case["specialist_sources"] + [selected]
    with pytest.raises(ValueError, match="conflicts"):
        sw.validate_manual_sources(case, manual)
    invalid = deepcopy(case)
    invalid["items"][0]["decision_workflow"]["retrieved_sources"][0].pop("stage_ids")
    with pytest.raises(ValueError, match="bound"):
        core.normalize_import(invalid)
    stage_changed = deepcopy(case)
    stage_changed["items"][0]["stage_id"] = "different"
    assert not sw.reviewed_source(stage_changed, stage_changed["items"][0], selected["source_id"])

    modeled = sw.save_model(case, iid, score_model(case, evidence=False), "Fixture SME")
    modeled = spaces.edit_feature(modeled, iid, "score", {}, [selected["source_id"]], "Fixture SME")
    modeled = sw.accept_scenario(modeled, iid, "below", "Fixture SME", "Recorded synthetic evidence gap.")
    with pytest.raises(ValueError, match="source linked to a visible feature"):
        sw.build_request(modeled, modeled["items"][0], "item_drafting", "openai")
    assert sw.build_request(modeled, modeled["items"][0], "item_drafting", "anthropic")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_mocked_transport_sees_only_selected_authorized_sources(monkeypatch, provider):
    case, iid, claims = selected_case()
    item = case["items"][0]
    for key, value in SETTINGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ITEMDEV_AGENT_ROUTES", json.dumps({"evidence_review": {"provider": provider}}))
    route = ai.specialist_config("evidence_review")
    request = sw.build_request(case, item, "evidence_review", provider)
    approved = provider == "anthropic"
    payload = claims if approved else {"claims": [], "practice_context": [], "conflicts": []}
    response = {"role_id": "evidence_review", "request_id": request["request_id"], "status": "proposed",
                "base_revision": case["version"], "payload": payload, "issues": []}
    sent = []
    def capture(outgoing):
        body = json.loads(outgoing.data)
        data = json.loads(body["messages"][0]["content"] if approved else body["input"])
        sent.append(data["role_input"]["source_excerpts"])
        assert "LOCAL-ONLY SOURCE MARKER" not in json.dumps(data)
        assert "selected_by" not in json.dumps(data) and "retrieval_permissions" not in json.dumps(data)
        return transport(outgoing.full_url, response)
    monkeypatch.setattr(ai, "_http", capture)
    received = ai.dispatch_specialist("evidence_review", request, True, approved_route=(provider, route["model"]))
    assert len(sent) == 1
    assert [s["locator"] for s in sent[0]] == (["decisions.md#Threshold"] if approved else [])
    if approved:
        run = sw.specialist_run(case, item, request, "evidence_review", route, response=received)
        held = sw.accept_proposal(case, run, "Fixture SME")
        assert not sw._claims(held, held["items"][0])
        reviewed = sw.accept_proposal(case, run, "Fixture SME", applicable_claim_ids=["claim-1"])
        assert sw._claims(reviewed, reviewed["items"][0], provider) == claims["claims"]


def test_undo_preserves_current_provider_restrictions(monkeypatch, tmp_path):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    monkeypatch.setenv("ITEMDEV_DATA_DIR", str(tmp_path))
    case, iid, _ = selected_case()
    first = storage.save_case(case)
    current = deepcopy(first)
    current["retrieval_permissions"][iid]["evidence-1"]["providers"] = []
    current["items"][0]["decision_workflow"]["retrieved_sources"][0]["providers"] = []
    storage.save_case(current, "Narrow selected source permissions", expected_revision=first["version"])
    with pytest.raises(ValueError, match="inspect the new request"):
        storage.verify_specialist_revision(first["id"], first["version"])
    with pytest.raises(ValueError, match="Source permissions changed"):
        storage.save_case(first, "Edit from another open session")
    restored = storage.recover(first["id"])
    for snapshot in (restored, storage.recover(first["id"], "redo")):
        item = snapshot["items"][0]
        assert snapshot["retrieval_permissions"][iid]["evidence-1"]["providers"] == []
        assert not item["decision_workflow"]["retrieved_sources"][0]["providers"]
        assert "decisions.md" not in json.dumps(sw.build_request(snapshot, item, "evidence_review", "anthropic"))


def test_confirming_visible_changes_preserves_a_withheld_features_linked_fact():
    case, item = task_case("unfolding")
    model = score_model(case, evidence=False)
    later = deepcopy(model["features"][0]) | {
        "feature_id": "later-measurement", "label": "Later measurement",
        "disclosure_stage_id": case["stages"][1]["id"], "fact_ids": ["early-fact"],
    }
    model["features"].append(later)
    case["facts"].append(deepcopy(case["facts"][0]) | {"id": "early-fact", "value": 9})
    for scenario in model["scenarios"]:
        scenario["fact_values"]["later-measurement"] = 11
    case = sw.save_model(case, item["id"], model, "Fixture SME")
    case = sw.accept_scenario(case, item["id"], "at", "Fixture SME", "Synthetic confirmation.", apply_values=True)
    assert {f["id"]: f["value"] for f in case["facts"]} == {"fact-score": 10, "fact-exception": False, "early-fact": 9}


def test_reused_bank_id_grants_follow_the_reviewed_version_across_undo(monkeypatch, tmp_path):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    monkeypatch.setenv("ITEMDEV_DATA_DIR", str(tmp_path))
    case, item = task_case()
    iid = item["id"]
    original = {"source_id": "reused-bank-id", "source_class": "bank", "excerpt": "Older wording.",
                "source_date": "1", "locator": "Bank example", "providers": ["openai"]}
    case = storage.save_case(sw.replace_retrieved_sources(case, iid, [original], "Original reviewer"))
    case = storage.save_case(sw.replace_retrieved_sources(case, iid, [], "Second reviewer"))
    newer = original | {"excerpt": "Revised wording.", "source_date": "2"}
    case = storage.save_case(sw.replace_retrieved_sources(case, iid, [newer], "Second reviewer"))
    assert "Revised wording" in json.dumps(sw.build_request(case, case["items"][0], "bank_retrieval", "openai"))
    storage.recover(case["id"])
    restored = storage.recover(case["id"])
    assert restored["items"][0]["decision_workflow"]["retrieved_sources"][0]["excerpt"] == "Older wording."
    assert "Older wording" not in json.dumps(sw.build_request(restored, restored["items"][0], "bank_retrieval", "openai"))
    storage.recover(case["id"], "redo")
    redone = storage.recover(case["id"], "redo")
    assert "Revised wording" in json.dumps(sw.build_request(redone, redone["items"][0], "bank_retrieval", "openai"))
