"""Offline synthetic acceptance fixtures for the Family Medicine workflow."""

from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import ai_adapter as ai
import cloud_entry
import core
import decision_model as dm
import learner_view as learner
import specialist_workflow as sw
import storage


SETTINGS = {
    "OPENAI_API_KEY": "mock-only-openai-credential",
    "ANTHROPIC_API_KEY": "mock-only-anthropic-credential",
    "ITEMDEV_OPENAI_MODEL": "mock-gpt",
    "ITEMDEV_ANTHROPIC_MODEL": "mock-claude",
    "ITEMDEV_AGENT_ROUTES": "{}",
    "ITEMDEV_HOSTED": "0",
    "ITEMDEV_AI_PROVIDER": "compatible",
    "ITEMDEV_AI_BASE_URL": "https://general.example.test/v1",
    "ITEMDEV_AI_MODEL": "mock-legacy",
    "ITEMDEV_AI_API_KEY": "mock-only-legacy-credential",
}


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    for k, v in SETTINGS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ITEMDEV_DATA_DIR", str(tmp_path))
    def forbid(*args, **kwargs):
        raise AssertionError("Live HTTP is forbidden in specialist tests.")
    monkeypatch.setattr(ai, "_http", forbid)


def predicate(feature, value, operator="equals"):
    return {"feature": feature, "operator": operator, "value": value}


def rule(rule_id, action, condition, evidence=True):
    return {"rule_id": rule_id, "action_ids": [action], "condition": condition,
            "evidence_claim_ids": ["claim-1"] if evidence else [],
            "applicability": "Labeled synthetic exercise only."}


def feature(case, name, values, stage=0):
    return {"feature_id": name, "label": name, "allowed_values": values, "unit": None,
            "classification": "decisive", "disclosure_stage_id": case["stages"][stage]["id"],
            "variation": "variable", "fact_ids": ["fact-" + name], "origin": "Synthetic fixture"}


def scenario(name, key, **values):
    return {"scenario_id": name, "fact_values": values, "keyed_action_ids": key,
            "outcome": "key_resolved" if key else "undetermined", "evidence_claim_ids": []}


def task_case(mode="standalone"):
    case = core.new_case(mode)
    case["patient"] = "Exact shared vignette for a synthetic test case."
    case["blueprint"]["objective"] = "Choose the next action from the rule for this decision."
    case["blueprint"]["domain"] = "Illustrative prevention"
    item = case["items"][0]
    item["lead_in"] = "What is the next action?"
    item["decision_workflow"] = {
        "decision": "Choose a screening action", "population": "Fictional adults",
        "setting": "Test primary care", "assessment_intent": "Apply the synthetic threshold",
        "examinee_population": "Synthetic learner cohort", "intended_key": "Action A",
    }
    case["specialist_sources"] = [
        {"source_id": "evidence-1", "source_class": "evidence summary", "source_date": "Fixture v1",
         "locator": "Rule 1", "excerpt": "Invented rule: use B at 10 or above, otherwise A, without exception.",
         "providers": ["openai", "anthropic"], "url": "https://example.test/fixture"},
        {"source_id": "bank-1", "source_class": "bank", "source_date": "Fixture v2",
         "locator": "Summary 1", "excerpt": "Synthetic bank summary: contrasting actions A and B.",
         "decision": "Choose a screening action", "providers": ["openai"],
         "observed_option_evidence": [{"item_version": "2", "population": "Synthetic comparison group",
                                     "administration": "Mock field test", "sample_size": 20,
                                     "denominator": 20, "selected": 6}]},
        {"source_id": "practice-1", "source_class": "practice summary", "source_date": "Fixture v1",
         "locator": "Scenario 1", "excerpt": "A synthetic clinic reports values at appointments.",
         "providers": ["anthropic"]},
        {"source_id": "local-only", "source_class": "evidence summary", "source_date": "Fixture v1",
         "locator": "Local", "excerpt": "LOCAL-ONLY SOURCE MARKER", "providers": []},
    ]
    case["facts"] = [
        {"id": "fact-score", "name": "Score", "value": 9, "status": "observed",
         "disclosed_at": case["stages"][0]["id"], "source": "private fact annotation"},
        {"id": "fact-exception", "name": "Exception", "value": False, "status": "observed",
         "disclosed_at": case["stages"][0]["id"]},
    ]
    item["options"] = [
        core.new_option("Action A", True, "Synthetic rationale A"),
        core.new_option("Action B", False, "Synthetic rationale B"),
    ]
    return case, item


def score_model(case, *, evidence=True):
    actions = [{"action_id": a, "label": f"Action {a}", "option_type": "Management"} for a in ("A", "B")]
    return {
        "features": [feature(case, "score", [9, 10, 11]), feature(case, "exception", [False, True])],
        "decision_rules": [
            rule("low", "A", {"all": [predicate("score", 10, "lt"), predicate("exception", False)]}, evidence),
            rule("high", "B", {"all": [predicate("score", 10, "gte"), predicate("exception", False)]}, evidence),
        ],
        "actions": actions, "feasibility_constraints": [],
        "scenarios": [scenario("below", ["A"], score=9, exception=False),
                      scenario("at", ["B"], score=10, exception=False),
                      scenario("above", ["B"], score=11, exception=False)],
        "baseline_scenario_id": "below",
        "proposed_key": {"action_ids": ["A"], "source_claim_ids": ["claim-1"] if evidence else [],
                         "awaiting_sme_review": True},
    }


def fit_case():
    case, item = task_case("unfolding")
    case["items"][0]["stage_id"] = case["stages"][1]["id"]
    case["patient"] = "Synthetic FIT fixture. An asymptomatic average-risk 55-year-old chose FIT in U.S. primary care."
    case["facts"] = [
        {"id": "fact-lab_status", "name": "Lab status", "value": "processed", "disclosed_at": item["stage_id"]},
        {"id": "fact-fit_result", "name": "FIT result", "value": "positive", "disclosed_at": item["stage_id"]},
    ]
    item["option_count"] = 4
    item["decision_workflow"]["intended_key"] = ""
    values = [("fit-positive", "processed", "positive", "colonoscopy", "Arrange diagnostic colonoscopy"),
              ("fit-negative", "processed", "negative", "annual", "Repeat screening FIT in one year"),
              ("fit-rejected", "rejected", "no_reportable_result", "repeat", "Request a new FIT sample now")]
    conditions = [{"all": [predicate("lab_status", lab), predicate("fit_result", report)]}
                  for _, lab, report, _, _ in values]
    model = {
        "features": [feature(case, "lab_status", ["processed", "rejected"], stage=1),
                     feature(case, "fit_result", ["positive", "negative", "no_reportable_result"], stage=1)],
        "actions": [{"action_id": a, "label": label, "option_type": "Investigation"} for _, _, _, a, label in values],
        "feasibility_constraints": [{"constraint_id": "final-report", "condition": {"any": conditions},
                                    "rationale": "A valid result requires processing; a rejected sample has no reportable result.",
                                    "evidence_claim_ids": []}],
        "decision_rules": [rule(sid + "-rule", action, condition, False)
                           for (sid, _, _, action, _), condition in zip(values, conditions)],
        "scenarios": [scenario(sid, [action], lab_status=lab, fit_result=report)
                      for sid, lab, report, action, _ in values],
        "baseline_scenario_id": "fit-positive",
        "proposed_key": {"action_ids": ["colonoscopy"], "source_claim_ids": [], "awaiting_sme_review": True},
    }
    return case, item, model


def option(candidate_id, action_id, text, keyed=False, rule_id=None, **kwargs):
    return {"candidate_id": candidate_id, "action_id": action_id, "text": text,
            "attractive_fact_ids": [], "violated_rule_ids": [rule_id] if rule_id else [],
            "appropriate_in_scenario_ids": [], "evidence_claim_ids": [],
            "misconception_hypothesis": None if keyed else "Overgeneralize a routine action",
            "evidence_status": None if keyed else "author hypothesis", "role": "conditional",
            "eligibility": "keyed" if keyed else "eligible",
            "plausibility_rationale": None if keyed else "The alternative applies in a similar case.",
            "clinical_rationale": "The current report determines which rule applies.", **kwargs}


def evidence_payload():
    return {"claims": [{"claim_id": "claim-1", "source_id": "evidence-1", "locator": "Rule 1",
                         "source_date": "Fixture v1", "population": "Fictional adults",
                         "condition": "At threshold with no exception", "exclusions": ["Exception present"],
                         "timing": "On report review", "action": "Use action B at or above the threshold",
                         "direction": "Favors", "strength": None, "applicability": "Fixture exercise",
                         "limitations": ["Synthetic only"]}],
            "conflicts": [], "practice_context": [{"summary_id": "practice-1",
                                                    "observation": "Appointment reporting",
                                                    "applicable_context": "Synthetic clinic"}]}


def result_for(role, req, case, *, disagreement=False):
    data, accepted = req["role_input"], req["accepted_fields"]
    if role == "bank_retrieval":
        matches = [{"item_id": row["item_id"], "task_match": "Same decision", "useful_pattern": "Compare B",
                    "mismatch": None, "observed_option_evidence": row.get("observed_option_evidence", [])}
                   for row in data["bank_candidates"]]
        payload = {"matches": matches, "candidate_alternatives": [
            {"action": "Action B", "relevant_context": "Same illustrative threshold", "item_ids": [m["item_id"]]}
            for m in matches], "duplicate_flags": []}
    elif role == "evidence_review":
        payload = evidence_payload() if data["source_excerpts"] else {"claims": [], "conflicts": [], "practice_context": []}
    elif role == "key_feature_modeling":
        payload = score_model(case, evidence=bool(data["evidence_claims"]))
    elif role == "item_drafting":
        payload = {"scenario_id": accepted["scenario_id"], "stem": "The reviewed synthetic report is available.",
                   "lead_in": "What is the next action?", "selection_instruction": "Select the single best answer.",
                   "fact_ids_used": ["fact-score"], "drafting_notes": []}
    elif role == "distractor_modeling":
        payload = {"scenario_id": accepted["scenario_id"], "shortage_count": 0, "rejected_candidates": [],
                   "options": [option("opt-a-candidate", "A", "Action A", True),
                               option("opt-b-candidate", "B", "Action B", rule_id="high")]}
    elif role == "item_validation":
        pass_type = data["pass_type"]
        ids = ([data["options"][int(disagreement)]["option_id"]] if pass_type == "solve"
               else data["solve_result"]["defensible_option_ids"] if pass_type == "compare" else [])
        payload = {"pass_type": pass_type, "defensible_option_ids": ids, "competing_option_ids": [],
                   "missing_fact_ids": [], "findings": [], "recommendation": "review_ready"}
    else:
        payload = {"decision_summary": "Apply the supplied fixture rule.", "option_rationales": [
            {"option_id": o["option_id"], "candidate_text": "Explanation for " + o["text"],
             "reviewer_text": "Review the synthetic evidence at Rule 1.", "fact_ids": ["fact-score"],
             "rule_ids": [], "evidence_claim_ids": ["claim-1"]}
            for o in accepted["item"]["options"]
        ]}
    return {"role_id": role, "request_id": req["request_id"], "base_revision": req["target"]["base_revision"],
            "status": "proposed", "payload": payload, "issues": []}


def transport(url, response):
    envelope = (
        {"status": "completed", "output": [{"type": "message", "status": "completed", "role": "assistant",
                                           "content": [{"type": "output_text", "text": json.dumps(response)}]}]}
        if "openai.com" in url else
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(response)}]}
    )
    return json.dumps(envelope).encode(), "application/json"


def run(case, role, *, pass_type=None, disagreement=False, mutate=None):
    item = case["items"][0]
    route = ai.specialist_config(role)
    req = sw.build_request(case, item, role, route["provider"], pass_type=pass_type)
    res = result_for(role, req, case, disagreement=disagreement)
    if mutate:
        mutate(res)
    with patch.object(ai, "_http", side_effect=lambda http: transport(http.full_url, res)) as call:
        received = ai.dispatch_specialist(role, req, True, approved_route=(route["provider"], route["model"]))
    assert received == res
    outbound = call.call_args.args[0]
    assert call.call_count == 1
    body = json.loads(outbound.data)
    assert body["model"] == route["model"]
    assert all(v not in outbound.data.decode() for k, v in SETTINGS.items() if k.endswith("KEY"))
    if route["provider"] == "openai":
        assert outbound.full_url == "https://api.openai.com/v1/responses"
        assert outbound.get_header("Authorization") == "Bearer " + SETTINGS["OPENAI_API_KEY"]
        assert outbound.get_header("X-api-key") is None
        assert body["store"] is False and body["max_output_tokens"] > 0
        assert json.loads(body["input"]) == req and "Canonical output schema" in body["instructions"]
    else:
        assert outbound.full_url == "https://api.anthropic.com/v1/messages"
        assert outbound.get_header("X-api-key") == SETTINGS["ANTHROPIC_API_KEY"]
        assert outbound.get_header("Authorization") is None
        assert body["max_tokens"] > 0 and len(body["messages"]) == 1
        assert json.loads(body["messages"][0]["content"]) == req and "Canonical output schema" in body["system"]
    audit = sw.specialist_run(case, item, req, role, route, response=received)
    stored = storage.record_specialist_run(case["id"], audit)
    assert stored["version"] == case["version"]
    return stored, req, audit


def apply(case, audit):
    return storage.save_case(sw.accept_proposal(case, audit, "Fixture SME"), expected_revision=case["version"])


def test_all_roles_and_blinded_review_round_trip():
    case, _ = task_case()
    case = storage.save_case(core.normalize_import(case))
    case, bank_req, bank = run(case, "bank_retrieval")
    assert {r["source_class"] for r in bank_req["role_input"]["bank_candidates"]} == {"bank"}
    case = apply(case, bank)
    case, evidence_req, evidence = run(case, "evidence_review")
    assert [s["source_id"] for s in evidence_req["role_input"]["source_excerpts"]] == ["evidence-1"]
    assert [s["summary_id"] for s in evidence_req["role_input"]["practice_summaries"]] == ["practice-1"]
    case = apply(case, evidence)
    case, model_req, modeled = run(case, "key_feature_modeling")
    assert model_req["role_input"]["intended_key"] == "Action A"
    assert model_req["task"]["examinee_population"] == "Synthetic learner cohort"
    assert model_req["role_input"]["bank_patterns"]["matches"][0]["observed_option_evidence"][0]["sample_size"] == 20
    case = apply(case, modeled)
    iid = case["items"][0]["id"]
    case = storage.save_case(sw.accept_scenario(case, iid, "below", "Fixture SME", ""))
    for role in ("item_drafting", "distractor_modeling"):
        case, req, proposed = run(case, role)
        assert req["accepted_fields"]["scenario_id"] == proposed["response"]["payload"]["scenario_id"] == "below"
        neighbors = req["accepted_fields"]["decision_model"]["neighbor_scenarios"]
        assert len(neighbors) == (0 if role == "item_drafting" else 2)
        case = apply(case, proposed)
    case = storage.save_case(sw.approve_key(case, iid, "Fixture SME", ""))
    case, blind, solved = run(case, "item_validation", pass_type="solve")
    assert blind["accepted_fields"] == {}
    assert set(blind["task"]) == {"mode", "response_format"}
    assert "private fact annotation" not in json.dumps(blind) and "LOCAL-ONLY SOURCE MARKER" not in json.dumps(blind)
    assert solved["authorship"]["label"] == "Cross-provider blinded review"
    case, compare, compared = run(case, "item_validation", pass_type="compare")
    assert compared["comparison"]["status"] == "agrees"
    assert compare["role_input"]["solve_result"] == solved["response"]["payload"]
    assert all(compare["role_input"][k] == v for k, v in blind["role_input"].items() if k != "pass_type")
    case, rationale_req, rationale = run(case, "rationale_writing")
    assert rationale_req["role_input"]["option_judgments"][0]["rule_ids"] == ["low"]
    case = apply(case, rationale)
    assert case["items"][0]["options"][0]["rationale"] == "Explanation for Action A"
    author = core.export_markdown(case)
    assert "Specialist proposals and review passes" in author and "SME decision models and acceptances" in author
    assert "Specialist proposals" not in core.export_markdown(case, False)
    assert not any(v in author for k, v in SETTINGS.items() if k.endswith("KEY"))
    assert storage.load_library()["cases"][case["id"]] == case


def test_fit_feasibility_visibility_duplicate_and_shortage():
    case, item, model = fit_case()
    model["scenarios"] += dm.generate_scenarios(model)
    rows = dm.scenario_rows(model, case, item)
    assert len(rows) == 6 and sum(r["outcome"] == "infeasible" for r in rows.values()) == 3
    assert {sid: rows[sid]["keyed_action_ids"] for sid in ("fit-positive", "fit-negative", "fit-rejected")} == {
        "fit-positive": ["colonoscopy"], "fit-negative": ["annual"], "fit-rejected": ["repeat"]}
    assert rows["fit-positive"]["outcome"] == "key_resolved"
    assert rows["fit-negative"]["outcome"] == rows["fit-rejected"]["outcome"] == "key_changed"
    assert rows["fit-rejected"]["changes"] == {"lab_status": "rejected", "fit_result": "no_reportable_result"}
    hidden = dm.scenario_rows(model, case, item, stage_id=case["stages"][0]["id"])["fit-positive"]
    assert hidden["outcome"] == "undetermined" and hidden["missing_features"] == ["fit_result", "lab_status"]
    options = [
        option("c1", "colonoscopy", "Arrange diagnostic colonoscopy", True),
        option("c2", "annual", "Repeat screening FIT in one year", rule_id="fit-negative-rule"),
        option("c3", "repeat", "Request a new FIT sample now", rule_id="fit-rejected-rule"),
        option("c4", "repeat", "Collect another sample", rule_id="fit-rejected-rule"),
        option("c5", None, "Do something unmodeled") | {"eligibility": "review"},
        option("c6", "colonoscopy", "Colonoscopy as an equally defensible answer"),
    ]
    pool = dm.option_pool(model, rows["fit-positive"], options, 4)
    assert pool["shortage"] == 1 and len(pool["selected"]) == 3
    assert all(r["local_judgment"] == "review" for r in pool["rows"][3:])
    assert all(o["plausibility_rationale"] and o["clinical_rationale"] for o in pool["selected"][1:])
    accepted = sw.save_model(case, item["id"], model, "SME")
    positive = sw.accept_scenario(accepted, item["id"], "fit-positive", "SME", "Synthetic exercise.")
    negative = sw.accept_scenario(positive, item["id"], "fit-negative", "SME", "Synthetic exercise.", apply_values=True)
    status = sw.model_status(negative, negative["items"][0])
    assert status["selection_current"] and status["row"]["keyed_action_ids"] == ["annual"]
    assert "annual" in dm.option_pool(model, status["row"], options, 4)["missing_keys"]
    assert sw._state(negative["items"][0])["source_case_id"] == case["id"]


def test_threshold_edit_waits_for_review_and_unknown_does_not_negate():
    case, item = task_case("unfolding")
    model = score_model(case, evidence=False)
    case = sw.save_model(case, item["id"], model, "SME")
    case = sw.accept_scenario(case, item["id"], "below", "SME", "Synthetic threshold.")
    item = case["items"][0]
    assert sw.model_status(case, item)["row"]["keyed_action_ids"] == ["A"]
    case["facts"][0]["value"] = "10"
    status = sw.model_status(case, item)
    assert status["row"]["keyed_action_ids"] == ["B"] and not status["selection_current"]
    with pytest.raises(ValueError, match="Confirm the selected case"):
        sw.build_request(case, item, "item_drafting", "openai")
    case["facts"][0]["status"] = "unknown"
    assert sw.model_status(case, item)["row"]["missing_features"] == ["score"]
    case["facts"][0].update(status="observed", disclosed_at=case["stages"][1]["id"])
    assert sw.model_status(case, item)["row"]["outcome"] == "undetermined"
    case["facts"][0]["disclosed_at"] = item["stage_id"]
    case["facts"][1]["status"] = "not assessed"
    assert sw.model_status(case, item)["row"]["missing_features"] == ["exception"]
    case["facts"][1]["status"] = "absent"
    assert sw.model_status(case, item)["row"]["keyed_action_ids"] == ["B"]
    x = predicate("x", True)
    assert core.evaluate_decision_rule({"not": x}, {}) is None
    assert core.evaluate_decision_rule({"all": [x, predicate("y", True)]}, {"y": False}) is False
    assert core.evaluate_decision_rule({"any": [x, predicate("y", True)]}, {"y": True}) is True
    assert core.evaluate_rule({"not": x}, {}) is True


def test_ambiguous_reference_change_and_multiple_selection_membership():
    case, item = task_case()
    model = score_model(case, evidence=False)
    model["decision_rules"].append(rule("tie", "B", predicate("score", 9), False))
    original = dm.scenario_rows(model, case, item)
    assert original["below"]["outcome"] == "ambiguous"
    assert original["below"]["competing_action_ids"] == ["A", "B"]
    assert original["at"]["outcome"] == "key_resolved"
    model["baseline_scenario_id"] = "at"
    changed = dm.scenario_rows(model, case, item)
    assert changed["above"]["outcome"] == "key_retained"
    assert all(original[s]["keyed_action_ids"] == changed[s]["keyed_action_ids"] for s in original)
    item["format"] = "MS"
    assert dm.scenario_rows(model, case, item)["below"]["keyed_action_ids"] == ["A", "B"]
    model["decision_rules"][-1]["scope_condition"] = predicate("exception", True)
    assert dm.evaluate(model, model["scenarios"][0], {"score": 9}, "MS")["outcome"] == "undetermined"


def test_scoring_and_selection_count():
    _, item = task_case()
    item.update(format="MS", select_count=2, selection_instruction="Select exactly two.")
    item["options"] = [core.new_option(x, x in ("A", "C")) for x in "ABCD"]
    a, b, c, d = [o["id"] for o in item["options"]]
    patterns = [[], [a], [a, c], [a, b], [a, b, c, d]]
    for policy, expected in [("exact", [0, 0, 1, 0, 0]), ("partial", [0, .5, 1, 0, 0])]:
        item["scoring"] = policy
        assert [core.score_item(item, p)["score"] for p in patterns] == expected
    assert [core.selection_count_met(item, len(p)) for p in patterns] == [False, False, True, True, False]
    item["options"].reverse()
    assert core.score_item(item, [c, a])["score"] == 1


def test_stale_threshold_key_disagreement_and_revision_recovery():
    case, item = task_case()
    iid = item["id"]
    case["facts"][0]["value"] = 10
    case["empirical_records"] = [{"case_version": 1, "item_id": iid, "study": "Imported synthetic record"}]
    case["reviews"] = [{"id": "historical", "case_version": 1, "decision": "Ready for pilot", "valid": True}]
    case = storage.import_case(case)
    item = case["items"][0]
    case = storage.save_case(sw.approve_key(case, iid, "Fixture author", "Carried over on purpose for mismatch test."))
    case, blind, solved = run(case, "item_validation", pass_type="solve", disagreement=True)
    assert "Synthetic rationale A" not in json.dumps(blind)
    assert blind["role_input"]["options"][1]["option_id"] == solved["response"]["payload"]["defensible_option_ids"][0]
    case, _, compared = run(case, "item_validation", pass_type="compare")
    assert compared["comparison"]["status"] == "disagrees"
    assert any("key disagreement" in i for i in sw.review_findings(case, case["items"][0]))
    previous = deepcopy(case)
    case["items"][0]["lead_in"] = "New accepted wording"
    updated = storage.save_case(case)
    assert not sw.proposal_current(updated, solved)
    with pytest.raises(ValueError, match="newer revision"):
        storage.save_case(previous, expected_revision=previous["version"])
    storage.record_specialist_run(case["id"], solved)
    assert storage.load_library()["cases"][case["id"]]["items"][0]["lead_in"] == "New accepted wording"
    recovered = storage.recover(case["id"])
    assert not sw.proposal_current(recovered, solved)
    assert len(recovered["specialist_runs"]) == 2
    assert not recovered["reviews"][0]["valid"] and recovered["reviews"][0]["case_version"] == 1
    assert recovered["empirical_records"][0]["case_version"] == 1


@pytest.mark.parametrize("mode", ["appraisal", "unfolding"])
def test_exact_blinded_learner_view_and_cueing_isolation(mode):
    case, item = task_case(mode)
    item["recommendation"] = "Exact preliminary recommendation recorded separately."
    if mode == "unfolding":
        case["items"][1]["stem"] = "FUTURE SIBLING CONTENT"
        case["stages"][1]["new_information"] = "FUTURE DISCLOSURE CONTENT"
        case["stages"][0]["transition"] = "FUTURE TRANSITION CONTENT"
    case = storage.save_case(case)
    case, req, solved = run(case, "item_validation", pass_type="solve")
    assert req["role_input"] == (learner.item_view(case, item, opaque=True) | learner.stage_view(case, item["stage_id"]) | {
        "pass_type": "solve", "source_excerpts": req["role_input"]["source_excerpts"]})
    encoded = json.dumps(req)
    assert "Synthetic rationale" not in encoded and "FUTURE" not in encoded
    if mode == "appraisal":
        assert item["recommendation"] in encoded
    else:
        def cue(response):
            response["payload"]["findings"] = [{"severity": "review", "field": "sequence",
                "option_ids": [], "evidence_claim_ids": [], "issue": "A same-stage cue while items remain editable",
                "revise": "Move it after the prior decision locks. An after-lock disclosure can support a new task."}]
            response["payload"]["recommendation"] = "revise"
        case, cue_req, cued = run(case, "item_validation", pass_type="cueing", mutate=cue)
        assert cue_req["accepted_fields"] == {} and cue_req["visible_facts"] == []
        assert "FUTURE SIBLING CONTENT" in json.dumps(cue_req) and "FUTURE TRANSITION CONTENT" in json.dumps(cue_req)
        assert "locks" in cue_req["role_input"]["delivery_policy"]["response_lock"]
        assert cued["response"]["payload"]["defensible_option_ids"] == []
        assert sw._current_review(case, item, "solve") == solved


def test_explicit_override_and_unavailable_routes(monkeypatch):
    case, item = task_case()
    monkeypatch.setenv("ITEMDEV_AGENT_ROUTES", '{"bank_retrieval":{"provider":"anthropic","model":"mock-overridden"},"item_validation":{"provider":"invalid"}}')
    assert ai.specialist_config("bank_retrieval")["model"] == "mock-overridden"
    assert not ai.specialist_config("item_validation")["configured"]
    assert ai.specialist_config("item_drafting")["configured"]
    case = storage.save_case(case)
    _, req, audit = run(case, "bank_retrieval")
    assert req["role_input"]["bank_candidates"] == [] and audit["provider"] == "anthropic"
    with pytest.raises(ai.AdapterError, match="route changed"):
        ai.dispatch_specialist("bank_retrieval", req, True, approved_route=("openai", "mock-gpt"))
    with pytest.raises(ai.AdapterError, match="AI assistance is off"):
        ai.dispatch_specialist("bank_retrieval", req, False, approved_route=("anthropic", "mock-overridden"))
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert not ai.specialist_config("bank_retrieval")["configured"] and ai.specialist_config("item_drafting")["configured"]
    monkeypatch.delenv("OPENAI_API_KEY")
    assert not ai.specialist_config("item_drafting")["configured"]
    assert ai.get_config()["configured"]


@pytest.mark.parametrize("bad", ["shape", "incomplete", "refusal", "reference", "revision", "json"])
def test_invalid_specialist_responses_are_sanitized(bad):
    case, item = task_case()
    req = sw.build_request(case, item, "bank_retrieval", "openai")
    response = result_for("bank_retrieval", req, case)
    if bad == "shape":
        response["payload"].pop("matches")
    if bad == "reference":
        response["payload"]["matches"][0]["item_id"] = "invented"
    if bad == "revision":
        response["base_revision"] += 1
    raw, kind = transport("openai.com", response)
    envelope = json.loads(raw)
    if bad == "incomplete":
        envelope["status"] = "incomplete"
    if bad == "refusal":
        envelope["output"][0]["content"] = [{"type": "refusal", "refusal": SETTINGS["OPENAI_API_KEY"]}]
    if bad == "json":
        envelope["output"][0]["content"][0]["text"] = '{"unexpected":1,"unexpected":2}'
    with patch.object(ai, "_http", return_value=(json.dumps(envelope).encode(), kind)):
        with pytest.raises(ai.AdapterError) as error:
            ai.dispatch_specialist("bank_retrieval", req, True, approved_route=("openai", "mock-gpt"))
    assert SETTINGS["OPENAI_API_KEY"] not in str(error.value)


def test_evidence_conflict_holds_the_key_until_explicit_resolution():
    case, item = task_case()
    case = storage.save_case(case)
    def add_conflict(response):
        response["issues"] = [{"code": "GAP", "field": "key", "message": "Resolve evidence scope at the threshold.", "severity": "blocking"}]
    case, _, recorded = run(case, "evidence_review", mutate=add_conflict)
    case = apply(case, recorded)
    case = storage.save_case(sw.save_model(case, item["id"], score_model(case), "Fixture SME"))
    assert "Resolve evidence scope at the threshold." in sw._unresolved_evidence(case, case["items"][0])
    with pytest.raises(ValueError, match="material conflict"):
        sw.accept_scenario(case, item["id"], "below", "Fixture SME", "")
    case = sw.record_human_decision(case, item["id"], "Fixture SME", "Confirmed synthetic rule scope.", "evidence_resolution")
    assert sw.accept_scenario(case, item["id"], "below", "Fixture SME", "")["items"][0]["decision_workflow"]["selection"]


@pytest.mark.parametrize("hosted", [False, True], ids=["local-app", "hosted-wrapper"])
def test_specialist_authoring_through_both_streamlit_entrypoints(hosted, tmp_path):
    case, _ = task_case()
    case["author"] = "Fixture SME"
    case = core.normalize_import(case)
    cid = case["id"]
    iid = case["items"][0]["id"]
    base = Path(__file__).resolve().parent
    path = base.parent / "deploy/item_studio/streamlit_app.py" if hosted else base / "app.py"
    app = AppTest.from_file(str(path), default_timeout=30)
    if hosted:
        app.session_state["itemdev_hosted_library"] = {
            "schema_version": 1, "cases": {cid: case}, "history": {}, "redo": {}, "events": {},
        }
    else:
        storage.save_case(case)
    sent = []
    pending = ["bank_retrieval"]
    def http(request):
        body = json.loads(request.data)
        assert all(v not in request.data.decode() for k, v in SETTINGS.items() if k.endswith("KEY"))
        if request.full_url.endswith("/chat/completions"):
            assert request.get_header("Authorization") == "Bearer " + SETTINGS["ITEMDEV_AI_API_KEY"]
            sent.append("general")
            return b'{"choices":[{"message":{"content":"Mock legacy assistance."}}]}', "application/json"
        role = pending[0]
        if "openai.com" in request.full_url:
            assert request.get_header("Authorization") == "Bearer " + SETTINGS["OPENAI_API_KEY"]
            inputs = json.loads(body["input"])
        else:
            assert request.get_header("X-api-key") == SETTINGS["ANTHROPIC_API_KEY"]
            inputs = json.loads(body["messages"][0]["content"])
        sent.append(role)
        return transport(request.full_url, result_for(role, inputs, case))
    def current():
        return app.session_state["itemdev_cases"][cid]
    def verify_ui():
        assert not app.exception, [e.value for e in app.exception]
    prefix = f"{cid}::specialist:{iid}:"
    work = f"{cid}::decision:{iid}:"
    with patch.object(ai, "_http", side_effect=http), patch.object(cloud_entry, "configure_hosted_environment",
            side_effect=lambda: configure_mock_secrets()) if hosted else patch.object(cloud_entry, "ALLOWED_SECRET_ENV", cloud_entry.ALLOWED_SECRET_ENV):
        app.run()
        verify_ui()
        assert not sent and not app.toggle(key="itemdev_ai_enabled").value
        app.toggle(key="itemdev_ai_enabled").set_value(True).run()
        app.text_area(key=cid + "::ai:request").set_value("Explain the synthetic task.")
        app.button(key=cid + "::ai:ask").click().run()
        verify_ui()

        def invoke(role, pass_type=None):
            pending[0] = role
            app.selectbox(key=prefix + "role").set_value(role).run()
            if pass_type:
                app.selectbox(key=prefix + "pass").set_value(pass_type).run()
            route = ai.specialist_config(role)
            consent = prefix + f"approve:{role}:{pass_type}:{route['provider']}:{route['model']}"
            app.checkbox(key=consent).set_value(True).run()
            assert not app.button(key=prefix + "run").disabled
            app.button(key=prefix + "run").click().run()
            verify_ui()
            run = current()["specialist_runs"][-1]
            assert run["outcome"] == "proposed", (run, [e.value for e in app.error])
            return run

        def accept():
            run = current()["specialist_runs"][-1]
            assert not app.button(key=prefix + "accept").disabled, (pending[0], current()["version"], run, [e.value for e in app.error])
            app.text_input(key=prefix + "actor").set_value("Fixture SME")
            app.button(key=prefix + "accept").click().run()
            verify_ui()

        for role in ("bank_retrieval", "evidence_review", "key_feature_modeling"):
            invoke(role)
            accept()
        app.button(key=work + "confirm").click().run()
        verify_ui()
        assert current()["items"][0]["decision_workflow"]["selection"]["scenario_id"] == "below"
        for role in ("item_drafting", "distractor_modeling"):
            invoke(role)
            accept()
        app.button(key=work + "key").click().run()
        verify_ui()
        assert current()["items"][0]["decision_workflow"]["key_approval"]["option_ids"] == sw.key_ids(current()["items"][0])
        solve = invoke("item_validation", "solve")
        assert solve["authorship"]["label"] == "Cross-provider blinded review"
        comparison = invoke("item_validation", "compare")
        assert comparison["comparison"]["status"] == "agrees"
        invoke("rationale_writing")
        accept()
        assert current()["items"][0]["options"][0]["rationale"] == "Explanation for Action A"
        app.toggle(key="itemdev_ai_enabled").set_value(False).run()
        app.button(key="undo:standalone").click().run()
        verify_ui()
        assert current()["items"][0]["options"][0]["rationale"] != "Explanation for Action A"
        assert sent == ["general", "bank_retrieval", "evidence_review", "key_feature_modeling", "item_drafting",
                        "distractor_modeling", "item_validation", "item_validation", "rationale_writing"]
        assert len(current()["specialist_runs"]) == 8
        assert not any(v in core.export_markdown(current()) for k, v in SETTINGS.items() if k.endswith("KEY"))
        assert (tmp_path / "library.json").exists() is not hosted


def configure_mock_secrets():
    configure = _configure_original
    mocks = {k: v for k, v in SETTINGS.items() if k in cloud_entry.ALLOWED_SECRET_ENV}
    for k in mocks:
        os.environ.pop(k, None)
    return configure(mocks)


_configure_original = cloud_entry.configure_hosted_environment


def test_cueing_submitted_response_window_and_after_lock_context():
    case, _ = task_case("unfolding")
    decision = case["items"][0]
    key = "Arrange synthetic diagnostic follow-up"
    decision["options"][0]["text"] = key
    case["stages"][0]["new_information"] = f"The choice is to {key.lower()} while the item is open."
    case["stages"][0]["transition"] = f"The clinician proceeded to {key.lower()}."
    case["items"][1]["lead_in"] = "Which later synthetic result should be reassessed?"
    findings = [r for r in core.validate_case(case) if r["code"] in {"DISCLOSURE_CUE", "AFTER_LOCK_DISCLOSURE"}]
    assert [(r["code"], r["path"]) for r in findings] == [
        ("DISCLOSURE_CUE", "stages[0]"), ("AFTER_LOCK_DISCLOSURE", "stages[1]")]


def test_fact_reversion_still_requires_reaffirming_accepted_scenario():
    case, item = task_case()
    case = storage.save_case(sw.save_model(case, item["id"], score_model(case, evidence=False), "Fixture SME"))
    case = storage.save_case(sw.accept_scenario(case, item["id"], "below", "Fixture SME", "Synthetic only."))
    assert sw.model_status(case, case["items"][0])["selection_current"]
    case["facts"][0]["value"] = 10
    edited = storage.save_case(case, "Change synthetic score")
    assert sw.model_status(edited, edited["items"][0])["row"]["keyed_action_ids"] == ["B"]
    edited["facts"][0]["value"] = 9
    reverted = storage.save_case(edited, "Reenter baseline score")
    assert sw.model_status(reverted, reverted["items"][0])["row"]["keyed_action_ids"] == ["A"]
    assert not sw.model_status(reverted, reverted["items"][0])["selection_current"]


def test_drafting_cannot_change_locked_wording():
    case, item = task_case()
    item["decision_workflow"]["locked_wording_fields"] = ["lead_in"]
    case = sw.save_model(case, item["id"], score_model(case, evidence=False), "Fixture SME")
    case = sw.accept_scenario(case, item["id"], "below", "Fixture SME", "Synthetic only.")
    req = sw.build_request(case, case["items"][0], "item_drafting", "openai")
    assert req["accepted_fields"]["locked_wording"]["lead_in"] == item["lead_in"]
    response = result_for("item_drafting", req, case)
    response["payload"]["lead_in"] = "A different question?"
    with patch.object(ai, "_http", return_value=transport("openai.com", response)):
        with pytest.raises(ai.AdapterError, match="inconsistent references"):
            ai.dispatch_specialist("item_drafting", req, True, approved_route=("openai", "mock-gpt"))
