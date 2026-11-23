"""Role-scoped handoffs and versioned SME decisions for Item Studio."""

from copy import deepcopy
import hashlib
import json

import core
import decision_model as dm
import learner_view as learner
from ai_adapter import project_specialist_request, specialist_definition, specialist_registry
from jsonschema import Draft202012Validator


SOURCE_CLASSES = {"bank", "guideline", "study", "evidence summary", "practice summary", "misconception evidence"}
CLINICAL_CLASSES = {"guideline", "study", "evidence summary"}
LOCKABLE_WORDING = {"stem": "Stem", "lead_in": "Lead-in", "selection_instruction": "Selection instruction"}


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _state(item):
    return item.get("decision_workflow", {})


def task_for(case, item):
    blueprint = case["blueprint"]
    state = _state(item)
    return {
        "mode": case["mode"],
        "blueprint": {k: blueprint.get(k, "") for k in ("name", "version", "domain", "subdomain")},
        "objective": item.get("objective") or blueprint.get("objective", ""),
        "decision": state.get("decision", "") or item.get("task", ""),
        "assessment_intent": state.get("assessment_intent", ""),
        "examinee_population": state.get("examinee_population", ""),
        "setting": state.get("setting", ""), "population": state.get("population", ""),
        **({"decision_timing": state["decision_timing"]} if state.get("decision_timing") else {}),
        "reasoning_phase": item["phase"], "response_format": item["format"],
        **({"appraisal_dimension": item["dimension"]} if case["mode"] == "appraisal" else {}),
    }


def validate_sources(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Source records must be a JSON array of objects.")
    for row in rows:
        for k in ("source_id", "source_class", "source_date", "locator", "excerpt"):
            if not isinstance(row.get(k), str) or not row[k].strip():
                raise ValueError(f"Each source record needs {k}.")
        if row["source_class"] not in SOURCE_CLASSES:
            raise ValueError("Choose a supported bank, clinical, practice, or misconception source class.")
        allowed = row.get("providers", [])
        if not isinstance(allowed, list) or any(p not in ("openai", "anthropic") for p in allowed):
            raise ValueError("Choose explicitly authorized providers for each source record.")
        for k in ("item_ids", "stage_ids"):
            if k in row and (not isinstance(row[k], list) or any(not isinstance(x, str) for x in row[k])):
                raise ValueError(f"{k} must be a list of IDs.")
        for k in ("url", "title", "decision", "setting", "population", "option_type", "reasoning_phase",
                  "citation", "relative_file", "section", "decision_timing"):
            if k in row and not isinstance(row[k], str):
                raise ValueError(f"{k} must be text.")
        evidence = row.get("observed_option_evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(e, dict) for e in evidence):
            raise ValueError("Observed option evidence must be a list of records.")
        for record in evidence:
            if row["source_class"] != "bank" or any(not record.get(k) for k in ("item_version", "population", "administration", "sample_size", "denominator")):
                raise ValueError("Bank statistics require item version, population, administration, sample size, and denominator.")
    dm.by_id(rows, "source_id")


def retrieval_permission(source):
    content = {k: v for k, v in source.items()
               if k not in {"providers", "selected_by", "selected_at", "selected_from_revision", "permission_review"}}
    return {"collection": "bank" if source["source_class"] == "bank" else "clinical",
            "providers": list(source.get("providers", [])), "source_revision": _signature(content)}


def permitted_retrieval_providers(source, permission):
    if permission is None:
        return list(source.get("providers", []))
    if permission["source_revision"] != retrieval_permission(source)["source_revision"]:
        return []
    return [p for p in source.get("providers", []) if p in permission["providers"]]


def replace_retrieved_sources(case, item_id, records, actor):
    """Save only the writer's selected passages, bound to this item and disclosure."""
    validate_sources(records)
    actor = _human(actor)
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item.setdefault("decision_workflow", {})
    existing = dm.by_id(state.get("retrieved_sources", []), "source_id")
    permissions = candidate.setdefault("retrieval_permissions", {}).setdefault(item_id, {})
    for sid, source in existing.items():
        permissions.setdefault(sid, retrieval_permission(source))
    manual_ids = {s["source_id"] for s in case.get("specialist_sources", [])}
    selected = []
    for record in records:
        sid = record["source_id"]
        if sid in manual_ids:
            raise ValueError(f"Source ID {sid} already identifies a manually entered record.")
        if record == existing.get(sid):
            selected.append(deepcopy(record))
            continue
        if record.get("item_ids") and item_id not in record["item_ids"]:
            raise ValueError("The retrieved record is outside this item's authorized scope.")
        if record.get("stage_ids") and item["stage_id"] not in record["stage_ids"]:
            raise ValueError("The retrieved record is outside this disclosure stage.")
        if sid in existing and existing[sid]["stage_ids"] != [item["stage_id"]]:
            raise ValueError("This source ID is already selected at a different stage of the item.")
        if record["source_class"] in CLINICAL_CLASSES and not (record.get("citation") or record.get("url")):
            raise ValueError("A clinical passage needs its original citation. Import an uncited note as a practice summary.")
        snapshot = deepcopy(record) | {
            "item_ids": [item_id], "stage_ids": [item["stage_id"]],
            "selected_by": actor, "selected_at": core.timestamp(), "selected_from_revision": case["version"],
        }
        selected.append(snapshot)
        permissions[sid] = retrieval_permission(snapshot)
    state["retrieved_sources"] = selected
    _sources(candidate, item)
    return candidate


def validate_manual_sources(case, rows):
    validate_sources(rows)
    manual_ids = {s["source_id"] for s in rows}
    for item in case["items"] + case.get("archived_items", []):
        selected = _state(item).get("retrieved_sources", [])
        if manual_ids.intersection(s["source_id"] for s in selected):
            raise ValueError("A manual source ID conflicts with a selected retrieval record. Use distinct IDs.")


def _sources(case, item, provider=None):
    manual = case.get("specialist_sources", [])
    selected = _state(item).get("retrieved_sources", [])
    sources = manual + selected
    validate_sources(sources)
    selected_ids = {s["source_id"] for s in selected}
    permissions = case.get("retrieval_permissions", {}).get(item["id"], {})
    scope = task_for(case, item)
    rows = [
        source for source in sources
        if (provider is None or provider in permitted_retrieval_providers(source,
            permissions.get(source["source_id"]) if source["source_id"] in selected_ids else None))
        and (not source.get("item_ids") or item["id"] in source["item_ids"])
        and (not source.get("stage_ids") or item["stage_id"] in source["stage_ids"])
        and (source["source_id"] in selected_ids or source["source_class"] == "bank" or all(
            not source.get(k) or source[k].casefold() == scope[k].casefold()
            for k in ("setting", "population")
        ))
    ]
    if len(rows) > 100:
        raise ValueError("Scope source records to this decision to use at most 100 passages.")
    return rows


def reviewed_source(case, item, source_id):
    """Look up an author-visible passage applicable to this item and stage."""
    return next((s for s in _sources(case, item) if s["source_id"] == source_id), None)


def _approved_source(row):
    fields = ("source_id", "source_class", "source_date", "locator", "excerpt", "url", "title",
              "decision", "setting", "population", "option_type", "reasoning_phase",
              "citation", "decision_timing", "observed_option_evidence")
    approved = {k: row[k] for k in fields if k in row}
    if row["source_class"] == "bank":
        if isinstance(row.get("blueprint"), dict):
            approved["blueprint"] = {k: row["blueprint"][k] for k in ("name", "version", "domain", "subdomain", "objective")
                                     if isinstance(row["blueprint"].get(k), str)}
        if isinstance(row.get("options"), list):
            approved["options"] = [{k: o[k] for k in ("text", "option_id") if isinstance(o.get(k), str)}
                                   for o in row["options"] if isinstance(o, dict) and isinstance(o.get("text"), str)]
    return approved


def _rank_bank(rows, task, item):
    option_types = {o.get("option_type", "").casefold() for o in item["options"]} - {""}
    def rank(row):
        dimensions = ("decision", "setting", "population", "reasoning_phase")
        exact = sum(bool(row.get(k)) and row[k].casefold() == task[k].casefold() for k in dimensions)
        terms = set((task["decision"] + " " + task["objective"]).lower().split())
        overlap = len(terms & set(row["excerpt"].lower().split()))
        return exact, row.get("option_type", "").casefold() in option_types, overlap
    return sorted(rows, key=rank, reverse=True)


def _artifact_valid(record, case, item):
    if not record or record.get("requires_reassessment"):
        return False
    return record.get("basis") == _artifact_basis(case, item)


def _claims(case, item, provider=None):
    artifact = _state(item).get("evidence")
    if not _artifact_valid(artifact, case, item):
        return []
    allowed = {s["source_id"] for s in _sources(case, item, provider) if s["source_class"] in CLINICAL_CLASSES}
    retrieved_ids = {s["source_id"] for s in _state(item).get("retrieved_sources", [])}
    applicable = set(artifact.get("applicable_retrieved_claim_ids", []))
    return [c for c in artifact["payload"].get("claims", []) if c["source_id"] in allowed
            and (c["source_id"] not in retrieved_ids or c["claim_id"] in applicable)]


def _bank_patterns(case, item, provider):
    artifact = _state(item).get("bank")
    if not _artifact_valid(artifact, case, item):
        return {}
    allowed = {s["source_id"] for s in _sources(case, item, provider) if s["source_class"] == "bank"}
    patterns = artifact["payload"]
    return {
        "matches": [m for m in patterns["matches"] if m["item_id"] in allowed],
        "candidate_alternatives": [a for a in patterns["candidate_alternatives"]
                                   if a["item_ids"] and set(a["item_ids"]) <= allowed],
    }


def _matches(record, signature):
    return bool(record and not record.get("requires_reassessment") and record.get("signature") == signature)


def _artifact_basis(case, item):
    return _signature({"case": case["id"], "epoch": case.get("specialist_epoch"),
                       "stage": item["stage_id"], "task": task_for(case, item),
                       "source_records": _sources(case, item)})


def _model_basis(item):
    state = _state(item)
    return _signature({"evidence": state.get("evidence"), "intended_key": state.get("intended_key"),
                       "recommendation": item["recommendation"]})


def _artifact(case, item, payload, actor, run=None):
    artifact = {
        "payload": deepcopy(payload), "actor": actor, "time": core.timestamp(),
        "basis": _artifact_basis(case, item),
        "accepted_from_revision": case["version"], "decision": "accepted",
        "run_id": run["request_id"] if run else None, "provider": run["provider"] if run else None,
    }
    if _state(item).get("retrieved_sources"):
        claims = _artifact_claim_sources(case, item, payload)
        links = _artifact_sources(case, item, payload, claims)
        if claims:
            artifact["claim_sources"] = claims
        if links:
            artifact["source_links"] = links
    return artifact


def _artifact_claim_sources(case, item, payload):
    ids = set(payload.get("proposed_key", {}).get("source_claim_ids", []))
    for group in ("decision_rules", "feasibility_constraints", "scenarios", "options", "option_rationales"):
        for row in payload.get(group, []):
            ids.update(row.get("evidence_claim_ids", []))
    return ({c["claim_id"]: c["source_id"] for c in _claims(case, item) if c["claim_id"] in ids}
            | {c["claim_id"]: c["source_id"] for c in payload.get("claims", [])})


def _artifact_sources(case, item, payload, claim_sources=None):
    """Retain cited locators on the accepted revision even if the writer later removes a passage."""
    if claim_sources is None:
        claim_sources = _artifact_claim_sources(case, item, payload)
    cited = set(claim_sources.values())
    for row in payload.get("options", []):
        cited.update(row.get("bank_item_ids", []))
        cited.update(row.get("misconception_evidence_ids", []))
    cited.update(row["item_id"] for row in payload.get("matches", []))
    for row in payload.get("candidate_alternatives", []):
        cited.update(row.get("item_ids", []))
    cited.update(row["summary_id"] for row in payload.get("practice_context", []))
    return {s["source_id"]: _source_reference(s) for s in _sources(case, item) if s["source_id"] in cited}


def _source_reference(source):
    return {k: source[k] for k in ("source_id", "source_class", "source_date", "locator", "citation", "url",
                                   "title", "relative_file", "section") if k in source}


def _learner_snapshot(case, item):
    return learner.stage_view(case, item["stage_id"]) | learner.item_view(case, item) | {
        "facts": learner.facts(case, item["stage_id"]), "mode": case["mode"], "format": item["format"],
        "select_count": item.get("select_count"), "scoring": item["scoring"],
    }


def _view_signature(case, item):
    return _signature({"case": case["id"], "item": item["id"], "epoch": case.get("specialist_epoch"),
                       "view": _learner_snapshot(case, item), "sources": _sources(case, item)})


def _key_signature(case, item):
    state = _state(item)
    return _signature({
        "view": _view_signature(case, item), "key": sorted(key_ids(item)),
        "model": state.get("model"), "selection": state.get("selection"),
        "task": task_for(case, item), "option_status": [
            {k: o.get(k) for k in ("id", "eligible", "conditions", "modeled_action_id")}
            for o in item["options"]
        ],
    })


def key_ids(item):
    return [o["id"] for o in item["options"] if o["correct"]]


def _selection_signature(case, item, row):
    return _signature({
        "epoch": case.get("specialist_epoch"), "row": row, "facts": learner.facts(case, item["stage_id"]),
        "vignette": learner.stage_view(case, item["stage_id"]), "recommendation": item["recommendation"],
    })


def model_status(case, item):
    state = _state(item)
    model_record = state.get("model")
    if not model_record:
        return None
    model = model_record["payload"]
    selected = state.get("selection", {}).get("scenario_id", model["baseline_scenario_id"])
    row = dm.current_scenario(model, case, item, selected)
    return {"row": row, "model": model, "scope_current": (
        _artifact_valid(model_record, case, item) and model_record.get("model_basis") == _model_basis(item)
    ),
            "selection_current": _matches(state.get("selection"), _selection_signature(case, item, row))}


def _conflict_review_signature(case, item):
    return _signature({"case": case["id"], "epoch": case.get("specialist_epoch"),
                       "task": task_for(case, item), "sources": _sources(case, item),
                       "facts": learner.facts(case, item["stage_id"])})


def _unresolved_evidence(case, item):
    state = _state(item)
    signature = _conflict_review_signature(case, item)
    record = state.get("evidence_resolution", {})
    reviewed = record.get("reviewed_run_ids", []) if record.get("signature") == signature else []
    blockers = []
    if _artifact_valid(state.get("evidence"), case, item) and (
        record.get("signature") != signature or record.get("evidence") != _signature(state["evidence"])
    ):
        blockers += [c["resolution_needed"] for c in state["evidence"]["payload"].get("conflicts", [])]
    for run in _runs(case, item, "evidence_review"):
        if run.get("outcome") not in {"proposed", "needs_input", "blocked"} or run["request_id"] in reviewed:
            continue
        if run.get("evidence_basis") != signature:
            continue
        blockers += [i["message"] for i in run["response"]["issues"] if i["severity"] == "blocking"]
        blockers += [c["resolution_needed"] for c in run["response"]["payload"].get("conflicts", [])]
    return list(dict.fromkeys(blockers))


def _runs(case, item, role, pass_type=None):
    return [r for r in reversed(case.get("specialist_runs", []))
            if r["target"]["case_id"] == case["id"] and r["target"]["item_id"] == item["id"]
            and r["role"] == role and (pass_type is None or r.get("pass_type") == pass_type)]


def _current_review(case, item, pass_type, *, strict_revision=False):
    signature = _view_signature(case, item) if pass_type == "solve" else _key_signature(case, item)
    if pass_type == "cueing":
        signature = _cue_signature(case)
    return next((r for r in _runs(case, item, "item_validation", pass_type)
                 if r["outcome"] == "proposed" and r.get("snapshot") == signature
                 and (not strict_revision or r["target"]["base_revision"] == case["version"])), None)


def _cue_signature(case):
    return _signature({"case": case["id"], "epoch": case.get("specialist_epoch"),
                       "vignette": case["patient"], "sequence": learner.sequence(case),
                       "policy": learner.delivery_policy(case)})


def _assert_selection(case, item):
    status = model_status(case, item)
    if status is None or not status["scope_current"]:
        raise ValueError("Review a decision model for the current task, evidence, and disclosure stage.")
    if not status["selection_current"] or status["row"]["outcome"] not in dm.RESOLVED:
        raise ValueError("Confirm the selected case and its locally resolved key before drafting.")
    if _unresolved_evidence(case, item):
        raise ValueError("Record an SME resolution for the material evidence conflict or gap first.")
    return status


def _approved_key(case, item):
    approval = _state(item).get("key_approval", {})
    if not _matches(approval, _key_signature(case, item)):
        raise ValueError("Record the SME-approved option IDs for this item snapshot.")
    return key_ids(item)


def _project_model(case, item, status, claims, include_neighbors, allowed_sources):
    model, row = status["model"], status["row"]
    feature_ids = dm.visible_feature_ids(model, case, item["stage_id"])
    visible_features = [f for f in model["features"] if f["feature_id"] in feature_ids]
    linked = _state(item)["model"].get("feature_sources", {})
    if any(set(linked.get(fid, [])) - allowed_sources for fid in feature_ids):
        raise ValueError("The selected provider lacks permission for a source linked to a visible feature.")
    rules = []
    for rule in model["decision_rules"]:
        used = core._rule_features([rule.get("scope_condition", {}), rule["condition"]]) - {"scenario"}
        if not used <= feature_ids:
            continue
        if not include_neighbors and rule["rule_id"] not in row["triggered_rule_ids"]:
            continue
        if not set(rule["evidence_claim_ids"]) <= claims:
            continue
        rules.append(rule)
    neighbors = []
    if include_neighbors:
        for alt in dm.scenario_rows(model, case, item).values():
            if (alt["scenario_id"] != row["scenario_id"] and alt["outcome"] in dm.RESOLVED
                    and set(alt["evidence_claim_ids"]) <= claims):
                neighbors.append({k: alt[k] for k in ("scenario_id", "visible_values", "keyed_action_ids")})
    action_ids = set(row["keyed_action_ids"]) | {a for n in neighbors for a in n["keyed_action_ids"]}
    if include_neighbors:
        action_ids |= {a for r in rules for a in r["action_ids"]}
    return {
        "features": [{k: v for k, v in f.items() if k not in {"allowed_values", "origin"}} for f in visible_features],
        "actions": [a for a in model["actions"] if a["action_id"] in action_ids],
        "decision_rules": rules, "visible_values": row["visible_values"],
        "neighbor_scenarios": neighbors,
    }


def build_request(case, item, role, provider, *, pass_type=None):
    """Build one new request from persisted state and provider-authorized sources."""
    visible = learner.facts(case, item["stage_id"])
    task = task_for(case, item)
    sources = _sources(case, item, provider)
    state = _state(item)
    claims = _claims(case, item, provider)
    claim_ids = {c["claim_id"] for c in claims}
    inputs, accepted = {}, {}
    if role == "bank_retrieval":
        bank = _rank_bank([s for s in sources if s["source_class"] == "bank"], task, item)
        inputs["bank_candidates"] = [_approved_source(s) | {"item_id": s["source_id"]} for s in bank]
    elif role == "evidence_review":
        inputs["source_excerpts"] = [_approved_source(s) for s in sources if s["source_class"] in CLINICAL_CLASSES]
        inputs["practice_summaries"] = [_approved_source(s) | {"summary_id": s["source_id"]}
                                        for s in sources if s["source_class"] == "practice summary"]
    elif role == "key_feature_modeling":
        missing = [k for k in ("decision", "setting", "population", "objective") if not task[k].strip()]
        if missing:
            raise ValueError("Complete the decision, blueprint objective, setting and clinical population.")
        inputs = {"evidence_claims": claims, "bank_patterns": _bank_patterns(case, item, provider),
                  "author_constraints": [{"visible_stage_ids": sorted({item["stage_id"]} | {
                      s["stage_id"] for s in learner.stage_view(case, item["stage_id"])["visible_disclosures"]
                  })}]}
        visible_ids = {f["fact_id"] for f in visible}
        feature_names = {f.get("name") for f in case["features"]
                         if f.get("disclosed_at") in (None, "", "shared", *inputs["author_constraints"][0]["visible_stage_ids"])}
        inputs["author_constraints"].extend(c for c in case["constraints"] if core._rule_features(c) <= feature_names)
        if state.get("intended_key") or key_ids(item):
            inputs["intended_key"] = state.get("intended_key") or [o["text"] for o in item["options"] if o["correct"]]
        accepted["locked_fields"] = {"recommendation": item["recommendation"]} if case["mode"] == "appraisal" else {}
    elif role in {"item_drafting", "distractor_modeling"}:
        status = _assert_selection(case, item)
        row = status["row"]
        if set(row["evidence_claim_ids"]) - claim_ids:
            raise ValueError("The selected provider lacks authorization for one or more reviewed key sources.")
        accepted = {"scenario_id": row["scenario_id"], "keyed_action_ids": row["keyed_action_ids"],
                    "decision_model": _project_model(case, item, status, claim_ids, role == "distractor_modeling",
                                                     {s["source_id"] for s in sources})}
        if case["mode"] == "appraisal":
            accepted["preliminary_recommendation"] = item["recommendation"]
        if role == "item_drafting":
            accepted["locked_wording"] = {
                "shared_vignette": case["patient"],
                "visible_disclosures": learner.stage_view(case, item["stage_id"])["visible_disclosures"],
                **{k: item[k] for k in state.get("locked_wording_fields", [])},
            }
        else:
            if not item["lead_in"].strip() or not _matches(state.get("drafting"), _wording_signature(case, item)):
                raise ValueError("Confirm the wording for this scenario before modeling options.")
            accepted |= {k: item[k] for k in ("stem", "lead_in", "selection_instruction")}
            inputs = {"target_option_count": target_count(item), "bank_patterns": _bank_patterns(case, item, provider),
                      "misconception_labels": list(core.MISCONCEPTIONS),
                      "misconception_evidence": [_approved_source(s) for s in sources if s["source_class"] == "misconception evidence"]}
    elif role == "item_validation":
        if pass_type == "cueing":
            if case["mode"] != "unfolding":
                raise ValueError("Sequence cueing applies to unfolding cases.")
            inputs = {"pass_type": "cueing", "shared_vignette": case["patient"],
                      "stage_views": learner.sequence(case), "delivery_policy": learner.delivery_policy(case)}
            visible = []
        else:
            excerpts = [{k: s[k] for k in ("source_id", "source_date", "locator", "excerpt")} | {"evidence_claim_ids": [c["claim_id"] for c in claims
                         if c["source_id"] == s["source_id"]]}
                        for s in sources if s["source_class"] in CLINICAL_CLASSES]
            inputs = learner.stage_view(case, item["stage_id"]) | learner.item_view(case, item, opaque=True) | {
                "pass_type": pass_type, "source_excerpts": excerpts}
            if pass_type == "compare":
                solve = _current_review(case, item, "solve", strict_revision=True)
                if solve is None:
                    raise ValueError("Record a blinded solve against this exact item revision first.")
                if solve["provider"] != provider:
                    raise ValueError("Invoke a fresh blinded solve on the current provider route before comparing.")
                if solve.get("request_snapshot") != _signature({k: v for k, v in inputs.items() if k != "pass_type"}):
                    raise ValueError("The source projection changed. Record a fresh blinded solve.")
                accepted["approved_key_option_ids"] = [learner.option_token(item, oid) for oid in _approved_key(case, item)]
                inputs["solve_result"] = deepcopy(solve["response"]["payload"])
    elif role == "rationale_writing":
        approved = _approved_key(case, item)
        compare = _current_review(case, item, "compare")
        if (not compare or comparison_status(compare) != "agrees"
                or review_has_findings(compare)) and not _human_review_current(case, item):
            raise ValueError("Record a resolved comparison or a current human clinical review before writing rationales.")
        accepted = {"item": learner.stage_view(case, item["stage_id"]) | learner.item_view(case, item),
                    "approved_key_option_ids": approved}
        option_judgments = []
        selection = state.get("selection", {})
        permitted_rules = {r["rule_id"]: r for r in state.get("model", {}).get("payload", {}).get("decision_rules", [])
                           if set(r["evidence_claim_ids"]) <= claim_ids}
        allowed_sources = {s["source_id"] for s in sources}
        for o in item["options"]:
            row = {"option_id": o["id"], "keyed": o["correct"]}
            details = o.get("specialist_judgment", {})
            if (set(details.get("evidence_claim_ids", [])) <= claim_ids and _modeled_judgments_current(case, item)
                    and set(details.get("bank_item_ids", []) + details.get("misconception_evidence_ids", [])) <= allowed_sources):
                row |= {k: details[k] for k in ("plausibility_rationale", "clinical_rationale",
                        "violated_rule_ids", "evidence_claim_ids", "misconception_hypothesis", "evidence_status") if k in details}
            action = o.get("modeled_action_id")
            row["rule_ids"] = [rid for rid, rule in permitted_rules.items() if action in rule["action_ids"]
                               and rid in selection.get("triggered_rule_ids", []) + details.get("violated_rule_ids", [])]
            option_judgments.append(row)
        inputs = {"option_judgments": option_judgments, "evidence_claims": claims}

    request = {
        "request_id": core.uid("run"),
        "target": {"case_id": case["id"], "item_id": item["id"], "stage_id": item["stage_id"],
                   "base_revision": case["version"]},
        "task": task, "visible_facts": visible, "accepted_fields": accepted, "role_input": inputs,
    }
    return project_specialist_request(role, request)


def validate_handoff(role, request, result):
    """Cross-check IDs and semantics that the response schema cannot express."""
    p, data, accepted = result["payload"], request["role_input"], request["accepted_fields"]
    known_facts = {f["fact_id"] for f in request["visible_facts"]}
    claims = {r["claim_id"] for r in data.get("evidence_claims", [])}
    claims |= {c for s in data.get("source_excerpts", []) for c in s.get("evidence_claim_ids", [])}
    def subset(values, allowed):
        if set(values) - set(allowed) or len(set(values)) != len(values):
            raise ValueError("A reference is invalid or repeated.")
    if role == "bank_retrieval":
        records = dm.by_id(data["bank_candidates"], "item_id")
        for match in p.get("matches", []):
            allowed = records[match["item_id"]].get("observed_option_evidence", [])
            if any(row not in allowed for row in match["observed_option_evidence"]):
                raise ValueError("Observed option evidence must match the authorized historical record.")
        for entry in p.get("candidate_alternatives", []):
            subset(entry["item_ids"], records)
        for flag in p.get("duplicate_flags", []):
            subset([flag["item_id"]], records)
    elif role == "evidence_review":
        allowed = dm.by_id(data["source_excerpts"], "source_id")
        extracted = dm.by_id(p.get("claims", []), "claim_id")
        for claim in extracted.values():
            source = allowed[claim["source_id"]]
            if claim["locator"] != source["locator"] or claim["source_date"] not in (None, source["source_date"]):
                raise ValueError("The claim must cite the supplied locator and date.")
        for conflict in p.get("conflicts", []):
            subset(conflict["claim_ids"], extracted)
        subset([e["summary_id"] for e in p.get("practice_context", [])],
               [s["summary_id"] for s in data.get("practice_summaries", [])])
    elif role == "key_feature_modeling" and result["status"] == "proposed":
        stage_ids = {request["target"]["stage_id"]} | {
            stage for c in data.get("author_constraints", []) if isinstance(c, dict)
            for stage in c.get("visible_stage_ids", [])
        }
        dm.validate_model(p, stage_ids, known_facts, claims)
    elif role in {"item_drafting", "distractor_modeling"}:
        if "scenario_id" in p and p["scenario_id"] != accepted["scenario_id"]:
            raise ValueError("The response is for a different scenario.")
        subset(p.get("fact_ids_used", []), known_facts)
        locked = accepted.get("locked_wording", {})
        if any(p[k] != locked[k] for k in LOCKABLE_WORDING.keys() & locked.keys() & p.keys()):
            raise ValueError("The proposal changes wording locked by the writer.")
        model = accepted["decision_model"]
        actions = dm.by_id(model["actions"], "action_id")
        rules = dm.by_id(model["decision_rules"], "rule_id")
        neighbors = dm.by_id(model["neighbor_scenarios"], "scenario_id")
        evidence = {c for r in rules.values() for c in r["evidence_claim_ids"]}
        bank = {m["item_id"] for m in data.get("bank_patterns", {}).get("matches", [])} | {
            mid for a in data.get("bank_patterns", {}).get("candidate_alternatives", []) for mid in a["item_ids"]
        }
        for option in dm.by_id(p.get("options", []), "candidate_id").values():
            action = option["action_id"]
            if (action is None and option["eligibility"] != "review") or (action is not None and action not in actions):
                raise ValueError("Only review-only ideas may omit a modeled action ID.")
            subset(option["attractive_fact_ids"], known_facts)
            subset(option["violated_rule_ids"], rules)
            subset(option["evidence_claim_ids"], evidence)
            subset(option["appropriate_in_scenario_ids"], [
                sid for sid, n in neighbors.items() if action in n["keyed_action_ids"]
            ] + ([accepted["scenario_id"]] if action in accepted["keyed_action_ids"] else []))
            subset(option.get("bank_item_ids", []), bank)
            subset(option.get("misconception_evidence_ids", []),
                   [e["source_id"] for e in data.get("misconception_evidence", [])])
            if option.get("misconception_hypothesis") and not option.get("misconception_evidence_ids"):
                option["evidence_status"] = "author hypothesis"
    elif role == "item_validation":
        if "pass_type" in p and p["pass_type"] != data["pass_type"]:
            raise ValueError("The review uses a different pass.")
        ids = {o["option_id"] for o in data.get("options", [])}
        if data["pass_type"] == "cueing":
            ids = {o["option_id"] for s in data["stage_views"] for i in s["items"] for o in i["options"]}
        subset(p.get("missing_fact_ids", []), known_facts)
        for field in ("defensible_option_ids", "competing_option_ids"):
            subset(p.get(field, []), set() if data["pass_type"] == "cueing" else ids)
        if data["pass_type"] == "compare" and "defensible_option_ids" in p:
            for field in ("defensible_option_ids", "competing_option_ids", "missing_fact_ids"):
                if set(p[field]) != set(data["solve_result"][field]):
                    raise ValueError("The comparison must preserve the recorded solve.")
        for finding in p.get("findings", []):
            subset(finding["option_ids"], ids)
            subset(finding["evidence_claim_ids"], claims)
    elif role == "rationale_writing":
        ids = {o["option_id"] for o in accepted["item"]["options"]}
        rationales = dm.by_id(p.get("option_rationales", []), "option_id")
        if result["status"] == "proposed" and set(rationales) != ids:
            raise ValueError("Rationales must cover every displayed option.")
        for r in rationales.values():
            subset([r["option_id"]], ids)
            subset(r["fact_ids"], known_facts)
            subset(r["evidence_claim_ids"], claims)
            subset(r["rule_ids"], {rid for j in data["option_judgments"] if j["option_id"] == r["option_id"] for rid in j["rule_ids"]})


def target_count(item):
    return item.get("option_count") or max(2, len(item["options"]))


def item_for(case, item_id):
    item = next((i for i in case["items"] if i["id"] == item_id), None)
    if item is None or item["stage_id"] not in {s["id"] for s in case["stages"]}:
        raise ValueError("Select an existing item with a disclosure stage.")
    return item


def specialist_run(case, item, request, role, route, response=None, error=None):
    pass_type = request["role_input"].get("pass_type")
    row = {
        "request_id": request["request_id"], "target": deepcopy(request["target"]),
        "role": role, "pass_type": pass_type, "registry_version": specialist_registry()["version"],
        "provider": route["provider"], "model": route["model"], "time": core.timestamp(),
        "outcome": response["status"] if response else "failed", "base_snapshot": proposal_signature(case),
        "source_ids": [s["source_id"] for s in _sources(case, item, route["provider"])
                       if s["source_id"] in _request_source_ids(request, _claims(case, item, route["provider"]))],
    }
    if error:
        row["error"] = error
    if response:
        row["response"] = deepcopy(response)
    if role == "evidence_review":
        row["evidence_basis"] = _conflict_review_signature(case, item)
    if pass_type == "solve":
        row["snapshot"] = _view_signature(case, item)
        row["request_snapshot"] = _signature({k: v for k, v in request["role_input"].items() if k != "pass_type"})
        row["authorship"] = _authorship(case, item, route["provider"])
        row["option_map"] = {learner.option_token(item, o["id"]): o["id"] for o in item["options"]}
    elif pass_type == "compare":
        row["snapshot"] = _key_signature(case, item)
        solve = request["role_input"]["solve_result"]
        token_map = {learner.option_token(item, o["id"]): o["id"] for o in item["options"]}
        keyed = {token_map[token] for token in request["accepted_fields"]["approved_key_option_ids"]}
        solved = {token_map[token] for token in solve["defensible_option_ids"]}
        row["comparison"] = {
            "approved_key_option_ids": sorted(keyed), "solved_option_ids": sorted(solved),
            "competing_option_ids": [token_map[token] for token in solve["competing_option_ids"]],
            "missing_fact_ids": solve["missing_fact_ids"],
            "solve_request_id": _current_review(case, item, "solve", strict_revision=True)["request_id"],
            "status": "unresolved" if not solved or solve["competing_option_ids"] or solve["missing_fact_ids"]
                      or (item["format"] == "SBA" and len(solved) != 1)
                      else "agrees" if keyed == solved else "disagrees",
        }
    elif pass_type == "cueing":
        row["snapshot"] = _cue_signature(case)
    return row


def _request_source_ids(request, claims):
    ids = set()
    claim_sources = {c["claim_id"]: c["source_id"] for c in claims}
    def visit(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {"source_id", "item_id"} and isinstance(v, str):
                    ids.add(v)
                elif k in {"item_ids", "bank_item_ids"} and isinstance(v, list):
                    ids.update(v)
                elif k == "evidence_claim_ids" and isinstance(v, list):
                    ids.update(claim_sources[c] for c in v if c in claim_sources)
                else:
                    visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    visit(request["role_input"])
    visit(request["accepted_fields"])
    return ids


def proposal_signature(case):
    return _signature({k: v for k, v in case.items()
                       if k not in {"specialist_runs", "updated_at", "review_status", "reviews", "last_change"}})


def proposal_current(case, run):
    target = run["target"]
    return target["case_id"] == case["id"] and target["base_revision"] == case["version"] and (
        run.get("base_snapshot") == proposal_signature(case)
    )


def _authorship(case, item, solve_provider):
    state = _state(item)
    providers = {part: (state.get(part) or {}).get("provider") for part in ("model", "drafting", "options")}
    recorded = {p for p in providers.values() if p}
    general_ai = item.get("provenance") == "AI-proposed"
    if general_ai:
        label = ("Same-provider review · general AI contribution also recorded" if solve_provider in recorded
                 else "Blinded review · general authoring provider unrecorded")
    elif not recorded:
        label = "Manual authoring · provider-assisted blinded review" if item.get("provenance") == "human-authored" else "Authorship provider unrecorded · blinded review"
    elif solve_provider in recorded:
        label = "Same-provider review"
    else:
        label = "Cross-provider blinded review"
    return {"providers": providers, "solve_provider": solve_provider, "label": label}


def comparison_status(run):
    return run.get("comparison", {}).get("status")


def review_has_findings(run):
    response = run["response"]
    payload = response["payload"]
    return payload.get("recommendation") == "revise" or any(
        f["severity"] in {"review", "blocking"}
        for f in payload.get("findings", []) + response.get("issues", [])
    )


def _human_review_current(case, item):
    review = _state(item).get("clinical_review", {})
    return _matches(review, _key_signature(case, item)) and set(_clinical_run_ids(case, item)) <= set(review.get("reviewed_run_ids", []))


def _clinical_run_ids(case, item):
    signatures = {"solve": _view_signature(case, item), "compare": _key_signature(case, item)}
    return [r["request_id"] for r in _runs(case, item, "item_validation")
            if r.get("pass_type") in signatures and r.get("response") and r.get("snapshot") == signatures[r["pass_type"]]]


def _wording_signature(case, item):
    view = _learner_snapshot(case, item)
    view.pop("options")
    return _signature({"case": case["id"], "item": item["id"], "view": view,
                       "epoch": case.get("specialist_epoch"), "selection": _state(item).get("selection")})


def _judgment_signature(case, item):
    return _signature({"view": _view_signature(case, item), "model": _state(item).get("model"),
                       "selection": _state(item).get("selection"), "key": sorted(key_ids(item)),
                       "options": [{k: o.get(k) for k in ("id", "eligible", "conditions", "modeled_action_id",
                                   "specialist_judgment", "option_type", "role", "supports", "excludes",
                                   "misconception", "misconception_status")} for o in item["options"]]})


def _modeled_judgments_current(case, item):
    return _matches(_state(item).get("options"), _judgment_signature(case, item))


def _signed_action(case, item, actor, notes, signature):
    return {"actor": actor, "notes": notes, "signature": signature, "time": core.timestamp(),
            "accepted_from_revision": case["version"], "case_id": case["id"], "item_id": item["id"],
            "decision": "confirmed"}


def _human(actor):
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("Enter the SME or reviewer recording the decision.")
    return actor.strip()


def save_model(case, item_id, model, actor, *, run=None):
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item.setdefault("decision_workflow", {})
    task = task_for(candidate, item)
    missing = [k for k in ("decision", "objective", "setting", "population") if not task[k].strip()]
    if missing:
        raise ValueError("Complete the decision, objective, setting, and clinical population first.")
    dm.validate_model(model, {s["id"] for s in case["stages"]}, {f["id"] for f in case["facts"]},
                      {c["claim_id"] for c in _claims(candidate, item)})
    # Persist the prior approvals as provenance when changing the model or reference.
    feature_ids = {f["feature_id"] for f in model["features"]}
    prior_model = state.get("model", {})
    feature_sources = {fid: list(ids) for fid, ids in prior_model.get("feature_sources", {}).items() if fid in feature_ids}
    _remember_acceptance(state, ("model", "selection", "key_approval", "clinical_review", "cueing_review"))
    model = deepcopy(model)
    for feature in model["features"]:
        if not feature.get("origin"):
            feature["origin"] = f"Proposed by {run['provider']}" if run else "SME-authored"
    state["model"] = _artifact(candidate, item, model, _human(actor), run) | {"model_basis": _model_basis(item)}
    if not run and prior_model.get("provider") and any(
        row in model[group] for group in ("features", "actions", "decision_rules", "feasibility_constraints")
        for row in prior_model["payload"][group]
    ):
        state["model"]["provider"] = prior_model["provider"]
        state["model"]["source_run_id"] = prior_model.get("run_id") or prior_model.get("source_run_id")
    if feature_sources:
        state["model"]["feature_sources"] = feature_sources
        references = state["model"].setdefault("source_links", {})
        for ids in feature_sources.values():
            for sid in ids:
                source = reviewed_source(candidate, item, sid)
                if source:
                    references[sid] = _source_reference(source)
                elif sid in prior_model.get("source_links", {}):
                    references[sid] = deepcopy(prior_model["source_links"][sid])
    return candidate


def _remember_acceptance(state, keys):
    for key in keys:
        if key in state:
            state.setdefault("history", []).append({key: state.pop(key)})


def accept_evidence(case, item_id, payload, actor, *, run=None, applicable_claim_ids=()):
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item.setdefault("decision_workflow", {})
    schema = specialist_definition("evidence_review")["response_schema"]
    payload_schema = schema["properties"]["payload"] | {"required": schema["then"]["properties"]["payload"]["required"]}
    if not Draft202012Validator(payload_schema).is_valid(payload):
        raise ValueError("Supply valid claims, conflicts, and practice context using the evidence schema.")
    sources = _sources(candidate, item)
    validate_handoff("evidence_review", {"accepted_fields": {}, "visible_facts": [],
        "role_input": {"source_excerpts": [s for s in sources if s["source_class"] in CLINICAL_CLASSES],
                       "practice_summaries": [{"summary_id": s["source_id"]} for s in sources
                                              if s["source_class"] == "practice summary"]}}, {"payload": payload})
    retrieved = {s["source_id"] for s in state.get("retrieved_sources", [])}
    from_collection = {c["claim_id"]: c for c in payload["claims"] if c["source_id"] in retrieved}
    applicable = set(applicable_claim_ids)
    if applicable - from_collection.keys():
        raise ValueError("Select applicability only for claims linked to the retrieved clinical passages.")
    if any(not from_collection[c]["applicability"].strip() for c in applicable):
        raise ValueError("Explain how the selected passage applies to the clinical task.")
    _remember_acceptance(state, ["evidence", "evidence_resolution"])
    state["evidence"] = _artifact(candidate, item, payload, _human(actor), run)
    if from_collection:
        state["evidence"]["applicable_retrieved_claim_ids"] = sorted(applicable)
    return candidate


def accept_scenario(case, item_id, scenario_id, actor, notes, *, apply_values=False, as_new=False):
    """Confirm a reviewed key from either authored observations or a selected variant."""
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item["decision_workflow"]
    model = state["model"]["payload"]
    if not model_status(candidate, item)["scope_current"]:
        raise ValueError("Review the decision model against the updated task or evidence first.")
    if _unresolved_evidence(candidate, item):
        raise ValueError("Resolve the material conflict or evidence gap before confirming a key.")
    scenario = dm.by_id(model["scenarios"], "scenario_id")[scenario_id]
    if apply_values:
        visible_ids = dm.visible_feature_ids(model, candidate, item["stage_id"])
        facts = dm.by_id(candidate["facts"], "id")
        for feature in model["features"]:
            fid = feature["feature_id"]
            if fid not in visible_ids or feature["variation"] != "variable" or fid not in scenario["fact_values"]:
                continue
            for fact_id in feature.get("fact_ids", []):
                fact = facts[fact_id]
                fact["value"] = scenario["fact_values"][fid]
                if "status" in fact:
                    fact["status"] = "observed" if fact["value"] is not None else "unknown"
    row = dm.current_scenario(model, candidate, item, scenario_id)
    if row["outcome"] not in dm.RESOLVED:
        raise ValueError("A scenario needs a feasible, determinate key at this stage.")
    if not as_new and any(row["visible_values"].get(k) != v for k, v in
                          dm.scenario_rows(model, candidate, item)[scenario_id]["visible_values"].items()):
        raise ValueError("The case facts differ from this scenario. Record the edited case as a new scenario or apply the scenario values.")
    if as_new:
        values = scenario["fact_values"] | row["visible_values"]
        scenario_id = core.uid("scenario")
        model["scenarios"].append({"scenario_id": scenario_id, "fact_values": values,
                                   "keyed_action_ids": [], "outcome": "undetermined", "evidence_claim_ids": []})
        row = dm.current_scenario(model, candidate, item, scenario_id)
        if row["outcome"] not in dm.RESOLVED:
            raise ValueError("The edited case needs reviewed rules that apply to its new scenario.")
    actions = dm.by_id(model["actions"], "action_id")
    intended = state.get("intended_key", "").strip()
    proposed = [actions[a]["label"] for a in row["keyed_action_ids"]]
    conflicts = intended and " ".join(intended.casefold().split()) not in {
        " ".join(v.casefold().split()) for v in proposed + [", ".join(proposed)]
    }
    prior = state["model"].get("review_questions", [])
    if (conflicts or row["unsupported_rule_ids"] or prior) and not notes.strip():
        raise ValueError("Document how you resolved the intended-key difference, proposal issue, or unsupported rule.")
    if len(row["keyed_action_ids"]) > target_count(item):
        raise ValueError("Increase the target option count to accommodate the complete key.")
    _remember_acceptance(state, ("selection", "key_approval", "clinical_review"))
    state["selection"] = _signed_action(candidate, item, _human(actor), notes, _selection_signature(candidate, item, row)) | {
        "scenario_id": scenario_id, "keyed_action_ids": row["keyed_action_ids"],
        "triggered_rule_ids": row["triggered_rule_ids"], "evidence_claim_ids": row["evidence_claim_ids"],
    }
    if apply_values or as_new:
        state["source_case_id"] = state.get("source_case_id", case["id"])
    return candidate


def approve_key(case, item_id, actor, notes):
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item.setdefault("decision_workflow", {})
    if _unresolved_evidence(candidate, item):
        raise ValueError("Resolve the evidence conflict before approving this key.")
    if not item["lead_in"].strip() or len(item["options"]) < 2 or any(not o["text"].strip() for o in item["options"]):
        raise ValueError("Complete the lead-in and displayed option wording first.")
    core.score_item(item, key_ids(item))
    if not core.selection_count_met(item, len(key_ids(item))):
        raise ValueError("The approved key does not match the configured selection count.")
    if state.get("model"):
        status = _assert_selection(candidate, item)
        if len(item["options"]) != target_count(item):
            raise ValueError("Fill the option shortfall or change the configured option count.")
        model, row = status["model"], status["row"]
        actions = dm.by_id(model["actions"], "action_id")
        mapped = [o.get("modeled_action_id") for o in item["options"]]
        if set(mapped) - actions.keys() or len(mapped) != len(set(mapped)):
            raise ValueError("Link each displayed option to a distinct, reviewed modeled action.")
        mapped_keys = {o["modeled_action_id"] for o in item["options"] if o["correct"]}
        if mapped_keys != set(row["keyed_action_ids"]):
            raise ValueError("Link each keyed option to the correct approved action.")
        known_verdicts = set(row["inactive_rule_ids"] + row["superseded_rule_ids"])
        exclusions = {a for r in model["decision_rules"] if r["rule_id"] in known_verdicts for a in r["action_ids"]}
        if set(mapped) - mapped_keys - exclusions:
            raise ValueError("Review an applicable exclusion or precedence rule for each unkeyed action.")
        if len({actions[a]["option_type"].strip().casefold() for a in mapped}) != 1:
            raise ValueError("Keep displayed choices parallel in action type.")
        if not _modeled_judgments_current(candidate, item) and not notes.strip():
            raise ValueError("Reassess all displayed option judgments and record how the case or wording changes were resolved.")
        for option in item["options"]:
            if not option.get("eligible", True) or (
                option.get("conditions") and core.evaluate_decision_rule(
                    option["conditions"], row["visible_values"] | {"scenario": row["scenario_id"]}
                ) is not True
            ):
                raise ValueError("A displayed option is ineligible or its inclusion condition is unresolved.")
    _remember_acceptance(state, ("key_approval", "clinical_review"))
    state["key_approval"] = _signed_action(candidate, item, _human(actor), notes, _key_signature(candidate, item)) | {
        "option_ids": key_ids(item),
        "scenario_id": state.get("selection", {}).get("scenario_id"),
    }
    return candidate


def record_human_decision(case, item_id, actor, notes, kind):
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item.setdefault("decision_workflow", {})
    if not notes.strip():
        raise ValueError("Record your evidence and the question or finding you resolved.")
    if kind == "clinical_review":
        _approved_key(candidate, item)
        if _unresolved_evidence(candidate, item):
            raise ValueError("Resolve the evidence conflict first.")
        signature = _key_signature(candidate, item)
    elif kind == "drafting":
        selected = _assert_selection(candidate, item)["row"]
        if not item["lead_in"].strip():
            raise ValueError("Write the lead-in for the selected scenario first.")
        signature = _wording_signature(candidate, item)
    elif kind == "evidence_resolution":
        signature = _conflict_review_signature(candidate, item)
    elif kind == "cueing_review" and case["mode"] == "unfolding":
        signature = _cue_signature(candidate)
    else:
        raise ValueError("Choose a review decision that applies to this item.")
    _remember_acceptance(state, [kind])
    state[kind] = _signed_action(candidate, item, _human(actor), notes, signature)
    if kind == "evidence_resolution":
        state[kind]["evidence"] = _signature(state.get("evidence"))
        state[kind]["reviewed_run_ids"] = [r["request_id"] for r in _runs(case, item, "evidence_review")]
    elif kind == "clinical_review":
        state[kind]["reviewed_run_ids"] = _clinical_run_ids(case, item)
    elif kind == "cueing_review":
        state[kind]["reviewed_run_ids"] = [r["request_id"] for r in _runs(case, item, "item_validation", "cueing")]
    elif kind == "drafting":
        state[kind]["scenario_id"] = selected["scenario_id"]
        history = state.get("history", [])
        state[kind]["provider"] = next((r["drafting"].get("provider") for r in reversed(history)
                                        if "drafting" in r), None)
    return candidate


def assemble_candidate_options(case, item_id, payload, actor, *, run=None, selected_ids=None):
    """Apply the reviewed pool in one reversible update, preserving displayed option IDs."""
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item["decision_workflow"]
    actor = _human(actor)
    p = payload
    assembly_pool = p["options"]
    if selected_ids is not None:
        assembly_pool = [c for c in assembly_pool if c["candidate_id"] in selected_ids]
        if len(assembly_pool) != len(selected_ids):
            raise ValueError("The selected candidates have changed. Review the current pool.")
    previous_options = state.get("options", {})
    status = _assert_selection(candidate, item)
    if p["scenario_id"] != status["row"]["scenario_id"]:
        raise ValueError("These options target a different scenario.")
    pool = dm.option_pool(status["model"], status["row"], assembly_pool, target_count(item), case=candidate, item=item)
    if selected_ids is not None and {c["candidate_id"] for c in pool["selected"]} != set(selected_ids):
        raise ValueError("The selected options must be distinct, eligible, and fit the requested count.")
    if pool["missing_keys"] or pool["too_many_required"] or len(pool["selected"]) < 2:
        raise ValueError("Resolve missing keyed or eligible options before assembly.")
    _remember_acceptance(state, ["options", "key_approval", "clinical_review"])
    by_candidate = {o.get("specialist_judgment", {}).get("candidate_id", o["id"]): o for o in item["options"]}
    reserved_ids = {by_candidate[o["candidate_id"]]["id"] for o in pool["selected"] if o["candidate_id"] in by_candidate}
    by_action = {o["modeled_action_id"]: o for o in item["options"] if o.get("modeled_action_id") and o["id"] not in reserved_ids}
    item["option_count"] = target_count(item)
    actions = dm.by_id(status["model"]["actions"], "action_id")
    item["options"] = [
        (deepcopy(by_candidate.get(o["candidate_id"]) or by_action.get(o["action_id"])) or core.new_option()) | {
            "text": o["text"], "correct": o["eligibility"] == "keyed",
            "role": o["role"], "eligible": True, "modeled_action_id": o["action_id"],
            "option_type": actions[o["action_id"]]["option_type"],
            "supports": o.get("plausibility_rationale") or "", "excludes": o["clinical_rationale"],
            "rationale": o["clinical_rationale"], "misconception": o.get("misconception_hypothesis") or "",
            "misconception_status": o.get("evidence_status") or "author hypothesis",
            "specialist_judgment": deepcopy(o),
        } for o in pool["selected"]
    ]
    state["options"] = _artifact(candidate, item, p, actor, run) | {
        "scenario_id": p["scenario_id"], "signature": _judgment_signature(candidate, item),
    }
    if not run and previous_options.get("provider"):
        state["options"]["provider"] = previous_options["provider"]
        state["options"]["source_run_id"] = previous_options.get("run_id") or previous_options.get("source_run_id")
    return candidate


def accept_proposal(case, run, actor, *, applicable_claim_ids=()):
    if not proposal_current(case, run):
        raise ValueError("This proposal belongs to an older or different revision. Invoke the role again.")
    if run["outcome"] != "proposed":
        raise ValueError("Resolve the outstanding proposal issues before applying changes.")
    role, item_id = run["role"], run["target"]["item_id"]
    p = run["response"]["payload"]
    actor = _human(actor)
    if role == "key_feature_modeling":
        candidate = save_model(case, item_id, p, actor, run=run)
        _state(item_for(candidate, item_id))["model"]["review_questions"] = [
            i["message"] for i in run["response"]["issues"] if i["severity"] == "blocking"
        ]
        return candidate
    if role == "evidence_review":
        return accept_evidence(case, item_id, p, actor, run=run, applicable_claim_ids=applicable_claim_ids)
    candidate = deepcopy(case)
    item = item_for(candidate, item_id)
    state = item.setdefault("decision_workflow", {})
    if role == "bank_retrieval":
        _remember_acceptance(state, ["bank"])
        state["bank"] = _artifact(candidate, item, p, actor, run)
    elif role == "item_drafting":
        status = _assert_selection(candidate, item)
        if status["row"]["scenario_id"] != p["scenario_id"]:
            raise ValueError("This wording targets a different scenario.")
        if any(i["severity"] == "blocking" for i in run["response"]["issues"]):
            raise ValueError("Resolve the missing source or wording dependency before applying the draft.")
        if not p["lead_in"].strip():
            raise ValueError("The proposed item requires a lead-in.")
        _remember_acceptance(state, ["drafting", "key_approval", "clinical_review"])
        item.update({k: p[k] for k in LOCKABLE_WORDING if k not in state.get("locked_wording_fields", [])})
        item["required_facts"] = list(dict.fromkeys(item["required_facts"] + p["fact_ids_used"]))
        state["drafting"] = _artifact(candidate, item, p, actor, run) | {
            "scenario_id": p["scenario_id"], "signature": _wording_signature(candidate, item),
        }
    elif role == "distractor_modeling":
        if any(i["severity"] == "blocking" for i in run["response"]["issues"]):
            raise ValueError("Resolve the blocking proposal issue before assembling options.")
        return assemble_candidate_options(case, item_id, p, actor, run=run)
    elif role == "rationale_writing":
        _approved_key(candidate, item)
        if any(i["severity"] == "blocking" for i in run["response"]["issues"]):
            raise ValueError("Resolve the blocking rationale issue before applying explanations.")
        _remember_acceptance(state, ["rationales"])
        by_option = dm.by_id(p["option_rationales"], "option_id")
        for option in item["options"]:
            option["rationale"] = by_option[option["id"]]["candidate_text"]
        state["rationales"] = _artifact(candidate, item, p, actor, run) | {
            "signature": _key_signature(candidate, item),
            "scenario_id": state.get("selection", {}).get("scenario_id"),
        }
    else:
        raise ValueError("Review findings are retained as evidence and cannot apply edits or an approval.")
    return candidate


def review_findings(case, item):
    """Release gates for the optional modeled workflow and review findings."""
    issues = _unresolved_evidence(case, item)
    state = _state(item)
    if state.get("model"):
        status = model_status(case, item)
        if not status["scope_current"]:
            issues.append("The decision model needs review for the current task, evidence or stage.")
        row = status["row"]
        if row["outcome"] not in dm.RESOLVED:
            issues.extend(row["reasons"])
        elif not status["selection_current"]:
            issues.append("Case facts, disclosure or scenario changed. Reassess the selected key, options and rationales.")
        if state.get("options") and not _modeled_judgments_current(case, item) and (
            not _matches(state.get("key_approval"), _key_signature(case, item))
        ):
            issues.append("Reassess option eligibility and clinical judgments against the current case.")
        if row["unsupported_rule_ids"]:
            issues.append("Supply applicable clinical support for the keyed rule before pilot or release.")
        if not _human_review_current(case, item):
            issues.append("Record a current human clinical review of the modeled item.")
    if state.get("model") or state.get("key_approval"):
        try:
            _approved_key(case, item)
        except ValueError as exc:
            issues.append(str(exc))
    solve = _current_review(case, item, "solve")
    compare = _current_review(case, item, "compare")
    latest_solves = _runs(case, item, "item_validation", "solve")
    latest_compares = _runs(case, item, "item_validation", "compare")
    for recent, signature in ((latest_solves, _view_signature(case, item)), (latest_compares, _key_signature(case, item))):
        if recent and recent[0]["outcome"] in {"blocked", "needs_input"} and recent[0].get("snapshot") == signature and not _human_review_current(case, item):
            issues.extend(i["message"] for i in recent[0]["response"]["issues"] if i["severity"] != "info")
    if solve and not compare and not _human_review_current(case, item):
        issues.append("Complete the blinded key comparison or record a human adjudication.")
    if compare and (comparison_status(compare) != "agrees" or review_has_findings(compare) or (solve and review_has_findings(solve))) and not _human_review_current(case, item):
        issues.append("The blinded review has material findings or a key disagreement. Record the SME's clinical resolution.")
    if case["mode"] == "unfolding" and (state.get("model") or _runs(case, item, "item_validation", "cueing")):
        cueing = _current_review(case, item, "cueing")
        recorded = state.get("cueing_review", {})
        human_cueing = _matches(recorded, _cue_signature(case)) and (
            not cueing or cueing["request_id"] in recorded.get("reviewed_run_ids", [])
        )
        if (not cueing or review_has_findings(cueing)) and not human_cueing:
            issues.append("Record an unfolding cueing review for this sequence and resolve any material findings.")
    return list(dict.fromkeys(issues))


def mark_reassessment(previous, saved, reason):
    """Retain the accepted record and flag scope changes even after manual reversions."""
    originals = {i["id"]: _state(i) for i in previous["items"]}
    for item in saved["items"]:
        old = originals.get(item["id"], {})
        state = _state(item)
        tracked = {k for k in ("model", "selection", "drafting", "options", "key_approval",
                               "clinical_review", "rationales", "cueing_review")
                   if state.get(k) and state[k] == old.get(k) and not state[k].get("requires_reassessment")}
        stage_exists = item["stage_id"] in {s["id"] for s in saved["stages"]}
        if tracked and stage_exists:
            if state.get("model"):
                status = model_status(saved, item)
                if "model" in tracked and not status["scope_current"]:
                    state["model"].update(requires_reassessment=True, reassessment_reason=reason)
                tracked.discard("model")
                if "selection" in tracked and (not status["scope_current"] or not status["selection_current"]):
                    state["selection"].update(requires_reassessment=True, reassessment_reason=reason)
                tracked.discard("selection")
            signature_for = {
                "drafting": lambda: _wording_signature(saved, item),
                "options": lambda: _judgment_signature(saved, item),
                "key_approval": lambda: _key_signature(saved, item),
                "clinical_review": lambda: _key_signature(saved, item),
                "rationales": lambda: _key_signature(saved, item),
                "cueing_review": lambda: _cue_signature(saved),
            }
            tracked = {key for key in tracked if key not in signature_for or not _matches(state[key], signature_for[key]())}
        for key in tracked:
            state[key].update(requires_reassessment=True, reassessment_reason=reason)


def validate_workflow_import(case):
    """Validate structures consumed by the workflow while retaining historical outputs as records."""
    permissions = case.get("retrieval_permissions", {})
    if not isinstance(permissions, dict) or any(not isinstance(rows, dict) for rows in permissions.values()):
        raise ValueError("Saved retrieval permissions must be grouped by item and source.")
    for rows in permissions.values():
        for permission in rows.values():
            if (not isinstance(permission, dict) or permission.get("collection") not in ("clinical", "bank")
                    or not isinstance(permission.get("source_revision"), str)
                    or not isinstance(permission.get("providers"), list)
                    or any(p not in ("openai", "anthropic") for p in permission["providers"])):
                raise ValueError("Each passage permission needs a collection, source revision, and approved providers.")
    if "specialist_epoch" in case and not isinstance(case["specialist_epoch"], str):
        raise ValueError("The specialist revision marker must be text.")
    runs = case.get("specialist_runs", [])
    if not isinstance(runs, list) or any(not isinstance(r, dict) for r in runs):
        raise ValueError("Specialist history must be a list of records.")
    for run in runs:
        if (not all(isinstance(run.get(k), str) for k in ("request_id", "role", "outcome", "provider", "model"))
                or not isinstance(run.get("target"), dict)
                or not isinstance(run["target"].get("base_revision"), int)
                or not all(isinstance(run["target"].get(k), str) for k in ("case_id", "item_id", "stage_id"))):
            raise ValueError("Invalid specialist history envelope.")
        response = run.get("response")
        if response is not None and (
            not isinstance(response, dict) or not isinstance(response.get("issues"), list)
            or not isinstance(response.get("payload"), dict) or any(not isinstance(i, dict) for i in response["issues"])
        ):
            raise ValueError("Invalid specialist response record.")
    for item in case["items"] + case["archived_items"]:
        state = _state(item)
        if not isinstance(state, dict):
            raise ValueError("An item decision workflow must be an object.")
        selected_sources = state.get("retrieved_sources", [])
        validate_sources(selected_sources)
        for source in selected_sources:
            if len(source.get("stage_ids", [])) != 1 or len(source.get("item_ids", [])) != 1:
                raise ValueError("Each selected retrieval source must be bound to one item and disclosure stage.")
            if source["source_class"] in CLINICAL_CLASSES and not (source.get("citation") or source.get("url")):
                raise ValueError("A retrieved clinical passage needs its original citation.")
        if "candidate_reviews" in state:
            from authoring_spaces import validate_candidate_reviews
            validate_candidate_reviews(state["candidate_reviews"])
        for field in ("decision", "decision_timing", "assessment_intent", "examinee_population", "setting", "population", "intended_key"):
            if field in state and not isinstance(state[field], str):
                raise ValueError(f"Item decision workflow field {field} must be text.")
        locked = state.get("locked_wording_fields", [])
        if not isinstance(locked, list) or any(not isinstance(k, str) or k not in LOCKABLE_WORDING for k in locked):
            raise ValueError("An item contains an unknown wording lock.")
        for field in ("model", "selection", "drafting", "options", "evidence", "bank", "rationales",
                      "key_approval", "clinical_review", "cueing_review", "evidence_resolution"):
            if field in state and not isinstance(state[field], dict):
                raise ValueError(f"Item decision workflow field {field} must be an object.")
        if state.get("model"):
            try:
                model, selected = state["model"]["payload"], state.get("selection", {}).get("scenario_id")
                linked = state["model"].get("feature_sources", {})
                if not isinstance(linked, dict) or any(not isinstance(ids, list)
                    or any(not isinstance(sid, str) for sid in ids) for ids in linked.values()):
                    raise ValueError("Feature source links must map feature IDs to lists of source IDs.")
                # Missing former links may be a legitimate result of a later manual edit.
                stage_ids = {f["disclosure_stage_id"] for f in model["features"]}
                fact_ids = {fid for f in model["features"] for fid in f.get("fact_ids", [])}
                claim_ids = {cid for group in ("decision_rules", "feasibility_constraints", "scenarios")
                             for row in model[group] for cid in row["evidence_claim_ids"]}
                claim_ids.update(model["proposed_key"]["source_claim_ids"])
                dm.validate_model(model, stage_ids, fact_ids, claim_ids)
                if selected and selected not in dm.by_id(model["scenarios"], "scenario_id"):
                    raise ValueError("The selected scenario is not in the saved model.")
            except (KeyError, TypeError, AttributeError, RecursionError):
                raise ValueError("The saved decision model is incomplete or invalid.") from None
        evidence = state.get("evidence", {})
        applicable = evidence.get("applicable_retrieved_claim_ids", [])
        if not isinstance(applicable, list) or any(not isinstance(cid, str) for cid in applicable):
            raise ValueError("Applicable retrieved claims must be a list of claim IDs.")
        for name in ("model", "evidence", "bank", "options", "rationales"):
            claim_sources = state.get(name, {}).get("claim_sources", {})
            if not isinstance(claim_sources, dict) or any(not isinstance(sid, str) for sid in claim_sources.values()):
                raise ValueError("Recorded claim references must map claim IDs to source IDs.")
            links = state.get(name, {}).get("source_links", {})
            if not isinstance(links, dict) or any(not isinstance(ref, dict) or ref.get("source_id") != sid
                or any(not isinstance(v, str) for v in ref.values()) for sid, ref in links.items()):
                raise ValueError("A saved source reference must include its source ID and text provenance.")
    validate_manual_sources(case, case.get("specialist_sources", []))
