"""Local feature and candidate review against one item, disclosure stage, and scenario."""

from copy import deepcopy

from jsonschema import Draft202012Validator

import core
import decision_model as dm
import specialist_workflow as sw
from ai_adapter import specialist_definition


def display_value(value):
    if value is None:
        return "Unknown"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def predicate_text(predicate, labels):
    if isinstance(predicate, list):
        return " and ".join(predicate_text(p, labels) for p in predicate) or "Always"
    if not predicate:
        return "Always"
    for group, conj in (("all", " and "), ("any", " or ")):
        if group in predicate:
            return "(" + conj.join(predicate_text(p, labels) for p in predicate[group]) + ")"
    if "not" in predicate:
        return "not (" + predicate_text(predicate["not"], labels) + ")"
    ops = {"equals": "=", "not_equals": "≠", "gte": "≥", "lte": "≤", "gt": ">", "lt": "<",
           "exists": "is observed", "not_exists": "is undocumented", "not_in": "outside", "in": "in"}
    name = labels.get(predicate["feature"], predicate["feature"])
    operator = predicate.get("operator", "equals")
    op = ops.get(operator, operator)
    value = predicate.get("value")
    rhs = ", ".join(map(display_value, value)) if isinstance(value, list) else display_value(value)
    return f"{name} {op}" + (f" {rhs}" if "value" in predicate else "")


def _used_features(rule):
    return core._rule_features([rule.get("scope_condition", {}), rule["condition"]])


def claim_sources(item, field):
    """Resolve historical claim IDs against their accepted source revision."""
    state = sw._state(item)
    current = {c["claim_id"]: c["source_id"] for c in state.get("evidence", {}).get("payload", {}).get("claims", [])}
    return current | state.get(field, {}).get("claim_sources", {})


def _same_assignment(left, right):
    return left.keys() == right.keys() and all(dm.includes([value], right[k]) for k, value in left.items())


def space_context(case, item, scenario_id=None, changes=None):
    status = sw.model_status(case, item)
    if not status:
        raise ValueError("Review a decision model for this item first.")
    model = status["model"]
    scenario_id = scenario_id or status["row"]["scenario_id"]
    scenarios = dm.by_id(model["scenarios"], "scenario_id")
    if scenario_id not in scenarios:
        raise ValueError("The selected scenario is no longer in the reviewed model.")
    features = dm.by_id(model["features"], "feature_id")
    rows = dm.scenario_rows(model, case, item)
    selected = scenarios[scenario_id]
    row = status["row"] if scenario_id == status["row"]["scenario_id"] else rows[scenario_id]
    visible = dm.visible_feature_ids(model, case, item["stage_id"])
    values = selected["fact_values"] | {k: row["visible_values"].get(k) for k in visible}
    changes = changes or {}
    for fid, value in changes.items():
        if fid not in visible or features[fid]["variation"] != "variable":
            raise ValueError("Only a variable feature disclosed in the active stage can be varied.")
        if value is not None and not dm.includes(features[fid]["allowed_values"], value):
            raise ValueError("Choose a permitted value for the reviewed feature.")
    trial_values = values | changes
    trial = not _same_assignment(trial_values, values)
    if trial:
        matched = next((s for s in model["scenarios"] if _same_assignment(s["fact_values"], trial_values)), None)
        selected = matched or {"scenario_id": "preview-" + sw._signature(trial_values), "fact_values": trial_values}
        assignment = {k: v for k, v in trial_values.items() if k in visible and v is not None}
        problems = []
        if scenario_id == status["row"]["scenario_id"]:
            _, problems = dm.visible_assignment(model, scenarios[scenario_id], case, item["stage_id"], live=True)
        row = dm.evaluate(model, selected, assignment, item["format"], problems)
    baseline = rows[model["baseline_scenario_id"]]["keyed_action_ids"]
    if baseline and row["outcome"] in dm.RESOLVED and selected["scenario_id"] != model["baseline_scenario_id"]:
        row = row | {"outcome": "key_retained" if row["keyed_action_ids"] == baseline else "key_changed"}
    return {"status": status, "model": model, "row": row, "rows": rows, "selected": selected,
            "stage_id": item["stage_id"], "visible": visible, "values": trial_values,
            "base_id": scenario_id, "trial": trial,
            "confirmed": not trial and scenario_id == status["row"]["scenario_id"]
                         and status["scope_current"] and status["selection_current"]}


def confirm_trial(case, item_id, scenario_id, changes, actor, notes):
    item = sw.item_for(case, item_id)
    ctx = space_context(case, item, scenario_id, changes)
    if not ctx["trial"] or ctx["row"]["outcome"] not in dm.RESOLVED:
        raise ValueError("Preview a feasible change with a determinate key first.")
    if not ctx["status"]["scope_current"]:
        raise ValueError("Review the current model scope before confirming a scenario.")
    sid = ctx["selected"]["scenario_id"]
    candidate = deepcopy(case)
    if sid not in ctx["rows"]:
        sid = core.uid("scenario")
        model = sw.item_for(candidate, item_id)["decision_workflow"]["model"]["payload"]
        model["scenarios"].append({"scenario_id": sid, "fact_values": ctx["values"], "outcome": "undetermined",
                                   "evidence_claim_ids": [], "keyed_action_ids": []})
    return sw.accept_scenario(candidate, item_id, sid, actor, notes, apply_values=True)


def feature_rows(case, item, ctx):
    model = ctx["model"]
    stages = {"shared": "Shared", **{s["id"]: s["title"] for s in case["stages"]}}
    facts = {f["id"]: f.get("name") or f["id"] for f in case["facts"]}
    features = dm.by_id(model["features"], "feature_id")
    claim_refs = claim_sources(item, "model")
    details = {}
    for fid, feature in features.items():
        rules = [r for r in model["decision_rules"] if fid in _used_features(r)]
        constraints = [c for c in model["feasibility_constraints"] if fid in _used_features(c)]
        coupled = {k for c in constraints for k in _used_features(c) if k != fid and k in features}
        claims = sorted({k for r in rules + constraints for k in r["evidence_claim_ids"]})
        sources = set(item["decision_workflow"]["model"].get("feature_sources", {}).get(fid, []))
        sources.update(claim_refs[c] for c in claims if c in claim_refs)
        details[fid] = {
            "Feature": feature["label"], "Value": (display_value(ctx["row"]["visible_values"].get(fid))
                                                  if fid in ctx["visible"] else "Withheld at this stage"),
            "Permitted": ", ".join(map(display_value, feature["allowed_values"])), "Unit": feature.get("unit") or "",
            "Variation": feature["variation"], "Impact": feature["classification"],
            "Linked facts": ", ".join(f"{facts.get(f, f)} ({f})" for f in feature.get("fact_ids", [])),
            "Disclosure": stages.get(feature["disclosure_stage_id"], "Removed stage"),
            "Rules": ", ".join(r["rule_id"] for r in rules),
            "Coupled features": ", ".join(features[k]["label"] for k in sorted(coupled)),
            "Feasibility constraints": ", ".join(c["constraint_id"] for c in constraints),
            "Evidence claims": ", ".join(claims), "Sources": ", ".join(sorted(sources)),
            "Origin": feature.get("origin", ""),
        }
    return details


def edit_feature(case, item_id, feature_id, fields, source_ids, actor):
    item = sw.item_for(case, item_id)
    record = item["decision_workflow"]["model"]
    model = deepcopy(record["payload"])
    feature = dm.by_id(model["features"], "feature_id")[feature_id]
    if set(fields) - {"label", "allowed_values", "unit", "variation", "classification", "fact_ids", "disclosure_stage_id"}:
        raise ValueError("Choose an editable feature field.")
    if not fields.get("label", feature["label"]).strip():
        raise ValueError("Enter a feature label.")
    existing = record.get("feature_sources", {}).get(feature_id, [])
    available = dm.by_id(sw._sources(case, item), "source_id")
    if set(source_ids) - set(existing) - available.keys():
        raise ValueError("Choose a source available to the selected item.")
    feature.update(deepcopy(fields))
    updated = sw.save_model(case, item_id, model, actor)
    accepted = sw.item_for(updated, item_id)["decision_workflow"]["model"]
    accepted["feature_sources"] = deepcopy(record.get("feature_sources", {})) | {feature_id: list(source_ids)}
    if source_ids:
        accepted.setdefault("source_links", {}).update({
            sid: sw._source_reference(available[sid]) for sid in source_ids if sid in available
        })
    return updated


def _blank_candidate(action):
    return {
        "candidate_id": "modeled-" + action["action_id"], "action_id": action["action_id"],
        "text": action["label"], "eligibility": "review", "role": "conditional",
        "plausibility_rationale": None, "clinical_rationale": "", "attractive_fact_ids": [],
        "violated_rule_ids": [], "appropriate_in_scenario_ids": [], "evidence_claim_ids": [], "bank_item_ids": [],
        "misconception_hypothesis": None, "evidence_status": None,
    }


def _review_basis(case, item, ctx):
    return sw._signature({
        "item": sw._judgment_signature(case, item), "stage": item["stage_id"], "row": ctx["row"],
        "sources": sw._artifact_basis(case, item), "run": _pool_run(item),
    })


def _pool_run(item):
    artifact = sw._state(item).get("options", {})
    return artifact.get("run_id") or artifact.get("source_run_id")


def _displayed_edits(option):
    """Detect edits in the regular item editor since this option was assembled."""
    applied = option.get("specialist_judgment", {})
    if not applied:
        return {}
    fields = {"action_id": option.get("modeled_action_id"), "text": option["text"],
              "plausibility_rationale": option.get("supports") or None,
              "clinical_rationale": option.get("excludes") or "",
              "misconception_hypothesis": option.get("misconception") or None, "role": option["role"]}
    edits = {k: v for k, v in fields.items() if v != applied.get(k)}
    if "misconception_hypothesis" in edits:
        edits["misconception_evidence_ids"] = []
        edits["evidence_status"] = "author hypothesis" if edits["misconception_hypothesis"] else None
    return edits


def candidate_rows(case, item, ctx):
    """Project candidates from the accepted pool and modeled actions, then recheck the active case."""
    state = sw._state(item)
    artifact = state.get("options", {})
    base = {o["candidate_id"]: deepcopy(o) for o in artifact.get("payload", {}).get("options", [])}
    accepted_ids = set(base)
    for rejected in artifact.get("payload", {}).get("rejected_candidates", []):
        if not rejected["action"].strip():
            continue
        cid = "rejected-" + sw._signature(rejected)[:16]
        c = _blank_candidate({"action_id": cid, "label": rejected["action"]})
        c.update(candidate_id=cid, action_id=None, clinical_rationale=rejected["reason"])
        base.setdefault(cid, c)
    # Retain existing hand-authored item choices and identifiers when the SME links them.
    displayed = {}
    for o in item["options"]:
        action_id = o.get("modeled_action_id")
        candidate_id = o.get("specialist_judgment", {}).get("candidate_id", o["id"])
        displayed[candidate_id] = o
        if not o["text"].strip() and candidate_id not in base:
            continue
        if candidate_id not in base:
            c = _blank_candidate({"action_id": action_id or o["id"], "label": o["text"]})
            c.update(candidate_id=candidate_id, action_id=action_id, plausibility_rationale=o.get("supports") or None,
                     clinical_rationale=o.get("excludes") or o.get("rationale", ""),
                     misconception_hypothesis=o.get("misconception") or None,
                     evidence_status="author hypothesis" if o.get("misconception") else None,
                     role=o.get("role", "conditional"))
            base[candidate_id] = c
    library_actions = {a["action_id"] for a in ctx["model"]["actions"]}
    mapped = {c.get("action_id") for c in base.values()}
    for action in ctx["model"]["actions"]:
        if action["action_id"] not in mapped:
            c = _blank_candidate(action)
            base.setdefault(c["candidate_id"], c)
    accepted_current = bool(artifact) and ctx["confirmed"] and sw._modeled_judgments_current(case, item)
    reviews = state.get("candidate_reviews", {})
    basis = _review_basis(case, item, ctx)
    metadata = {}
    for cid, candidate in base.items():
        review = reviews.get(cid)
        source_current = accepted_current and cid in accepted_ids
        decision = "Pending"
        if source_current:
            if cid in displayed:
                decision = "Selected"
            else:
                decision = "Eligible" if candidate["eligibility"] in {"eligible", "keyed"} else "Hold"
        actor = artifact.get("actor") if source_current else ""
        if review and review.get("run_id") == _pool_run(item):
            candidate.update(deepcopy(review["candidate"]))
            if review["basis"] == basis:
                decision, actor = review["decision"], review["actor"]
            else:
                decision, actor = "Reassess", review["actor"]
        elif cid in accepted_ids and not accepted_current:
            decision, actor = "Reassess", artifact.get("actor", "")
        if cid in displayed:
            edits = _displayed_edits(displayed[cid])
            if edits and (not review or review.get("displayed_edits") != sw._signature(edits)):
                candidate.update(edits)
        if decision in {"Reassess", "Pending", "Hold"}:
            candidate["eligibility"] = "review"
        action = candidate.get("action_id")
        neighbors = [sid for sid, row in ctx["rows"].items() if sid != ctx["row"]["scenario_id"]
                     and row["outcome"] in dm.RESOLVED and action in row["keyed_action_ids"]]
        inactive = set(ctx["row"]["inactive_rule_ids"] + ctx["row"]["superseded_rule_ids"])
        exclusions = [r for r in ctx["model"]["decision_rules"] if action in r["action_ids"] and r["rule_id"] in inactive]
        metadata[cid] = {"decision": decision, "actor": actor, "neighbors": neighbors, "exclusions": exclusions,
                         "modeled": action in library_actions}
    # Evaluate selected rows first so an unselected duplicate cannot displace an SME selection.
    ordered = sorted(base.values(), key=lambda c: metadata[c["candidate_id"]]["decision"] != "Selected")
    pool = dm.option_pool(ctx["model"], ctx["row"], ordered, sw.target_count(item), case=case, item=item)
    return [r | metadata[r["candidate_id"]] for r in pool["rows"]]


def validate_candidate_reviews(reviews):
    if not isinstance(reviews, dict):
        raise ValueError("Candidate reviews must be keyed by candidate ID.")
    schema = specialist_definition("distractor_modeling")["response_schema"]
    candidate_schema = schema["properties"]["payload"]["properties"]["options"]["items"]
    validator = Draft202012Validator(candidate_schema)
    for cid, review in reviews.items():
        if (not isinstance(review, dict) or not isinstance(review.get("basis"), str)
                or not isinstance(review.get("actor"), str) or review.get("decision") not in {"Hold", "Selected", "Eligible"}
                or not validator.is_valid(review.get("candidate")) or review["candidate"]["candidate_id"] != cid):
            raise ValueError("Invalid candidate review or candidate data.")
        links = review.get("source_links", {})
        if not isinstance(links, dict) or any(
            not isinstance(ref, dict) or ref.get("source_id") != sid or
            not all(isinstance(value, str) for value in ref.values()) for sid, ref in links.items()
        ):
            raise ValueError("Invalid candidate source reference.")
        claims = review.get("claim_sources", {})
        if not isinstance(claims, dict) or any(not isinstance(sid, str) or sid not in links for sid in claims.values()):
            raise ValueError("Invalid candidate claim source.")


def review_candidate(case, item_id, cid, fields, decision, actor, scenario_id):
    item = sw.item_for(case, item_id)
    ctx = space_context(case, item, scenario_id)
    rows = candidate_rows(case, item, ctx)
    row = dm.by_id(rows, "candidate_id")[cid]
    allowed = {"action_id", "text", "clinical_rationale", "plausibility_rationale", "attractive_fact_ids",
               "violated_rule_ids", "evidence_claim_ids", "bank_item_ids", "role", "misconception_hypothesis"}
    if set(fields) - allowed:
        raise ValueError("Choose an editable candidate field.")
    c = deepcopy({k: v for k, v in row.items() if k not in
                  {"decision", "actor", "neighbors", "exclusions", "modeled", "local_judgment", "review_reason"}})
    c.update(fields)
    if c["action_id"] and c["action_id"] not in {a["action_id"] for a in ctx["model"]["actions"]}:
        raise ValueError("Link the candidate to a reviewed action or leave it unmodeled.")
    c["eligibility"] = ("review" if decision == "Hold" else
                        "keyed" if c["action_id"] in ctx["row"]["keyed_action_ids"] else "eligible")
    if not c["clinical_rationale"].strip():
        raise ValueError("Explain the clinical judgment or what remains unresolved.")
    if not set(c["attractive_fact_ids"]) <= {f["id"] for f in core.visible_facts(case, item["stage_id"])}:
        raise ValueError("Choose only findings disclosed at this stage.")
    if not set(c["violated_rule_ids"]) <= {r["rule_id"] for r in ctx["model"]["decision_rules"]}:
        raise ValueError("Choose a rule in the active model.")
    if not set(c["evidence_claim_ids"]) <= {r["claim_id"] for r in sw._claims(case, item)}:
        raise ValueError("Choose claims accepted for the current scope.")
    if any((sw.reviewed_source(case, item, key) or {}).get("source_class") != "bank" for key in c.get("bank_item_ids", [])):
        raise ValueError("Choose an available approved bank record.")
    evidence = c.get("misconception_evidence_ids", [])
    if (c.get("misconception_hypothesis") != row.get("misconception_hypothesis") or not evidence or
        any((sw.reviewed_source(case, item, sid) or {}).get("source_class") != "misconception evidence" for sid in evidence)):
        c.pop("misconception_evidence_ids", None)
        c["evidence_status"] = "author hypothesis" if c.get("misconception_hypothesis") else None
    c["appropriate_in_scenario_ids"] = [sid for sid, r in ctx["rows"].items()
        if sid != scenario_id and r["outcome"] in dm.RESOLVED and c["action_id"] in r["keyed_action_ids"]]
    if decision != "Hold":
        if not ctx["confirmed"]:
            raise ValueError("Confirm the scenario and current key before selecting eligible candidates.")
        other_selected = [r for r in rows if decision == "Selected" and r["decision"] == "Selected" and r["candidate_id"] != cid]
        pool = dm.option_pool(ctx["model"], ctx["row"], [*other_selected, c], sw.target_count(item), case=case, item=item)
        if pool["rows"][-1]["local_judgment"] == "review":
            raise ValueError(pool["rows"][-1]["review_reason"])
    review = {"decision": decision, "actor": sw._human(actor), "time": core.timestamp(),
              "scenario_id": scenario_id, "stage_id": item["stage_id"], "basis": _review_basis(case, item, ctx),
              "accepted_from_revision": case["version"], "candidate": c}
    review["run_id"] = _pool_run(item)
    for option in item["options"]:
        if option.get("specialist_judgment", {}).get("candidate_id", option["id"]) == cid:
            review["displayed_edits"] = sw._signature(_displayed_edits(option))
            break
    source_links = sw._artifact_sources(case, item, {"options": [c]})
    if source_links:
        review["source_links"] = source_links
        review["claim_sources"] = {claim["claim_id"]: claim["source_id"] for claim in sw._claims(case, item)
                                  if claim["claim_id"] in c["evidence_claim_ids"]}
    validate_candidate_reviews({cid: review})
    result = deepcopy(case)
    sw.item_for(result, item_id)["decision_workflow"].setdefault("candidate_reviews", {})[cid] = review
    return result


def assemble_reviewed(case, item_id, scenario_id, actor):
    item = sw.item_for(case, item_id)
    ctx = space_context(case, item, scenario_id)
    if not ctx["confirmed"]:
        raise ValueError("Confirm the displayed scenario before assembling reviewed candidates.")
    rows = candidate_rows(case, item, ctx)
    selected = {r["candidate_id"] for r in rows if r["decision"] == "Selected"}
    if not selected:
        raise ValueError("Select the keyed action and eligible alternatives for assembly.")
    options = []
    for r in rows:
        if not r["text"].strip():
            continue
        option = {k: deepcopy(v) for k, v in r.items() if k not in
                  {"decision", "actor", "neighbors", "exclusions", "modeled", "local_judgment", "review_reason"}}
        if not option["clinical_rationale"].strip():
            option["clinical_rationale"] = "Awaiting clinical review."
        options.append(option)
    rejected = sw._state(item).get("options", {}).get("payload", {}).get("rejected_candidates", [])
    payload = {"scenario_id": scenario_id, "options": options, "rejected_candidates": deepcopy(rejected),
               "shortage_count": max(0, sw.target_count(item) - sum(r["local_judgment"] != "review" for r in rows))}
    case_after = sw.assemble_candidate_options(case, item_id, payload, actor, selected_ids=selected)
    after_item = sw.item_for(case_after, item_id)
    reviews = sw._state(after_item).get("candidate_reviews", {})
    before_basis = _review_basis(case, item, ctx)
    after_basis = _review_basis(case_after, after_item, space_context(case_after, after_item, scenario_id))
    for record in reviews.values():
        if record["basis"] == before_basis:
            record["basis"] = after_basis
    return case_after
