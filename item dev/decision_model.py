"""SME-reviewed decision rules, scenario evaluation, and option eligibility."""

from itertools import chain, islice, product
import json

from jsonschema import Draft202012Validator

import core


RESOLVED = {"key_resolved", "key_retained", "key_changed"}
UNKNOWN_STATES = {"", "unknown", "unassessed", "not assessed", "undocumented", "not documented"}


def by_id(rows, field):
    values = {r[field]: r for r in rows}
    if len(values) != len(rows):
        raise ValueError(f"Duplicate {field} in the proposal.")
    return values


def includes(values, value):
    # Booleans and numbers have different clinical meanings despite Python equality.
    return any(v == value and (isinstance(v, bool) == isinstance(value, bool)) for v in values)


def value_of(fact, feature):
    state = str(fact.get("status", "")).strip().casefold().replace("_", " ")
    if state in UNKNOWN_STATES - {""}:
        return None
    value = fact.get("value")
    if isinstance(value, str) and value.strip().casefold() == "absent":
        state = "absent"
    if state == "absent" and "absent" in feature["allowed_values"]:
        return "absent"
    if state == "absent" and includes(feature["allowed_values"], False):
        return False
    if state in {"unusable", "rejected"}:
        return value if isinstance(value, str) and includes(feature["allowed_values"], value) else state
    if state in {"absent", "normal"} and value in (None, ""):
        return state
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().casefold().replace("_", " ") in UNKNOWN_STATES:
            return None
        if not includes(feature["allowed_values"], value):
            try:
                return json.loads(value.strip().lower())
            except ValueError:
                pass
    return value


def _validate_predicate(condition, features, scenarios):
    core._validate_rule_shape(condition)
    if core._rule_features(condition) - set(features) - {"scenario"}:
        raise ValueError("A predicate references an unknown feature.")
    if isinstance(condition, list):
        for part in condition:
            _validate_predicate(part, features, scenarios)
    elif condition:
        compounds = set(condition) & {"all", "any", "not"}
        if compounds:
            if len(compounds) != 1 or set(condition) - compounds - {"scenario"}:
                raise ValueError("Use one logical operator per compound predicate.")
            _validate_predicate(condition[next(iter(compounds))], features, scenarios)
        elif set(condition) - {"feature", "operator", "value", "scenario"}:
            raise ValueError("A comparison contains unsupported fields.")
        elif condition.get("operator", "equals") not in {"exists", "not_exists"} and "value" not in condition:
            raise ValueError("A comparison needs a value.")
        if condition.get("scenario") and condition["scenario"] not in scenarios:
            raise ValueError("A predicate references an unknown scenario.")


def validate_model(model, stage_ids, fact_ids, claim_ids):
    """Validate editable and provider-proposed models before either drives a handoff."""
    from ai_adapter import specialist_definition
    schema = specialist_definition("key_feature_modeling")["response_schema"]
    payload_schema = schema["properties"]["payload"] | {
        "required": schema["then"]["properties"]["payload"]["required"],
    }
    if not Draft202012Validator(payload_schema).is_valid(model):
        raise ValueError("The model needs registry fields for features, rules, actions, scenarios, and a proposed key.")
    features = by_id(model["features"], "feature_id")
    actions = by_id(model["actions"], "action_id")
    scenarios = by_id(model["scenarios"], "scenario_id")
    if not features or not actions or not model["decision_rules"] or model["baseline_scenario_id"] not in scenarios:
        raise ValueError("Supply features, actions, rules, and an existing baseline scenario.")
    baseline = scenarios[model["baseline_scenario_id"]]["fact_values"]
    for feature_id, feature in features.items():
        if not feature["allowed_values"]:
            raise ValueError(f"Feature {feature_id} needs permitted generation values.")
        if feature["disclosure_stage_id"] not in {*stage_ids, "shared"}:
            raise ValueError(f"Feature {feature_id} has an unknown disclosure stage.")
        if set(feature.get("fact_ids", [])) - fact_ids:
            raise ValueError(f"Feature {feature_id} references an unavailable fact.")
    constraints = by_id(model["feasibility_constraints"], "constraint_id")
    rules = by_id(model["decision_rules"], "rule_id")
    for rule in [*constraints.values(), *rules.values()]:
        _validate_predicate(rule["condition"], features, scenarios)
        if "scope_condition" in rule:
            _validate_predicate(rule["scope_condition"], features, scenarios)
        if set(rule["evidence_claim_ids"]) - claim_ids:
            raise ValueError("A rule or constraint references an unavailable clinical claim.")
        if "action_ids" in rule and (not rule["action_ids"] or set(rule["action_ids"]) - actions.keys()):
            raise ValueError("A decision rule references an unavailable action.")
    for scenario in scenarios.values():
        for key, value in scenario["fact_values"].items():
            if key not in features or (value is not None and not includes(features[key]["allowed_values"], value)):
                raise ValueError(f"Scenario {scenario['scenario_id']} contains an unknown feature or an out-of-range value.")
        if any(f["variation"] == "fixed" and scenario["fact_values"].get(k) != baseline.get(k) for k, f in features.items()):
            raise ValueError("Scenarios in a model must preserve fixed features.")
        if set(scenario["keyed_action_ids"]) - actions.keys() or set(scenario["evidence_claim_ids"]) - claim_ids:
            raise ValueError("A scenario references an unavailable action or clinical claim.")
    proposed = model["proposed_key"]
    if set(proposed["action_ids"]) - actions.keys() or set(proposed["source_claim_ids"]) - claim_ids:
        raise ValueError("The proposed key references an unavailable action or clinical claim.")


def visible_feature_ids(model, case, stage_id):
    stages = [s["id"] for s in case["stages"]]
    disclosed = {"shared", *stages[:stages.index(stage_id) + 1]}
    facts = {f["id"] for f in core.visible_facts(case, stage_id)}
    return {f["feature_id"] for f in model["features"]
            if f["disclosure_stage_id"] in disclosed and set(f.get("fact_ids", [])) <= facts}


def visible_assignment(model, scenario, case, stage_id, *, live=False):
    feature_ids = visible_feature_ids(model, case, stage_id)
    visible = {f["id"]: f for f in core.visible_facts(case, stage_id)}
    values = {}
    problems = []
    for feature in model["features"]:
        fid = feature["feature_id"]
        links = feature.get("fact_ids", [])
        if fid not in feature_ids:
            continue
        value = scenario["fact_values"].get(fid)
        if live and links:
            observed = [value_of(visible[link], feature) for link in links]
            if any(v != observed[0] for v in observed[1:]):
                problems.append(f"Linked facts for {fid} disagree.")
                continue
            value = observed[0]
            if feature.get("unit") and any(visible[link].get("unit") not in (None, "", feature["unit"]) for link in links):
                problems.append(f"Check the evidence scope and units for {fid}.")
                continue
            if feature["variation"] == "fixed" and value is not None and value != scenario["fact_values"].get(fid):
                problems.append(f"{fid} changed outside the fixed decision scope.")
        if value is not None and not includes(feature["allowed_values"], value):
            problems.append(f"{fid} has a value outside the reviewed range or categories.")
        elif value is not None:
            values[fid] = value
    return values, problems


def evaluate(model, scenario, visible_values, response_format, problems=()):
    """Calculate the key from the rules. Scenario annotations are proposal history."""
    assignment = visible_values | {"scenario": scenario["scenario_id"]}
    # Feasibility is about the underlying combination even if a report is still hidden.
    underlying = scenario["fact_values"] | assignment
    infeasible = [c["rationale"] for c in model["feasibility_constraints"]
                  if core.evaluate_decision_rule(c["condition"], underlying) is False]
    unknown_feasibility = [c for c in model["feasibility_constraints"]
                           if core.evaluate_decision_rule(c["condition"], underlying) is None]
    triggered, unresolved, inactive = [], [], []
    missing = set()
    for rule in model["decision_rules"]:
        if rule["condition"] and isinstance(rule["condition"], dict) and rule["condition"].get("scenario") not in (None, "", scenario["scenario_id"]):
            inactive.append(rule)
            continue
        predicates = [rule.get("scope_condition", {}), rule["condition"]]
        judgment = core.evaluate_decision_rule(predicates, assignment)
        if judgment is None:
            missing.update(core._rule_features(predicates) - assignment.keys())
        {True: triggered, False: inactive, None: unresolved}[judgment].append(rule)

    best = triggered
    possible = unresolved
    if response_format == "SBA" and triggered:
        rank = max(r.get("priority", 0) for r in triggered)
        best = [r for r in triggered if r.get("priority", 0) == rank]
        possible = [r for r in unresolved if r.get("priority", 0) >= rank]
    keys = {a for r in best for a in r["action_ids"]}
    possible_keys = {a for r in possible for a in r["action_ids"]}
    higher_rule_can_displace = response_format == "SBA" and bool(best) and any(
        r.get("priority", 0) > best[0].get("priority", 0) and set(r["action_ids"]) != keys
        for r in possible
    )
    status = "key_resolved"
    reasons = list(problems)
    if infeasible:
        status = "infeasible"
        reasons += infeasible
    elif problems or unknown_feasibility or not keys or possible_keys - keys or higher_rule_can_displace:
        status = "undetermined"
        if unknown_feasibility:
            for constraint in unknown_feasibility:
                missing.update(core._rule_features(constraint["condition"]) - {k for k, v in underlying.items() if v is not None})
            reasons.append("Information is missing for a feasibility constraint.")
        if not keys:
            reasons.append("No applicable rule establishes a key within the reviewed evidence scope.")
        if possible_keys - keys or higher_rule_can_displace:
            reasons.append("An unresolved rule could change the key or the keyed set.")
    elif response_format == "SBA" and len(keys) > 1:
        status = "ambiguous"
        reasons.append("Multiple actions satisfy the best applicable rules.")
    return {
        "scenario_id": scenario["scenario_id"], "outcome": status,
        "keyed_action_ids": sorted(keys) if status in RESOLVED else [],
        "competing_action_ids": sorted(keys | possible_keys) if status not in RESOLVED else [],
        "triggered_rule_ids": [r["rule_id"] for r in triggered],
        "unresolved_rule_ids": [r["rule_id"] for r in unresolved],
        "inactive_rule_ids": [r["rule_id"] for r in inactive],
        "superseded_rule_ids": [r["rule_id"] for r in triggered if r not in best],
        "missing_features": sorted(missing), "reasons": reasons, "visible_values": visible_values,
        "evidence_claim_ids": sorted({c for r in best for c in r["evidence_claim_ids"]}),
        "unsupported_rule_ids": [r["rule_id"] for r in best if not r["evidence_claim_ids"]],
    }


def scenario_rows(model, case, item, *, stage_id=None):
    stage_id = stage_id or item["stage_id"]
    scenarios = by_id(model["scenarios"], "scenario_id")
    baseline = scenarios[model["baseline_scenario_id"]]["fact_values"]
    rows = {}
    for sid, scenario in scenarios.items():
        values, problems = visible_assignment(model, scenario, case, stage_id)
        row = evaluate(model, scenario, values, item["format"], problems)
        row["changes"] = {k: v for k, v in scenario["fact_values"].items() if v != baseline.get(k)}
        rows[sid] = row
    reference = rows[model["baseline_scenario_id"]]["keyed_action_ids"]
    for sid, row in rows.items():
        if sid != model["baseline_scenario_id"] and reference and row["outcome"] in RESOLVED:
            row["outcome"] = "key_retained" if row["keyed_action_ids"] == reference else "key_changed"
    return rows


def current_scenario(model, case, item, selected):
    scenario = by_id(model["scenarios"], "scenario_id")[selected]
    values, problems = visible_assignment(model, scenario, case, item["stage_id"], live=True)
    underlying = scenario["fact_values"].copy()
    # An explicitly unknown observation replaces a scenario default for active feasibility.
    visible_ids = visible_feature_ids(model, case, item["stage_id"])
    for feature in model["features"]:
        fid, links = feature["feature_id"], feature.get("fact_ids", [])
        if links and fid in visible_ids and fid not in values:
            underlying.pop(fid, None)
    return evaluate(model, scenario | {"fact_values": underlying}, values, item["format"], problems)


def _boundary_values(feature, model):
    allowed = feature["allowed_values"]
    numeric = sorted(v for v in allowed if type(v) in (int, float))
    candidates = []
    def scan(rule):
        if isinstance(rule, list):
            for part in rule:
                scan(part)
        elif rule:
            for key in ("any", "all", "not"):
                if key in rule:
                    scan(rule[key])
                    return
            threshold = rule.get("value")
            if (rule.get("feature") == feature["feature_id"] and rule.get("operator") in {"gt", "gte", "lt", "lte"}
                    and type(threshold) in (int, float)):
                below = [v for v in numeric if v < threshold]
                above = [v for v in numeric if v > threshold]
                candidates.extend(below[-1:] + [v for v in numeric if v == threshold] + above[:1])
    for rule in model["decision_rules"]:
        scan([rule["condition"], rule.get("scope_condition", {})])
    ordered = []
    for value in candidates + allowed:
        if not includes(ordered, value):
            ordered.append(value)
    return ordered


def generate_scenarios(model, limit=200):
    """Enumerate bounded permitted values, including coupled changes and reviewed boundary values."""
    baseline = by_id(model["scenarios"], "scenario_id")[model["baseline_scenario_id"]]
    variable = [f for f in model["features"] if f["variation"] == "variable"]
    if not variable:
        return []
    existing = {json.dumps(s["fact_values"], sort_keys=True) for s in model["scenarios"]}
    rows = []
    ids = [f["feature_id"] for f in variable]
    choices = [_boundary_values(feature, model) for feature in variable]
    one_change = ({fid: v} for fid, values in zip(ids, choices) for v in values)
    coupled = (dict(zip(ids, values)) for values in product(*choices))
    for change in islice(chain(one_change, coupled), limit):
        values = baseline["fact_values"] | change
        signature = json.dumps(values, sort_keys=True)
        if signature not in existing:
            existing.add(signature)
            rows.append({"scenario_id": core.uid("scenario"), "fact_values": values,
                         "keyed_action_ids": [], "outcome": "undetermined", "evidence_claim_ids": []})
    return rows


def option_pool(model, selected, proposed, count, *, case=None, item=None):
    """Recheck candidate judgments and count distinct choices without overriding an SME key."""
    keys = set(selected["keyed_action_ids"])
    actions = by_id(model["actions"], "action_id")
    keyed_types = {actions[k]["option_type"].strip().casefold() for k in keys}
    inapplicable = set(selected["inactive_rule_ids"] + selected["superseded_rule_ids"])
    rules = by_id(model["decision_rules"], "rule_id")
    used_actions, used_text, included, rows = set(), set(), [], []
    for candidate in proposed:
        action = candidate["action_id"]
        text = " ".join(candidate["text"].casefold().split()).strip(" .?!")
        verdict, reason = "review", ""
        if not text:
            reason = "Option wording is missing."
        elif not keys:
            reason = "Resolve the scenario key first."
        elif action not in actions:
            reason = "Unmodeled action. Review and model it before inclusion."
        elif action in used_actions or text in used_text:
            reason = "Duplicate action or option wording."
        elif action in keys and candidate["eligibility"] != "keyed":
            reason = "This action is now keyed or equally defensible. Reassess its judgment."
        elif action in keys:
            verdict = "keyed"
        elif candidate["eligibility"] == "eligible" and candidate.get("plausibility_rationale") and candidate.get("clinical_rationale"):
            if actions[action]["option_type"].strip().casefold() not in keyed_types:
                reason = "Action type differs from the key."
            elif not any(action in rules[r]["action_ids"] for r in set(candidate["violated_rule_ids"]) & inapplicable):
                reason = "Identify an inactive action rule that excludes the alternative in this case."
            elif not candidate["clinical_rationale"].strip():
                reason = "Explain why this action is inferior or incorrect."
            elif case is not None and item is not None and set(candidate["attractive_fact_ids"]) - {f["id"] for f in core.visible_facts(case, item["stage_id"])}:
                reason = "An attractive finding is unavailable in this stage."
            else:
                verdict = "eligible"
        else:
            reason = "Review plausibility and the clinical exclusion before inclusion."
        if verdict in {"keyed", "eligible"}:
            used_actions.add(action)
            used_text.add(text)
            included.append(candidate | {"eligibility": verdict})
        rows.append(candidate | {"local_judgment": verdict, "review_reason": reason})
    required = [o for o in included if o["eligibility"] == "keyed" or o["role"] == "essential"]
    optional = sorted((o for o in included if o not in required), key=lambda o: o["role"] != "conditional")
    return {"rows": rows, "shortage": max(0, count - len(included)),
            "missing_keys": sorted(keys - {o["action_id"] for o in included if o["eligibility"] == "keyed"}),
            "too_many_required": len(required) > count,
            "selected": (required + optional)[:count]}
