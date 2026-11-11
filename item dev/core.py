"""Inspectable, provider-independent domain logic for family medicine item authoring.

This module performs structural checks. It does not certify clinical correctness or
psychometric validity. All data are JSON-compatible and clinical rules are authored.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import json
import math
import random
import re
from string import Template
from uuid import uuid4

SCHEMA_VERSION = 1
MODES = {"standalone": "Standalone MCQ", "unfolding": "Unfolding Case", "appraisal": "Clinical Appraisal MCQ"}
MODE_LABELS = MODES
REASONING_PHASES = ["Identify & Interpret", "Form & Prioritize", "Decide", "Act", "Reassess & Update"]
REASONING_GUIDANCE = {
    REASONING_PHASES[0]: "Recognize meaningful findings and interpret them in this patient's context.",
    REASONING_PHASES[1]: "Compare hypotheses and prioritize problems by likelihood, urgency, and patient concerns.",
    REASONING_PHASES[2]: "Choose an appropriate investigation, treatment, referral, preventive, or follow-up option.",
    REASONING_PHASES[3]: "Implement, explain, or coordinate the chosen plan; an MCQ samples judgment, not actual performance.",
    REASONING_PHASES[4]: "Use new findings or outcomes to update the working diagnosis, priorities, or plan.",
}
APPRAISAL_DIMENSIONS = ["Clinical safety and triage", "Patient-specific relevance", "Clinical evidence and interpretation", "Diagnostic reasoning and prioritization", "Diagnostic testing appropriateness", "Management and referral appropriateness"]
APPRAISAL_GUIDANCE = {
    APPRAISAL_DIMENSIONS[0]: "Does the recommendation recognize urgency, immediate risk, and appropriate timing?",
    APPRAISAL_DIMENSIONS[1]: "Does it incorporate the actual history, findings, conditions, medications, and priorities?",
    APPRAISAL_DIMENSIONS[2]: "Are its claims supported by the available findings and appropriate evidence?",
    APPRAISAL_DIMENSIONS[3]: "Does it prioritize the working diagnosis and consequential alternatives appropriately?",
    APPRAISAL_DIMENSIONS[4]: "Are investigations necessary, informative, proportionate, and appropriately sequenced?",
    APPRAISAL_DIMENSIONS[5]: "Are treatment, monitoring, follow-up, referral, and care setting suitable?",
}
FRAMEWORK_CROSSWALK = {
    APPRAISAL_DIMENSIONS[0]: {"IMPACT": ["Timeliness", "Comprehensiveness"], "RACCCA": ["Appropriateness", "Completeness"]},
    APPRAISAL_DIMENSIONS[1]: {"IMPACT": ["Integration", "Applicability"], "RACCCA": ["Relevance", "Accuracy"]},
    APPRAISAL_DIMENSIONS[2]: {"IMPACT": ["Mastery", "Precision"], "RACCCA": ["Accuracy", "Completeness"]},
    APPRAISAL_DIMENSIONS[3]: {"IMPACT": ["Mastery", "Comprehensiveness"], "RACCCA": ["Accuracy", "Completeness"]},
    APPRAISAL_DIMENSIONS[4]: {"IMPACT": ["Precision", "Applicability", "Timeliness"], "RACCCA": ["Relevance", "Appropriateness"]},
    APPRAISAL_DIMENSIONS[5]: {"IMPACT": ["Applicability", "Comprehensiveness", "Timeliness"], "RACCCA": ["Appropriateness", "Completeness", "Accuracy"]},
}
MISCONCEPTIONS = {
    "Premature closure": "Stop considering alternatives before sufficient evidence supports the decision. Identify which distinguishing finding was overlooked.",
    "Failure to update": "Retain an earlier judgment despite new evidence or a clinical course that should change it. Record exactly what changed.",
    "Evidence misinterpretation": "Misread or overstate the significance of a finding or result. Separate absent evidence from evidence of absence.",
    "Overtesting": "Request an investigation that does not improve the current decision or is disproportionate to its benefit.",
    "Undertesting": "Omit information needed to distinguish consequential alternatives or support a defensible action.",
    "Unsupported treatment": "Transfer a treatment from a different condition or context without the supporting features.",
    "Patient context omitted": "Ignore a relevant medication, comorbidity, preference, contraindication, or practical constraint.",
    "Failure to escalate": "Overlook findings that warrant more urgent assessment or a different level of care.",
    "Unnecessary escalation": "Choose an unjustified higher intensity of care despite evidence supporting a proportionate plan.",
    "Poor follow-up planning": "Omit an appropriate reassessment interval, return precautions, or plan for an ineffective intervention.",
    "Unsupported criticism": "Criticize an appropriate recommendation or infer an error from information the case does not establish.",
    "Knowledge gap": "The response may reflect missing factual knowledge. A selected option alone cannot establish the cause.",
}
BLUEPRINT = {
    "Acute and urgent care": ["Respiratory symptoms", "Abdominal symptoms", "Triage and disposition"],
    "Chronic and longitudinal care": ["Cardiometabolic health", "Medication review", "Multimorbidity"],
    "Prevention and health promotion": ["Screening", "Counseling", "Immunization"],
    "Care across the life course": ["Child and adolescent health", "Adult health", "Older adult care"],
    "Professional and patient-centered practice": ["Shared decisions", "Communication", "Clinical recommendation appraisal"],
}
OPTION_TYPES = ["", "Diagnosis", "Investigation", "Management", "Disposition", "Follow-up action", "Appraisal judgment", "Other"]
STAGE_PRESETS = {
    3: ["Assessment", "Management", "Follow-up"],
    4: ["Initial assessment", "Interpretation", "Management", "Follow-up"],
}
REVIEW_STAGES = ["Model development review", "Independent model review", "Item/case review", "Pilot preparation and field testing", "Release decision"]
REVIEW_STATUSES = ["Draft", "Model review", "Item/case review", "Ready for pilot", "Field-tested", "Approved for use"]
SCORING_DESCRIPTIONS = {
    "exact": "1 point only when the selected option set exactly matches the keyed set; otherwise 0. No credit for unselected distractors.",
    "partial": "(correct options selected minus incorrect options selected) divided by the number of correct options, clipped to 0–1. Unselected options earn no credit. Selecting no options earns 0.",
}


def uid(prefix="id"):
    return f"{prefix}_{uuid4().hex[:12]}"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_option(text="", correct=False, rationale="", misconception=""):
    return {"id": uid("opt"), "text": text, "correct": correct, "rationale": rationale,
            "misconception": misconception, "misconception_status": "author hypothesis",
            "role": "conditional", "eligible": True, "conditions": [], "supports": "", "excludes": "", "option_type": ""}


def new_stage(index=1):
    return {"id": uid("stage"), "title": f"Stage {index}", "new_information": "", "transition": "", "elapsed": ""}


def new_item(stage_id):
    return {"id": uid("item"), "stage_id": stage_id, "title": "New clinical decision", "phase": REASONING_PHASES[0],
            "secondary_phases": [], "format": "SBA", "stem": "", "lead_in": "", "options": [new_option() for _ in range(4)],
            "scoring": "exact", "select_count": None, "option_count": None, "selection_instruction": "Select the single best answer.",
            "recommendation": "", "dimension": APPRAISAL_DIMENSIONS[0], "claims": [], "ratings": {},
            "required_facts": [], "sources": [], "review_status": "Draft", "provenance": "human-authored"}


def new_case(mode="standalone", example=False):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    stages = [new_stage(i + 1) for i in range(4 if mode == "unfolding" else 1)]
    case = {"schema_version": SCHEMA_VERSION, "id": uid("case"), "mode": mode,
            "title": f"Untitled {MODES[mode].lower()}", "idea": "", "blueprint": {"name": "Illustrative family medicine blueprint", "version": "1", "domain": "", "subdomain": "", "objective": ""},
            "patient": "", "facts": [], "stages": stages, "items": [new_item(s["id"]) for s in stages],
            "features": [], "constraints": [], "sources": [], "reviews": [], "archived_items": [], "archived_stages": [],
            "version": 1, "updated_at": timestamp(), "review_status": "Draft", "scenario": "", "provenance": "human-authored",
            "author": "", "clinical_review_note": "Clinical correctness requires SME review; no learner response data are present."}
    return _example(case) if example else case


def _example(case):
    """Educational prototype using the previously verified NICE acute cough guidance."""
    case.update(title=f"Educational example · acute cough · {MODES[case['mode']]}",
                idea="Assess proportionate management and follow-up for an uncomplicated acute respiratory presentation.",
                patient="An otherwise healthy adult presents with a short history of cough associated with an uncomplicated upper respiratory tract infection. The patient is not systemically very unwell, has no features suggesting a serious illness, is not at higher risk of complications, and has no underlying airways disease.",
                scenario="uncomplicated acute cough", provenance="educational fixture — SME review required")
    case["blueprint"].update(domain="Acute and urgent care", subdomain="Respiratory symptoms", objective="Use the stated clinical context to choose proportionate management and appropriate return advice.")
    case["sources"] = [{"title": "NICE NG120 — Cough (acute): antimicrobial prescribing, recommendations", "url": "https://www.nice.org.uk/guidance/ng120/chapter/recommendations", "note": "Teaching prototype grounded in recommendations 1.1.4–1.1.9; not operationally reviewed."}]
    decisions = [
        ("Initial assessment", "Identify & Interpret", "Which information is most relevant before applying an uncomplicated acute cough management pathway?", [
            ("Whether serious illness, systemic unwellness, complication risk, or underlying airways disease is present", True, "These contextual findings affect the applicability of the stated pathway.", ""),
            ("The patient's preferred brand of cough remedy alone", False, "A product preference does not replace assessment of relevant clinical context.", "Patient context omitted"),
            ("Whether an antibiotic was prescribed for a previous unrelated illness alone", False, "Previous prescribing alone does not establish a current indication.", "Evidence misinterpretation"),
            ("Whether every patient with a cough received the same treatment", False, "The decision must use this patient's features rather than a uniform treatment assumption.", "Patient context omitted")]),
        ("Interpretation", "Form & Prioritize", "Which interpretation is best supported by the clinical information explicitly provided?", [
            ("The stated uncomplicated pathway can be considered, with continued attention to change or deterioration", True, "The vignette explicitly supplies the features required for the teaching pathway, while future change still matters.", ""),
            ("The presence of cough alone proves a bacterial cause", False, "Cough alone does not establish a bacterial cause or an antibiotic indication.", "Evidence misinterpretation"),
            ("An unreported examination result can be assumed normal", False, "Unknown findings must not be replaced by invented normal results.", "Evidence misinterpretation"),
            ("No subsequent reassessment could ever be necessary", False, "A current uncomplicated presentation does not rule out later deterioration.", "Failure to update")]),
        ("Management", "Decide", "Which is the most appropriate initial management approach for the uncomplicated presentation described?", [
            ("Self-care advice, explanation of the expected course, and clear return precautions", True, "NICE recommends self-care and advice; routine antibiotics are not indicated in this stated context.", ""),
            ("Immediate antibiotics solely because a cough is present", False, "Routine antibiotic treatment is not indicated for the stated uncomplicated upper respiratory infection.", "Unsupported treatment"),
            ("Bronchodilator treatment despite no underlying airways disease", False, "NICE advises against bronchodilators for this context without underlying airways disease.", "Unsupported treatment"),
            ("Corticosteroid treatment despite no underlying airways disease", False, "NICE advises against corticosteroids for this context without underlying airways disease.", "Unsupported treatment")]),
        ("Follow-up", "Reassess & Update", "Which follow-up advice is supported by the guidance?", [
            ("Seek medical help if symptoms worsen rapidly or significantly, or the patient becomes systemically very unwell", True, "These are explicit return precautions in NICE NG120.", ""),
            ("Seek medical help if symptoms do not improve after 3–4 weeks", True, "NICE includes lack of improvement after 3–4 weeks as a reason to seek medical help.", ""),
            ("Ignore significant deterioration because the original presentation was uncomplicated", False, "New clinical information warrants reassessment.", "Failure to update"),
            ("Assume that all new symptoms have already been assessed", False, "New symptoms are new information, not automatically covered by the earlier assessment.", "Premature closure")]),
    ]
    chosen = decisions if case["mode"] == "unfolding" else [decisions[2]]
    for i, (stage, item, decision) in enumerate(zip(case["stages"], case["items"], chosen)):
        title, phase, lead, options = decision
        stage["title"] = title
        if i:
            stage["new_information"] = "The case continues as explicitly described. No unstated treatment or examination result is assumed."
        if i < len(chosen) - 1:
            # A transition leads out of its stage, so the final stage has none.
            stage["transition"] = "The next item assesses a different decision using the information then shown."
        item.update(title=title, phase=phase, lead_in=lead,
                    options=[new_option(*option) for option in options], sources=[case["sources"][0]["url"]])
        if title == "Follow-up":
            item.update(format="MS", selection_instruction="Select all applicable options.")
        item["provenance"] = "educational fixture — SME review required"
    if case["mode"] == "appraisal":
        item = case["items"][0]
        item.update(title="Appraise a preliminary plan", phase="Decide", dimension="Management and referral appropriateness", format="MS", selection_instruction="Select all supported criticisms.",
                    recommendation="Start an antibiotic, a bronchodilator, and an oral corticosteroid solely because the patient has a cough. No follow-up advice is needed.",
                    lead_in="Which criticisms of the preliminary plan are supported by the displayed case and NICE guidance?",
                    options=[new_option("Routine antibiotic treatment is not supported in the uncomplicated context described.", True, "NG120 recommendation 1.1.8 advises against antibiotics in this context."),
                             new_option("Bronchodilator and corticosteroid treatment lack the stated underlying airways disease indication.", True, "NG120 recommendation 1.1.6 advises against these treatments in this context without an underlying airways disease."),
                             new_option("Follow-up and return advice can be omitted because deterioration is impossible.", False, "The case does not establish that deterioration is impossible; return advice remains relevant.", "Failure to update"),
                             new_option("The plan is wrong because an unreported allergy is definitely present.", False, "An unreported allergy is unknown and cannot be invented as a criticism.", "Unsupported criticism")],
                    claims=[{"claim": "Routine treatment solely because of cough", "judgment": "Contradicted", "evidence": "The vignette explicitly states the uncomplicated context without underlying airways disease.", "dimension": "Management and referral appropriateness", "rationale": "The treatments conflict with the stated NG120 pathway.", "source": case["sources"][0]["url"]}])
    case["facts"] = [{"id": uid("fact"), "name": "Underlying airways disease", "value": "absent", "disclosed_at": case["stages"][0]["id"], "source": "Educational vignette"}]
    return case


def touch_case(case, reason="Content changed"):
    """Invalidate current approval, retaining historical review evidence."""
    case["version"] = int(case.get("version", 1)) + 1
    case["updated_at"] = timestamp()
    case["review_status"] = "Draft"
    for item in case.get("items", []):
        item["review_status"] = "Draft"
    for review in case.get("reviews", []):
        if review.get("decision") in {"Approved", "Approved for use", "Ready for pilot"} or review.get("status") in {"Approved", "Approved for use", "Ready for pilot"}:
            review["requires_reassessment"] = True
            review["valid"] = False
            review["reassessment_reason"] = reason
    return case


def _positive_count(n, name="count"):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} must be a positive integer")


def resize_items(case, n):
    _positive_count(n, "Item count")
    current = len(case["items"])
    if n == current:
        return case
    if n < current:
        case.setdefault("archived_items", [])[0:0] = deepcopy(case["items"][n:])
        case["items"] = case["items"][:n]
    else:
        archive = case.setdefault("archived_items", [])
        while len(case["items"]) < n:
            if archive:
                restored = archive.pop(0)
                if not any(i["id"] == restored["id"] for i in case["items"]):
                    case["items"].append(restored)
            else:
                case["items"].append(new_item(case["stages"][0]["id"]))
    return touch_case(case, "Item count changed")


def resize_stages(case, n):
    _positive_count(n, "Stage count")
    current = len(case["stages"])
    if n == current:
        return case
    if n < current:
        case.setdefault("archived_stages", [])[0:0] = deepcopy(case["stages"][n:])
        case["stages"] = case["stages"][:n]
        # References are deliberately retained and flagged for explicit author repair.
    else:
        archive = case.setdefault("archived_stages", [])
        while len(case["stages"]) < n:
            case["stages"].append(archive.pop(0) if archive else new_stage(len(case["stages"]) + 1))
    return touch_case(case, "Disclosure stages changed; dependent items and facts need review")


def apply_stage_preset(case, n):
    """Resize to an editable 3- or 4-stage scaffold; only default titles are renamed."""
    if n not in STAGE_PRESETS:
        raise ValueError("Stage presets are available for 3 or 4 stages")
    resize_stages(case, n)
    for index, (stage, title) in enumerate(zip(case["stages"], STAGE_PRESETS[n]), 1):
        if not stage.get("title", "").strip() or re.fullmatch(r"Stage \d+", stage["title"].strip()):
            stage["title"] = title
    return touch_case(case, f"Applied the {n}-stage preset; existing content was kept")


def duplicate_stage(case, stage_id):
    """Insert a copy of a stage after it. Items are not copied or moved."""
    index = next((i for i, s in enumerate(case["stages"]) if s["id"] == stage_id), None)
    if index is None:
        raise ValueError("The stage to duplicate does not exist")
    copy = deepcopy(case["stages"][index])
    copy["id"] = uid("stage")
    copy["title"] = (copy.get("title") or "Stage") + " · copy"
    case["stages"].insert(index + 1, copy)
    return touch_case(case, "Duplicated a disclosure stage")


def remove_stage(case, stage_id):
    """Archive one stage. Dependent items and facts keep their reference for explicit repair."""
    if len(case["stages"]) <= 1:
        raise ValueError("A case needs at least one disclosure stage")
    index = next((i for i, s in enumerate(case["stages"]) if s["id"] == stage_id), None)
    if index is None:
        raise ValueError("The stage to remove does not exist")
    case.setdefault("archived_stages", []).insert(0, case["stages"].pop(index))
    return touch_case(case, "Removed a disclosure stage; dependent items and facts need review")


def scoring_patterns(item):
    """Score representative response patterns so the author can inspect the policy."""
    options = item.get("options", [])
    keyed = [o["id"] for o in options if o.get("correct")]
    unkeyed = [o["id"] for o in options if not o.get("correct")]
    patterns = [("No options selected", [])]
    if item.get("format", "SBA") == "SBA":
        patterns += [("Keyed option", keyed[:1])] + ([("An unkeyed option", unkeyed[:1])] if unkeyed else [])
    else:
        patterns.append(("Exact keyed set", keyed))
        if len(keyed) > 1:
            patterns.append(("One keyed option only", keyed[:1]))
        if unkeyed:
            patterns.append(("Keyed set plus one unkeyed option", keyed + unkeyed[:1]))
            patterns.append(("One unkeyed option only", unkeyed[:1]))
        patterns.append(("All options selected", keyed + unkeyed))
    rows = []
    for label, selected in patterns:
        result = score_item(item, selected)
        rows.append({"Response pattern": label, "Keyed selected": result["correct_selected"],
                     "Unkeyed selected": result["incorrect_selected"], "Keyed missed": result["missed"],
                     "Score": result["score"]})
    return rows


def evidence_map(case):
    """Rows tracing visible information to the decision, key, anticipated errors and next event."""
    stages = case.get("stages", [])
    order = {s["id"]: i for i, s in enumerate(stages)}
    rows = []
    for number, item in enumerate(case.get("items", []), 1):
        index = order.get(item.get("stage_id"))
        stage = stages[index] if index is not None else None
        facts = visible_facts(case, item.get("stage_id")) if stage else []
        disclosed = [s.get("new_information", "").strip() for s in stages[:index + 1]] if stage else []
        visible = ["Shared vignette"] if case.get("patient", "").strip() else []
        visible += [f"{f.get('name', '')}: {f.get('value', '')}" for f in facts]
        visible += [text for text in disclosed if text]
        if stage is None:
            following = "Stage missing — reassign this item"
        elif stage.get("transition", "").strip():
            following = stage["transition"].strip()
        elif index + 1 < len(stages):
            following = stages[index + 1].get("new_information", "").strip() or "No transition or new information recorded"
        else:
            following = "End of case"
        errors = sorted({o["misconception"] for o in item.get("options", []) if not o.get("correct") and o.get("misconception")})
        phases = [item.get("phase", "")] + [p for p in item.get("secondary_phases", []) if p != item.get("phase")]
        rows.append({
            "Item": f"{number}. {item.get('title', '')}",
            "Stage": stage["title"] if stage else "Missing stage",
            "Visible information": " | ".join(visible) or "Nothing recorded",
            "Reasoning phase(s)": " + ".join(p for p in phases if p),
            "Expected judgment": item.get("lead_in", ""),
            "Format": item.get("format", "SBA"),
            "Keyed response(s)": " | ".join(o.get("text", "") for o in item.get("options", []) if o.get("correct")) or "No key",
            "Anticipated reasoning errors": "; ".join(errors) or "None recorded",
            "What happens next": following,
        })
    return rows


def combination_table(case, limit=200):
    """Enumerate authored feature combinations and whether the constraints permit them."""
    features = [f for f in case.get("features", []) if f.get("name") and f.get("values")]
    if not features:
        return {"rows": [], "total": 0, "permitted": 0, "truncated": False}
    total = math.prod(len(f["values"]) for f in features)
    if total > 50000:
        raise ValueError("Feature space exceeds 50,000 combinations; narrow authored ranges")
    rows, permitted = [], 0
    for combination in product(*(f["values"] for f in features)):
        assignment = {"scenario": case.get("scenario", "")}
        for feature, value in zip(features, combination):
            assignment[feature["name"]] = value
            assignment[feature.get("id", feature["name"])] = value
        allowed = all(evaluate_rule(rule, assignment) for rule in case.get("constraints", []))
        permitted += allowed
        if len(rows) < limit:
            rows.append({f["name"]: str(v) for f, v in zip(features, combination)} | {"Permitted": "Yes" if allowed else "No"})
    return {"rows": rows, "total": total, "permitted": permitted, "truncated": total > limit}


def split_import(payload):
    """Return the case payloads in a single-case, variant-set, or library export."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Import is not valid UTF-8 JSON") from exc
    if isinstance(payload, list):
        cases = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        cases = payload["cases"]
    elif isinstance(payload, dict) and isinstance(payload.get("cases"), dict):
        cases = list(payload["cases"].values())
    else:
        cases = [payload]
    if not cases:
        raise ValueError("The import contains no cases")
    if len(cases) > 200:
        raise ValueError("The import contains more than 200 cases; split it into smaller files")
    return cases


def score_item(item, selected_ids):
    options = item.get("options", [])
    all_ids = {o["id"] for o in options}
    selected = set(selected_ids)
    unknown = selected - all_ids
    if unknown:
        raise ValueError("Response contains an unknown option identifier")
    keyed = {o["id"] for o in options if o.get("correct")}
    if not keyed:
        raise ValueError("The item has no keyed answer")
    if item.get("format", "SBA") == "SBA":
        if len(keyed) != 1:
            raise ValueError("Single-best-answer items require exactly one key")
        if len(selected) > 1:
            raise ValueError("Single-best-answer response selects more than one option")
        value = float(selected == keyed)
        policy = "exact"
    else:
        policy = item.get("scoring", "exact")
        if policy == "exact":
            value = float(selected == keyed)
        elif policy == "partial":
            value = max(0.0, min(1.0, (len(selected & keyed) - len(selected - keyed)) / len(keyed)))
        else:
            raise ValueError("Scoring must be exact or partial")
    return {"score": value, "max_score": 1.0, "exact_match": selected == keyed,
            "selected": sorted(selected), "keyed": sorted(keyed), "correct_selected": len(selected & keyed),
            "incorrect_selected": len(selected - keyed), "missed": len(keyed - selected),
            "policy": policy, "description": SCORING_DESCRIPTIONS[policy]}


def visible_facts(case, stage_id):
    stage_order = {stage["id"]: i for i, stage in enumerate(case.get("stages", []))}
    if stage_id not in stage_order:
        return []
    return [fact for fact in case.get("facts", []) if fact.get("disclosed_at") in (None, "", "shared") or stage_order.get(fact.get("disclosed_at"), math.inf) <= stage_order[stage_id]]


def _issue(severity, code, path, message):
    return {"severity": severity, "code": code, "path": path, "message": message}


def validate_case(case):
    """Return explainable structural findings and separate SME review prompts."""
    try:
        case = normalize_import(case)
    except (TypeError, ValueError) as exc:
        return [_issue("error", "SCHEMA", "case", str(exc))]
    findings = []
    def add(level, code, path, message):
        findings.append(_issue(level, code, path, message))
    for field in ("title", "patient"):
        if not case.get(field, "").strip():
            add("warning", "MISSING_INPUT", field, f"Enter the {field} before submitting for review.")
    for field in ("domain", "subdomain", "objective"):
        if not case["blueprint"].get(field, "").strip():
            add("warning", "BLUEPRINT", f"blueprint.{field}", f"Specify the blueprint {field}; starter categories are illustrative.")
    if not case["sources"]:
        add("review", "SOURCE_REVIEW", "sources", "Add and verify clinical source evidence; structural checks do not establish clinical correctness.")
    stage_ids = [s["id"] for s in case["stages"]]
    if len(set(stage_ids)) != len(stage_ids):
        add("error", "DUPLICATE_ID", "stages", "Stage identifiers must be unique.")
    if not stage_ids:
        add("error", "NO_STAGE", "stages", "Create at least one disclosure stage.")
    fact_ids = [f.get("id", "") for f in case["facts"]]
    if len(set(fact_ids)) != len(fact_ids):
        add("error", "DUPLICATE_ID", "facts", "Fact identifiers must be unique.")
    for fidx, fact in enumerate(case["facts"]):
        if fact.get("disclosed_at") not in (None, "", "shared", *stage_ids):
            add("error", "FACT_STAGE", f"facts[{fidx}].disclosed_at", "This fact refers to a missing disclosure stage; restore it or explicitly reassign disclosure.")
    item_ids = [i["id"] for i in case["items"]]
    if len(set(item_ids)) != len(item_ids):
        add("error", "DUPLICATE_ID", "items", "Item identifiers must be unique.")
    if not item_ids:
        add("error", "NO_ITEM", "items", "Create at least one item.")
    all_option_ids = []
    for idx, item in enumerate(case["items"]):
        path = f"items[{idx}]"
        if item["stage_id"] not in stage_ids:
            add("error", "ITEM_STAGE", path + ".stage_id", "This item's disclosure stage is missing; restore or explicitly reassign it.")
        if not item["lead_in"].strip():
            add("warning", "LEAD_IN", path + ".lead_in", "Write one clear decision or appraisal task.")
        if item["phase"] not in REASONING_PHASES:
            add("warning", "REASONING_PHASE", path + ".phase", "Choose a primary phase from the proposed reasoning guide.")
        opts = item["options"]
        texts = [o["text"].strip().casefold() for o in opts if o["text"].strip()]
        if len(opts) < 2:
            add("error", "OPTION_COUNT", path + ".options", "At least two options are needed for an MCQ.")
        if len(texts) != len(set(texts)):
            add("error", "DUPLICATE_OPTION", path + ".options", "Options with identical text overlap; revise them.")
        keyed = [o for o in opts if o["correct"]]
        if item["format"] == "SBA" and len(keyed) != 1:
            add("error", "SBA_KEY", path + ".options", "Single-best-answer requires exactly one keyed option.")
        if item["format"] == "MS" and not keyed:
            add("error", "MS_KEY", path + ".options", "Multiple-selection requires a complete nonempty keyed option set.")
        if item["format"] == "MS" and item.get("select_count") is not None and item["select_count"] != len(keyed):
            add("error", "SELECT_COUNT", path + ".select_count", "The instructed selection count does not match the keyed option set.")
        if item["format"] == "MS" and keyed and len(keyed) == len(opts):
            add("warning", "MS_ALL_KEYED", path + ".options", "Every option is keyed; add a defensible unkeyed alternative or reconsider the task.")
        if item["format"] == "MS" and not item.get("selection_instruction", "").strip():
            add("warning", "SELECTION_INSTRUCTION", path + ".selection_instruction", "State whether all applicable options or a specified number must be selected.")
        for oidx, option in enumerate(opts):
            opath = f"{path}.options[{oidx}]"
            all_option_ids.append(option["id"])
            if not option["text"].strip():
                add("warning", "OPTION_TEXT", opath + ".text", "Enter the option text; do not pad with filler.")
            if option["text"].strip() and not option["rationale"].strip():
                add("review", "OPTION_RATIONALE", opath + ".rationale", "Explain this option using visible case evidence.")
            if option.get("correct") and not option.get("eligible", True):
                add("error", "INELIGIBLE_KEY", opath, "A keyed option is marked ineligible. Review the key and eligibility rule.")
            if not option["correct"] and option["text"].strip() and not option.get("misconception", "").strip():
                add("review", "MISCONCEPTION", opath + ".misconception", "Record a proposed reasoning error or explicitly note that its cause is unknown.")
            if option.get("conditions"):
                known = {f.get("id") for f in case["features"]} | {f.get("name") for f in case["features"]}
                for name in _rule_features(option["conditions"]):
                    if name not in known and name != "scenario":
                        add("error", "CONDITION_FEATURE", opath + ".conditions", f"Eligibility references unknown feature {name!r}.")
        visible_ids = {f["id"] for f in visible_facts(case, item["stage_id"])}
        for fact_id in item.get("required_facts", []):
            if fact_id not in fact_ids:
                add("error", "UNKNOWN_FACT", path + ".required_facts", f"Required fact {fact_id!r} is missing.")
            elif fact_id not in visible_ids:
                add("error", "FUTURE_FACT", path + ".required_facts", "The key depends on a fact disclosed only in a future stage.")
        if case["mode"] == "appraisal":
            if not item["recommendation"].strip():
                add("warning", "APPRAISAL_RECOMMENDATION", path + ".recommendation", "Preserve the exact preliminary recommendation being appraised.")
            if item["dimension"] not in APPRAISAL_DIMENSIONS:
                add("warning", "APPRAISAL_DIMENSION", path + ".dimension", "Choose a proposed appraisal dimension.")
            if not item["claims"]:
                add("review", "CLAIM_EVIDENCE", path + ".claims", "Annotate the relevant claim, evidence, and expected appraisal judgment.")
            for cidx, claim in enumerate(item["claims"]):
                if not claim.get("evidence", "").strip() or not claim.get("rationale", "").strip():
                    add("review", "CLAIM_EVIDENCE", f"{path}.claims[{cidx}]", "Supply evidence and rationale; insufficient information must not be silently treated as false.")
    if len(set(all_option_ids)) != len(all_option_ids):
        add("error", "DUPLICATE_ID", "items.options", "Option identifiers must be unique throughout a case.")
    _cross_item_findings(case, add)
    for idx, stage in enumerate(case["stages"]):
        if idx and not stage.get("new_information", "").strip() and not stage.get("transition", "").strip():
            add("review", "TRANSITION", f"stages[{idx}]", "Explain the next decision's purpose or new information; another item does not require a new event.")
    feature_names = {f.get("name") for f in case["features"]} | {f.get("id") for f in case["features"]}
    for cidx, rule in enumerate(case["constraints"]):
        for name in _rule_features(rule):
            if name not in feature_names and name != "scenario":
                add("error", "CONSTRAINT_FEATURE", f"constraints[{cidx}]", f"Rule references unknown feature {name!r}.")
    if not case["reviews"]:
        add("review", "EXPERT_REVIEW", "reviews", "Complete model and item review. No clinical or psychometric approval is inferred from these checks.")
    if case["mode"] == "unfolding":
        add("review", "SEQUENCE_REVIEW", "stages", "Review chronology, treatment exposure, answer cueing, and whether the learner should update judgments. These require SME judgment.")
    return findings


_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def _norm_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _cross_item_findings(case, add):
    """Deterministic prompts about cueing, repetition and placeholders; SME judgment decides."""
    order = {s["id"]: i for i, s in enumerate(case["stages"])}
    items = case["items"]
    seen = {}
    for idx, item in enumerate(items):
        for option in item["options"]:
            text = _norm_text(option["text"])
            if text:
                seen.setdefault(text, set()).add(idx)
    for text, where in seen.items():
        if len(where) > 1:
            first = min(where)
            numbers = ", ".join(str(i + 1) for i in sorted(where))
            add("review", "REPEATED_OPTION", f"items[{first}].options", f"The same option text appears in items {numbers}. Check for cueing and whether its role should change with the evidence.")
    leads = {}
    for idx, item in enumerate(items):
        lead = _norm_text(item["lead_in"])
        if lead:
            leads.setdefault((item["stage_id"], lead), []).append(idx)
    for (_, _lead), where in leads.items():
        if len(where) > 1:
            add("review", "REPEATED_DECISION", f"items[{where[1]}].lead_in", f"Items {', '.join(str(i + 1) for i in where)} ask the same question from the same disclosed evidence. Give each item a distinct decision.")
    for aidx, source in enumerate(items):
        keys = [_norm_text(o["text"]) for o in source["options"] if o["correct"] and len(_norm_text(o["text"])) >= 12]
        if not keys or source["stage_id"] not in order:
            continue
        for bidx, other in enumerate(items):
            if bidx == aidx or order.get(other["stage_id"], -1) < order[source["stage_id"]]:
                continue
            visible = _norm_text(other["stem"] + " " + other["lead_in"])
            if any(key in visible for key in keys):
                add("review", "ANSWER_CUE", f"items[{bidx}].stem", f"Item {bidx + 1} wording contains the keyed answer to item {aidx + 1}. Confirm this disclosure is deliberate and cannot be seen before item {aidx + 1} is answered.")
    names = {f.get("name") for f in case["features"]}
    def scan(path, text):
        for name in sorted(set(_PLACEHOLDER.findall(text or ""))):
            if name not in names:
                add("warning", "PLACEHOLDER", path, f"Placeholder {{{{{name}}}}} does not match an authored feature name and will appear unreplaced.")
    scan("patient", case["patient"])
    for sidx, stage in enumerate(case["stages"]):
        scan(f"stages[{sidx}].new_information", stage.get("new_information", ""))
        scan(f"stages[{sidx}].transition", stage.get("transition", ""))
    for idx, item in enumerate(items):
        scan(f"items[{idx}].stem", item["stem"])
        scan(f"items[{idx}].lead_in", item["lead_in"])
        for oidx, option in enumerate(item["options"]):
            scan(f"items[{idx}].options[{oidx}].text", option["text"])


def _ensure(value, expected, path):
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        typename = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
        raise ValueError(f"{path}: expected {typename}")


def normalize_import(payload):
    """Validate every structure consumed by the UI; preserve additional JSON fields."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Import is not valid UTF-8 JSON") from exc
    _ensure(payload, dict, "case")
    if "case" in payload and isinstance(payload["case"], dict):
        payload = payload["case"]
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("case: all fields must contain finite JSON-compatible values") from exc
    mode = payload.get("mode")
    if mode not in MODES:
        raise ValueError("case.mode: expected standalone, unfolding, or appraisal")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("case.schema_version: unsupported schema version")
    result = deepcopy(payload)
    base = new_case(mode)
    for key, value in base.items():
        result.setdefault(key, deepcopy(value))
    for key in ("id", "title", "idea", "patient", "review_status", "scenario", "updated_at"):
        _ensure(result[key], str, f"case.{key}")
    _ensure(result["version"], int, "case.version")
    if result["version"] < 1:
        raise ValueError("case.version: must be positive")
    _ensure(result["blueprint"], dict, "case.blueprint")
    for key in ("domain", "subdomain", "objective"):
        result["blueprint"].setdefault(key, "")
        _ensure(result["blueprint"][key], str, f"case.blueprint.{key}")
    for key in ("stages", "items", "facts", "features", "constraints", "sources", "reviews", "archived_items", "archived_stages"):
        _ensure(result[key], list, f"case.{key}")
        for idx, element in enumerate(result[key]):
            _ensure(element, dict, f"case.{key}[{idx}]")
    for key in ("stages", "archived_stages"):
        for idx, stage in enumerate(result[key]):
            for field in ("id", "title", "new_information", "transition"):
                stage.setdefault(field, "" if field != "id" else uid("stage"))
                _ensure(stage[field], str, f"case.{key}[{idx}].{field}")
    for key in ("items", "archived_items"):
        for idx, item in enumerate(result[key]):
            path = f"case.{key}[{idx}]"
            defaults = new_item(result["stages"][0]["id"] if result["stages"] else "")
            # Never manufacture keyed content or option rows when importing an explicit list.
            for field, default in defaults.items():
                item.setdefault(field, deepcopy(default))
            for field in ("id", "stage_id", "title", "phase", "format", "stem", "lead_in", "scoring", "recommendation", "dimension", "selection_instruction"):
                _ensure(item[field], str, f"{path}.{field}")
            if item["format"] not in {"SBA", "MS"}:
                raise ValueError(f"{path}.format: expected SBA or MS")
            if item["scoring"] not in {"exact", "partial"}:
                raise ValueError(f"{path}.scoring: expected exact or partial")
            if item.get("select_count") is not None:
                _ensure(item["select_count"], int, f"{path}.select_count")
                if item["select_count"] < 1:
                    raise ValueError(f"{path}.select_count: must be positive or null")
            if item.get("option_count") is not None:
                _ensure(item["option_count"], int, f"{path}.option_count")
                if item["option_count"] < 2:
                    raise ValueError(f"{path}.option_count: must be at least 2 or null")
            for field in ("options", "claims", "required_facts", "sources", "secondary_phases"):
                _ensure(item[field], list, f"{path}.{field}")
            for field in ("required_facts", "secondary_phases"):
                for fidx, value in enumerate(item[field]):
                    _ensure(value, str, f"{path}.{field}[{fidx}]")
            _ensure(item["ratings"], dict, f"{path}.ratings")
            for ridx, (dimension, rating) in enumerate(item["ratings"].items()):
                _ensure(dimension, str, f"{path}.ratings key")
                _ensure(rating, (dict, int, float, str, type(None)), f"{path}.ratings.{dimension}")
            for cidx, claim in enumerate(item["claims"]):
                _ensure(claim, dict, f"{path}.claims[{cidx}]")
                for field in ("claim", "evidence", "rationale", "judgment", "dimension", "source"):
                    if field in claim:
                        _ensure(claim[field], str, f"{path}.claims[{cidx}].{field}")
            for oidx, option in enumerate(item["options"]):
                opath = f"{path}.options[{oidx}]"
                _ensure(option, dict, opath)
                for field, default in new_option().items():
                    option.setdefault(field, deepcopy(default))
                for field in ("id", "text", "rationale", "misconception", "role", "supports", "excludes", "option_type"):
                    _ensure(option[field], str, f"{opath}.{field}")
                _ensure(option["correct"], bool, f"{opath}.correct")
                _ensure(option["eligible"], bool, f"{opath}.eligible")
                _ensure(option["conditions"], (list, dict), f"{opath}.conditions")
                _validate_rule_shape(option["conditions"], f"{opath}.conditions")
    for idx, fact in enumerate(result["facts"]):
        fact.setdefault("id", uid("fact"))
        for field in ("id", "name"):
            fact.setdefault(field, "")
            _ensure(fact[field], str, f"case.facts[{idx}].{field}")
        if fact.get("disclosed_at") is not None:
            _ensure(fact["disclosed_at"], str, f"case.facts[{idx}].disclosed_at")
    for idx, feature in enumerate(result["features"]):
        feature.setdefault("id", uid("feature"))
        feature.setdefault("name", "")
        feature.setdefault("values", [])
        _ensure(feature["id"], str, f"case.features[{idx}].id")
        _ensure(feature["name"], str, f"case.features[{idx}].name")
        _ensure(feature["values"], list, f"case.features[{idx}].values")
        if feature.get("disclosed_at") is not None:
            _ensure(feature["disclosed_at"], str, f"case.features[{idx}].disclosed_at")
    for idx, rule in enumerate(result["constraints"]):
        _validate_rule_shape(rule, f"case.constraints[{idx}]")
    for idx, source in enumerate(result["sources"]):
        for key in ("title", "url", "note"):
            if key in source:
                _ensure(source[key], str, f"case.sources[{idx}].{key}")
    return result


_OPERATORS = {"equals", "not_equals", "in", "not_in", "gt", "gte", "lt", "lte", "exists", "not_exists"}


def _validate_rule_shape(rule, path="rule"):
    if isinstance(rule, list):
        for idx, part in enumerate(rule):
            _validate_rule_shape(part, f"{path}[{idx}]")
        return
    _ensure(rule, dict, path)
    if not rule:
        return
    for compound in ("all", "any"):
        if compound in rule:
            _ensure(rule[compound], list, f"{path}.{compound}")
            _validate_rule_shape(rule[compound], f"{path}.{compound}")
            return
    if "not" in rule:
        _validate_rule_shape(rule["not"], path + ".not")
        return
    _ensure(rule.get("feature"), str, path + ".feature")
    op = rule.get("operator", "equals")
    if op not in _OPERATORS:
        raise ValueError(f"{path}.operator: unsupported operator")
    if op in {"in", "not_in"}:
        _ensure(rule.get("value"), list, path + ".value")
    if "scenario" in rule:
        _ensure(rule["scenario"], str, path + ".scenario")


def _rule_features(rule):
    if isinstance(rule, list):
        return set().union(*(_rule_features(r) for r in rule)) if rule else set()
    if not isinstance(rule, dict) or not rule:
        return set()
    if "all" in rule:
        return _rule_features(rule["all"])
    if "any" in rule:
        return _rule_features(rule["any"])
    if "not" in rule:
        return _rule_features(rule["not"])
    return {rule.get("feature", "")}


def evaluate_rule(rule, assignment):
    if isinstance(rule, list):
        return all(evaluate_rule(part, assignment) for part in rule)
    if not rule:
        return True
    if rule.get("scenario") and rule["scenario"] != assignment.get("scenario"):
        return True  # This scenario-specific restriction is not applicable.
    if "all" in rule:
        return all(evaluate_rule(part, assignment) for part in rule["all"])
    if "any" in rule:
        return any(evaluate_rule(part, assignment) for part in rule["any"])
    if "not" in rule:
        return not evaluate_rule(rule["not"], assignment)
    feature, op, right = rule.get("feature"), rule.get("operator", "equals"), rule.get("value")
    if op == "exists":
        return feature in assignment
    if op == "not_exists":
        return feature not in assignment
    if feature not in assignment:
        return False
    left = assignment[feature]
    try:
        return {"equals": lambda: left == right, "not_equals": lambda: left != right,
                "in": lambda: left in right, "not_in": lambda: left not in right,
                "gt": lambda: left > right, "gte": lambda: left >= right,
                "lt": lambda: left < right, "lte": lambda: left <= right}[op]()
    except (TypeError, ValueError, KeyError):
        return False


def _render(value, assignment):
    if isinstance(value, str):
        text = value
        for key, replacement in assignment.items():
            text = text.replace("{{" + key + "}}", str(replacement))
        return text
    if isinstance(value, list):
        return [_render(part, assignment) for part in value]
    if isinstance(value, dict):
        return {key: (_render(part, assignment) if key not in {"conditions", "constraints", "features"} else deepcopy(part)) for key, part in value.items()}
    return value


def generate_variants(case, count=1, seed=1):
    """Assemble only authored combinations. Raises on shortage; never invents content.

    Features use ``name`` + ``values``. Text placeholders use ``{{feature name}}``.
    Constraints and option eligibility use structured, non-executable rules.
    The finite search is capped at 50,000 combinations to bound resource use.
    """
    _positive_count(count, "Variant count")
    case = normalize_import(case)
    features = case["features"]
    names = [f["name"] for f in features]
    if len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("Feature names must be nonempty and unique")
    values = [f["values"] for f in features]
    if any(not vals for vals in values):
        raise ValueError("Each feature needs at least one authored value")
    combinations = math.prod(len(vals) for vals in values)
    if combinations > 50000:
        raise ValueError("Feature space exceeds 50,000 combinations; narrow authored ranges before assembly")
    rng = random.Random(seed)
    possible = list(product(*values)) if features else [()]
    rng.shuffle(possible)
    variants, signatures = [], set()
    for combination in possible:
        assignment = {"scenario": case.get("scenario", "")}
        for f, value in zip(features, combination):
            assignment[f["name"]] = value
            assignment[f["id"]] = value
        if not all(evaluate_rule(rule, assignment) for rule in case["constraints"]):
            continue
        rendered = _render(deepcopy(case), assignment)
        invalid_options = False
        reasons = []
        for original, item in zip(case["items"], rendered["items"]):
            eligible = []
            for original_option, option in zip(original["options"], item["options"]):
                marked = original_option.get("eligible", True)
                allowed = marked and evaluate_rule(original_option.get("conditions", []), assignment)
                if original_option.get("correct") and not allowed:
                    invalid_options = True
                    break
                if allowed:
                    eligible.append(option)
                else:
                    reasons.append({"item": item["title"], "option": option["text"], "included": False,
                                    "reason": "Marked ineligible by the author" if not marked else "Eligibility conditions not met for this feature assignment"})
            if invalid_options:
                break
            if len(eligible) < 2 or not any(o["correct"] for o in eligible):
                invalid_options = True
                break
            required = item.get("option_count") or len(original["options"])
            essential = [o for o in eligible if o["correct"] or o.get("role") == "essential"]
            conditional = [o for o in eligible if not o["correct"] and o.get("role") == "conditional"]
            pool = [o for o in eligible if not o["correct"] and o.get("role") not in {"essential", "conditional"}]
            selected = essential + conditional
            if required < len(selected) or len(eligible) < required:
                invalid_options = True
                break
            sampled = rng.sample(pool, required - len(selected))
            selected += sampled
            chosen = {o["id"] for o in selected}
            for option in eligible:
                if option["correct"]:
                    why = "Keyed response; always included"
                elif option.get("role") == "essential":
                    why = "Essential contrast; always included when eligible"
                elif option.get("role") == "conditional":
                    why = "Conditional alternative; eligibility conditions met"
                elif option["id"] in chosen:
                    why = "Sampled from the eligible random pool"
                else:
                    why = "Eligible, but not drawn from the random pool"
                reasons.append({"item": item["title"], "option": option["text"], "included": option["id"] in chosen, "reason": why})
            item["options"] = selected
            if item["format"] == "SBA" and sum(o["correct"] for o in selected) != 1:
                invalid_options = True
                break
            if item["format"] == "MS" and item.get("select_count") is not None and item["select_count"] != sum(o["correct"] for o in selected):
                invalid_options = True
                break
        if invalid_options:
            continue
        # Do not label a duplicate text combination as new item content.
        signature = json.dumps({"patient": rendered["patient"], "stages": rendered["stages"], "items": rendered["items"]}, sort_keys=True)
        if signature in signatures:
            continue
        signatures.add(signature)
        rendered["id"] = uid("case")
        rendered["title"] = f"{case['title']} · variant {len(variants) + 1}"
        rendered["generated_from"] = {"case_id": case["id"], "version": case["version"], "seed": seed,
                                      "assignment": {f["name"]: assignment[f["name"]] for f in features},
                                      "generator": "deterministic assembly (no AI model)", "option_reasons": reasons}
        rendered["reviews"] = []
        rendered["review_status"] = "Draft"
        rendered["provenance"] = "deterministic assembly from authored elements"
        variants.append(rendered)
        if len(variants) == count:
            return variants
    raise ValueError(f"Only {len(variants)} distinct permissible variant(s) could be assembled; requested {count}. Revise constraints, placeholders, or eligible option pools. No filler was generated.")


def export_markdown(case, include_answers=True):
    case = normalize_import(case)
    lines = [f"# {case['title']}", "", f"Mode: {MODES[case['mode']]}", "", case["patient"], ""]
    if include_answers:
        bp = case["blueprint"]
        lines += [f"Blueprint: {bp.get('domain', '')} → {bp.get('subdomain', '')}", f"Objective: {bp.get('objective', '')}",
                  f"Case version: {case['version']} · Review state: {case['review_status']}", ""]
    for stage_idx, stage in enumerate(case["stages"]):
        lines += [f"## {stage['title']}", "", stage.get("new_information", ""), ""]
        for item in [i for i in case["items"] if i["stage_id"] == stage["id"]]:
            lines += [f"### {item['title']}", "", item["stem"], ""]
            if case["mode"] == "appraisal":
                lines += ["Proposed recommendation:", "", item["recommendation"], ""]
            lines += [item["lead_in"], "", item.get("selection_instruction", ""), ""]
            for idx, option in enumerate(item["options"]):
                label = chr(65 + idx) if idx < 26 else str(idx + 1)
                lines += [f"{label}. {option['text']}"]
            lines.append("")
            if include_answers:
                keys = [str(idx + 1) for idx, opt in enumerate(item["options"]) if opt["correct"]]
                lines += [f"Keyed option(s): {', '.join(keys)}", f"Reasoning phase: {item['phase']}", f"Scoring: {SCORING_DESCRIPTIONS[item.get('scoring', 'exact')]}", ""]
                for idx, option in enumerate(item["options"]):
                    lines += [f"- Option {idx + 1}: {option['rationale']} · Proposed misconception: {option.get('misconception') or 'not specified'}"]
                if case["mode"] == "appraisal":
                    lines += ["", f"Appraisal dimension: {item['dimension']}", "", "Claim review:", "", "```json", json.dumps(item["claims"], indent=2, ensure_ascii=False), "```", "", "Recommendation rubric ratings:", "", "```json", json.dumps(item["ratings"], indent=2, ensure_ascii=False), "```"]
                lines.append("")
        if stage.get("transition"):
            lines += [stage["transition"], ""]
    if include_answers:
        lines += ["## Sources", ""]
        for source in case["sources"]:
            lines += [f"- [{source.get('title') or source.get('url') or 'Source'}]({source.get('url', '')}) — {source.get('note', '')}"]
        lines += ["", "## Review record", "", "```json", json.dumps(case["reviews"], indent=2, ensure_ascii=False), "```", "", "## Structural and expert-review findings", ""]
        lines += [f"- {f['severity'].upper()} · {f['path']}: {f['message']}" for f in validate_case(case)]
        lines += ["", "Framework tags are proposed authoring categories; clinical and psychometric validity require separate evidence."]
    return "\n".join(lines).rstrip() + "\n"
