"""Target task proposal, validation, and SME sibling confirmation."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .evidence import DATA_NOTICE, check_span, item_payload
from .schemas import ALIGNMENTS, SCHEMA_VERSION, SME_ALIGNMENTS, TASK_SCHEMA, TASK_STATUSES

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
NEW_TARGET_ID = "NEW-TARGET"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def system_prompt(name):
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def build_task_payload(target, candidates):
    """``candidates`` is a list of (record, cosine). Only these items are sent."""
    return {
        "data_notice": DATA_NOTICE,
        "schema_version": SCHEMA_VERSION,
        "target_item_id": target["item_id"],
        "target": item_payload(target),
        "candidates": [item_payload(r, c) for r, c in candidates],
    }


def validate_task_output(output, payload):
    """Drop references to unknown items and unverifiable quotes; return (clean, issues)."""
    items = {payload["target"]["item_id"]: payload["target"]}
    items.update({c["item_id"]: c for c in payload["candidates"]})
    issues = []
    clean = deepcopy(output)
    if clean.get("target_item_id") != payload["target_item_id"]:
        issues.append("The response named a different target item; the requested target is used.")
        clean["target_item_id"] = payload["target_item_id"]
    task = clean["target_task"]
    known = [i for i in task.get("source_item_ids", []) if i in items]
    if len(known) != len(task.get("source_item_ids", [])):
        issues.append("Removed unknown item IDs from the task's source items.")
    task["source_item_ids"] = known
    spans = []
    for span in task.get("evidence_spans", []):
        problems = check_span(span, items)
        if problems:
            issues.extend(f"Task evidence rejected: {p}." for p in problems)
        else:
            spans.append(span)
    task["evidence_spans"] = spans
    if not spans:
        issues.append("No verifiable evidence supports the proposed task. Review it against the item text.")
    alignments = {}
    for row in clean.get("candidate_task_alignment", []):
        iid = row.get("item_id")
        if iid not in items or iid == payload["target_item_id"]:
            issues.append(f"Ignored an alignment for unknown or target item '{iid}'.")
            continue
        if row.get("proposed_alignment") not in ALIGNMENTS:
            row["proposed_alignment"] = "uncertain"
        alignments[iid] = row
    for cand in payload["candidates"]:
        if cand["item_id"] not in alignments:
            issues.append(f"No alignment proposal was returned for {cand['item_id']}.")
            alignments[cand["item_id"]] = {"item_id": cand["item_id"], "apparent_task": "", "proposed_alignment": "uncertain",
                                           "evidence": "No proposal returned.", "confidence_status": "uncertain"}
    clean["candidate_task_alignment"] = [alignments[c["item_id"]] for c in payload["candidates"]]
    return clean, issues


def propose_task(llm, target, candidates):
    payload = build_task_payload(target, candidates)
    raw, usage = llm.complete(system_prompt("target_task"), payload, TASK_SCHEMA, "target_task")
    clean, issues = validate_task_output(raw, payload)
    return {"raw": raw, "proposal": clean, "issues": issues, "usage": usage,
            "model": getattr(llm, "model", "unknown"), "created_at": now(),
            "candidate_ids": [c["item_id"] for c in payload["candidates"]]}


def new_review(target_id, candidate_ids):
    return {
        "target_id": target_id,
        "candidate_ids": list(candidate_ids),
        "runs": [],
        "task": None,              # the SME-editable task
        "task_status": "unreviewed",
        "decisions": {cid: "unreviewed" for cid in candidate_ids},
        "include_all_retrieved": False,
        "history": [],
    }


def attach_proposal(review, run, reviewer):
    """Record a proposal. An SME-reviewed task or decisions are never overwritten."""
    review["runs"].append(run)
    proposed = run["proposal"]["target_task"]
    if review["task_status"] == "unreviewed":
        review["task"] = {k: proposed[k] for k in ("name", "domain", "decision_or_action", "rationale")}
    _log(review, reviewer, "proposal_recorded", None, {"model": run["model"], "issues": len(run["issues"])})
    return review


def latest_alignment(review):
    if not review["runs"]:
        return {}
    return {r["item_id"]: r for r in review["runs"][-1]["proposal"]["candidate_task_alignment"]}


def set_task(review, reviewer, status, edits=None):
    if status not in TASK_STATUSES:
        raise ValueError(f"Unknown task status '{status}'.")
    before = deepcopy({"task": review["task"], "status": review["task_status"]})
    if edits:
        review["task"] = {**(review["task"] or {}), **edits}
        status = "edited" if status == "accepted" and before["task"] and review["task"] != before["task"] else status
    review["task_status"] = status
    _log(review, reviewer, "task_review", before, {"task": review["task"], "status": status})


def set_decision(review, reviewer, item_id, decision):
    if decision not in SME_ALIGNMENTS:
        raise ValueError(f"Unknown alignment '{decision}'.")
    if item_id not in review["decisions"]:
        raise KeyError(item_id)
    before = review["decisions"][item_id]
    if before != decision:
        review["decisions"][item_id] = decision
        _log(review, reviewer, "candidate_decision", {item_id: before}, {item_id: decision})


def set_include_all(review, reviewer, include_all):
    if review["include_all_retrieved"] != bool(include_all):
        _log(review, reviewer, "denominator_switch", review["include_all_retrieved"], bool(include_all))
        review["include_all_retrieved"] = bool(include_all)


def task_confirmed(review):
    return review is not None and review["task_status"] in ("accepted", "edited") and bool(review["task"])


def accepted_siblings(review):
    """Distinct sibling IDs in the denominator. The target is never included."""
    if review is None:
        return []
    if review["include_all_retrieved"]:
        chosen = review["candidate_ids"]
    else:
        chosen = [cid for cid in review["candidate_ids"] if review["decisions"].get(cid) == "same_task"]
    out = []
    for cid in chosen:
        if cid != review["target_id"] and cid not in out:
            out.append(cid)
    return out


def _log(review, reviewer, action, before, after):
    review["history"].append({"timestamp": now(), "reviewer": reviewer or "anonymous",
                              "action": action, "before": before, "after": after})
