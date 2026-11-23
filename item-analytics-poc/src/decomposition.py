"""Four-space feature decomposition, evidence validation, canonicalization and SME review state.

Raw model proposals are kept verbatim in run records. Canonical features carry
an immutable ``ai`` snapshot plus SME-editable fields. An SME-touched feature is
never overwritten by a rerun; the new proposal waits in ``pending_updates``.
"""

from copy import deepcopy
import json
import re
import uuid

from .evidence import DATA_NOTICE, check_span, item_payload
from .schemas import (DECOMPOSITION_SCHEMA, DOMINANCE, FEATURE_PREFIX, FEATURE_TYPES, REVIEW_STATUSES,
                      SCHEMA_VERSION)
from .task_alignment import now, system_prompt, task_confirmed

DEFAULT_CHUNK_CHARS = 30_000
EDITABLE = ("canonical_label", "definition", "feature_type", "task_relevance", "dominance", "procedure_group",
            "sequence_order", "supporting_items", "review_status", "sme_importance", "sme_note")


class DecompositionGateError(RuntimeError):
    pass


# --------------------------------------------------------------------------- payloads

def build_payloads(target, siblings, task, max_chars=DEFAULT_CHUNK_CHARS):
    """Bounded requests: each contains the confirmed task, the target, and a subset of siblings."""
    base = {"data_notice": DATA_NOTICE, "schema_version": SCHEMA_VERSION, "confirmed_task": task,
            "target_item_id": target["item_id"], "target": item_payload(target)}
    base_size = len(json.dumps(base, ensure_ascii=False))
    chunks, current, size = [], [], base_size
    for record in siblings:
        p = item_payload(record)
        p_size = len(json.dumps(p, ensure_ascii=False))
        if current and size + p_size > max_chars:
            chunks.append(current)
            current, size = [], base_size
        current.append(p)
        size += p_size
    if current or not chunks:
        chunks.append(current)
    return [{**deepcopy(base), "siblings": chunk, "accepted_sibling_ids": [p["item_id"] for p in chunk]}
            for chunk in chunks]


# --------------------------------------------------------------------------- validation

def validate_feature(raw, items_by_id):
    """Return a cleaned feature with ``validation_issues``; unverifiable evidence is removed."""
    f = deepcopy(raw)
    issues = []
    if f.get("feature_type") not in FEATURE_TYPES:
        issues.append(f"unknown feature type '{f.get('feature_type')}'")
        f["feature_type"] = "declarative_key"
    if f["feature_type"] == "procedural_key":
        if f.get("dominance") not in DOMINANCE:
            f["dominance"] = "unknown"
    else:
        f["dominance"] = None
    spans = []
    for span in f.get("source_spans", []):
        problems = check_span(span, items_by_id)
        if not problems and span["evidence_type"] == "clinically_inferred" and not (f.get("inference_rationale") or "").strip():
            problems = [f"inferred evidence for item {span['item_id']} lacks a reasoning rationale"]
        if not problems and f["feature_type"] == "distractor_feature":
            item = items_by_id[span["item_id"]]
            option = next((o for o in item["options"] if o["option_id"] == span.get("option_id")), None)
            if option is None:
                problems = [f"distractor evidence for item {span['item_id']} does not cite an option"]
            elif option["is_key"] is None:
                problems = [f"item {span['item_id']} has no usable key; distractor analysis is pending"]
            elif option["is_key"]:
                problems = [f"option {option['option_id']} of item {span['item_id']} is the key, not a distractor"]
        if problems:
            issues.extend(problems)
        else:
            spans.append(span)
    f["source_spans"] = spans
    evidenced = []
    for s in spans:
        if s["item_id"] not in evidenced:
            evidenced.append(s["item_id"])
    claimed = [i for i in f.get("supporting_items", [])]
    unknown = [i for i in claimed if i not in items_by_id]
    if unknown:
        issues.append(f"removed unknown item IDs {', '.join(map(str, unknown))}")
    unsupported = [i for i in claimed if i in items_by_id and i not in evidenced]
    if unsupported:
        issues.append(f"no verifiable evidence for claimed support from {', '.join(unsupported)}")
    f["supporting_items"] = evidenced
    f["option_refs"] = sorted({(s["item_id"], s["option_id"]) for s in spans if s.get("option_id")})
    f["option_refs"] = [{"item_id": i, "option_id": o} for i, o in f["option_refs"]]
    f["evidence_types"] = sorted({s["evidence_type"] for s in spans})
    f["validation_issues"] = issues
    f["raw_feature_ids"] = [raw.get("feature_id")]
    return f


def normalize_label(label):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/%.-]", " ", (label or "").lower())).strip()


def canonical_key(feature):
    key = (feature["feature_type"], normalize_label(feature["canonical_label"]))
    if feature["feature_type"] == "procedural_key":
        key += (feature.get("dominance") or "unknown",)
    return key


def canonicalize(features):
    """Merge exact normalized-label matches within a type (and dominance for procedures)."""
    merged = {}
    for f in features:
        key = canonical_key(f)
        if key not in merged:
            merged[key] = deepcopy(f)
            continue
        m = merged[key]
        for iid in f["supporting_items"]:
            if iid not in m["supporting_items"]:
                m["supporting_items"].append(iid)
        seen = {json.dumps(s, sort_keys=True) for s in m["source_spans"]}
        m["source_spans"] += [s for s in f["source_spans"] if json.dumps(s, sort_keys=True) not in seen]
        refs = {(r["item_id"], r["option_id"]) for r in m["option_refs"]} | {(r["item_id"], r["option_id"]) for r in f["option_refs"]}
        m["option_refs"] = [{"item_id": i, "option_id": o} for i, o in sorted(refs)]
        m["evidence_types"] = sorted(set(m["evidence_types"]) | set(f["evidence_types"]))
        m["validation_issues"] += [i for i in f["validation_issues"] if i not in m["validation_issues"]]
        m["raw_feature_ids"] += f["raw_feature_ids"]
        if f.get("inference_rationale") and f["inference_rationale"] not in (m.get("inference_rationale") or ""):
            m["inference_rationale"] = "; ".join(x for x in (m.get("inference_rationale"), f["inference_rationale"]) if x)
    return list(merged.values())


# --------------------------------------------------------------------------- store

def new_store():
    return {"runs": [], "features": {}, "order": [], "pending_updates": {}, "history": [], "basis": None}


def _next_id(store, ftype):
    prefix = FEATURE_PREFIX[ftype]
    used = [int(fid[1:]) for fid in store["features"] if fid.startswith(prefix) and fid[1:].isdigit()]
    return f"{prefix}{(max(used) + 1 if used else 1):03d}"


_AI_FIELDS = ("canonical_label", "definition", "feature_type", "task_relevance", "dominance", "procedure_group",
              "sequence_order", "supporting_items", "source_spans", "option_refs", "evidence_types",
              "inference_rationale", "proposed_misconception", "llm_uncertainty_note", "validation_issues",
              "raw_feature_ids")


def _from_proposal(fid, proposal, run_id):
    ai = {k: deepcopy(proposal.get(k)) for k in _AI_FIELDS}
    unsupported = not proposal["supporting_items"]
    feature = deepcopy(ai)
    feature.update({
        "feature_id": fid, "ai": ai, "run_ids": [run_id],
        "review_status": "needs_adjudication" if (unsupported or proposal["validation_issues"]) else "pending",
        "sme_importance": None, "sme_note": "", "sme_touched": False,
        "merged_into": None, "merged_from": [], "related": [],
    })
    return feature


def integrate(store, proposals, run, reviewer):
    """Add canonical proposals from a run without overwriting SME decisions."""
    store["runs"].append(run)
    by_key = {canonical_key(f["ai"]): fid for fid, f in store["features"].items()}
    added, refreshed, pending = [], [], []
    for proposal in proposals:
        fid = by_key.get(canonical_key(proposal))
        if fid is None:
            fid = _next_id(store, proposal["feature_type"])
            store["features"][fid] = _from_proposal(fid, proposal, run["run_id"])
            store["order"].append(fid)
            by_key[canonical_key(proposal)] = fid
            added.append(fid)
        elif not store["features"][fid]["sme_touched"]:
            store["features"][fid] = _from_proposal(fid, proposal, run["run_id"]) | {
                "run_ids": store["features"][fid]["run_ids"] + [run["run_id"]]}
            refreshed.append(fid)
        else:
            store["pending_updates"][fid] = {"run_id": run["run_id"], "proposal": deepcopy(proposal)}
            store["features"][fid]["run_ids"].append(run["run_id"])
            pending.append(fid)
    _log(store, reviewer, "integrate_run", None, {"run_id": run["run_id"], "added": added,
                                                  "refreshed": refreshed, "pending_confirmation": pending})
    return {"added": added, "refreshed": refreshed, "pending": pending}


def decompose(llm, target, sibling_records, review, store, reviewer, max_chars=DEFAULT_CHUNK_CHARS):
    """Run the decomposition in bounded chunks. Validated output from completed chunks is kept on failure."""
    if not task_confirmed(review):
        raise DecompositionGateError("Accept or edit the target task before extracting features.")
    task = review["task"]
    payloads = build_payloads(target, sibling_records, task, max_chars=max_chars)
    items_by_id = {target["item_id"]: item_payload(target)}
    items_by_id.update({r["item_id"]: item_payload(r) for r in sibling_records})
    validated, raws, usage, errors = [], [], {"input_tokens": 0, "output_tokens": 0}, []
    for n, payload in enumerate(payloads, start=1):
        try:
            raw, u = llm.complete(system_prompt("feature_decomposition"), payload, DECOMPOSITION_SCHEMA,
                                  "feature_decomposition")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Chunk {n} of {len(payloads)}: {exc}")
            continue
        raws.append(raw)
        for k in usage:
            usage[k] += u.get(k, 0)
        chunk_items = {payload["target"]["item_id"]: items_by_id[payload["target"]["item_id"]]}
        chunk_items.update({p["item_id"]: items_by_id[p["item_id"]] for p in payload["siblings"]})
        validated.extend(validate_feature(f, chunk_items) for f in raw.get("features", []))
    canonical = canonicalize(validated)
    run = {
        "run_id": uuid.uuid4().hex[:12], "created_at": now(), "model": getattr(llm, "model", "unknown"),
        "target_id": target["item_id"], "sibling_ids": [r["item_id"] for r in sibling_records],
        "task": deepcopy(task), "chunks": len(payloads), "raw_outputs": raws, "errors": errors, "usage": usage,
        "include_all_retrieved": review["include_all_retrieved"],
    }
    result = integrate(store, canonical, run, reviewer) if raws else {"added": [], "refreshed": [], "pending": []}
    if not raws:
        store["runs"].append(run)
    store["basis"] = {"target_id": target["item_id"], "sibling_ids": run["sibling_ids"]}
    return run, result


# --------------------------------------------------------------------------- SME edits

def update_feature(store, fid, changes, reviewer, known_items=None):
    f = store["features"][fid]
    clean = {}
    for key, value in changes.items():
        if key not in EDITABLE:
            raise ValueError(f"'{key}' is not editable.")
        if key == "sme_importance" and value not in (None, 1, 2, 3, 4, 5):
            raise ValueError("Importance must be unrated or 1–5.")
        if key == "review_status" and value not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status '{value}'.")
        if key == "feature_type" and value not in FEATURE_TYPES:
            raise ValueError(f"Unknown feature type '{value}'.")
        if key == "dominance" and value is not None and value not in DOMINANCE:
            raise ValueError(f"Unknown dominance '{value}'.")
        if key == "supporting_items":
            value = list(dict.fromkeys(value))
            if known_items is not None and any(i not in known_items for i in value):
                raise ValueError("Supporting items must be existing item IDs.")
        if key == "sequence_order" and value is not None:
            value = int(value)
        if f.get(key) != value:
            clean[key] = value
    if not clean:
        return False
    before = {k: deepcopy(f.get(k)) for k in clean}
    f.update(clean)
    content = {"canonical_label", "definition", "feature_type", "supporting_items", "dominance"}
    if content & clean.keys() and "review_status" not in clean and f["review_status"] in ("pending", "accepted", "needs_adjudication"):
        f["review_status"] = "edited"
    if f["feature_type"] != "procedural_key":
        f["dominance"] = None
    elif f.get("dominance") is None:
        f["dominance"] = "unknown"
    f["sme_touched"] = True
    _log(store, reviewer, "edit_feature", {fid: before}, {fid: clean})
    return True


def accept_pending_update(store, fid, reviewer):
    """Explicitly replace AI-derived evidence of an SME-touched feature with the newer proposal."""
    pending = store["pending_updates"].pop(fid)
    f = store["features"][fid]
    before = {k: deepcopy(f.get(k)) for k in ("supporting_items", "source_spans", "option_refs", "ai")}
    proposal = pending["proposal"]
    f["ai"] = {k: deepcopy(proposal.get(k)) for k in _AI_FIELDS}
    for k in ("supporting_items", "source_spans", "option_refs", "evidence_types", "validation_issues"):
        f[k] = deepcopy(proposal[k])
    _log(store, reviewer, "accept_pending_update", {fid: before}, {fid: {"run_id": pending["run_id"]}})


def discard_pending_update(store, fid, reviewer):
    pending = store["pending_updates"].pop(fid)
    _log(store, reviewer, "discard_pending_update", None, {fid: {"run_id": pending["run_id"]}})


def merge_features(store, keep_id, other_ids, reviewer):
    keep = store["features"][keep_id]
    others = [store["features"][o] for o in other_ids if o != keep_id]
    if any(o["feature_type"] != keep["feature_type"] for o in others):
        raise ValueError("Only features of the same type can be merged. Change a feature's type first if needed.")
    before = deepcopy({fid: store["features"][fid] for fid in [keep_id] + list(other_ids)})
    for o in others:
        for iid in o["supporting_items"]:
            if iid not in keep["supporting_items"]:
                keep["supporting_items"].append(iid)
        seen = {json.dumps(s, sort_keys=True) for s in keep["source_spans"]}
        keep["source_spans"] += [s for s in o["source_spans"] if json.dumps(s, sort_keys=True) not in seen]
        refs = {(r["item_id"], r["option_id"]) for r in keep["option_refs"] + o["option_refs"]}
        keep["option_refs"] = [{"item_id": i, "option_id": oid} for i, oid in sorted(refs)]
        keep["evidence_types"] = sorted(set(keep["evidence_types"]) | set(o["evidence_types"]))
        keep["merged_from"].append(o["feature_id"])
        o["review_status"] = "merged"
        o["merged_into"] = keep_id
        o["sme_touched"] = True
    keep["sme_touched"] = True
    if keep["review_status"] in ("pending", "accepted", "needs_adjudication"):
        keep["review_status"] = "edited"
    _log(store, reviewer, "merge_features", before, {"kept": keep_id, "merged": list(other_ids)})


def split_feature(store, fid, item_ids, new_label, reviewer):
    """Move the support for ``item_ids`` (and their spans) to a new feature of the same type."""
    src = store["features"][fid]
    item_ids = [i for i in item_ids if i in src["supporting_items"]]
    if not item_ids or not new_label.strip():
        raise ValueError("Choose supporting items to move and a label for the new feature.")
    before = deepcopy(src)
    new_id = _next_id(store, src["feature_type"])
    moved_spans = [s for s in src["source_spans"] if s["item_id"] in item_ids]
    new = deepcopy(src)
    new.update({
        "feature_id": new_id, "canonical_label": new_label.strip(), "supporting_items": item_ids,
        "source_spans": moved_spans,
        "option_refs": [r for r in src["option_refs"] if r["item_id"] in item_ids],
        "review_status": "edited", "sme_touched": True, "merged_from": [], "merged_into": None,
        "related": [fid], "split_from": fid, "sme_importance": None, "sme_note": "",
    })
    src["supporting_items"] = [i for i in src["supporting_items"] if i not in item_ids]
    src["source_spans"] = [s for s in src["source_spans"] if s["item_id"] not in item_ids]
    src["option_refs"] = [r for r in src["option_refs"] if r["item_id"] not in item_ids]
    src["related"] = list(dict.fromkeys(src["related"] + [new_id]))
    src["sme_touched"] = True
    if src["review_status"] in ("pending", "accepted", "needs_adjudication"):
        src["review_status"] = "edited"
    store["features"][new_id] = new
    store["order"].insert(store["order"].index(fid) + 1, new_id)
    _log(store, reviewer, "split_feature", {fid: before}, {"kept": fid, "new": new_id, "items": item_ids})
    return new_id


def editor_changes(before_rows, after_rows):
    """Translate edited review-table rows into ``update_feature`` changes, keyed by feature ID.

    Rows are matched by ``feature_id`` (never by position). Importance arrives as
    "unrated" or "1"–"5"; blank procedure groups and orders become None.
    """
    def blank(v):
        return v is None or v == "" or (isinstance(v, float) and v != v)

    before = {r["feature_id"]: r for r in before_rows}
    out = {}
    for after in after_rows:
        prev = before.get(after["feature_id"])
        if prev is None:
            continue
        changes = {}
        for key in ("canonical_label", "definition", "review_status", "sme_note", "procedure_group"):
            if key in after and after[key] != prev[key] and not (blank(after[key]) and blank(prev[key])):
                changes[key] = None if (key == "procedure_group" and blank(after[key])) else after[key]
        if "importance" in after and after["importance"] != prev["importance"]:
            changes["sme_importance"] = None if after["importance"] in ("unrated", None) else int(after["importance"])
        if "dominance" in after and after["dominance"] != prev["dominance"] and not blank(after["dominance"]):
            changes["dominance"] = after["dominance"]
        if "sequence_order" in after:
            a = None if blank(after["sequence_order"]) else int(after["sequence_order"])
            b = None if blank(prev["sequence_order"]) else int(prev["sequence_order"])
            if a != b:
                changes["sequence_order"] = a
        if changes.get("review_status") == "merged":
            changes.pop("review_status")  # merging goes through merge_features
        if changes:
            out[after["feature_id"]] = changes
    return out


def features_frame_rows(store, types=None, view="all"):
    """Rows for display; view is all, accepted, rejected or pending."""
    rows = []
    for fid in store["order"]:
        f = store["features"][fid]
        if types and f["feature_type"] not in types:
            continue
        status = f["review_status"]
        if view == "accepted" and status not in ("accepted", "edited"):
            continue
        if view == "rejected" and status not in ("rejected", "not_required", "unsupported", "merged"):
            continue
        rows.append(f)
    return rows


def _log(store, reviewer, action, before, after):
    store["history"].append({"timestamp": now(), "reviewer": reviewer or "anonymous", "action": action,
                             "before": before, "after": after})
