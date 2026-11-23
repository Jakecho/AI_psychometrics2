"""Searchable library of feature spaces (current and archived) by task, domain and feature type."""

import pandas as pd

from . import feature_frequency as ff
from .schemas import FEATURE_TYPE_LABELS, FEATURE_TYPES, INACTIVE_STATUSES
from .task_alignment import accepted_siblings

NO_TASK = "(task not confirmed)"
NO_DOMAIN = "(no domain)"


def collect_spaces(current_review, current_store, archived, records_by_id=None):
    """One entry per space: the active review plus each archived review that has a feature store."""
    raw = []
    if current_review is not None or current_store:
        raw.append(("current", current_review, current_store))
    for n, a in enumerate(archived or [], start=1):
        raw.append((f"archived {n}", a.get("task_review"), a.get("store")))
    spaces = []
    for name, review, store in raw:
        if review is None and not store:
            continue
        task = (review or {}).get("task") or {}
        confirmed = (review or {}).get("task_status") in ("accepted", "edited")
        target = (review or {}).get("target_id") or ((store or {}).get("basis") or {}).get("target_id")
        item_domain = None
        if records_by_id and target in records_by_id:
            item_domain = records_by_id[target]["roles"].get("domain")
        spaces.append({
            "space": name, "target_id": target, "task": task.get("name") if confirmed and task.get("name") else NO_TASK,
            "decision_or_action": task.get("decision_or_action") or "",
            "domain": task.get("domain") or (str(item_domain) if item_domain else NO_DOMAIN),
            "task_status": (review or {}).get("task_status", "unreviewed"),
            "siblings": accepted_siblings(review) if review else [],
            "store": store or {"order": [], "features": {}},
        })
    return spaces


def space_summary(spaces):
    rows = []
    for s in spaces:
        feats = [s["store"]["features"][fid] for fid in s["store"]["order"]]
        active = [f for f in feats if f["review_status"] not in INACTIVE_STATUSES]
        row = {"space": s["space"], "task": s["task"], "domain": s["domain"], "target_item": s["target_id"],
               "task_status": s["task_status"], "confirmed_siblings_N": len(s["siblings"])}
        for t in FEATURE_TYPES:
            row[FEATURE_TYPE_LABELS[t]] = sum(f["feature_type"] == t for f in active)
        row["removed"] = sum(f["review_status"] == "removed" for f in feats)
        rows.append(row)
    return pd.DataFrame(rows, columns=["space", "task", "domain", "target_item", "task_status", "confirmed_siblings_N",
                                       *[FEATURE_TYPE_LABELS[t] for t in FEATURE_TYPES], "removed"])


def space_features(spaces):
    """One row per feature (merged duplicates excluded) with its space's task and domain."""
    rows = []
    for s in spaces:
        freq = ff.frequency_table(s["store"], s["siblings"], s["target_id"], "sme").set_index("feature_id") \
            if s["store"]["order"] else None
        for fid in s["store"]["order"]:
            f = s["store"]["features"][fid]
            if f["review_status"] == "merged":
                continue
            approved = freq is not None and fid in freq.index
            n = len(s["siblings"])
            rows.append({
                "space": s["space"], "task": s["task"], "domain": s["domain"], "target_item": s["target_id"],
                "space_type": FEATURE_TYPE_LABELS[f["feature_type"]], "feature_id": fid,
                "label": f["canonical_label"], "definition": f.get("definition") or "",
                "status": f["review_status"], "origin": "SME-added" if f.get("origin") == "sme" else "AI proposal",
                "importance": f.get("sme_importance"),
                "coverage (SME-approved)": freq.loc[fid, "coverage"] if approved else ("N/A" if not n else "not approved"),
                "supporting_items": ", ".join(f["supporting_items"]),
                "options": ", ".join(f"{r['item_id']}:{r['option_id']}" for r in f.get("option_refs", [])),
                "evidence": ", ".join(f.get("evidence_types", [])),
            })
    return pd.DataFrame(rows, columns=["space", "task", "domain", "target_item", "space_type", "feature_id", "label",
                                       "definition", "status", "origin", "importance", "coverage (SME-approved)",
                                       "supporting_items", "options", "evidence"])


def filter_features(df, tasks=None, domains=None, space_types=None, statuses=None, text=""):
    out = df
    if tasks:
        out = out[out["task"].isin(tasks)]
    if domains:
        out = out[out["domain"].isin(domains)]
    if space_types:
        out = out[out["space_type"].isin(space_types)]
    if statuses:
        out = out[out["status"].isin(statuses)]
    text = (text or "").strip().lower()
    if text:
        hay = (out["label"] + " " + out["definition"] + " " + out["task"] + " " + out["domain"]).str.lower()
        out = out[hay.str.contains(text, regex=False)]
    return out.reset_index(drop=True)
