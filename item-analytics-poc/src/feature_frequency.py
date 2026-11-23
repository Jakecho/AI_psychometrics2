"""Item-level feature incidence and frequencies over the confirmed sibling set.

For N distinct confirmed siblings, A[i,f] = 1 if item i supports canonical
feature f; n_f = Σ_i A[i,f]; coverage = 100·n_f/N. Each item counts once per
feature however many spans or options support it. The target is reported
separately and is not part of N.
"""

import pandas as pd

from .schemas import APPROVED_STATUSES


def _features(store, source):
    out = []
    for fid in store["order"]:
        f = store["features"][fid]
        if source == "sme":
            if f["review_status"] in APPROVED_STATUSES:
                out.append((fid, f, f["supporting_items"], f["option_refs"]))
        elif source == "raw":
            if not f.get("split_from") and f.get("origin") != "sme":
                out.append((fid, f, f["ai"]["supporting_items"], f["ai"]["option_refs"]))
        else:
            raise ValueError("source must be 'sme' or 'raw'")
    return out


def incidence_matrix(store, item_ids, source="sme"):
    item_ids = list(dict.fromkeys(item_ids))
    feats = _features(store, source)
    data = {fid: [1 if iid in set(support) else 0 for iid in item_ids] for fid, _f, support, _r in feats}
    return pd.DataFrame(data, index=pd.Index(item_ids, name="item_id"), dtype=int)


def frequency_table(store, sibling_ids, target_id, source="sme"):
    siblings = [i for i in dict.fromkeys(sibling_ids) if i != target_id]
    n = len(siblings)
    sib_set = set(siblings)
    rows = []
    for fid, f, support, refs in _features(store, source):
        support_set = set(support)
        hits = [i for i in siblings if i in support_set]
        option_count = None
        if f["feature_type"] == "distractor_feature":
            option_count = len({(r["item_id"], r["option_id"]) for r in refs if r["item_id"] in sib_set})
        rows.append({
            "feature_id": fid,
            "feature_type": f["feature_type"],
            "canonical_label": f["canonical_label"],
            "item_frequency": len(hits),
            "denominator_N": n,
            "coverage_pct": round(100 * len(hits) / n, 1) if n else None,
            "coverage": f"{len(hits)}/{n} = {100 * len(hits) / n:.0f}%" if n else "N/A (no confirmed siblings)",
            "supporting_sibling_ids": ", ".join(hits),
            "in_target": target_id in support_set,
            "distractor_option_count": option_count,
            "evidence_types": ", ".join(f.get("evidence_types", [])),
            "review_status": f["review_status"],
            "sme_importance": f.get("sme_importance"),
            "incidence_source": "SME-approved" if source == "sme" else "Raw AI proposal",
        })
    return pd.DataFrame(rows, columns=[
        "feature_id", "feature_type", "canonical_label", "item_frequency", "denominator_N", "coverage_pct",
        "coverage", "supporting_sibling_ids", "in_target", "distractor_option_count", "evidence_types",
        "review_status", "sme_importance", "incidence_source",
    ])


def long_incidence(store, target_id, sibling_ids, source="sme"):
    """Rows of item_id, role, feature_id, feature_type, value for reproducible export."""
    items = [target_id] + [i for i in dict.fromkeys(sibling_ids) if i != target_id]
    rows = []
    for fid, f, support, _refs in _features(store, source):
        support_set = set(support)
        for iid in items:
            rows.append({"item_id": iid, "role": "target" if iid == target_id else "sibling", "feature_id": fid,
                         "feature_type": f["feature_type"], "value": int(iid in support_set)})
    return pd.DataFrame(rows, columns=["item_id", "role", "feature_id", "feature_type", "value"])
