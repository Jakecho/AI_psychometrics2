"""Option-level review: every option of every unique item, with key status and linked features."""

import pandas as pd

KEY_PENDING = {"missing": "Key not supplied — distractor analysis pending",
               "unresolved": "Key could not be matched to an option — distractor analysis pending"}


def option_review(records_by_id, target_id, sibling_ids, store=None):
    links = {}
    if store:
        for fid in store["order"]:
            f = store["features"][fid]
            if f["feature_type"] != "distractor_feature" or f["review_status"] == "merged":
                continue
            for ref in f["option_refs"]:
                links.setdefault((ref["item_id"], ref["option_id"]), []).append(fid)
    rows = []
    for iid in [target_id] + [i for i in dict.fromkeys(sibling_ids) if i != target_id]:
        record = records_by_id.get(iid)
        if record is None:
            continue
        for o in record["options"]:
            role = "key" if o["is_key"] else "distractor" if o["is_key"] is False else "unknown"
            rows.append({
                "item_id": iid,
                "item_role": "target" if iid == target_id else "sibling",
                "option_id": o["option_id"],
                "option_text": o["text"],
                "option_role": role,
                "key_status": record["key_status"],
                "linked_distractor_features": ", ".join(links.get((iid, o["option_id"]), [])),
                "analysis_status": KEY_PENDING.get(record["key_status"],
                                                   "Not a distractor" if role == "key" else "Available for review"),
            })
    return pd.DataFrame(rows, columns=["item_id", "item_role", "option_id", "option_text", "option_role", "key_status",
                                       "linked_distractor_features", "analysis_status"])
