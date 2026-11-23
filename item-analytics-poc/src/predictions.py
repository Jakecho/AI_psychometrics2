"""Similarity-based difficulty and classification heuristics, with grouped holdout diagnostics.

These are exploratory nearest-neighbour heuristics, not IRT calibration or
validated classifiers. Difficulty is only aggregated when the user confirms
that labels are on one linked scale; items from different scale groups are
never pooled.
"""

import math

import numpy as np
import pandas as pd

MULTI_LABEL_SEPARATORS = (";", "|")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- difficulty

def difficulty_estimate(neighbors, records_by_id, alpha=1.0, min_neighbors=2, scale_confirmed=False):
    """``neighbors``: DataFrame with item_id and cosine_similarity."""
    rows = []
    for _, n in neighbors.iterrows():
        r = records_by_id[n["item_id"]]
        rows.append({"item_id": n["item_id"], "cosine_similarity": float(n["cosine_similarity"]),
                     "difficulty": _num(r["roles"].get("difficulty")), "scale_group": r["roles"].get("scale_group")})
    table = pd.DataFrame(rows, columns=["item_id", "cosine_similarity", "difficulty", "scale_group"])
    labeled = table.dropna(subset=["difficulty"])
    out = {"table": table, "n_neighbors": len(table), "n_labeled": len(labeled), "estimate": None,
           "status": None, "message": ""}
    if len(labeled):
        b = labeled["difficulty"].to_numpy(dtype=float)
        out.update({"unweighted_mean": float(b.mean()), "sd": float(b.std(ddof=1)) if len(b) > 1 else None,
                    "min": float(b.min()), "max": float(b.max())})
    scales = {s for s in labeled["scale_group"] if s not in (None, "")}
    if len(scales) > 1:
        out.update(status="non_comparable",
                   message=f"Neighbours come from {len(scales)} scale groups ({', '.join(map(str, sorted(scales)))}). "
                           "Difficulties from unlinked scales are not averaged.")
        out.pop("unweighted_mean", None)
        return out
    if len(labeled) < min_neighbors:
        out.update(status="abstain", message=f"Only {len(labeled)} labelled neighbours (minimum {min_neighbors}).")
        return out
    if not scale_confirmed:
        out.update(status="exploratory",
                   message="Scale comparability is not confirmed. The mean is descriptive only and no estimate is given.")
        return out
    w = np.maximum(labeled["cosine_similarity"].to_numpy(dtype=float), 0.0) ** alpha
    if w.sum() <= 0:
        out.update(status="abstain", message="All neighbour weights are zero (no positive cosine similarity).")
        return out
    out.update(status="estimate", estimate=float((w * labeled["difficulty"].to_numpy()).sum() / w.sum()),
               message="Similarity-weighted neighbour mean. Heuristic; not an IRT calibration.")
    return out


def near_duplicate_families(index, threshold=0.95):
    """Connected components of items with cosine >= threshold, as a default holdout grouping."""
    n = len(index.item_ids)
    sims = index.vectors @ index.vectors.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if not index.valid[i]:
            continue
        for j in np.nonzero(sims[i, i + 1:] >= threshold)[0] + i + 1:
            if index.valid[j]:
                parent[find(i)] = find(int(j))
    return {iid: f"auto-{find(i)}" for i, iid in enumerate(index.item_ids)}


def _groups(records, index, family_role=True, dup_threshold=0.95):
    auto = near_duplicate_families(index, dup_threshold)
    out = {}
    for r in records:
        fam = r["roles"].get("family") if family_role else None
        out[r["item_id"]] = f"family-{fam}" if fam not in (None, "") else auto[r["item_id"]]
    return out


def _neighbours(index, pos, candidate_positions, k):
    sims = index.vectors[candidate_positions] @ index.vectors[pos]
    order = sorted(range(len(candidate_positions)), key=lambda j: (-sims[j], index.item_ids[candidate_positions[j]]))[:k]
    return [(candidate_positions[j], float(sims[j])) for j in order]


def evaluate_difficulty(index, records, k=5, alpha=1.0, min_neighbors=2, dup_threshold=0.95, min_items=6):
    """Grouped leave-one-family-out evaluation against a naive mean baseline."""
    pos_of = {iid: i for i, iid in enumerate(index.item_ids)}
    by_id = {r["item_id"]: r for r in records}
    labeled = [r for r in records if _num(r["roles"].get("difficulty")) is not None and index.valid[pos_of[r["item_id"]]]]
    scales = {r["roles"].get("scale_group") for r in labeled} - {None, ""}
    if len(scales) > 1:
        return {"status": "non_comparable", "message": "Labels span more than one scale group; evaluate within one linked scale."}
    if len(labeled) < min_items:
        return {"status": "insufficient", "message": f"Holdout needs at least {min_items} labelled items (found {len(labeled)})."}
    groups = _groups(records, index, dup_threshold=dup_threshold)
    rows = []
    for r in labeled:
        train = [x for x in labeled if groups[x["item_id"]] != groups[r["item_id"]]]
        if len(train) < min_neighbors:
            continue
        nb = _neighbours(index, pos_of[r["item_id"]], [pos_of[x["item_id"]] for x in train], k)
        b = np.array([_num(by_id[index.item_ids[p]]["roles"]["difficulty"]) for p, _ in nb])
        w = np.maximum(np.array([s for _, s in nb]), 0) ** alpha
        pred = float((w * b).sum() / w.sum()) if w.sum() > 0 else None
        baseline = float(np.mean([_num(x["roles"]["difficulty"]) for x in train]))
        rows.append({"item_id": r["item_id"], "group": groups[r["item_id"]], "domain": r["roles"].get("domain"),
                     "observed": _num(r["roles"]["difficulty"]), "knn_prediction": pred, "baseline_mean": baseline})
    table = pd.DataFrame(rows)
    if table.empty:
        return {"status": "insufficient", "message": "No item had enough out-of-family neighbours."}
    scored = table.dropna(subset=["knn_prediction"])

    def metrics(col):
        err = scored[col] - scored["observed"]
        return {"MAE": float(err.abs().mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean())}

    summary = pd.DataFrame([{"method": "weighted kNN", **metrics("knn_prediction")},
                            {"method": "naive mean baseline", **metrics("baseline_mean")}])
    per_domain = None
    if scored["domain"].notna().any():
        per_domain = (scored.assign(abs_error=(scored["knn_prediction"] - scored["observed"]).abs())
                      .groupby("domain", dropna=False)["abs_error"].agg(["count", "mean"]).reset_index()
                      .rename(columns={"mean": "kNN_MAE", "count": "n"}))
    return {"status": "ok", "table": table, "summary": summary, "per_domain": per_domain,
            "n_evaluated": len(scored), "n_abstained": int(table["knn_prediction"].isna().sum()),
            "n_groups": table["group"].nunique(),
            "message": "Each item was predicted only from items outside its family/near-duplicate group."}


# --------------------------------------------------------------------------- classification

def split_labels(value):
    if value in (None, ""):
        return []
    text = str(value)
    for sep in MULTI_LABEL_SEPARATORS:
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text.strip()]


def classify(neighbors, records_by_id, role="domain", min_labeled=2, margin=0.1):
    rows = []
    for _, n in neighbors.iterrows():
        labels = split_labels(records_by_id[n["item_id"]]["roles"].get(role))
        rows.append({"item_id": n["item_id"], "cosine_similarity": float(n["cosine_similarity"]),
                     "labels": labels, "weight": max(float(n["cosine_similarity"]), 0.0)})
    table = pd.DataFrame(rows, columns=["item_id", "cosine_similarity", "labels", "weight"])
    labeled = table[table["labels"].map(len) > 0]
    out = {"table": table, "n_labeled": len(labeled), "prediction": None, "support": None, "status": None, "message": ""}
    if len(labeled) < min_labeled:
        out.update(status="abstain", message=f"Only {len(labeled)} labelled neighbours (minimum {min_labeled}).")
        return out
    total = labeled["weight"].sum()
    if total <= 0:
        out.update(status="abstain", message="No neighbour has a positive similarity weight.")
        return out
    support = {}
    for _, r in labeled.iterrows():
        for lab in r["labels"]:
            support[lab] = support.get(lab, 0.0) + r["weight"]
    support = pd.DataFrame(sorted(((k, v / total) for k, v in support.items()), key=lambda kv: (-kv[1], kv[0])),
                           columns=["label", "weighted_support"])
    out["support"] = support
    if labeled["labels"].map(len).max() > 1:
        out.update(status="multi_label", message="Neighbours carry multiple labels. Support is shown per label; no single label is forced.")
        return out
    top = support["weighted_support"].tolist()
    if len(top) > 1 and top[0] - top[1] < margin:
        out.update(status="abstain", message=f"Top labels are within the {margin:.2f} support margin (tie or weak evidence).")
        return out
    out.update(status="prediction", prediction=support["label"].iloc[0],
               message="Similarity-weighted vote among labelled neighbours. Heuristic; requires SME confirmation.")
    return out


def evaluate_classification(index, records, role="domain", k=5, min_labeled=2, margin=0.1, dup_threshold=0.95, min_items=6):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    pos_of = {iid: i for i, iid in enumerate(index.item_ids)}
    by_id = {r["item_id"]: r for r in records}
    single = [r for r in records if len(split_labels(r["roles"].get(role))) == 1 and index.valid[pos_of[r["item_id"]]]]
    multi = sum(len(split_labels(r["roles"].get(role))) > 1 for r in records)
    if len(single) < min_items:
        return {"status": "insufficient", "message": f"Holdout needs at least {min_items} single-label items (found {len(single)})."}
    groups = _groups(records, index, dup_threshold=dup_threshold)
    rows = []
    for r in single:
        train = [x for x in single if groups[x["item_id"]] != groups[r["item_id"]]]
        nb = _neighbours(index, pos_of[r["item_id"]], [pos_of[x["item_id"]] for x in train], k)
        nb_frame = pd.DataFrame([{"item_id": index.item_ids[p], "cosine_similarity": s} for p, s in nb],
                                columns=["item_id", "cosine_similarity"])
        res = classify(nb_frame, by_id, role=role, min_labeled=min_labeled, margin=margin)
        rows.append({"item_id": r["item_id"], "observed": split_labels(r["roles"][role])[0],
                     "predicted": res["prediction"], "status": res["status"]})
    table = pd.DataFrame(rows)
    decided = table[table["predicted"].notna()]
    out = {"status": "ok", "table": table, "n_abstained": int(table["predicted"].isna().sum()),
           "n_multi_label_skipped": int(multi), "class_counts": table["observed"].value_counts().rename_axis("label").reset_index(name="n")}
    if decided.empty:
        out.update(status="all_abstained", message="Every held-out item abstained.")
        return out
    labels = sorted(set(decided["observed"]) | set(decided["predicted"]))
    cm = confusion_matrix(decided["observed"], decided["predicted"], labels=labels)
    out.update({
        "accuracy": float((decided["observed"] == decided["predicted"]).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(decided["observed"], decided["predicted"])),
        "macro_f1": float(f1_score(decided["observed"], decided["predicted"], average="macro", zero_division=0)),
        "confusion": pd.DataFrame(cm, index=pd.Index(labels, name="observed"), columns=pd.Index(labels, name="predicted")),
        "per_class_recall": decided.assign(correct=decided["observed"] == decided["predicted"])
                                   .groupby("observed")["correct"].mean().rename("recall").reset_index(),
        "misclassified": decided[decided["observed"] != decided["predicted"]],
        "message": "Grouped leave-one-family-out; abstentions are excluded from accuracy and reported separately.",
    })
    return out


def label_leakage_warning(mapping, role):
    col = getattr(mapping, role, None)
    if col and col in mapping.text_columns:
        return f"The '{col}' column is part of the embedded text, so neighbours can match on the label itself. Remove it from the embedding text before relying on this."
    return None
