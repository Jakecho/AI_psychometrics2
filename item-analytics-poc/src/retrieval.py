"""Exact cosine retrieval over unique items, with optional BM25 and RRF fusion.

Cosine similarity in the original embedding space is always the primary
ranking and the only score a minimum-similarity threshold applies to. BM25 and
RRF are separate, labelled columns.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

MODES = ("top_k", "threshold", "both")
DEFAULT_THRESHOLD = 0.85  # an illustrative, user-adjustable heuristic, not a sibling criterion


def cosine_scores(index, query_vector):
    """``index.vectors @ q`` on unit vectors. Invalid index rows score NaN."""
    q = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("The query vector is zero or non-finite.")
    scores = index.vectors @ (q / norm)
    scores = np.clip(scores.astype(np.float64), -1.0, 1.0)
    scores[~index.valid] = np.nan
    return scores


def select(scores, item_ids, mode="top_k", k=5, threshold=DEFAULT_THRESHOLD, exclude_id=None, allowed_ids=None):
    """Rank unique items by cosine (descending, ties by item ID) and apply the filter."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    seen = set()
    rows = []
    allowed = set(allowed_ids) if allowed_ids is not None else None
    for iid, s in zip(item_ids, scores):
        if iid in seen or iid == exclude_id or (allowed is not None and iid not in allowed) or not np.isfinite(s):
            continue
        seen.add(iid)
        rows.append((iid, float(s)))
    rows.sort(key=lambda r: (-r[1], str(r[0])))
    if mode in ("threshold", "both"):
        rows = [r for r in rows if r[1] >= threshold]
    if mode in ("top_k", "both"):
        rows = rows[:max(0, min(int(k), len(rows)))]
    return pd.DataFrame(
        [{"rank": i + 1, "item_id": iid, "cosine_similarity": s} for i, (iid, s) in enumerate(rows)],
        columns=["rank", "item_id", "cosine_similarity"],
    )


def threshold_counts(scores, thresholds=(0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)):
    finite = np.asarray([s for s in scores if np.isfinite(s)])
    return pd.DataFrame({"minimum_cosine": thresholds,
                         "items_at_or_above": [int((finite >= t).sum()) for t in thresholds]})


# --------------------------------------------------------------------------- BM25

_TOKEN = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")


def tokenize(text):
    """Lowercase tokens; keeps numbers with units/ratios (e.g. 82/50, 7.4) and splits hyphenated words
    while also keeping the joined form (e.g. 'lead-in' -> lead, in, leadin)."""
    text = (text or "").lower()
    tokens = []
    for raw in re.findall(r"[a-z0-9./-]+", text):
        parts = [p for p in raw.split("-") if p]
        for p in parts:
            tokens.extend(_TOKEN.findall(p))
        if len(parts) > 1:
            tokens.append("".join(parts))
    return tokens


class BM25:
    """Okapi BM25 over a small in-memory corpus."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.asarray([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        if not q or self.avgdl == 0:
            return out
        for i, d in enumerate(self.docs):
            denom_base = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + denom_base)
            out[i] = s
        return out


def ranks(values, item_ids):
    """1-based ranks, descending, ties broken by item ID."""
    order = sorted(range(len(values)), key=lambda i: (-values[i] if np.isfinite(values[i]) else math.inf, str(item_ids[i])))
    out = np.empty(len(values), dtype=int)
    for r, i in enumerate(order, start=1):
        out[i] = r
    return out


def hybrid_search(scores, bm25_scores, item_ids, mode="top_k", k=5, threshold=DEFAULT_THRESHOLD,
                  exclude_id=None, allowed_ids=None, k0=60, w_cosine=1.0, w_bm25=1.0):
    """RRF of separately generated cosine and BM25 ranks.

    ``hybrid_score`` is a ranking measure, not a similarity. The threshold, when
    used, is applied to the original cosine column only.
    """
    cos_rank = ranks(scores, item_ids)
    bm_rank = ranks(bm25_scores, item_ids)
    rrf = w_cosine / (k0 + cos_rank) + w_bm25 / (k0 + bm_rank)
    allowed = set(allowed_ids) if allowed_ids is not None else None
    rows = []
    seen = set()
    for i, iid in enumerate(item_ids):
        if iid in seen or iid == exclude_id or (allowed is not None and iid not in allowed) or not np.isfinite(scores[i]):
            continue
        seen.add(iid)
        if mode in ("threshold", "both") and scores[i] < threshold:
            continue
        rows.append({"item_id": iid, "cosine_similarity": float(scores[i]), "bm25_score": float(bm25_scores[i]),
                     "hybrid_score": float(rrf[i]), "cosine_rank": int(cos_rank[i]), "bm25_rank": int(bm_rank[i])})
    rows.sort(key=lambda r: (-r["hybrid_score"], str(r["item_id"])))
    if mode in ("top_k", "both"):
        rows = rows[:max(0, int(k))]
    frame = pd.DataFrame(rows, columns=["item_id", "cosine_similarity", "bm25_score", "hybrid_score",
                                        "cosine_rank", "bm25_rank"])
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame
