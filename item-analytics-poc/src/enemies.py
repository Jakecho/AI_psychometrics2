"""Enemy-item *candidate* flags with independent rule triggers and evidence.

Nothing here certifies a true enemy pair. Every flag is a candidate for SME
review; intentionally linked items (shared stimulus/family) are reported as
linked sets rather than enemies.
"""

from datetime import datetime, timezone
import re

import numpy as np
import pandas as pd

from .llm import _VALUE

ENEMY_DECISIONS = ("unreviewed", "enemy_pair", "not_enemy", "intentional_linked_set", "needs_discussion")
DEFAULTS = {"cos_threshold": 0.92, "stem_jaccard": 0.8, "option_jaccard": 0.5, "shared_values_min": 2,
            "candidate_floor": 0.6}


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/.%-]", " ", (text or "").lower())).strip()


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+(?:[./][a-z0-9]+)*", (text or "").lower()))


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _prepare(r):
    text = " ".join(v for v in r["fields"].values() if v)
    enemies = {e.strip() for e in re.split(r"[,;|\s]+", str(r["roles"].get("enemy") or "")) if e.strip()}
    return {
        "id": r["item_id"], "norm": _norm(text), "tokens": _tokens(text),
        "options": {_norm(o["text"]) for o in r["options"] if o["text"]},
        "keys": [_norm(o["text"]) for o in r["options"] if o["is_key"]],
        "values": {m.group(0).lower().replace(" ", "") for m in _VALUE.finditer(text)},
        "family": r["roles"].get("family"), "enemies": enemies,
    }


def enemy_candidates(records, index=None, scope_ids=None, **params):
    p = {**DEFAULTS, **params}
    scope = [r for r in records if scope_ids is None or r["item_id"] in set(scope_ids)]
    prep = [_prepare(r) for r in scope]
    sims = None
    if index is not None:
        pos = [index.position(r["item_id"]) for r in scope]
        if all(x is not None for x in pos):
            vec = index.vectors[pos]
            sims = vec @ vec.T
            valid = index.valid[pos]
            sims[~valid, :] = np.nan
            sims[:, ~valid] = np.nan
    if sims is None and len(scope) > 400:
        raise ValueError("Generate embeddings first, or narrow the scope to 400 items or fewer for lexical-only screening.")
    rows = []
    for i in range(len(prep)):
        a = prep[i]
        for j in range(i + 1, len(prep)):
            b = prep[j]
            cos = float(sims[i, j]) if sims is not None and np.isfinite(sims[i, j]) else None
            flagged_meta = b["id"] in a["enemies"] or a["id"] in b["enemies"]
            linked = a["family"] not in (None, "") and a["family"] == b["family"]
            if not (flagged_meta or linked or a["norm"] == b["norm"]
                    or cos is None or cos >= p["candidate_floor"]):
                continue
            triggers, evidence, concerns = [], [], set()
            stem_j = _jaccard(a["tokens"], b["tokens"])
            opt_j = _jaccard(a["options"], b["options"]) if len(a["options"]) > 1 and len(b["options"]) > 1 else 0.0
            if a["norm"] == b["norm"]:
                triggers.append("exact_duplicate_text")
                concerns.add("duplication")
            elif stem_j >= p["stem_jaccard"]:
                triggers.append("near_duplicate_text")
                concerns.add("duplication")
                evidence.append(f"stem token overlap {stem_j:.2f}")
            if cos is not None and cos >= p["cos_threshold"]:
                triggers.append("high_cosine")
                concerns.add("semantic_similarity")
                evidence.append(f"cosine {cos:.3f}")
            if opt_j >= p["option_jaccard"]:
                triggers.append("option_overlap")
                concerns.add("duplication")
                evidence.append(f"option-set overlap {opt_j:.2f}")
            shared = a["values"] & b["values"]
            if len(shared) >= p["shared_values_min"]:
                triggers.append("shared_precise_data")
                concerns.add("shared_case_data")
                evidence.append("shared values: " + ", ".join(sorted(shared)))
            for src, dst in ((a, b), (b, a)):
                for key in src["keys"]:
                    if len(key) >= 12 and key in dst["norm"]:
                        triggers.append("answer_leakage")
                        concerns.add("cueing_answer_disclosure")
                        evidence.append(f"key of {src['id']} appears in the text of {dst['id']}")
            if flagged_meta:
                triggers.append("existing_enemy_metadata")
                concerns.add("existing_metadata_flag")
                evidence.append("listed in the enemy column")
            if linked:
                triggers.append("same_linked_set")
                concerns.add("shared_stimulus_dependence")
                evidence.append(f"same family/stimulus '{a['family']}'")
            if not triggers:
                continue
            only_linked = set(triggers) == {"same_linked_set"}
            rows.append({
                "pair_id": "|".join(sorted((a["id"], b["id"]))), "item_a": a["id"], "item_b": b["id"],
                "cosine_similarity": cos, "stem_token_jaccard": round(stem_j, 3), "option_jaccard": round(opt_j, 3),
                "triggers": ", ".join(dict.fromkeys(triggers)), "concerns": ", ".join(sorted(concerns)),
                "evidence": "; ".join(dict.fromkeys(evidence)),
                "flag_status": "linked set — review intent" if only_linked else "candidate enemy (unverified)",
            })
    frame = pd.DataFrame(rows, columns=["pair_id", "item_a", "item_b", "cosine_similarity", "stem_token_jaccard",
                                        "option_jaccard", "triggers", "concerns", "evidence", "flag_status"])
    return frame.sort_values(["cosine_similarity", "pair_id"], ascending=[False, True], na_position="last").reset_index(drop=True)


def record_decision(decisions, pair_id, decision, reviewer, note=""):
    if decision not in ENEMY_DECISIONS:
        raise ValueError(f"Unknown decision '{decision}'.")
    a, b = pair_id.split("|", 1)
    key = "|".join(sorted((a, b)))
    history = decisions.get(key, {}).get("history", [])
    entry = {"decision": decision, "reviewer": reviewer or "anonymous", "note": note,
             "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    decisions[key] = {**entry, "history": history + [entry]}
    return decisions[key]
