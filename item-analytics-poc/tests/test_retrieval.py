import numpy as np

from src import retrieval as rt
from tests.conftest import make_item, manual_index, records_from_rows

# Hand-computed: q = (1, 0); cosines are a=1.0, b=0.8, c=0.6, d=-1.0, z invalid (zero vector).
IDS = ["c", "a", "d", "b", "z"]
VECS = [[0.6, 0.8], [2, 0], [-1, 0], [0.8, 0.6], [0, 0]]


def test_cosine_ordering_negative_and_zero_handling():
    idx = manual_index(IDS, VECS)
    scores = rt.cosine_scores(idx, np.array([3.0, 0.0]))
    assert np.allclose(scores[:4], [0.6, 1.0, -1.0, 0.8]) and np.isnan(scores[4])
    res = rt.select(scores, IDS, "top_k", k=10)
    assert list(res["item_id"]) == ["a", "b", "c", "d"]
    assert np.allclose(res["cosine_similarity"], [1.0, 0.8, 0.6, -1.0])


def test_self_exclusion_top_k_threshold_and_both():
    idx = manual_index(IDS, VECS)
    scores = rt.cosine_scores(idx, idx.vectors[1])
    assert list(rt.select(scores, IDS, "top_k", k=2, exclude_id="a")["item_id"]) == ["b", "c"]
    assert list(rt.select(scores, IDS, "threshold", threshold=0.85)["item_id"]) == ["a"]
    assert list(rt.select(scores, IDS, "threshold", threshold=0.85, exclude_id="a")["item_id"]) == []
    assert list(rt.select(scores, IDS, "both", k=1, threshold=0.5)["item_id"]) == ["a"]
    assert len(rt.select(scores, IDS, "top_k", k=50)) == 4  # never more than available valid items


def test_ties_are_stable_by_item_id():
    idx = manual_index(["m", "k", "x"], [[1, 0], [1, 0], [0, 1]])
    scores = rt.cosine_scores(idx, np.array([1.0, 0.0]))
    assert list(rt.select(scores, idx.item_ids, "top_k", k=3)["item_id"]) == ["k", "m", "x"]


def test_top_k_counts_unique_items_and_options_are_preserved():
    rows = [make_item(f"U-{i}", f"Stem {i}?", tuple(f"opt{i}{j}" for j in range(6)), key="A") for i in range(8)]
    records, _, _ = records_from_rows(rows)
    vectors = [[1.0, i / 10] for i in range(8)]
    idx = manual_index([r["item_id"] for r in records], vectors)
    res = rt.select(rt.cosine_scores(idx, np.array([1.0, 0.0])), idx.item_ids, "top_k", k=5)
    assert len(res) == 5 and res["item_id"].is_unique
    by_id = {r["item_id"]: r for r in records}
    assert all(len(by_id[i]["options"]) == 6 for i in res["item_id"])  # five distractors, one result slot each


def test_semantic_match_survives_poor_bm25_and_threshold_uses_cosine_only():
    # "sem" is 0.86 cosine but shares no words with the query; "lex" shares words but is 0.5 cosine.
    ids = ["sem", "lex"]
    scores = np.array([0.86, 0.5])
    docs = ["completely different wording here", "chest pain first action nurse"]
    bm = rt.BM25(docs).scores("chest pain first action")
    assert bm[0] == 0 and bm[1] > 0
    semantic = rt.select(scores, ids, "threshold", threshold=0.85)
    assert list(semantic["item_id"]) == ["sem"]
    hybrid = rt.hybrid_search(scores, bm, ids, "threshold", threshold=0.85)
    assert list(hybrid["item_id"]) == ["sem"]
    assert (hybrid["hybrid_score"] < 0.85).all()  # RRF values are far below 0.85 yet the item is kept by cosine
    assert {"cosine_similarity", "bm25_score", "hybrid_score"} <= set(hybrid.columns)
    ranked = rt.hybrid_search(scores, bm, ids, "top_k", k=2)
    assert set(ranked["item_id"]) == {"sem", "lex"}


def test_tokenizer_keeps_units_ratios_and_hyphenated_forms():
    toks = rt.tokenize("BP 86/50 mmHg, post-operative day 2; pH 7.4")
    assert {"86/50", "mmhg", "post", "operative", "postoperative", "7.4"} <= set(toks)


def test_threshold_counts_describe_distribution():
    table = rt.threshold_counts([0.9, 0.86, 0.5, float("nan")], thresholds=(0.85, 0.5))
    assert table["items_at_or_above"].tolist() == [2, 3]
