import numpy as np
import pandas as pd
import pytest

from src import enemies as en
from src import predictions as pr
from tests.conftest import make_item, manual_index, records_from_rows


def neighbours(pairs):
    return pd.DataFrame([{"item_id": i, "cosine_similarity": s} for i, s in pairs])


def by_id(rows, **kw):
    records, mapping, _ = records_from_rows(rows, **kw)
    return {r["item_id"]: r for r in records}, records, mapping


def test_difficulty_requires_labels_and_never_uses_p_value():
    ids, _, mapping = by_id([make_item("P-1", "a?", p_value="0.7"), make_item("P-2", "b?", p_value="0.4")])
    assert mapping.p_value == "p_value" and mapping.difficulty is None
    res = pr.difficulty_estimate(neighbours([("P-1", 0.9), ("P-2", 0.8)]), ids, scale_confirmed=True)
    assert res["status"] == "abstain" and res["n_labeled"] == 0 and res["estimate"] is None


def test_unlinked_scales_are_never_averaged_and_confirmed_scale_matches_hand_value():
    rows = [make_item("R-1", "a?", rasch_b="1.0", scale="FormA"), make_item("R-2", "b?", rasch_b="-1.0", scale="FormB")]
    ids, _, _ = by_id(rows)
    res = pr.difficulty_estimate(neighbours([("R-1", 0.9), ("R-2", 0.8)]), ids, scale_confirmed=True)
    assert res["status"] == "non_comparable" and res["estimate"] is None and "unweighted_mean" not in res
    rows = [make_item("R-1", "a?", rasch_b="1.0", scale="FormA"), make_item("R-2", "b?", rasch_b="-1.0", scale="FormA"),
            make_item("R-3", "c?", rasch_b="3.0", scale="FormA")]
    ids, _, _ = by_id(rows)
    nb = neighbours([("R-1", 0.9), ("R-2", 0.6), ("R-3", -0.2)])
    unconfirmed = pr.difficulty_estimate(nb, ids, scale_confirmed=False)
    assert unconfirmed["status"] == "exploratory" and unconfirmed["estimate"] is None
    res = pr.difficulty_estimate(nb, ids, alpha=1.0, scale_confirmed=True)
    assert res["status"] == "estimate"
    assert res["estimate"] == pytest.approx((0.9 * 1.0 + 0.6 * -1.0) / 1.5)  # negative cosine gets weight 0
    assert res["unweighted_mean"] == pytest.approx(1.0) and res["min"] == -1.0 and res["max"] == 3.0
    zero = pr.difficulty_estimate(neighbours([("R-1", -0.1), ("R-2", 0.0)]), ids, scale_confirmed=True)
    assert zero["status"] == "abstain"


def test_grouped_holdout_prevents_family_leakage_and_reports_baseline():
    # Families of near-identical items share a difficulty; leave-one-out without grouping would be perfect.
    rows, vecs = [], []
    for fam, b in enumerate([-2.0, -1.0, 0.5, 1.0, 2.0]):
        for j in range(2):
            rows.append(make_item(f"F{fam}-{j}", f"family {fam} item {j}?", rasch_b=str(b), family=f"fam{fam}"))
            vecs.append([np.cos(fam), np.sin(fam), 0.001 * j])
    records, _, _ = records_from_rows(rows)
    idx = manual_index([r["item_id"] for r in records], vecs)
    out = pr.evaluate_difficulty(idx, records, k=2, min_items=6)
    assert out["status"] == "ok" and out["n_groups"] == 5
    table = out["table"]
    assert (table["knn_prediction"] != table["observed"]).all()  # no prediction used its own family
    assert set(out["summary"]["method"]) == {"weighted kNN", "naive mean baseline"}
    assert {"MAE", "RMSE", "bias"} <= set(out["summary"].columns)


def test_classification_abstains_and_handles_multi_label():
    rows = [make_item("L-1", "a?", domain="Cardio"), make_item("L-2", "b?", domain="Renal"),
            make_item("L-3", "c?", domain="Cardio; Renal"), make_item("L-4", "d?", domain="")]
    ids, _, _ = by_id(rows)
    tie = pr.classify(neighbours([("L-1", 0.8), ("L-2", 0.8)]), ids, margin=0.1)
    assert tie["status"] == "abstain" and tie["prediction"] is None
    multi = pr.classify(neighbours([("L-1", 0.9), ("L-3", 0.8)]), ids)
    assert multi["status"] == "multi_label" and multi["prediction"] is None
    none = pr.classify(neighbours([("L-4", 0.9)]), ids)
    assert none["status"] == "abstain"
    clear = pr.classify(neighbours([("L-1", 0.9), ("L-1", 0.9), ("L-2", 0.2)]), ids, margin=0.1)
    assert clear["status"] == "prediction" and clear["prediction"] == "Cardio"


def test_classification_holdout_metrics_and_leakage_warning():
    rows, vecs = [], []
    for n in range(8):
        rows.append(make_item(f"C-{n}", f"item {n}?", domain="A" if n < 4 else "B"))
        v = np.zeros(10)
        v[0 if n < 4 else 1] = 1.0
        v[2 + n] = 0.9  # distinct within class (cosine 0.55), so items are not near-duplicate families
        vecs.append(v)
    records, mapping, _ = records_from_rows(rows)
    idx = manual_index([r["item_id"] for r in records], vecs)
    out = pr.evaluate_classification(idx, records, role="domain", k=3, margin=0.0)
    assert out["status"] == "ok" and out["accuracy"] == 1.0
    assert out["balanced_accuracy"] == 1.0 and out["macro_f1"] == 1.0
    assert list(out["confusion"].index) == ["A", "B"]
    assert pr.label_leakage_warning(mapping, "domain") is None
    mapping.text_columns = ["stem", "domain"]
    assert "embedded text" in pr.label_leakage_warning(mapping, "domain")


def test_enemy_flags_are_candidates_with_evidence():
    rows = [make_item("E-1", "A 70-year-old has BP 80/40 mmHg and HR 120 bpm. What first?", ("Give fluids now please", "b", "c")),
            make_item("E-2", "A 70-year-old has BP 80/40 mmHg and HR 120 bpm. What first?", ("Give fluids now please", "b", "c")),
            make_item("E-3", "Explain why you would give fluids now please to this patient?", ("x", "y")),
            make_item("E-4", "Stimulus part one?", ("p", "q"), family="CASE-1"),
            make_item("E-5", "Stimulus part two?", ("r", "s"), family="CASE-1", enemy="")]
    records, _, _ = records_from_rows(rows)
    table = en.enemy_candidates(records, None)
    pairs = dict(zip(table["pair_id"], table["triggers"]))
    assert "exact_duplicate_text" in pairs["E-1|E-2"] and "option_overlap" in pairs["E-1|E-2"]
    assert "shared_precise_data" in pairs["E-1|E-2"]
    assert "answer_leakage" in pairs["E-1|E-3"]
    linked = table[table["pair_id"] == "E-4|E-5"].iloc[0]
    assert linked["triggers"] == "same_linked_set" and linked["flag_status"].startswith("linked set")
    assert not table["flag_status"].str.contains("confirmed").any()
    decisions = {}
    en.record_decision(decisions, "E-2|E-1", "enemy_pair", "sme", "duplicate")
    assert "E-1|E-2" in decisions and decisions["E-1|E-2"]["reviewer"] == "sme"
    with pytest.raises(ValueError):
        en.record_decision(decisions, "E-1|E-2", "auto_confirmed", "sme")


def test_existing_enemy_metadata_and_cosine_trigger():
    rows = [make_item("M-1", "first?", enemy="M-2"), make_item("M-2", "second?"), make_item("M-3", "third?")]
    records, _, _ = records_from_rows(rows)
    idx = manual_index(["M-1", "M-2", "M-3"], [[1, 0], [0, 1], [1, 0.05]])
    table = en.enemy_candidates(records, idx, cos_threshold=0.95)
    pairs = dict(zip(table["pair_id"], table["triggers"]))
    assert "existing_enemy_metadata" in pairs["M-1|M-2"]
    assert "high_cosine" in pairs["M-1|M-3"]
