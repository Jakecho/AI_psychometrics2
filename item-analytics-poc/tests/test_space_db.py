"""The dedicated space database: save, search, update, remove/restore, delete/purge, conflicts and backup."""

import json
import sqlite3

import pytest

from src import decomposition as dc
from src import space_db as sdb
from src import task_alignment as ta
from tests.test_decomposition import bank  # noqa: F401 - fixture


@pytest.fixture
def db(tmp_path):
    return sdb.SpaceDB(tmp_path / "spaces.sqlite")


def build_space(by_id, review):
    store = dc.new_store()
    d = dc.add_feature(store, by_id, "declarative_key", "Tachycardia signals compromise", "sme",
                       supporting_items=["K-1", "K-2"], rationale="Both items depend on it.", importance=5)
    p = dc.add_feature(store, by_id, "procedural_key", "Stabilize first", "sme", supporting_items=["K-1"],
                       excerpt="Which action is first?", dominance="dominant", sequence_order=1)
    i = dc.add_feature(store, by_id, "instance_context", "Heart rate 125 bpm", "sme", supporting_items=["K-1"], excerpt="125 bpm")
    x = dc.add_feature(store, by_id, "distractor_feature", "Waiting delays care", "sme", option_refs=[("K-1", "C")],
                       misconception="Underestimates urgency (hypothesis).")
    return store, {"D": d, "P": p, "I": i, "X": x}


def test_save_search_and_reload_a_space(bank, db):
    by_id, review = bank
    store, ids = build_space(by_id, review)
    sid, version = db.save_space("Team A", review, store, by_id["K-0"], ta.accepted_siblings(review), "sme", source_name="bank.csv")
    assert version == 1
    spaces = db.spaces()
    row = spaces.iloc[0]
    assert (row["task_name"], row["domain"], row["siblings_N"]) == ("Prioritize", "Acute", 3)
    assert (row["Declarative"], row["Procedural"], row["Instance"], row["Distractor"]) == (1, 1, 1, 1)
    els = db.elements()
    assert len(els) == 4 and set(els["space_type"]) == {"Declarative", "Procedural", "Instance", "Distractor"}
    decl = els[els["label"] == "Tachycardia signals compromise"].iloc[0]
    assert decl["coverage"] == "2/3 = 67%" and decl["importance"] == 5 and decl["origin"] == "SME-added"
    loaded = db.get_space(sid)
    assert loaded["space"]["sibling_ids"] == ["K-1", "K-2", "K-3"]
    spans = next(e for e in loaded["elements"] if e["feature_id"] == ids["P"])["source_spans"]
    assert spans[0]["excerpt"] == "Which action is first?"


def test_search_filters_by_task_domain_type_status_and_keyword(bank, db):
    by_id, review = bank
    store, _ = build_space(by_id, review)
    db.save_space("Team A", review, store, by_id["K-0"], ta.accepted_siblings(review), "sme")
    other = ta.new_review("K-2", ["K-1"])
    other["task"] = {"name": "Interpret findings", "decision_or_action": "", "domain": "Renal", "rationale": ""}
    ta.set_task(other, "sme", "accepted")
    store2 = dc.new_store()
    dc.add_feature(store2, by_id, "instance_context", "Low pressure 50% off", "sme", supporting_items=["K-2"], rationale="r")
    db.save_space("Team B", other, store2, by_id["K-2"], ["K-1"], "sme")
    assert db.facets()["tasks"] == ["Interpret findings", "Prioritize"] and db.facets()["workspaces"] == ["Team A", "Team B"]
    assert db.elements(tasks=["Interpret findings"])["label"].tolist() == ["Low pressure 50% off"]
    assert len(db.elements(domains=["Acute"])) == 4
    assert db.elements(space_types=["distractor_feature"])["label"].tolist() == ["Waiting delays care"]
    assert db.elements(text="urgency")["label"].tolist() == ["Waiting delays care"]  # misconception text
    assert db.elements(text="50%")["label"].tolist() == ["Low pressure 50% off"]  # LIKE wildcards are escaped
    assert len(db.elements(text="%")) == 1
    assert db.spaces(workspace="Team B")["task_name"].tolist() == ["Interpret findings"]
    assert db.elements(statuses=["accepted"], workspace="Team A").shape[0] == 4


def test_update_remove_restore_with_history_and_version_checks(bank, db):
    by_id, review = bank
    store, ids = build_space(by_id, review)
    sid, v = db.save_space("Team A", review, store, by_id["K-0"], ta.accepted_siblings(review), "sme")
    eid = f"{sid}:{ids['D']}"
    v = db.update_element(eid, {"label": "Tachycardia indicates compromise", "importance": 4, "note": "reworded"}, "lead", v)
    assert v == 2
    with pytest.raises(sdb.ConflictError):
        db.update_element(eid, {"importance": 3}, "other", 1)  # stale version
    with pytest.raises(sdb.SpaceDBError, match="procedural"):
        db.update_element(eid, {"dominance": "dominant"}, "lead", v)
    with pytest.raises(sdb.SpaceDBError, match="1–5"):
        db.update_element(eid, {"importance": 7}, "lead", v)
    with pytest.raises(sdb.SpaceDBError, match="Remove/Restore"):
        db.update_element(eid, {"status": "removed"}, "lead", v)
    v = db.update_element(f"{sid}:{ids['P']}", {"dominance": "alternative", "sequence_order": 2}, "lead", v)
    xid = f"{sid}:{ids['X']}"
    assert db.set_removed([xid], True, "lead", {sid: v}, reason="weak") == [xid]
    assert xid not in set(db.elements()["element_id"])
    assert db.elements(include_removed=True).set_index("element_id").loc[xid, "status"] == "removed"
    v = db.spaces().iloc[0]["version"]
    assert db.set_removed([xid], False, "lead", {sid: v}) == [xid]
    assert db.elements().set_index("element_id").loc[xid, "status"] == "accepted"
    actions = db.history(sid)["action"].tolist()
    assert {"create_space", "update_element", "remove_element", "restore_element"} <= set(actions)
    before = json.loads(db.history(sid).query("action == 'update_element'").iloc[-1]["before"])
    assert before["label"] == "Tachycardia signals compromise"


def test_resave_updates_in_place_and_detects_conflicting_edits(bank, db):
    by_id, review = bank
    store, ids = build_space(by_id, review)
    sib = ta.accepted_siblings(review)
    sid, v = db.save_space("Team A", review, store, by_id["K-0"], sib, "sme")
    dc.update_feature(store, ids["I"], {"sme_importance": 2}, "sme")
    dc.remove_features(store, [ids["X"]], "sme")
    sid2, v2 = db.save_space("Team A", review, store, by_id["K-0"], sib, "sme", space_id=sid, expected_version=v)
    assert (sid2, v2) == (sid, 2) and len(db.spaces()) == 1
    els = db.elements(include_removed=True).set_index("feature_id")
    assert els.loc[ids["I"], "importance"] == 2 and els.loc[ids["X"], "status"] == "removed"
    db.update_element(f"{sid}:{ids['D']}", {"note": "edited in the library"}, "lead", v2)
    with pytest.raises(sdb.ConflictError):
        db.save_space("Team A", review, store, by_id["K-0"], sib, "sme", space_id=sid, expected_version=v2)
    copy_id, _ = db.save_space("Team A", review, store, by_id["K-0"], sib, "sme")  # "save as new"
    assert copy_id != sid and len(db.spaces()) == 2
    assert db.elements().set_index("element_id").loc[f"{sid}:{ids['D']}", "note"] == "edited in the library"


def test_delete_restore_purge_and_saving_requires_confirmed_task(bank, db):
    by_id, review = bank
    store, _ = build_space(by_id, review)
    unconfirmed = ta.new_review("K-0", ["K-1"])
    with pytest.raises(sdb.SpaceDBError, match="Confirm the target task"):
        db.save_space("Team A", unconfirmed, store, by_id["K-0"], [], "sme")
    with pytest.raises(sdb.SpaceDBError, match="no elements"):
        db.save_space("Team A", review, dc.new_store(), by_id["K-0"], [], "sme")
    sid, v = db.save_space("Team A", review, store, by_id["K-0"], ta.accepted_siblings(review), "sme")
    with pytest.raises(sdb.SpaceDBError, match="Delete the space first"):
        db.purge_space(sid, "lead", confirm=True)
    v = db.set_space_deleted(sid, True, "lead", v)
    assert db.spaces().empty and len(db.spaces(include_deleted=True)) == 1 and db.elements().empty
    v = db.set_space_deleted(sid, False, "lead", v)
    assert len(db.spaces()) == 1
    db.set_space_deleted(sid, True, "lead", v)
    with pytest.raises(sdb.SpaceDBError, match="Confirm"):
        db.purge_space(sid, "lead")
    db.purge_space(sid, "lead", confirm=True)
    assert db.spaces(include_deleted=True).empty
    with sqlite3.connect(db.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM elements").fetchone()[0] == 0
    assert db.history(sid).iloc[0]["action"] == "purge_space"  # tombstone kept


def test_backup_roundtrip_and_update_space_metadata(bank, db, tmp_path):
    by_id, review = bank
    store, _ = build_space(by_id, review)
    sid, v = db.save_space("Team A", review, store, by_id["K-0"], ta.accepted_siblings(review), "sme")
    v = db.update_space(sid, {"domain": "Emergency", "notes": "pilot"}, "lead", v)
    with pytest.raises(sdb.SpaceDBError):
        db.update_space(sid, {"task_name": " "}, "lead", v)
    backup = db.export_json()
    other = sdb.SpaceDB(tmp_path / "other.sqlite")
    assert other.import_json(backup, "lead") == {"added": 1, "replaced": 0, "skipped": 0}
    assert other.import_json(backup, "lead") == {"added": 0, "replaced": 0, "skipped": 1}
    assert other.import_json(backup, "lead", on_conflict="replace")["replaced"] == 1
    assert other.spaces().iloc[0]["domain"] == "Emergency" and len(other.elements()) == 4
    with pytest.raises(sdb.SpaceDBError):
        other.import_json({"format": "something else"}, "lead")
    assert "sk-" not in json.dumps(backup)


def test_database_is_off_on_hosted_deployments_unless_configured(tmp_path):
    path, reason = sdb.resolve_path(env={}, hosted=True)
    assert path is None and "shared" in reason
    path, _ = sdb.resolve_path(env={sdb.ENV_VAR: str(tmp_path / "x.sqlite")}, hosted=True)
    assert path == tmp_path / "x.sqlite"
    assert sdb.resolve_path(env={sdb.ENV_VAR: "off"}, hosted=False)[0] is None
    assert sdb.resolve_path(env={}, hosted=False)[0] == sdb.DEFAULT_PATH


def test_database_editor_changes_match_by_element_id():
    before = [{"element_id": "s:D001", "label": "a", "definition": "", "status": "accepted", "importance": float("nan"),
               "dominance": None, "procedure_group": None, "sequence_order": float("nan"), "note": "", "misconception": None,
               "space_version": 3},
              {"element_id": "s:X001", "label": "x", "definition": "", "status": "removed", "importance": 2.0,
               "dominance": None, "procedure_group": None, "sequence_order": None, "note": "", "misconception": "m",
               "space_version": 3}]
    after = [dict(before[1], status="accepted", note="n"), dict(before[0], importance="4", label="a2")]
    assert sdb.editor_changes(before, after) == {"s:X001": ({"note": "n"}, 3),
                                                 "s:D001": ({"label": "a2", "importance": 4}, 3)}


def test_editor_changes_to_several_elements_of_one_space_chain_versions(bank, db):
    by_id, review = bank
    store, ids = build_space(by_id, review)
    sid, v = db.save_space("Team A", review, store, by_id["K-0"], ta.accepted_siblings(review), "sme")
    changes = {f"{sid}:{ids['D']}": ({"importance": 3}, v), f"{sid}:{ids['I']}": ({"note": "context only"}, v),
               f"{sid}:{ids['X']}": ({"label": "Delaying care"}, v)}
    applied, errors = sdb.apply_editor_changes(db, changes, "lead")
    assert errors == [] and len(applied) == 3
    assert db.spaces().iloc[0]["version"] == v + 3
    stale = {f"{sid}:{ids['D']}": ({"importance": 1}, v)}  # someone else's older read
    applied, errors = sdb.apply_editor_changes(db, stale, "other")
    assert applied == [] and "changed since you opened it" in errors[0]
    assert db.elements(statuses=["accepted"]).shape[0] == 4
