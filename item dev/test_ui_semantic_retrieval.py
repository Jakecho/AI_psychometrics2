"""Retrieve, select, inspect the provider handoff and undo in both Item Studio entrypoints."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ai_adapter
import core
import specialist_workflow as workflow
import storage
import ui_semantic_retrieval as ui


APP_DIR = Path(__file__).resolve().parent


def test_replacing_a_collection_checks_archived_stages_and_removed_passages():
    bank = {"source_id": "example", "source_class": "bank", "source_date": "v1", "locator": "Example 1",
            "excerpt": "Approved bank record.", "providers": ["openai"]}
    saved = bank | {"item_ids": ["old"], "stage_ids": ["baseline"], "selected_by": "Original writer"}
    previous = workflow.retrieval_permission(saved | {"source_id": "removed"})
    clinical = workflow.retrieval_permission(saved | {"source_id": "clinical", "source_class": "practice summary",
                                                     "providers": ["anthropic"]})
    case = {"version": 6, "items": [], "archived_items": [
        {"id": "old", "decision_workflow": {"retrieved_sources": [saved]}},
    ], "retrieval_permissions": {"old": {"removed": previous, "clinical": clinical}}}
    assert ui._collection_restrictions(case, "bank", [bank]) == {("old", "removed"): set()}
    for record in (bank | {"stage_ids": ["later"]}, bank | {"item_ids": ["other"]},
                   bank | {"excerpt": "A different approved passage."}, bank | {"providers": []}):
        changes = ui._collection_restrictions(case, "bank", [record])
        assert changes == {("old", "example"): set(), ("old", "removed"): set()}
    committed = []
    ui._narrow_access(case, changes, "Reviewer", lambda value, reason, specialist: committed.append(value))
    assert committed[0]["retrieval_permissions"]["old"] == {
        "example": workflow.retrieval_permission(saved) | {"providers": []},
        "removed": previous | {"providers": []}, "clinical": clinical,
    }
    restricted = committed[0]["archived_items"][0]["decision_workflow"]["retrieved_sources"][0]
    assert restricted["providers"] == []
    assert restricted["selected_by"] == "Original writer"
    assert restricted["stage_ids"] == ["baseline"]
    assert saved["providers"] == ["openai"]

    # A restored, older snapshot with the same ID cannot acquire a newer revision's approval.
    newer = workflow.retrieval_permission(saved | {"excerpt": "A newer version of this example."})
    case["retrieval_permissions"]["old"]["example"] = newer
    assert ui._collection_restrictions(case, "bank", [bank]) == {("old", "example"): set(), ("old", "removed"): set()}
    ui._narrow_access(case, {("old", "example"): {"openai"}}, "Reviewer",
                      lambda value, reason, specialist: committed.append(value))
    assert committed[-1]["retrieval_permissions"]["old"]["example"] == newer | {"providers": []}
    assert committed[-1]["archived_items"][0]["decision_workflow"]["retrieved_sources"][0]["providers"] == []


@pytest.mark.parametrize("hosted", [False, True], ids=["local-folder", "hosted-approved-import"])
def test_writer_controls_selected_sources_and_provider_handoffs(tmp_path, monkeypatch, hosted):
    monkeypatch.setenv("ITEMDEV_DATA_DIR", str(tmp_path / "library"))
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    def no_network(*args, **kwargs):
        pytest.fail("Local retrieval and source selection must never call a provider.")
    monkeypatch.setattr(ai_adapter, "_http", no_network)
    case = core.new_case("standalone")
    case["author"] = "Synthetic reviewer"
    case["blueprint"]["objective"] = "Choose follow-up after a positive FIT."
    item = case["items"][0]
    item["decision_workflow"] = {"decision": "Evaluate an abnormal stool screening test", "setting": "Primary care",
                                 "population": "Adult", "assessment_intent": "Choose a follow-up investigation"}
    cid, iid = case["id"], item["id"]
    case["specialist_sources"] = [
        {"source_id": "manual-1", "source_date": "V1", "source_class": "bank", "locator": "Manual example",
         "excerpt": "A manually recorded fictional alternative.", "providers": ["openai"]},
    ]
    path = APP_DIR.parent / "deploy/item_studio/streamlit_app.py" if hosted else APP_DIR / "app.py"
    app = AppTest.from_file(str(path), default_timeout=30)
    if hosted:
        app.session_state["itemdev_hosted_library"] = {
            "schema_version": 1, "cases": {cid: case}, "history": {}, "redo": {}, "events": {},
        }
    else:
        storage.save_case(case)
    app.run()
    assert not app.exception
    prefix = f"{cid}::decision:{iid}:retrieval:"
    app.file_uploader(key=prefix + "files_bank").set_value(("invalid.json", b"{}", "application/json")).run()
    app.button(key=prefix + "load").click().run()
    assert app.error and not app.exception
    assert "manual-1" in app.text_area(key=f"{cid}::decision:{iid}:sources").value
    assert not app.session_state["itemdev_retrieval_catalog:" + cid]
    bank = [
        {"source_id": "bank-fit", "title": "Abnormal screening follow-up", "source_date": "Synthetic v1",
         "source_class": "bank", "locator": "Synthetic bank / 1", "excerpt": "Action after a positive FIT result.",
         "providers": ["openai"], "population": "Adult", "setting": "Primary care"},
    ]
    bank.append(deepcopy(bank[0]) | {"source_id": "bank-local", "title": "Local-only comparison",
                                     "excerpt": "LOCAL-ONLY positive FIT example.", "providers": []})
    app.file_uploader(key=prefix + "files_bank").set_value(
        ("bank.json", json.dumps(bank).encode(), "application/json")
    )
    app.multiselect(key=prefix + "grants_bankFalse").set_value(["openai"]).run()
    app.button(key=prefix + "load").click().run()
    assert not app.exception
    app.selectbox(key=prefix + "load_collection").set_value("clinical").run()

    note = b"""---
source_class: guideline
source_date: Synthetic version 2
citation: Fictional guideline, section 4
---
# Synthetic passage
Follow-up investigation after a positive fecal immunochemical test in adult primary care.
"""
    if hosted:
        assert not any(f.label == "Obsidian or Markdown folder" for f in app.text_input)
        app.file_uploader(key=prefix + "files_clinical").set_value(("followup.md", note, "text/markdown"))
    else:
        vault = tmp_path / "synthetic-vault"
        vault.mkdir()
        file = vault / "followup.md"
        file.write_bytes(note)
        (tmp_path / "outside.md").write_text("OUT-OF-SCOPE-PASSAGE positive FIT.")
        (vault / "escape.md").symlink_to(tmp_path / "outside.md")
        file.chmod(0o444)
        vault.chmod(0o555)
        app.radio(key=prefix + "origin").set_value("Local Markdown folder").run()
        app.text_input(key=prefix + "folder").set_value(str(vault))
    app.multiselect(key=prefix + "grants_clinical" + str(not hosted)).set_value(["anthropic"]).run()
    app.button(key=prefix + "load").click().run()
    assert not app.exception
    if not hosted:
        assert file.read_bytes() == note
        assert {p.name for p in vault.iterdir()} == {"followup.md", "escape.md"}
        vault.chmod(0o755)
    assert prefix + "results" not in app.session_state
    app.button(key=prefix + "search").click().run()
    assert not app.exception
    results = app.session_state[prefix + "results"]
    assert {r["source"]["source_id"] for r in results["rows"]["bank"]} == {"bank-fit", "bank-local"}
    assert results["rows"]["clinical"]
    assert "OUT-OF-SCOPE-PASSAGE" not in json.dumps(results)
    passage = next(r["source"] for r in results["rows"]["clinical"] if "immunochemical" in r["source"]["excerpt"])
    assert "followup.md" in passage["locator"]
    assert "Synthetic passage" in passage["locator"]

    def current():
        snapshot = app.session_state["itemdev_cases"][cid]
        return snapshot, workflow.item_for(snapshot, iid)

    snapshot, active = current()
    assert "retrieved_sources" not in active["decision_workflow"]
    pre = workflow.build_request(snapshot, active, "bank_retrieval", "openai")
    assert [r["item_id"] for r in pre["role_input"]["bank_candidates"]] == ["manual-1"]

    app.selectbox(key=prefix + "search_access").set_value("anthropic").run()
    assert not any(e.label == "Results to add to this item" for e in app.multiselect)
    app.button(key=prefix + "search").click().run()
    assert not app.session_state[prefix + "results"]["rows"]["bank"]
    app.selectbox(key=prefix + "search_access").set_value("local").run()
    app.button(key=prefix + "search").click().run()
    search_id = app.session_state[prefix + "results"]["id"]
    chosen = ["bank:bank-fit", "bank:bank-local", "clinical:" + passage["source_id"]]
    app.multiselect(key=prefix + "chosen_" + search_id).set_value(chosen).run()
    app.button(key=prefix + "save").click().run()
    assert not app.exception
    snapshot, active = current()
    selected = {s["source_id"] for s in active["decision_workflow"]["retrieved_sources"]}
    assert selected == {"bank-fit", "bank-local", passage["source_id"]}
    bank_handoff = workflow.build_request(snapshot, active, "bank_retrieval", "openai")
    assert {r["item_id"] for r in bank_handoff["role_input"]["bank_candidates"]} == {"bank-fit", "manual-1"}
    clinical_handoff = workflow.build_request(snapshot, active, "evidence_review", "anthropic")
    assert passage["excerpt"] in json.dumps(clinical_handoff)
    assert "LOCAL-ONLY" not in json.dumps([bank_handoff, clinical_handoff])
    assert "OUT-OF-SCOPE-PASSAGE" not in json.dumps([bank_handoff, clinical_handoff])
    assert passage["source_id"] not in json.dumps(workflow.build_request(snapshot, active, "evidence_review", "openai"))

    app.multiselect(key=prefix + "remove_ids").set_value(["bank-fit"]).run()
    app.button(key=prefix + "remove").click().run()
    assert not app.exception
    snapshot, active = current()
    assert "bank-fit" not in json.dumps(workflow.build_request(snapshot, active, "bank_retrieval", "openai"))
    app.button(key="undo:standalone").click().run()
    assert not app.exception
    snapshot, active = current()
    assert {s["source_id"] for s in active["decision_workflow"]["retrieved_sources"]} == selected

    # The writer can tighten one saved passage independently of its collection.
    sid = passage["source_id"]
    app.multiselect(key=prefix + "keep_" + sid).set_value([]).run()
    app.button(key=prefix + "restrict_" + sid).click().run()
    assert not app.exception
    snapshot, active = current()
    assert snapshot["retrieval_permissions"][iid][sid]["providers"] == []
    assert sid not in json.dumps(workflow.build_request(snapshot, active, "evidence_review", "anthropic"))

    # A replacement bank retains its source IDs but withdraws the collection grant.
    app.selectbox(key=prefix + "load_collection").set_value("bank").run()
    app.file_uploader(key=prefix + "files_bank").set_value(
        ("revised.json", json.dumps(bank).encode(), "application/json")
    )
    app.multiselect(key=prefix + "grants_bankFalse").set_value([]).run()
    app.button(key=prefix + "load").click().run()
    assert not app.exception
    snapshot, active = current()
    assert snapshot["retrieval_permissions"][iid]["bank-fit"]["providers"] == []
    retained = {s["source_id"]: s for s in active["decision_workflow"]["retrieved_sources"]}
    assert retained["bank-fit"]["selected_by"] == "Synthetic reviewer"
    assert retained["bank-fit"]["permission_review"]["actor"] == "Synthetic reviewer"
    assert retained["bank-fit"]["stage_ids"] == [item["stage_id"]]
    assert retained["bank-fit"]["excerpt"] == bank[0]["excerpt"]
    handoff = workflow.build_request(snapshot, active, "bank_retrieval", "openai")
    assert [r["item_id"] for r in handoff["role_input"]["bank_candidates"]] == ["manual-1"]
    app.selectbox(key=prefix + "search_access").set_value("openai").run()
    app.button(key=prefix + "search").click().run()
    assert not app.session_state[prefix + "results"]["rows"]["bank"]
    app.button(key="undo:standalone").click().run()
    snapshot, active = current()
    assert snapshot["retrieval_permissions"][iid]["bank-fit"]["providers"] == []
    assert "bank-fit" not in json.dumps(workflow.build_request(snapshot, active, "bank_retrieval", "openai"))

    # A reopened local draft or restored hosted library has no search index.
    reopened = AppTest.from_file(str(path), default_timeout=30)
    if hosted:
        reopened.session_state["itemdev_hosted_library"] = deepcopy(app.session_state["itemdev_hosted_library"])
    reopened.run()
    assert not reopened.exception
    assert not reopened.session_state["itemdev_retrieval_catalog:" + cid]
    stored = reopened.session_state["itemdev_cases"][cid]
    handoff = workflow.build_request(stored, workflow.item_for(stored, iid), "bank_retrieval", "openai")
    assert [r["item_id"] for r in handoff["role_input"]["bank_candidates"]] == ["manual-1"]
    assert storage.library_path().exists() is not hosted
