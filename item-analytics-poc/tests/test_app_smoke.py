"""Streamlit AppTest smoke and end-to-end checks in offline mock mode."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import src.decomposition
import src.embedding
import src.ingest
import src.retrieval

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
SECTIONS = ["Import", "Search", "Decomposition", "Distractors", "Space Library", "Embedding Visualization",
            "Difficulty", "Classification", "Enemy Candidates", "Exports"]


def new_app():
    return AppTest.from_file(APP, default_timeout=120)


def click(at, label):
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, at.exception


def goto(at, section):
    at.radio(key="section").set_value(section).run()
    assert not at.exception, at.exception


def load_demo(at):
    at.run()
    click(at, "Load synthetic demo bank")
    click(at, "Apply mapping")
    click(at, "Generate embeddings")
    return at


def test_every_section_renders_with_no_data_and_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = new_app()
    at.run()
    assert not at.exception
    for section in SECTIONS:
        goto(at, section)


def test_end_to_end_mock_workflow():
    at = load_demo(new_app())
    assert len(at.session_state["records"]) == 12
    assert at.session_state["index"].valid.all()
    goto(at, "Search")
    at.selectbox(key="target_pick").set_value("S-01").run()
    at.radio(key="search_mode").set_value("Top K").run()
    click(at, "Search")
    results = at.session_state["search"]["results"]
    assert len(results) == 5 and "S-01" not in set(results["item_id"])
    assert any("illustrative" in (s.help or "") for s in at.slider)

    goto(at, "Decomposition")
    click(at, "Propose target task and candidate alignment")
    click(at, "Accept task")
    review = at.session_state["task_review"]
    assert review["task_status"] in ("accepted", "edited")
    for cid in review["candidate_ids"]:
        at.radio(key=f"align_S-01_{cid}").set_value("same_task" if cid != "S-08" else "different_task").run()
    sib = [c for c in at.session_state["task_review"]["candidate_ids"] if c != "S-08"]
    click(at, "Extract features")
    store = at.session_state["store"]
    assert store["features"] and all(f["review_status"] in ("pending", "needs_adjudication") for f in store["features"].values())
    assert any(f["canonical_label"] == "Prioritize the first action" for f in store["features"].values())

    for section in SECTIONS[3:]:
        goto(at, section)
    goto(at, "Difficulty")
    assert any("not an IRT calibration" in w.value for w in at.warning)
    goto(at, "Enemy Candidates")
    click(at, "Screen for candidates")
    table = at.session_state["enemy_table"]
    assert "S-02|S-05" in set(table["pair_id"])
    goto(at, "Exports")
    assert not at.exception
    assert set(sib) <= set(at.session_state["task_review"]["decisions"])


def test_feature_edits_persist_across_reruns_and_sections():
    at = load_demo(new_app())
    goto(at, "Search")
    click(at, "Search")
    goto(at, "Decomposition")
    click(at, "Propose target task and candidate alignment")
    click(at, "Accept task")
    click(at, "Extract features")
    store = at.session_state["store"]
    fid = store["order"][0]
    src.decomposition.update_feature(store, fid, {"review_status": "accepted", "sme_importance": 4}, "sme")
    goto(at, "Embedding Visualization")
    goto(at, "Decomposition")
    at.run()
    f = at.session_state["store"]["features"][fid]
    assert (f["review_status"], f["sme_importance"]) == ("accepted", 4)
    click(at, "Extract features")  # rerun never overwrites the reviewed feature
    f = at.session_state["store"]["features"][fid]
    assert (f["review_status"], f["sme_importance"]) == ("accepted", 4)


def test_two_users_are_isolated(tmp_path):
    a = load_demo(new_app())
    other = pd.DataFrame({"item_id": ["B-1", "B-2", "B-3"], "stem": ["Private one?", "Private two?", "Private three?"]})
    b = new_app()
    b.run()
    assert b.session_state["records"] is None and b.session_state["index"] is None
    b.session_state["raw_df"] = other
    b.session_state["mapping"] = src.ingest.detect_mapping(other)
    b.session_state["file_name"] = "user_b.csv"
    b.run()
    click(b, "Apply mapping")
    a.run()
    a_ids = {r["item_id"] for r in a.session_state["records"]}
    b_ids = {r["item_id"] for r in b.session_state["records"]}
    assert a_ids.isdisjoint(b_ids) and b_ids == {"B-1", "B-2", "B-3"}
    assert b.session_state["index"] is None and a.session_state["index"] is not None
    # no module-level container holds either user's content
    for module in (src.ingest, src.embedding, src.retrieval, src.decomposition):
        for value in vars(module).values():
            if isinstance(value, (list, dict, set)):
                text = repr(value)
                assert "Private one" not in text and "S-01" not in text


def test_openai_mode_blocks_requests_until_consent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real-0000000000")
    at = new_app()
    at.run()
    click(at, "Load synthetic demo bank")
    click(at, "Apply mapping")
    at.radio(key="llm_mode").set_value("OpenAI").run()
    button = next(b for b in at.button if b.label == "Generate embeddings")
    assert button.disabled
    assert all("sk-test" not in (m.value or "") for m in list(at.markdown) + list(at.caption) + list(at.text))


def test_sme_adds_and_removes_space_elements_in_the_app():
    at = load_demo(new_app())
    goto(at, "Search")
    click(at, "Search")
    goto(at, "Decomposition")
    click(at, "Propose target task and candidate alignment")
    click(at, "Accept task")
    click(at, "Extract features")
    before = set(at.session_state["store"]["features"])
    at.selectbox(key="add_type").set_value("instance_context").run()
    next(t for t in at.text_input if t.label == "Label").input("Fever value")
    next(m for m in at.multiselect if m.label == "Supporting items").set_value(["S-01"])
    next(t for t in at.text_input if t.label.startswith("Quote")).input("38.9 °C")
    click(at, "Add to space")
    store = at.session_state["store"]
    added = sorted(set(store["features"]) - before)
    assert len(added) == 1 and store["features"][added[0]]["origin"] == "sme"
    next(m for m in at.multiselect if m.label == "Elements to remove").set_value(added).run()
    click(at, "Remove from space")
    assert at.session_state["store"]["features"][added[0]]["review_status"] == "removed"
    goto(at, "Space Library")
    at.multiselect(key="lib_task").set_value(["Prioritize the first action"]).run()
    assert not at.exception
    shown = at.dataframe[-1].value
    assert len(shown) and set(shown["task"]) == {"Prioritize the first action"} and "removed" not in set(shown["status"])


def test_new_items_from_excel_dropdown_and_batch_search():
    from src import ingest
    at = load_demo(new_app())
    at.session_state["ni_df"] = ingest.read_table((ROOT / "samples" / "synthetic_queries.xlsx").read_bytes(), "q.xlsx")
    at.session_state["ni_name"] = "synthetic_queries.xlsx"
    at.radio(key="ni_how").set_value("Excel / CSV file").run()
    assert next(m for m in at.multiselect if m.label.startswith("Text columns")).value == ["vignette", "stem"]
    click(at, "Add 4 items")
    ni = at.session_state["new_items"]
    assert [r["item_id"] for r in ni["records"]] == ["NEW:QX-1", "NEW:QX-2", "NEW:QX-3", "NEW:QX-4"]
    assert set(ni["vectors"]) == {"NEW:QX-1", "NEW:QX-2", "NEW:QX-3", "NEW:QX-4"}  # embedded on add (mock)
    goto(at, "Search")
    at.radio(key="target_source").set_value("New item").run()
    at.selectbox(key="new_pick").set_value("NEW:QX-2").run()
    click(at, "Search")
    search = at.session_state["search"]
    assert search["target"]["item_id"] == "NEW:QX-2" and search["target"]["source_item_id"] == "QX-2"
    assert len(search["results"]) == 5 and not search["results"]["item_id"].str.startswith("NEW:").any()
    click(at, "Search all 4 new items")
    batch = at.session_state["query_batch"]["table"]
    assert sorted(batch["new_item_id"].unique()) == ["QX-1", "QX-2", "QX-3", "QX-4"]
    assert batch.groupby("new_item_id")["item_id"].count().eq(5).all()
    goto(at, "Decomposition")
    click(at, "Propose target task and candidate alignment")
    assert at.session_state["task_review"]["target_id"] == "NEW:QX-2"


def test_typed_new_items_accumulate_and_can_be_removed():
    at = load_demo(new_app())
    for text in ("A client has low blood pressure and confusion. Which action is first?",
                 "A client with diabetes asks about sick-day rules. Which statement shows understanding?"):
        at.text_area(key="ni_text").input(text)
        click(at, "Add item")
    ni = at.session_state["new_items"]
    assert [r["item_id"] for r in ni["records"]] == ["NEW:T001", "NEW:T002"] and len(ni["vectors"]) == 2
    next(m for m in at.multiselect if m.label == "Remove new items").set_value(["NEW:T001"]).run()
    click(at, "Remove")
    assert [r["item_id"] for r in at.session_state["new_items"]["records"]] == ["NEW:T002"]
    goto(at, "Search")
    at.radio(key="target_source").set_value("New item").run()
    click(at, "Search")
    assert at.session_state["search"]["target"]["item_id"] == "NEW:T002"




def test_spaces_are_saved_searched_updated_and_removed_in_the_database():
    import os
    from src import space_db as sdb
    at = load_demo(new_app())
    goto(at, "Search")
    click(at, "Search")
    goto(at, "Decomposition")
    click(at, "Propose target task and candidate alignment")
    click(at, "Accept task")
    for cid in at.session_state["task_review"]["candidate_ids"]:
        at.radio(key=f"align_S-01_{cid}").set_value("same_task" if cid != "S-08" else "different_task").run()
    click(at, "Extract features")
    store = at.session_state["store"]
    for fid in store["order"][:3]:
        src.decomposition.update_feature(store, fid, {"review_status": "accepted"}, "sme")
    at.text_input(key="db_workspace").input("Team A").run()
    click(at, "Save to database")
    link = at.session_state["store"]["db_link"]
    db = sdb.SpaceDB(os.environ["ITEM_ANALYTICS_DB_PATH"])
    saved = db.spaces()
    assert saved["space_id"].tolist() == [link["space_id"]] and saved.iloc[0]["workspace"] == "Team A"
    assert saved.iloc[0]["task_name"] == "Prioritize the first action" and saved.iloc[0]["siblings_N"] == 4
    click(at, "Update saved space")  # re-save in place with the version it holds
    assert at.session_state["store"]["db_link"]["version"] == 2 and len(db.spaces()) == 1
    goto(at, "Space Library")
    at.multiselect(key="db_task").set_value(["Prioritize the first action"]).run()
    at.text_input(key="db_text").input("Prioritize").run()
    assert not at.exception
    element_id = f"{link['space_id']}:{store['order'][0]}"
    next(m for m in at.multiselect if m.label == "Remove elements").set_value([element_id]).run()
    click(at, "Remove")
    assert db.elements(include_removed=True).set_index("element_id").loc[element_id, "status"] == "removed"
    # the session copy is now stale: updating from the session is refused, saving as new is allowed
    goto(at, "Decomposition")
    click(at, "Update saved space")
    assert any("changed since you opened it" in e.value for e in at.error)
    click(at, "Save as new space")
    assert len(db.spaces()) == 2
    goto(at, "Space Library")
    click(at, "Delete space")
    assert len(db.spaces()) == 1 and len(db.spaces(include_deleted=True)) == 2


def test_database_section_reports_when_disabled(monkeypatch):
    monkeypatch.setenv("ITEM_ANALYTICS_DB_PATH", "off")
    at = load_demo(new_app())
    goto(at, "Space Library")
    assert any("disabled by configuration" in i.value for i in at.info)
