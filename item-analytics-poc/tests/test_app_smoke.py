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
SECTIONS = ["Data & API", "Search", "Decomposition", "Distractors", "Embedding Visualization",
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
