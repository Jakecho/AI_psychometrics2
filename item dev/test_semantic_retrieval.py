"""Offline collection, scope, and provenance checks using invented records only."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

import core
import semantic_retrieval as retrieval


@pytest.fixture
def query_case():
    case = core.new_case("unfolding")
    case["patient"] = "Private learner vignette outside retrieval context"
    case["blueprint"].update(domain="Synthetic training", subdomain="Investigation",
                             objective="Choose a workup for shortness of breath after a return visit.")
    stage = case["stages"][0]
    stage.update(title="Return visit", elapsed="One year")
    case["stages"][1]["new_information"] = "Future secret"
    item = case["items"][0]
    item["decision_workflow"] = {"decision": "Investigate breathlessness", "population": "Adult",
                                 "setting": "Family practice", "intended_key": "Private writer key"}
    item["options"][0].update(text="Private answer option", correct=True, option_type="Investigation")
    return case, item


@pytest.fixture
def bank():
    source = {"source_id": "bank-a", "source_class": "bank", "source_date": "Synthetic version 4",
              "locator": "Bank item 17, summary", "summary": "Synthetic case of dyspnea at reassessment.",
              "providers": ["openai", "anthropic"], "decision_timing": "Annual", "population": "Adult",
              "option_type": "Investigation", "setting": "Family practice",
              "blueprint": {"domain": "Educational cases", "objective": "Investigate dyspnea"},
              "options": [{"id": "17a", "text": "Synthetic investigation", "internal": "PRIVATE COMMENT"}],
              "observed_option_evidence": [{"item_version": "3", "population": "Synthetic cohort",
                                            "administration": "Simulated sample", "sample_size": 10,
                                            "denominator": 10, "selected": 4, "private_raw_record": "PRIVATE RAW"}]}
    return [source, source | {"source_id": "bank-b", "population": "Pediatric", "decision_timing": "Initial visit"},
            source | {"source_id": "bank-hidden", "stage_ids": ["later"]},
            source | {"source_id": "bank-unapproved", "approved": False}]


def test_both_collections_rank_paraphrases_and_preserve_provenance(query_case, bank, tmp_path, monkeypatch):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    case, item = query_case
    query = retrieval.build_query(case, item)
    assert query["stage_id"] == item["stage_id"]
    assert query["decision_timing"] == "One year"
    assert all(v not in str(query) for v in ("Private learner vignette", "Private writer key", "Future secret", "Private answer option"))
    records = retrieval.import_collection(json.dumps(bank), "approved_bank.json", "bank", approved_providers=["openai"])
    ranked = retrieval.search(records, query, "bank", provider="openai")
    assert [r["source"]["source_id"] for r in ranked] == ["bank-a", "bank-b"]
    best = ranked[0]
    assert best["similarity"] > 0 and best["score"] > ranked[1]["score"]
    assert any("dyspnea" in why for why in best["reasons"])
    assert any("Different population" in why for why in ranked[1]["reasons"])
    assert best["source"]["options"] == [{"option_id": "17a", "text": "Synthetic investigation"}]
    assert best["source"]["observed_option_evidence"][0]["sample_size"] == 10
    assert "PRIVATE" not in str(records)
    assert best["source"]["source_class"] == "bank" and best["source"]["source_date"] == "Synthetic version 4"
    assert best["source"]["locator"] == "Bank item 17, summary"
    assert retrieval.search(records, query, "bank", provider="anthropic") == []

    selected = tmp_path / "training references"
    selected.mkdir()
    body = """---
source_class: guideline
version: "Synthetic v2"
citation: "Invented demonstration guideline, section 3"
providers: [openai, anthropic]
population: Adult
decision_timing: Annual
---
# Invented teaching guide
## Subsequent visit
A synthetic adult describes dyspnea during reassessment. The fictitious task is diagnostic evaluation.
"""
    (selected / "Example.md").write_text(body)
    (selected / "Other population.md").write_text(body.replace("population: Adult", "population: Pediatric")
                                                 .replace("decision_timing: Annual", "decision_timing: Initial visit"))
    references = retrieval.load_markdown_folder(selected, approved_providers=["openai"])
    findings = retrieval.search(references, query, "clinical", provider="openai")
    assert [r["source"]["relative_file"] for r in findings] == ["Example.md", "Other population.md"]
    ref = findings[0]["source"]
    assert ref["source_class"] == "guideline" and ref["source_date"] == "Synthetic v2"
    assert ref["locator"] == "Example.md#Invented teaching guide / Subsequent visit"
    assert ref["citation"] == "Invented demonstration guideline, section 3"
    assert ref["providers"] == ["openai"] and str(tmp_path) not in str(ref)


def test_folder_boundary_and_read_only_access(tmp_path, monkeypatch):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    chosen = tmp_path / "approved"
    chosen.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("PRIVATE OUTSIDE PASSAGE")
    (chosen / "outside link.md").symlink_to(outside)
    (chosen / "other folder").symlink_to(tmp_path, target_is_directory=True)
    (chosen / ".obsidian").mkdir()
    (chosen / ".obsidian" / "plugin.md").write_text("PRIVATE PLUGIN TEXT")
    (chosen / "nonmarkdown.txt").write_text("PRIVATE NONMARKDOWN")
    note = chosen / "Note.md"
    note.write_text("# Approved heading\nSynthetic manually entered passage. ![[../outside]]")
    note.chmod(0o444)
    chosen.chmod(0o555)
    native_open = os.open
    modes = []
    def check_mode(name, flags, *args, **kwargs):
        modes.append(flags)
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        return native_open(name, flags, *args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "open", check_mode)
            rows = retrieval.load_markdown_folder(chosen)
        assert modes and len(rows) == 1
        assert rows[0]["source_class"] == "practice summary" and rows[0]["providers"] == []
        assert "PRIVATE" not in str(rows) and not any(r["source_id"] == "outside" for r in rows)
        assert note.read_text().startswith("# Approved") and (note.stat().st_mode & 0o777) == 0o444
        with pytest.raises(ValueError, match="readable"):
            retrieval.load_markdown_folder(chosen / "other folder")
    finally:
        chosen.chmod(0o755)
    monkeypatch.setenv("ITEMDEV_HOSTED", "1")
    with pytest.raises(ValueError, match="hosted"):
        retrieval.load_markdown_folder(chosen)


def test_imported_notes_match_local_snapshots_and_do_not_self_authorize(tmp_path, monkeypatch, query_case):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    text = """---
source_class: study
providers:
  - anthropic
version: illustration
---
# Synthetic comparator
Longer follow-up of dyspnea is assessed.
Citation: Invented Study https://example.test/fiction
"""
    file = tmp_path / "Study.md"
    file.write_text(text)
    local = retrieval.load_markdown_folder(tmp_path, approved_providers=["openai", "anthropic"])
    monkeypatch.setenv("ITEMDEV_HOSTED", "1")
    uploaded = retrieval.import_collection(text.encode(), "Study.md", "clinical", approved_providers=["anthropic"])
    assert uploaded == local
    assert uploaded[0]["source_class"] == "study" and uploaded[0]["url"] == "https://example.test/fiction"
    assert uploaded[0]["providers"] == ["anthropic"]
    query = retrieval.build_query(*query_case)
    before = deepcopy(uploaded)
    assert not retrieval.search(uploaded, query, "clinical", provider="openai")
    assert retrieval.search(uploaded, query, "clinical", provider="anthropic")
    assert uploaded == before
    assert retrieval.import_collection(text, "Study.md", "clinical")[0]["providers"] == []
    unverified = text.replace("Citation: Invented Study https://example.test/fiction\n", "")
    assert retrieval.import_collection(unverified, "Study.md", "clinical")[0]["source_class"] == "practice summary"
    with pytest.raises(ValueError, match="relative"):
        retrieval.import_collection(text, "../outside.md", "clinical")


def test_permissions_filter_before_semantic_indexing(query_case, bank, monkeypatch):
    case, item = query_case
    query = retrieval.build_query(case, item, "FICT10 4.8")
    bank[0]["summary"] += " FICT10 threshold 4.8."
    bank[1].update(summary="PRIVATE PROVIDER PASSAGE", providers=["anthropic"])
    bank[2]["summary"] = "OUTSIDE STAGE PASSAGE"
    bank[3]["summary"] = "RETRACTED PASSAGE"
    sources = retrieval.import_collection(json.dumps(bank), "sample.json", "bank", approved_providers=["openai", "anthropic"])
    real = retrieval._similarities
    def assert_scope(documents, query_text):
        assert len(documents) == 1 and "FICT10" in documents[0]
        assert all(token not in str(documents) for token in ("PRIVATE", "OUTSIDE STAGE", "RETRACTED"))
        return real(documents, query_text)
    monkeypatch.setattr(retrieval, "_similarities", assert_scope)
    matches = retrieval.search(sources, query, "bank", provider="openai")
    assert any("fict10" in why and "4.8" in why for why in matches[0]["reasons"])
    assert matches[0]["source"]["providers"] == ["anthropic", "openai"]


def test_semantic_vectors_cover_nonoverlapping_phrases_and_learned_context():
    docs = ["dyspnea", "hypotension", "syncope"]
    assert set(retrieval._words("shortness of breath")).isdisjoint(retrieval._words(docs[0]))
    scores = retrieval._similarities(docs, "shortness of breath")
    assert scores[0] > 0 and scores[1] == scores[2] == 0

    # Invented vocabulary. A and B share corpus contexts but no common literals.
    docs = ["zota", "breli", "zota halna mopri", "breli halna mopri",
            "helta", "helta xemra frani", "suvo", "suvo divra curin"]
    scores = retrieval._similarities(docs, "zota")
    assert scores[1] > scores[4] + .01 and scores[1] > scores[6] + .01


def test_exact_threshold_retains_the_comparison_direction(query_case):
    case, item = query_case
    query = retrieval.build_query(case, item, exact_terms=">= 4.8")
    bank = [{"source_id": sid, "source_date": "1", "locator": sid, "summary": "Fixture threshold " + threshold}
            for sid, threshold in (("a", "≤4.8"), ("b", "≥4.8"))]
    sources = retrieval.import_collection(json.dumps(bank), "cutoffs.json", "bank")
    higher, lower = retrieval.search(sources, query, "bank")
    assert higher["source"]["source_id"] == "b" and higher["score"] > lower["score"]
    assert any(">=4.8" in reason for reason in higher["reasons"])


def test_citation_links_survive_import_without_treating_vault_paths_as_publications():
    citation = "Invented teaching guide, version 3 (https://example.test/guide)"
    note = "---\nsource_class: guideline\nsource: " + citation + "\n---\n# Rule\nTeaching summary."
    row = retrieval.import_collection(note, "guide.md", "clinical")[0]
    assert (row["source_class"], row["citation"], row["url"]) == (
        "guideline", citation, "https://example.test/guide",
    )
    local = retrieval.import_collection(note.replace(citation, '"[[Private note]]"'), "guide.md", "clinical")[0]
    assert local["source_class"] == "practice summary" and "citation" not in local
    for body in ("See our calendar at https://example.test/events", "Citation: [[Another local note]]"):
        ordinary = "---\nsource_class: guideline\n---\n# Rule\n" + body
        assert retrieval.import_collection(ordinary, "guide.md", "clinical")[0]["source_class"] == "practice summary"
    record = {"source_id": "study-1", "source_class": "study", "version": "3", "locator": "Section 1",
              "summary": "Invented reference for training.", "citation": citation}
    assert retrieval.import_collection(json.dumps([record]), "study.json", "clinical")[0]["url"] == row["url"]


@pytest.mark.parametrize("payload", [
    "---\nproviders: [openai]\nproviders: [anthropic]\n---\nDo not index.",
    "---\nproviders: both\n---\nDo not index.",
    "---\nproviders: [arbitrary-server]\n---\nDo not index.",
    "---\nproviders: [openai\n---\nDo not index.",
])
def test_ambiguous_permissions_fail_closed(payload):
    with pytest.raises(ValueError):
        retrieval.import_collection(payload, "x.md", "clinical", approved_providers=["openai", "anthropic"])


def test_bounded_import_and_read_failure_keep_existing_manual_sources(tmp_path, monkeypatch, query_case):
    monkeypatch.setenv("ITEMDEV_HOSTED", "0")
    case, item = query_case
    case["specialist_sources"] = [{"source_id": "manual", "source_class": "practice summary",
                                   "source_date": "Synthetic v1", "locator": "Entered by SME",
                                   "excerpt": "Synthetic local example", "providers": []}]
    snapshot = deepcopy(case)
    with pytest.raises(ValueError, match="readable"):
        retrieval.load_markdown_folder(tmp_path / "unavailable")
    with pytest.raises(ValueError, match="512 KiB"):
        retrieval.import_collection(b"x" * (retrieval.MAX_MARKDOWN_BYTES + 1), "x.md", "clinical")
    assert case == snapshot
    from specialist_workflow import _sources
    assert _sources(case, item) == snapshot["specialist_sources"]
