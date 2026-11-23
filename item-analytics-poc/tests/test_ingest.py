import io

import pandas as pd
import pytest

from src import ingest
from src.text_builder import fingerprint
from tests.conftest import csv_bytes, make_item, records_from_rows


def test_variable_option_counts_are_preserved_in_csv():
    rows = [make_item("I-1", "Two options?", ("Yes", "No"), key="B"),
            make_item("I-2", "Four options?", ("a1", "a2", "a3", "a4"), key="C"),
            make_item("I-3", "Six options?", ("b1", "b2", "b3", "b4", "b5", "b6"), key="F")]
    records, mapping, _ = records_from_rows(rows)
    assert mapping.option_columns == [f"option_{c}" for c in "ABCDEF"]
    counts = {r["item_id"]: len(r["options"]) for r in records}
    assert counts == {"I-1": 2, "I-2": 4, "I-3": 6}
    six = next(r for r in records if r["item_id"] == "I-3")
    assert [o["text"] for o in six["options"]] == ["b1", "b2", "b3", "b4", "b5", "b6"]
    assert [o["option_id"] for o in six["options"] if o["is_key"]] == ["F"]


def test_xlsx_with_mixed_ids_and_types_and_sheet_choice():
    df = pd.DataFrame({"ID": [101, "A-2", 3.0], "Stem": ["s1", "s2", "s3"], "Choice A": ["x", "y", "z"],
                       "Choice B": ["p", "q", "r"], "Answer": ["A", "b", "2"], "Rasch_b": [0.5, -1, 1.25]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"note": ["cover sheet"]}).to_excel(w, sheet_name="Cover", index=False)
        df.to_excel(w, sheet_name="Items", index=False)
    data = buf.getvalue()
    assert ingest.list_sheets(data, "bank.xlsx") == ["Cover", "Items"]
    table = ingest.read_table(data, "bank.xlsx", sheet="Items")
    mapping = ingest.detect_mapping(table)
    assert (mapping.item_id, mapping.key, mapping.difficulty) == ("ID", "Answer", "Rasch_b")
    assert mapping.option_columns == ["Choice A", "Choice B"]
    records, _ = ingest.build_records(table, mapping)
    assert [r["item_id"] for r in records] == ["101", "A-2", "3"]
    assert [[o["option_id"] for o in r["options"] if o["is_key"]] for r in records] == [["A"], ["B"], ["B"]]
    assert records[2]["metadata"]["Rasch_b"] == 1.25


def test_manual_mapping_overrides_detection():
    df = ingest.read_table(csv_bytes(pd.DataFrame({"code": ["x1"], "scenario": ["A case"], "question": ["Which?"],
                                                   "alt1": ["one"], "alt2": ["two"], "correct": ["alt2"]})), "b.csv")
    mapping = ingest.ColumnMapping(item_id="code", text_columns=["scenario", "question"], option_columns=["alt1", "alt2"], key="correct")
    records, report = ingest.build_records(df, mapping)
    assert records[0]["selected_embed_text"] == "scenario: A case\nquestion: Which?"
    assert [o["option_id"] for o in records[0]["options"]] == ["A", "B"]
    # "alt2" is neither an option ID nor option text, so no key is invented
    assert records[0]["key_status"] == "unresolved"
    assert all(o["is_key"] is None for o in records[0]["options"])


def test_conflicting_duplicate_ids_raise_and_exact_duplicates_are_reported():
    a = make_item("D-1", "Same stem?")
    with pytest.raises(ingest.IngestError, match="D-1"):
        records_from_rows([a, make_item("D-1", "Different stem?")])
    records, _, report = records_from_rows([a, dict(a), make_item("D-2", "Other?")])
    assert [r["item_id"] for r in records] == ["D-1", "D-2"]
    assert report["exact_duplicates_removed"] == [{"source_row": 3, "duplicate_of_row": 2}]
    assert records[1]["source_row"] == 4


def test_key_parsing_never_invents_answers():
    opts = [{"option_id": c, "text": t, "is_key": None} for c, t in zip("ABC", ["Red", "Green", "Blue"])]
    assert ingest.parse_key("b", opts) == ({"B"}, "single")
    assert ingest.parse_key("A, C", opts) == ({"A", "C"}, "multiple")
    assert ingest.parse_key("3", opts) == ({"C"}, "single")
    assert ingest.parse_key("green", opts) == ({"B"}, "single")
    assert ingest.parse_key("", opts) == (set(), "missing")
    assert ingest.parse_key("E", opts) == (set(), "unresolved")
    assert ingest.parse_key("7", opts) == (set(), "unresolved")


def test_malformed_numeric_difficulty_is_rejected_with_rows():
    rows = [make_item("N-1", "One?", rasch_b="0.4"), make_item("N-2", "Two?", rasch_b="hard")]
    with pytest.raises(ingest.IngestError, match="rows 3"):
        records_from_rows(rows)


def test_encoding_bom_whitespace_nulls_and_missing_ids():
    text = "﻿stem,option_A,option_B,domain\n  Padded stem?  ,a,b,\nSecond?,c,,Cardio\n"
    df = ingest.read_table(text.encode("utf-8"), "x.csv")
    records, report = ingest.build_records(df, ingest.detect_mapping(df))
    assert report["synthetic_ids"]
    assert [r["item_id"] for r in records] == ["ROW-0002", "ROW-0003"]
    assert records[0]["fields"]["stem"] == "Padded stem?"
    assert records[0]["metadata"]["domain"] is None
    assert len(records[1]["options"]) == 1  # blank option cell is not an option
    with pytest.raises(ingest.IngestError, match="UTF-8"):
        ingest.read_table("stem\ncaf\xe9".encode("latin-1"), "bad.csv")


def test_rows_without_embedding_text_are_reported_not_imputed():
    records, _, report = records_from_rows([make_item("E-1", "Text?"), make_item("E-2", "")])
    assert [r["item_id"] for r in records] == ["E-1"]
    assert report["skipped_empty_text"] == [{"source_row": 3, "item_id": "E-2"}]


def test_concatenation_reproducible_metadata_preserved_and_text_change_invalidates():
    rows = [make_item("C-1", "Stem one?", vignette="A case.", region="North", custom="keep me")]
    r1, mapping, _ = records_from_rows(rows)
    r2, _, _ = records_from_rows(rows)
    assert r1[0]["selected_embed_text"] == r2[0]["selected_embed_text"] == "vignette: A case.\nstem: Stem one?"
    assert r1[0]["metadata"]["custom"] == "keep me" and r1[0]["metadata"]["region"] == "North"
    before = fingerprint(mapping.embed_settings())
    mapping.text_columns = ["stem"]
    assert fingerprint(mapping.embed_settings()) != before
    df = ingest.read_table(csv_bytes(pd.DataFrame(rows)), "b.csv")
    r3, _ = ingest.build_records(df, mapping)
    assert r3[0]["embed_text_hash"] != r1[0]["embed_text_hash"]


def test_option_text_inclusion_is_explicit_and_warned():
    rows = [make_item("O-1", "Which?", ("first", "second"))]
    records, mapping, report = records_from_rows(rows)
    assert "first" not in records[0]["selected_embed_text"]
    records, mapping, report = records_from_rows(rows, include_options_in_embed=True)
    assert "A. first" in records[0]["selected_embed_text"] and "key" not in records[0]["selected_embed_text"].lower()
    assert any("leakage" in w for w in report["warnings"])


def test_invalid_mappings_are_rejected():
    df = ingest.read_table(csv_bytes(pd.DataFrame([make_item("M-1", "x?")])), "b.csv")
    with pytest.raises(ingest.IngestError, match="at least one text column"):
        ingest.build_records(df, ingest.ColumnMapping(item_id="item_id"))
    with pytest.raises(ingest.IngestError, match="key column"):
        ingest.build_records(df, ingest.ColumnMapping(item_id="item_id", text_columns=["stem", "key"], key="key"))
    with pytest.raises(ingest.IngestError, match="not in the file"):
        ingest.build_records(df, ingest.ColumnMapping(text_columns=["missing"]))
