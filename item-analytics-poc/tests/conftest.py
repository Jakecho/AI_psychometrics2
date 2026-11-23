"""Shared fixtures. Every test runs with outbound network connections blocked."""

from pathlib import Path
import socket
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import embedding as em  # noqa: E402
from src import ingest  # noqa: E402
from src.text_builder import fingerprint  # noqa: E402

SAMPLE = ROOT / "samples" / "synthetic_items.csv"


@pytest.fixture(autouse=True)
def isolated_space_db(monkeypatch, tmp_path):
    """Every test (including AppTest runs) uses its own temporary space database."""
    monkeypatch.setenv("ITEM_ANALYTICS_DB_PATH", str(tmp_path / "space_library.sqlite"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that attempts a non-local connection."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in ("127.0.0.1", "localhost", "::1") and not host.startswith("/"):
            raise RuntimeError(f"Network access attempted to {host}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def make_item(item_id, stem, options=("Alpha", "Beta", "Gamma", "Delta"), key="A", **meta):
    row = {"item_id": item_id, "stem": stem, "key": key}
    for i, text in enumerate(options):
        row[f"option_{chr(65 + i)}"] = text
    row.update(meta)
    return row


def records_from_rows(rows, **mapping_overrides):
    df = ingest.read_table(csv_bytes(pd.DataFrame(rows).fillna("")), "bank.csv")
    mapping = ingest.detect_mapping(df)
    for k, v in mapping_overrides.items():
        setattr(mapping, k, v)
    records, report = ingest.build_records(df, mapping)
    return records, mapping, report


def manual_index(item_ids, vectors, valid=None, model="hand"):
    vecs, ok = em.l2_normalize(np.asarray(vectors, dtype=np.float32))
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    return em.EmbeddingIndex(item_ids=list(item_ids), text_hashes=[f"h{i}" for i in item_ids], vectors=vecs, valid=ok,
                             model=model, dimensions=vecs.shape[1], settings_fingerprint="fp", template="t")


@pytest.fixture
def sample_bank():
    df = ingest.read_table(SAMPLE.read_bytes(), SAMPLE.name)
    mapping = ingest.detect_mapping(df)
    records, report = ingest.build_records(df, mapping)
    return df, mapping, records, report


@pytest.fixture
def sample_index(sample_bank):
    _df, mapping, records, report = sample_bank
    fp = fingerprint(mapping.embed_settings())
    return em.generate_embeddings(records, em.MockEmbedder(), em.MOCK_MODEL, None, fp, report["template"])
