import io
from pathlib import Path
import re
import zipfile

import pandas as pd
import pytest

from src import embedding as em
from src import storage
from src.llm import OpenAIStructuredLLM

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "prompts").glob("*.md")),
           ROOT / "samples" / "synthetic_items.csv", ROOT / ".streamlit" / "config.toml"]


def test_no_hard_coded_credentials_in_source():
    patterns = [r"sk-[A-Za-z0-9_-]{16,}", r"(?i)password\s*[:=]\s*['\"][^'\"]+['\"]",
                r"(?i)api_key\s*=\s*['\"][A-Za-z0-9_-]{8,}['\"]", r"postgres(ql)?://"]
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for p in patterns:
            assert not re.search(p, text), f"{p} in {path.name}"


def test_no_shared_streamlit_caches_hold_user_content():
    for path in [ROOT / "app.py", *(ROOT / "src").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"@st\.cache|st\.cache_(data|resource)\s*\(", text), path.name


def test_bundle_and_exports_exclude_keys_and_neutralise_formulas(sample_bank, sample_index, monkeypatch):
    secret = "sk-test-DO-NOT-EXPORT-1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _, mapping, records, _ = sample_bank
    review = {"settings": {"mode": "OpenAI", "llm_model": "m"}}
    bundle = storage.export_bundle(records, mapping, sample_index, review, "bank.csv")
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        for name in z.namelist():
            assert secret.encode() not in z.read(name)
    frame = pd.DataFrame({"text": ["=HYPERLINK(\"x\")", "+1", "-cmd", "@SUM(A1)", "plain", None], "n": [1, 2, 3, 4, 5, 6]})
    out = storage.safe_frame(frame)
    assert out["text"].tolist()[:5] == ["'=HYPERLINK(\"x\")", "'+1", "'-cmd", "'@SUM(A1)", "plain"]
    assert out["n"].tolist() == [1, 2, 3, 4, 5, 6]


def test_external_clients_require_explicit_consent():
    with pytest.raises(em.ConsentError):
        em.OpenAIEmbedder("sk-anything", consent=False)
    with pytest.raises(em.ConsentError):
        OpenAIStructuredLLM("sk-anything", "model", consent=False)


def test_error_messages_do_not_echo_request_content():
    class Boom(Exception):
        status_code = 401

    msg = em.safe_error(Boom("Incorrect API key provided: sk-abc... and the item text"))
    assert msg == "Boom (HTTP 401)"
