"""Embedding adapters, batching with bounded retries, and the in-memory vector index.

OpenAI is imported only when a real request is made. ``MockEmbedder`` provides a
deterministic offline vector space for demonstrations and tests; it is labelled
as a lexical mock, not a semantic model.
"""

from dataclasses import dataclass, field
import hashlib
import re
import time

import numpy as np

DEFAULT_MODEL = "text-embedding-3-small"
# Approximate list prices in USD per 1M input tokens, for the on-screen estimate
# only. Verify current pricing before budgeting.
APPROX_PRICE_PER_M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
MOCK_MODEL = "mock-lexical-hash-v1"


class ConsentError(PermissionError):
    """An external request was attempted without the user's explicit consent."""


class EmbeddingError(RuntimeError):
    pass


def estimate_tokens(texts):
    """Rough token estimate (≈4 characters per token) for consent and cost display."""
    return int(sum(len(t) for t in texts) / 4) + len(texts)


def estimate_cost(model, tokens):
    price = APPROX_PRICE_PER_M.get(model)
    return None if price is None else tokens / 1_000_000 * price


def l2_normalize(matrix):
    """Unit-normalise rows; zero or non-finite rows become zero vectors and are reported invalid."""
    matrix = np.asarray(matrix, dtype=np.float32)
    finite = np.isfinite(matrix).all(axis=1)
    safe = np.where(finite[:, None], matrix, 0.0)
    norms = np.linalg.norm(safe, axis=1)
    valid = finite & (norms > 1e-12)
    out = np.zeros_like(safe)
    out[valid] = safe[valid] / norms[valid, None]
    return out.astype(np.float32), valid


@dataclass
class EmbeddingIndex:
    item_ids: list
    text_hashes: list
    vectors: np.ndarray  # N x d, unit-normalised float32 (invalid rows are zero)
    valid: np.ndarray  # bool mask
    model: str
    dimensions: int
    settings_fingerprint: str
    template: str
    usage_tokens: int = 0
    errors: list = field(default_factory=list)
    requested_dimensions: int | None = None

    def position(self, item_id):
        try:
            return self.item_ids.index(item_id)
        except ValueError:
            return None

    def matches(self, records, settings_fingerprint):
        """True when this index was built for exactly these records and settings."""
        return (self.settings_fingerprint == settings_fingerprint
                and self.item_ids == [r["item_id"] for r in records]
                and self.text_hashes == [r["embed_text_hash"] for r in records])


class MockEmbedder:
    """Deterministic hashed word/character-trigram vectors; no network access."""

    name = MOCK_MODEL
    external = False

    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def embed(self, texts, model=None, dimensions=None):
        dim = dimensions or self.dimensions
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            grams = words + [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
                out[row, h % dim] += 1.0 if (h >> 63) & 1 else -1.0
        return out, estimate_tokens(texts)


class OpenAIEmbedder:
    """Server-side OpenAI Embeddings client. The key stays in this object only."""

    external = True

    def __init__(self, api_key, consent, client=None):
        if not consent:
            raise ConsentError("Confirm consent to send item text to OpenAI before generating embeddings.")
        if not api_key and client is None:
            raise EmbeddingError("Provide an OpenAI API key.")
        self._api_key = api_key
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily: slow and only needed for live calls
            self._client = OpenAI(api_key=self._api_key, max_retries=0, timeout=60)
        return self._client

    def embed(self, texts, model=DEFAULT_MODEL, dimensions=None):
        kwargs = {"model": model, "input": list(texts)}
        if dimensions:
            kwargs["dimensions"] = int(dimensions)
        response = self.client.embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda d: d.index)
        if len(data) != len(texts):
            raise EmbeddingError("The embedding response did not include one vector per input.")
        usage = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
        return np.asarray([d.embedding for d in data], dtype=np.float32), usage


def is_transient(exc):
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
        return True
    return type(exc).__name__ in {"RateLimitError", "APITimeoutError", "APIConnectionError",
                                  "InternalServerError", "TimeoutError", "ConnectionError"}


def safe_error(exc):
    """A user-facing error string that never echoes request content or credentials."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")


def call_with_retries(fn, max_retries=4, base_delay=1.0, sleep=time.sleep):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not is_transient(exc):
                raise
            sleep(min(base_delay * (2 ** attempt), 30.0))
            attempt += 1


def generate_embeddings(records, embedder, model, dimensions, settings_fingerprint, template,
                        existing=None, batch_size=64, max_retries=4, sleep=time.sleep, progress=None,
                        cancel=None):
    """Embed every record, reusing vectors from ``existing`` whose text hash and settings match.

    Returns a new index; ``existing`` is never mutated. On a batch failure the
    affected rows are left invalid and listed in ``errors`` so the caller can
    retry without losing completed work.
    """
    n = len(records)
    reuse = {}
    if (existing is not None and existing.settings_fingerprint == settings_fingerprint
            and existing.model == model and existing.requested_dimensions == dimensions):
        for pos, (iid, h) in enumerate(zip(existing.item_ids, existing.text_hashes)):
            if existing.valid[pos]:
                reuse[(iid, h)] = existing.vectors[pos]
    raw = [None] * n
    todo = []
    for i, r in enumerate(records):
        cached = reuse.get((r["item_id"], r["embed_text_hash"]))
        if cached is not None:
            raw[i] = cached
        else:
            todo.append(i)
    usage, errors, dim = 0, [], None
    if reuse:
        dim = len(next(iter(reuse.values())))
    done = n - len(todo)
    for start in range(0, len(todo), batch_size):
        if cancel and cancel():
            errors.append({"item_ids": [records[i]["item_id"] for i in todo[start:]], "error": "Cancelled"})
            break
        batch = todo[start:start + batch_size]
        texts = [records[i]["selected_embed_text"] for i in batch]
        try:
            vectors, tokens = call_with_retries(lambda: embedder.embed(texts, model=model, dimensions=dimensions),
                                                max_retries=max_retries, sleep=sleep)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise EmbeddingError("The embedding response has an unexpected shape.")
            if dim is None:
                dim = vectors.shape[1]
            elif vectors.shape[1] != dim:
                raise EmbeddingError(f"Inconsistent embedding dimension ({vectors.shape[1]} vs {dim}).")
            for i, v in zip(batch, vectors):
                raw[i] = v
            usage += int(tokens or 0)
        except Exception as exc:  # noqa: BLE001
            errors.append({"item_ids": [records[i]["item_id"] for i in batch], "error": safe_error(exc)})
        done += len(batch)
        if progress:
            progress(done, n)
    if dim is None:
        raise EmbeddingError("No embeddings were produced. " + (errors[0]["error"] if errors else ""))
    matrix = np.zeros((n, dim), dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    for i, v in enumerate(raw):
        if v is not None:
            matrix[i] = v
            present[i] = True
    vectors, valid = l2_normalize(matrix)
    valid &= present
    zero_rows = [records[i]["item_id"] for i in range(n) if present[i] and not valid[i]]
    if zero_rows:
        errors.append({"item_ids": zero_rows, "error": "Zero or non-finite vector"})
    return EmbeddingIndex(
        item_ids=[r["item_id"] for r in records],
        text_hashes=[r["embed_text_hash"] for r in records],
        vectors=vectors, valid=valid, model=model, dimensions=int(dim),
        settings_fingerprint=settings_fingerprint, template=template,
        usage_tokens=(existing.usage_tokens if existing is not None and reuse else 0) + usage,
        errors=errors, requested_dimensions=dimensions,
    )


def embed_queries(texts, embedder, index, batch_size=64, sleep=time.sleep):
    """Embed several query texts with the same model and dimensions as the index.

    Returns (unit vectors, valid mask, tokens). Queries are not added to the index.
    """
    texts = list(texts)
    if not texts or any(not (t or "").strip() for t in texts):
        raise EmbeddingError("Every query needs text.")
    dims = index.dimensions if index.model == MOCK_MODEL else index.requested_dimensions
    parts, tokens = [], 0
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        vectors, used = call_with_retries(lambda: embedder.embed(chunk, model=index.model, dimensions=dims), sleep=sleep)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(chunk), index.dimensions):
            raise EmbeddingError("The query embedding dimension does not match the index. Regenerate embeddings.")
        parts.append(vectors)
        tokens += int(used or 0)
    vec, valid = l2_normalize(np.vstack(parts))
    return vec, valid, tokens


def embed_query(text, embedder, index):
    """Embed free text with the same model and dimensions as the index."""
    if not text.strip():
        raise EmbeddingError("Enter query text.")
    vec, valid, _ = embed_queries([text], embedder, index)
    if not valid[0]:
        raise EmbeddingError("The query produced a zero vector.")
    return vec[0]
