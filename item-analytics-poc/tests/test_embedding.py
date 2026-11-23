from types import SimpleNamespace

import numpy as np
import pytest

from src import embedding as em
from src import storage


class CountingEmbedder:
    def __init__(self, dim=8, fail_batches=(), transient_failures=0):
        self.calls, self.dim, self.fail_batches, self.transient = [], dim, set(fail_batches), transient_failures

    def embed(self, texts, model=None, dimensions=None):
        self.calls.append(list(texts))
        if self.transient:
            self.transient -= 1
            err = RuntimeError("rate limited")
            err.status_code = 429
            raise err
        if len(self.calls) - 1 in self.fail_batches:
            raise ValueError("bad request")
        rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
        return rng.normal(size=(len(texts), self.dim)), 10 * len(texts)


def test_mock_vectors_have_declared_dimension_and_unit_norm(sample_bank, sample_index):
    _, _, records, _ = sample_bank
    assert sample_index.vectors.shape == (len(records), 256)
    assert sample_index.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(sample_index.vectors[sample_index.valid], axis=1), 1.0, atol=1e-5)
    assert sample_index.matches(records, sample_index.settings_fingerprint)


def test_reuse_avoids_repeat_calls_and_failure_keeps_existing_state(sample_bank):
    _, _, records, report = sample_bank
    first = CountingEmbedder()
    idx = em.generate_embeddings(records, first, "m", None, "fp", "t", batch_size=5)
    assert idx.valid.all() and len(first.calls) == 3
    again = CountingEmbedder()
    same = em.generate_embeddings(records, again, "m", None, "fp", "t", existing=idx)
    assert again.calls == [] and np.allclose(same.vectors, idx.vectors, atol=1e-6)
    snapshot = idx.vectors.copy()
    failing = CountingEmbedder(fail_batches={0})
    partial = em.generate_embeddings(records, failing, "m", None, "fp-new", "t", existing=idx, batch_size=5)
    assert np.array_equal(idx.vectors, snapshot)  # existing index untouched
    assert (~partial.valid).sum() == 5 and partial.errors[0]["error"] == "ValueError"
    retry = em.generate_embeddings(records, CountingEmbedder(), "m", None, "fp-new", "t", existing=partial, batch_size=5)
    assert retry.valid.all()


def test_transient_errors_retry_with_bounded_backoff(sample_bank):
    _, _, records, _ = sample_bank
    sleeps = []
    emb = CountingEmbedder(transient_failures=2)
    idx = em.generate_embeddings(records[:3], emb, "m", None, "fp", "t", sleep=sleeps.append)
    assert idx.valid.all() and sleeps == [1.0, 2.0]
    stubborn = CountingEmbedder(transient_failures=99)
    with pytest.raises(em.EmbeddingError):
        em.generate_embeddings(records[:3], stubborn, "m", None, "fp", "t", sleep=lambda s: None, max_retries=2)


def test_zero_and_non_finite_vectors_are_invalid():
    vecs, valid = em.l2_normalize(np.array([[3, 4], [0, 0], [np.nan, 1]], dtype=np.float32))
    assert valid.tolist() == [True, False, False]
    assert np.allclose(vecs[0], [0.6, 0.8]) and not vecs[1:].any()


def test_openai_embedder_requires_consent_and_sends_expected_request():
    with pytest.raises(em.ConsentError):
        em.OpenAIEmbedder("sk-test", consent=False)
    captured = {}

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            data = [SimpleNamespace(index=1, embedding=[0.0, 1.0]), SimpleNamespace(index=0, embedding=[1.0, 0.0])]
            return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=7))

    embedder = em.OpenAIEmbedder(None, consent=True, client=SimpleNamespace(embeddings=FakeEmbeddings()))
    vectors, usage = embedder.embed(["a", "b"], model="text-embedding-3-small", dimensions=256)
    assert captured == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 256}
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]] and usage == 7


def test_bundle_reimport_detects_mismatch(sample_bank, sample_index):
    _, mapping, records, _ = sample_bank
    bundle = storage.export_bundle(records, mapping, sample_index, {})
    loaded = storage.import_bundle(bundle, current_records=records)
    assert np.array_equal(loaded["index"].vectors, sample_index.vectors)
    assert loaded["index"].model == sample_index.model and loaded["index"].dimensions == 256
    changed = [dict(r) for r in records]
    changed[0] = {**changed[0], "embed_text_hash": "different"}
    with pytest.raises(storage.BundleError, match="different items or embedding text"):
        storage.import_bundle(bundle, current_records=changed)
