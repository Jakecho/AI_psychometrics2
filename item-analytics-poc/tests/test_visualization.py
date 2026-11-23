import numpy as np
import pytest

from src import retrieval as rt
from src import visualization as viz
from tests.conftest import manual_index


def test_projection_cache_ignores_colour_and_handles_tiny_samples(sample_bank, sample_index):
    _, _, records, _ = sample_bank
    cache = {}
    coords, meta = viz.get_projection(cache, sample_index, "pca", seed=1)
    assert len(cache) == 1 and len(coords) == len(records) and len(meta["explained_variance"]) == 2
    by_id = {r["item_id"]: r for r in records}
    cat = viz.plot_frame(coords, by_id, "domain", {"S-01": "target"})
    num = viz.plot_frame(coords, by_id, "rasch_b")
    fig_c, kind_c = viz.scatter_figure(cat, "domain", hidden=["Maternal care"])
    fig_n, kind_n = viz.scatter_figure(num, "rasch_b")
    assert (kind_c, kind_n) == ("categorical", "numeric")
    assert viz.get_projection(cache, sample_index, "pca", seed=1)[0] is coords and len(cache) == 1
    assert any(t.name == "rasch_b: (missing)" for t in fig_n.data)  # S-10 has no difficulty
    tiny = manual_index(["a", "b"], [[1, 0], [0, 1]])
    with pytest.raises(viz.ProjectionUnavailable):
        viz.project(tiny, "pca")
    four = manual_index(list("abcd"), np.eye(4))
    with pytest.raises(viz.ProjectionUnavailable, match="t-SNE"):
        viz.project(four, "tsne")


def test_tsne_caps_perplexity_and_is_deterministic(sample_index):
    a, meta = viz.project(sample_index, "tsne", seed=7, perplexity=50)
    b, _ = viz.project(sample_index, "tsne", seed=7, perplexity=50)
    assert meta["perplexity"] < len(sample_index.item_ids)
    assert np.allclose(a[["x", "y"]].to_numpy(), b[["x", "y"]].to_numpy())


def test_retrieval_uses_original_space_not_projection(sample_index):
    pos = sample_index.position("S-01")
    before = rt.select(rt.cosine_scores(sample_index, sample_index.vectors[pos]), sample_index.item_ids, "top_k", 5, exclude_id="S-01")
    viz.project(sample_index, "pca")
    after = rt.select(rt.cosine_scores(sample_index, sample_index.vectors[pos]), sample_index.item_ids, "top_k", 5, exclude_id="S-01")
    assert before.equals(after)


def test_heatmap_groups_by_feature_type():
    from tests.test_frequency import store_with
    from src import feature_frequency as ff

    store = store_with([("distractor_feature", "x", ["S1"], [("S1", "B")]), ("declarative_key", "d", ["S1", "S2"], [])])
    m = ff.incidence_matrix(store, ["S1", "S2"])
    fig = viz.heatmap_figure(m, store)
    assert list(fig.data[0].x)[0].startswith("D·")
