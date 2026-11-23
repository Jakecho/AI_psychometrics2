"""2D projections of item embeddings and interpretable feature plots.

Projections are for display only. Retrieval always ranks in the original
embedding space; 2D distances are not cosine similarities.
"""

import numpy as np
import pandas as pd

from .schemas import FEATURE_TYPE_LABELS, FEATURE_TYPES
from .text_builder import fingerprint

ROLE_SYMBOLS = {"target": "star", "accepted sibling": "diamond", "rejected candidate": "x",
                "uncertain candidate": "triangle-up", "retrieved": "circle-open", "other": "circle"}


class ProjectionUnavailable(ValueError):
    pass


def umap_available():
    try:
        import umap  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def projection_key(index, method, params):
    return fingerprint({"settings": index.settings_fingerprint, "hashes": index.text_hashes, "model": index.model,
                        "method": method, "params": params})


def project(index, method="pca", seed=42, perplexity=30.0, pca_dims=30, n_neighbors=15, min_dist=0.1):
    valid = np.asarray(index.valid, dtype=bool)
    X = index.vectors[valid].astype(np.float64)
    ids = [iid for iid, v in zip(index.item_ids, valid) if v]
    n = len(ids)
    if n < 3:
        raise ProjectionUnavailable("At least 3 embedded items are needed for a 2D projection.")
    meta = {"method": method, "seed": seed}
    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(X)
        meta["explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    elif method == "tsne":
        if n < 5:
            raise ProjectionUnavailable("t-SNE needs at least 5 items.")
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        perp = float(min(perplexity, max(1.0, (n - 1) / 3)))
        if perp >= n:
            raise ProjectionUnavailable("Perplexity must be less than the number of items.")
        reduced = PCA(n_components=min(pca_dims, n - 1, X.shape[1]), random_state=seed).fit_transform(X)
        coords = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", metric="cosine").fit_transform(reduced)
        meta["perplexity"] = perp
    elif method == "umap":
        if not umap_available():
            raise ProjectionUnavailable("UMAP is not installed. Install umap-learn or use PCA/t-SNE.")
        if n < 5:
            raise ProjectionUnavailable("UMAP needs at least 5 items.")
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=int(min(n_neighbors, n - 1)), min_dist=min_dist,
                            metric="cosine", random_state=seed)
        coords = reducer.fit_transform(X)
        meta["n_neighbors"] = int(min(n_neighbors, n - 1))
    else:
        raise ValueError(f"Unknown projection method '{method}'.")
    return pd.DataFrame({"item_id": ids, "x": coords[:, 0], "y": coords[:, 1]}), meta


def get_projection(cache, index, method, **params):
    """Per-session cache keyed by embedding fingerprint, method, parameters and seed."""
    key = projection_key(index, method, params)
    if key not in cache:
        cache[key] = project(index, method, **params)
    return cache[key]


def is_numeric(series):
    values = series.dropna()
    return len(values) > 0 and pd.to_numeric(values, errors="coerce").notna().all()


def plot_frame(coords, records_by_id, color_by=None, roles=None, scores=None):
    df = coords.copy()
    df["role"] = [(roles or {}).get(i, "other") for i in df["item_id"]]
    df["stem_preview"] = [_preview(records_by_id[i]) for i in df["item_id"]]
    if scores is not None:
        df["cosine_to_target"] = [scores.get(i) for i in df["item_id"]]
    if color_by:
        df[color_by] = [records_by_id[i]["metadata"].get(color_by) for i in df["item_id"]]
    return df


def _preview(record, n=110):
    text = " ".join(v for v in record["fields"].values() if v)
    return text if len(text) <= n else text[: n - 1] + "…"


def scatter_figure(df, color_by=None, color_kind="auto", hidden=None, title=""):
    import plotly.express as px

    data = df.copy()
    kind = None
    if color_by:
        kind = color_kind if color_kind != "auto" else ("numeric" if is_numeric(data[color_by]) else "categorical")
        if kind == "numeric":
            data[color_by] = pd.to_numeric(data[color_by], errors="coerce")
        else:
            data[color_by] = data[color_by].map(lambda v: "(missing)" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            if hidden:
                data = data[~data[color_by].isin(hidden)]
    hover = {"item_id": True, "stem_preview": True, "role": True, "x": False, "y": False}
    if "cosine_to_target" in data:
        hover["cosine_to_target"] = ":.3f"
    if color_by:
        hover[color_by] = True
    kwargs = {"color": color_by} if color_by else {}
    if kind == "numeric":
        kwargs["color_continuous_scale"] = "Viridis"
    fig = px.scatter(data, x="x", y="y", symbol="role", symbol_map=ROLE_SYMBOLS, hover_data=hover,
                     custom_data=["item_id"], title=title, **kwargs)
    fig.update_traces(marker={"size": 10, "line": {"width": 0.5, "color": "#333"}})
    if kind == "numeric" and data[color_by].isna().any():
        missing = data[data[color_by].isna()]
        fig.add_scatter(x=missing["x"], y=missing["y"], mode="markers", name=f"{color_by}: (missing)",
                        marker={"color": "#bbbbbb", "size": 9, "symbol": "circle-open"},
                        text=missing["item_id"], hoverinfo="text")
    fig.update_layout(xaxis_title="Projection axis 1", yaxis_title="Projection axis 2",
                      legend_title_text="", height=560, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return fig, kind


def heatmap_figure(matrix, store):
    """Binary item × feature incidence grouped by feature type."""
    import plotly.graph_objects as go

    if matrix.empty or matrix.shape[1] == 0:
        return None
    order = [fid for t in FEATURE_TYPES for fid in matrix.columns if store["features"][fid]["feature_type"] == t]
    m = matrix[order]
    labels = [store["features"][fid]["canonical_label"] for fid in order]
    types = [FEATURE_TYPE_LABELS[store["features"][fid]["feature_type"]] for fid in order]
    hover = [[f"Item {iid}<br>{fid} ({types[j]})<br>{labels[j]}<br>{'present' if m.iloc[i, j] else 'absent'}"
              for j, fid in enumerate(order)] for i, iid in enumerate(m.index)]
    fig = go.Figure(go.Heatmap(z=m.values, x=[f"{t[0]}·{fid}" for t, fid in zip(types, order)], y=list(m.index),
                               colorscale=[[0, "#f3f4f6"], [1, "#1f6f8b"]], zmin=0, zmax=1, showscale=False,
                               hoverinfo="text", text=hover, xgap=2, ygap=2))
    fig.update_layout(height=max(260, 34 * len(m.index) + 120), margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis={"tickangle": -45}, yaxis={"autorange": "reversed"})
    return fig
