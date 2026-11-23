"""Portable project bundles and safe tabular exports.

A bundle carries records, the column mapping, vectors with their manifest, and
review state. It never carries API keys: callers pass explicit state, not the
whole session.
"""

from datetime import datetime, timezone
import io
import json
import zipfile

import numpy as np
import pandas as pd

from . import APP_VERSION
from .embedding import EmbeddingIndex
from .ingest import ColumnMapping

BUNDLE_VERSION = 1
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class BundleError(ValueError):
    pass


def neutralize_formula(value):
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def safe_frame(df):
    """Prefix text cells that a spreadsheet would execute as a formula."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(neutralize_formula)
    out.columns = [neutralize_formula(str(c)) for c in out.columns]
    return out


def to_csv_bytes(df):
    return safe_frame(df).to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_frame(df).to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def to_json_bytes(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def export_bundle(records, mapping, index=None, review_state=None, source_name=None):
    manifest = {
        "bundle_version": BUNDLE_VERSION, "app_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_name": source_name, "item_count": len(records),
        "item_ids": [r["item_id"] for r in records], "text_hashes": [r["embed_text_hash"] for r in records],
        "mapping": mapping.to_dict(), "embedding": None,
        "contains_api_keys": False,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if index is not None:
            manifest["embedding"] = {
                "model": index.model, "dimensions": index.dimensions, "requested_dimensions": index.requested_dimensions,
                "settings_fingerprint": index.settings_fingerprint, "template": index.template,
                "item_ids": index.item_ids, "text_hashes": index.text_hashes, "usage_tokens": index.usage_tokens,
            }
            vec = io.BytesIO()
            np.savez_compressed(vec, vectors=index.vectors, valid=index.valid)
            z.writestr("vectors.npz", vec.getvalue())
        z.writestr("manifest.json", to_json_bytes(manifest))
        z.writestr("records.json", to_json_bytes(records))
        z.writestr("review.json", to_json_bytes(review_state or {}))
    return buf.getvalue()


def import_bundle(data, current_records=None):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        manifest = json.loads(z.read("manifest.json"))
        records = json.loads(z.read("records.json"))
        review = json.loads(z.read("review.json")) if "review.json" in z.namelist() else {}
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        raise BundleError("This is not a valid project bundle.") from None
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise BundleError("The bundle was created by an incompatible version of the app.")
    ids = [r["item_id"] for r in records]
    hashes = [r["embed_text_hash"] for r in records]
    if ids != manifest["item_ids"] or hashes != manifest["text_hashes"]:
        raise BundleError("The bundle's records do not match its manifest.")
    if current_records is not None:
        if [r["item_id"] for r in current_records] != ids or [r["embed_text_hash"] for r in current_records] != hashes:
            raise BundleError("The bundle was built from different items or embedding text than the loaded data. "
                              "Load the bundle on its own or regenerate embeddings.")
    index = None
    emb = manifest.get("embedding")
    if emb:
        try:
            arrays = np.load(io.BytesIO(z.read("vectors.npz")))
            vectors, valid = arrays["vectors"].astype(np.float32), arrays["valid"].astype(bool)
        except (KeyError, ValueError):
            raise BundleError("The bundle's vectors are missing or unreadable.") from None
        if emb["item_ids"] != ids or emb["text_hashes"] != hashes:
            raise BundleError("The vectors were built for different items or text. Regenerate embeddings.")
        if vectors.shape != (len(ids), emb["dimensions"]) or valid.shape != (len(ids),):
            raise BundleError(f"Vector shape {vectors.shape} does not match {len(ids)} items × {emb['dimensions']} dimensions.")
        if not np.isfinite(vectors).all():
            raise BundleError("The bundle contains non-finite vector values.")
        index = EmbeddingIndex(item_ids=ids, text_hashes=hashes, vectors=vectors, valid=valid, model=emb["model"],
                               dimensions=int(emb["dimensions"]), settings_fingerprint=emb["settings_fingerprint"],
                               template=emb["template"], usage_tokens=int(emb.get("usage_tokens") or 0),
                               requested_dimensions=emb.get("requested_dimensions"))
    return {"manifest": manifest, "records": records, "mapping": ColumnMapping.from_dict(manifest["mapping"]),
            "index": index, "review": review}
