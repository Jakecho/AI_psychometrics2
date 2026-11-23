"""Item Bank Analytics & Cognitive Feature Decomposition — proof of concept.

All user data, vectors, keys and reviews live in ``st.session_state`` only. No
``st.cache_data``/``st.cache_resource`` is used for user content.
"""

import hashlib
import json
import os
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from src import APP_VERSION
from src import decomposition as dc
from src import distractors as ds
from src import embedding as em
from src import enemies as en
from src import feature_frequency as ff
from src import ingest
from src import predictions as pr
from src import retrieval as rt
from src import storage
from src import task_alignment as ta
from src import visualization as viz
from src.llm import LLMError, MockStructuredLLM, OpenAIStructuredLLM
from src.schemas import (DOMINANCE, FEATURE_TYPE_LABELS, FEATURE_TYPES, IMPORTANCE_ANCHORS, REVIEW_STATUSES,
                         SME_ALIGNMENTS)
from src.text_builder import fingerprint

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "samples" / "synthetic_items.csv"
SECTIONS = ["Data & API", "Search", "Decomposition", "Distractors", "Embedding Visualization",
            "Difficulty", "Classification", "Enemy Candidates", "Exports"]
NONE = "(none)"

st.set_page_config(page_title="Item Bank Analytics", page_icon="🧩", layout="wide")

DEFAULTS = {
    "reviewer": "", "raw_df": None, "file_name": None, "file_hash": None, "sheet": None, "mapping": None,
    "records": None, "report": None, "index": None, "search": None, "task_review": None,
    "store": None, "enemy_decisions": {}, "enemy_table": None, "class_overrides": {}, "proj_cache": {},
    "usage_log": [], "archived_reviews": [], "stale_notice": None, "notice": None, "store_rev": 0,
    "llm_mode": "Offline mock (no network)", "consent_embed": False, "consent_llm": False,
    "api_key_input": "", "embed_model": em.DEFAULT_MODEL, "embed_dims": 0, "llm_model": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
ss = st.session_state
# Keep settings whose widgets live on one section when another section is shown.
for _k in ("llm_mode", "consent_embed", "consent_llm", "api_key_input", "embed_model", "embed_dims", "llm_model",
           "freq_source", "scale_confirmed"):
    if _k in ss:
        ss[_k] = ss[_k]


# =========================================================================== helpers

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def md(text):
    """Escape item text for markdown rendering (no HTML is ever enabled)."""
    return _MD_SPECIAL.sub(r"\\\1", str(text if text is not None else ""))


def records_by_id():
    return {r["item_id"]: r for r in ss.records or []}


def mock_mode():
    return ss.llm_mode.startswith("Offline")


def api_key():
    """(key, source). The value never leaves the server process."""
    if ss.get("api_key_input"):
        return ss.api_key_input, "entered for this session"
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]), "Streamlit secrets"
    except Exception:  # noqa: BLE001 - no secrets file configured
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], "environment"
    return None, "not set"


def embed_settings():
    m = ss.mapping
    dims = int(ss.get("embed_dims") or 0) or None
    model = em.MOCK_MODEL if mock_mode() else (ss.get("embed_model") or em.DEFAULT_MODEL).strip()
    return model, dims, fingerprint({**m.embed_settings(), "model": model, "dimensions": dims})


def get_embedder():
    if mock_mode():
        return em.MockEmbedder()
    key, _ = api_key()
    return em.OpenAIEmbedder(key, consent=ss.consent_embed)


def get_llm():
    if mock_mode():
        return MockStructuredLLM()
    key, _ = api_key()
    return OpenAIStructuredLLM(key, (ss.get("llm_model") or "").strip(), consent=ss.consent_llm)


def log_usage(kind, model, tokens):
    ss.usage_log.append({"kind": kind, "model": model, **tokens})


def bump_store():
    ss.store_rev += 1


def item_card(record, title=None, show_key=True, cosine=None):
    with st.container(border=True):
        head = title or f"**{md(record['item_id'])}**"
        if cosine is not None:
            head += f" · cosine {cosine:.3f}"
        st.markdown(head)
        meta = [f"{k}: {md(v)}" for k, v in (("domain", record["roles"].get("domain")), ("task", record["roles"].get("task"))) if v]
        if meta:
            st.caption(" · ".join(meta))
        for label, text in record["fields"].items():
            if text:
                st.markdown(f"*{md(label)}:* {md(text)}")
        if record["options"]:
            for o in record["options"]:
                mark = " ✅ **key**" if (show_key and o["is_key"]) else ""
                st.markdown(f"- **{md(o['option_id'])}.** {md(o['text'])}{mark}")
        else:
            st.caption("No answer options mapped.")
        if record["key_status"] in ("missing", "unresolved"):
            st.caption(f"Key {record['key_status']}" + (f" (raw value: {md(record['key_raw'])})" if record.get("key_raw") else "")
                       + " — key-dependent analysis is pending; no key is inferred.")


def require_records():
    if not ss.records:
        st.info("Load an item file in **Data & API** first.")
        st.stop()


def require_index():
    if ss.index is None:
        st.info("Generate or import embeddings in **Data & API** first.")
        st.stop()


def invalidate_for_new_data(reason):
    """Archive (never discard) review work when the data or embedding basis changes."""
    if ss.task_review or (ss.store and ss.store["features"]):
        ss.archived_reviews.append({"reason": reason, "task_review": ss.task_review, "store": ss.store,
                                    "search_target": (ss.search or {}).get("target", {}).get("item_id")})
        ss.stale_notice = f"{reason} Previous task and feature reviews were archived and remain exportable from Exports."
    ss.search = None
    ss.task_review = None
    ss.store = None
    ss.enemy_table = None
    ss.proj_cache = {}
    bump_store()


# =========================================================================== sidebar & status

with st.sidebar:
    st.markdown("### 🧩 Item Bank Analytics")
    st.caption(f"Proof of concept v{APP_VERSION} · database-free · session-only storage")
    section = st.radio("Section", SECTIONS, key="section", label_visibility="collapsed")
    st.divider()
    st.text_input("Reviewer ID or pseudonym", key="reviewer", help="Recorded with SME decisions. Do not enter credentials.")
    st.caption("Session data is lost on refresh or disconnect. Export a project bundle to keep your work.")

st.title("Item Bank Analytics & Cognitive Feature Decomposition")
if ss.notice:
    st.success(ss.notice)
    ss.notice = None
if ss.stale_notice:
    st.warning(ss.stale_notice)

recs = ss.records or []
index = ss.index
siblings = ta.accepted_siblings(ss.task_review) if ss.task_review else []
features = [f for f in (ss.store or {}).get("features", {}).values() if f["review_status"] != "merged"]
reviewed = [f for f in features if f["review_status"] not in ("pending", "needs_adjudication")]
key_val, key_src = api_key()
tokens_used = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) + u.get("tokens", 0) for u in ss.usage_log)
cols = st.columns(7)
cols[0].metric("Items", len(recs), help=ss.file_name or "No file loaded")
cols[1].metric("Embedded", int(index.valid.sum()) if index is not None else 0,
               help=f"{index.model} · {index.dimensions} dims" if index is not None else "No vectors")
cols[2].metric("Retrieved", len(ss.search["results"]) if ss.search else 0)
cols[3].metric("Confirmed siblings (N)", len(siblings))
cols[4].metric("Features", len(features))
cols[5].metric("SME-reviewed", f"{100 * len(reviewed) / len(features):.0f}%" if features else "—")
cols[6].metric("API", "mock" if mock_mode() else ("key ready" if key_val else "no key"), help=f"OpenAI key: {key_src}")
if tokens_used:
    mock_only = all("mock" in u["kind"] or str(u.get("model", "")).startswith("mock") for u in ss.usage_log)
    st.caption(f"Estimated usage this session: about {tokens_used:,} tokens"
               + (" (mock mode — no API calls were made)." if mock_only else " (see provider billing for actual cost)."))


# =========================================================================== Data & API

def page_data():
    st.header("1 · Data & API")
    src_col, demo_col = st.columns([3, 1])
    with src_col:
        upload = st.file_uploader("Upload an item bank (.csv or .xlsx, up to 1,000 items)", type=["csv", "xlsx", "xls"])
    with demo_col:
        st.write("")
        if st.button("Load synthetic demo bank", help="12 invented items for demonstration only.", width="stretch"):
            _load_bytes(SAMPLE.read_bytes(), SAMPLE.name)
    if upload is not None:
        data = upload.getvalue()
        h = hashlib.sha256(data).hexdigest()
        sheets = []
        try:
            sheets = ingest.list_sheets(data, upload.name)
        except ingest.IngestError as exc:
            st.error(str(exc))
        sheet = st.selectbox("Worksheet", sheets) if len(sheets) > 1 else (sheets[0] if sheets else None)
        if h != ss.file_hash or sheet != ss.sheet:
            if ss.records and h != ss.file_hash:
                st.warning("Loading a new file replaces the current data. Downstream reviews will be archived.")
            if st.button("Load this file", type="primary"):
                _load_bytes(data, upload.name, sheet)
    with st.expander("Import a project bundle (.zip)"):
        bundle = st.file_uploader("Project bundle", type=["zip"], key="bundle_upload")
        if bundle is not None and st.button("Import bundle"):
            try:
                loaded = storage.import_bundle(bundle.getvalue())
            except storage.BundleError as exc:
                st.error(str(exc))
            else:
                _apply_bundle(loaded, bundle.name)

    if ss.raw_df is None:
        st.info("Upload a file, load the synthetic demo bank, or import a project bundle.")
        return
    df = ss.raw_df
    st.subheader("Schema")
    st.caption(f"{ss.file_name} · {len(df)} rows · {len(df.columns)} columns")
    st.dataframe(ingest.profile_columns(df), hide_index=True, width="stretch")
    with st.expander("Preview all columns"):
        st.dataframe(df, hide_index=True, width="stretch")
    _mapping_form(df)
    if ss.records:
        _report()
        _api_and_embeddings()


def _load_bytes(data, name, sheet=None):
    try:
        df = ingest.read_table(data, name, sheet)
    except ingest.IngestError as exc:
        st.error(str(exc))
        return
    if ss.records:
        invalidate_for_new_data("A new file was loaded.")
    ss.raw_df, ss.file_name, ss.file_hash, ss.sheet = df, name, hashlib.sha256(data).hexdigest(), sheet
    ss.mapping = ingest.detect_mapping(df)
    ss.records, ss.report, ss.index = None, None, None
    for k in [k for k in ss.keys() if str(k).startswith("map_")]:
        del ss[k]
    ss.notice = f"Loaded {name}. Review the suggested column mapping, then apply it."
    st.rerun()


def _apply_bundle(loaded, name):
    if ss.records:
        invalidate_for_new_data("A project bundle was imported.")
    ss.records = loaded["records"]
    ss.mapping = loaded["mapping"]
    ss.index = loaded["index"]
    ss.raw_df = ingest.records_frame(loaded["records"])
    ss.file_name = f"{name} (bundle of {loaded['manifest'].get('source_name') or 'unknown source'})"
    ss.file_hash = None
    ss.report = {"items": len(loaded["records"]), "template": loaded["index"].template if loaded["index"] else "",
                 "exact_duplicates_removed": [], "skipped_empty_text": [], "warnings": [], "key_status": {}}
    review = loaded["review"]
    ss.task_review = review.get("task_review")
    ss.store = review.get("store")
    ss.enemy_decisions = review.get("enemy_decisions", {})
    ss.class_overrides = review.get("class_overrides", {})
    ss.search = _search_from_bundle(review.get("search"))
    ss.archived_reviews = review.get("archived_reviews", [])
    for k in [k for k in ss.keys() if str(k).startswith("map_")]:
        del ss[k]
    bump_store()
    ss.notice = "Project bundle imported. Vectors, mapping and reviews were validated against the bundle manifest."
    st.rerun()


def _search_from_bundle(data):
    if not data:
        return None
    data = dict(data)
    data["results"] = pd.DataFrame(data["results"])
    return data


def _opt(cols, current):
    return [NONE] + cols, (cols.index(current) + 1 if current in cols else 0)


def _mapping_form(df):
    st.subheader("Column mapping")
    m = ss.mapping
    cols = list(df.columns)
    with st.form("mapping_form"):
        text_cols = st.multiselect("Text columns to embed (in this order)", cols, default=[c for c in m.text_columns if c in cols],
                                   key="map_text", help="Default: vignette, stem and lead-in. The key and labels are never embedded by default.")
        option_cols = st.multiselect("Answer option columns (any number)", cols, default=[c for c in m.option_columns if c in cols], key="map_opts")
        include_opts = st.toggle("Include option text in the embedded text", value=m.include_options_in_embed, key="map_inc",
                                 help="Increases lexical overlap and can surface answer-option leakage. The key marker is never embedded.")
        c1, c2, c3 = st.columns(3)
        picks = {}
        roles = [("item_id", "Item ID", c1), ("key", "Answer key", c1), ("domain", "Domain", c1), ("task", "Task", c1),
                 ("category", "Category / blueprint", c2), ("difficulty", "Difficulty (e.g. Rasch b)", c2),
                 ("p_value", "Classical p-value", c2), ("point_biserial", "Point-biserial", c2),
                 ("enemy", "Enemy list", c3), ("rationale", "Rationale", c3), ("family", "Linked set / stimulus ID", c3),
                 ("scale_group", "Scale / calibration group", c3)]
        for role, label, col in roles:
            options, idx = _opt(cols, getattr(m, role))
            picks[role] = col.selectbox(label, options, index=idx, key=f"map_{role}")
        categorical = st.multiselect("Treat these numeric-looking columns as categorical", cols,
                                     default=[c for c, t in m.type_overrides.items() if t == "categorical"], key="map_cat")
        submitted = st.form_submit_button("Apply mapping", type="primary")
    if not submitted:
        return
    new = ingest.ColumnMapping(
        text_columns=text_cols, option_columns=option_cols, include_options_in_embed=include_opts,
        type_overrides={c: "categorical" for c in categorical},
        **{r: (None if v == NONE else v) for r, v in picks.items()},
    )
    try:
        records, report = ingest.build_records(df, new)
    except ingest.IngestError as exc:
        st.error(str(exc))
        return
    old_hashes = [r["embed_text_hash"] for r in ss.records] if ss.records else None
    if old_hashes is not None and old_hashes != [r["embed_text_hash"] for r in records]:
        invalidate_for_new_data("The embedded text changed, so earlier search and review results are stale.")
    ss.mapping, ss.records, ss.report = new, records, report
    if ss.index is not None:
        model, dims, fp = embed_settings()
        if not ss.index.matches(records, fp):
            ss.index = None
            ss.notice = "Mapping applied. The embedded text or settings changed, so the previous vectors were cleared."
            st.rerun()
    ss.notice = f"Mapping applied: {len(records)} unique items."
    st.rerun()


def _report():
    r = ss.report
    st.subheader("Normalized items")
    c = st.columns(4)
    c[0].metric("Unique items", r["items"])
    c[1].metric("Exact duplicates removed", len(r["exact_duplicates_removed"]))
    c[2].metric("Rows without embed text", len(r["skipped_empty_text"]))
    c[3].metric("Keys resolved", sum(v for k, v in r["key_status"].items() if k in ("single", "multiple")))
    if r.get("synthetic_ids"):
        st.info("No item ID column is mapped; stable IDs were created from source row numbers (ROW-0002…).")
    for w in r["warnings"]:
        st.warning(w)
    if r["exact_duplicates_removed"]:
        st.caption("Removed exact duplicates: " + ", ".join(f"row {d['source_row']} = row {d['duplicate_of_row']}" for d in r["exact_duplicates_removed"]))
    if r["skipped_empty_text"]:
        st.caption("Skipped (no text in embedded columns): " + ", ".join(f"{d['item_id']} (row {d['source_row']})" for d in r["skipped_empty_text"]))
    if r.get("key_status"):
        st.caption("Key status: " + ", ".join(f"{k} {v}" for k, v in r["key_status"].items()))
    st.caption(f"Embedding template: `{r['template']}`")
    with st.expander("Embedded text preview (first 3 items)"):
        for rec in ss.records[:3]:
            st.text(f"[{rec['item_id']}]\n{rec['selected_embed_text']}")
    with st.expander("Normalized item preview"):
        st.dataframe(ingest.records_frame(ss.records), hide_index=True, width="stretch")
        pick = st.selectbox("Show item", [r["item_id"] for r in ss.records], key="preview_item")
        item_card(records_by_id()[pick])


def _api_and_embeddings():
    st.subheader("API mode, consent and embeddings")
    st.radio("Analysis mode", ["Offline mock (no network)", "OpenAI"], key="llm_mode", horizontal=True,
             help="Mock mode runs the whole workflow offline with deterministic lexical vectors and heuristic proposals. "
                  "Its output is labelled and is not AI or SME analysis.")
    if mock_mode():
        st.info("Mock mode: no external requests are made. Vectors are lexical hashes and proposals are heuristics, for demonstration only.")
    else:
        key, src = api_key()
        st.text_input("OpenAI API key (kept in this session only)", type="password", key="api_key_input",
                      help="Not written to disk, exports, logs or the URL. Streamlit secrets or the OPENAI_API_KEY environment variable are used when this is empty.")
        st.caption(f"Key source: {src}.")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Embedding model", key="embed_model")
        c2.number_input("Embedding dimensions (0 = model default)", min_value=0, max_value=3072, step=64, key="embed_dims",
                        help="Only for models that support shortened embeddings.")
        c3.text_input("Model for task and feature analysis", key="llm_model", placeholder="A model ID enabled for your account",
                      help="Must support structured JSON-schema output.")
        st.warning("**Privacy notice.** Generating embeddings sends the embedded text of every item to OpenAI. Task proposal and feature "
                   "decomposition send only the target and the candidates you confirm. Item content may be proprietary; you can "
                   "decline and continue exploring metadata locally.")
        st.checkbox("I consent to sending item embedding text to OpenAI for embeddings", key="consent_embed")
        st.checkbox("I consent to sending the target and confirmed items to OpenAI for task and feature analysis", key="consent_llm")

    model, dims, fp = embed_settings()
    texts = [r["selected_embed_text"] for r in ss.records]
    current = ss.index is not None and ss.index.matches(ss.records, fp)
    todo = len(texts) if not current else int((~ss.index.valid).sum())
    est = em.estimate_tokens(texts)
    cost = em.estimate_cost(model, est)
    st.caption(f"{len(texts)} items · about {est:,} tokens" + (f" · approx. ${cost:.4f} at list price" if cost is not None and not mock_mode() else "")
               + f" · model {model}" + (f" · {dims} dims" if dims else ""))
    if ss.index is not None and not current:
        st.warning("The existing vectors were built with different text or settings. Regenerating replaces them; search results become stale.")
    if current and todo == 0:
        st.success(f"All {len(texts)} items are embedded with {ss.index.model} ({ss.index.dimensions} dims). Search works offline from here.")
    label = "Generate embeddings" if not current else f"Retry {todo} failed items"
    if (not current or todo) and st.button(label, type="primary", disabled=not mock_mode() and not ss.consent_embed):
        _generate(model, dims, fp)
    if not mock_mode() and not ss.consent_embed:
        st.caption("Embedding is disabled until you give consent.")
    if ss.index is not None and ss.index.errors:
        with st.expander(f"{len(ss.index.errors)} embedding issue(s)"):
            for e in ss.index.errors:
                st.write(f"{e['error']}: {', '.join(e['item_ids'][:20])}")


def _generate(model, dims, fp):
    try:
        embedder = get_embedder()
    except (em.ConsentError, em.EmbeddingError) as exc:
        st.error(str(exc))
        return
    bar = st.progress(0.0, text="Embedding…")
    previous = ss.index
    try:
        new = em.generate_embeddings(ss.records, embedder, model, dims, fp, ss.report["template"], existing=previous,
                                     progress=lambda d, n: bar.progress(d / n, text=f"Embedded {d}/{n}"))
    except em.EmbeddingError as exc:
        st.error(f"{exc} Existing vectors were kept.")
        return
    if previous is not None and previous.settings_fingerprint != fp and ss.search:
        invalidate_for_new_data("Embeddings were regenerated with different settings.")
    ss.index = new
    ss.proj_cache = {}
    reused = previous.usage_tokens if previous is not None and previous.settings_fingerprint == fp else 0
    log_usage("embedding (mock estimate)" if mock_mode() else "embedding", model, {"tokens": max(0, new.usage_tokens - reused)})
    ss.notice = f"Embedded {int(new.valid.sum())}/{len(ss.records)} items." + (" Some items failed; use Retry." if new.errors else "")
    st.rerun()


# =========================================================================== Search

def page_search():
    st.header("2 · Cosine similarity search")
    require_records()
    require_index()
    by_id = records_by_id()
    st.caption("Ranking is exact cosine similarity in the original embedding space. Keyword ranking is optional and secondary.")
    source = st.radio("Target", ["Item from the bank", "New item text"], horizontal=True, key="target_source")
    target = None
    if source == "Item from the bank":
        ids = [r["item_id"] for r in ss.records]
        tid = st.selectbox("Target item", ids, key="target_pick",
                           format_func=lambda i: f"{i} — {viz._preview(by_id[i], 80)}")
        target = by_id[tid]
    else:
        text = st.text_area("Target item text (vignette, stem and lead-in)", key="new_text", height=120)
        opts = st.text_area("Options (optional, one per line, e.g. 'A. option text')", key="new_opts", height=100)
        key = st.text_input("Key (optional option ID)", key="new_key")
        target = _new_target(text, opts, key)
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Filter", ["Top K", "Minimum cosine similarity", "Both"], key="search_mode")
    k = c2.number_input("Top K (unique items)", 1, max(1, len(ss.records)), min(5, len(ss.records)), key="search_k")
    thr = c3.slider("Minimum cosine similarity", -1.0, 1.0, rt.DEFAULT_THRESHOLD, 0.01, key="search_thr",
                    help="0.85 is an illustrative, adjustable heuristic — not a probability and not proof of the same task.")
    with st.expander("Metadata filter and optional keyword ranking"):
        meta_cols = [NONE] + list(ss.records[0]["metadata"].keys())
        fcol = st.selectbox("Restrict candidates by metadata column", meta_cols, key="search_fcol")
        allowed = None
        if fcol != NONE:
            values = sorted({str(r["metadata"].get(fcol)) for r in ss.records})
            chosen = st.multiselect("Allowed values", values, default=values, key="search_fvals")
            allowed = [r["item_id"] for r in ss.records if str(r["metadata"].get(fcol)) in chosen]
            st.caption(f"{len(allowed)} candidate items match this filter.")
        use_bm25 = st.checkbox("Add BM25 keyword scores (secondary)", key="search_bm25")
        hybrid = st.checkbox("Rank by RRF fusion of cosine and BM25 ranks", key="search_hybrid", disabled=not use_bm25,
                             help="RRF is a rank-fusion measure, not a similarity. Any threshold still applies to cosine only.")
    if st.button("Search", type="primary", disabled=target is None):
        _run_search(target, {"Top K": "top_k", "Minimum cosine similarity": "threshold", "Both": "both"}[mode],
                    int(k), float(thr), allowed, use_bm25, hybrid)
    _show_search()


def _new_target(text, opts, key):
    if not text.strip():
        return None
    options = []
    for n, line in enumerate([l for l in opts.splitlines() if l.strip()]):
        m = re.match(r"^\s*([A-Za-z])[.)]\s*(.+)$", line)
        options.append({"option_id": m.group(1).upper() if m else chr(65 + n), "text": (m.group(2) if m else line).strip(),
                        "is_key": None, "source_column": None})
    key_ids, status = ingest.parse_key(key, options)
    if status in ("single", "multiple"):
        for o in options:
            o["is_key"] = o["option_id"] in key_ids
    return {"internal_id": ta.NEW_TARGET_ID, "item_id": ta.NEW_TARGET_ID, "source_row": None,
            "fields": {"Item text": text.strip()}, "field_order": ["Item text"], "options": options,
            "key_raw": key or None, "key_status": status, "roles": {}, "metadata": {},
            "selected_embed_text": f"Item text: {text.strip()}", "embed_text_hash": None, "source_fingerprint": None}


def _run_search(target, mode, k, thr, allowed, use_bm25, hybrid):
    index = ss.index
    if target["item_id"] == ta.NEW_TARGET_ID:
        if index.model == em.MOCK_MODEL and not mock_mode():
            st.error("These vectors were built in mock mode. Switch to mock mode or regenerate embeddings with OpenAI.")
            return
        if index.model != em.MOCK_MODEL and mock_mode():
            st.error("These vectors were built with OpenAI. Switch to OpenAI mode to embed new text with the same model.")
            return
        try:
            q = em.embed_query(target["selected_embed_text"], get_embedder(), index)
        except Exception as exc:  # noqa: BLE001
            st.error(f"The query could not be embedded: {em.safe_error(exc) if not isinstance(exc, (em.ConsentError, em.EmbeddingError)) else exc}")
            return
        exclude = None
    else:
        pos = index.position(target["item_id"])
        if pos is None or not index.valid[pos]:
            st.error("The target item has no valid vector. Retry embeddings for it first.")
            return
        q = index.vectors[pos]
        exclude = target["item_id"]
    scores = rt.cosine_scores(index, q)
    if use_bm25:
        bm = rt.BM25([r["selected_embed_text"] for r in ss.records]).scores(target["selected_embed_text"])
        if hybrid:
            results = rt.hybrid_search(scores, bm, index.item_ids, mode, k, thr, exclude, allowed)
        else:
            results = rt.select(scores, index.item_ids, mode, k, thr, exclude, allowed)
            bm_by = dict(zip(index.item_ids, bm))
            results["bm25_score"] = results["item_id"].map(bm_by)
    else:
        results = rt.select(scores, index.item_ids, mode, k, thr, exclude, allowed)
    old = ss.task_review
    if old and (old["target_id"] != target["item_id"] or old["candidate_ids"] != list(results["item_id"])):
        if any(v != "unreviewed" for v in old["decisions"].values()) or (ss.store and ss.store["features"]):
            ss.archived_reviews.append({"reason": "A new search replaced the candidate set.", "task_review": old,
                                        "store": ss.store, "search_target": old["target_id"]})
            ss.notice = "The previous task and feature review was archived (available in Exports)."
        ss.task_review, ss.store = None, None
        bump_store()
    ss.search = {"target": target, "results": results, "mode": mode, "k": k, "threshold": thr,
                 "bm25": use_bm25, "hybrid": hybrid, "filtered": allowed is not None,
                 "scores": {iid: float(s) for iid, s in zip(index.item_ids, scores) if s == s and iid != exclude}}
    st.rerun()


def _show_search():
    s = ss.search
    if not s:
        return
    by_id = records_by_id()
    res = s["results"]
    st.subheader(f"{len(res)} unique candidate item(s)")
    label = {"top_k": f"Top {s['k']}", "threshold": f"cosine ≥ {s['threshold']:.2f}", "both": f"cosine ≥ {s['threshold']:.2f}, then top {s['k']}"}[s["mode"]]
    st.caption(f"Filter: {label}" + (" · RRF hybrid ranking (threshold applied to cosine)" if s["hybrid"] else "")
               + (" · metadata filter on" if s["filtered"] else ""))
    with st.expander("Similarity distribution across thresholds"):
        st.dataframe(rt.threshold_counts(list(s["scores"].values())), hide_index=True)
        st.caption("Counts of items at or above each cosine value. They describe this retrieval only; they are not same-task probabilities.")
    if res.empty:
        st.warning("No items met the filter. Lower the minimum cosine similarity, raise K, or widen the metadata filter. Results are never padded.")
    else:
        table = res.copy()
        for role in ("domain", "task"):
            table[role] = table["item_id"].map(lambda i: by_id[i]["roles"].get(role))
        table["options"] = table["item_id"].map(lambda i: len(by_id[i]["options"]))
        table["key_status"] = table["item_id"].map(lambda i: by_id[i]["key_status"])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"cosine_similarity": st.column_config.NumberColumn(format="%.4f"),
                                    "hybrid_score": st.column_config.NumberColumn("hybrid_score (RRF rank measure)", format="%.5f")})
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Target**")
        item_card(s["target"], title=f"**Target · {md(s['target']['item_id'])}**")
    with right:
        st.markdown("**Candidates**")
        for _, row in res.iterrows():
            with st.expander(f"{row['item_id']} · cosine {row['cosine_similarity']:.3f}"):
                item_card(by_id[row["item_id"]], cosine=row["cosine_similarity"])
    c1, c2 = st.columns(2)
    c1.download_button("Search results (CSV)", storage.to_csv_bytes(_results_export()), "search_results.csv", "text/csv")
    c2.download_button("Search results (JSON)", storage.to_json_bytes(_results_export().to_dict("records")), "search_results.json", "application/json")
    st.info("Next: confirm which candidates measure the same task in **Decomposition**.")


def _results_export():
    s = ss.search
    by_id = records_by_id()
    rows = []
    for _, r in s["results"].iterrows():
        rec = by_id[r["item_id"]]
        row = {**r.to_dict(), "target_item_id": s["target"]["item_id"]}
        row.update({f"field_{k}": v for k, v in rec["fields"].items()})
        row.update({f"option_{o['option_id']}": o["text"] for o in rec["options"]})
        row["key"] = ",".join(o["option_id"] for o in rec["options"] if o["is_key"]) or None
        row.update({k: v for k, v in rec["metadata"].items() if k not in row})
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== Decomposition

def page_decomposition():
    st.header("3 · Target task, sibling confirmation and feature decomposition")
    require_records()
    if not ss.search:
        st.info("Run a search first.")
        return
    by_id = records_by_id()
    target = ss.search["target"]
    cand_ids = list(ss.search["results"]["item_id"])
    if ss.task_review is None:
        ss.task_review = ta.new_review(target["item_id"], cand_ids)
    review = ss.task_review
    reviewer = ss.reviewer or "anonymous"

    st.subheader("Step 1 · Target task")
    llm_blocked = not mock_mode() and not (ss.consent_llm and ss.get("llm_model"))
    if llm_blocked:
        st.caption("OpenAI analysis needs a model ID and consent in **Data & API**.")
    if st.button("Propose target task and candidate alignment", disabled=llm_blocked,
                 help=f"Sends the target and {len(cand_ids)} retrieved candidate(s)."):
        try:
            run = ta.propose_task(get_llm(), target, [(by_id[c], ss.search["scores"].get(c)) for c in cand_ids])
        except (LLMError, em.ConsentError) as exc:
            st.error(str(exc))
        else:
            ta.attach_proposal(review, run, reviewer)
            log_usage("task", run["model"], run["usage"])
            st.rerun()
    alignment = ta.latest_alignment(review)
    if review["runs"]:
        run = review["runs"][-1]
        prop = run["proposal"]["target_task"]
        with st.container(border=True):
            st.markdown(f"**Proposed task:** {md(prop['name'])}  \n**Decision/action:** {md(prop['decision_or_action'])}  \n"
                        f"**Domain:** {md(prop['domain'])}  \n**Rationale:** {md(prop['rationale'])}")
            for span in prop["evidence_spans"]:
                st.caption(f"Evidence · {md(span['item_id'])} · {md(span['field'])}: “{md(span['excerpt'])}” ({span['evidence_type']})")
            for flag in prop.get("ambiguity_flags", []):
                st.warning(f"Ambiguity: {flag}")
            for issue in run["issues"]:
                st.warning(f"Validation: {issue}")
            st.caption(f"Model: {run['model']} · {run['created_at']}")
    if review["task"] is not None:
        with st.form("task_form"):
            t = review["task"]
            name = st.text_input("Target task (cognitive demand)", t.get("name", ""))
            action = st.text_input("Decision or action required", t.get("decision_or_action", ""))
            domain = st.text_input("Content domain (kept separate from the task)", t.get("domain", ""))
            rationale = st.text_area("Rationale", t.get("rationale", ""), height=80)
            c1, c2 = st.columns(2)
            accept = c1.form_submit_button("Accept task", type="primary")
            reject = c2.form_submit_button("Reject task")
        if accept or reject:
            edits = {"name": name, "decision_or_action": action, "domain": domain, "rationale": rationale}
            ta.set_task(review, reviewer, "accepted" if accept else "rejected", edits if accept else None)
            st.rerun()
        st.caption(f"Task status: **{review['task_status']}**")
    elif not review["runs"]:
        if st.button("Write the task manually"):
            review["task"] = {"name": "", "decision_or_action": "", "domain": "", "rationale": "Entered by the SME."}
            st.rerun()

    st.subheader("Step 2 · Confirm same-task siblings")
    st.caption("High cosine similarity is retrieval evidence only. Mark each candidate after comparing it with the target.")
    for cid in cand_ids:
        rec = by_id[cid]
        a = alignment.get(cid)
        with st.expander(f"{cid} · cosine {ss.search['scores'].get(cid, float('nan')):.3f} · SME: {review['decisions'][cid]}"
                         + (f" · AI: {a['proposed_alignment']}" if a else ""), expanded=False):
            left, right = st.columns(2)
            with left:
                item_card(target, title=f"**Target · {md(target['item_id'])}**")
            with right:
                item_card(rec)
            if a:
                st.markdown(f"**AI apparent task:** {md(a['apparent_task'])} · **proposal:** {a['proposed_alignment']} ({a['confidence_status']})  \n"
                            f"**Evidence:** {md(a['evidence'])}")
            choice = st.radio("SME decision", SME_ALIGNMENTS, index=SME_ALIGNMENTS.index(review["decisions"][cid]),
                              key=f"align_{target['item_id']}_{cid}", horizontal=True,
                              format_func=lambda v: v.replace("_", " "))
            if choice != review["decisions"][cid]:
                ta.set_decision(review, reviewer, cid, choice)
                st.rerun()
    include_all = st.toggle("Use all retrieved candidates as the denominator (recorded in the report)",
                            value=review["include_all_retrieved"], key="include_all")
    if include_all != review["include_all_retrieved"]:
        ta.set_include_all(review, reviewer, include_all)
        st.rerun()
    sib = ta.accepted_siblings(review)
    st.info(f"Denominator N = {len(sib)} distinct {'retrieved' if review['include_all_retrieved'] else 'SME-confirmed'} sibling(s)"
            f"{': ' + ', '.join(sib) if sib else ''}. The target is reported separately and is not counted in N.")

    st.subheader("Step 3 · Extract features")
    if not ta.task_confirmed(review):
        st.warning("Accept or edit the target task before extracting features.")
    else:
        payloads = dc.build_payloads(target, [by_id[i] for i in sib], review["task"])
        st.caption(f"Sends the confirmed task, the target and {len(sib)} sibling(s) in {len(payloads)} bounded request(s). "
                   "SME-reviewed features are never overwritten; new evidence for them waits for your confirmation.")
        if st.button("Extract features", type="primary", disabled=llm_blocked):
            if ss.store is None:
                ss.store = dc.new_store()
            try:
                run, result = dc.decompose(get_llm(), target, [by_id[i] for i in sib], review, ss.store, reviewer)
            except (dc.DecompositionGateError, em.ConsentError, LLMError) as exc:
                st.error(str(exc))
            else:
                log_usage("decomposition", run["model"], run["usage"])
                bump_store()
                msg = f"{len(result['added'])} new, {len(result['refreshed'])} refreshed, {len(result['pending'])} awaiting your confirmation."
                if run["errors"]:
                    st.session_state.notice = msg + " Some chunks failed: " + "; ".join(run["errors"])
                else:
                    ss.notice = msg
                st.rerun()
    if ss.store and ss.store["features"]:
        _feature_review(target, sib)


def _feature_review(target, sib):
    store = ss.store
    reviewer = ss.reviewer or "anonymous"
    by_id = records_by_id()
    st.subheader("Step 4 · SME feature review")
    if ss.store.get("basis") and ss.store["basis"]["sibling_ids"] != sib:
        st.warning("The confirmed sibling set changed since the last extraction. Frequencies use the current set; "
                   "re-extract to propose features for newly confirmed items.")
    c1, c2 = st.columns(2)
    view = c1.radio("Show", ["all", "pending", "accepted", "rejected"], horizontal=True, key="feat_view",
                    format_func=lambda v: {"all": "All proposed", "pending": "Needs review", "accepted": "Accepted", "rejected": "Rejected/merged"}[v])
    source = c2.radio("Frequency basis", ["sme", "raw"], horizontal=True, key="freq_source",
                      format_func=lambda v: "SME-approved incidence" if v == "sme" else "Raw AI incidence")
    freq = ff.frequency_table(store, sib, target["item_id"], source="raw")
    freq_sme = ff.frequency_table(store, sib, target["item_id"], source="sme").set_index("feature_id")
    raw_by = freq.set_index("feature_id")
    st.caption("Importance anchors: " + " · ".join(IMPORTANCE_ANCHORS.values()) + ". Ratings are independent of how many features exist.")
    tabs = st.tabs([FEATURE_TYPE_LABELS[t] for t in FEATURE_TYPES])
    for tab, ftype in zip(tabs, FEATURE_TYPES):
        with tab:
            rows = dc.features_frame_rows(store, [ftype], "all")
            if view == "pending":
                rows = [f for f in rows if f["review_status"] in ("pending", "needs_adjudication")]
            elif view == "accepted":
                rows = [f for f in rows if f["review_status"] in ("accepted", "edited")]
            elif view == "rejected":
                rows = [f for f in rows if f["review_status"] in ("rejected", "not_required", "unsupported", "merged")]
            if not rows:
                st.caption("No features in this view.")
                continue
            _feature_table(rows, ftype, raw_by, freq_sme, target["item_id"], len(sib), reviewer)
    _feature_tools(store, reviewer, sib, target)
    st.markdown("**Feature frequency** — each feature counts once per distinct confirmed sibling.")
    shown = ff.frequency_table(store, sib, target["item_id"], source=source)
    if shown.empty:
        st.caption("No features meet this basis yet (SME-approved basis counts only accepted or edited features).")
    else:
        st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Frequencies describe the current confirmed set. They are not population prevalence or evidence that a feature drives difficulty.")


def _feature_table(rows, ftype, raw_by, sme_by, target_id, n, reviewer):
    def cov(fid, table):
        if fid in table.index:
            return table.loc[fid, "coverage"]
        return f"0/{n} = 0%" if n else "N/A"

    data = pd.DataFrame([{
        "feature_id": f["feature_id"], "canonical_label": f["canonical_label"], "definition": f["definition"],
        "review_status": f["review_status"], "importance": "unrated" if f["sme_importance"] is None else str(f["sme_importance"]),
        "dominance": f.get("dominance") or "", "procedure_group": f.get("procedure_group") or "",
        "sequence_order": f.get("sequence_order"), "sme_note": f.get("sme_note") or "",
        "SME coverage": cov(f["feature_id"], sme_by), "raw AI coverage": cov(f["feature_id"], raw_by),
        "in target": target_id in f["supporting_items"], "supporting items": ", ".join(f["supporting_items"]),
        "options": ", ".join(f"{r['item_id']}:{r['option_id']}" for r in f["option_refs"]),
        "evidence": ", ".join(f.get("evidence_types", [])), "issues": "; ".join(f.get("validation_issues", [])),
    } for f in rows])
    columns = ["feature_id", "canonical_label", "definition", "review_status", "importance"]
    if ftype == "procedural_key":
        columns += ["dominance", "procedure_group", "sequence_order"]
    columns += ["sme_note", "SME coverage", "raw AI coverage", "in target", "supporting items"]
    if ftype == "distractor_feature":
        columns += ["options"]
    columns += ["evidence", "issues"]
    statuses = [s for s in REVIEW_STATUSES if s != "merged"]
    edited = st.data_editor(
        data[columns], hide_index=True, width="stretch", key=f"fe_{ftype}_{ss.store_rev}",
        disabled=["feature_id", "SME coverage", "raw AI coverage", "in target", "supporting items", "options", "evidence", "issues"],
        column_config={
            "canonical_label": st.column_config.TextColumn("label", width="medium"),
            "definition": st.column_config.TextColumn(width="medium"),
            "review_status": st.column_config.SelectboxColumn("status", options=statuses + ["merged"], required=True),
            "importance": st.column_config.SelectboxColumn("importance 1–5", options=["unrated", "1", "2", "3", "4", "5"], required=True),
            "dominance": st.column_config.SelectboxColumn(options=list(DOMINANCE)),
            "sequence_order": st.column_config.NumberColumn("order", min_value=1, step=1),
        },
    )
    changed = False
    for fid, changes in dc.editor_changes(data[columns].to_dict("records"), edited.to_dict("records")).items():
        try:
            changed |= dc.update_feature(ss.store, fid, changes, reviewer)
        except ValueError as exc:
            st.error(f"{fid}: {exc}")
    if changed:
        bump_store()  # new editor key: positional edits must not re-apply to rows that moved
        st.rerun()


def _feature_tools(store, reviewer, sib, target):
    active = [fid for fid in store["order"] if store["features"][fid]["review_status"] != "merged"]
    label = lambda fid: f"{fid} · {store['features'][fid]['canonical_label'][:70]}"  # noqa: E731
    with st.expander("Provenance, pending AI updates and history"):
        fid = st.selectbox("Feature", active, format_func=label, key="prov_pick")
        f = store["features"][fid]
        st.markdown(f"**{md(f['canonical_label'])}** ({FEATURE_TYPE_LABELS[f['feature_type']]}) — status {f['review_status']}")
        if f.get("inference_rationale"):
            st.markdown(f"*Inference rationale:* {md(f['inference_rationale'])}")
        if f.get("proposed_misconception"):
            st.markdown(f"*Misconception hypothesis (unverified):* {md(f['proposed_misconception'])}")
        for s in f["source_spans"]:
            where = f"option {s['option_id']}" if s.get("option_id") else s["field"]
            quote = f"“{md(s['excerpt'])}”" if s["excerpt"] else "(no quotation — inferred)"
            st.caption(f"{md(s['item_id'])} · {md(where)} · {s['evidence_type']}: {quote}")
        for issue in f.get("validation_issues", []):
            st.warning(issue)
        if f.get("llm_uncertainty_note"):
            st.caption(f"Model note: {f['llm_uncertainty_note']}")
        with st.popover("Original AI proposal"):
            st.json(f["ai"])
        if fid in store["pending_updates"]:
            st.info("A later run proposed different evidence for this SME-reviewed feature.")
            st.json(store["pending_updates"][fid]["proposal"]["supporting_items"])
            c1, c2 = st.columns(2)
            if c1.button("Apply the new AI evidence", key=f"apply_{fid}"):
                dc.accept_pending_update(store, fid, reviewer)
                bump_store()
                st.rerun()
            if c2.button("Keep my reviewed version", key=f"discard_{fid}"):
                dc.discard_pending_update(store, fid, reviewer)
                st.rerun()
        new_type = st.selectbox("Change feature type", FEATURE_TYPES, index=FEATURE_TYPES.index(f["feature_type"]),
                                key=f"type_{fid}", format_func=FEATURE_TYPE_LABELS.get)
        if new_type != f["feature_type"] and st.button("Apply type change", key=f"apply_type_{fid}"):
            dc.update_feature(store, fid, {"feature_type": new_type}, reviewer)
            bump_store()
            st.rerun()
        items = [target["item_id"]] + sib
        support = st.multiselect("Supporting items (SME)", items, default=[i for i in f["supporting_items"] if i in items], key=f"sup_{fid}_{ss.store_rev}")
        if set(support) != set(i for i in f["supporting_items"] if i in items) and st.button("Save supporting items", key=f"save_sup_{fid}"):
            keep_other = [i for i in f["supporting_items"] if i not in items]
            dc.update_feature(store, fid, {"supporting_items": keep_other + support}, reviewer, known_items=set(records_by_id()) | {target["item_id"]})
            bump_store()
            st.rerun()
        hist = [h for h in store["history"] if fid in json.dumps(h.get("after"), default=str) or fid in json.dumps(h.get("before"), default=str)]
        if hist:
            st.dataframe(pd.DataFrame([{"timestamp": h["timestamp"], "reviewer": h["reviewer"], "action": h["action"]} for h in hist]),
                         hide_index=True)
    with st.expander("Merge or split features"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Merge** (same type; support is united, provenance kept)")
            keep = st.selectbox("Keep", active, format_func=label, key="merge_keep")
            same = [x for x in active if x != keep and store["features"][x]["feature_type"] == store["features"][keep]["feature_type"]]
            others = st.multiselect("Merge into it", same, format_func=label, key=f"merge_others_{ss.store_rev}")
            if st.button("Merge", disabled=not others):
                dc.merge_features(store, keep, others, reviewer)
                bump_store()
                st.rerun()
        with c2:
            st.markdown("**Split** (move some items' support to a new feature)")
            src = st.selectbox("Feature to split", active, format_func=label, key="split_src")
            move = st.multiselect("Items to move", store["features"][src]["supporting_items"], key=f"split_items_{ss.store_rev}")
            new_label = st.text_input("Label for the new feature", key=f"split_label_{ss.store_rev}")
            if st.button("Split", disabled=not (move and new_label.strip())):
                dc.split_feature(store, src, move, new_label, reviewer)
                bump_store()
                st.rerun()


# =========================================================================== Distractors

def page_distractors():
    st.header("4 · Distractor review")
    require_records()
    if not ss.search:
        st.info("Run a search and confirm siblings first.")
        return
    target = ss.search["target"]
    sib = ta.accepted_siblings(ss.task_review)
    by_id = records_by_id() | {target["item_id"]: target}
    table = ds.option_review(by_id, target["item_id"], sib, ss.store)
    st.caption("All options of the target and each confirmed sibling. Only incorrect options of items with a usable key contribute distractor features.")
    st.dataframe(table, hide_index=True, width="stretch")
    if ss.store:
        freq = ff.frequency_table(ss.store, sib, target["item_id"], source=ss.get("freq_source", "sme"))
        freq = freq[freq["feature_type"] == "distractor_feature"]
        st.subheader("Distractor feature space")
        st.caption("Item frequency counts each sibling once; option count counts distinct sibling options. They use different denominators.")
        if freq.empty:
            st.caption("No distractor features on this basis yet.")
        else:
            st.dataframe(freq[["feature_id", "canonical_label", "item_frequency", "denominator_N", "coverage",
                               "distractor_option_count", "in_target", "review_status", "sme_importance"]],
                         hide_index=True, width="stretch")
            hyps = [(fid, f) for fid, f in ss.store["features"].items()
                    if f["feature_type"] == "distractor_feature" and f.get("proposed_misconception") and f["review_status"] != "merged"]
            if hyps:
                with st.expander("Misconception hypotheses (require SME verification)"):
                    for fid, f in hyps:
                        st.markdown(f"**{fid}** {md(f['canonical_label'])} — {md(f['proposed_misconception'])}")


# =========================================================================== Visualization

def page_visualization():
    st.header("5 · Embedding and feature visualization")
    require_records()
    require_index()
    by_id = records_by_id()
    st.caption("2D positions are projections for display. Distances in the plot are not cosine similarities; search always ranks in the original space.")
    c1, c2, c3, c4 = st.columns(4)
    methods = ["pca", "tsne"] + (["umap"] if viz.umap_available() else [])
    method = c1.selectbox("Projection", methods, format_func={"pca": "PCA (fast)", "tsne": "t-SNE", "umap": "UMAP"}.get, key="viz_method")
    seed = c2.number_input("Random seed", 0, 10_000, 42, key="viz_seed")
    meta_cols = [NONE] + list(ss.records[0]["metadata"].keys())
    color_by = c3.selectbox("Colour by metadata", meta_cols, key="viz_color")
    kind = c4.radio("Colour type", ["auto", "categorical", "numeric"], horizontal=True, key="viz_kind")
    params = {"seed": int(seed)}
    if method == "tsne":
        params["perplexity"] = float(st.slider("Perplexity", 2.0, 50.0, 30.0, 1.0, key="viz_perp",
                                               help="Capped below the number of items. Cluster shape and global distances are unstable."))
    try:
        coords, meta = viz.get_projection(ss.proj_cache, ss.index, method, **params)
    except viz.ProjectionUnavailable as exc:
        st.warning(str(exc))
        return
    roles = {}
    if ss.search:
        for iid in ss.search["results"]["item_id"]:
            roles[iid] = "retrieved"
        if ss.task_review:
            for iid, d in ss.task_review["decisions"].items():
                roles[iid] = {"same_task": "accepted sibling", "different_task": "rejected candidate",
                              "uncertain": "uncertain candidate"}.get(d, "retrieved")
        roles[ss.search["target"]["item_id"]] = "target"
    frame = viz.plot_frame(coords, by_id, None if color_by == NONE else color_by, roles,
                           ss.search["scores"] if ss.search else None)
    hidden = None
    if color_by != NONE and (kind == "categorical" or (kind == "auto" and not viz.is_numeric(frame[color_by]))):
        cats = sorted(frame[color_by].map(lambda v: "(missing)" if v is None else str(v)).unique())
        shown = st.multiselect("Show categories", cats, default=cats, key=f"viz_cats_{color_by}")
        hidden = [c for c in cats if c not in shown]
    fig, used = viz.scatter_figure(frame, None if color_by == NONE else color_by, kind, hidden,
                                   title=f"{method.upper()} projection of {len(frame)} items")
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="viz_chart")
    if meta.get("explained_variance"):
        st.caption("PCA explained variance: " + ", ".join(f"{v:.1%}" for v in meta["explained_variance"]))
    picked = None
    try:
        points = event.selection.points if event else []
        if points:
            picked = points[0].get("customdata", [None])[0]
    except AttributeError:
        picked = None
    ids = list(frame["item_id"])
    choice = st.selectbox("Item details", ids, index=ids.index(picked) if picked in ids else 0, key="viz_item")
    item_card(by_id[choice])
    st.download_button("Plot data (CSV)", storage.to_csv_bytes(frame), f"projection_{method}.csv", "text/csv")

    if ss.store and ss.search:
        st.subheader("Item × feature incidence")
        target_id = ss.search["target"]["item_id"]
        sib = ta.accepted_siblings(ss.task_review)
        source = st.radio("Incidence", ["sme", "raw"], horizontal=True, key="heat_source",
                          format_func=lambda v: "SME-approved" if v == "sme" else "Raw AI")
        matrix = ff.incidence_matrix(ss.store, [target_id] + sib, source=source)
        fig = viz.heatmap_figure(matrix, ss.store)
        if fig is None:
            st.caption("No features on this basis yet.")
        else:
            st.plotly_chart(fig, width="stretch")
            st.caption("Rows: target then confirmed siblings. Columns grouped Declarative, Procedural, Instance, Distractor.")
        st.download_button("Incidence (long CSV)", storage.to_csv_bytes(ff.long_incidence(ss.store, target_id, sib, source)),
                           "item_feature_incidence.csv", "text/csv")


# =========================================================================== Difficulty

def page_difficulty():
    st.header("6 · Difficulty heuristic")
    require_records()
    if not ss.mapping.difficulty:
        st.info("Map an empirical difficulty column (for example Rasch b) in **Data & API** to use this section. "
                "A classical p-value is never substituted for difficulty.")
        return
    require_index()
    st.warning("This is a similarity-weighted neighbour average, not an IRT calibration. It needs empirical validation before any operational use.")
    confirmed = st.checkbox(f"I confirm the values in '{ss.mapping.difficulty}' are on one calibrated, linked scale with the same direction and units",
                            key="scale_confirmed")
    c1, c2 = st.columns(2)
    alpha = c1.slider("Weight exponent α (w = max(cosine, 0)^α)", 0.0, 5.0, 1.0, 0.5, key="diff_alpha")
    min_n = c2.number_input("Minimum labelled neighbours", 1, 20, 2, key="diff_min")
    if ss.search and not ss.search["results"].empty:
        by_id = records_by_id()
        res = pr.difficulty_estimate(ss.search["results"], by_id, alpha=alpha, min_neighbors=int(min_n), scale_confirmed=confirmed)
        st.subheader(f"Target {ss.search['target']['item_id']}")
        if res["status"] == "estimate":
            st.metric(f"Weighted neighbour estimate ({ss.mapping.difficulty})", f"{res['estimate']:.3f}")
        elif res["status"] == "exploratory":
            st.info(res["message"])
        else:
            st.warning(res["message"])
        if "unweighted_mean" in res:
            st.caption(f"Descriptive: unweighted mean {res['unweighted_mean']:.3f}, SD {res['sd'] if res['sd'] is None else round(res['sd'], 3)}, "
                       f"min {res['min']:.3f}, max {res['max']:.3f}, labelled N {res['n_labeled']} of {res['n_neighbors']} neighbours. "
                       "The SD describes the neighbours; it is not a confidence interval.")
        st.dataframe(res["table"], hide_index=True, width="stretch")
    else:
        st.info("Run a search to see neighbour difficulties for a target.")
    st.subheader("Grouped holdout diagnostics")
    st.caption("Each labelled item is predicted only from items outside its linked family or near-duplicate group, against a naive mean baseline.")
    k = st.number_input("Neighbours (k)", 1, 50, 5, key="diff_k")
    if st.button("Run holdout evaluation", disabled=not confirmed):
        out = pr.evaluate_difficulty(ss.index, ss.records, k=int(k), alpha=alpha, min_neighbors=int(min_n))
        if out["status"] != "ok":
            st.warning(out["message"])
        else:
            st.dataframe(out["summary"], hide_index=True)
            st.caption(f"{out['n_evaluated']} items evaluated across {out['n_groups']} groups; {out['n_abstained']} abstained. {out['message']}")
            if out["per_domain"] is not None:
                st.dataframe(out["per_domain"], hide_index=True)
            with st.expander("Per-item predictions"):
                st.dataframe(out["table"], hide_index=True, width="stretch")
    if not confirmed:
        st.caption("Holdout evaluation is available after you confirm scale comparability.")


# =========================================================================== Classification

def page_classification():
    st.header("7 · Classification heuristic")
    require_records()
    roles = [r for r in ("domain", "task", "category") if getattr(ss.mapping, r)]
    if not roles:
        st.info("Map a domain, task or category column in **Data & API** to use this section.")
        return
    require_index()
    role_labels = {r: f"{r} ({getattr(ss.mapping, r)})" for r in roles}
    role = st.selectbox("Label to predict", roles, format_func=role_labels.get, key="cls_role")
    warning = pr.label_leakage_warning(ss.mapping, role)
    if warning:
        st.warning(warning)
    st.caption("Predict only within one shared taxonomy. Domain and task are kept separate. The model never overwrites supplied labels.")
    c1, c2 = st.columns(2)
    margin = c1.slider("Abstain when the top two labels are within", 0.0, 0.5, 0.1, 0.05, key="cls_margin")
    min_n = c2.number_input("Minimum labelled neighbours", 1, 20, 2, key="cls_min")
    if ss.search and not ss.search["results"].empty:
        target = ss.search["target"]
        res = pr.classify(ss.search["results"], records_by_id(), role=role, min_labeled=int(min_n), margin=margin)
        st.subheader(f"Target {target['item_id']}")
        given = target.get("roles", {}).get(role)
        if given:
            st.caption(f"Supplied label: {md(given)} (kept as the label of record).")
        if res["status"] == "prediction":
            st.metric("Suggested label", res["prediction"])
        else:
            st.info(res["message"])
        if res["support"] is not None:
            st.dataframe(res["support"], hide_index=True)
        st.dataframe(res["table"], hide_index=True, width="stretch")
        options = sorted({lab for r in ss.records for lab in pr.split_labels(r["roles"].get(role))})
        with st.form("cls_override"):
            choice = st.selectbox("SME label decision", ["(no decision)"] + options, key="cls_choice")
            note = st.text_input("Note")
            if st.form_submit_button("Record decision") and choice != "(no decision)":
                ss.class_overrides[f"{target['item_id']}|{role}"] = {"label": choice, "note": note, "reviewer": ss.reviewer or "anonymous",
                                                                     "suggested": res["prediction"], "status": res["status"]}
                st.rerun()
        rec = ss.class_overrides.get(f"{target['item_id']}|{role}")
        if rec:
            st.success(f"SME decision recorded: {rec['label']} (suggestion was {rec['suggested'] or 'abstain'}).")
    else:
        st.info("Run a search to classify a target from its neighbours.")
    st.subheader("Grouped holdout diagnostics")
    k = st.number_input("Neighbours (k)", 1, 50, 5, key="cls_k")
    if st.button("Run holdout evaluation", key="cls_eval"):
        out = pr.evaluate_classification(ss.index, ss.records, role=role, k=int(k), min_labeled=int(min_n), margin=margin)
        if out["status"] not in ("ok",):
            st.warning(out.get("message", out["status"]))
        else:
            c = st.columns(3)
            c[0].metric("Accuracy (decided)", f"{out['accuracy']:.2f}")
            c[1].metric("Balanced accuracy", f"{out['balanced_accuracy']:.2f}")
            c[2].metric("Macro-F1", f"{out['macro_f1']:.2f}")
            st.caption(f"{out['n_abstained']} abstained · {out['n_multi_label_skipped']} multi-label items skipped. {out['message']}")
            st.dataframe(out["confusion"], width="stretch")
            st.dataframe(out["per_class_recall"], hide_index=True)
            st.dataframe(out["class_counts"], hide_index=True)
            if not out["misclassified"].empty:
                with st.expander("Misclassified items"):
                    st.dataframe(out["misclassified"], hide_index=True)


# =========================================================================== Enemies

def page_enemies():
    st.header("8 · Enemy-item candidates")
    require_records()
    st.warning("Every flag is a candidate for SME review. Similarity alone never establishes an enemy pair or an exposure risk.")
    scope_opts = ["Whole bank"] + (["Target and retrieved candidates"] if ss.search else [])
    scope = st.radio("Scope", scope_opts, horizontal=True, key="enemy_scope")
    c = st.columns(4)
    cos_t = c[0].number_input("High cosine ≥", 0.0, 1.0, en.DEFAULTS["cos_threshold"], 0.01, key="en_cos")
    stem_t = c[1].number_input("Stem token overlap ≥", 0.0, 1.0, en.DEFAULTS["stem_jaccard"], 0.05, key="en_stem")
    opt_t = c[2].number_input("Option-set overlap ≥", 0.0, 1.0, en.DEFAULTS["option_jaccard"], 0.05, key="en_opt")
    val_t = c[3].number_input("Shared precise values ≥", 1, 10, en.DEFAULTS["shared_values_min"], key="en_val")
    if st.button("Screen for candidates", type="primary"):
        ids = None
        if scope != "Whole bank":
            ids = [ss.search["target"]["item_id"]] + list(ss.search["results"]["item_id"])
        try:
            ss.enemy_table = en.enemy_candidates(ss.records, ss.index, ids, cos_threshold=cos_t, stem_jaccard=stem_t,
                                                 option_jaccard=opt_t, shared_values_min=int(val_t))
        except ValueError as exc:
            st.error(str(exc))
    table = ss.enemy_table
    if table is None:
        return
    if table.empty:
        st.info("No pairs met any rule at these settings.")
        return
    table = table.copy()
    table["sme_decision"] = table["pair_id"].map(lambda p: ss.enemy_decisions.get(p, {}).get("decision", "unreviewed"))
    st.dataframe(table, hide_index=True, width="stretch")
    with st.form("enemy_decision"):
        pair = st.selectbox("Pair", list(table["pair_id"]))
        decision = st.selectbox("SME decision", en.ENEMY_DECISIONS, format_func=lambda v: v.replace("_", " "))
        note = st.text_input("Note")
        if st.form_submit_button("Record decision"):
            en.record_decision(ss.enemy_decisions, pair, decision, ss.reviewer, note)
            st.rerun()
    pick = st.selectbox("Compare pair side by side", list(table["pair_id"]), key="enemy_view")
    by_id = records_by_id()
    ia, ib = pick.split("|", 1)
    a, b = st.columns(2)
    with a:
        item_card(by_id[ia])
    with b:
        item_card(by_id[ib])


# =========================================================================== Exports

def review_state():
    s = None
    if ss.search:
        s = {k: v for k, v in ss.search.items() if k != "results"}
        s["results"] = ss.search["results"].to_dict("records")
    return {"task_review": ss.task_review, "store": ss.store, "enemy_decisions": ss.enemy_decisions,
            "class_overrides": ss.class_overrides, "search": s, "archived_reviews": ss.archived_reviews,
            "settings": {"mode": ss.llm_mode, "llm_model": ss.get("llm_model"), "embed_model": ss.get("embed_model")}}


def feature_rows():
    rows = []
    for fid in (ss.store or {}).get("order", []):
        f = ss.store["features"][fid]
        rows.append({
            "feature_id": fid, "feature_type": f["feature_type"], "canonical_label": f["canonical_label"],
            "definition": f["definition"], "review_status": f["review_status"], "sme_importance": f["sme_importance"],
            "sme_note": f["sme_note"], "dominance": f.get("dominance"), "procedure_group": f.get("procedure_group"),
            "sequence_order": f.get("sequence_order"), "supporting_items": ", ".join(f["supporting_items"]),
            "ai_supporting_items": ", ".join(f["ai"]["supporting_items"]),
            "option_refs": ", ".join(f"{r['item_id']}:{r['option_id']}" for r in f["option_refs"]),
            "evidence_types": ", ".join(f.get("evidence_types", [])), "ai_label": f["ai"]["canonical_label"],
            "merged_into": f.get("merged_into"), "merged_from": ", ".join(f.get("merged_from", [])),
            "split_from": f.get("split_from"), "validation_issues": "; ".join(f.get("validation_issues", [])),
            "proposed_misconception": f.get("proposed_misconception"),
        })
    return pd.DataFrame(rows)


def evidence_rows():
    rows = []
    for fid in (ss.store or {}).get("order", []):
        for s in ss.store["features"][fid]["source_spans"]:
            rows.append({"feature_id": fid, **s})
    return pd.DataFrame(rows)


def page_exports():
    st.header("9 · Exports")
    require_records()
    st.caption("Exports never contain API keys. Text cells that could run as spreadsheet formulas are neutralised.")
    by_id = records_by_id()
    c1, c2, c3 = st.columns(3)
    c1.download_button("Unique items (CSV)", storage.to_csv_bytes(ingest.records_frame(ss.records)), "unique_items.csv", "text/csv", width="stretch")
    c2.download_button("Project bundle (.zip)", storage.export_bundle(ss.records, ss.mapping, ss.index, review_state(), ss.file_name),
                       "item_analytics_project.zip", "application/zip", width="stretch",
                       help="Records, mapping, vectors with their manifest, and reviews. Re-import from Data & API.")
    if ss.search:
        c3.download_button("Search results (CSV)", storage.to_csv_bytes(_results_export()), "search_results.csv", "text/csv", width="stretch")
    if ss.store and ss.search:
        target_id = ss.search["target"]["item_id"]
        sib = ta.accepted_siblings(ss.task_review)
        sheets = {
            "features": feature_rows(),
            "frequency_sme": ff.frequency_table(ss.store, sib, target_id, "sme"),
            "frequency_raw": ff.frequency_table(ss.store, sib, target_id, "raw"),
            "incidence_long": ff.long_incidence(ss.store, target_id, sib, "sme"),
            "evidence_map": evidence_rows(),
            "options": ds.option_review(by_id | {target_id: ss.search["target"]}, target_id, sib, ss.store),
            "task_review": pd.DataFrame([{"target_id": target_id, "task_status": ss.task_review["task_status"],
                                          "include_all_retrieved": ss.task_review["include_all_retrieved"],
                                          "denominator_N": len(sib), **(ss.task_review["task"] or {})}]),
            "candidate_decisions": pd.DataFrame([{"item_id": k, "sme_decision": v} for k, v in ss.task_review["decisions"].items()]),
            "audit_history": pd.DataFrame([{k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in h.items()}
                                           for h in ss.task_review["history"] + ss.store["history"]]),
        }
        d1, d2, d3 = st.columns(3)
        d1.download_button("Reviewed features (XLSX)", storage.to_excel_bytes(sheets), "reviewed_features.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        d2.download_button("Reviewed features (CSV)", storage.to_csv_bytes(sheets["features"]), "reviewed_features.csv", "text/csv", width="stretch")
        d3.download_button("Item-feature incidence (CSV)", storage.to_csv_bytes(sheets["incidence_long"]), "item_feature_incidence.csv", "text/csv", width="stretch")
        st.download_button("Raw model runs and reviews (JSON)", storage.to_json_bytes(review_state()), "review_state.json", "application/json")
    if ss.enemy_table is not None and not ss.enemy_table.empty:
        t = ss.enemy_table.copy()
        t["sme_decision"] = t["pair_id"].map(lambda p: ss.enemy_decisions.get(p, {}).get("decision", "unreviewed"))
        st.download_button("Enemy candidates and decisions (CSV)", storage.to_csv_bytes(t), "enemy_candidates.csv", "text/csv")
    if ss.archived_reviews:
        st.caption(f"{len(ss.archived_reviews)} archived review set(s) are included in the bundle and the JSON export.")


PAGES = dict(zip(SECTIONS, [page_data, page_search, page_decomposition, page_distractors, page_visualization,
                            page_difficulty, page_classification, page_enemies, page_exports]))
PAGES[section]()
