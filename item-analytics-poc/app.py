"""Item Bank Analytics & Cognitive Feature Decomposition — proof of concept.

All user data, vectors, keys and reviews live in ``st.session_state`` only. No
``st.cache_data``/``st.cache_resource`` is used for user content.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

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
from src import space_db as sdb
from src import space_library as sl
from src import storage
from src import task_alignment as ta
from src import visualization as viz
from src.llm import LLMError, MockStructuredLLM, OpenAIStructuredLLM
from src.schemas import (DOMINANCE, FEATURE_TYPE_LABELS, FEATURE_TYPES, IMPORTANCE_ANCHORS, REVIEW_STATUSES,
                         SME_ALIGNMENTS)
from src.text_builder import fingerprint, sha256_text

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "samples" / "synthetic_items.csv"
SECTIONS = ["Import", "Search", "Decomposition", "Distractors", "Space Library", "Embedding Visualization",
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
    "new_items": {"records": [], "vectors": {}, "basis": None}, "ni_df": None, "ni_name": None, "ni_hash": None,
    "query_batch": None, "db_workspace": "default", "db_rev": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, (dict, list)) else v
ss = st.session_state
# Keep settings whose widgets live on one section when another section is shown.
for _k in ("llm_mode", "consent_embed", "consent_llm", "api_key_input", "embed_model", "embed_dims", "llm_model",
           "freq_source", "scale_confirmed", "db_workspace"):
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


def get_db():
    """(SpaceDB or None, reason). Off on hosted deployments unless configured; never cached across sessions."""
    secrets = {}
    try:
        if sdb.ENV_VAR in st.secrets:
            secrets = {sdb.ENV_VAR: str(st.secrets[sdb.ENV_VAR])}
    except Exception:  # noqa: BLE001 - no secrets file
        pass
    path, reason = sdb.resolve_path(secrets=secrets)
    if path is None:
        return None, reason
    try:
        return sdb.SpaceDB(path), reason
    except (sdb.SpaceDBError, sqlite3.Error, OSError) as exc:
        return None, f"The space database could not be opened: {exc}"


def save_session_space(review, store, target, sib, as_new=False):
    """Save or update a session space in the database; the store remembers its database link."""
    db, reason = get_db()
    if db is None:
        st.error(reason)
        return
    link = store.get("db_link") if not as_new else None
    if link and link.get("path") != str(db.path):
        link = None
    try:
        sid, version = db.save_space(ss.db_workspace, review, store, target, sib, ss.reviewer or "anonymous",
                                     source_name=ss.file_name, space_id=link["space_id"] if link else None,
                                     expected_version=link["version"] if link else None)
    except sdb.ConflictError as exc:
        st.error(f"{exc} Use **Save as new space** to keep both versions.")
        return
    except sdb.SpaceDBError as exc:
        st.error(str(exc))
        return
    store["db_link"] = {"space_id": sid, "version": version, "path": str(db.path)}
    ss.db_rev += 1
    ss.notice = f"{'Updated' if link else 'Saved'} space {sid} (version {version}) in workspace '{ss.db_workspace}'."
    st.rerun()


def save_controls(review, store, target, sib, key):
    link = store.get("db_link")
    a, b, c = st.columns([2, 1, 1], vertical_alignment="bottom")
    a.text_input("Workspace (library name)", key="db_workspace", help="Groups spaces in the database. Not an access control.")
    if b.button("Update saved space" if link else "Save to database", type="primary", key=f"save_{key}", width="stretch"):
        save_session_space(review, store, target, sib)
    if link and c.button("Save as new space", key=f"saveas_{key}", width="stretch"):
        save_session_space(review, store, target, sib, as_new=True)
    if link:
        st.caption(f"Linked to database space {link['space_id']} (version {link['version']}). A later edit there must be reloaded "
                   "before this space can overwrite it.")


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
        st.info("Load an item file in **Import** first.")
        st.stop()


def require_index():
    if ss.index is None:
        st.info("Generate or import embeddings in **Import** first.")
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

st.markdown("<style>.block-container{padding-top:2.2rem;padding-bottom:2rem}"
            "h3{margin-bottom:.2rem} div[data-testid='stExpander'] details summary p{font-size:.92rem}</style>",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧩 Item Bank Analytics")
    st.caption(f"Proof of concept v{APP_VERSION} · database-free · session-only storage")
    section = st.radio("Section", SECTIONS, key="section", label_visibility="collapsed")
    st.divider()
    st.text_input("Reviewer ID or pseudonym", key="reviewer", help="Recorded with SME decisions. Do not enter credentials.")
    st.caption("Session data is lost on refresh or disconnect. Export a project bundle to keep your work.")

recs = ss.records or []
index = ss.index
siblings = ta.accepted_siblings(ss.task_review) if ss.task_review else []
features = [f for f in (ss.store or {}).get("features", {}).values() if f["review_status"] not in ("merged", "removed")]
reviewed = [f for f in features if f["review_status"] not in ("pending", "needs_adjudication")]
key_val, key_src = api_key()
tokens_used = sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) + u.get("tokens", 0) for u in ss.usage_log)
status = [
    f"**{len(recs)}** bank items",
    f"**{len(ss.new_items['records'])}** new items",
    f"**{int(index.valid.sum()) if index is not None else 0}** embedded" + (f" ({index.model}, {index.dimensions}d)" if index is not None else ""),
    f"**{len(ss.search['results']) if ss.search else 0}** retrieved",
    f"siblings N = **{len(siblings)}**",
    f"**{len(features)}** features" + (f" ({100 * len(reviewed) / len(features):.0f}% reviewed)" if features else ""),
    "API: " + ("mock" if mock_mode() else ("key ready" if key_val else "no key")),
]
if tokens_used:
    mock_only = all("mock" in u["kind"] or str(u.get("model", "")).startswith("mock") for u in ss.usage_log)
    status.append(f"≈{tokens_used:,} tokens" + (" (mock, no API calls)" if mock_only else " (estimate)"))
st.caption(" · ".join(status))
if ss.notice:
    st.success(ss.notice)
    ss.notice = None
if ss.stale_notice:
    st.warning(ss.stale_notice)


# =========================================================================== Import

def page_import():
    st.subheader("1 · Import")
    bank_tab, new_tab, api_tab = st.tabs(["Bank import", "New items", "API & embeddings"])
    with bank_tab:
        _bank_import()
    with new_tab:
        _new_items_panel()
    with api_tab:
        if ss.records:
            _api_and_embeddings()
        else:
            st.info("Import a bank first.")


def _bank_import():
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        upload = st.file_uploader("Item bank (.csv or .xlsx, up to 1,000 items)", type=["csv", "xlsx", "xls"])
    with c2:
        if st.button("Load synthetic demo bank", help="12 invented items for demonstration only.", width="stretch"):
            _load_bytes(SAMPLE.read_bytes(), SAMPLE.name)
        with st.popover("Import project bundle", width="stretch"):
            bundle = st.file_uploader("Project bundle (.zip)", type=["zip"], key="bundle_upload")
            if bundle is not None and st.button("Import bundle"):
                try:
                    loaded = storage.import_bundle(bundle.getvalue())
                except storage.BundleError as exc:
                    st.error(str(exc))
                else:
                    _apply_bundle(loaded, bundle.name)
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
                st.caption("⚠️ Loading a new bank replaces the current one; downstream reviews are archived.")
            if st.button("Load this file", type="primary"):
                _load_bytes(data, upload.name, sheet)
    if ss.raw_df is None:
        st.caption("Upload a bank, load the synthetic demo bank, or import a project bundle.")
        return
    df = ss.raw_df
    with st.expander(f"Schema and preview — {ss.file_name} · {len(df)} rows · {len(df.columns)} columns"):
        st.dataframe(ingest.profile_columns(df), hide_index=True, width="stretch")
        st.dataframe(df, hide_index=True, width="stretch", height=240)
    _mapping_form(df)
    if ss.records:
        _report()


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
    ss.new_items = {"records": review.get("new_items", []), "vectors": {}, "basis": None}
    for k in [k for k in ss.keys() if str(k).startswith("map_")]:
        del ss[k]
    bump_store()
    ss.notice = "Project bundle imported. Vectors, mapping and reviews were validated; re-embed new items if you need them."
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
    m = ss.mapping
    cols = list(df.columns)
    with st.form("mapping_form"):
        st.markdown("**Column mapping**")
        a, b = st.columns(2)
        text_cols = a.multiselect("Text columns to embed (in this order)", cols, default=[c for c in m.text_columns if c in cols],
                                  key="map_text", help="Default: vignette, stem and lead-in. The key and labels are never embedded by default.")
        option_cols = b.multiselect("Answer option columns (any number)", cols, default=[c for c in m.option_columns if c in cols], key="map_opts")
        grid = st.columns(4)
        picks = {}
        roles = [("item_id", "Item ID"), ("key", "Answer key"), ("domain", "Domain"), ("task", "Task"),
                 ("category", "Category / blueprint"), ("difficulty", "Difficulty (e.g. Rasch b)"),
                 ("p_value", "Classical p-value"), ("point_biserial", "Point-biserial"),
                 ("enemy", "Enemy list"), ("rationale", "Rationale"), ("family", "Linked set / stimulus ID"),
                 ("scale_group", "Scale / calibration group")]
        for n, (role, label) in enumerate(roles):
            options, idx = _opt(cols, getattr(m, role))
            picks[role] = grid[n % 4].selectbox(label, options, index=idx, key=f"map_{role}")
        a, b = st.columns(2)
        categorical = a.multiselect("Treat these numeric-looking columns as categorical", cols,
                                    default=[c for c, t in m.type_overrides.items() if t == "categorical"], key="map_cat")
        include_opts = b.toggle("Include option text in the embedded text", value=m.include_options_in_embed, key="map_inc",
                                help="Increases lexical overlap and can surface answer-option leakage. The key marker is never embedded.")
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
    ss.notice = f"Mapping applied: {len(records)} unique items. Next: API & embeddings."
    st.rerun()


def _report():
    r = ss.report
    keys = sum(v for k, v in r["key_status"].items() if k in ("single", "multiple"))
    st.caption(f"**{r['items']}** unique items · {len(r['exact_duplicates_removed'])} exact duplicate(s) removed · "
               f"{len(r['skipped_empty_text'])} row(s) without text skipped · keys resolved for {keys} · "
               f"template `{r['template']}`")
    if r.get("synthetic_ids"):
        st.caption("No item ID column is mapped; stable IDs were created from source row numbers (ROW-0002…).")
    for w in r["warnings"]:
        st.warning(w)
    with st.expander("Normalized items and embedded text"):
        if r["exact_duplicates_removed"]:
            st.caption("Removed duplicates: " + ", ".join(f"row {d['source_row']} = row {d['duplicate_of_row']}" for d in r["exact_duplicates_removed"]))
        if r["skipped_empty_text"]:
            st.caption("Skipped: " + ", ".join(f"{d['item_id']} (row {d['source_row']})" for d in r["skipped_empty_text"]))
        st.dataframe(ingest.records_frame(ss.records), hide_index=True, width="stretch", height=240)
        pick = st.selectbox("Show item", [r["item_id"] for r in ss.records], key="preview_item")
        rec = records_by_id()[pick]
        a, b = st.columns(2)
        with a:
            item_card(rec)
        with b:
            st.text(rec["selected_embed_text"])


# --------------------------------------------------------------------------- new items

NEW_PREFIX = "NEW:"


def _index_basis():
    return (ss.index.model, ss.index.requested_dimensions, ss.index.settings_fingerprint) if ss.index is not None else None


def _new_vectors():
    """Vectors of new items, only while they match the current bank index."""
    ni = ss.new_items
    return ni["vectors"] if ni["basis"] is not None and ni["basis"] == _index_basis() else {}


def _mode_mismatch(index):
    if index.model == em.MOCK_MODEL and not mock_mode():
        return "The bank vectors were built in mock mode. Switch to mock mode or regenerate bank embeddings with OpenAI."
    if index.model != em.MOCK_MODEL and mock_mode():
        return "The bank vectors were built with OpenAI. Switch to OpenAI mode to embed new items with the same model."
    return None


def _embed_blocker():
    if ss.index is None:
        return "Generate bank embeddings first (API & embeddings tab); new items are embedded with the bank's model."
    if _mode_mismatch(ss.index):
        return _mode_mismatch(ss.index)
    if not mock_mode() and not ss.consent_embed:
        return "Embedding new items sends their text to OpenAI; give embedding consent in the API & embeddings tab."
    return None


def _embed_pending():
    """Embed new items that have no vector for the current bank index. Returns a message or None."""
    ni = ss.new_items
    if ni["basis"] != _index_basis():
        ni["vectors"], ni["basis"] = {}, None
    pending = [r for r in ni["records"] if r["item_id"] not in ni["vectors"]]
    if not pending:
        return None
    blocker = _embed_blocker()
    if blocker:
        return blocker
    try:
        vecs, valid, tokens = em.embed_queries([r["selected_embed_text"] for r in pending], get_embedder(), ss.index)
    except (em.ConsentError, em.EmbeddingError) as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"The new items could not be embedded: {em.safe_error(exc)}"
    ni["basis"] = _index_basis()
    for r, v, ok in zip(pending, vecs, valid):
        if ok:
            ni["vectors"][r["item_id"]] = v
    log_usage("new-item embedding (mock estimate)" if mock_mode() else "new-item embedding", ss.index.model, {"tokens": tokens})
    return None


def _add_new_items(records):
    """Add or replace new items by ID, then embed them when allowed."""
    ni = ss.new_items
    incoming = {r["item_id"] for r in records}
    replaced = [r["item_id"] for r in ni["records"] if r["item_id"] in incoming]
    ni["records"] = [r for r in ni["records"] if r["item_id"] not in incoming] + records
    for iid in replaced:
        ni["vectors"].pop(iid, None)
    problem = _embed_pending()
    msg = f"Added {len(records)} new item(s)" + (f" ({len(replaced)} replaced by ID)" if replaced else "") + "."
    ss.notice = msg + (" Embedded with the bank's model." if not problem else " Not embedded yet: " + problem)
    st.rerun()


def _typed_item(text, opts, key, given_id):
    options = []
    for n, line in enumerate([l for l in opts.splitlines() if l.strip()]):
        m = re.match(r"^\s*([A-Za-z])[.)]\s*(.+)$", line)
        options.append({"option_id": m.group(1).upper() if m else chr(65 + n), "text": (m.group(2) if m else line).strip(),
                        "is_key": None, "source_column": None})
    key_ids, status = ingest.parse_key(key, options)
    if status in ("single", "multiple"):
        for o in options:
            o["is_key"] = o["option_id"] in key_ids
    typed = sum(r.get("origin") == "typed" for r in ss.new_items["records"])
    source_id = (given_id or "").strip() or f"T{typed + 1:03d}"
    embed_text = f"Item text: {text.strip()}"
    return {"internal_id": NEW_PREFIX + source_id, "item_id": NEW_PREFIX + source_id, "source_item_id": source_id,
            "origin": "typed", "source_row": None, "fields": {"Item text": text.strip()}, "field_order": ["Item text"],
            "options": options, "key_raw": key or None, "key_status": status, "roles": {}, "metadata": {},
            "selected_embed_text": embed_text, "embed_text_hash": sha256_text(embed_text), "source_fingerprint": None}


def _new_items_panel():
    if not ss.records:
        st.info("Import a bank first; new items are compared against it.")
        return
    how = st.radio("Add new items by", ["Typing text", "Excel / CSV file"], horizontal=True, key="ni_how")
    if how == "Typing text":
        with st.form("ni_text_form", clear_on_submit=True):
            text = st.text_area("Item text (vignette, stem and lead-in, or any description of the task)", height=110, key="ni_text")
            a, b, c = st.columns([3, 1, 1])
            opts = a.text_area("Options (optional, one per line: 'A. text')", height=90, key="ni_opts")
            key = b.text_input("Key (optional)", key="ni_key")
            given = c.text_input("ID (optional)", key="ni_id")
            if st.form_submit_button("Add item", type="primary"):
                if not text.strip():
                    st.error("Enter the item text.")
                else:
                    _add_new_items([_typed_item(text, opts, key, given)])
    else:
        _new_items_file()
    _new_items_table()


def _new_items_file():
    upload = st.file_uploader("New items file — one row per item (.csv or .xlsx)", type=["csv", "xlsx", "xls"], key="ni_upload")
    if upload is not None:
        data = upload.getvalue()
        h = hashlib.sha256(data).hexdigest()
        try:
            sheets = ingest.list_sheets(data, upload.name)
        except ingest.IngestError as exc:
            st.error(str(exc))
            sheets = []
        sheet = st.selectbox("Worksheet", sheets, key="ni_sheet") if len(sheets) > 1 else (sheets[0] if sheets else None)
        if (h, sheet) != ss.ni_hash and st.button("Load file"):
            try:
                ss.ni_df = ingest.read_table(data, upload.name, sheet)
            except ingest.IngestError as exc:
                st.error(str(exc))
            else:
                ss.ni_name, ss.ni_hash = upload.name, (h, sheet)
                st.rerun()
    df = ss.ni_df
    if df is None:
        st.caption("Columns are mapped like the bank; columns whose names match the bank's text columns are preselected.")
        return
    cols = list(df.columns)
    guess = ingest.detect_mapping(df)
    bank_text = [c for c in ss.mapping.text_columns if c in cols]
    with st.form("ni_map_form"):
        st.caption(f"{ss.ni_name or 'File'} · {len(df)} rows · {len(cols)} columns")
        a, b = st.columns(2)
        text_cols = a.multiselect("Text columns (in this order)", cols, default=bank_text or guess.text_columns)
        opt_cols = b.multiselect("Option columns (optional)", cols, default=guess.option_columns)
        c1, c2, c3, c4 = st.columns(4)
        ids, idx = _opt(cols, guess.item_id)
        id_col = c1.selectbox("ID column", ids, index=idx)
        keys, kidx = _opt(cols, guess.key)
        key_col = c2.selectbox("Key column", keys, index=kidx)
        doms, didx = _opt(cols, guess.domain)
        dom_col = c3.selectbox("Domain column", doms, index=didx)
        tasks, tidx = _opt(cols, guess.task)
        task_col = c4.selectbox("Task column", tasks, index=tidx)
        if st.form_submit_button(f"Add {len(df)} items", type="primary"):
            mapping = ingest.ColumnMapping(item_id=None if id_col == NONE else id_col, text_columns=text_cols,
                                           option_columns=opt_cols, key=None if key_col == NONE else key_col,
                                           domain=None if dom_col == NONE else dom_col, task=None if task_col == NONE else task_col,
                                           include_options_in_embed=ss.mapping.include_options_in_embed)
            try:
                records, _unused = ingest.build_records(df, mapping)
            except ingest.IngestError as exc:
                st.error(str(exc))
            else:
                for r in records:
                    r["source_item_id"] = r["item_id"]
                    r["item_id"] = r["internal_id"] = NEW_PREFIX + r["item_id"]
                    r["origin"] = "file"
                _add_new_items(records)


def _new_items_table():
    ni = ss.new_items
    if not ni["records"]:
        return
    vecs = _new_vectors()
    table = pd.DataFrame([{"id": r["source_item_id"], "source": r.get("origin", ""), "text": viz._preview(r, 90),
                           "options": len(r["options"]), "key": r["key_status"], "embedded": r["item_id"] in vecs}
                          for r in ni["records"]])
    st.dataframe(table, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(table)))
    pending = int((~table["embedded"]).sum())
    blocker = _embed_blocker() if pending else None
    a, b, c = st.columns([2, 1, 1], vertical_alignment="bottom")
    drop = a.multiselect("Remove new items", [r["item_id"] for r in ni["records"]], key=f"ni_drop_{len(ni['records'])}",
                         format_func=lambda i: i[len(NEW_PREFIX):])
    if b.button("Remove", disabled=not drop, width="stretch"):
        ni["records"] = [r for r in ni["records"] if r["item_id"] not in drop]
        for iid in drop:
            ni["vectors"].pop(iid, None)
        st.rerun()
    if pending and c.button(f"Embed {pending} pending", width="stretch", disabled=bool(blocker)):
        problem = _embed_pending()
        if problem:
            st.error(problem)
        else:
            st.rerun()
    if blocker:
        st.caption(blocker)


# --------------------------------------------------------------------------- API & embeddings

def _api_and_embeddings():
    st.radio("Analysis mode", ["Offline mock (no network)", "OpenAI"], key="llm_mode", horizontal=True,
             help="Mock mode runs the whole workflow offline with deterministic lexical vectors and heuristic proposals. "
                  "Its output is labelled and is not AI or SME analysis.")
    if mock_mode():
        st.caption("Mock mode: no external requests. Vectors are lexical hashes and proposals are heuristics, for demonstration only.")
    else:
        key, src = api_key()
        a, b = st.columns([2, 1], vertical_alignment="bottom")
        a.text_input("OpenAI API key (kept in this session only)", type="password", key="api_key_input",
                     help="Not written to disk, exports, logs or the URL. Streamlit secrets or the OPENAI_API_KEY environment variable are used when this is empty.")
        b.caption(f"Key source: {src}.")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Embedding model", key="embed_model")
        c2.number_input("Embedding dimensions (0 = model default)", min_value=0, max_value=3072, step=64, key="embed_dims",
                        help="Only for models that support shortened embeddings.")
        c3.text_input("Model for task and feature analysis", key="llm_model", placeholder="A model ID enabled for your account",
                      help="Must support structured JSON-schema output.")
        st.caption("🔒 **Privacy.** Embedding sends the embedded text of every bank item and new item to OpenAI. Task proposal and "
                   "decomposition send only the target and the candidates you confirm. You can decline and keep exploring locally.")
        st.checkbox("I consent to sending item embedding text to OpenAI for embeddings", key="consent_embed")
        st.checkbox("I consent to sending the target and confirmed items to OpenAI for task and feature analysis", key="consent_llm")

    model, dims, fp = embed_settings()
    texts = [r["selected_embed_text"] for r in ss.records]
    current = ss.index is not None and ss.index.matches(ss.records, fp)
    todo = len(texts) if not current else int((~ss.index.valid).sum())
    est = em.estimate_tokens(texts)
    cost = em.estimate_cost(model, est)
    a, b = st.columns([3, 1], vertical_alignment="center")
    if current and todo == 0:
        a.success(f"All {len(texts)} bank items are embedded with {ss.index.model} ({ss.index.dimensions} dims). Search works offline from here.")
    else:
        a.caption(f"{len(texts)} bank items · about {est:,} tokens"
                  + (f" · approx. ${cost:.4f} at list price" if cost is not None and not mock_mode() else "")
                  + f" · model {model}" + (f" · {dims} dims" if dims else ""))
        if ss.index is not None and not current:
            a.caption("⚠️ The existing vectors were built with different text or settings; regenerating replaces them.")
        label = "Generate embeddings" if not current else f"Retry {todo} failed items"
        if b.button(label, type="primary", disabled=not mock_mode() and not ss.consent_embed, width="stretch"):
            _generate(model, dims, fp)
        if not mock_mode() and not ss.consent_embed:
            a.caption("Embedding is disabled until you give consent.")
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
    msg = f"Embedded {int(new.valid.sum())}/{len(ss.records)} bank items." + (" Some items failed; use Retry." if new.errors else "")
    if ss.new_items["records"]:
        problem = _embed_pending()
        msg += " New items were embedded too." if not problem else f" New items not embedded: {problem}"
    ss.notice = msg
    st.rerun()


# =========================================================================== Search

def page_search():
    st.subheader("2 · Cosine similarity search")
    require_records()
    require_index()
    by_id = records_by_id()
    ni = ss.new_items
    a, b = st.columns([1, 3], vertical_alignment="bottom")
    source = a.radio("Target", ["Bank item", "New item"], horizontal=True, key="target_source")
    target = None
    with b:
        if source == "Bank item":
            tid = st.selectbox("Bank item", [r["item_id"] for r in ss.records], key="target_pick",
                               format_func=lambda i: f"{i} — {viz._preview(by_id[i], 90)}")
            target = by_id[tid]
        elif not ni["records"]:
            st.info("No new items yet — add them in **Import → New items** (typed text or an Excel/CSV file).")
        else:
            by_new = {r["item_id"]: r for r in ni["records"]}
            vecs = _new_vectors()
            labels = {i: f"{i[len(NEW_PREFIX):]}{'' if i in vecs else ' (not embedded)'} — {viz._preview(r, 80)}"
                      for i, r in by_new.items()}
            nid = st.selectbox(f"New item ({len(by_new)})", list(by_new), key="new_pick", format_func=labels.get)
            target = by_new[nid]
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2], vertical_alignment="bottom")
    mode = c1.radio("Filter", ["Top K", "Minimum cosine similarity", "Both"], key="search_mode", horizontal=True)
    k = c2.number_input("Top K", 1, max(1, len(ss.records)), min(5, len(ss.records)), key="search_k",
                        help="Counts unique items, never options.")
    thr = c3.slider("Minimum cosine similarity", -1.0, 1.0, rt.DEFAULT_THRESHOLD, 0.01, key="search_thr",
                    help="0.85 is an illustrative, adjustable heuristic — not a probability and not proof of the same task.")
    with c4.popover("Metadata filter · keyword ranking", width="stretch"):
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
    mode_key = {"Top K": "top_k", "Minimum cosine similarity": "threshold", "Both": "both"}[mode]
    b1, b2, _ = st.columns([1, 1.4, 3])
    if b1.button("Search", type="primary", disabled=target is None, width="stretch"):
        _run_search(target, mode_key, int(k), float(thr), allowed, use_bm25, hybrid)
    embedded_new = [r for r in ni["records"] if r["item_id"] in _new_vectors()]
    if source == "New item" and embedded_new:
        if b2.button(f"Search all {len(embedded_new)} new items", width="stretch"):
            _batch_search(embedded_new, mode_key, int(k), float(thr), allowed)
        _show_batch()
    _show_search()


def _batch_search(items, mode, k, thr, allowed):
    vecs = _new_vectors()
    rows = []
    by_id = records_by_id()
    for q in items:
        res = rt.select(rt.cosine_scores(ss.index, vecs[q["item_id"]]), ss.index.item_ids, mode, k, thr, None, allowed)
        if res.empty:
            rows.append({"new_item_id": q["source_item_id"], "rank": None, "item_id": None, "cosine_similarity": None,
                         "domain": None, "task": None, "note": "no bank item met the filter"})
        for _, r in res.iterrows():
            rec = by_id[r["item_id"]]
            rows.append({"new_item_id": q["source_item_id"], "rank": int(r["rank"]), "item_id": r["item_id"],
                         "cosine_similarity": r["cosine_similarity"], "domain": rec["roles"].get("domain"),
                         "task": rec["roles"].get("task"), "note": ""})
    ss.query_batch = {"table": pd.DataFrame(rows), "mode": mode, "k": k, "threshold": thr}
    st.rerun()


def _show_batch():
    b = ss.query_batch
    if not b:
        return
    table = b["table"]
    label = {"top_k": f"top {b['k']}", "threshold": f"cosine ≥ {b['threshold']:.2f}", "both": f"cosine ≥ {b['threshold']:.2f}, top {b['k']}"}[b["mode"]]
    with st.expander(f"Batch results · {table['new_item_id'].nunique()} new items · {label} · cosine only", expanded=True):
        st.dataframe(table, hide_index=True, width="stretch", height=min(360, 38 + 35 * len(table)),
                     column_config={"cosine_similarity": st.column_config.NumberColumn(format="%.4f")})
        st.download_button("Batch results (CSV)", storage.to_csv_bytes(table), "batch_search_results.csv", "text/csv")


def _run_search(target, mode, k, thr, allowed, use_bm25, hybrid):
    index = ss.index
    if target["item_id"].startswith(NEW_PREFIX):
        vec = _new_vectors().get(target["item_id"])
        if vec is None:
            problem = _embed_pending()
            vec = _new_vectors().get(target["item_id"])
            if vec is None:
                st.error(problem or "This new item could not be embedded.")
                return
        q, exclude = vec, None
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
            ss.notice = "The previous task and feature review was archived (see Space Library and Exports)."
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
    st.subheader("3 · Target task, sibling confirmation and feature decomposition")
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
        st.caption("OpenAI analysis needs a model ID and consent in **Import**.")
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
                    format_func=lambda v: {"all": "All proposed", "pending": "Needs review", "accepted": "Accepted", "rejected": "Rejected/merged/removed"}[v])
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
                rows = [f for f in rows if f["review_status"] in ("rejected", "not_required", "unsupported", "merged", "removed")]
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
    st.markdown("**Save this space for future use**")
    save_controls(ss.task_review, store, target, sib, "decomp")


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
    statuses = [s for s in REVIEW_STATUSES if s not in ("merged", "removed")]
    edited = st.data_editor(
        data[columns], hide_index=True, width="stretch", key=f"fe_{ftype}_{ss.store_rev}",
        disabled=["feature_id", "SME coverage", "raw AI coverage", "in target", "supporting items", "options", "evidence", "issues"],
        column_config={
            "canonical_label": st.column_config.TextColumn("label", width="medium"),
            "definition": st.column_config.TextColumn(width="medium"),
            "review_status": st.column_config.SelectboxColumn("status", options=statuses + ["merged", "removed"], required=True),
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
    provenance = [fid for fid in store["order"] if store["features"][fid]["review_status"] != "merged"]
    active = [fid for fid in provenance if store["features"][fid]["review_status"] != "removed"]
    label = lambda fid: f"{fid} · {store['features'][fid]['canonical_label'][:70]}"  # noqa: E731
    _add_remove_tools(store, reviewer, sib, target, active, label)
    if not provenance:
        return
    with st.expander("Provenance, pending AI updates and history"):
        fid = st.selectbox("Feature", provenance, format_func=label, key="prov_pick")
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
    if not active:
        return
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


ADD_LABELS = {"declarative_key": "Key feature — declarative", "procedural_key": "Key feature — procedural",
              "instance_context": "Instance feature", "distractor_feature": "Distractor candidate"}


def _add_remove_tools(store, reviewer, sib, target, active, label):
    by_id = records_by_id() | {target["item_id"]: target}
    space_items = [target["item_id"]] + [i for i in sib if i in by_id]
    with st.expander("Add an element to a space"):
        ftype = st.selectbox("Space", FEATURE_TYPES, format_func=ADD_LABELS.get, key="add_type")
        option_labels = {f"{iid}:{o['option_id']}": f"{iid} · {o['option_id']}. {o['text'][:70]}"
                         for iid in space_items for o in by_id[iid]["options"] if o["is_key"] is False}
        if ftype == "distractor_feature" and not option_labels:
            st.caption("No incorrect options with a known key are in this space, so no distractor candidate can be added.")
        with st.form(f"add_form_{ftype}", clear_on_submit=True):
            name = st.text_input("Label")
            definition = st.text_area("Definition", height=70)
            params = {}
            if ftype == "distractor_feature":
                picked = st.multiselect("Incorrect options it comes from", list(option_labels), format_func=option_labels.get)
                params["option_refs"] = [tuple(p.split(":", 1)) for p in picked]
                params["misconception"] = st.text_input("Misconception hypothesis (unverified)")
            else:
                params["supporting_items"] = st.multiselect("Supporting items", space_items)
                params["excerpt"] = st.text_input("Quote (must appear verbatim in each selected item)")
                params["rationale"] = st.text_area("Or an inference rationale (when the feature is not stated in the text)", height=60)
                if ftype == "procedural_key":
                    c1, c2, c3 = st.columns(3)
                    params["dominance"] = c1.selectbox("Dominance", DOMINANCE)
                    params["procedure_group"] = c2.text_input("Procedure group")
                    params["sequence_order"] = c3.number_input("Order (0 = none)", 0, 50, 0) or None
            c1, c2 = st.columns(2)
            imp = c1.selectbox("Importance", ["unrated", "1", "2", "3", "4", "5"])
            note = c2.text_input("SME note")
            if st.form_submit_button("Add to space", type="primary"):
                try:
                    fid = dc.add_feature(store, by_id, ftype, name, reviewer, definition=definition, note=note,
                                         importance=None if imp == "unrated" else int(imp), **params)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    bump_store()
                    ss.notice = f"Added {fid} to the {ADD_LABELS[ftype].lower()} space."
                    st.rerun()
    removed = [fid for fid in store["order"] if store["features"][fid]["review_status"] == "removed"]
    with st.expander(f"Remove or restore elements ({len(removed)} removed)"):
        st.caption("Removal takes an element out of its space, its counts and exports of approved features. It is recorded "
                   "and reversible; a later AI run cannot bring it back without your confirmation.")
        c1, c2 = st.columns(2)
        with c1:
            to_remove = st.multiselect("Elements to remove", active, format_func=label, key=f"rm_pick_{ss.store_rev}")
            reason = st.text_input("Reason (recorded)", key=f"rm_reason_{ss.store_rev}")
            if st.button("Remove from space", disabled=not to_remove):
                dc.remove_features(store, to_remove, reviewer, reason)
                bump_store()
                st.rerun()
        with c2:
            to_restore = st.multiselect("Removed elements", removed, format_func=label, key=f"rs_pick_{ss.store_rev}")
            if st.button("Restore", disabled=not to_restore):
                dc.restore_features(store, to_restore, reviewer)
                bump_store()
                st.rerun()


# =========================================================================== Distractors

def page_distractors():
    st.subheader("4 · Distractor review")
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
                    if f["feature_type"] == "distractor_feature" and f.get("proposed_misconception") and f["review_status"] not in ("merged", "removed")]
            if hyps:
                with st.expander("Misconception hypotheses (require SME verification)"):
                    for fid, f in hyps:
                        st.markdown(f"**{fid}** {md(f['canonical_label'])} — {md(f['proposed_misconception'])}")


# =========================================================================== Feature spaces

def page_spaces():
    st.subheader("5 · Space library")
    require_records()
    session_tab, db_tab = st.tabs(["Session spaces", "Database"])
    with session_tab:
        _session_spaces()
    with db_tab:
        _database_spaces()


def _session_space_entries():
    by_id = records_by_id() | {r["item_id"]: r for r in ss.new_items["records"]}
    if ss.search:
        by_id[ss.search["target"]["item_id"]] = ss.search["target"]
    entries = []
    if ss.task_review is not None and ss.store:
        entries.append(("current", ss.task_review, ss.store))
    for n, a in enumerate(ss.archived_reviews, start=1):
        if a.get("task_review") and a.get("store"):
            entries.append((f"archived {n}", a["task_review"], a["store"]))
    out = []
    for name, review, store in entries:
        tid = review.get("target_id")
        out.append((name, review, store, by_id.get(tid) or {"item_id": tid, "fields": {}}))
    return out


def _session_spaces():
    spaces = sl.collect_spaces(ss.task_review, ss.store, ss.archived_reviews, records_by_id())
    if not spaces:
        st.info("No feature spaces yet. Build one in **Decomposition**; earlier spaces are kept here when you start a new search.")
        return
    st.dataframe(sl.space_summary(spaces), hide_index=True, width="stretch")
    entries = _session_space_entries()
    if entries:
        names = [e[0] for e in entries]
        pick = st.selectbox("Space to save", names, key="session_save_pick",
                            format_func=lambda n: f"{n} — {next(e for e in entries if e[0] == n)[1].get('task', {}).get('name') or '(task not confirmed)'}")
        name, review, store, target = next(e for e in entries if e[0] == pick)
        save_controls(review, store, target, ta.accepted_siblings(review), f"lib_{pick}")
    with st.expander("Search elements in session spaces"):
        table = sl.space_features(spaces)
        c1, c2, c3, c4 = st.columns(4)
        tasks = c1.multiselect("Task", sorted(table["task"].unique()) if len(table) else [], key="lib_task")
        domains = c2.multiselect("Domain", sorted(table["domain"].unique()) if len(table) else [], key="lib_domain")
        types = c3.multiselect("Space", [FEATURE_TYPE_LABELS[t] for t in FEATURE_TYPES], key="lib_type")
        statuses = c4.multiselect("Status", sorted(table["status"].unique()) if len(table) else [], key="lib_status")
        a, b = st.columns([3, 1], vertical_alignment="bottom")
        text = a.text_input("Keyword (label, definition, task or domain)", key="lib_text")
        include_removed = b.toggle("Include removed", key="lib_removed")
        shown = sl.filter_features(table, tasks, domains, types, statuses, text)
        if not include_removed:
            shown = shown[shown["status"] != "removed"]
        st.caption(f"{len(shown)} of {len(table)} elements")
        st.dataframe(shown, hide_index=True, width="stretch")
        st.download_button("Filtered elements (CSV)", storage.to_csv_bytes(shown), "session_space_elements.csv", "text/csv")


DB_TYPE_LABELS = {t: FEATURE_TYPE_LABELS[t] for t in FEATURE_TYPES}


def _database_spaces():
    db, reason = get_db()
    if db is None:
        st.info(reason)
        return
    actor = ss.reviewer or "anonymous"
    st.caption(f"{reason} · shared by everyone who can open this app · stores spaces with their quoted item evidence; "
               "item banks and vectors are not stored.")
    for err in st.session_state.pop("db_errors", []):
        st.error(err)
    facets = db.facets(include_deleted=True)
    c1, c2, c3, c4 = st.columns(4)
    ws = c1.selectbox("Workspace", ["(all)"] + facets["workspaces"], key="db_ws")
    tasks = c2.multiselect("Task", facets["tasks"], key="db_task")
    domains = c3.multiselect("Domain", facets["domains"], key="db_domain")
    types = c4.multiselect("Space", list(DB_TYPE_LABELS), format_func=DB_TYPE_LABELS.get, key="db_type")
    a, s_, b, c = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
    text = a.text_input("Keyword (label, definition, note, misconception, task or domain)", key="db_text")
    statuses = s_.multiselect("Status", list(sdb.EDITABLE_STATUSES), key="db_status",
                              help="For example accepted + edited for SME-approved elements only.")
    include_removed = b.toggle("Removed elements", key="db_removed")
    include_deleted = c.toggle("Deleted spaces", key="db_deleted")
    workspace = None if ws == "(all)" else ws
    spaces = db.spaces(workspace, tasks, domains, include_deleted=include_deleted)
    if spaces.empty:
        st.caption("No saved spaces match. Save one from **Decomposition** or the Session spaces tab.")
    else:
        st.dataframe(spaces, hide_index=True, width="stretch", height=min(260, 38 + 35 * len(spaces)))
    if include_removed and statuses:
        statuses = statuses + ["removed"]
    elements = db.elements(workspace, tasks, domains, types, statuses or None, text, include_removed, include_deleted)
    st.caption(f"{len(elements)} element(s). Edit cells to update; changes are saved immediately and recorded in the history.")
    if not elements.empty:
        _database_editor(db, elements, actor)
    with st.expander("Remove, restore or delete"):
        _database_actions(db, elements, spaces, actor)
    with st.expander("Edit space details and history"):
        _database_space_details(db, spaces, actor)
    with st.expander("Backup and restore"):
        st.download_button("Download library backup (JSON)", storage.to_json_bytes(db.export_json()),
                           "space_library_backup.json", "application/json")
        up = st.file_uploader("Restore from a backup (JSON)", type=["json"], key="db_backup_upload")
        mode = st.radio("If a space already exists", ["skip", "replace"], horizontal=True, key="db_import_mode")
        if up is not None and st.button("Restore backup"):
            try:
                counts = db.import_json(json.loads(up.getvalue()), actor, on_conflict=mode)
            except (sdb.SpaceDBError, ValueError, sqlite3.Error) as exc:
                st.error(f"Backup not restored: {exc}")
            else:
                ss.db_rev += 1
                ss.notice = f"Backup restored: {counts['added']} added, {counts['replaced']} replaced, {counts['skipped']} skipped."
                st.rerun()


def _database_editor(db, elements, actor):
    columns = ["element_id", "task", "domain", "space_type", "label", "definition", "status", "importance", "dominance",
               "procedure_group", "sequence_order", "note", "misconception", "coverage", "supporting_items", "options",
               "evidence", "origin", "space_version", "updated_by"]
    data = elements[columns].copy()
    data["importance"] = data["importance"].map(lambda v: "unrated" if v is None or v != v else str(int(v)))
    edited = st.data_editor(
        data, hide_index=True, width="stretch", key=f"dbed_{ss.db_rev}",
        disabled=["element_id", "task", "domain", "space_type", "coverage", "supporting_items", "options", "evidence",
                  "origin", "space_version", "updated_by"],
        column_config={
            "element_id": st.column_config.TextColumn("id", width="small"),
            "label": st.column_config.TextColumn(width="medium"),
            "definition": st.column_config.TextColumn(width="medium"),
            "status": st.column_config.SelectboxColumn(options=list(sdb.EDITABLE_STATUSES) + ["removed"], required=True),
            "importance": st.column_config.SelectboxColumn("importance 1–5", options=["unrated", "1", "2", "3", "4", "5"], required=True),
            "dominance": st.column_config.SelectboxColumn(options=list(DOMINANCE)),
            "sequence_order": st.column_config.NumberColumn("order", min_value=1, step=1),
        },
    )
    changes = sdb.editor_changes(data.to_dict("records"), edited.to_dict("records"))
    if not changes:
        return
    _applied, errors = sdb.apply_editor_changes(db, changes, actor)
    ss.db_rev += 1
    if errors:
        st.session_state["db_errors"] = errors
    st.rerun()


def _database_actions(db, elements, spaces, actor):
    labels = {r["element_id"]: f"{r['label'][:60]} · {r['space_type']} · {r['task']}" for _, r in elements.iterrows()}
    versions = {r["space_id"]: int(r["space_version"]) for _, r in elements.iterrows()}
    c1, c2 = st.columns(2)
    with c1:
        active = [e for e in labels if elements.set_index("element_id").loc[e, "status"] != "removed"]
        pick = st.multiselect("Remove elements", active, format_func=labels.get, key=f"db_rm_{ss.db_rev}")
        reason = st.text_input("Reason (recorded)", key=f"db_rm_reason_{ss.db_rev}")
        if st.button("Remove", disabled=not pick, key="db_rm_btn"):
            _db_call(lambda: db.set_removed(pick, True, actor, versions, reason))
    with c2:
        removed = [e for e in labels if elements.set_index("element_id").loc[e, "status"] == "removed"]
        pick = st.multiselect("Restore removed elements", removed, format_func=labels.get, key=f"db_rs_{ss.db_rev}",
                              help="Turn on 'Removed elements' above to list them.")
        if st.button("Restore", disabled=not pick, key="db_rs_btn"):
            _db_call(lambda: db.set_removed(pick, False, actor, versions))
    if spaces.empty:
        return
    st.divider()
    space_labels = {r["space_id"]: f"{r['space_id']} · {r['task_name']} · {r['domain']} · {r['status']}" for _, r in spaces.iterrows()}
    a, b, c, d = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
    sid = a.selectbox("Space", list(space_labels), format_func=space_labels.get, key="db_space_pick")
    row = spaces.set_index("space_id").loc[sid]
    if row["status"] == "active":
        if b.button("Delete space", key="db_del"):
            _db_call(lambda: db.set_space_deleted(sid, True, actor, int(row["version"])))
    else:
        if b.button("Restore space", key="db_undel"):
            _db_call(lambda: db.set_space_deleted(sid, False, actor, int(row["version"])))
        confirm = c.checkbox("Permanently", key=f"db_purge_ok_{sid}", help="Purging cannot be undone; the history keeps a tombstone.")
        if d.button("Purge", disabled=not confirm, key="db_purge"):
            _db_call(lambda: db.purge_space(sid, actor, confirm=True))


def _database_space_details(db, spaces, actor):
    if spaces.empty:
        st.caption("No spaces.")
        return
    sid = st.selectbox("Space", list(spaces["space_id"]), key="db_detail_pick",
                       format_func=lambda s: f"{s} · {spaces.set_index('space_id').loc[s, 'task_name']}")
    row = spaces.set_index("space_id").loc[sid]
    with st.form(f"db_space_form_{sid}_{row['version']}"):
        a, b = st.columns(2)
        task = a.text_input("Task", row["task_name"])
        domain = b.text_input("Domain", row["domain"])
        a, b = st.columns(2)
        decision = a.text_input("Decision or action", row["decision_or_action"])
        workspace = b.text_input("Workspace", row["workspace"])
        notes = st.text_area("Notes", row["notes"], height=70)
        if st.form_submit_button("Save details"):
            _db_call(lambda: db.update_space(sid, {"task_name": task, "domain": domain, "decision_or_action": decision,
                                                   "workspace": workspace, "notes": notes}, actor, int(row["version"])))
    st.dataframe(db.history(sid), hide_index=True, width="stretch", height=220)


def _db_call(fn):
    try:
        fn()
    except sdb.SpaceDBError as exc:
        st.error(str(exc))
        return
    ss.db_rev += 1
    st.rerun()


# =========================================================================== Visualization

def page_visualization():
    st.subheader("6 · Embedding and feature visualization")
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
    st.subheader("7 · Difficulty heuristic")
    require_records()
    if not ss.mapping.difficulty:
        st.info("Map an empirical difficulty column (for example Rasch b) in **Import** to use this section. "
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
    st.subheader("8 · Classification heuristic")
    require_records()
    roles = [r for r in ("domain", "task", "category") if getattr(ss.mapping, r)]
    if not roles:
        st.info("Map a domain, task or category column in **Import** to use this section.")
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
    st.subheader("9 · Enemy-item candidates")
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
            "new_items": ss.new_items["records"],
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
    st.subheader("10 · Exports")
    require_records()
    st.caption("Exports never contain API keys. Text cells that could run as spreadsheet formulas are neutralised.")
    by_id = records_by_id()
    c1, c2, c3 = st.columns(3)
    c1.download_button("Unique items (CSV)", storage.to_csv_bytes(ingest.records_frame(ss.records)), "unique_items.csv", "text/csv", width="stretch")
    c2.download_button("Project bundle (.zip)", storage.export_bundle(ss.records, ss.mapping, ss.index, review_state(), ss.file_name),
                       "item_analytics_project.zip", "application/zip", width="stretch",
                       help="Records, mapping, vectors with their manifest, and reviews. Re-import from Import → Bank import.")
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
    spaces = sl.collect_spaces(ss.task_review, ss.store, ss.archived_reviews, by_id)
    if spaces:
        st.download_button("Feature space library — all spaces (CSV)", storage.to_csv_bytes(sl.space_features(spaces)),
                           "feature_space_library.csv", "text/csv")
    if ss.archived_reviews:
        st.caption(f"{len(ss.archived_reviews)} archived review set(s) are included in the bundle and the JSON export.")


PAGES = dict(zip(SECTIONS, [page_import, page_search, page_decomposition, page_distractors, page_spaces, page_visualization,
                            page_difficulty, page_classification, page_enemies, page_exports]))
PAGES[section]()
