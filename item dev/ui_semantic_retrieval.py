"""Search authorized source collections and save the writer's selected passages."""

from copy import deepcopy
from pathlib import Path
import re

import streamlit as st

import core
import semantic_retrieval as retrieval
import specialist_workflow as workflow
import storage


COLLECTIONS = {"bank": "Structured item bank", "clinical": "Clinical references"}
PROVIDERS = {"openai": "OpenAI", "anthropic": "Anthropic"}
SEARCH_ACCESS = {"local": "All locally approved records", **PROVIDERS}


def _load_uploads(files, collection, providers):
    if not files:
        raise ValueError("Choose at least one approved collection file.")
    if len(files) > retrieval.MAX_FILES or sum(f.size for f in files) > retrieval.MAX_IMPORT_BYTES:
        raise ValueError(f"Import up to {retrieval.MAX_FILES} files totaling {retrieval.MAX_IMPORT_BYTES // 1024 ** 2} MiB per collection.")
    records = []
    for upload in files:
        records.extend(retrieval.import_collection(
            upload.getvalue(), upload.name, collection, approved_providers=providers,
        ))
        if len(records) > retrieval.MAX_RECORDS:
            raise ValueError(f"Import up to {retrieval.MAX_RECORDS} records per collection.")
    workflow.validate_sources(records)
    return records


def _collection_restrictions(case, collection, records):
    """Find existing grants the replacement collection no longer supports."""
    current = {r["source_id"]: r for r in records}
    restrictions = {}
    present = set()
    permissions = case.get("retrieval_permissions", {})
    for item in case["items"] + case.get("archived_items", []):
        for saved in item.get("decision_workflow", {}).get("retrieved_sources", []):
            key = item["id"], saved["source_id"]
            snapshot = workflow.retrieval_permission(saved)
            permission = permissions.get(item["id"], {}).get(saved["source_id"], snapshot)
            if snapshot["collection"] != collection or permission["source_revision"] != snapshot["source_revision"]:
                continue
            present.add(key)
            if not permission["providers"]:
                continue
            record = current.get(saved["source_id"])
            permitted = set()
            if record and (not record.get("item_ids") or item["id"] in record["item_ids"]):
                stage_allowed = not record.get("stage_ids") or set(saved.get("stage_ids", [])) <= set(record["stage_ids"])
                scoped = record | {"stage_ids": saved.get("stage_ids", []), "item_ids": saved.get("item_ids", [])}
                if stage_allowed and workflow.retrieval_permission(scoped)["source_revision"] == snapshot["source_revision"]:
                    permitted = set(workflow.permitted_retrieval_providers(saved, permission)).intersection(
                        record.get("providers", [])
                    )
            if permitted != set(permission["providers"]):
                restrictions[key] = permitted
    # With no current snapshot to compare, historical text can only be restored locally.
    for iid, by_source in permissions.items():
        for sid, permission in by_source.items():
            if permission["providers"] and (iid, sid) not in present and permission["collection"] == collection:
                restrictions[iid, sid] = set()
    return restrictions


def _narrow_access(case, restrictions, actor, commit):
    """Commit provider reductions while retaining the reviewed passage and its stage."""
    if not actor.strip():
        raise ValueError("Enter the writer recording the permission change.")
    candidate = deepcopy(case)
    review = {"actor": actor.strip(), "at": core.timestamp(), "case_revision": case["version"]}
    permissions = candidate.setdefault("retrieval_permissions", {})
    for (iid, sid), requested in restrictions.items():
        previous = permissions.get(iid, {}).get(sid)
        if previous is not None:
            previous["providers"] = sorted(set(previous["providers"]).intersection(requested))
    for item in candidate["items"] + candidate.get("archived_items", []):
        for saved in item.get("decision_workflow", {}).get("retrieved_sources", []):
            key = item["id"], saved["source_id"]
            if key in restrictions:
                by_source = permissions.setdefault(item["id"], {})
                permission = by_source.setdefault(saved["source_id"], workflow.retrieval_permission(saved))
                allowed = workflow.permitted_retrieval_providers(saved, permission)
                saved["providers"] = [p for p in allowed if p in restrictions[key]]
                permission["providers"] = list(saved["providers"])
                saved["permission_review"] = review.copy()
    commit(candidate, "Narrow saved source permissions", specialist=True)


def _collection_controls(case, catalog, actor, commit, prefix):
    with st.expander("Load or replace a search collection"):
        collection = st.selectbox("Collection to load", list(COLLECTIONS), format_func=COLLECTIONS.get,
                                  key=prefix + "load_collection")
        local_directory = False
        if collection == "clinical" and not storage.hosted_mode():
            source = st.radio("Clinical reference source", ["Approved import", "Local Markdown folder"],
                              horizontal=True, key=prefix + "origin")
            local_directory = source == "Local Markdown folder"
        if local_directory:
            folder = st.text_input("Obsidian or Markdown folder", placeholder="/path/to/approved/folder",
                                   key=prefix + "folder")
            st.caption("Choose the folder whose notes you are authorized to search. The index reads Markdown files in that folder.")
        else:
            types = ["json"] if collection == "bank" else ["json", "md", "markdown"]
            uploads = st.file_uploader(f"Approved {COLLECTIONS[collection].lower()} files",
                                       type=types, accept_multiple_files=True,
                                       max_upload_size=retrieval.MAX_IMPORT_BYTES // 1024 ** 2,
                                       key=prefix + "files_" + collection)
            st.caption("A JSON array can contain multiple source records with individual provider permissions."
                       + (" You can also import individual Markdown notes." if collection == "clinical" else ""))
        grants = st.multiselect("Providers authorized for this collection", list(PROVIDERS),
                                format_func=PROVIDERS.get, key=prefix + "grants_" + collection + str(local_directory))
        st.caption("Leave providers empty for local use. Individual source restrictions also apply.")
        has_saved_grants = bool(_collection_restrictions(case, collection, []))
        if has_saved_grants:
            st.caption("Loading this collection also updates provider access across this case. Changed, missing and previously removed passages become local-only. Matching saved passages keep the providers still authorized.")
        if st.button("Load approved collection", key=prefix + "load", disabled=not (
            folder.strip() if local_directory else uploads
        ) or (has_saved_grants and not actor.strip())):
            try:
                if local_directory:
                    records = retrieval.load_markdown_folder(folder.strip(), approved_providers=grants)
                    origin = Path(folder.strip()).name
                else:
                    records = _load_uploads(uploads, collection, grants)
                    origin = ", ".join(f.name for f in uploads[:2])
                    if len(uploads) > 2:
                        origin += f" and {len(uploads) - 2} more files"
                restrictions = _collection_restrictions(case, collection, records)
                if restrictions:
                    _narrow_access(case, restrictions, actor, commit)
            except (ValueError, OSError, UnicodeError) as exc:
                st.error(str(exc))
            else:
                catalog[collection] = {"records": records, "version": core.uid("index"), "origin": origin}
                if restrictions:
                    st.session_state["itemdev_notice"] = (
                        f"Loaded {len(records)} records. Narrowed provider access for {len(restrictions)} saved passages."
                    )
                    st.rerun()
                st.success(f"Loaded {len(records)} records into {COLLECTIONS[collection].lower()}.")
        if catalog:
            for key, entry in catalog.items():
                st.text(f"{COLLECTIONS[key]} · {entry['origin']} · {len(entry['records'])} records loaded in this session")


def _result_title(source):
    title = f"{source.get('title') or source['source_id']} · {source['locator']}"
    return re.sub(r"([\\`*_\[\]()<>])", r"\\\1", title)


def _display_source(source, *, result=None):
    """Render excerpts and source metadata as data, including literal Markdown."""
    st.text("Source ID · " + source["source_id"])
    st.text(f"{source['source_class']} · {source['source_date']} · {source['locator']}")
    if source.get("citation"):
        st.text("Citation · " + str(source["citation"]))
    if source.get("url"):
        st.text("Source URL · " + str(source["url"]))
    if source.get("relative_file") and source["relative_file"] not in source["locator"]:
        st.text("File · " + source["relative_file"])
    if source.get("section") and source["section"] not in source["locator"]:
        st.text("Section · " + source["section"])
    providers = ", ".join(PROVIDERS[p] for p in source.get("providers", []) if p in PROVIDERS)
    st.caption("Provider access · " + (providers or "Local use only"))
    scope = {k: source[k] for k in ("decision", "decision_timing", "population", "setting", "reasoning_phase",
                                    "option_type", "stage_title", "item_ids", "stage_ids") if source.get(k)}
    blueprint = source.get("blueprint", {})
    if isinstance(blueprint, dict):
        scope.update({f"blueprint {k}": v for k, v in blueprint.items() if v})
    if scope:
        st.write("Applicable scope")
        for key, value in scope.items():
            st.text(f"{key.replace('_', ' ').capitalize()} · {', '.join(value) if isinstance(value, list) else value}")
    if source.get("selected_by"):
        st.text(f"Selected by {source['selected_by']} · {source.get('selected_at', '')} · case revision {source.get('selected_from_revision', '')}")
    review = source.get("permission_review")
    if isinstance(review, dict) and review.get("actor"):
        st.text(f"Provider access reviewed by {review['actor']} · {review.get('at', '')}")
    st.text(source["excerpt"])
    if result is not None:
        st.caption(f"Semantic similarity · {result['similarity']:.3f}")
        for reason in result["reasons"]:
            st.text(reason)
    if source.get("options"):
        st.write("Approved bank options")
        for option in source["options"]:
            if isinstance(option, dict):
                st.text(" · ".join(filter(None, [option.get("option_id"), option.get("text")])))
    if source.get("observed_option_evidence"):
        st.write("Historical option performance from the source administration")
        st.dataframe(source["observed_option_evidence"], hide_index=True, width="stretch")


def _save(case, item, rows, actor, commit):
    try:
        candidate = workflow.replace_retrieved_sources(case, item["id"], rows, actor)
        commit(candidate, "Update selected retrieval sources", specialist=True)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
    else:
        st.rerun()


def _saved_selection(case, item, saved, actor, commit, prefix):
    if not saved:
        return
    labels = {r["source_id"]: f"{r.get('title') or r['source_id']} · {r['locator']}" for r in saved}
    with st.expander(f"Saved sources for this item ({len(saved)})"):
        for row in saved:
            with st.expander(_result_title(row)):
                permission = case.get("retrieval_permissions", {}).get(item["id"], {}).get(row["source_id"])
                allowed = workflow.permitted_retrieval_providers(row, permission)
                _display_source(row | {"providers": allowed})
                if allowed:
                    sid = row["source_id"]
                    keep = st.multiselect("Keep provider access for this passage", allowed, default=allowed,
                                          format_func=PROVIDERS.get, key=prefix + "keep_" + sid)
                    if st.button("Save narrower provider access", key=prefix + "restrict_" + sid,
                                 disabled=set(keep) == set(allowed) or not actor.strip()):
                        try:
                            _narrow_access(case, {(item["id"], sid): keep}, actor, commit)
                        except (ValueError, OSError) as exc:
                            st.error(str(exc))
                        else:
                            st.rerun()
        remove = st.multiselect("Sources to remove from this item", list(labels),
                                format_func=labels.get, key=prefix + "remove_ids")
        if st.button("Remove selected sources", disabled=not remove or not actor.strip(), key=prefix + "remove"):
            _save(case, item, [r for r in saved if r["source_id"] not in remove], actor, commit)


def render_retrieval(case, item, commit, prefix):
    """Render session-local collection controls and reversible per-item selections."""
    prefix += "retrieval:"
    catalog = st.session_state.setdefault("itemdev_retrieval_catalog:" + case["id"], {})
    with st.expander("Search bank and clinical references"):
        actor = st.text_input("Writer selecting sources", case.get("author", ""), key=prefix + "actor")
        _collection_controls(case, catalog, actor, commit, prefix)
        saved = item.get("decision_workflow", {}).get("retrieved_sources", [])
        if not catalog and not saved:
            st.caption("Load an approved collection to search. You can also enter individual bank summaries and evidence passages in the source editor.")
            return
        _saved_selection(case, item, saved, actor, commit, prefix)
        if not catalog:
            return
        available = [c for c in COLLECTIONS if c in catalog]
        collections = st.multiselect("Collections to search", available, default=available, format_func=COLLECTIONS.get,
                                     key=prefix + "collections_" + "_".join(available))
        access = st.selectbox("Search access", list(SEARCH_ACCESS), format_func=SEARCH_ACCESS.get,
                              key=prefix + "search_access")
        terms = st.text_input("Exact terms to prioritize", key=prefix + "exact",
                              placeholder="Test name, threshold or named criterion")
        try:
            query = retrieval.build_query(case, item, exact_terms=terms)
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.expander("Task and stage used for search"):
            for field, value in query.items():
                if field not in {"item_id", "stage_id"} and value:
                    st.text(f"{field.replace('_', ' ').capitalize()} · {value}")
        context = {"query": query, "item": item["id"], "stage": item["stage_id"], "access": access,
                   "collections": {c: catalog[c]["version"] for c in collections}}
        if st.button("Search selected collections", key=prefix + "search", disabled=not collections):
            try:
                results = {c: retrieval.search(catalog[c]["records"], query, c,
                                                provider=None if access == "local" else access)
                           for c in collections}
            except (ValueError, OSError) as exc:
                st.session_state.pop(prefix + "results", None)
                st.error(str(exc))
                return
            st.session_state[prefix + "results"] = {"context": context, "rows": results, "id": core.uid("search")}
        previous = st.session_state.get(prefix + "results")
        if not previous:
            return
        if previous["context"] != context:
            st.caption("Search again to use the current task, stage, collections and provider access.")
            return
        saved_ids = {s["source_id"] for s in saved}
        manual_ids = {s["source_id"] for s in case.get("specialist_sources", [])}
        options, labels = {}, {}
        for collection, results in previous["rows"].items():
            st.markdown("#### " + COLLECTIONS[collection])
            if not results:
                st.caption("No matching records within the selected scope and permissions.")
            for result in results:
                source = result["source"]
                sid = source["source_id"]
                with st.expander(_result_title(source)):
                    _display_source(source, result=result)
                    if sid in saved_ids:
                        st.caption("This source is already saved for the item.")
                    elif sid in manual_ids:
                        st.caption("The manual source editor already uses this source ID.")
                if sid in manual_ids or sid in saved_ids:
                    continue
                token = collection + ":" + sid
                options[token] = source
                labels[token] = f"{COLLECTIONS[collection]} · {source.get('title') or sid} · {source['locator']}"
        if options:
            chosen = st.multiselect("Results to add to this item", list(options), format_func=labels.get,
                                    key=prefix + "chosen_" + previous["id"])
            st.caption("Saving keeps the selected passages and their source references with this item. Run a specialist separately to send the approved context.")
            if st.button("Save selected sources", disabled=not chosen or not actor.strip(), key=prefix + "save"):
                picked = [deepcopy(options[k]) for k in chosen]
                if access != "local":
                    for row in picked:
                        row["providers"] = [p for p in row.get("providers", []) if p == access]
                _save(case, item, saved + picked, actor, commit)
