"""Manual Streamlit authoring workspace; no AI or provider access is required."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Callable

import pandas as pd
import streamlit as st

from core import (
    APPRAISAL_DIMENSIONS, APPRAISAL_GUIDANCE, BLUEPRINT, FRAMEWORK_CROSSWALK,
    MISCONCEPTIONS, OPTION_TYPES, REASONING_GUIDANCE, REASONING_PHASES, REVIEW_STAGES,
    REVIEW_STATUSES, SCORING_DESCRIPTIONS, STAGE_PRESETS, apply_stage_preset,
    combination_table, duplicate_stage, evidence_map, export_markdown, generate_variants,
    new_item, new_option, remove_stage, resize_items, resize_stages, score_item,
    scoring_patterns, timestamp, uid, validate_case, visible_facts, normalize_import,
)


def widget_prefix(case_id):
    """Key prefix for content widgets of one case.

    Streamlit identifies a keyed widget by its key, so a browser keeps showing
    its own value when the saved case changes underneath it (undo, a local
    command, an applied proposal). Bumping the generation gives every content
    widget a new identity so it is rebuilt from the saved case.
    """
    generation = st.session_state.get("itemdev_widget_generation", {}).get(case_id, 0)
    return f"{case_id}::" + (f"r{generation}::" if generation else "")


def refresh_widgets(case_id):
    """Rebuild this case's content widgets from saved data on the next run."""
    generations = st.session_state.setdefault("itemdev_widget_generation", {})
    generations[case_id] = generations.get(case_id, 0) + 1
    keep = {case_id + "::selected_item_id", case_id + "::editor_tab"}
    for key in list(st.session_state):
        if isinstance(key, str) and key.startswith(case_id + "::") and key not in keep and not key.startswith(case_id + "::ai:"):
            del st.session_state[key]


def _table_key(prefix, name):
    return f"{prefix}{name}::{st.session_state.get(prefix + name + '_rev', 0)}"


def _table_saved(prefix, name):
    # A saved table is rebuilt from the case so pending row edits are not replayed.
    st.session_state[prefix + name + "_rev"] = st.session_state.get(prefix + name + "_rev", 0) + 1


def _choice(label, values, current, key, **kwargs):
    values = list(values)
    if current not in values:
        values.append(current)
    return st.selectbox(label, values, index=values.index(current), key=key, **kwargs)


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def _load_json(text, expected, label):
    try:
        value = json.loads(text)
        if not isinstance(value, expected):
            raise ValueError("Expected " + (expected.__name__ if isinstance(expected, type) else " or ".join(t.__name__ for t in expected)))
        return value
    except (ValueError, TypeError) as exc:
        st.error(f"{label}: {exc}. Existing saved content has been kept.")
        return None


def _mutation(case, commit, reason):
    commit(case, reason)
    st.rerun()


def _guidance(case, item, prefix):
    st.markdown("#### Writing guide")
    st.caption("Available with AI off · proposed authoring frameworks")
    with st.expander("Reasoning phase", expanded=True):
        st.markdown(f"**{item['phase']}**")
        st.write(REASONING_GUIDANCE.get(item["phase"], "Define the intended reasoning judgment."))
        st.caption("One phase does not require one item or one disclosure stage. The guide is a proposed synthesis, not a validated learner subscale.")
    if case["mode"] == "appraisal":
        with st.expander("Clinical appraisal criterion", expanded=True):
            dimension = item.get("dimension", APPRAISAL_DIMENSIONS[0])
            st.markdown(f"**{dimension}**")
            st.write(APPRAISAL_GUIDANCE.get(dimension, "Select the criterion you intend to assess."))
            crosswalk = FRAMEWORK_CROSSWALK.get(dimension, {})
            st.caption("Proposed crosswalk: IMPACT · " + ", ".join(crosswalk.get("IMPACT", [])))
            st.caption("RACCCA · " + ", ".join(crosswalk.get("RACCCA", [])))
            st.caption("The local six-dimension taxonomy is an adaptation; neither source establishes family medicine MCQ score validity.")
    with st.expander("Misconceptions and distractors", expanded=True):
        name = st.selectbox("Explore a possible reasoning error", list(MISCONCEPTIONS), key=prefix + "guide_error")
        st.write(MISCONCEPTIONS[name])
        st.caption("Link the error to a finding the option neglects or misuses. Verify why the option is tempting and why it is incorrect under the lead-in. A selection alone does not prove a misconception.")
    with st.expander("What makes this a clear item?"):
        st.markdown("- One focused task tied to the objective.\n- Enough visible evidence to support the key.\n- Comparable, plausible options.\n- An explanation for every keyed and unkeyed option.\n- An explicit response format and scoring rule.")
        if case["mode"] == "unfolding":
            st.write("At each decision, identify what changed. Later findings must not retrospectively make an earlier reasonable decision wrong.")
    with st.expander("Input and scoring help"):
        st.write("A single-best-answer item has exactly one best key. A multiple-selection item has a complete keyed set. Define whether all applicable choices or a specified number should be selected.")
        st.write("Exact-set scoring earns 1 only for the exact set. Partial scoring here uses correct selections minus incorrect selections, divided by the number keyed, with a 0–1 bound. It does not reward unselected distractors.")


def _metadata(case, prefix):
    a, b = st.columns([3, 1])
    with a:
        case["title"] = st.text_input("Title", case["title"], key=prefix + "title")
    with b:
        case["author"] = st.text_input("Author / SME", case.get("author", ""), key=prefix + "author", help="Used to distinguish an independent reviewer from the model author.")
    case["idea"] = st.text_area("Clinical idea", case.get("idea", ""), height=85, key=prefix + "idea", help="Start in your own words. You can complete the full workflow without AI.")
    with st.expander("Blueprint and learning objective", expanded=not bool(case["blueprint"].get("objective"))):
        bp = case["blueprint"]
        st.caption("The starter domains are illustrative. Enter your organization's labels and version when appropriate.")
        c1, c2 = st.columns(2)
        with c1:
            bp["name"] = st.text_input("Blueprint name", bp.get("name", "Illustrative family medicine blueprint"), key=prefix + "bp_name")
            bp["domain"] = st.text_input("Top-level domain", bp.get("domain", ""), key=prefix + "domain", help="Examples: " + "; ".join(BLUEPRINT))
        with c2:
            bp["version"] = st.text_input("Blueprint version", bp.get("version", "1"), key=prefix + "bp_version")
            bp["subdomain"] = st.text_input("Middle-level subdomain / topic", bp.get("subdomain", ""), key=prefix + "subdomain")
        bp["objective"] = st.text_area("Learning objective", bp.get("objective", ""), height=75, key=prefix + "objective")
        st.caption(f"{bp.get('domain') or 'Domain'} → {bp.get('subdomain') or 'Subdomain'} → {bp.get('objective') or 'Objective'}")
    case["patient"] = st.text_area("Shared clinical vignette", case.get("patient", ""), height=145, key=prefix + "patient", help="Only include facts available to the learner from the beginning. Later information belongs in a disclosure stage.")


def _structure(case, commit, prefix):
    with st.expander("Items and disclosure stages", expanded=case["mode"] == "unfolding"):
        a, b, c, d = st.columns([1, 1, 1, 1])
        counts = (max(1, len(case["items"])), max(1, len(case["stages"])))
        if st.session_state.get(prefix + "seen_counts") != counts:
            # The saved counts changed (here, by a command, or by undo): show them.
            st.session_state[prefix + "seen_counts"] = counts
            st.session_state[prefix + "count_items"], st.session_state[prefix + "count_stages"] = counts
        nitems = a.number_input("Items in this case", min_value=1, step=1, key=prefix + "count_items")
        nstages = b.number_input("Disclosure stages", min_value=1, step=1, key=prefix + "count_stages")
        if c.button("Apply counts", key=prefix + "apply_counts", width="stretch"):
            resize_items(case, int(nitems))
            resize_stages(case, int(nstages))
            _mutation(case, commit, "Change item and stage counts")
        if d.button("Add one item", key=prefix + "add_item", width="stretch"):
            case["items"].append(new_item(case["stages"][0]["id"]))
            _mutation(case, commit, "Add item")
        presets = st.columns([1, 1, 2])
        for column, size in zip(presets, sorted(STAGE_PRESETS)):
            if column.button(f"{size}-stage preset", key=prefix + f"preset_{size}", width="stretch", help="Sets the stage count and names untitled stages: " + " → ".join(STAGE_PRESETS[size]) + ". A starting scaffold only; items and written stage content are kept."):
                apply_stage_preset(case, size)
                _mutation(case, commit, f"Apply {size}-stage preset")
        st.caption("Item count, stage count, reasoning phases, and variant count are independent. Removed content is kept in recoverable history. Removing a stage preserves references and flags them for explicit reassignment.")
        if case.get("archived_items") or case.get("archived_stages"):
            st.caption(f"Recoverable: {len(case.get('archived_items', []))} item(s), {len(case.get('archived_stages', []))} stage(s). Increase the corresponding count to restore them.")
        stage_ids = [s["id"] for s in case["stages"]]
        stages = {s["id"]: s for s in case["stages"]}
        if st.session_state.get(prefix + "stage_picker") not in stage_ids:
            st.session_state.pop(prefix + "stage_picker", None)
        sid = st.selectbox("Edit disclosure stage", stage_ids, format_func=lambda x: f"{stage_ids.index(x) + 1}. {stages[x]['title']}", key=prefix + "stage_picker")
        stage = stages[sid]
        sp = prefix + sid + "::"
        stage["title"] = st.text_input("Stage title", stage["title"], key=sp + "title")
        stage["new_information"] = st.text_area("Newly disclosed information", stage.get("new_information", ""), height=85, key=sp + "information")
        stage["transition"] = st.text_area("Transition and purpose of the next decision", stage.get("transition", ""), height=75, key=sp + "transition", help="State what actually happened. Do not silently assume a correct answer to an earlier item.")
        stage["elapsed"] = st.text_input("Time elapsed / sequence note", stage.get("elapsed", ""), key=sp + "elapsed")
        st.caption("Stage order: " + " → ".join(s["title"] for s in case["stages"]))
        here = [i["title"] for i in case["items"] if i["stage_id"] == sid]
        st.caption(f"Items using this stage ({len(here)}): " + ("; ".join(here) if here else "none yet — assign one with ‘Information available at this stage’ below."))
        a, b, c, d, e = st.columns(5)
        index = stage_ids.index(sid)
        if a.button("Move earlier", disabled=index == 0, key=sp + "up", width="stretch"):
            case["stages"][index - 1], case["stages"][index] = case["stages"][index], case["stages"][index - 1]
            _mutation(case, commit, "Reorder disclosure stages")
        if b.button("Move later", disabled=index == len(stage_ids) - 1, key=sp + "down", width="stretch"):
            case["stages"][index + 1], case["stages"][index] = case["stages"][index], case["stages"][index + 1]
            _mutation(case, commit, "Reorder disclosure stages")
        if c.button("Add stage", key=sp + "add_stage", width="stretch"):
            resize_stages(case, len(case["stages"]) + 1)
            _mutation(case, commit, "Add disclosure stage")
        if d.button("Duplicate stage", key=sp + "duplicate_stage", width="stretch", help="Copies the stage text. Items are not copied."):
            duplicate_stage(case, sid)
            _mutation(case, commit, "Duplicate disclosure stage")
        if e.button("Remove stage", key=sp + "remove_stage", disabled=len(stage_ids) <= 1, width="stretch", help="The stage is kept in recoverable history. Items and facts that used it are flagged for reassignment, not deleted."):
            remove_stage(case, sid)
            _mutation(case, commit, "Remove disclosure stage")


def _appraisal_fields(item, prefix, commit, case):
    item["recommendation"] = st.text_area("Preliminary recommendation being appraised", item.get("recommendation", ""), height=130, key=prefix + "recommendation", help="Preserve the exact assessment or plan under evaluation. It may be sound, partly sound, or flawed.")
    item["dimension"] = _choice("Primary appraisal dimension", APPRAISAL_DIMENSIONS, item.get("dimension", APPRAISAL_DIMENSIONS[0]), prefix + "dimension")
    with st.expander("Claim-to-evidence appraisal annotations"):
        fields = ["claim", "judgment", "evidence", "dimension", "rationale", "source"]
        rows = [{f: claim.get(f, "") for f in fields} for claim in item.get("claims", [])]
        with st.form(prefix + "claim_form"):
            edited = st.data_editor(pd.DataFrame(rows, columns=fields), num_rows="dynamic", hide_index=True, width="stretch", key=_table_key(prefix, "claims_table"), column_config={"judgment": st.column_config.SelectboxColumn(options=["Supported", "Contradicted", "Insufficient information", "Not applicable"]), "dimension": st.column_config.SelectboxColumn(options=APPRAISAL_DIMENSIONS)})
            if st.form_submit_button("Save claim annotations"):
                item["claims"] = [row for row in edited.fillna("").astype(str).to_dict("records") if any(v.strip() for v in row.values())]
                _table_saved(prefix, "claims_table")
                _mutation(case, commit, "Edit appraisal evidence annotations")
        st.caption("Insufficient information is not the same as false. These annotations are author review data, not the learner key.")
    with st.expander("Recommendation rubric ratings"):
        st.caption("Optional proposed anchors: 1 = substantial concern; 2 = revision needed; 3 = minor concern; 4 = supported for this case. Missing is not zero. This is not a validated pass/fail scale or a learner score.")
        ratings = item.setdefault("ratings", {})
        for dimension in APPRAISAL_DIMENSIONS:
            rp = prefix + dimension + "::"
            previous = ratings.get(dimension, {})
            if not isinstance(previous, dict):
                previous = {"rating": previous, "rationale": ""}
            choices = ["Not rated", "1", "2", "3", "4"]
            value = str(previous.get("rating", "Not rated"))
            if value not in choices:
                value = "Not rated"
            cols = st.columns([2, 3])
            with cols[0]:
                rating = _choice(dimension, choices, value, rp + "rating")
            with cols[1]:
                rationale = st.text_input(f"Rating rationale · {dimension}", previous.get("rationale", ""), key=rp + "rationale")
            if rating != "Not rated" or rationale:
                ratings[dimension] = {"rating": None if rating == "Not rated" else int(rating), "rationale": rationale}
            else:
                ratings.pop(dimension, None)


def _option_editor(case, item, commit, prefix):
    st.markdown("#### Options and distractor formulation")
    for index, option in enumerate(item["options"]):
        op = prefix + option["id"] + "::"
        label = f"Option {index + 1}" + (" · keyed" if option.get("correct") else "")
        with st.expander(label, expanded=index < 2):
            option["text"] = st.text_area("Option text", option.get("text", ""), height=75, key=op + "text")
            a, b, c = st.columns([1, 1, 1])
            option["correct"] = a.checkbox("Keyed response", value=option.get("correct", False), key=op + "correct", help="For SBA, exactly one option can be keyed. For MS, key every correct selection.")
            option["eligible"] = b.checkbox("Eligible for assembly", value=option.get("eligible", True), key=op + "eligible")
            with c:
                option["option_type"] = _choice("Option type", OPTION_TYPES, option.get("option_type", ""), op + "option_type", format_func=lambda x: x or "Not specified")
            option["role"] = _choice("Selection role for assembled variants", ["essential", "conditional", "random"], option.get("role", "conditional"), op + "role", help="Essential: always included when eligible. Conditional: included whenever its eligibility conditions are met. Random: sampled from the eligible pool to reach the option count. Keyed options are always included.")
            option["rationale"] = st.text_area("Why this option is keyed or unkeyed", option.get("rationale", ""), height=80, key=op + "rationale", help="Tie the explanation to information visible at this decision point.")
            option["misconception"] = _choice("Proposed reasoning error", [""] + list(MISCONCEPTIONS) + ["Unknown cause / other"], option.get("misconception", ""), op + "misconception")
            option["misconception_status"] = _choice("Evidence status of this proposed error", ["author hypothesis", "expert-supported", "empirically supported"], option.get("misconception_status", "author hypothesis"), op + "misconception_status")
            option["supports"] = st.text_input("What makes it attractive?", option.get("supports", ""), key=op + "supports")
            option["excludes"] = st.text_input("What distinguishes it from the keyed judgment?", option.get("excludes", ""), key=op + "excludes")
            with st.expander("Advanced eligibility conditions"):
                st.caption('Example: [{"feature":"age","operator":"gte","value":18}]. Conditions are data, not executable code. Use the cognitive model to define authored feature values.')
                text = st.text_area("Condition rules (JSON)", _json(option.get("conditions", [])), key=op + "conditions", height=80)
                parsed = _load_json(text, (list, dict), "Eligibility conditions") if text.strip() else []
                if parsed is not None:
                    option["conditions"] = parsed
            if st.button("Remove this option", key=op + "remove", disabled=len(item["options"]) <= 2):
                item["options"].pop(index)
                _mutation(case, commit, "Remove option; review keyed set")
    if st.button("Add option", key=prefix + "add_option"):
        item["options"].append(new_option())
        _mutation(case, commit, "Add option")
    shown = st.number_input("Options shown per assembled variant (0 = all authored options)", min_value=0, max_value=len(item["options"]), value=min(int(item.get("option_count") or 0), len(item["options"])), step=1, key=prefix + "option_count", help="Only used when assembling variants. A smaller number samples ‘random’ distractors after keyed, essential and conditional options are placed. A shortage is reported; nothing is padded.")
    item["option_count"] = int(shown) if int(shown) >= 2 else None


def _write(case, commit, prefix, selection_key):
    _metadata(case, prefix)
    _structure(case, commit, prefix)
    if not case["items"]:
        st.info("Add an item to begin writing.")
        return
    ids = [i["id"] for i in case["items"]]
    by_id = {i["id"]: i for i in case["items"]}
    if st.session_state.get(selection_key) not in ids:
        st.session_state.pop(selection_key, None)
    selected = st.selectbox("Current item", ids, format_func=lambda x: f"{ids.index(x)+1}. {by_id[x]['title']}", key=selection_key)
    item = by_id[selected]
    ip = prefix + item["id"] + "::"
    a, b, c, d = st.columns(4)
    index = ids.index(selected)
    if a.button("Earlier item", key=ip + "earlier", disabled=index == 0):
        case["items"][index - 1], case["items"][index] = case["items"][index], case["items"][index - 1]
        _mutation(case, commit, "Reorder items")
    if b.button("Later item", key=ip + "later", disabled=index == len(ids) - 1):
        case["items"][index + 1], case["items"][index] = case["items"][index], case["items"][index + 1]
        _mutation(case, commit, "Reorder items")
    if c.button("Duplicate", key=ip + "duplicate"):
        duplicate = deepcopy(item)
        duplicate["id"] = uid("item")
        duplicate["title"] += " · copy"
        for option in duplicate["options"]:
            option["id"] = uid("opt")
        duplicate["review_status"] = "Draft"
        case["items"].insert(index + 1, duplicate)
        _mutation(case, commit, "Duplicate item")
    if d.button("Archive item", key=ip + "archive", disabled=len(ids) <= 1):
        case.setdefault("archived_items", []).append(case["items"].pop(index))
        _mutation(case, commit, "Archive item")
    main, guide = st.columns([3, 1.2], gap="large")
    with main:
        item["title"] = st.text_input("Item title", item["title"], key=ip + "title")
        stage_map = {s["id"]: s["title"] for s in case["stages"]}
        item["stage_id"] = _choice("Information available at this stage", stage_map, item["stage_id"], ip + "stage", format_func=lambda x: stage_map.get(x, "Missing stage — reassign"))
        item["phase"] = _choice("Primary reasoning phase", REASONING_PHASES, item["phase"], ip + "phase")
        others = [phase for phase in REASONING_PHASES if phase != item["phase"]]
        item["secondary_phases"] = st.multiselect("Secondary reasoning phases (optional)", others, default=[x for x in item.get("secondary_phases", []) if x in others], key=ip + "secondary_phases", help="Phases can overlap and recur. Secondary tags do not create extra items or stages.")
        item["stem"] = st.text_area("Item-specific stem / additional context", item.get("stem", ""), height=95, key=ip + "stem", help="The shared vignette is automatically included in learner preview. Do not duplicate it here.")
        if case["mode"] == "appraisal":
            _appraisal_fields(item, ip, commit, case)
        item["lead_in"] = st.text_area("Focused lead-in", item.get("lead_in", ""), height=80, key=ip + "lead_in")
        old_format = item["format"]
        item["format"] = st.radio("Response format", ["SBA", "MS"], index=0 if old_format == "SBA" else 1, format_func=lambda x: "Single best answer" if x == "SBA" else "Multiple selection", horizontal=True, key=ip + "format")
        if old_format != item["format"]:
            st.warning("Response format changed. Review all keyed and unkeyed options and scoring; existing content is preserved.")
        item["selection_instruction"] = st.text_input("Learner selection instruction", item.get("selection_instruction", ""), key=ip + "instruction", help="For example, select all applicable options, or select exactly two.")
        if item["format"] == "MS":
            item["scoring"] = _choice("Scoring policy", ["exact", "partial"], item.get("scoring", "exact"), ip + "scoring", format_func=lambda x: "Exact keyed set" if x == "exact" else "Partial credit with incorrect-selection penalty")
            st.caption(SCORING_DESCRIPTIONS[item["scoring"]])
            count = st.number_input("Specified selection count (0 = all applicable)", min_value=0, value=int(item.get("select_count") or 0), step=1, key=ip + "select_count")
            item["select_count"] = int(count) or None
        else:
            item["scoring"] = "exact"
            item["select_count"] = None
        if any(o.get("correct") for o in item["options"]):
            with st.expander("Scoring preview · representative response patterns"):
                try:
                    st.dataframe(pd.DataFrame(scoring_patterns(item)), hide_index=True, width="stretch")
                    st.caption("The total can hide which options were chosen. Review option-level patterns, especially for safety-related judgments.")
                except ValueError as exc:
                    st.info(f"Not scorable yet: {exc}.")
        fact_ids = [f["id"] for f in case.get("facts", [])]
        if fact_ids:
            fmap = {f["id"]: f.get("name", f["id"]) for f in case["facts"]}
            current = [x for x in item.get("required_facts", []) if x in fact_ids]
            item["required_facts"] = st.multiselect("Facts required to justify this key", fact_ids, default=current, format_func=lambda x: fmap[x], key=ip + "required_facts", help="Validation will flag a key that depends on a fact disclosed only later.")
        _option_editor(case, item, commit, ip)
    with guide:
        _guidance(case, item, ip)


def _model(case, commit, prefix, add_cases=None):
    st.markdown("### Cognitive model and information sources")
    st.caption("Blueprint → clinical task → relevant features → permitted values and constraints → item wording and options. Clinical rules are authored restrictions, not universal diagnostic truths.")
    case["scenario"] = st.text_input("Scenario / outcome being modeled", case.get("scenario", ""), key=prefix + "scenario")
    with st.expander("Shared facts and disclosure", expanded=True):
        cols = ["id", "name", "value", "disclosed_at", "source"]
        # Show readable stage names in the table; store stable stage identifiers.
        labels = {"shared": "Shared · from the start"}
        for number, stage in enumerate(case["stages"], 1):
            labels[stage["id"]] = f"Stage {number} · {stage['title']}"
        def label_for(value):
            return labels.get(value or "shared", f"Missing stage · {value}")
        rows = [{k: str(f.get(k, "")) for k in cols} | {"disclosed_at": label_for(f.get("disclosed_at"))} for f in case.get("facts", [])]
        back = {label: stage_id for stage_id, label in labels.items()} | {label_for(f.get("disclosed_at")): f.get("disclosed_at") or "shared" for f in case.get("facts", [])}
        st.caption("Record each fact once and choose when the learner first sees it. Use the value to distinguish not assessed, unknown, absent and normal.")
        with st.form(prefix + "facts_form"):
            table = st.data_editor(pd.DataFrame(rows, columns=cols), num_rows="dynamic", hide_index=True, width="stretch", key=_table_key(prefix, "facts_table"), column_config={"id": None, "disclosed_at": st.column_config.SelectboxColumn("Visible from", options=list(dict.fromkeys(list(labels.values()) + [r["disclosed_at"] for r in rows]))), "name": "Fact", "value": "Value / state", "source": "Source / provenance"})
            if st.form_submit_button("Save facts and disclosure"):
                facts = [row for row in table.fillna("").astype(str).to_dict("records") if row.get("name", "").strip() or row.get("value", "").strip()]
                for fact in facts:
                    fact["id"] = fact.get("id") or uid("fact")
                    fact["disclosed_at"] = back.get(fact.get("disclosed_at"), "shared")
                case["facts"] = facts
                _table_saved(prefix, "facts_table")
                _mutation(case, commit, "Edit facts and disclosure")
    with st.expander("Feature values and variant templates", expanded=True):
        st.caption('Use authored values as a JSON list, such as ["mild", "hacking"] or [19,20,21]. Insert {{feature name}} into vignette or item text to assemble variants. Values must be clinically reviewed.')
        fields = ["id", "name", "values", "disclosed_at"]
        rows = [{"id": f.get("id", ""), "name": f.get("name", ""), "values": json.dumps(f.get("values", []), ensure_ascii=False), "disclosed_at": f.get("disclosed_at", "shared")} for f in case.get("features", [])]
        with st.form(prefix + "features_form"):
            table = st.data_editor(pd.DataFrame(rows, columns=fields), num_rows="dynamic", hide_index=True, width="stretch", key=_table_key(prefix, "features_table"), column_config={"id": None, "values": "Permitted values (JSON list)"})
            if st.form_submit_button("Save feature values"):
                features, valid = [], True
                for row in table.fillna("").astype(str).to_dict("records"):
                    if not row["name"].strip() and not row["values"].strip():
                        continue
                    values = _load_json(row["values"] or "[]", list, f"Feature {row['name'] or '(unnamed)'}")
                    if values is None:
                        valid = False
                        continue
                    features.append({"id": row.get("id") or uid("feature"), "name": row["name"], "values": values, "disclosed_at": row.get("disclosed_at") or "shared"})
                if valid:
                    case["features"] = features
                    _table_saved(prefix, "features_table")
                    _mutation(case, commit, "Edit feature values")
        try:
            combos = combination_table(case)
        except ValueError as exc:
            combos = None
            st.warning(str(exc))
        if combos and combos["total"]:
            st.markdown(f"**Constraint matrix** · {combos['permitted']} of {combos['total']} authored combinations are permitted by the current rules.")
            st.dataframe(pd.DataFrame(combos["rows"]), hide_index=True, width="stretch", height=min(320, 38 * (len(combos["rows"]) + 1)))
            if combos["truncated"]:
                st.caption(f"Showing the first {len(combos['rows'])} combinations.")
            st.caption("‘Permitted’ reflects authored restrictions only. It is not a statement of clinical truth or likelihood.")
    with st.expander("Constraint rules"):
        st.caption('Rules are ANDed. Example: [{"feature":"age","operator":"gte","value":18}]. Operators: equals, not_equals, in, not_in, gt, gte, lt, lte, exists, not_exists. Compound all/any/not rules are supported. These are data, not executable code.')
        with st.form(prefix + "constraints_form"):
            rules = st.text_area("Constraint data (JSON list)", _json(case.get("constraints", [])), height=150, key=prefix + "constraints_json")
            if st.form_submit_button("Save constraint rules"):
                parsed = _load_json(rules, list, "Constraints")
                if parsed is not None:
                    candidate = deepcopy(case)
                    candidate["constraints"] = parsed
                    if any(f["code"] == "SCHEMA" for f in validate_case(candidate)):
                        st.error("The rule shape is invalid. Check feature names, operators, and value types.")
                    else:
                        case["constraints"] = parsed
                        _mutation(case, commit, "Edit constraint rules")
    with st.expander("Clinical sources", expanded=True):
        fields = ["title", "url", "note"]
        rows = [{k: row.get(k, "") for k in fields} for row in case.get("sources", [])]
        with st.form(prefix + "sources_form"):
            table = st.data_editor(pd.DataFrame(rows, columns=fields), num_rows="dynamic", hide_index=True, width="stretch", key=_table_key(prefix, "sources_table"))
            if st.form_submit_button("Save sources"):
                candidate = deepcopy(case)
                candidate["sources"] = [row for row in table.fillna("").astype(str).to_dict("records") if any(v.strip() for v in row.values())]
                try:
                    normalized = normalize_import(candidate)
                    case["sources"] = normalized["sources"]
                    _table_saved(prefix, "sources_table")
                    _mutation(case, commit, "Edit clinical sources")
                except ValueError as exc:
                    st.error(str(exc))
    with st.expander("Assemble deterministic variants"):
        st.caption("Works with AI off. No new clinical prose is invented. Eligibility is checked before sampling; an insufficient option pool is reported.")
        count = st.number_input("Number of case variants", min_value=1, value=1, step=1, key=prefix + "variant_count")
        seed = st.number_input("Reproducible sampling seed", min_value=0, value=1, step=1, key=prefix + "variant_seed")
        if st.button("Assemble variants", key=prefix + "assemble"):
            try:
                variants = generate_variants(case, int(count), int(seed))
                st.session_state[prefix + "assembled_variants"] = variants
                st.success(f"Assembled {len(variants)} permissible case variant(s); expert review is still required.")
            except ValueError as exc:
                st.session_state.pop(prefix + "assembled_variants", None)
                st.error(str(exc))
        variants = st.session_state.get(prefix + "assembled_variants")
        if variants and variants[0].get("generated_from", {}).get("version") != case.get("version"):
            st.session_state.pop(prefix + "assembled_variants", None)
            variants = None
            st.info("The case changed after assembly. Assemble again to use the current content.")
        if variants:
            st.markdown("**Assembled variants and option-selection reasons**")
            for number, variant in enumerate(variants, 1):
                origin = variant.get("generated_from", {})
                with st.expander(f"Variant {number} · " + (", ".join(f"{k} = {v}" for k, v in origin.get("assignment", {}).items()) or "no varying features")):
                    st.write(variant["patient"] or "_No shared vignette._")
                    reasons = origin.get("option_reasons", [])
                    if reasons:
                        st.dataframe(pd.DataFrame([{"Item": r["item"], "Option": r["option"], "Included": "Yes" if r["included"] else "No", "Reason": r["reason"]} for r in reasons]), hide_index=True, width="stretch")
            a, b = st.columns(2)
            a.download_button("Download assembled variants (JSON)", _json({"schema_version": 1, "cases": variants}), file_name="item_dev_variants.json", mime="application/json", key=prefix + "variants_download", width="stretch")
            if add_cases is not None and b.button("Add variants to the draft library", key=prefix + "variants_to_library", width="stretch", help="Each variant becomes its own draft with its origin recorded. Review is still required."):
                add_cases(deepcopy(variants))
                st.session_state.pop(prefix + "assembled_variants", None)
                st.session_state["itemdev_notice"] = f"Added {len(variants)} assembled variant(s) to the draft library as unreviewed drafts."
                st.rerun()


def _review(case, commit, prefix):
    st.markdown("### Validation and recorded review")
    issues = validate_case(case)
    a, b, c = st.columns(3)
    a.metric("Structural errors", sum(f["severity"] == "error" for f in issues))
    b.metric("Missing input / warnings", sum(f["severity"] == "warning" for f in issues))
    c.metric("Expert review prompts", sum(f["severity"] == "review" for f in issues))
    st.caption("Structural checks, clinical judgments, and empirical psychometric evidence are different. No learner response data or psychometric results are manufactured.")
    for issue in issues:
        text = f"**{issue['path']}** · {issue['message']}"
        {"error": st.error, "warning": st.warning, "review": st.info}[issue["severity"]](text)
    with st.expander("Record a model or item review", expanded=True):
        with st.form(prefix + "review_form"):
            stage = st.selectbox("Review step", REVIEW_STAGES, key=prefix + "review_stage")
            reviewer = st.text_input("Reviewer name", key=prefix + "reviewer")
            decision = st.selectbox("Recorded decision / status", REVIEW_STATUSES + ["Revisions required"], key=prefix + "review_decision")
            comments = st.text_area("Evidence, comments, and required revisions", key=prefix + "review_comments")
            a, b, c = st.columns(3)
            labels = ["Not rated", "1 · Reject", "2 · Major revision", "3 · Minor revision", "4 · Accept"]
            content = a.selectbox("Content", labels, key=prefix + "review_content")
            logic = b.selectbox("Logic", labels, key=prefix + "review_logic")
            presentation = c.selectbox("Presentation", labels, key=prefix + "review_presentation")
            explicit = st.checkbox("I am recording my own review decision for this content version.", key=prefix + "review_confirm")
            if st.form_submit_button("Record review decision"):
                author = case.get("author", "").strip()
                errors = [f for f in issues if f["severity"] == "error"]
                if not reviewer.strip() or not explicit:
                    st.error("Enter the responsible reviewer and acknowledge the recorded decision.")
                elif stage == "Independent model review" and (not author or reviewer.strip().casefold() == author.casefold()):
                    st.error("An independent review needs a named model author and a different reviewer.")
                elif decision in {"Ready for pilot", "Approved for use"} and errors:
                    st.error("Resolve structural errors before recording this release status.")
                elif decision == "Approved for use" and stage != "Release decision":
                    st.error("Operational approval must be explicitly recorded at the Release decision step.")
                else:
                    case.setdefault("reviews", []).append({"id": uid("review"), "stage": stage, "reviewer": reviewer.strip(), "decision": decision, "case_version": case["version"], "timestamp": timestamp(), "content": content, "logic": logic, "presentation": presentation, "comments": comments, "requires_reassessment": False, "valid": True})
                    case["review_status"] = decision
                    _mutation(case, commit, "Record review decision")
    if case.get("reviews"):
        st.markdown("#### Review history")
        st.dataframe(pd.DataFrame(case["reviews"]), hide_index=True, width="stretch")
    with st.expander("Model validation table"):
        stage_map = {s["id"]: s["title"] for s in case["stages"]}
        table = []
        for item in case["items"]:
            table.append({"Item": item["title"], "Stage": stage_map.get(item["stage_id"], "Missing"), "Objective": case["blueprint"].get("objective", ""), "Phase": item["phase"], "Format": item["format"], "Keyed options": "; ".join(o["text"] for o in item["options"] if o["correct"]), "Required facts": ", ".join(item.get("required_facts", [])), "Appraisal dimension": item.get("dimension", "") if case["mode"] == "appraisal" else ""})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    a, b, c = st.columns(3)
    a.download_button("Author Markdown", export_markdown(case, True), file_name="case_author.md", mime="text/markdown", key=prefix + "author_md")
    b.download_button("Learner Markdown", export_markdown(case, False), file_name="case_learner.md", mime="text/markdown", key=prefix + "learner_md")
    c.download_button("Full case JSON", _json(case), file_name="case.json", mime="application/json", key=prefix + "case_json")


def _preview(case, prefix):
    st.markdown("### Learner preview")
    st.caption("This is a preview, not an examination administration system. Keys and rationales appear only after a response is submitted.")
    stages = case["stages"]
    if not stages:
        st.info("Create a stage first.")
        return
    state_key = prefix + "preview_state"
    state = st.session_state.setdefault(state_key, {"index": 0, "responses": {}, "submitted": {}, "version": case.get("version"), "attempt": 0})
    if state.get("version") != case.get("version"):
        state.update(index=0, responses={}, submitted={}, version=case.get("version"))
    attempt = f"{state['attempt']}::" if state.get("attempt") else ""
    state["index"] = min(state["index"], len(stages) - 1)
    st.markdown(case["patient"] or "_Shared vignette has not been written._")
    shared = [f for f in case.get("facts", []) if f.get("disclosed_at") in (None, "", "shared")]
    if shared:
        st.caption("Shared facts · " + " · ".join(f"{f['name']}: {f.get('value', '')}" for f in shared))
    unfolding = case["mode"] == "unfolding"
    if not unfolding:
        indices = list(range(len(stages)))
    else:
        indices = list(range(state["index"] + 1))
        st.progress((state["index"] + 1) / len(stages), text=f"Stage {state['index'] + 1} of {len(stages)}")
    for sidx in indices:
        stage = stages[sidx]
        if sidx and stages[sidx - 1].get("transition"):
            st.write(stages[sidx - 1]["transition"])
        st.markdown(f"#### {stage['title']}")
        if stage.get("elapsed"):
            st.caption("Time / sequence · " + stage["elapsed"])
        if stage.get("new_information"):
            st.write(stage["new_information"])
        facts = [f for f in visible_facts(case, stage["id"]) if f.get("disclosed_at") == stage["id"]]
        if facts:
            st.caption(" · ".join(f"{f['name']}: {f.get('value', '')}" for f in facts))
        for item in [i for i in case["items"] if i["stage_id"] == stage["id"]]:
            pp = prefix + "preview::" + attempt + item["id"] + "::"
            st.markdown(f"**{item['title']}**")
            if item.get("stem"):
                st.write(item["stem"])
            if case["mode"] == "appraisal":
                st.info(item.get("recommendation") or "No preliminary recommendation entered.")
            st.write(item["lead_in"] or "No lead-in entered.")
            st.caption(item.get("selection_instruction", ""))
            locked = unfolding and sidx < state["index"]
            options = item["options"]
            opts = {o["id"]: o for o in options}
            with st.form(pp + "response"):
                if item["format"] == "SBA":
                    selected = st.radio("Choose one", list(opts), index=None, format_func=lambda oid: opts[oid]["text"] or "(blank option)", disabled=locked, key=pp + "choice")
                    selected_ids = [selected] if selected else []
                else:
                    selected_ids = []
                    for idx, option in enumerate(options):
                        if st.checkbox(option["text"] or f"Blank option {idx + 1}", disabled=locked, key=pp + option["id"]):
                            selected_ids.append(option["id"])
                if st.form_submit_button("Submit response", disabled=locked):
                    state["responses"][item["id"]] = selected_ids
                    try:
                        state["submitted"][item["id"]] = score_item(item, selected_ids)
                    except ValueError as exc:
                        state["submitted"][item["id"]] = {"error": str(exc)}
            result = state["submitted"].get(item["id"])
            if result:
                if "error" in result:
                    st.error("This draft cannot be scored: " + result["error"])
                else:
                    st.success(f"Preview score: {result['score']:.2f} / {result['max_score']:.0f}")
                    with st.expander("Submitted-response feedback"):
                        for option in options:
                            st.markdown(f"**{'Keyed' if option['correct'] else 'Unkeyed'} · {option['text']}**")
                            st.write(option.get("rationale") or "No rationale recorded.")
                        st.caption(result["description"])
    if unfolding:
        current = stages[state["index"]]
        current_items = [i for i in case["items"] if i["stage_id"] == current["id"]]
        submitted = all(i["id"] in state["submitted"] and "error" not in state["submitted"][i["id"]] for i in current_items)
        if state["index"] < len(stages) - 1:
            if st.button("Advance and lock this stage", disabled=not submitted, key=prefix + "preview_next"):
                state["index"] += 1
                st.rerun()
        else:
            st.caption("End of case. Earlier-stage responses remain locked.")
    orphans = [i["title"] for i in case["items"] if i["stage_id"] not in {s["id"] for s in stages}]
    if orphans:
        st.warning("Not shown because their disclosure stage is missing: " + "; ".join(orphans) + ". Reassign them in the Write tab.")
    if st.button("Restart learner preview", key=prefix + "preview_restart"):
        for key in list(st.session_state):
            if isinstance(key, str) and key.startswith(prefix + "preview::"):
                del st.session_state[key]
        st.session_state[state_key] = {"index": 0, "responses": {}, "submitted": {}, "version": case.get("version"), "attempt": state.get("attempt", 0) + 1}
        st.rerun()


def _case_map(case, prefix):
    st.markdown("### Case map")
    bp = case["blueprint"]
    st.caption("Blueprint trace · " + " → ".join([bp.get("name") or "Blueprint", bp.get("domain") or "domain not set", bp.get("subdomain") or "subdomain not set", bp.get("objective") or "objective not set"]))
    issues = validate_case(case)
    per_item = {}
    for issue in issues:
        if issue["path"].startswith("items["):
            index = int(issue["path"][6:issue["path"].index("]")])
            per_item.setdefault(index, {"error": 0, "warning": 0, "review": 0})[issue["severity"]] += 1
    st.markdown("#### Stages and their items")
    st.caption(f"{len(case['items'])} item(s) across {len(case['stages'])} disclosure stage(s). These counts are independent.")
    stage_ids = {s["id"] for s in case["stages"]}
    def line(index, item):
        counts = per_item.get(index, {})
        flags = ", ".join(f"{counts[k]} {label}" for k, label in (("error", "error(s)"), ("warning", "missing input"), ("review", "review prompt(s)")) if counts.get(k)) or "no findings"
        return f"- **{index + 1}. {item['title']}** · {item['phase']} · {'single best answer' if item['format'] == 'SBA' else 'multiple selection'} · {flags}"
    for number, stage in enumerate(case["stages"], 1):
        st.markdown(f"**Stage {number} · {stage['title']}**")
        lines = [line(i, item) for i, item in enumerate(case["items"]) if item["stage_id"] == stage["id"]]
        st.markdown("\n".join(lines) if lines else "- _No items use this stage._")
    orphans = [line(i, item) for i, item in enumerate(case["items"]) if item["stage_id"] not in stage_ids]
    if orphans:
        st.error("Items whose disclosure stage is missing — reassign them in the Write tab:")
        st.markdown("\n".join(orphans))
    st.markdown("#### Evidence-to-decision map")
    st.caption("Visible information → reasoning judgment → keyed response → anticipated errors → what happens next. An MCQ samples judgment; it does not show a learner's internal reasoning.")
    st.dataframe(pd.DataFrame(evidence_map(case)), hide_index=True, width="stretch")
    phases = {}
    for item in case["items"]:
        phases[item["phase"]] = phases.get(item["phase"], 0) + 1
    st.markdown("#### Reasoning-phase coverage")
    st.dataframe(pd.DataFrame([{"Reasoning phase": phase, "Items with this primary phase": phases.get(phase, 0)} for phase in REASONING_PHASES]), hide_index=True, width="stretch")
    st.caption("Coverage is descriptive. A case need not sample every phase, and a phase can recur.")


def render_editor(case: dict, commit: Callable, add_cases: Callable | None = None) -> None:
    """Render all manual tools; caller persists field edits after this returns.

    ``add_cases(list_of_cases)`` saves assembled variants as new library drafts.
    """
    prefix = widget_prefix(case["id"])
    write, overview, model, review, preview = st.tabs(["✎ Write", "☰ Case map", "◇ Cognitive model", "✓ Validate & review", "▷ Learner preview"], key=case["id"] + "::editor_tab", on_change="rerun")
    with write:
        _write(case, commit, prefix, case["id"] + "::selected_item_id")
    with overview:
        _case_map(case, prefix)
    with model:
        _model(case, commit, prefix, add_cases)
    with review:
        _review(case, commit, prefix)
    with preview:
        _preview(case, prefix)
