"""Author controls for decision modeling and explicitly invoked specialists."""

from copy import deepcopy
import json

import streamlit as st

import core
import decision_model as dm
import specialist_workflow as workflow
import storage
from ai_adapter import AdapterError, dispatch_specialist, specialist_config, specialist_registry
from ui_authoring_spaces import render_authoring_spaces
from ui_semantic_retrieval import render_retrieval


TASK_FIELDS = {
    "decision": "Decision to assess", "setting": "Clinical setting", "population": "Clinical population",
    "decision_timing": "Decision timing",
    "examinee_population": "Intended examinees", "assessment_intent": "Observable assessment intent",
    "intended_key": "Writer-proposed key",
}


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def _change(commit, operation, reason):
    try:
        commit(operation(), reason, specialist=True)
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        st.error(str(exc))
    else:
        st.rerun()


def _source_controls(case, commit, prefix):
    with st.expander("Bank summaries and evidence passages"):
        st.caption("Supply authorized excerpts or labeled synthetic records. Assign a stable source ID, source class, date or version, locator and excerpt. Optionally scope by item, stage, population and setting. An empty providers list keeps the record local.")
        st.caption("Source classes include bank, guideline, study, evidence summary, practice summary, and misconception evidence.")
        with st.form(prefix + "sources_form"):
            text = st.text_area("Source records (JSON)", _json(case.get("specialist_sources", [])),
                                placeholder='[{"source_id":"fixture-1","source_class":"evidence summary","source_date":"Fixture v1","locator":"Rule 1","excerpt":"Synthetic threshold rule","providers":[]}]',
                                height=120, key=prefix + "sources")
            if st.form_submit_button("Save source records"):
                def update():
                    rows = json.loads(text)
                    workflow.validate_manual_sources(case, rows)
                    return deepcopy(case) | {"specialist_sources": rows}
                _change(commit, update, "Edit specialist source records")


def _record_row(row, actions):
    return {
        "Scenario": row["scenario_id"], "Outcome": row["outcome"],
        "Key": ", ".join(actions.get(a, a) for a in row["keyed_action_ids"]),
        "Changes": ", ".join(f"{k} = {v}" for k, v in row.get("changes", {}).items()),
        "Competing or possible": ", ".join(actions.get(a, a) for a in row["competing_action_ids"]),
        "Triggered rules": ", ".join(row["triggered_rule_ids"]),
        "Unresolved rules": ", ".join(row["unresolved_rule_ids"]),
        "Missing findings": ", ".join(row["missing_features"]), "Evidence": ", ".join(row["evidence_claim_ids"]),
        "Reason": " ".join(row["reasons"]),
    }


def _scenario_controls(case, item, commit, prefix, actor, notes, status, add_cases):
    model, current = status["model"], status["row"]
    rows = dm.scenario_rows(model, case, item)
    labels = {a["action_id"]: a["label"] for a in model["actions"]}
    st.markdown("#### Scenario evaluation")
    st.dataframe([_record_row(r, labels) for r in rows.values()], hide_index=True, width="stretch")
    st.write("Current case using the selected scenario and live stage facts")
    live = _record_row(current, labels)
    st.write(f"**{live['Scenario']}** · {live['Outcome']}")
    for label in ("Key", "Competing or possible", "Triggered rules", "Unresolved rules", "Missing findings", "Reason"):
        if live[label]:
            st.write(f"**{label}** · {live[label]}")
    if not status["scope_current"]:
        st.warning("Reconfirm the model against the updated task, source records, or stage.")
    elif not status["selection_current"]:
        st.warning("Confirm the current scenario and locally evaluated key. Review existing wording, options, and rationales after changes.")
    if current["unsupported_rule_ids"]:
        st.warning("Clinical support is missing for " + ", ".join(current["unsupported_rule_ids"]) + ". Document the SME judgment and evidence gap.")
    state = item["decision_workflow"]
    intended = state.get("intended_key")
    if intended:
        st.write("Writer-proposed key · " + intended)
    for question in state["model"].get("review_questions", []):
        st.warning(question)
    if current["evidence_claim_ids"]:
        full_claims = dm.by_id(workflow._claims(case, item), "claim_id")
        sources = dm.by_id(workflow._sources(case, item), "source_id")
        with st.expander("Rule evidence and applicability"):
            for id_ in current["evidence_claim_ids"]:
                if id_ not in full_claims:
                    st.warning(f"{id_} requires fresh evidence review.")
                    continue
                claim = full_claims[id_]
                source = sources[claim["source_id"]]
                st.write(f"{id_} · {source['source_id']} · {source['source_date']} · {source['locator']}")
                st.write(source["excerpt"])
                st.write(claim)
                if source.get("url", "").startswith(("http://", "https://")):
                    st.link_button("Open clinical source", source["url"])
    feasible = [key for key, row in rows.items() if row["outcome"] != "infeasible"]
    chosen = current["scenario_id"]
    if feasible:
        chosen = st.selectbox("Scenario to review", feasible, index=feasible.index(current["scenario_id"])
                              if current["scenario_id"] in feasible else 0, key=prefix + "scenario")
        apply_values = st.checkbox("Apply changed values to linked facts visible at this stage",
                                   key=prefix + "apply_values")
        first, second = st.columns(2)
        if first.button("Confirm scenario and key", disabled=not status["scope_current"], key=prefix + "confirm"):
            _change(commit, lambda: workflow.accept_scenario(case, item["id"], chosen, actor, notes,
                               apply_values=apply_values), "Confirm decision scenario and action key")
        if second.button("Confirm edited facts as new scenario", disabled=not status["scope_current"], key=prefix + "confirm_new"):
            _change(commit, lambda: workflow.accept_scenario(case, item["id"], current["scenario_id"], actor, notes,
                               as_new=True), "Confirm edited decision scenario")
        if add_cases and st.button("Save this scenario as a linked draft", disabled=not status["scope_current"], key=prefix + "save_variant"):
            try:
                draft = workflow.accept_scenario(case, item["id"], chosen, actor, notes, apply_values=True)
                draft["title"] += f" · {chosen}"
                draft["generated_from"] = {"case_id": case["id"], "version": case["version"],
                                           "item_id": item["id"], "scenario_id": chosen,
                                           "generator": "reviewed decision scenario"}
                add_cases([draft])
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
    render_authoring_spaces(case, item, status, actor, notes, prefix, commit, scenario_id=chosen)
    with st.expander("Feasibility and visibility checks"):
        stage_map = {s["id"]: s["title"] for s in case["stages"]}
        sid = st.selectbox("Stage for a withheld-information check", list(stage_map), format_func=stage_map.get,
                           index=list(stage_map).index(item["stage_id"]), key=prefix + "stage")
        visibility = dm.scenario_rows(model, case, item, stage_id=sid)
        st.dataframe([_record_row(visibility[current["scenario_id"]], labels)], hide_index=True, width="stretch")
        signature = workflow.proposal_signature(case)
        if st.button("Calculate more permitted-value combinations", key=prefix + "generate"):
            generated = deepcopy(model)
            generated["scenarios"] += dm.generate_scenarios(model)
            st.session_state[prefix + "generated"] = dm.scenario_rows(generated, case, item)
            st.session_state[prefix + "generated_data"] = generated
            st.session_state[prefix + "generated_signature"] = signature
        if st.session_state.get(prefix + "generated_signature") == signature:
            generated_rows = st.session_state[prefix + "generated"]
            st.dataframe([_record_row(r, labels) for r in generated_rows.values()], hide_index=True, width="stretch")
            if st.button("Keep feasible generated scenarios", key=prefix + "save_generated"):
                generated = deepcopy(st.session_state[prefix + "generated_data"])
                generated["scenarios"] = [s for s in generated["scenarios"]
                    if generated_rows[s["scenario_id"]]["outcome"] != "infeasible" or s in model["scenarios"]]
                _change(commit, lambda: workflow.save_model(case, item["id"], generated, actor),
                        "Save locally generated decision scenarios")


def render_workflow(case, item, commit, add_cases=None):
    if not item:
        return
    prefix = case["id"] + "::decision:" + item["id"] + ":"
    with st.expander("Family Medicine decision workflow"):
        _source_controls(case, commit, prefix)
        render_retrieval(case, item, commit, prefix)
        state = item.get("decision_workflow", {})
        with st.form(prefix + "task_form"):
            st.markdown("#### Target task")
            values = {k: st.text_input(label, state.get(k, ""), key=prefix + k) for k, label in TASK_FIELDS.items()}
            values["locked_wording_fields"] = st.multiselect(
                "Keep this wording when a specialist drafts", list(workflow.LOCKABLE_WORDING),
                default=state.get("locked_wording_fields", []), format_func=workflow.LOCKABLE_WORDING.get,
                key=prefix + "wording_locks",
            )
            if st.form_submit_button("Save target task"):
                def update():
                    candidate = deepcopy(case)
                    workflow.item_for(candidate, item["id"]).setdefault("decision_workflow", {}).update(values)
                    return candidate
                _change(commit, update, "Edit decision task")
        actor = st.text_input("SME or clinical reviewer", case.get("author", ""), key=prefix + "actor")
        notes = st.text_area("Decision evidence and review notes", key=prefix + "notes", height=90)
        try:
            status = workflow.model_status(case, item)
            conflicts = workflow._unresolved_evidence(case, item)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
            return
        if conflicts:
            for conflict in conflicts:
                st.warning(conflict)
            if st.button("Record evidence resolution", key=prefix + "evidence_resolution"):
                _change(commit, lambda: workflow.record_human_decision(case, item["id"], actor, notes, "evidence_resolution"),
                        "Record SME evidence resolution")
        with st.expander("Review or edit clinical claims"):
            evidence = state.get("evidence", {}).get("payload", {"claims": [], "practice_context": [], "conflicts": []})
            with st.form(prefix + "evidence_form"):
                raw = st.text_area("Clinical claims and conflicts (JSON)", _json(evidence), height=140, key=prefix + "evidence")
                applicable = ""
                if any(s["source_class"] in workflow.CLINICAL_CLASSES for s in state.get("retrieved_sources", [])):
                    applicable = st.text_input("Retrieved claim IDs reviewed as applicable",
                        ", ".join(state.get("evidence", {}).get("applicable_retrieved_claim_ids", [])),
                        help="Enter comma-separated claim IDs after checking each passage against the population, timing and exceptions. Only these retrieved claims can support a rule.",
                        key=prefix + "applicable_claims")
                if st.form_submit_button("Accept clinical evidence"):
                    def accept():
                        return workflow.accept_evidence(case, item["id"], json.loads(raw), actor,
                            applicable_claim_ids=[v.strip() for v in applicable.split(",") if v.strip()])
                    _change(commit, accept, "Accept SME-reviewed clinical evidence")
        with st.expander("Review or edit features, constraints, and action rules"):
            empty = {"features": [], "feasibility_constraints": [], "actions": [], "decision_rules": [],
                     "scenarios": [], "baseline_scenario_id": "baseline",
                     "proposed_key": {"action_ids": [], "source_claim_ids": [], "awaiting_sme_review": True}}
            with st.form(prefix + "model_form"):
                raw = st.text_area("Decision model (JSON)", _json(status["model"] if status else empty), height=280, key=prefix + "model")
                st.caption("Separate feasibility from action rules. Use feature IDs in predicates and link features to case fact IDs to reevaluate edits. For a numeric threshold, include reviewed values below, at, and above it where representable.")
                if st.form_submit_button("Accept reviewed decision model"):
                    _change(commit, lambda: workflow.save_model(case, item["id"], json.loads(raw), actor),
                            "Accept SME-reviewed decision model")
        if status:
            _scenario_controls(case, item, commit, prefix, actor, notes, status, add_cases)
            if st.button("Confirm manual wording for this scenario", key=prefix + "manual_wording"):
                _change(commit, lambda: workflow.record_human_decision(case, item["id"], actor, notes, "drafting"),
                        "Confirm human-authored wording for the selected scenario")
        if state.get("rationales") and not workflow._matches(state["rationales"], workflow._key_signature(case, item)):
            st.warning("The accepted rationale refers to a different item snapshot. Reassess explanations against the displayed case and key.")
        st.write("Displayed keyed options", [o["text"] for o in item["options"] if o["correct"]])
        if st.button("Confirm displayed option key", key=prefix + "key"):
            _change(commit, lambda: workflow.approve_key(case, item["id"], actor, notes), "Record SME-approved option IDs")
        if st.button("Record human clinical review", key=prefix + "clinical_review"):
            _change(commit, lambda: workflow.record_human_decision(case, item["id"], actor, notes, "clinical_review"),
                    "Record human clinical item review")
        if case["mode"] == "unfolding" and st.button("Record human cueing review", key=prefix + "cueing_review"):
            _change(commit, lambda: workflow.record_human_decision(case, item["id"], actor, notes, "cueing_review"),
                    "Record human sequence cueing review")
        for issue in workflow.review_findings(case, item):
            st.warning(issue)
        with st.expander("Specialist run history"):
            for run in reversed(case.get("specialist_runs", [])):
                if run["target"].get("item_id") != item["id"]:
                    continue
                label = "current" if workflow.proposal_current(case, run) else "historical"
                st.write(f"{run['role']} {run.get('pass_type') or ''} · {run['provider']} {run['model']} · revision {run['target']['base_revision']} · {run['outcome']} · {label}")
                if run.get("authorship"):
                    st.caption(run["authorship"]["label"])
                if run.get("comparison"):
                    st.write(run["comparison"])
                if run.get("response"):
                    st.json(run["response"], expanded=False)


def render_specialists(case, item, enabled, commit):
    if not item:
        return
    prefix = case["id"] + "::specialist:" + item["id"] + ":"
    with st.expander("Family Medicine specialists"):
        try:
            registry = specialist_registry()
        except AdapterError as exc:
            st.warning(str(exc))
            return
        names = {a["role_id"]: a["name"] for a in registry["agents"]}
        st.caption("Each request uses the current item and the authorized context shown below.")
        role = st.selectbox("Specialist", list(names), format_func=names.get, key=prefix + "role")
        pass_type = None
        if role == "item_validation":
            passes = ["solve", "compare"] + (["cueing"] if case["mode"] == "unfolding" else [])
            pass_type = st.selectbox("Review pass", passes, key=prefix + "pass")
        route = specialist_config(role)
        st.caption(f"{route['name']} · {route['model'] or 'Model unset'} · registry {registry['version']}")
        if not route["configured"]:
            st.info(route["reason"])
        if not enabled:
            st.caption("Turn on AI assistance to invoke a specialist. Local decision checks remain available.")
            return
        approved = st.checkbox(f"Approve sending this role’s projected context to {route['name']} ({route['model'] or 'unset'})",
                               key=prefix + f"approve:{role}:{pass_type}:{route['provider']}:{route['model']}")
        try:
            if route["configured"]:
                request = workflow.build_request(case, item, role, route["provider"], pass_type=pass_type)
                if role in {"bank_retrieval", "evidence_review"} and not any(request["role_input"].values()):
                    st.caption("No source passages are authorized for this role and provider.")
                with st.expander("Inspect outgoing context"):
                    st.json(request, expanded=False)
            else:
                request = None
        except (AdapterError, ValueError, KeyError, TypeError) as exc:
            request = None
            st.info(str(exc))
        if st.button("Run selected specialist", disabled=not approved or request is None, key=prefix + "run"):
            response = error = None
            try:
                storage.verify_specialist_revision(case["id"], request["target"]["base_revision"])
            except ValueError as exc:
                st.warning(str(exc))
                return
            try:
                with st.spinner("Waiting for the selected specialist…"):
                    response = dispatch_specialist(role, request, enabled, approved_route=(route["provider"], route["model"]))
            except AdapterError as exc:
                error = str(exc)
            if not st.session_state.get("itemdev_ai_enabled", enabled):
                return
            run = workflow.specialist_run(case, item, request, role, route, response=response, error=error)
            try:
                commit(deepcopy(case) | {"specialist_runs": case.get("specialist_runs", []) + [run]},
                       "Record specialist run", specialist=True)
            except ValueError as exc:
                st.warning(str(exc))
                return
            st.rerun()
        run = next(iter(workflow._runs(case, item, role, pass_type)), None)
        if not run:
            return
        if not workflow.proposal_current(case, run):
            st.caption(f"Source revision {run['target']['base_revision']} · available in history.")
        if run.get("error"):
            st.error(run["error"])
        response = run.get("response")
        if not response:
            return
        st.caption(f"{run['outcome'].replace('_', ' ').title()} · {run['provider']} · revision {run['target']['base_revision']}")
        for issue in response["issues"]:
            {"info": st.info, "review": st.warning, "blocking": st.error}[issue["severity"]](issue["message"])
        if run.get("authorship"):
            st.caption(run["authorship"]["label"])
        if run.get("comparison"):
            st.write("Local comparison · " + run["comparison"]["status"])
        st.json(response["payload"], expanded=False)
        if role == "distractor_modeling" and workflow.proposal_current(case, run):
            status = workflow.model_status(case, item)
            if status:
                pool = dm.option_pool(status["model"], status["row"], response["payload"].get("options", []),
                                      workflow.target_count(item), case=case, item=item)
                st.caption(f"Local option shortfall · {pool['shortage']} for {workflow.target_count(item)} requested.")
                for row in pool["rows"]:
                    if row["review_reason"]:
                        st.caption(row["text"] + " · " + row["review_reason"])
        if role != "item_validation":
            applicable = []
            if role == "evidence_review" and workflow.proposal_current(case, run):
                selected = dm.by_id(item.get("decision_workflow", {}).get("retrieved_sources", []), "source_id")
                claims = {c["claim_id"]: c for c in response["payload"].get("claims", []) if c["source_id"] in selected}
                if claims:
                    for cid, claim in claims.items():
                        source = selected[claim["source_id"]]
                        with st.expander(f"Review claim {cid} against its passage"):
                            st.text(f"{source['source_date']} · {source['locator']}")
                            st.text(source.get("citation") or source.get("url", ""))
                            st.text(source["excerpt"])
                            st.write({k: claim.get(k) for k in ("population", "condition", "exclusions", "timing",
                                                               "action", "applicability", "limitations")})
                    applicable = st.multiselect("Retrieved claims applicable to this decision", list(claims),
                        help="Select the clinical claims you have reviewed for use in decision rules.",
                        key=prefix + "applicable:" + run["request_id"])
            actor = st.text_input("SME accepting the proposal", case.get("author", ""), key=prefix + "actor")
            if st.button("Accept proposal into draft", disabled=run["outcome"] != "proposed" or not workflow.proposal_current(case, run),
                         key=prefix + "accept"):
                _change(commit, lambda: workflow.accept_proposal(case, run, actor, applicable_claim_ids=applicable),
                        "Accept " + names[role].lower() + " proposal")
