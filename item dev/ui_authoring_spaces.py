"""Feature and distractor editors in the selected item's Cognitive model tab."""

import json

import streamlit as st

import authoring_spaces as spaces
import core
import decision_model as dm
import specialist_workflow as workflow


def _save(commit, edit, reason):
    try:
        candidate = edit()
        commit(candidate, reason, specialist=True)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
    else:
        st.rerun()


def _list_field(label, choices, current, key):
    return st.multiselect(label, list(choices), default=[k for k in current if k in choices],
                          format_func=choices.get, key=key)


def _show_sources(case, item, source_ids, key, recorded=None):
    if not source_ids:
        return
    for sid in dict.fromkeys(source_ids):
        source = workflow.reviewed_source(case, item, sid)
        snapshot = (recorded or {}).get(sid)
        with st.expander(f"Inspect source · {sid}"):
            if snapshot and (not source or snapshot != workflow._source_reference(source)):
                st.caption("Recorded with the reviewed revision")
                st.text(" · ".join(snapshot.get(k, "") for k in ("source_class", "source_date", "locator")))
                if snapshot.get("citation"):
                    st.text(snapshot["citation"])
                if snapshot.get("url", "").startswith(("https://", "http://")):
                    st.link_button("Open recorded citation", snapshot["url"], key=f"{key}:recorded:{sid}")
            if source:
                st.text(f"{source['source_class']} · {source['source_date']} · {source['locator']}")
                st.text(source["excerpt"])
                if source.get("citation"):
                    st.text(source["citation"])
                if source.get("url", "").startswith(("https://", "http://")):
                    st.link_button("Open cited source", source["url"], key=f"{key}:link:{sid}")
            else:
                st.caption("This passage is outside the current selection or scope.")


def _constraints(ctx):
    features = {f["feature_id"]: f["label"] for f in ctx["model"]["features"]}
    for constraint in ctx["model"]["feasibility_constraints"]:
        st.write(f"**{constraint['constraint_id']}** · {constraint['rationale']}")
        st.caption(spaces.predicate_text(constraint["condition"], features))


def _feature_editor(case, item, ctx, actor, prefix, commit):
    features = dm.by_id(ctx["model"]["features"], "feature_id")
    rows = spaces.feature_rows(case, item, ctx)
    st.dataframe(list(rows.values()), width="stretch", hide_index=True)
    _constraints(ctx)
    actions = dm.by_id(ctx["model"]["actions"], "action_id")
    st.markdown("**Feasible modeled scenarios at this disclosure**")
    st.dataframe([
        {"Scenario": sid, "Outcome": r["outcome"],
         "Key": ", ".join(actions[a]["label"] for a in r["keyed_action_ids"]),
         "Missing findings": ", ".join(r["missing_features"]),
         "Competing actions": ", ".join(actions[a]["label"] for a in r["competing_action_ids"])}
        for sid, r in ctx["rows"].items() if r["outcome"] != "infeasible"
    ], width="stretch", hide_index=True)
    fid = st.selectbox("Feature to inspect or edit", list(features), format_func=lambda f: features[f]["label"],
                       key=prefix + "feature")
    feature = features[fid]
    record = workflow._state(item)["model"]
    st.caption(f"{feature.get('origin', 'SME-authored')} · model accepted by {record.get('actor', 'Unknown reviewer')} from revision {record.get('accepted_from_revision', 'unknown')}")
    all_sources = {s["source_id"]: s.get("title") or s["source_id"] for s in workflow._sources(case, item)}
    linked_sources = record.get("feature_sources", {}).get(fid, [])
    for sid in linked_sources:
        all_sources.setdefault(sid, sid + " (recorded source)")
    claims = spaces.claim_sources(item, "model")
    informed = [r for r in ctx["model"]["decision_rules"] + ctx["model"]["feasibility_constraints"]
                if fid in spaces._used_features(r)]
    source_ids = list(linked_sources) + [claims[c] for r in informed for c in r["evidence_claim_ids"] if c in claims]
    _show_sources(case, item, source_ids, prefix + "feature", record.get("source_links"))
    fp = prefix + "edit_feature:" + fid + ":"
    with st.form(fp):
        label = st.text_input("Feature label", feature["label"], key=fp + "label")
        allowed = st.text_area("Permitted feature values", "\n".join(json.dumps(v, ensure_ascii=False) for v in feature["allowed_values"]),
                               key=fp + "allowed", help='One value per line. Use numbers or true/false for typed values. Quote text that looks like a number.')
        unit = st.text_input("Measurement unit", feature.get("unit") or "", key=fp + "unit")
        variation = st.selectbox("Variation", ["fixed", "variable"], index=["fixed", "variable"].index(feature["variation"]), key=fp + "variation")
        classes = ["decisive", "supporting", "contextual"]
        impact = st.selectbox("Decision impact", classes, index=classes.index(feature["classification"]), key=fp + "impact")
        stages = {"shared": "Shared", **{s["id"]: s["title"] for s in case["stages"]}}
        disclosure = st.selectbox("Feature disclosure", list(stages), format_func=stages.get,
                                  index=list(stages).index(feature["disclosure_stage_id"]) if feature["disclosure_stage_id"] in stages else None,
                                  key=fp + "stage")
        fact_ids = _list_field("Linked case facts", {f["id"]: f.get("name") or f["id"] for f in case["facts"]},
                               feature.get("fact_ids", []), fp + "facts")
        source_ids = _list_field("Source records informing this feature", all_sources, linked_sources, fp + "sources")
        if st.form_submit_button("Save reviewed feature"):
            def edit():
                values = []
                for line in allowed.splitlines():
                    if line.strip():
                        try:
                            value = json.loads(line)
                        except ValueError:
                            value = line.strip()
                        values.append(value)
                return spaces.edit_feature(case, item["id"], fid, {"label": label.strip(), "allowed_values": values,
                    "unit": unit.strip() or None, "variation": variation, "classification": impact,
                    "fact_ids": fact_ids, "disclosure_stage_id": disclosure}, source_ids, actor)
            _save(commit, edit, "Edit reviewed decision feature")


def _distractor_editor(case, item, ctx, actor, prefix, commit):
    rows = spaces.candidate_rows(case, item, ctx)
    eligible = sum(r["local_judgment"] in {"eligible", "keyed"} for r in rows)
    target = workflow.target_count(item)
    st.write(f"Requested options **{target}** · Distinct eligible or keyed **{eligible}** · Shortfall **{max(0, target - eligible)}**")
    actions = {a["action_id"]: a["label"] for a in ctx["model"]["actions"]}
    features = {f["feature_id"]: f["label"] for f in ctx["model"]["features"]}
    facts = {f["id"]: f.get("name") or f["id"] for f in core.visible_facts(case, item["stage_id"])}
    claims = dm.by_id(workflow._claims(case, item), "claim_id")
    state = workflow._state(item)
    claim_refs = spaces.claim_sources(item, "options")
    def review_record(row):
        record = state.get("candidate_reviews", {}).get(row["candidate_id"], {})
        return record if record.get("run_id") == spaces._pool_run(item) else {}

    def sources(row):
        review = review_record(row)
        links = review.get("source_links", {})
        refs = claim_refs | review.get("claim_sources", {})
        return list(dict.fromkeys([*row.get("bank_item_ids", []), *row.get("misconception_evidence_ids", []),
            *(refs[c] for c in row["evidence_claim_ids"] if c in refs), *links]))
    st.dataframe([
        {"Candidate": r["text"], "Action": actions.get(r["action_id"], "Unmodeled"), "Decision": r["decision"],
         "Current eligibility": r["local_judgment"], "Appeal": r["plausibility_rationale"] or "",
         "Clinical judgment": r["clinical_rationale"], "Resolution needed": r["review_reason"],
         "Where keyed": ", ".join(r["neighbors"]), "Excluding rules": ", ".join(rule["rule_id"] for rule in r["exclusions"]),
         "Attractive findings": ", ".join(facts.get(f, f) for f in r["attractive_fact_ids"]),
         "Misconception": r.get("misconception_hypothesis") or "", "Evidence status": r.get("evidence_status") or "",
         "Source IDs": ", ".join(sources(r)), "Reviewer": r["actor"]}
        for r in rows
    ], width="stretch", hide_index=True)
    if not rows:
        return
    by_id = dm.by_id(rows, "candidate_id")
    cid = st.selectbox("Candidate to review", list(by_id), format_func=lambda k: by_id[k]["text"] or k, key=prefix + "candidate")
    row = by_id[cid]
    if row["decision"] == "Reassess":
        st.warning("This candidate's prior judgment requires review for the displayed facts and scenario.")
    for rule in row["exclusions"]:
        condition = [rule.get("scope_condition"), rule["condition"]] if rule.get("scope_condition") else rule["condition"]
        st.write(f"**{rule['rule_id']}** · {spaces.predicate_text(condition, features)}")
        for fid in sorted(spaces._used_features(rule) & ctx["visible"]):
            st.caption(f"{features[fid]} · {spaces.display_value(ctx['row']['visible_values'].get(fid))}")
    recorded = state.get("options", {}).get("source_links", {}) | review_record(row).get("source_links", {})
    _show_sources(case, item, sources(row), prefix + "candidate", recorded)
    fp = prefix + "edit_candidate:" + cid + ":"
    with st.form(fp):
        wording = st.text_area("Candidate wording", row["text"], key=fp + "text")
        act = {"": "Unmodeled idea"} | actions
        selected_action = st.selectbox("Candidate action", list(act), format_func=act.get,
                                       index=list(act).index(row["action_id"]) if row["action_id"] in act else 0, key=fp + "action")
        roles = ["essential", "conditional", "random"]
        role = st.selectbox("Option role", roles, index=roles.index(row["role"]), key=fp + "role")
        appeal = st.text_area("Why would an examinee choose it?", row["plausibility_rationale"] or "", key=fp + "appeal")
        clinical = st.text_area("Why is it keyed, inferior, or unresolved here?", row["clinical_rationale"], key=fp + "judgment")
        attractive = _list_field("Findings that make it attractive", facts, row["attractive_fact_ids"], fp + "facts")
        rules = {r["rule_id"]: r["rule_id"] for r in ctx["model"]["decision_rules"]}
        exclusions = _list_field("Rules explaining exclusion", rules, row["violated_rule_ids"], fp + "rules")
        references = _list_field("Accepted evidence claims", {k: k + " · " + c["action"] for k, c in claims.items()},
                                 row["evidence_claim_ids"], fp + "claims")
        bank = {s["source_id"]: s.get("title") or s["source_id"] for s in workflow._sources(case, item)
                if s["source_class"] == "bank"}
        patterns = _list_field("Approved bank patterns", bank, row.get("bank_item_ids", []), fp + "bank")
        hypothesis = st.text_input("Reasoning-error hypothesis", row.get("misconception_hypothesis") or "", key=fp + "hypothesis")
        decisions = {"Hold": "Retain for review", "Eligible": "Eligible for this case", "Selected": "Select for assembly"}
        decision = st.selectbox("SME disposition", list(decisions), format_func=decisions.get,
                                index=list(decisions).index(row["decision"]) if row["decision"] in decisions else 0,
                                key=fp + "decision")
        if st.form_submit_button("Save candidate decision", disabled=ctx["trial"]):
            _save(commit, lambda: spaces.review_candidate(case, item["id"], cid, {
                "action_id": selected_action or None, "text": wording, "role": role, "clinical_rationale": clinical.strip(),
                "plausibility_rationale": appeal.strip() or None, "attractive_fact_ids": attractive,
                "violated_rule_ids": exclusions, "evidence_claim_ids": references, "bank_item_ids": patterns,
                "misconception_hypothesis": hypothesis.strip() or None,
            }, decision, actor, ctx["base_id"]), "Record candidate eligibility and assembly decision")
    if st.button("Assemble selected candidates", key=prefix + "assemble", disabled=not ctx["confirmed"]):
        _save(commit, lambda: spaces.assemble_reviewed(case, item["id"], ctx["base_id"], actor),
              "Assemble reviewed candidate options")


def render_authoring_spaces(case, item, status, actor, notes, prefix, commit, *, scenario_id=None):
    ctx = spaces.space_context(case, item, scenario_id)
    prefix += "spaces:" + ctx["base_id"] + ":"
    stage = next(s["title"] for s in case["stages"] if s["id"] == item["stage_id"])
    st.caption(f"Disclosure · {stage}   |   Scenario · {ctx['base_id']}")
    changes = {}
    with st.expander("Try a change in case facts"):
        for feature in status["model"]["features"]:
            fid = feature["feature_id"]
            if feature["variation"] != "variable" or fid not in ctx["visible"]:
                continue
            original = ctx["values"].get(fid)
            allowed = feature["allowed_values"]
            choices = [original] + [v for v in allowed if not dm.includes([original], v)]
            selected = st.selectbox(feature["label"], list(range(len(choices))), key=prefix + "trial:" + fid + ":" + str(case["version"]),
                                    format_func=lambda index, vals=choices: spaces.display_value(vals[index]))
            if selected:
                changes[fid] = choices[selected]
        if changes:
            ctx = spaces.space_context(case, item, scenario_id, changes)
            verdict = ctx["row"]
            labels = {a["action_id"]: a["label"] for a in ctx["model"]["actions"]}
            st.write(f"**Trial outcome** · {verdict['outcome']}")
            if verdict["keyed_action_ids"]:
                st.write("Trial key · " + ", ".join(labels[k] for k in verdict["keyed_action_ids"]))
            for reason in verdict["reasons"]:
                st.warning(reason)
            if st.button("Confirm trial scenario and key", key=prefix + "confirm_trial",
                         disabled=verdict["outcome"] not in dm.RESOLVED or not status["scope_current"]):
                _save(commit, lambda: spaces.confirm_trial(case, item["id"], ctx["base_id"], changes, actor, notes),
                      "Confirm previewed feature combination and scenario")
    labels = {a["action_id"]: a["label"] for a in ctx["model"]["actions"]}
    row = ctx["row"]
    st.write(f"**Evaluated key** · " + (", ".join(labels[a] for a in row["keyed_action_ids"]) or row["outcome"]))
    if row["missing_features"]:
        st.write("Missing findings · " + ", ".join(row["missing_features"]))
    if row["competing_action_ids"]:
        st.write("Competing actions · " + ", ".join(labels[a] for a in row["competing_action_ids"]))
    feature_tab, distractor_tab = st.tabs(["Feature space", "Distractor space"])
    with feature_tab:
        _feature_editor(case, item, ctx, actor, prefix, commit)
    with distractor_tab:
        _distractor_editor(case, item, ctx, actor, prefix, commit)
