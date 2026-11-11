"""Optional assistant panel over the same reversible authoring action layer."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Callable
from uuid import uuid4

import streamlit as st

from ai_adapter import (
    AdapterError,
    get_config,
    parse_command,
    propose_item,
    request_assistance,
    transcribe_audio,
)
from core import (
    APPRAISAL_GUIDANCE,
    MISCONCEPTIONS,
    REASONING_GUIDANCE,
    new_item,
    resize_items,
    resize_stages,
    touch_case,
    validate_case,
)


def _selected_item(case: dict) -> dict | None:
    selected_id = st.session_state.get(case["id"] + "::selected_item_id")
    return next((item for item in case.get("items", []) if item["id"] == selected_id), None) or next(iter(case.get("items", [])), None)


def _apply_local(case: dict, command: str, commit: Callable) -> None:
    """Dispatch a known local command; never send unknown text to a provider."""
    action = parse_command(command)
    if action is None:
        st.info("This is not an exact local command. Try ‘add another question’, ‘create two items’, ‘set this item to multiple-selection’, or ‘undo’. No change was made.")
        return
    kind = action["action"]
    if kind in {"undo", "redo", "open_mode", "preview"}:
        st.session_state["itemdev_requested_action"] = action | {"case_id": case["id"]}
        st.rerun()
    if kind == "validate":
        findings = validate_case(case)
        if not findings:
            st.success("No structural findings. Clinical correctness still requires SME review.")
        else:
            for finding in findings[:12]:
                message = f"{finding['path']}: {finding['message']}"
                if finding["severity"] == "error":
                    st.error(message)
                elif finding["severity"] == "warning":
                    st.warning(message)
                else:
                    st.info(message)
        return
    candidate = deepcopy(case)
    item = _selected_item(candidate)
    if kind == "add_item":
        if not candidate.get("stages"):
            st.warning("Create a disclosure stage first.")
            return
        stage_id = item["stage_id"] if item else candidate["stages"][0]["id"]
        created = new_item(stage_id)
        candidate["items"].append(created)
        touch_case(candidate, "Added an item using a local command")
        reason = "Added an item without changing the shared vignette"
    elif kind == "add_stage":
        resize_stages(candidate, len(candidate.get("stages", [])) + 1)
        reason = "Added a disclosure stage using a local command"
    elif kind == "resize_items":
        resize_items(candidate, action["count"])
        reason = f"Set case to {action['count']} items; removed content remains recoverable"
    elif kind == "set_format":
        if not item:
            st.warning("Select an item before changing its format.")
            return
        item["format"] = action["format"]
        item["select_count"] = None
        item["selection_instruction"] = "Select the single best answer." if action["format"] == "SBA" else "Select all applicable options."
        touch_case(candidate, "Response format changed; key and scoring need review")
        reason = "Changed response format and retained options for explicit key review"
    else:
        st.info("This command is not available. No change was made.")
        return
    commit(candidate, reason)
    st.rerun()


def _offline_guidance(case: dict, item: dict | None) -> None:
    with st.expander("Reasoning and misconception guidance", expanded=False):
        if item:
            phase = item.get("phase", "")
            st.markdown(f"**Current reasoning phase:** {phase or 'Not yet selected'}")
            st.write(REASONING_GUIDANCE.get(phase, "Choose the clinical judgment this item should assess."))
            if case.get("mode") == "appraisal":
                dimension = item.get("dimension", "")
                st.markdown(f"**Appraisal focus:** {dimension or 'Not yet selected'}")
                st.write(APPRAISAL_GUIDANCE.get(dimension, "Link each judgment to the displayed recommendation and clinical evidence."))
            if case.get("mode") == "unfolding":
                st.caption("Ask what information is visible now, what changed, and whether the expected judgment should update. Future evidence cannot justify an earlier key.")
        else:
            st.write("Create or select an item to link its reasoning phase, appraisal dimension, and proposed misconceptions.")
        misconception = st.selectbox("Explore a proposed reasoning error", list(MISCONCEPTIONS), key=case["id"] + "::ai:misconception_guide")
        st.write(MISCONCEPTIONS[misconception])
        st.caption("This is authoring guidance. A wrong response alone does not prove a learner's misconception or cognitive bias.")


def _remember_request(prefix: str) -> str:
    token = uuid4().hex
    st.session_state[prefix + "request_token"] = token
    return token


def _request_still_current(prefix: str, token: str) -> bool:
    return bool(st.session_state.get(prefix + "enabled")) and st.session_state.get(prefix + "request_token") == token


def render_assistant(case: dict, enabled: bool, commit: Callable) -> None:
    """Render manual help, deterministic commands, and optional AI proposals.

    ``commit(candidate, reason)`` must persist the supplied copy through the main
    app's undoable action layer. Deferred navigation/undo events are placed in
    ``itemdev_requested_action`` for the root to consume before creating widgets.
    """
    prefix = case["id"] + "::ai:"
    st.session_state[prefix + "enabled"] = enabled is True
    item = _selected_item(case)
    config = get_config()

    st.subheader("Authoring support")
    st.caption("Shared manual guidance and optional AI. All edits use the same saved case.")
    _offline_guidance(case, item)

    with st.expander("Local typed commands · no AI connection", expanded=False):
        st.caption("Exact commands work with AI off. Commands use the selected item; they do not invent patient facts or answer keys.")
        command = st.text_input("Local command", placeholder="Add another question", key=prefix + "local_command")
        if st.button("Run local command", key=prefix + "run_command", width="stretch"):
            try:
                _apply_local(case, command, commit)
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"The local command could not be applied: {exc}")
        st.caption("Supported: add item, add stage, create N items, change this item to SBA/MS, validate, preview, undo, redo, open a mode.")

    if not enabled:
        st.session_state[prefix + "request_token"] = uuid4().hex
        for key in ("proposal", "proposal_base", "proposal_item_id"):
            st.session_state.pop(prefix + key, None)
        st.info("AI is off. Manual guidance, deterministic commands, editing, validation, and saving remain available. No AI request is sent.")
        return

    if config.get("error"):
        st.warning(config["error"])
    if config["configured"]:
        st.caption(f"Configured provider: {config['provider']} · model: {config['model']}. Configuration is not a connection test.")
    else:
        st.info("AI is enabled but a provider/model is not configured. Continue manually; no live AI connection is simulated.")

    if item:
        st.caption("Target item: " + (item.get("title") or "Untitled clinical decision"))
    else:
        st.caption("Create an item to request a structured proposal.")
    request = st.text_area("Ask or request a revision", placeholder="Explain which findings support this key, or suggest a clearer lead-in.", key=prefix + "request", height=110)
    action = st.selectbox("Request type", ["Question or explanation", "Review case", "Suggest distractors", "Appraise recommendation"], key=prefix + "task")
    task = {"Question or explanation": "chat", "Review case": "review", "Suggest distractors": "distractors", "Appraise recommendation": "appraisal"}[action]
    ask, draft = st.columns(2)
    if ask.button("Ask AI", key=prefix + "ask", disabled=not config["configured"], width="stretch"):
        token = _remember_request(prefix)
        try:
            with st.spinner("Requesting authoring assistance…"):
                answer = request_assistance(case, request, enabled, task=task)
            if _request_still_current(prefix, token):
                st.session_state[prefix + "answer"] = answer
            else:
                st.info("The request was superseded or AI was switched off; its result was not retained.")
        except AdapterError as exc:
            st.error(str(exc))
    if draft.button("Propose item", key=prefix + "propose", disabled=not config["configured"] or item is None, width="stretch"):
        token = _remember_request(prefix)
        try:
            with st.spinner("Preparing a structured draft…"):
                proposal = propose_item(case, item, request, enabled)
            if _request_still_current(prefix, token):
                st.session_state[prefix + "proposal"] = proposal
                st.session_state[prefix + "proposal_base"] = json.dumps(case, sort_keys=True, ensure_ascii=False)
                st.session_state[prefix + "proposal_item_id"] = item["id"]
            else:
                st.info("The request was superseded or AI was switched off; its proposal was not retained.")
        except AdapterError as exc:
            st.error(str(exc))

    if st.session_state.get(prefix + "answer"):
        with st.expander("AI draft response · requires SME judgment", expanded=True):
            st.markdown(st.session_state[prefix + "answer"])
            st.caption("This is a draft suggestion, not independent clinical validation or approval.")

    proposal = st.session_state.get(prefix + "proposal")
    if proposal:
        with st.expander("Review the proposed item before applying", expanded=True):
            st.markdown("**" + proposal.get("title", "Proposed item") + "**")
            st.write(proposal.get("stem", ""))
            st.write(proposal.get("lead_in", ""))
            st.caption(f"Response format: {proposal.get('format')} · {proposal.get('selection_instruction', '')}")
            if proposal.get("recommendation"):
                st.markdown("**Preliminary recommendation**")
                st.write(proposal["recommendation"])
            for number, option in enumerate(proposal.get("options", []), 1):
                st.markdown(f"**{number}. {option['text']}** {'— KEY' if option.get('correct') else ''}")
                st.write(option.get("rationale", ""))
                if option.get("misconception"):
                    st.caption("Proposed reasoning error: " + option["misconception"])
            current = json.dumps(case, sort_keys=True, ensure_ascii=False)
            stale = current != st.session_state.get(prefix + "proposal_base")
            if stale:
                st.warning("The case changed after this proposal was generated. Generate a fresh proposal against the current evidence.")
            if st.button("Apply candidate to draft", key=prefix + "apply", disabled=stale, width="stretch"):
                candidate = deepcopy(case)
                target_id = st.session_state[prefix + "proposal_item_id"]
                index = next((i for i, value in enumerate(candidate["items"]) if value["id"] == target_id), None)
                if index is None:
                    st.warning("The proposed target no longer exists. Generate a fresh proposal.")
                else:
                    candidate["items"][index] = deepcopy(proposal)
                    touch_case(candidate, "Applied an AI-proposed item; clinical and option review required")
                    commit(candidate, "Applied AI-proposed draft after author review")
                    st.session_state.pop(prefix + "proposal", None)
                    st.rerun()

    if st.button("Discard AI drafts", key=prefix + "discard", width="stretch"):
        st.session_state[prefix + "request_token"] = uuid4().hex
        for key in ("answer", "proposal", "proposal_base", "proposal_item_id"):
            st.session_state.pop(prefix + key, None)
        st.rerun()

    with st.expander("Recorded voice message", expanded=False):
        if not config["voice_configured"]:
            st.info("Voice transcription is not configured. Text and manual editing remain available.")
        else:
            st.caption("Record a message, transcribe it explicitly, then review the text before using it. This is recorded input, not a live hands-free conversation.")
            if hasattr(st, "audio_input"):
                recording = st.audio_input("Record a voice message", key=prefix + "audio")
            else:
                recording = None
                st.info("This Streamlit version does not provide a microphone recorder. Use text or update the app environment.")
            if st.button("Transcribe recording", key=prefix + "transcribe", disabled=recording is None, width="stretch"):
                token = _remember_request(prefix)
                try:
                    with st.spinner("Transcribing recorded message…"):
                        transcript = transcribe_audio(recording.getvalue(), enabled)
                    if _request_still_current(prefix, token):
                        st.session_state[prefix + "transcript"] = transcript
                except AdapterError as exc:
                    st.error(str(exc))
            transcript = st.text_area("Review or edit the transcript", key=prefix + "transcript", height=100)
            use = st.radio("Use transcript as", ["Local command", "AI question", "Append to item stem"], key=prefix + "transcript_use")
            if st.button("Use reviewed transcript", key=prefix + "use_transcript", disabled=not transcript.strip(), width="stretch"):
                if use == "Local command":
                    _apply_local(case, transcript, commit)
                elif use == "AI question":
                    if not config["configured"]:
                        st.warning("Configure a chat model before submitting an AI question. The transcript is preserved.")
                    else:
                        token = _remember_request(prefix)
                        try:
                            with st.spinner("Requesting authoring assistance…"):
                                answer = request_assistance(case, transcript, enabled)
                            if _request_still_current(prefix, token):
                                st.session_state[prefix + "answer"] = answer
                                st.rerun()
                        except AdapterError as exc:
                            st.error(str(exc))
                elif item:
                    candidate = deepcopy(case)
                    target = next(value for value in candidate["items"] if value["id"] == item["id"])
                    target["stem"] = (target.get("stem", "").rstrip() + "\n\n" + transcript.strip()).lstrip()
                    touch_case(candidate, "Appended reviewed voice dictation to an item stem")
                    commit(candidate, "Appended reviewed voice dictation to selected item")
                    st.rerun()
                else:
                    st.warning("Select an item before appending dictation.")
