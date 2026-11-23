"""Structured-output model adapters.

``OpenAIStructuredLLM`` calls the Responses API with a strict JSON schema.
``MockStructuredLLM`` produces deterministic, clearly labelled heuristic output
for offline demonstrations and tests; it is not an AI or SME judgement.
"""

import json
import re

from .embedding import ConsentError, call_with_retries, safe_error
from .schemas import SCHEMA_VERSION

MOCK_NOTE = "Mock heuristic output for offline demonstration — not an AI or SME judgement."


class LLMError(RuntimeError):
    pass


class OpenAIStructuredLLM:
    external = True

    def __init__(self, api_key, model, consent, client=None, max_output_tokens=16000):
        if not consent:
            raise ConsentError("Confirm consent to send the target and confirmed items to OpenAI first.")
        if not model:
            raise LLMError("Enter the OpenAI model ID to use for structured analysis.")
        if not api_key and client is None:
            raise LLMError("Provide an OpenAI API key.")
        self._api_key = api_key
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, max_retries=0, timeout=180)
        return self._client

    def complete(self, system, payload, schema, name, sleep=None):
        def call():
            return self.client.responses.create(
                model=self.model,
                instructions=system,
                input=json.dumps(payload, ensure_ascii=False),
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                store=False,
                max_output_tokens=self.max_output_tokens,
            )
        last = None
        for _attempt in range(2):  # one re-request on malformed output
            try:
                response = call_with_retries(call, **({"sleep": sleep} if sleep else {}))
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"The model request failed: {safe_error(exc)}.") from None
            status = getattr(response, "status", "completed")
            if status != "completed":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None) or status
                raise LLMError(f"The model response was incomplete ({reason}). Reduce the number of items or retry.")
            for item in getattr(response, "output", []) or []:
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "refusal":
                        raise LLMError("The model declined the request. No changes were applied.")
            try:
                data = json.loads(response.output_text)
                if not isinstance(data, dict):
                    raise ValueError
            except (ValueError, TypeError, AttributeError) as exc:
                last = exc
                continue
            usage = getattr(response, "usage", None)
            tokens = {"input_tokens": getattr(usage, "input_tokens", 0) or 0,
                      "output_tokens": getattr(usage, "output_tokens", 0) or 0}
            return data, tokens
        raise LLMError("The model returned output that is not valid JSON. No changes were applied.") from last


# --------------------------------------------------------------------------- mock

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text or "") if s.strip()]


def lead_in(item):
    """(field, sentence) of the item's lead-in: the last question sentence, else the last sentence."""
    fields = list(item["fields"].items())
    for name, text in reversed(fields):
        for sentence in reversed(_sentences(text)):
            if sentence.endswith("?"):
                return name, sentence
    name, text = fields[-1]
    return name, (_sentences(text) or [text])[-1]


_DECISIONS = [
    (r"\b(first|priority|prioriti[sz]e|initial|immediate(ly)?)\b", "Prioritize the first action", "Select the first or highest-priority action"),
    (r"\bnext\b", "Select the next action", "Choose the next step in a sequence"),
    (r"\b(teach|teaching|instruct|education|understanding)\b", "Evaluate patient teaching", "Judge or select a teaching response"),
    (r"\b(indicate|suggest|most likely|interpret|expect)\b", "Interpret clinical findings", "Interpret findings to identify their meaning"),
]


def decision_category(sentence):
    for pattern, name, action in _DECISIONS:
        if re.search(pattern, sentence, re.I):
            return name, action
    return "Select the best response", "Select the best answer to the lead-in"


_VALUE = re.compile(
    r"\b\d+(?:\.\d+)?(?:/\d+)?\s?(?:mmHg|mg/dL|mmol/L|mEq/L|mg|mcg|mL|kg|%|°[CF]|bpm|breaths/min|units?|hours?|days?|weeks?)"
    r"|\b\d+-(?:year|month|week|day)-old\b",
    re.I,
)


def mock_task(payload):
    target = payload["target"]
    field, sentence = lead_in(target)
    name, action = decision_category(sentence)
    domain = target.get("metadata", {}).get("domain") or "Not specified in metadata"
    alignments = []
    for cand in payload.get("candidates", []):
        c_field, c_sentence = lead_in(cand)
        c_name, _ = decision_category(c_sentence)
        same_domain = (cand.get("metadata", {}).get("domain") or domain) == domain
        proposal = "same_task" if c_name == name and same_domain else "uncertain" if c_name == name else "different_task"
        alignments.append({
            "item_id": cand["item_id"], "apparent_task": c_name, "proposed_alignment": proposal,
            "evidence": f"Lead-in ({c_field}): \"{c_sentence}\"", "confidence_status": "uncertain",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "target_item_id": target["item_id"],
        "target_task": {
            "name": name, "domain": str(domain), "decision_or_action": action,
            "rationale": f"{MOCK_NOTE} Derived from the wording of the lead-in.",
            "source_item_ids": [target["item_id"]],
            "evidence_spans": [{"item_id": target["item_id"], "field": field, "option_id": None,
                                "excerpt": sentence, "evidence_type": "explicit_text"}],
            "ambiguity_flags": [] if name != "Select the best response" else ["Lead-in does not name a specific decision."],
        },
        "candidate_task_alignment": alignments,
    }


def _feature(fid, label, ftype, items, spans, definition, **extra):
    base = {
        "feature_id": fid, "canonical_label": label, "definition": definition, "feature_type": ftype,
        "task_relevance": "", "dominance": None, "procedure_group": None, "sequence_order": None,
        "supporting_items": items, "source_spans": spans, "inference_rationale": "",
        "proposed_misconception": None, "llm_uncertainty_note": MOCK_NOTE,
    }
    base.update(extra)
    return base


def mock_decompose(payload):
    task = payload.get("confirmed_task", {})
    items = [payload["target"]] + list(payload.get("siblings", []))
    features = []
    n = 0

    def next_id():
        nonlocal n
        n += 1
        return f"M{n:03d}"

    for item in items:
        iid = item["item_id"]
        field, sentence = lead_in(item)
        name, action = decision_category(sentence)
        features.append(_feature(
            next_id(), name, "procedural_key", [iid],
            [{"item_id": iid, "field": field, "option_id": None, "excerpt": sentence, "evidence_type": "explicit_text"}],
            action, dominance="dominant", procedure_group="lead-in decision", sequence_order=1,
            task_relevance=f"Required by the confirmed task: {task.get('name', '')}",
        ))
        domain = item.get("metadata", {}).get("domain")
        if domain:
            first_field = next(iter(item["fields"]))
            features.append(_feature(
                next_id(), f"Core knowledge: {domain}", "declarative_key", [iid],
                [{"item_id": iid, "field": first_field, "option_id": None, "excerpt": "", "evidence_type": "clinically_inferred"}],
                f"Domain knowledge needed to answer the item ({domain}).",
                inference_rationale="Inferred from the item's mapped domain; not stated as a fact in the text.",
            ))
        keyed = [o for o in item["options"] if o["is_key"]]
        for o in keyed:
            features.append(_feature(
                next_id(), f"Knows the keyed response: {o['text']}", "declarative_key", [iid],
                [{"item_id": iid, "field": "option", "option_id": o["option_id"], "excerpt": o["text"], "evidence_type": "explicit_text"}],
                "Knowledge expressed by the keyed option.",
            ))
        for fname, text in item["fields"].items():
            for m in _VALUE.finditer(text):
                features.append(_feature(
                    next_id(), f"Presentation value: {m.group(0)}", "instance_context", [iid],
                    [{"item_id": iid, "field": fname, "option_id": None, "excerpt": m.group(0), "evidence_type": "explicit_text"}],
                    "A scenario-specific value in the presentation.",
                ))
        if item.get("key_status") in ("single", "multiple"):
            for o in item["options"]:
                if o["is_key"] is False:
                    features.append(_feature(
                        next_id(), f"Selects alternative: {o['text']}", "distractor_feature", [iid],
                        [{"item_id": iid, "field": "option", "option_id": o["option_id"], "excerpt": o["text"], "evidence_type": "explicit_text"}],
                        "An incorrect option; the reasoning error behind it is a hypothesis.",
                        proposed_misconception="Hypothesis only — requires SME verification.",
                    ))
    return {"schema_version": SCHEMA_VERSION, "target_item_id": payload["target"]["item_id"], "features": features}


class MockStructuredLLM:
    external = False
    model = "mock-heuristic-v1"

    def complete(self, system, payload, schema, name, sleep=None):
        if name == "target_task":
            return mock_task(payload), {"input_tokens": 0, "output_tokens": 0}
        if name == "feature_decomposition":
            return mock_decompose(payload), {"input_tokens": 0, "output_tokens": 0}
        raise LLMError(f"Unknown structured task '{name}'.")
