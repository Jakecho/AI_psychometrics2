"""Shared vocabularies and the versioned JSON schemas for structured model output.

The schemas use the subset accepted by strict structured output: every object
lists all of its properties as required, nullable values use a type union, and
``additionalProperties`` is false. Application-side validation in
``task_alignment`` and ``decomposition`` performs the checks a schema cannot
express (existing item/option IDs, exact excerpts, key status).
"""

SCHEMA_VERSION = "1.0"

FEATURE_TYPES = ("declarative_key", "procedural_key", "instance_context", "distractor_feature")
FEATURE_TYPE_LABELS = {
    "declarative_key": "Declarative",
    "procedural_key": "Procedural",
    "instance_context": "Instance",
    "distractor_feature": "Distractor",
}
FEATURE_PREFIX = {"declarative_key": "D", "procedural_key": "P", "instance_context": "I", "distractor_feature": "X"}
DOMINANCE = ("dominant", "alternative", "contingent", "unknown")
EVIDENCE_TYPES = ("explicit_text", "clinically_inferred")
ALIGNMENTS = ("same_task", "different_task", "uncertain")
SME_ALIGNMENTS = ("unreviewed",) + ALIGNMENTS
TASK_STATUSES = ("unreviewed", "accepted", "edited", "rejected")
REVIEW_STATUSES = (
    "pending", "accepted", "edited", "rejected", "not_required",
    "context_only", "ambiguous", "unsupported", "needs_adjudication", "merged",
)
# Statuses that count toward SME-approved incidence.
APPROVED_STATUSES = ("accepted", "edited")
IMPORTANCE_ANCHORS = {1: "1 · peripheral", 2: "2 · low", 3: "3 · moderate", 4: "4 · high", 5: "5 · essential"}

_NULLABLE_STR = {"type": ["string", "null"]}

_SPAN = {
    "type": "object",
    "additionalProperties": False,
    "required": ["item_id", "field", "option_id", "excerpt", "evidence_type"],
    "properties": {
        "item_id": {"type": "string"},
        "field": {"type": "string", "description": "A field name from the item payload, or 'option'."},
        "option_id": _NULLABLE_STR,
        "excerpt": {"type": "string", "description": "Exact substring for explicit_text; may be empty for clinically_inferred."},
        "evidence_type": {"type": "string", "enum": list(EVIDENCE_TYPES)},
    },
}

TASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "target_item_id", "target_task", "candidate_task_alignment"],
    "properties": {
        "schema_version": {"type": "string"},
        "target_item_id": {"type": "string"},
        "target_task": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "domain", "decision_or_action", "rationale", "source_item_ids",
                         "evidence_spans", "ambiguity_flags"],
            "properties": {
                "name": {"type": "string"},
                "domain": {"type": "string"},
                "decision_or_action": {"type": "string"},
                "rationale": {"type": "string"},
                "source_item_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_spans": {"type": "array", "items": _SPAN},
                "ambiguity_flags": {"type": "array", "items": {"type": "string"}},
            },
        },
        "candidate_task_alignment": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["item_id", "apparent_task", "proposed_alignment", "evidence", "confidence_status"],
                "properties": {
                    "item_id": {"type": "string"},
                    "apparent_task": {"type": "string"},
                    "proposed_alignment": {"type": "string", "enum": list(ALIGNMENTS)},
                    "evidence": {"type": "string"},
                    "confidence_status": {"type": "string", "enum": ["likely", "uncertain"]},
                },
            },
        },
    },
}

FEATURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["feature_id", "canonical_label", "definition", "feature_type", "task_relevance",
                 "dominance", "procedure_group", "sequence_order", "supporting_items", "source_spans",
                 "inference_rationale", "proposed_misconception", "llm_uncertainty_note"],
    "properties": {
        "feature_id": {"type": "string"},
        "canonical_label": {"type": "string"},
        "definition": {"type": "string"},
        "feature_type": {"type": "string", "enum": list(FEATURE_TYPES)},
        "task_relevance": {"type": "string"},
        "dominance": {"type": ["string", "null"], "enum": list(DOMINANCE) + [None]},
        "procedure_group": _NULLABLE_STR,
        "sequence_order": {"type": ["integer", "null"]},
        "supporting_items": {"type": "array", "items": {"type": "string"}},
        "source_spans": {"type": "array", "items": _SPAN},
        "inference_rationale": {"type": "string"},
        "proposed_misconception": _NULLABLE_STR,
        "llm_uncertainty_note": {"type": "string"},
    },
}

DECOMPOSITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "target_item_id", "features"],
    "properties": {
        "schema_version": {"type": "string"},
        "target_item_id": {"type": "string"},
        "features": {"type": "array", "items": FEATURE_SCHEMA},
    },
}
