"""Deterministic construction of the text sent for embedding."""

import hashlib
import json


def build_embed_text(fields, field_order, options=None, include_options=False):
    """Join the selected fields in a fixed order; options are appended only when requested.

    The answer key is never included, so the embedding does not carry the keyed
    response as a marker.
    """
    lines = []
    for label in field_order:
        value = fields.get(label)
        if value:
            lines.append(f"{label}: {value}")
    if include_options and options:
        lines.append("Options:")
        lines.extend(f"{o['option_id']}. {o['text']}" for o in options)
    return "\n".join(lines)


def template_description(field_order, include_options):
    parts = [f"{label}: {{{label}}}" for label in field_order]
    if include_options:
        parts.append("Options: {option_id}. {option_text} (all options, no key marker)")
    return " \\n ".join(parts)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj):
    """Stable SHA-256 of a JSON-serialisable object."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))
