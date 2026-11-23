"""Guard the specialist registry's version against silent contract edits.

Each specialist run records ``registry_version``. If the prompts, schemas,
routing or request projection change while the version stays the same,
runs made under different contracts become indistinguishable. When this
test fails after an intentional edit, bump ``version`` in the registry and
add the new version and fingerprint to ``KNOWN_CONTRACTS``.
"""

import hashlib
import json

import ai_adapter as ai

CONTRACT_KEYS = ("providers", "routing", "request_schema", "request_projection", "agents")

# version -> SHA-256 of the canonical JSON of CONTRACT_KEYS
KNOWN_CONTRACTS = {
    "1.3": "1500c69923445be6f25c114279e2d96ba52a9e75b1ed17b9d830d74b287eb5f1",
}


def contract_fingerprint(registry):
    contract = {key: registry[key] for key in CONTRACT_KEYS}
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_registry_contracts_match_their_recorded_version():
    registry = ai.specialist_registry()
    version = registry["version"]
    assert version in KNOWN_CONTRACTS, (
        f"Registry version {version} has no recorded fingerprint. Add it to KNOWN_CONTRACTS."
    )
    assert contract_fingerprint(registry) == KNOWN_CONTRACTS[version], (
        f"The registry contracts changed but its version is still {version}. "
        "Bump the registry version and record the new fingerprint."
    )


def test_registry_defines_the_seven_specialists_once():
    roles = [agent["role_id"] for agent in ai.specialist_registry()["agents"]]
    assert sorted(roles) == sorted({
        "bank_retrieval", "evidence_review", "key_feature_modeling", "item_drafting",
        "distractor_modeling", "item_validation", "rationale_writing",
    })
