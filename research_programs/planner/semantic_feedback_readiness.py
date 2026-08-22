from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CONFIG_PATH = Path(".research/readiness/semantic_feedback_discovery_execution_v1.json")
ALLOWED_METADATA_FIELDS = (
    "opaque_task_id",
    "opaque_ref_ids",
    "opaque_signature_id",
    "position",
    "nuisance_bucket",
)


class ReadinessError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_digest(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(config))


def reconstruct_seed(identity: dict[str, Any]) -> bytes:
    seed = hashlib.sha256(identity["derivation_input"].encode("utf-8")).digest()
    if seed.hex() != identity["seed_hex"]:
        raise ReadinessError(f"codebook seed mismatch: {identity['id']}")
    return seed


def generate_code(seed: bytes, signature_sha256: str) -> tuple[float, ...]:
    if len(signature_sha256) != 64:
        raise ReadinessError("signature_sha256 must be 64 hex chars")
    int(signature_sha256, 16)
    raw = hashlib.shake_256(seed + signature_sha256.encode("utf-8")).digest(48)
    scale = math.sqrt(384.0)
    values: list[float] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            bit = (byte >> shift) & 1
            values.append((1.0 if bit else -1.0) / scale)
    if len(values) != 384:
        raise AssertionError("generator dimension drift")
    return tuple(values)


def reconstruct_codebooks(config: dict[str, Any], signatures: Iterable[str]) -> dict[str, dict[str, tuple[float, ...]]]:
    ordered = sorted(signatures)
    if len(set(ordered)) != len(ordered):
        raise ReadinessError("duplicate signature identity")
    result: dict[str, dict[str, tuple[float, ...]]] = {}
    for identity in config["codebooks"]:
        seed = reconstruct_seed(identity)
        result[identity["id"]] = {sig: generate_code(seed, sig) for sig in ordered}
    if len(result) != 2 or len(set(result)) != 2:
        raise ReadinessError("exactly two distinct codebooks required")
    return result


def validate_composition_split(config: dict[str, Any]) -> dict[str, Any]:
    split = config["composition_split"]
    train = set(split["train"])
    stress = set(split["stress"])
    if train & stress:
        raise ReadinessError("INVALID_GEOMETRY_ESTIMAND_SIGNATURE_OVERLAP")
    train_atoms = {atom for sig in train for atom in sig.split("|")}
    stress_atoms = {atom for sig in stress for atom in sig.split("|")}
    if not stress_atoms <= train_atoms:
        raise ReadinessError("unseen stress atom")
    return {"id": split["id"], "train_count": len(train), "stress_count": len(stress), "atoms_seen": sorted(stress_atoms)}


@dataclass(frozen=True)
class DonorUnit:
    unit_id: str
    episode_id: str
    semantic_signature: str
    planner_seed: int
    split_id: str
    intervention_position: int
    remaining_distance_bucket: str
    hand_mode: str
    feedback_norm_bucket: str
    feedback_norm_raw: float


def select_wrong_semantic_donor(target: DonorUnit, candidates: Iterable[DonorUnit]) -> DonorUnit | None:
    eligible = [
        c for c in candidates
        if c.unit_id != target.unit_id
        and c.semantic_signature != target.semantic_signature
        and c.planner_seed == target.planner_seed
        and c.split_id == target.split_id
        and c.intervention_position == target.intervention_position
        and c.remaining_distance_bucket == target.remaining_distance_bucket
        and c.hand_mode == target.hand_mode
        and c.feedback_norm_bucket == target.feedback_norm_bucket
    ]
    eligible.sort(key=lambda c: (abs(c.feedback_norm_raw - target.feedback_norm_raw), c.episode_id, c.unit_id))
    return eligible[0] if eligible else None


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in ALLOWED_METADATA_FIELDS}


def validate_collision_fixture(config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["collision_fixture"]
    a, b = fixture["collision_pair"]
    if canonical_json(_metadata(a)) != canonical_json(_metadata(b)):
        raise ReadinessError("INVALID_SHORTCUT_NOT_EXCLUDED")
    if a["semantic_feedback"] == b["semantic_feedback"] or a["required_next_action"] == b["required_next_action"]:
        raise ReadinessError("INVALID_SEMANTIC_CHANNEL_NOT_USABLE")
    ia, ib = fixture["inverse_metadata_pair"]
    if canonical_json(_metadata(ia)) == canonical_json(_metadata(ib)):
        raise ReadinessError("inverse metadata must differ")
    if ia["semantic_feedback"] != ib["semantic_feedback"] or ia["required_next_action"] != ib["required_next_action"]:
        raise ReadinessError("INVALID_METADATA_INVARIANCE")
    return {
        "fixture_id": fixture["id"],
        "metadata_only_collision": "INDETERMINATE_MUST_NOT_SEPARATE_TARGETS",
        "semantic_oracle_collision": "MUST_SEPARATE_TARGETS",
        "inverse_metadata_target_invariant": True,
    }


def validate_checkpoint_binding(expected_a3: str, arm_checkpoints: dict[str, str], retrained_arms: Iterable[str] = ()) -> str:
    required = {"A3", "A4", "A5", "WRONG_SEMANTIC_DONOR"}
    if set(arm_checkpoints) != required:
        return "INVALID_CHECKPOINT_OR_RETRAINING"
    if any(value != expected_a3 for value in arm_checkpoints.values()):
        return "INVALID_CHECKPOINT_OR_RETRAINING"
    if set(retrained_arms) & {"A4", "A5", "WRONG_SEMANTIC_DONOR"}:
        return "INVALID_CHECKPOINT_OR_RETRAINING"
    return "EVALUATED"


def encode_state(*, checkpoint_valid: bool = True, shortcut_valid: bool = True, oracle_valid: bool = True,
                 metadata_invariant: bool = True, split_valid: bool = True, both_codebooks_valid: bool = True,
                 downstream_opportunities: int = 1, intervention_applicable: bool = True, donor_available: bool = True,
                 terminated_before_intervention: bool = False, endpoint_defined: bool = True) -> str:
    if not checkpoint_valid:
        return "INVALID_CHECKPOINT_OR_RETRAINING"
    if not shortcut_valid:
        return "INVALID_SHORTCUT_NOT_EXCLUDED"
    if not oracle_valid:
        return "INVALID_SEMANTIC_CHANNEL_NOT_USABLE"
    if not metadata_invariant:
        return "INVALID_METADATA_INVARIANCE"
    if not split_valid:
        return "INVALID_GEOMETRY_ESTIMAND_SIGNATURE_OVERLAP"
    if not both_codebooks_valid:
        return "INVALID_GEOMETRY_ESTIMAND_NO_PARTIAL_AGGREGATION"
    if downstream_opportunities == 0:
        return "NOT_EVALUATED_NO_DOWNSTREAM_OPPORTUNITY"
    if not intervention_applicable:
        return "NOT_EVALUATED_INTERVENTION_NOT_APPLICABLE"
    if not donor_available:
        return "NOT_EVALUATED_DONOR_UNAVAILABLE"
    if terminated_before_intervention:
        return "CENSORED_TERMINATED_BEFORE_PLANNED_INTERVENTION"
    if not endpoint_defined:
        return "NOT_APPLICABLE_ENDPOINT_UNDEFINED"
    return "EVALUATED"


def endpoint_placeholders(state: str) -> dict[str, Any]:
    return {
        "validity_state": state,
        "immediate_next_step_effect": None,
        "free_running_total_effect": None,
        "outcome_bearing": False,
    }


def readiness_replay(config: dict[str, Any]) -> dict[str, Any]:
    synthetic_signatures = [hashlib.sha256(s.encode("utf-8")).hexdigest() for s in ("alpha", "beta", "gamma")]
    codebooks = reconstruct_codebooks(config, synthetic_signatures)
    split = validate_composition_split(config)
    collision = validate_collision_fixture(config)

    target = DonorUnit("u-target", "e-9", "sig-a", 17, "dev", 3, "r2", "left", "n1", 1.0)
    candidates = [
        DonorUnit("u-z", "e-2", "sig-b", 17, "dev", 3, "r2", "left", "n1", 1.2),
        DonorUnit("u-a", "e-1", "sig-c", 17, "dev", 3, "r2", "left", "n1", 0.8),
    ]
    donor = select_wrong_semantic_donor(target, candidates)
    checkpoint_state = validate_checkpoint_binding("ckpt-A3", {k: "ckpt-A3" for k in ("A3", "A4", "A5", "WRONG_SEMANTIC_DONOR")})

    branches = {
        "valid": encode_state(),
        "checkpoint": encode_state(checkpoint_valid=False),
        "shortcut": encode_state(shortcut_valid=False),
        "oracle": encode_state(oracle_valid=False),
        "metadata": encode_state(metadata_invariant=False),
        "split": encode_state(split_valid=False),
        "codebooks": encode_state(both_codebooks_valid=False),
        "no_opportunity": encode_state(downstream_opportunities=0),
        "not_applicable": encode_state(intervention_applicable=False),
        "donor_unavailable": encode_state(donor_available=False),
        "censored": encode_state(terminated_before_intervention=True),
        "endpoint_undefined": encode_state(endpoint_defined=False),
    }
    payload = {
        "readiness_id": config["id"],
        "config_digest": config_digest(config),
        "codebook_ids": sorted(codebooks),
        "codebook_probe_digest": sha256_text(canonical_json({
            key: {sig: list(vector[:8]) for sig, vector in value.items()} for key, value in codebooks.items()
        })),
        "split": split,
        "collision": collision,
        "selected_donor": None if donor is None else {"episode_id": donor.episode_id, "unit_id": donor.unit_id},
        "checkpoint_state": checkpoint_state,
        "validity_branches": branches,
        "endpoint_placeholders": endpoint_placeholders("EVALUATED"),
        "scientific_execution": False,
        "held_out_access": False,
        "claim_bearing_evidence": False,
        "go_latent": "NOT EVALUATED",
    }
    payload["replay_digest"] = sha256_text(canonical_json(payload))
    return payload
