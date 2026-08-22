"""Preexecution-only helpers for frozen semantic Discovery arms.

This module implements deterministic A3r random-code targets, exact-checkpoint
identity guards, and the frozen A5 matching construction. It does not execute
or interpret scientific outcomes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from research_programs.planner.semantic_feedback_readiness import (
    CONFIG_PATH as READINESS_CONFIG_PATH,
    generate_code,
    load_config,
    reconstruct_seed,
    validate_authority_binding,
)

SEMANTIC_DIMENSION = 384
A3R_VARIANT = "A3r"
A5_CONTRACT_PATH = Path("docs/controls/planner_latent_ablation_contract_v1.yaml")


class SemanticArmError(ValueError):
    """Fail-closed semantic Discovery arm construction error."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if not value.startswith("sha256:") or len(value) != 71:
        raise SemanticArmError(f"{label} must be sha256:<64 hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise SemanticArmError(f"{label} must be sha256:<64 hex>") from exc


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json(list(tensor.shape)) + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def parameter_shape_manifest(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(tensor.shape) for name, tensor in sorted(state.items())}


def validate_a3r_checkpoint_independence(
    *,
    a3_state: Mapping[str, torch.Tensor],
    a3r_state: Mapping[str, torch.Tensor],
    a3_checkpoint_id: str,
    a3r_checkpoint_id: str,
    require_trained_state_difference: bool = True,
) -> dict[str, str]:
    if parameter_shape_manifest(a3_state) != parameter_shape_manifest(a3r_state):
        raise SemanticArmError("A3r parameter shapes must exactly match A3")
    if not a3_checkpoint_id or not a3r_checkpoint_id or a3_checkpoint_id == a3r_checkpoint_id:
        raise SemanticArmError("A3r must use a separate checkpoint identity")
    a3_hash = state_dict_sha256(a3_state)
    a3r_hash = state_dict_sha256(a3r_state)
    if require_trained_state_difference and a3_hash == a3r_hash:
        raise SemanticArmError("A3 checkpoint/state reuse for trained A3r is forbidden")
    return {"a3_state_sha256": a3_hash, "a3r_state_sha256": a3r_hash}


def validate_exact_a3_checkpoint(
    *, expected_a3_state_sha256: str, arm_state_sha256: str, retrained: bool
) -> None:
    _validate_sha256(expected_a3_state_sha256, "expected A3 state")
    _validate_sha256(arm_state_sha256, "arm state")
    if retrained or arm_state_sha256 != expected_a3_state_sha256:
        raise SemanticArmError("INVALID_CHECKPOINT_OR_RETRAINING")


def _codebook_identity(codebook_id: str, path: Path = READINESS_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(path)
    validate_authority_binding(config)
    matches = [item for item in config["codebooks"] if item["id"] == codebook_id]
    if len(matches) != 1:
        raise SemanticArmError("unknown or duplicate frozen A3r codebook identity")
    return matches[0]


def a3r_codebook_identity_digest(
    codebook_id: str, path: Path = READINESS_CONFIG_PATH
) -> str:
    return _sha256(_canonical_json(_codebook_identity(codebook_id, path)))


def a3r_targets(
    signature_sha256s: Iterable[str],
    *,
    codebook_id: str,
    total_steps: int = 17,
    readiness_path: Path = READINESS_CONFIG_PATH,
) -> torch.Tensor:
    signatures = list(signature_sha256s)
    if not signatures or len(signatures) > total_steps:
        raise SemanticArmError("A3r signature count must be in [1, total_steps]")
    identity = _codebook_identity(codebook_id, readiness_path)
    seed = reconstruct_seed(identity)
    rows = [
        torch.tensor(generate_code(seed, signature), dtype=torch.float32)
        for signature in signatures
    ]
    rows.extend(
        torch.zeros(SEMANTIC_DIMENSION, dtype=torch.float32)
        for _ in range(total_steps - len(rows))
    )
    result = torch.stack(rows).unsqueeze(0)
    norms = result[0, : len(signatures)].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0):
        raise SemanticArmError("frozen A3r codebook target normalization drift")
    return result


@dataclass(frozen=True)
class A5Unit:
    unit_id: str
    unit_hash: str
    base_task_id: str
    planner_seed: int
    split: str
    step_index: int
    remaining_distance_bucket: str
    hand_mode: str
    semantic_signature: str
    intent_id: str
    semantic_artifact_sha256: str

    def __post_init__(self) -> None:
        _validate_sha256(self.unit_hash, "A5 unit_hash")
        _validate_sha256(self.semantic_artifact_sha256, "A5 semantic_artifact_sha256")

    @property
    def stratum(self) -> tuple[int, str, int, str, str]:
        return (
            self.planner_seed,
            self.split,
            self.step_index,
            self.remaining_distance_bucket,
            self.hand_mode,
        )

    @property
    def stratum_hash(self) -> str:
        return _sha256(_canonical_json(self.stratum))


def _foreign_allowed(source: A5Unit, candidate: A5Unit) -> bool:
    return (
        source.base_task_id != candidate.base_task_id
        and source.step_index == candidate.step_index
        and source.remaining_distance_bucket == candidate.remaining_distance_bucket
        and source.hand_mode == candidate.hand_mode
        and source.semantic_signature != candidate.semantic_signature
        and source.intent_id != candidate.intent_id
        and source.stratum == candidate.stratum
    )


def _edge_key(source: A5Unit, candidate: A5Unit, contract_hash: str) -> str:
    payload = (source.unit_hash + candidate.unit_hash + contract_hash).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _augment_matching(
    source_hash: str,
    visited: set[str],
    candidate_for_source: Mapping[str, list[str]],
    candidate_owner: dict[str, str],
) -> bool:
    for candidate_hash in candidate_for_source[source_hash]:
        if candidate_hash in visited:
            continue
        visited.add(candidate_hash)
        owner = candidate_owner.get(candidate_hash)
        if owner is None or _augment_matching(
            owner, visited, candidate_for_source, candidate_owner
        ):
            candidate_owner[candidate_hash] = source_hash
            return True
    return False


def a5_contract_sha256(path: Path = A5_CONTRACT_PATH) -> str:
    if not path.is_file():
        raise SemanticArmError("frozen A5 ablation contract is missing")
    return _sha256(path.read_bytes())


def construct_a5_derangement(
    units: Iterable[A5Unit], *, contract_path: Path = A5_CONTRACT_PATH
) -> dict[str, str]:
    rows = list(units)
    if not rows or len({row.unit_hash for row in rows}) != len(rows):
        raise SemanticArmError("A5 units must be non-empty with unique unit_hash")
    contract_hash = a5_contract_sha256(contract_path)
    grouped: dict[tuple[int, str, int, str, str], list[A5Unit]] = {}
    for row in rows:
        grouped.setdefault(row.stratum, []).append(row)

    full_mapping: dict[str, str] = {}
    for stratum in sorted(grouped):
        source_rows = sorted(
            grouped[stratum], key=lambda row: (row.base_task_id, row.semantic_artifact_sha256)
        )
        by_hash = {row.unit_hash: row for row in source_rows}
        candidate_for_source: dict[str, list[str]] = {}
        for source in source_rows:
            candidates = [
                candidate
                for candidate in source_rows
                if _foreign_allowed(source, candidate)
            ]
            candidates.sort(key=lambda candidate: _edge_key(source, candidate, contract_hash))
            candidate_for_source[source.unit_hash] = [
                candidate.unit_hash for candidate in candidates
            ]

        candidate_owner: dict[str, str] = {}
        for source in source_rows:
            if not _augment_matching(
                source.unit_hash, set(), candidate_for_source, candidate_owner
            ):
                raise SemanticArmError("BLOCKED_CONTROL_CONSTRUCTION")

        mapping = {
            source_hash: candidate_hash
            for candidate_hash, source_hash in candidate_owner.items()
        }
        if set(mapping) != set(by_hash) or len(set(mapping.values())) != len(by_hash):
            raise SemanticArmError("A5 perfect derangement construction failed")
        for source_hash, candidate_hash in mapping.items():
            if source_hash == candidate_hash or not _foreign_allowed(
                by_hash[source_hash], by_hash[candidate_hash]
            ):
                raise SemanticArmError("A5 mapping violates frozen foreign requirements")
        full_mapping.update(mapping)

    if len(full_mapping) != len(rows) or len(set(full_mapping.values())) != len(rows):
        raise SemanticArmError("A5 donor reuse or incomplete coverage")
    return dict(sorted(full_mapping.items()))


def a5_mapping_digest(mapping: Mapping[str, str]) -> str:
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise SemanticArmError("A5 mapping must be complete and donor-unique")
    if any(source == donor for source, donor in mapping.items()):
        raise SemanticArmError("A5 self-assignment forbidden")
    for source, donor in mapping.items():
        _validate_sha256(source, "A5 source mapping hash")
        _validate_sha256(donor, "A5 donor mapping hash")
    return _sha256(_canonical_json(dict(sorted(mapping.items()))))


def a5_mapping_manifest(
    *,
    units: Iterable[A5Unit],
    mapping: Mapping[str, str],
    a3_checkpoint_sha256: str,
    contract_path: Path = A5_CONTRACT_PATH,
) -> dict[str, Any]:
    _validate_sha256(a3_checkpoint_sha256, "A3 checkpoint")
    rows = {unit.unit_hash: unit for unit in units}
    if set(rows) != set(mapping) or set(rows) != set(mapping.values()):
        raise SemanticArmError("A5 manifest requires perfect complete derangement")
    mapping_digest = a5_mapping_digest(mapping)
    entries: list[dict[str, Any]] = []
    for source_hash, foreign_hash in sorted(mapping.items()):
        source = rows[source_hash]
        foreign = rows[foreign_hash]
        if not _foreign_allowed(source, foreign):
            raise SemanticArmError("A5 manifest mapping violates frozen filters")
        entries.append(
            {
                "source_unit_hash": source_hash,
                "source_base_task_id": source.base_task_id,
                "step_index": source.step_index,
                "foreign_unit_hash": foreign_hash,
                "foreign_base_task_id": foreign.base_task_id,
                "source_semantic_sha256": source.semantic_artifact_sha256,
                "foreign_semantic_sha256": foreign.semantic_artifact_sha256,
                "stratum_hash": source.stratum_hash,
            }
        )
    payload = {
        "schema_version": "semantic-discovery-a5-mapping/0.1",
        "a3_checkpoint_sha256": a3_checkpoint_sha256,
        "contract_sha256": a5_contract_sha256(contract_path),
        "mapping_sha256": mapping_digest,
        "mappings": entries,
        "excluded_units": [],
    }
    return {**payload, "manifest_sha256": _sha256(_canonical_json(payload))}
