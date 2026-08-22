"""Preexecution-only helpers for frozen semantic Discovery arms.

This module implements deterministic A3r random-code targets, exact-checkpoint
identity guards, and the frozen A5 matching construction. It does not execute
or interpret scientific outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    if len(set(signatures)) != len(signatures):
        raise SemanticArmError("A3r exact semantic signature identities must be unique per step")
    identity = _codebook_identity(codebook_id, readiness_path)
    seed = reconstruct_seed(identity)
    rows = [torch.tensor(generate_code(seed, signature), dtype=torch.float32) for signature in signatures]
    rows.extend(torch.zeros(SEMANTIC_DIMENSION, dtype=torch.float32) for _ in range(total_steps - len(rows)))
    result = torch.stack(rows).unsqueeze(0)
    norms = result[0, : len(signatures)].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0):
        raise SemanticArmError("frozen A3r codebook target normalization drift")
    return result


@dataclass(frozen=True)
class A5Unit:
    unit_id: str
    base_task_id: str
    planner_seed: int
    split: str
    step_index: int
    remaining_distance_bucket: str
    hand_mode: str
    semantic_signature: str
    intent_id: str
    semantic_artifact_sha256: str

    @property
    def unit_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.__dict__)).hexdigest()

    @property
    def stratum(self) -> tuple[int, str, int, str, str]:
        return (
            self.planner_seed,
            self.split,
            self.step_index,
            self.remaining_distance_bucket,
            self.hand_mode,
        )


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


def a5_contract_sha256(path: Path = A5_CONTRACT_PATH) -> str:
    if not path.is_file():
        raise SemanticArmError("frozen A5 ablation contract is missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construct_a5_derangement(
    units: Iterable[A5Unit], *, contract_path: Path = A5_CONTRACT_PATH
) -> dict[str, str]:
    rows = list(units)
    if not rows or len({row.unit_id for row in rows}) != len(rows):
        raise SemanticArmError("A5 units must be non-empty with unique unit_id")
    contract_hash = a5_contract_sha256(contract_path)
    grouped: dict[tuple[int, str, int, str, str], list[A5Unit]] = {}
    for row in rows:
        grouped.setdefault(row.stratum, []).append(row)

    full_mapping: dict[str, str] = {}
    for stratum in sorted(grouped):
        source_rows = sorted(
            grouped[stratum], key=lambda row: (row.base_task_id, row.semantic_artifact_sha256)
        )
        by_id = {row.unit_id: row for row in source_rows}
        candidate_for_source: dict[str, list[str]] = {}
        for source in source_rows:
            candidates = [candidate for candidate in source_rows if _foreign_allowed(source, candidate)]
            candidates.sort(key=lambda candidate: _edge_key(source, candidate, contract_hash))
            candidate_for_source[source.unit_id] = [candidate.unit_id for candidate in candidates]

        candidate_owner: dict[str, str] = {}

        def augment(source_id: str, visited: set[str]) -> bool:
            for candidate_id in candidate_for_source[source_id]:
                if candidate_id in visited:
                    continue
                visited.add(candidate_id)
                owner = candidate_owner.get(candidate_id)
                if owner is None or augment(owner, visited):
                    candidate_owner[candidate_id] = source_id
                    return True
            return False

        for source in source_rows:
            if not augment(source.unit_id, set()):
                raise SemanticArmError("BLOCKED_CONTROL_CONSTRUCTION")

        mapping = {source_id: candidate_id for candidate_id, source_id in candidate_owner.items()}
        if set(mapping) != set(by_id) or len(set(mapping.values())) != len(by_id):
            raise SemanticArmError("A5 perfect derangement construction failed")
        for source_id, candidate_id in mapping.items():
            if source_id == candidate_id or not _foreign_allowed(by_id[source_id], by_id[candidate_id]):
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
    return _sha256(_canonical_json(dict(sorted(mapping.items()))))
