"""Versioned formal runtime/1.1 acceptance semantics.

This module is internal to ``scripts.fixed_target_contract.validate_acceptance_bundle``.
It deliberately does not expose an independent final-verdict API.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import scripts.fixed_target_contract as ft

FORMAL_ACCEPTANCE_VERSION = "toy-quality-fixed-target-acceptance/1.1"
FORMAL_PROVENANCE_VERSION = "toy-quality-fixed-target-formal-provenance/1.1"
FORMAL_EXECUTION_CONTEXT = "formal-fixed-target"
FORMAL_TRAINING_MODE = "TRAINED_IN_ATTEMPT_SHARDED"
FORMAL_EVALUATOR_VERSION = "development-quality-evaluation/0.1-runtime1.1-sharded/1.0"
EXPECTED_VARIANTS = ("A2", "A3", "A4")
EXPECTED_SEEDS = (17, 29, 43)
_RUNTIME_ENV_FIELDS = (
    "ATEN_CPU_CAPABILITY",
    "MKL_CBWR",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_formal_target_contract(contract: dict[str, Any]) -> None:
    ft.validate_target_contract(contract)
    if contract["microcode_policy"]["mode"] == "minimum":
        raise ValueError("FIXED_TARGET_FORMAL_MICROCODE_POLICY_NOT_EXACT")


def _prepare_contract_environment(contract: dict[str, Any]) -> None:
    """Set process controls before lazy torch-backed semantic validation."""
    ft.validate_target_contract(contract)
    for name in _RUNTIME_ENV_FIELDS:
        expected = str(contract[name])
        current = os.environ.get(name)
        if current not in {None, expected}:
            raise RuntimeError(f"FIXED_TARGET_ENV_MISMATCH:{name}")
        os.environ[name] = expected
    if "torch" not in sys.modules:
        ft.prepare_process_environment(contract)


def formal_provenance_sha256(value: dict[str, Any]) -> str:
    payload = copy.deepcopy(value)
    payload.pop("formal_provenance_sha256", None)
    return ft.sha256_value(payload)


def _unit_result_hash(evaluation_root: Path, variant: str, seed: int) -> str:
    rows = []
    for line in (evaluation_root / "task-results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("variant") == variant and row.get("seed") == seed:
            rows.append(row)
    rows.sort(key=lambda row: row["task_id"])
    expected_tasks = list(ft.HISTORICAL_ORDERED_EVAL_TASK_IDS)
    if [row.get("task_id") for row in rows] != expected_tasks:
        raise ValueError("FIXED_TARGET_FORMAL_UNIT_TASK_COVERAGE_MISMATCH")
    payload = b"".join(ft.canonical_bytes(row) + b"\n" for row in rows)
    return ft.sha256_bytes(payload)


def build_formal_provenance(
    attempt_root: Path,
    attempt_index: int,
    *,
    workflow_run_id: int,
    job_id: int,
    workflow_sha: str,
) -> dict[str, Any]:
    """Seal attempt-local orchestration provenance before foundation packaging."""
    contract = _read(attempt_root / "target-contract.json")
    _validate_formal_target_contract(contract)
    _prepare_contract_environment(contract)
    from scripts.fixed_target_quality_sharded import (  # noqa: PLC0415
        unit_identity,
        validate_attempt,
        validate_unit_manifest,
    )

    attempt, validated_contract = validate_attempt(attempt_root)
    if attempt["execution_context"] != FORMAL_EXECUTION_CONTEXT:
        raise ValueError("FIXED_TARGET_FORMAL_EXECUTION_CONTEXT_REQUIRED")
    if workflow_run_id < 1 or job_id < 1:
        raise ValueError("FIXED_TARGET_FORMAL_WORKFLOW_PROVENANCE_INVALID")
    if workflow_sha != attempt["execution_implementation_commit"]:
        raise ValueError("FIXED_TARGET_WORKFLOW_IMPLEMENTATION_MISMATCH")
    units = []
    seen: set[tuple[str, int]] = set()
    for variant in EXPECTED_VARIANTS:
        for seed in EXPECTED_SEEDS:
            unit_root = attempt_root / "units" / variant / f"seed-{seed}"
            manifest = _read(unit_root / "unit-manifest.json")
            validate_unit_manifest(manifest, attempt, validated_contract)
            key = (manifest["variant"], manifest["seed"])
            if key in seen:
                raise ValueError("FIXED_TARGET_FORMAL_DUPLICATE_UNIT")
            seen.add(key)
            if manifest["unit_manifest_sha256"] != unit_identity(manifest):
                raise ValueError("FIXED_TARGET_FORMAL_UNIT_IDENTITY_MISMATCH")
            units.append(copy.deepcopy(manifest))
    expected = {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS}
    if seen != expected:
        raise ValueError("FIXED_TARGET_FORMAL_UNIT_COVERAGE_MISMATCH")
    provenance: dict[str, Any] = {
        "formal_provenance_version": FORMAL_PROVENANCE_VERSION,
        "attempt_index": attempt_index,
        "attempt_identity_sha256": attempt["attempt_identity_sha256"],
        "workflow_run_id": workflow_run_id,
        "job_id": job_id,
        "workflow_sha": workflow_sha,
        "execution_context": attempt["execution_context"],
        "execution_implementation_commit": attempt["execution_implementation_commit"],
        "scientific_parent_implementation_commit": attempt[
            "scientific_parent_implementation_commit"
        ],
        "target_contract_sha256": attempt["target_contract_sha256"],
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation_sha256": attempt["target_observation_sha256"],
        "source_inventory_sha256": attempt["source_inventory_sha256"],
        "units": units,
        "formal_provenance_sha256": "",
    }
    provenance["formal_provenance_sha256"] = formal_provenance_sha256(provenance)
    return provenance


def _checkpoint_lineage_sha256(unit: dict[str, Any]) -> str:
    return ft.sha256_value(
        {
            "attempt_identity_sha256": unit["attempt_identity_sha256"],
            "variant": unit["variant"],
            "seed": unit["seed"],
            "unit_manifest_sha256": unit["unit_manifest_sha256"],
            "checkpoint_manifest_sha256": unit["checkpoint_manifest_sha256"],
        }
    )


def _unit_projection(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_identity_sha256": unit["attempt_identity_sha256"],
        "variant": unit["variant"],
        "seed": unit["seed"],
        "unit_manifest_sha256": unit["unit_manifest_sha256"],
        "checkpoint_manifest_sha256": unit["checkpoint_manifest_sha256"],
        "checkpoint_lineage_sha256": _checkpoint_lineage_sha256(unit),
        "task_results_sha256": unit["task_results_sha256"],
    }


def _validate_formal_provenance(
    attempt_root: Path,
    attempt_index: int,
    *,
    preflight: dict[str, Any],
    execution: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    provenance = _read(attempt_root / "formal-provenance.json")
    ft._validate_schema(
        "fixed_target_formal_provenance.schema.json",
        provenance,
        "FIXED_TARGET_FORMAL_PROVENANCE_SCHEMA_INVALID",
    )
    if provenance["formal_provenance_version"] != FORMAL_PROVENANCE_VERSION:
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_VERSION_MISMATCH")
    if provenance["attempt_index"] != attempt_index:
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_INDEX_MISMATCH")
    if provenance["formal_provenance_sha256"] != formal_provenance_sha256(provenance):
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_HASH_MISMATCH")
    if provenance["execution_context"] != FORMAL_EXECUTION_CONTEXT:
        raise ValueError("FIXED_TARGET_FORMAL_EXECUTION_CONTEXT_REQUIRED")
    if provenance["workflow_sha"] != provenance["execution_implementation_commit"]:
        raise ValueError("FIXED_TARGET_WORKFLOW_IMPLEMENTATION_MISMATCH")

    bindings = {
        "execution_implementation_commit": execution["implementation_commit"],
        "scientific_parent_implementation_commit": execution[
            "scientific_parent_implementation_commit"
        ],
        "target_contract_sha256": execution["target_contract_sha256"],
        "runtime_contract_sha256": execution["runtime_contract_sha256"],
        "target_observation_sha256": execution["target_observation_sha256"],
        "source_inventory_sha256": execution["source_inventory_sha256"],
    }
    for field, expected in bindings.items():
        if provenance[field] != expected:
            raise ValueError(f"FIXED_TARGET_FORMAL_PROVENANCE_BINDING_MISMATCH:{field}")
    if provenance["execution_implementation_commit"] != preflight["implementation_commit"]:
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_IMPLEMENTATION_MISMATCH")
    if provenance["execution_implementation_commit"] != evaluation_config.get(
        "implementation_commit"
    ):
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_IMPLEMENTATION_MISMATCH")
    if provenance["attempt_identity_sha256"] != evaluation_config.get(
        "attempt_identity_sha256"
    ):
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_ATTEMPT_IDENTITY_MISMATCH")
    if provenance["execution_context"] != evaluation_config.get("execution_context"):
        raise ValueError("FIXED_TARGET_FORMAL_PROVENANCE_EXECUTION_CONTEXT_MISMATCH")

    contract = preflight["target_contract"]
    _prepare_contract_environment(contract)
    from scripts.fixed_target_quality_sharded import (  # noqa: PLC0415
        unit_identity,
        validate_unit_manifest,
    )

    attempt_binding = {
        "attempt_identity_sha256": provenance["attempt_identity_sha256"],
        "scientific_parent_implementation_commit": provenance[
            "scientific_parent_implementation_commit"
        ],
        "execution_implementation_commit": provenance["execution_implementation_commit"],
        "execution_context": provenance["execution_context"],
        "runtime_contract_sha256": provenance["runtime_contract_sha256"],
        "target_observation_sha256": provenance["target_observation_sha256"],
        "source_inventory_sha256": provenance["source_inventory_sha256"],
    }
    units = provenance["units"]
    if len(units) != 9:
        raise ValueError("FIXED_TARGET_FORMAL_UNIT_COVERAGE_MISMATCH")
    seen: set[tuple[str, int]] = set()
    unit_hashes: set[str] = set()
    unit_projection = []
    evaluation_root = attempt_root / "evaluation"
    for unit in units:
        validate_unit_manifest(unit, attempt_binding, contract)
        if unit["unit_manifest_sha256"] != unit_identity(unit):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_IDENTITY_MISMATCH")
        if unit["attempt_identity_sha256"] != provenance["attempt_identity_sha256"]:
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_ATTEMPT_BINDING_MISMATCH")
        key = (unit["variant"], unit["seed"])
        if key in seen:
            raise ValueError("FIXED_TARGET_FORMAL_DUPLICATE_UNIT")
        seen.add(key)
        if unit["unit_manifest_sha256"] in unit_hashes:
            raise ValueError("FIXED_TARGET_FORMAL_DUPLICATE_UNIT_IDENTITY")
        unit_hashes.add(unit["unit_manifest_sha256"])
        if unit["dataset_hash"] != evaluation_config.get("dataset_manifest_hash"):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_DATASET_MISMATCH")
        if unit["ordered_train_task_ids"] != evaluation_config.get("train_task_ids"):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_TRAIN_SPLIT_MISMATCH")
        if unit["ordered_eval_task_ids"] != evaluation_config.get("eval_task_ids"):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_EVAL_SPLIT_MISMATCH")
        checkpoint = (
            evaluation_root
            / "training-runs"
            / unit["variant"]
            / f"seed-{unit['seed']}"
            / "checkpoint-manifest.json"
        )
        if unit["checkpoint_manifest_sha256"] != ft.sha256_bytes(checkpoint.read_bytes()):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_CHECKPOINT_BINDING_MISMATCH")
        if unit["task_results_sha256"] != _unit_result_hash(
            evaluation_root, unit["variant"], unit["seed"]
        ):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_RESULT_BINDING_MISMATCH")
        unit_projection.append(_unit_projection(unit))
    expected = {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS}
    if seen != expected or len(unit_hashes) != 9:
        raise ValueError("FIXED_TARGET_FORMAL_UNIT_COVERAGE_MISMATCH")
    unit_projection.sort(key=lambda item: (item["variant"], item["seed"]))
    return {
        "attempt_identity_sha256": provenance["attempt_identity_sha256"],
        "formal_provenance_sha256": provenance["formal_provenance_sha256"],
        "units": unit_projection,
    }


def derive_formal_attempt_summary(attempt_root: Path, attempt_index: int) -> dict[str, Any]:
    """Re-derive one formal attempt from packaged primary artifacts."""
    expected_top = {
        "attempt_manifest.json",
        "preflight.json",
        "execution-evidence.json",
        "formal-provenance.json",
        "evaluation",
    }
    if not attempt_root.is_dir() or {path.name for path in attempt_root.iterdir()} != expected_top:
        raise ValueError("FIXED_TARGET_FORMAL_ATTEMPT_TOP_LEVEL_COVERAGE_MISMATCH")
    manifest = _read(attempt_root / "attempt_manifest.json")
    ft.validate_attempt_manifest(attempt_root, manifest, attempt_index)
    preflight = _read(attempt_root / "preflight.json")
    execution = _read(attempt_root / "execution-evidence.json")
    ft.validate_execution_evidence_manifest(execution)
    contract = preflight["target_contract"]
    _validate_formal_target_contract(contract)
    target_hash = ft.target_contract_sha256(contract)
    if preflight["target_contract_sha256"] != target_hash:
        raise ValueError("FIXED_TARGET_FORMAL_TARGET_HASH_MISMATCH")
    runtime = preflight["runtime_contract"]
    ft.validate_runtime_contract(runtime, contract)
    runtime_hash = ft.runtime_contract_sha256(runtime)
    if preflight["runtime_contract_sha256"] != runtime_hash:
        raise ValueError("FIXED_TARGET_FORMAL_RUNTIME_HASH_MISMATCH")
    observation = preflight["target_observation"]
    ft.validate_observation_against_contract(contract, observation)
    inventory = ft.validate_sharded_source_inventory(
        preflight["source_inventory"],
        implementation_commit=preflight["implementation_commit"],
    )
    ft.require_trusted_implementation_commit(preflight["implementation_commit"])
    _prepare_contract_environment(contract)

    evaluation_root = attempt_root / "evaluation"
    evaluation_config = _read(evaluation_root / "evaluation-config.json")
    training_configs = [
        _read(path)
        for path in sorted(evaluation_root.glob("training-runs/*/seed-*/training-config.json"))
    ]
    attempt_binding = {
        "implementation_commit": preflight["implementation_commit"],
        "target_contract_sha256": target_hash,
        "runtime_contract_sha256": runtime_hash,
        "target_observation_sha256": observation["observation_sha256"],
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "training_execution_mode": evaluation_config.get("training_execution_mode"),
        "observed_optimizer_foreach": execution["observed_optimizer_foreach"],
        "observed_optimizer_fused": execution["observed_optimizer_fused"],
    }
    acceptance_binding = {
        "implementation_commit": preflight["implementation_commit"],
        "target_contract_sha256": target_hash,
        "runtime_contract_sha256": runtime_hash,
    }
    ft.validate_execution_binding_contract(
        execution,
        acceptance=acceptance_binding,
        attempt=attempt_binding,
        preflight=preflight,
        evaluation_config=evaluation_config,
        target_observation=observation,
        training_configs=training_configs,
    )
    if evaluation_config.get("training_execution_mode") != FORMAL_TRAINING_MODE:
        raise ValueError("FIXED_TARGET_FORMAL_TRAINING_MODE_MISMATCH")
    if evaluation_config.get("execution_context") != FORMAL_EXECUTION_CONTEXT:
        raise ValueError("FIXED_TARGET_FORMAL_EXECUTION_CONTEXT_REQUIRED")
    if execution.get("execution_context") != FORMAL_EXECUTION_CONTEXT:
        raise ValueError("FIXED_TARGET_FORMAL_EXECUTION_CONTEXT_REQUIRED")
    if execution.get("evaluator_version") != FORMAL_EVALUATOR_VERSION:
        raise ValueError("FIXED_TARGET_FORMAL_EVALUATOR_MISMATCH")
    claims = ft._derive_claim_identities(evaluation_root)
    observed_foreach, observed_fused = ft._derive_optimizer_execution(evaluation_root)
    if observed_foreach is not False or observed_fused is not False:
        raise ValueError("FIXED_TARGET_OPTIMIZER_EXECUTION_MISMATCH")
    if (
        execution["observed_optimizer_foreach"] is not False
        or execution["observed_optimizer_fused"] is not False
    ):
        raise ValueError("FIXED_TARGET_OPTIMIZER_EXECUTION_MISMATCH")
    if execution["evaluation_root_identity"] != claims["replay_hash"]:
        raise ValueError("FIXED_TARGET_EVALUATION_ROOT_IDENTITY_MISMATCH")
    provenance = _validate_formal_provenance(
        attempt_root,
        attempt_index,
        preflight=preflight,
        execution=execution,
        evaluation_config=evaluation_config,
    )
    return {
        "attempt_index": attempt_index,
        "attempt_manifest_sha256": manifest["attempt_manifest_sha256"],
        "workflow_run_id": provenance["workflow_run_id"],
        "job_id": provenance["job_id"],
        "workflow_sha": provenance["workflow_sha"],
        "execution_implementation_commit": preflight["implementation_commit"],
        "scientific_parent_implementation_commit": execution[
            "scientific_parent_implementation_commit"
        ],
        "target_contract": copy.deepcopy(contract),
        "target_contract_sha256": target_hash,
        "runtime_contract": copy.deepcopy(runtime),
        "runtime_contract_sha256": runtime_hash,
        "target_observation_sha256": observation["observation_sha256"],
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "execution_evidence_sha256": execution["execution_evidence_sha256"],
        "observed_optimizer_foreach": observed_foreach,
        "observed_optimizer_fused": observed_fused,
        "claim_identities": copy.deepcopy(claims),
        "canonical_result_identity": ft.canonical_result_identity(claims),
        "training_execution_mode": FORMAL_TRAINING_MODE,
        "successful_full_evaluation": True,
        "result": "PASS",
        "attempt_identity_sha256": provenance["attempt_identity_sha256"],
        "formal_provenance_sha256": provenance["formal_provenance_sha256"],
        "units": copy.deepcopy(provenance["units"]),
        "execution_provenance": {
            "evaluator_version": execution["evaluator_version"],
            "evaluator_source_sha256": execution["evaluator_source_sha256"],
            "execution_topology": execution["execution_topology"],
            "scientific_policy_sha256": execution["scientific_policy_sha256"],
            "requirements_lock_sha256": execution["requirements_lock_sha256"],
            "scientific_parent_implementation_commit": execution[
                "scientific_parent_implementation_commit"
            ],
        },
    }


def _validate_cross_attempt_independence(summaries: list[dict[str, Any]]) -> None:
    if len(summaries) != 3:
        raise ValueError("FIXED_TARGET_FORMAL_ATTEMPT_COUNT_MISMATCH")
    if [summary["attempt_index"] for summary in summaries] != [1, 2, 3]:
        raise ValueError("FIXED_TARGET_FORMAL_ATTEMPT_ORDER_MISMATCH")
    attempt_ids = [summary["attempt_identity_sha256"] for summary in summaries]
    if len(set(attempt_ids)) != 3:
        raise ValueError("FIXED_TARGET_FORMAL_ATTEMPT_REUSE")
    attempt_manifests = [summary["attempt_manifest_sha256"] for summary in summaries]
    if len(set(attempt_manifests)) != 3:
        raise ValueError("FIXED_TARGET_FORMAL_ATTEMPT_ARTIFACT_REUSE")
    provenance_ids = [summary["formal_provenance_sha256"] for summary in summaries]
    if len(set(provenance_ids)) != 3:
        raise ValueError("FIXED_TARGET_FORMAL_ATTEMPT_REUSE")

    seen_units: set[str] = set()
    seen_checkpoint_lineages: set[str] = set()
    for summary in summaries:
        units = summary["units"]
        if len(units) != 9:
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_COVERAGE_MISMATCH")
        unit_ids = {unit["unit_manifest_sha256"] for unit in units}
        if len(unit_ids) != 9 or seen_units & unit_ids:
            raise ValueError("FIXED_TARGET_FORMAL_CROSS_ATTEMPT_UNIT_REUSE")
        if any(
            unit["attempt_identity_sha256"] != summary["attempt_identity_sha256"]
            for unit in units
        ):
            raise ValueError("FIXED_TARGET_FORMAL_UNIT_ATTEMPT_BINDING_MISMATCH")
        lineage_ids = {unit["checkpoint_lineage_sha256"] for unit in units}
        if len(lineage_ids) != 9 or seen_checkpoint_lineages & lineage_ids:
            raise ValueError("FIXED_TARGET_FORMAL_CROSS_ATTEMPT_CHECKPOINT_REUSE")
        seen_units |= unit_ids
        seen_checkpoint_lineages |= lineage_ids
    if len(seen_units) != 27 or len(seen_checkpoint_lineages) != 27:
        raise ValueError("FIXED_TARGET_FORMAL_CROSS_ATTEMPT_UNIT_REUSE")

    claims = [summary["claim_identities"] for summary in summaries]
    if any(claim != claims[0] for claim in claims[1:]):
        raise ValueError("FIXED_TARGET_FORMAL_CROSS_ATTEMPT_CLAIM_MISMATCH")
    stable_fields = (
        "workflow_run_id",
        "job_id",
        "workflow_sha",
        "execution_implementation_commit",
        "scientific_parent_implementation_commit",
        "target_contract_sha256",
        "runtime_contract_sha256",
        "target_observation_sha256",
        "source_inventory_sha256",
        "observed_optimizer_foreach",
        "observed_optimizer_fused",
        "training_execution_mode",
        "execution_provenance",
    )
    first = summaries[0]
    for summary in summaries[1:]:
        for field in stable_fields:
            if summary[field] != first[field]:
                raise ValueError(f"FIXED_TARGET_FORMAL_CROSS_ATTEMPT_PROVENANCE_MISMATCH:{field}")


def build_acceptance_record(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the canonical version-1.1 envelope from re-derived attempt provenance."""
    _validate_cross_attempt_independence(summaries)
    first = summaries[0]
    workflow_run_id = first["workflow_run_id"]
    job_id = first["job_id"]
    workflow_sha = first["workflow_sha"]
    execution_implementation_commit = first["execution_implementation_commit"]
    if workflow_run_id < 1 or job_id < 1:
        raise ValueError("FIXED_TARGET_FORMAL_WORKFLOW_PROVENANCE_INVALID")
    if workflow_sha != execution_implementation_commit:
        raise ValueError("FIXED_TARGET_WORKFLOW_IMPLEMENTATION_MISMATCH")

    attempts = []
    for summary in summaries:
        attempts.append(
            {
                "attempt_index": summary["attempt_index"],
                "attempt_identity_sha256": summary["attempt_identity_sha256"],
                "attempt_manifest_sha256": summary["attempt_manifest_sha256"],
                "execution_context": FORMAL_EXECUTION_CONTEXT,
                "target_observation_sha256": summary["target_observation_sha256"],
                "source_inventory_sha256": summary["source_inventory_sha256"],
                "execution_evidence_sha256": summary["execution_evidence_sha256"],
                "formal_provenance_sha256": summary["formal_provenance_sha256"],
                "observed_optimizer_foreach": False,
                "observed_optimizer_fused": False,
                "training_execution_mode": FORMAL_TRAINING_MODE,
                "units": copy.deepcopy(summary["units"]),
                "claim_identities": copy.deepcopy(summary["claim_identities"]),
                "canonical_result_identity": summary["canonical_result_identity"],
                "successful_full_evaluation": True,
                "result": "PASS",
            }
        )
    value: dict[str, Any] = {
        "acceptance_version": FORMAL_ACCEPTANCE_VERSION,
        "status": "FIXED_TARGET_ACCEPTED",
        "workflow_run_id": workflow_run_id,
        "job_id": job_id,
        "workflow_sha": workflow_sha,
        "execution_implementation_commit": execution_implementation_commit,
        "scientific_parent_implementation_commit": ft.HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "target_contract": copy.deepcopy(first["target_contract"]),
        "target_contract_sha256": first["target_contract_sha256"],
        "runtime_contract": copy.deepcopy(first["runtime_contract"]),
        "runtime_contract_sha256": first["runtime_contract_sha256"],
        "attempts": attempts,
        "cross_attempt_comparison": {"status": "PASS", "mismatches": []},
        "accepted": True,
        "acceptance_identity": "",
    }
    value["acceptance_identity"] = ft.acceptance_identity_sha256(value)
    ft._validate_schema(
        "fixed_target_acceptance_v1_1.schema.json",
        value,
        "FIXED_TARGET_FORMAL_ACCEPTANCE_SCHEMA_INVALID",
    )
    return value


def _validate_formal_bundle_semantics(
    root: Path,
    acceptance: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Internal semantic derivation; final verdict is issued only by the caller in ft."""
    root = Path(root)
    expected_root = {"acceptance.json", "attempt-1", "attempt-2", "attempt-3"}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_root:
        raise ValueError("FIXED_TARGET_BUNDLE_ROOT_COVERAGE_MISMATCH")
    ft._validate_schema(
        "fixed_target_acceptance_v1_1.schema.json",
        acceptance,
        "FIXED_TARGET_FORMAL_ACCEPTANCE_SCHEMA_INVALID",
    )
    if acceptance["acceptance_version"] != FORMAL_ACCEPTANCE_VERSION:
        raise ValueError("FIXED_TARGET_FORMAL_ACCEPTANCE_VERSION_MISMATCH")
    if acceptance["acceptance_identity"] != ft.acceptance_identity_sha256(acceptance):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_IDENTITY_MISMATCH")
    if acceptance["workflow_sha"] != acceptance["execution_implementation_commit"]:
        raise ValueError("FIXED_TARGET_WORKFLOW_IMPLEMENTATION_MISMATCH")
    if acceptance["scientific_parent_implementation_commit"] != (
        ft.HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT
    ):
        raise ValueError("FIXED_TARGET_SCIENTIFIC_PARENT_MISMATCH")
    ft.require_trusted_implementation_commit(acceptance["execution_implementation_commit"])

    summaries = [
        derive_formal_attempt_summary(root / f"attempt-{index}", index)
        for index in range(1, 4)
    ]
    _validate_cross_attempt_independence(summaries)
    expected = build_acceptance_record(summaries)
    if acceptance != expected:
        raise ValueError("FIXED_TARGET_FORMAL_ACCEPTANCE_BINDING_MISMATCH")
    return expected, summaries


def _validate_formal_acceptance_bundle_v1_1(
    root: Path,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Internal version dispatch target for the authoritative bundle validator."""
    validated, summaries = _validate_formal_bundle_semantics(root, acceptance)
    return {
        "valid": True,
        "accepted": True,
        "status": validated["status"],
        "acceptance_version": FORMAL_ACCEPTANCE_VERSION,
        "implementation_commit": validated["execution_implementation_commit"],
        "attempt_count": len(summaries),
        "unit_count": sum(len(summary["units"]) for summary in summaries),
        "runtime_1_1_execution_validation": "ENABLED_AND_VALIDATED",
    }
