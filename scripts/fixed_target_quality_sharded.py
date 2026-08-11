"""Runtime/1.1 sharded execution for the frozen quality-v0.1 science."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import shutil
import uuid
from pathlib import Path

import torch
from jsonschema import Draft202012Validator

from planner_toy import quality as q
from scripts.fixed_target_contract import (
    HISTORICAL_QUALITY_DATASET_HASH,
    HISTORICAL_QUALITY_EVALUATOR_VERSION,
    HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
    build_runtime_contract,
    build_scientific_policy,
    canonical_bytes,
    collect_target_observation,
    observation_sha256,
    runtime_contract_sha256,
    sha256_bytes,
    sha256_value,
    target_contract_sha256,
    validate_observation_against_contract,
    validate_runtime_contract,
    validate_scientific_policy,
    validate_target_contract,
)

ATTEMPT_VERSION = "toy-quality-fixed-target-sharded-attempt/1.0"
UNIT_VERSION = "toy-quality-fixed-target-sharded-unit/1.0"
EVALUATOR_VERSION = "development-quality-evaluation/0.1-runtime1.1-sharded/1.0"
EXECUTION_MODE = "TRAINED_IN_ATTEMPT_SHARDED"
VARIANTS = ("A2", "A3", "A4")
SEEDS = (17, 29, 43)
ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUN_FILES = {
    "checkpoint-manifest.json",
    "initialization.pt",
    "optimizer-evidence.json",
    "optimizer-state.pt",
    "trained.pt",
    "training-config.json",
    "training-report.json",
}


def execution_source_inventory() -> dict:
    paths = (
        "scripts/fixed_target_quality_sharded.py",
        "scripts/run_fixed_target_quality_evaluation.py",
        "planner_toy/schemas/fixed_target_quality_unit.schema.json",
    )
    return {
        "scientific_parent": q.source_identity(),
        "execution_files": [
            {"path": path, "sha256": sha256_bytes((ROOT / path).read_bytes())} for path in paths
        ],
    }


def collect_runtime11_observation(contract: dict) -> dict:
    """Collect after the historical module has configured torch once."""
    cpu = {}
    for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            cpu[key.strip()] = value.strip()
    build_number, build_date = platform.python_build()
    observation = {
        "observation_version": "toy-quality-fixed-cpu-target-observation/1.0",
        "target_contract_version": contract["target_contract_version"],
        "target_contract_sha256": target_contract_sha256(contract),
        "os": platform.system(),
        "os_version": contract["os_version"],
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
        "cpu_vendor": cpu["vendor_id"],
        "cpu_family": cpu["cpu family"],
        "cpu_model": cpu["model"],
        "cpu_stepping": cpu["stepping"],
        "cpu_model_name": cpu["model name"],
        "microcode": cpu["microcode"],
        "cpu_flags": sorted(set(cpu["flags"].split())),
        "logical_cpu_count": os.cpu_count(),
        "runner_type": "self-hosted-dedicated",
        "runner_image": os.environ.get("FIXED_TARGET_RUNNER_IMAGE", ""),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": [build_number, build_date],
        "python_compiler": platform.python_compiler(),
        "pip_version": importlib.metadata.version("pip"),
        "torch_version": torch.__version__,
        "torch_build_configuration_sha256": sha256_bytes(torch.__config__.show().encode()),
        "mkl_available": torch.backends.mkl.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "actual_atten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "MKL_CBWR": os.environ.get("MKL_CBWR"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "observation_sha256": "",
    }
    observation["observation_sha256"] = observation_sha256(observation)
    validate_observation_against_contract(contract, observation)
    return observation


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SHARDED_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    )


def _without(value: dict, field: str) -> dict:
    return {key: item for key, item in value.items() if key != field}


def attempt_identity(descriptor: dict) -> str:
    # The nonce binds units within an attempt, while this identity is intentionally
    # excluded from numerical claim comparison between independently fresh attempts.
    return sha256_value(_without(descriptor, "attempt_identity_sha256"))


def unit_identity(manifest: dict) -> str:
    return sha256_value(_without(manifest, "unit_manifest_sha256"))


def initialize_attempt(
    root: Path,
    target_contract: dict,
    nonce: str | None = None,
    observation: dict | None = None,
) -> dict:
    if root.exists() and any(root.iterdir()):
        raise ValueError("SHARDED_ATTEMPT_DIRECTORY_NOT_EMPTY")
    validate_target_contract(target_contract)
    runtime = build_runtime_contract(target_contract)
    observation = observation or collect_target_observation(target_contract)
    policy = build_scientific_policy()
    inventory = execution_source_inventory()
    dataset = q.generate(17)
    descriptor = {
        "attempt_version": ATTEMPT_VERSION,
        "attempt_nonce": nonce or str(uuid.uuid4()),
        "implementation_commit": HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "execution_evaluator": EVALUATOR_VERSION,
        "scientific_policy_sha256": policy["scientific_policy_sha256"],
        "target_contract_sha256": target_contract_sha256(target_contract),
        "runtime_contract_sha256": runtime_contract_sha256(runtime),
        "target_observation_sha256": observation["observation_sha256"],
        "source_inventory_sha256": sha256_value(inventory),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "ordered_train_task_ids": sorted(row["task_id"] for row in dataset["train"]),
        "ordered_eval_task_ids": sorted(row["task_id"] for row in dataset["validation"]),
        "attempt_identity_sha256": "",
    }
    descriptor["attempt_identity_sha256"] = attempt_identity(descriptor)
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "attempt-id.json", descriptor)
    _write(root / "target-contract.json", target_contract)
    _write(root / "runtime-contract.json", runtime)
    _write(root / "scientific-policy.json", policy)
    _write(root / "source-inventory.json", inventory)
    _write(root / "initial-observation.json", observation)
    return descriptor


def validate_attempt(root: Path) -> tuple[dict, dict]:
    attempt = _read(root / "attempt-id.json")
    if attempt.get("attempt_identity_sha256") != attempt_identity(attempt):
        raise ValueError("SHARDED_ATTEMPT_IDENTITY_MISMATCH")
    contract = _read(root / "target-contract.json")
    validate_target_contract(contract)
    runtime = _read(root / "runtime-contract.json")
    validate_runtime_contract(runtime, contract)
    if runtime != build_runtime_contract(contract):
        raise ValueError("SHARDED_RUNTIME_CONTRACT_DRIFT")
    policy = _read(root / "scientific-policy.json")
    validate_scientific_policy(policy)
    initial_observation = _read(root / "initial-observation.json")
    validate_observation_against_contract(contract, initial_observation)
    persisted_inventory = _read(root / "source-inventory.json")
    if persisted_inventory != execution_source_inventory():
        raise ValueError("SHARDED_EXECUTION_SOURCE_INVENTORY_DRIFT")
    expected = {
        "implementation_commit": HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "execution_evaluator": EVALUATOR_VERSION,
        "target_contract_sha256": target_contract_sha256(contract),
        "runtime_contract_sha256": runtime_contract_sha256(runtime),
        "scientific_policy_sha256": policy["scientific_policy_sha256"],
        "target_observation_sha256": initial_observation["observation_sha256"],
        "source_inventory_sha256": sha256_value(persisted_inventory),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "ordered_train_task_ids": ["bw-00000001", "bw-00000002", "bw-00000003"],
        "ordered_eval_task_ids": ["bw-00000004", "bw-00000005"],
    }
    for field, value in expected.items():
        if attempt.get(field) != value:
            raise ValueError(f"SHARDED_ATTEMPT_CONTRACT_MISMATCH:{field}")
    return attempt, contract


def _train_runtime11(rows: list[dict], variant: str, seed: int, output: Path, dataset_hash: str):
    """Historical _train semantics with only runtime/1.1 AdamW flags changed."""
    model = q.LockedPlanner(seed, variant).cpu()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "initialization.pt")
    optimizer_named = q._optimizer_named_parameters(model)
    active, dormant = q._optimizer_parameter_policy(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in optimizer_named],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    validate_runtime11_optimizer(optimizer)
    for _ in range(q.EPOCHS):
        for row in sorted(rows, key=lambda item: item["task_id"]):
            action, arg1, arg2 = q.labels(row)
            valid = len(row["oracle_work_plan"])
            target = q.targets(row)
            shifted = torch.cat([torch.zeros_like(target[:, :1]), target[:, :-1]], 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                q.canonical_task_encoding(row),
                action,
                arg1,
                arg2,
                semantic_feedback=shifted if variant in {"A3", "A4"} else None,
            )
            flat = action[:, :valid].flatten()
            loss = torch.nn.functional.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
            one = flat != q.ACTIONS["END"]
            two = (flat == q.ACTIONS["UNSTACK"]) | (flat == q.ACTIONS["STACK"])
            if one.any():
                loss += torch.nn.functional.cross_entropy(
                    logits.arg1[:, :valid].flatten(0, 1)[one], arg1[:, :valid].flatten()[one]
                )
            if two.any():
                loss += torch.nn.functional.cross_entropy(
                    logits.arg2[:, :valid].flatten(0, 1)[two], arg2[:, :valid].flatten()[two]
                )
            if variant in {"A3", "A4"}:
                loss += (1 - (logits.z_semantic[:, :valid] * target[:, :valid]).sum(-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    # Reuse the frozen serializer/manifest builder by giving it the already-computed
    # objects through a small local equivalent of its persistence tail.
    return _persist_runtime11(
        model, optimizer, output, variant, seed, rows, dataset_hash, active, dormant
    )


def _persist_runtime11(
    model, optimizer, output, variant, seed, rows, dataset_hash, active, dormant
):
    from planner_toy.numeric_identity import (
        canonical_state_dict_sha256,
        canonical_torch_object_sha256,
        exact_torch_object_sha256,
    )
    from planner_toy.training import state_dict_sha256

    initial = torch.load(output / "initialization.pt", map_location="cpu", weights_only=True)
    torch.save(model.state_dict(), output / "trained.pt")
    state = optimizer.state_dict()
    torch.save(state, output / "optimizer-state.pt")
    optimizer_evidence = {
        "schema_version": "toy-quality-optimizer-evidence/0.1",
        "torch_object_encoding_version": q.TORCH_OBJECT_ENCODING_VERSION,
        "optimizer_state_sha256": exact_torch_object_sha256(state),
        "canonical_optimizer_state_sha256": canonical_torch_object_sha256(state),
        "active_parameter_names": active,
        "parameter_group_count": 1,
        "update_count": 9,
    }
    _write(output / "optimizer-evidence.json", optimizer_evidence)
    identity = {
        "architecture_stage": q.MAPPING[variant][0],
        "implementation_variant": variant,
        "experimental_arm": q.MAPPING[variant][1],
        "target_type": q.MAPPING[variant][2],
    }
    config = {
        "schema_version": "toy-quality-training-config/0.1",
        "variant_identity": identity,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "train_task_ids": [r["task_id"] for r in rows],
        "epochs": 3,
        "updates": 9,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "checkpoint_policy": "final_epoch_only_no_heldout_selection",
        "inventory_hash": q._file_hash(
            q.ROOT / "docs/architecture/planner_module_inventory_v1.yaml"
        ),
        "task_encoding_hash": q._file_hash(q.ROOT / "docs/architecture/task_encoding_v1.yaml"),
    }
    _write(output / "training-config.json", config)
    _write(
        output / "training-report.json",
        {
            "schema_version": "toy-quality-training-report/0.1",
            "updates": 9,
            "final_checkpoint_selected_without_heldout": True,
        },
    )
    manifest = {
        "schema_version": "toy-quality-checkpoint/0.1",
        "variant_identity": identity,
        "torch_object_encoding_version": q.TORCH_OBJECT_ENCODING_VERSION,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "train_task_ids": config["train_task_ids"],
        "epochs": 3,
        "updates": 9,
        "config_hash": q._file_hash(output / "training-config.json"),
        "initialization_path": "initialization.pt",
        "initialization_file_sha256": q._file_hash(output / "initialization.pt"),
        "initialization_state_dict_sha256": state_dict_sha256(initial),
        "canonical_initialization_state_dict_sha256": canonical_state_dict_sha256(initial),
        "trained_path": "trained.pt",
        "trained_file_sha256": q._file_hash(output / "trained.pt"),
        "trained_state_dict_sha256": state_dict_sha256(model.state_dict()),
        "canonical_trained_state_dict_sha256": canonical_state_dict_sha256(model.state_dict()),
        "optimizer_state_path": "optimizer-state.pt",
        "optimizer_state_file_sha256": q._file_hash(output / "optimizer-state.pt"),
        "optimizer_state_sha256": optimizer_evidence["optimizer_state_sha256"],
        "canonical_optimizer_state_sha256": optimizer_evidence["canonical_optimizer_state_sha256"],
        "optimizer_evidence_path": "optimizer-evidence.json",
        "optimizer_evidence_file_sha256": q._file_hash(output / "optimizer-evidence.json"),
        "training_report_path": "training-report.json",
        "training_report_file_sha256": q._file_hash(output / "training-report.json"),
        "active_parameter_names": active,
        "dormant_parameter_names": dormant,
        "training_execution_mode": EXECUTION_MODE,
        "checkpoint_origin_run_hash": None,
        "reuse_source_manifest_hash": None,
        "deterministic_training_replay_status": "CANONICAL_DETERMINISTIC",
    }
    _write(output / "checkpoint-manifest.json", manifest)
    return model, manifest


def validate_runtime11_optimizer(optimizer) -> None:
    if optimizer.defaults.get("foreach") is not False:
        raise ValueError("RUNTIME_1_1_OPTIMIZER_FOREACH_REQUIRED_FALSE")
    if optimizer.defaults.get("fused") is not False:
        raise ValueError("RUNTIME_1_1_OPTIMIZER_FUSED_REQUIRED_FALSE")


def validate_runtime11_optimizer_state(state: dict, model, expected_updates: int) -> None:
    """Apply the frozen structural validator plus runtime/1.1 option requirements."""
    groups = state.get("param_groups") if isinstance(state, dict) else None
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("RUNTIME_1_1_OPTIMIZER_PARAMETER_GROUP_MISMATCH")
    if groups[0].get("foreach") is not False:
        raise ValueError("RUNTIME_1_1_OPTIMIZER_FOREACH_REQUIRED_FALSE")
    if groups[0].get("fused") is not False:
        raise ValueError("RUNTIME_1_1_OPTIMIZER_FUSED_REQUIRED_FALSE")
    historical_shape = copy.deepcopy(state)
    historical_shape["param_groups"][0]["foreach"] = None
    historical_shape["param_groups"][0]["fused"] = None
    q._validate_optimizer_state(historical_shape, model, expected_updates)


def run_unit(root: Path, variant: str, seed: int, observation: dict | None = None) -> dict:
    attempt, contract = validate_attempt(root)
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("SHARDED_UNIT_NOT_IN_POLICY")
    output = root / "units" / variant / f"seed-{seed}"
    if output.exists():
        raise ValueError("SHARDED_UNIT_ALREADY_EXISTS")
    observation = observation or collect_target_observation(contract)
    if observation["observation_sha256"] != attempt["target_observation_sha256"]:
        raise ValueError("SHARDED_UNIT_TARGET_OBSERVATION_DRIFT")
    dataset = q.generate(17)
    train = sorted(dataset["train"], key=lambda row: row["task_id"])
    evaluation = sorted(dataset["validation"], key=lambda row: row["task_id"])
    model, checkpoint = _train_runtime11(
        train, variant, seed, output / "training", dataset["dataset_hash"]
    )
    rows = []
    for row in evaluation:
        result = q._evaluate(
            row, variant, seed, model, checkpoint, output / "evidence" / row["task_id"]
        )
        result["evidence_root"] = f"evidence/{variant}/seed-{seed}/{row['task_id']}"
        rows.append(result)
    # Re-read all sealed attempt inputs and observe the host again after execution;
    # a unit is invalid if orchestration files or claim-bearing host fields changed
    # while its optimizer trajectory was running.
    final_attempt, _ = validate_attempt(root)
    if final_attempt != attempt:
        raise ValueError("SHARDED_ATTEMPT_CHANGED_DURING_UNIT")
    final_observation = collect_runtime11_observation(contract)
    if final_observation["observation_sha256"] != observation["observation_sha256"]:
        raise ValueError("SHARDED_TARGET_CHANGED_DURING_UNIT")
    (output / "task-results.jsonl").write_bytes(
        b"".join(canonical_bytes(row) + b"\n" for row in rows)
    )
    manifest = {
        "unit_evidence_version": UNIT_VERSION,
        "attempt_identity_sha256": attempt["attempt_identity_sha256"],
        "implementation_commit": attempt["implementation_commit"],
        "variant": variant,
        "seed": seed,
        "dataset_hash": dataset["dataset_hash"],
        "ordered_train_task_ids": attempt["ordered_train_task_ids"],
        "ordered_eval_task_ids": attempt["ordered_eval_task_ids"],
        "epochs": 3,
        "updates": 9,
        "training_execution_mode": EXECUTION_MODE,
        "optimizer": {
            "class": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "observed_foreach": False,
            "observed_fused": False,
        },
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation_sha256": observation["observation_sha256"],
        "source_inventory_sha256": attempt["source_inventory_sha256"],
        "observation": observation,
        "checkpoint_manifest_sha256": q._file_hash(
            output / "training" / "checkpoint-manifest.json"
        ),
        "task_results_sha256": q._file_hash(output / "task-results.jsonl"),
        "unit_manifest_sha256": "",
    }
    manifest["unit_manifest_sha256"] = unit_identity(manifest)
    validate_unit_manifest(manifest, attempt, contract)
    _write(output / "unit-manifest.json", manifest)
    return manifest


def validate_unit_manifest(manifest: dict, attempt: dict, contract: dict) -> None:
    schema = _read(ROOT / "planner_toy/schemas/fixed_target_quality_unit.schema.json")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.path)
    )
    if errors:
        raise ValueError(f"SHARDED_UNIT_SCHEMA_INVALID:{errors[0].message}")
    if manifest["unit_manifest_sha256"] != unit_identity(manifest):
        raise ValueError("SHARDED_UNIT_MANIFEST_STALE")
    for field in (
        "attempt_identity_sha256",
        "implementation_commit",
        "runtime_contract_sha256",
        "target_observation_sha256",
        "source_inventory_sha256",
    ):
        if manifest[field] != attempt[field]:
            raise ValueError(f"SHARDED_UNIT_BINDING_MISMATCH:{field}")
    validate_observation_against_contract(contract, manifest["observation"])
    if observation_sha256(manifest["observation"]) != manifest["target_observation_sha256"]:
        raise ValueError("SHARDED_UNIT_OBSERVATION_STALE")


def validate_unit_artifacts(unit: Path, manifest: dict) -> None:
    expected_top_level = {"training", "evidence", "task-results.jsonl", "unit-manifest.json"}
    if {path.name for path in unit.iterdir()} != expected_top_level:
        raise ValueError("SHARDED_UNIT_TOP_LEVEL_COVERAGE_MISMATCH")
    training = unit / "training"
    if {path.name for path in training.iterdir()} != TRAINING_RUN_FILES:
        raise ValueError("SHARDED_UNIT_TRAINING_COVERAGE_MISMATCH")
    if (
        q._file_hash(training / "checkpoint-manifest.json")
        != manifest["checkpoint_manifest_sha256"]
    ):
        raise ValueError("SHARDED_UNIT_CHECKPOINT_MANIFEST_STALE")
    if q._file_hash(unit / "task-results.jsonl") != manifest["task_results_sha256"]:
        raise ValueError("SHARDED_UNIT_TASK_RESULTS_STALE")
    rows = [json.loads(line) for line in (unit / "task-results.jsonl").read_text().splitlines()]
    if [(row["variant"], row["seed"], row["task_id"]) for row in rows] != [
        (manifest["variant"], manifest["seed"], task_id)
        for task_id in manifest["ordered_eval_task_ids"]
    ]:
        raise ValueError("SHARDED_UNIT_TASK_RESULT_COVERAGE_MISMATCH")
    evidence = unit / "evidence"
    if {path.name for path in evidence.iterdir()} != set(manifest["ordered_eval_task_ids"]):
        raise ValueError("SHARDED_UNIT_EVIDENCE_TASK_COVERAGE_MISMATCH")
    for row in rows:
        required = {
            "planner-request.json",
            "attempt-log.jsonl",
            "episode-plan-manifest.json",
            "episode-log.json",
            "evaluation-result.json",
        }
        if row["plan_generation_success"]:
            required.add("work-plan.json")
        if manifest["variant"] in {"A3", "A4"}:
            required.update(
                {"semantic-trace.json", "semantic-latents.f32", "projected-feedback.f32"}
            )
        task_root = evidence / row["task_id"]
        if {path.name for path in task_root.iterdir()} != required:
            raise ValueError("SHARDED_UNIT_EVIDENCE_FILE_COVERAGE_MISMATCH")
    checkpoint = _read(training / "checkpoint-manifest.json")
    if (
        checkpoint.get("training_execution_mode") != EXECUTION_MODE
        or checkpoint.get("checkpoint_origin_run_hash") is not None
        or checkpoint.get("reuse_source_manifest_hash") is not None
    ):
        raise ValueError("SHARDED_UNIT_REUSED_LINEAGE_REJECTED")
    if (
        checkpoint.get("variant_identity", {}).get("implementation_variant") != manifest["variant"]
        or checkpoint.get("seed") != manifest["seed"]
    ):
        raise ValueError("SHARDED_UNIT_CHECKPOINT_IDENTITY_MISMATCH")
    optimizer_state = torch.load(
        training / "optimizer-state.pt", map_location="cpu", weights_only=True
    )
    model = q.LockedPlanner(manifest["seed"], manifest["variant"]).cpu()
    validate_runtime11_optimizer_state(optimizer_state, model, manifest["updates"])


def assemble(root: Path) -> dict:
    attempt, contract = validate_attempt(root)
    expected = {(v, s) for v in VARIANTS for s in SEEDS}
    unit_root = root / "units"
    expected_variant_directories = set(VARIANTS)
    if {path.name for path in unit_root.iterdir()} != expected_variant_directories:
        raise ValueError("SHARDED_EXTRA_OR_MISSING_VARIANT_DIRECTORY")
    for variant in VARIANTS:
        expected_seed_directories = {f"seed-{seed}" for seed in SEEDS}
        if {path.name for path in (unit_root / variant).iterdir()} != expected_seed_directories:
            raise ValueError("SHARDED_EXTRA_OR_MISSING_SEED_DIRECTORY")
    found = {}
    for path in sorted((root / "units").glob("*/seed-*/unit-manifest.json")):
        manifest = _read(path)
        validate_unit_manifest(manifest, attempt, contract)
        validate_unit_artifacts(path.parent, manifest)
        key = (manifest["variant"], manifest["seed"])
        if key in found:
            raise ValueError("SHARDED_DUPLICATE_UNIT")
        found[key] = (path.parent, manifest)
    if set(found) != expected:
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected)
        raise ValueError(f"SHARDED_UNIT_COVERAGE_MISMATCH:{missing}:{extra}")
    output = root / "evaluation"
    shutil.rmtree(output, ignore_errors=True)
    rows = []
    for variant, seed in sorted(expected):
        unit, _ = found[(variant, seed)]
        shutil.copytree(unit / "training", output / "training-runs" / variant / f"seed-{seed}")
        shutil.copytree(unit / "evidence", output / "evidence" / variant / f"seed-{seed}")
        rows += [
            json.loads(line) for line in (unit / "task-results.jsonl").read_text().splitlines()
        ]
    rows.sort(key=lambda row: (row["variant"], row["seed"], row["task_id"]))
    dataset = q.generate(17)
    config = {
        "schema_version": q.VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "variant_mapping": {
            v: {
                "architecture_stage": q.MAPPING[v][0],
                "implementation_variant": v,
                "experimental_arm": q.MAPPING[v][1],
                "target_type": q.MAPPING[v][2],
            }
            for v in VARIANTS
        },
        "train_task_ids": ["bw-00000001", "bw-00000002", "bw-00000003"],
        "eval_task_ids": ["bw-00000004", "bw-00000005"],
        "epochs": 3,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "checkpoint_policy": "final_epoch_only_no_heldout_selection",
        "training_execution_mode": EXECUTION_MODE,
        "dataset_manifest_hash": dataset["dataset_hash"],
        "attempt_identity_sha256": attempt["attempt_identity_sha256"],
    }
    _write(output / "evaluation-config.json", config)
    _write(
        output / "dataset-manifest.json",
        {
            "dataset_hash": dataset["dataset_hash"],
            "train_task_ids": ["bw-00000001", "bw-00000002", "bw-00000003"],
            "eval_task_ids": ["bw-00000004", "bw-00000005"],
        },
    )
    (output / "task-results.jsonl").write_bytes(
        b"".join(canonical_bytes(row) + b"\n" for row in rows)
    )
    per = {
        f"{v}/seed-{s}": q.summarize([r for r in rows if r["variant"] == v and r["seed"] == s])
        for v in VARIANTS
        for s in SEEDS
    }
    _write(output / "per-seed-summary.json", per)
    _write(output / "aggregate-summary.json", q.aggregate_summaries(per, rows, VARIANTS, SEEDS))
    _write(
        output / "paired-comparisons.json",
        [q.paired(rows, a, b) for a, b in (("A3", "A2"), ("A4", "A2"), ("A3", "A4"))],
    )
    (output / "human-readable-examples.md").write_text(
        q._examples(
            rows, sorted(dataset["validation"], key=lambda r: r["task_id"]), VARIANTS, SEEDS
        )
    )
    _write(output / "task-results-semantic.json", q.task_results_semantic_projection(rows))
    names = [
        "evaluation-config.json",
        "dataset-manifest.json",
        "task-results.jsonl",
        "task-results-semantic.json",
        "per-seed-summary.json",
        "aggregate-summary.json",
        "paired-comparisons.json",
        "human-readable-examples.md",
    ]
    manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "artifact_hashes": {n: q._file_hash(output / n) for n in names},
        "checkpoint_manifest_hashes": {
            str(p.relative_to(output)): q._file_hash(p)
            for p in sorted(output.glob("training-runs/*/seed-*/*"))
            if p.is_file()
        },
        "evidence_artifact_hashes": {
            str(p.relative_to(output)): q._file_hash(p)
            for p in sorted(output.glob("evidence/*/seed-*/*/*"))
        },
    }
    _write(output / "evaluation-manifest.json", manifest)
    payload = {
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "dataset_hash": HISTORICAL_QUALITY_DATASET_HASH,
        "training_execution_mode": EXECUTION_MODE,
        "task_results": q.task_results_semantic_projection(rows),
        "checkpoints": {
            k: v
            for k, v in manifest["checkpoint_manifest_hashes"].items()
            if k.endswith("checkpoint-manifest.json")
        },
    }
    _write(output / "canonical-semantic-payload.json", payload)
    manifest["artifact_hashes"]["canonical-semantic-payload.json"] = q._file_hash(
        output / "canonical-semantic-payload.json"
    )
    _write(output / "evaluation-manifest.json", manifest)
    replay = sha256_value(payload)
    (output / "replay-hash.txt").write_text(replay + "\n")
    return {"replay_hash": replay, "units": 9}
