"""Runtime/1.1 sharded execution for the frozen quality-v0.1 science."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path

import torch
from jsonschema import Draft202012Validator

from planner_toy import quality as q
from scripts.fixed_target_contract import (
    ATTEMPT_MANIFEST_VERSION,
    HISTORICAL_QUALITY_EVALUATOR_VERSION,
    HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
    attempt_manifest_sha256,
    build_runtime_contract,
    build_scientific_policy,
    canonical_bytes,
    execution_evidence_sha256,
    observation_sha256,
    require_trusted_implementation_commit,
    requirements_lock_sha256_at_commit,
    runtime_contract_sha256,
    sha256_bytes,
    sha256_value,
    sharded_source_inventory_at_commit,
    target_contract_sha256,
    validate_observation_against_contract,
    validate_runtime_contract,
    validate_scientific_policy,
    validate_target_contract,
)

ATTEMPT_VERSION = "toy-quality-fixed-target-sharded-attempt/1.0"
UNIT_VERSION = "toy-quality-fixed-target-sharded-unit/1.0"
SHARDED_SOURCE_INVENTORY_VERSION = "toy-quality-fixed-target-sharded-source-inventory/1.0"
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
SHARDED_REPLAY_VERSION = "toy-quality-sharded-canonical-replay/1.0"
SHARDED_STABLE_REPLAY_FIELDS = {
    "schema_version",
    "evaluator_version",
    "scientific_parent_evaluator",
    "variants",
    "seeds",
    "variant_mapping",
    "train_task_ids",
    "eval_task_ids",
    "epochs",
    "optimizer",
    "checkpoint_policy",
    "training_execution_mode",
    "dataset_manifest_hash",
    "implementation_commit",
    "scientific_parent_implementation_commit",
    "evaluator_source_sha256",
    "requirements_lock_sha256",
    "target_contract_sha256",
    "runtime_contract_sha256",
    "target_observation_sha256",
    "source_inventory_sha256",
    "scientific_policy_sha256",
    "execution_topology",
    "checkpoint_origin_run_hash",
    "reuse_source_manifest_hash",
}
SHARDED_OPERATIONAL_FIELDS = {"attempt_identity_sha256", "execution_context"}
QUALIFICATION_RUNTIME_FIELDS = (
    "python_implementation",
    "python_version",
    "python_build",
    "python_compiler",
    "torch_version",
    "torch_build_configuration_sha256",
    "ATEN_CPU_CAPABILITY",
    "actual_atten_cpu_capability",
    "MKL_CBWR",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "torch_num_threads",
    "torch_num_interop_threads",
    "mkldnn_enabled",
    "deterministic_algorithms",
    "deterministic_warn_only",
)


def qualification_runtime_profile(contract: dict, observation: dict) -> dict:
    """Bind qualification to numerical runtime controls, not physical target identity."""
    expected = {field: contract[field] for field in QUALIFICATION_RUNTIME_FIELDS}
    observed = {field: observation[field] for field in QUALIFICATION_RUNTIME_FIELDS}
    mismatches = sorted(field for field in expected if expected[field] != observed[field])
    profile = {
        "qualification_runtime_profile_version": (
            "toy-quality-runtime1.1-qualification-profile/1.0"
        ),
        "semantics": "NUMERICAL_RUNTIME_COMPATIBILITY_NOT_FORMAL_TARGET_IDENTITY",
        "expected": expected,
        "observed": observed,
        "optimizer_foreach": False,
        "optimizer_fused": False,
        "compatible": not mismatches,
        "mismatches": mismatches,
    }
    if mismatches:
        raise ValueError(f"SHARDED_QUALIFICATION_RUNTIME_MISMATCH:{','.join(mismatches)}")
    return profile


def sharded_config_semantic_projection(config: dict) -> dict:
    """Remove only the attempt-local binding from the closed sharded config."""
    expected = SHARDED_STABLE_REPLAY_FIELDS | SHARDED_OPERATIONAL_FIELDS
    if set(config) != expected:
        raise ValueError("SHARDED_REPLAY_CONFIG_FIELDS_MISMATCH")
    return {field: config[field] for field in sorted(SHARDED_STABLE_REPLAY_FIELDS)}


def sharded_canonical_replay_payload(root: Path, manifest: dict, rows: list[dict]) -> dict:
    """Versioned cross-attempt identity without nonce-derived orchestration data."""
    payload = q.canonical_replay_payload(root, manifest, rows)
    config = _read(root / "evaluation-config.json")
    payload["schema_version"] = SHARDED_REPLAY_VERSION
    payload["config_semantic_hash"] = sha256_value(
        sharded_config_semantic_projection(config)
    )
    return payload


def sharded_canonical_replay_hash(root: Path, manifest: dict, rows: list[dict]) -> str:
    return sha256_value(sharded_canonical_replay_payload(root, manifest, rows))


def checkout_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def execution_source_inventory(execution_commit: str) -> dict:
    """PR19 inventory extended with every claim-bearing runtime/1.1 source."""
    return sharded_source_inventory_at_commit(execution_commit)


def validate_checkout_source_inventory(inventory: dict) -> None:
    for entry in inventory["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or sha256_bytes(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"SHARDED_EXECUTION_SOURCE_TREE_DRIFT:{entry['path']}")


def _read_os_pretty_name() -> str:
    values = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("NAME") or platform.platform()


def _runner_identity(execution_context: str) -> tuple[str, str]:
    if execution_context == "qualification-only":
        return "codex-managed-container-qualification-only", os.environ.get(
            "CAAS_IMAGE_VERSION", "unidentified-codex-image"
        )
    image_path = Path("/etc/planning-model-runner-image-id")
    if not image_path.is_file():
        raise RuntimeError("FIXED_TARGET_TRUSTED_RUNNER_IMAGE_ID_MISSING")
    return "self-hosted-dedicated", image_path.read_text(encoding="utf-8").strip()


def collect_runtime11_observation(contract: dict, execution_context: str) -> dict:
    """Collect after the historical module has configured torch once."""
    processors = []
    for block in Path("/proc/cpuinfo").read_text().strip().split("\n\n"):
        cpu = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                cpu[key.strip()] = value.strip()
        if cpu:
            processors.append(cpu)
    claim_fields = (
        "vendor_id",
        "cpu family",
        "model",
        "stepping",
        "model name",
        "microcode",
        "flags",
    )
    if not processors or any(
        tuple(processor.get(field) for field in claim_fields)
        != tuple(processors[0].get(field) for field in claim_fields)
        for processor in processors[1:]
    ):
        raise RuntimeError("FIXED_TARGET_HETEROGENEOUS_CPU_IDENTITY")
    cpu = processors[0]
    build_number, build_date = platform.python_build()
    runner_type, runner_image = _runner_identity(execution_context)
    observation = {
        "observation_version": "toy-quality-fixed-cpu-target-observation/1.0",
        "target_contract_version": contract["target_contract_version"],
        "target_contract_sha256": target_contract_sha256(contract),
        "os": platform.system(),
        "os_version": _read_os_pretty_name(),
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
        "runner_type": runner_type,
        "runner_image": runner_image,
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
    if execution_context == "formal-fixed-target":
        validate_observation_against_contract(contract, observation)
    else:
        # Qualification records actual context without claiming formal target acceptance.
        for field in ("runner_type", "runner_image"):
            if observation[field] == contract[field]:
                raise ValueError(f"QUALIFICATION_CONTEXT_MUST_NOT_IMPERSONATE_TARGET:{field}")
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
    execution_implementation_commit: str,
    execution_context: str,
    nonce: str | None = None,
    observation: dict | None = None,
) -> dict:
    if root.exists() and any(root.iterdir()):
        raise ValueError("SHARDED_ATTEMPT_DIRECTORY_NOT_EMPTY")
    validate_target_contract(target_contract)
    runtime = build_runtime_contract(target_contract)
    if execution_context not in {"formal-fixed-target", "qualification-only"}:
        raise ValueError("SHARDED_EXECUTION_CONTEXT_INVALID")
    if execution_implementation_commit != checkout_commit():
        raise ValueError("SHARDED_EXECUTION_COMMIT_NOT_CHECKED_OUT")
    if execution_context == "formal-fixed-target":
        require_trusted_implementation_commit(execution_implementation_commit)
    live_observation = collect_runtime11_observation(target_contract, execution_context)
    if observation is not None and observation != live_observation:
        raise ValueError("SHARDED_SUPPLIED_OBSERVATION_NOT_LIVE")
    observation = live_observation
    qualification_profile = (
        qualification_runtime_profile(target_contract, observation)
        if execution_context == "qualification-only"
        else {
            "qualification_runtime_profile_version": (
                "toy-quality-runtime1.1-qualification-profile/1.0"
            ),
            "semantics": "NOT_APPLICABLE_FORMAL_FIXED_TARGET",
            "compatible": None,
        }
    )
    policy = build_scientific_policy()
    inventory = execution_source_inventory(execution_implementation_commit)
    validate_checkout_source_inventory(inventory)
    dataset = q.generate(17)
    descriptor = {
        "attempt_version": ATTEMPT_VERSION,
        "attempt_nonce": nonce or str(uuid.uuid4()),
        "scientific_parent_implementation_commit": HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "execution_implementation_commit": execution_implementation_commit,
        "execution_context": execution_context,
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "execution_evaluator": EVALUATOR_VERSION,
        "scientific_policy_sha256": policy["scientific_policy_sha256"],
        "target_contract_sha256": target_contract_sha256(target_contract),
        "runtime_contract_sha256": runtime_contract_sha256(runtime),
        "target_observation_sha256": observation["observation_sha256"],
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "qualification_runtime_profile_sha256": sha256_value(qualification_profile),
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
    _write(root / "qualification-runtime-profile.json", qualification_profile)
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
    qualification_profile = _read(root / "qualification-runtime-profile.json")
    if attempt["execution_context"] == "formal-fixed-target":
        require_trusted_implementation_commit(attempt["execution_implementation_commit"])
    live_observation = collect_runtime11_observation(contract, attempt["execution_context"])
    if initial_observation != live_observation:
        raise ValueError("SHARDED_INITIAL_OBSERVATION_NOT_LIVE")
    if attempt["execution_context"] == "formal-fixed-target":
        validate_observation_against_contract(contract, initial_observation)
        if qualification_profile != {
            "qualification_runtime_profile_version": (
                "toy-quality-runtime1.1-qualification-profile/1.0"
            ),
            "semantics": "NOT_APPLICABLE_FORMAL_FIXED_TARGET",
            "compatible": None,
        }:
            raise ValueError("SHARDED_FORMAL_QUALIFICATION_PROFILE_INVALID")
    else:
        actual_runner_type, actual_runner_image = _runner_identity("qualification-only")
        if initial_observation.get("os_version") != _read_os_pretty_name():
            raise ValueError("SHARDED_QUALIFICATION_OS_OBSERVATION_FORGED")
        if (
            initial_observation.get("runner_type") != actual_runner_type
            or initial_observation.get("runner_image") != actual_runner_image
        ):
            raise ValueError("SHARDED_QUALIFICATION_RUNNER_OBSERVATION_FORGED")
        if qualification_profile != qualification_runtime_profile(contract, initial_observation):
            raise ValueError("SHARDED_QUALIFICATION_RUNTIME_PROFILE_MISMATCH")
    persisted_inventory = _read(root / "source-inventory.json")
    if persisted_inventory != execution_source_inventory(
        attempt["execution_implementation_commit"]
    ):
        raise ValueError("SHARDED_EXECUTION_SOURCE_INVENTORY_DRIFT")
    validate_checkout_source_inventory(persisted_inventory)
    if attempt["execution_implementation_commit"] != checkout_commit():
        raise ValueError("SHARDED_EXECUTION_COMMIT_NOT_CHECKED_OUT")
    expected = {
        "scientific_parent_implementation_commit": HISTORICAL_QUALITY_IMPLEMENTATION_COMMIT,
        "execution_implementation_commit": attempt["execution_implementation_commit"],
        "execution_context": attempt["execution_context"],
        "scientific_parent_evaluator": HISTORICAL_QUALITY_EVALUATOR_VERSION,
        "execution_evaluator": EVALUATOR_VERSION,
        "target_contract_sha256": target_contract_sha256(contract),
        "runtime_contract_sha256": runtime_contract_sha256(runtime),
        "scientific_policy_sha256": policy["scientific_policy_sha256"],
        "target_observation_sha256": initial_observation["observation_sha256"],
        "source_inventory_sha256": persisted_inventory["source_inventory_sha256"],
        "qualification_runtime_profile_sha256": sha256_value(qualification_profile),
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
    q.configure_canonical_cpu_runtime(seed)
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
        "checkpoint_policy": "final_epoch_only_no_heldout_selection",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
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
        "checkpoint_policy": "final_epoch_only_no_heldout_selection",
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
    live_observation = collect_runtime11_observation(contract, attempt["execution_context"])
    if observation is not None and observation != live_observation:
        raise ValueError("SHARDED_SUPPLIED_OBSERVATION_NOT_LIVE")
    observation = live_observation
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
    final_observation = collect_runtime11_observation(contract, attempt["execution_context"])
    if final_observation["observation_sha256"] != observation["observation_sha256"]:
        raise ValueError("SHARDED_TARGET_CHANGED_DURING_UNIT")
    (output / "task-results.jsonl").write_bytes(
        b"".join(canonical_bytes(row) + b"\n" for row in rows)
    )
    manifest = {
        "unit_evidence_version": UNIT_VERSION,
        "attempt_identity_sha256": attempt["attempt_identity_sha256"],
        "scientific_parent_implementation_commit": attempt[
            "scientific_parent_implementation_commit"
        ],
        "execution_implementation_commit": attempt["execution_implementation_commit"],
        "execution_context": attempt["execution_context"],
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
        "scientific_parent_implementation_commit",
        "execution_implementation_commit",
        "execution_context",
        "runtime_contract_sha256",
        "target_observation_sha256",
        "source_inventory_sha256",
    ):
        if manifest[field] != attempt[field]:
            raise ValueError(f"SHARDED_UNIT_BINDING_MISMATCH:{field}")
    if manifest["execution_context"] == "formal-fixed-target":
        validate_observation_against_contract(contract, manifest["observation"])
    if observation_sha256(manifest["observation"]) != manifest["target_observation_sha256"]:
        raise ValueError("SHARDED_UNIT_OBSERVATION_STALE")


def validate_unit_artifacts(unit: Path, manifest: dict) -> None:
    from planner_toy.numeric_identity import (
        canonical_state_dict_sha256,
        canonical_torch_object_sha256,
        exact_torch_object_sha256,
    )
    from planner_toy.training import state_dict_sha256

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
    Draft202012Validator(
        _read(ROOT / "planner_toy/schemas/fixed_target_quality_checkpoint_manifest.schema.json")
    ).validate(checkpoint)
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
    expected_checkpoint_semantics = {
        "dataset_hash": manifest["dataset_hash"],
        "train_task_ids": manifest["ordered_train_task_ids"],
        "epochs": 3,
        "updates": 9,
        "checkpoint_policy": "final_epoch_only_no_heldout_selection",
        "initialization_path": "initialization.pt",
        "trained_path": "trained.pt",
        "optimizer_state_path": "optimizer-state.pt",
        "optimizer_evidence_path": "optimizer-evidence.json",
        "training_report_path": "training-report.json",
        "torch_object_encoding_version": q.TORCH_OBJECT_ENCODING_VERSION,
        "deterministic_training_replay_status": "CANONICAL_DETERMINISTIC",
    }
    for field, expected in expected_checkpoint_semantics.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"SHARDED_CHECKPOINT_SEMANTIC_MISMATCH:{field}")
    optimizer_state = torch.load(
        training / "optimizer-state.pt", map_location="cpu", weights_only=True
    )
    model = q.LockedPlanner(manifest["seed"], manifest["variant"]).cpu()
    validate_runtime11_optimizer_state(optimizer_state, model, manifest["updates"])
    initialization = torch.load(
        training / "initialization.pt", map_location="cpu", weights_only=True
    )
    trained = torch.load(training / "trained.pt", map_location="cpu", weights_only=True)
    optimizer_evidence = _read(training / "optimizer-evidence.json")
    training_config = _read(training / "training-config.json")
    training_report = _read(training / "training-report.json")
    schema_root = ROOT / "planner_toy/schemas"
    for schema_name, value in (
        ("toy_quality_training_config.schema.json", training_config),
        ("toy_quality_training_report.schema.json", training_report),
        ("toy_quality_optimizer_evidence.schema.json", optimizer_evidence),
    ):
        Draft202012Validator(_read(schema_root / schema_name)).validate(value)
    expected_bindings = {
        "initialization_file_sha256": q._file_hash(training / "initialization.pt"),
        "initialization_state_dict_sha256": state_dict_sha256(initialization),
        "canonical_initialization_state_dict_sha256": canonical_state_dict_sha256(initialization),
        "trained_file_sha256": q._file_hash(training / "trained.pt"),
        "trained_state_dict_sha256": state_dict_sha256(trained),
        "canonical_trained_state_dict_sha256": canonical_state_dict_sha256(trained),
        "optimizer_state_file_sha256": q._file_hash(training / "optimizer-state.pt"),
        "optimizer_state_sha256": exact_torch_object_sha256(optimizer_state),
        "canonical_optimizer_state_sha256": canonical_torch_object_sha256(optimizer_state),
        "config_hash": q._file_hash(training / "training-config.json"),
        "training_report_file_sha256": q._file_hash(training / "training-report.json"),
        "optimizer_evidence_file_sha256": q._file_hash(training / "optimizer-evidence.json"),
    }
    for field, expected in expected_bindings.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"SHARDED_CHECKPOINT_BINDING_MISMATCH:{field}")
    expected_optimizer_evidence = {
        "schema_version": "toy-quality-optimizer-evidence/0.1",
        "torch_object_encoding_version": q.TORCH_OBJECT_ENCODING_VERSION,
        "optimizer_state_sha256": expected_bindings["optimizer_state_sha256"],
        "canonical_optimizer_state_sha256": expected_bindings["canonical_optimizer_state_sha256"],
        "active_parameter_names": checkpoint["active_parameter_names"],
        "parameter_group_count": 1,
        "update_count": 9,
    }
    if optimizer_evidence != expected_optimizer_evidence:
        raise ValueError("SHARDED_OPTIMIZER_EVIDENCE_MISMATCH")
    expected_training = {
        "variant_identity": checkpoint["variant_identity"],
        "seed": manifest["seed"],
        "dataset_hash": manifest["dataset_hash"],
        "train_task_ids": manifest["ordered_train_task_ids"],
        "epochs": 3,
        "updates": 9,
        "checkpoint_policy": "final_epoch_only_no_heldout_selection",
    }
    for field, expected in expected_training.items():
        if training_config.get(field) != expected:
            raise ValueError(f"SHARDED_TRAINING_CONFIG_MISMATCH:{field}")
    if training_config.get("optimizer") != {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
    }:
        raise ValueError("SHARDED_TRAINING_CONFIG_MISMATCH:optimizer")
    expected_identity = {
        "architecture_stage": q.MAPPING[manifest["variant"]][0],
        "implementation_variant": manifest["variant"],
        "experimental_arm": q.MAPPING[manifest["variant"]][1],
        "target_type": q.MAPPING[manifest["variant"]][2],
    }
    if checkpoint["variant_identity"] != expected_identity:
        raise ValueError("SHARDED_VARIANT_IDENTITY_MISMATCH")
    expected_active, expected_dormant = q._optimizer_parameter_policy(model)
    if (
        checkpoint["active_parameter_names"] != expected_active
        or checkpoint["dormant_parameter_names"] != expected_dormant
    ):
        raise ValueError("SHARDED_PARAMETER_POLICY_MISMATCH")
    if any(not torch.equal(trained[name], initialization[name]) for name in expected_dormant):
        raise ValueError("SHARDED_DORMANT_PARAMETER_CHANGED")
    if training_config.get("inventory_hash") != q._file_hash(
        q.ROOT / "docs/architecture/planner_module_inventory_v1.yaml"
    ) or training_config.get("task_encoding_hash") != q._file_hash(
        q.ROOT / "docs/architecture/task_encoding_v1.yaml"
    ):
        raise ValueError("SHARDED_CANONICAL_ARCHITECTURE_BINDING_MISMATCH")
    canonical_dataset = q.generate(17)
    if manifest["dataset_hash"] != canonical_dataset["dataset_hash"]:
        raise ValueError("SHARDED_CANONICAL_DATASET_MISMATCH")
    if training_report != {
        "schema_version": "toy-quality-training-report/0.1",
        "updates": 9,
        "final_checkpoint_selected_without_heldout": True,
    }:
        raise ValueError("SHARDED_TRAINING_REPORT_MISMATCH")
    model.load_state_dict(trained)
    model.eval()
    dataset_rows = {row["task_id"]: row for row in q.generate(17)["validation"]}
    for persisted in rows:
        expected_evidence_root = (
            f"evidence/{manifest['variant']}/seed-{manifest['seed']}/{persisted['task_id']}"
        )
        if persisted.get("evidence_root") != expected_evidence_root:
            raise ValueError("SHARDED_EVIDENCE_ROOT_MISMATCH")
        Draft202012Validator(_read(schema_root / "toy_quality_task_result.schema.json")).validate(
            persisted
        )
        task = dataset_rows[persisted["task_id"]]
        task_root = evidence / persisted["task_id"]
        objects = q.validate_persisted_quality_evidence(
            root=task_root, task=task, checkpoint=checkpoint
        )
        if q.quality_evidence_semantic_hash(**objects) != persisted["evidence_hash"]:
            raise ValueError("SHARDED_EVIDENCE_SEMANTIC_HASH_MISMATCH")
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            reproduced = q._evaluate(
                task,
                manifest["variant"],
                manifest["seed"],
                model,
                checkpoint,
                Path(temporary),
            )
        reproduced["evidence_root"] = persisted["evidence_root"]
        q.validate_task_result_semantics(persisted, reproduced)
    with tempfile.TemporaryDirectory() as replay_directory:
        replay_model, replay_checkpoint = _train_runtime11(
            sorted(canonical_dataset["train"], key=lambda row: row["task_id"]),
            manifest["variant"],
            manifest["seed"],
            Path(replay_directory),
            canonical_dataset["dataset_hash"],
        )
    if (
        state_dict_sha256(initialization) != replay_checkpoint["initialization_state_dict_sha256"]
        or state_dict_sha256(replay_model.state_dict()) != checkpoint["trained_state_dict_sha256"]
        or replay_checkpoint["optimizer_state_sha256"] != checkpoint["optimizer_state_sha256"]
    ):
        raise ValueError("SHARDED_DETERMINISTIC_TRAINING_REPLAY_MISMATCH")


def validate_persisted_runtime11_run(
    evaluation_root: Path,
    evaluation_config: dict,
    variant: str,
    seed: int,
) -> None:
    """Independently replay one packaged sharded run from primary artifacts.

    Packaging intentionally omits orchestration-only unit manifests.  This adapter
    reconstructs only that envelope and delegates to the same recursive semantic
    validator used by unit verification; no persisted verification receipt is trusted.
    """
    import tempfile

    training = evaluation_root / "training-runs" / variant / f"seed-{seed}"
    evidence = evaluation_root / "evidence" / variant / f"seed-{seed}"
    all_rows = [
        json.loads(line)
        for line in (evaluation_root / "task-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows = [row for row in all_rows if row["variant"] == variant and row["seed"] == seed]
    rows.sort(key=lambda row: row["task_id"])
    with tempfile.TemporaryDirectory() as temporary:
        unit = Path(temporary) / "unit"
        unit.mkdir()
        (unit / "training").symlink_to(training.resolve(), target_is_directory=True)
        (unit / "evidence").symlink_to(evidence.resolve(), target_is_directory=True)
        (unit / "task-results.jsonl").write_bytes(
            b"".join(canonical_bytes(row) + b"\n" for row in rows)
        )
        _write(unit / "unit-manifest.json", {})
        manifest = {
            "variant": variant,
            "seed": seed,
            "dataset_hash": evaluation_config["dataset_manifest_hash"],
            "ordered_train_task_ids": evaluation_config["train_task_ids"],
            "ordered_eval_task_ids": evaluation_config["eval_task_ids"],
            "epochs": evaluation_config["epochs"],
            "updates": 9,
            "checkpoint_manifest_sha256": q._file_hash(
                training / "checkpoint-manifest.json"
            ),
            "task_results_sha256": q._file_hash(unit / "task-results.jsonl"),
        }
        validate_unit_artifacts(unit, manifest)


def _unit_tree_identity(unit: Path) -> str:
    return sha256_value(
        {
            path.relative_to(unit).as_posix(): sha256_bytes(path.read_bytes())
            for path in sorted(unit.rglob("*"))
            if path.is_file()
        }
    )


def verify_qualification_unit(root: Path, variant: str, seed: int) -> dict:
    """Independently replay one unit in a short qualification-only process."""
    attempt, contract = validate_attempt(root)
    if attempt["execution_context"] != "qualification-only":
        raise ValueError("SHARDED_QUALIFICATION_VERIFICATION_CONTEXT_REQUIRED")
    unit = root / "units" / variant / f"seed-{seed}"
    manifest = _read(unit / "unit-manifest.json")
    validate_unit_manifest(manifest, attempt, contract)
    validate_unit_artifacts(unit, manifest)
    post_observation = collect_runtime11_observation(contract, "qualification-only")
    if post_observation["observation_sha256"] != attempt["target_observation_sha256"]:
        raise ValueError("SHARDED_TARGET_CHANGED_DURING_VERIFICATION")
    receipt = {
        "verification_version": "toy-quality-sharded-unit-verification/1.0",
        "semantics": "QUALIFICATION_ONLY_NOT_FORMAL_ACCEPTANCE",
        "attempt_identity_sha256": attempt["attempt_identity_sha256"],
        "variant": variant,
        "seed": seed,
        "unit_manifest_sha256": manifest["unit_manifest_sha256"],
        "unit_tree_sha256": _unit_tree_identity(unit),
        "target_observation_sha256": attempt["target_observation_sha256"],
        "verification_observation_sha256": post_observation["observation_sha256"],
        "verification_sha256": "",
    }
    receipt["verification_sha256"] = sha256_value(
        _without(receipt, "verification_sha256")
    )
    _write(root / "qualification-verifications" / variant / f"seed-{seed}.json", receipt)
    return receipt


def _validate_qualification_receipt(root: Path, unit: Path, manifest: dict) -> None:
    receipt = _read(
        root
        / "qualification-verifications"
        / manifest["variant"]
        / f"seed-{manifest['seed']}.json"
    )
    expected = {
        "verification_version": "toy-quality-sharded-unit-verification/1.0",
        "semantics": "QUALIFICATION_ONLY_NOT_FORMAL_ACCEPTANCE",
        "attempt_identity_sha256": manifest["attempt_identity_sha256"],
        "variant": manifest["variant"],
        "seed": manifest["seed"],
        "unit_manifest_sha256": manifest["unit_manifest_sha256"],
        "unit_tree_sha256": _unit_tree_identity(unit),
        "target_observation_sha256": manifest["target_observation_sha256"],
        "verification_observation_sha256": manifest["target_observation_sha256"],
        "verification_sha256": receipt.get("verification_sha256"),
    }
    if receipt != expected or receipt["verification_sha256"] != sha256_value(
        _without(receipt, "verification_sha256")
    ):
        raise ValueError("SHARDED_QUALIFICATION_VERIFICATION_INVALID")


def assemble(root: Path, *, qualification_receipts: bool = False) -> dict:
    attempt, contract = validate_attempt(root)
    if qualification_receipts and attempt["execution_context"] != "qualification-only":
        raise ValueError("SHARDED_QUALIFICATION_ASSEMBLY_CONTEXT_REQUIRED")
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
        if qualification_receipts:
            _validate_qualification_receipt(root, path.parent, manifest)
        else:
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
        "implementation_commit": attempt["execution_implementation_commit"],
        "scientific_parent_implementation_commit": attempt[
            "scientific_parent_implementation_commit"
        ],
        "evaluator_source_sha256": attempt["source_inventory_sha256"],
        "requirements_lock_sha256": requirements_lock_sha256_at_commit(
            attempt["execution_implementation_commit"]
        ),
        "target_contract_sha256": attempt["target_contract_sha256"],
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation_sha256": attempt["target_observation_sha256"],
        "source_inventory_sha256": attempt["source_inventory_sha256"],
        "scientific_policy_sha256": attempt["scientific_policy_sha256"],
        "execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES",
        "execution_context": attempt["execution_context"],
        "checkpoint_origin_run_hash": None,
        "reuse_source_manifest_hash": None,
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
        "variant_mapping": config["variant_mapping"],
        "evaluator_source_sha256": config["evaluator_source_sha256"],
        "implementation_commit": config["implementation_commit"],
        "scientific_parent_implementation_commit": config[
            "scientific_parent_implementation_commit"
        ],
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
    payload = sharded_canonical_replay_payload(output, manifest, rows)
    _write(output / "canonical-semantic-payload.json", payload)
    manifest["artifact_hashes"]["canonical-semantic-payload.json"] = q._file_hash(
        output / "canonical-semantic-payload.json"
    )
    _write(output / "evaluation-manifest.json", manifest)
    replay = sharded_canonical_replay_hash(output, manifest, rows)
    (output / "replay-hash.txt").write_text(replay + "\n")
    policy = _read(root / "scientific-policy.json")
    evidence = {
        "execution_evidence_version": "toy-quality-fixed-target-execution-evidence/1.0",
        "implementation_commit": attempt["execution_implementation_commit"],
        "scientific_parent_implementation_commit": attempt[
            "scientific_parent_implementation_commit"
        ],
        "execution_topology": "SHARDED_VARIANT_SEED_SUBPROCESSES",
        "execution_context": attempt["execution_context"],
        "target_contract_sha256": attempt["target_contract_sha256"],
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation_sha256": attempt["target_observation_sha256"],
        "source_inventory_sha256": attempt["source_inventory_sha256"],
        "scientific_policy": policy,
        "scientific_policy_sha256": policy["scientific_policy_sha256"],
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_source_sha256": attempt["source_inventory_sha256"],
        "requirements_lock_sha256": config["requirements_lock_sha256"],
        "dataset_identity": {
            "dataset_hash": dataset["dataset_hash"],
            "ordered_train_task_ids": config["train_task_ids"],
            "ordered_eval_task_ids": config["eval_task_ids"],
        },
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "epochs": 3,
        "updates_per_run": 9,
        "optimizer_class": "AdamW",
        "optimizer_hyperparameters": {
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
        "gradient_clipping": {"algorithm": "clip_grad_norm_", "max_norm": 1.0},
        "observed_optimizer_foreach": False,
        "observed_optimizer_fused": False,
        "evaluation_root_identity": replay,
        "execution_evidence_sha256": "",
    }
    evidence["execution_evidence_sha256"] = execution_evidence_sha256(evidence)
    _write(root / "execution-evidence.json", evidence)
    return {"replay_hash": replay, "units": 9}


def package_foundation_attempt(root: Path, destination: Path, attempt_index: int) -> dict:
    """Package a validated sharded attempt in the PR19 attempt directory shape."""
    attempt, _ = validate_attempt(root)
    if not (root / "evaluation/replay-hash.txt").is_file():
        raise ValueError("SHARDED_EVALUATION_NOT_ASSEMBLED")
    if destination.exists():
        raise ValueError("SHARDED_FOUNDATION_DESTINATION_EXISTS")
    destination.mkdir(parents=True)
    shutil.copytree(root / "evaluation", destination / "evaluation")
    shutil.copy2(root / "execution-evidence.json", destination / "execution-evidence.json")
    preflight = {
        "implementation_commit": attempt["execution_implementation_commit"],
        "target_contract": _read(root / "target-contract.json"),
        "target_contract_sha256": attempt["target_contract_sha256"],
        "runtime_contract": _read(root / "runtime-contract.json"),
        "runtime_contract_sha256": attempt["runtime_contract_sha256"],
        "target_observation": _read(root / "initial-observation.json"),
        "source_inventory": _read(root / "source-inventory.json"),
    }
    _write(destination / "preflight.json", preflight)
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = path.relative_to(destination).as_posix()
            files[relative] = sha256_bytes(path.read_bytes())
    manifest = {
        "attempt_manifest_version": ATTEMPT_MANIFEST_VERSION,
        "attempt_index": attempt_index,
        "files": files,
        "attempt_manifest_sha256": "",
    }
    manifest["attempt_manifest_sha256"] = attempt_manifest_sha256(manifest)
    _write(destination / "attempt_manifest.json", manifest)
    return manifest
