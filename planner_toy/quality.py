# ruff: noqa: E501, E702
"""Development-only held-out quality evaluation for the existing toy planners."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jsonschema import Draft202012Validator

from .canonical import canonical_bytes, sha256
from .dataset import generate
from .domain import goal_satisfied, validate_state
from .e2e import (
    A2Planner,
    evaluate_frozen_plan,
    quality_evidence_semantic_hash,
    validate_persisted_quality_evidence,
)
from .model import LockedPlanner, canonical_task_encoding
from .semantic import targets
from .training import ACTIONS, labels, state_dict_sha256

VERSION = "development-quality-evaluation/0.1"
SEEDS = (17, 29, 43)
VARIANTS = ("A2", "A3", "A4")
MAPPING = {
    "A2": ("STAGE_1", "A2-structured-baseline", "NONE"),
    "A3": ("STAGE_2A", "A3a-codebook", "DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK"),
    "A4": ("STAGE_2A", "A3a-zero", "DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK"),
}
EPOCHS = 3
MAX_STEPS = 17
ROOT = Path(__file__).parents[1]
SOURCE_FILES = (
    "planner_toy/canonical.py", "planner_toy/dataset.py", "planner_toy/domain.py", "planner_toy/e2e.py",
    "planner_toy/model.py", "planner_toy/quality.py", "planner_toy/semantic.py",
    "planner_toy/training.py", "planner_toy/schemas/toy_quality_evaluation.schema.json",
    "scripts/run_toy_quality_evaluation.py",
    "planner_toy/schemas/toy_quality_aggregate_summary.schema.json",
    "planner_toy/schemas/toy_quality_compact_summary.schema.json",
    "planner_toy/schemas/toy_quality_dataset_manifest.schema.json",
    "planner_toy/schemas/toy_quality_evaluation_manifest.schema.json",
    "planner_toy/schemas/toy_quality_paired_comparison.schema.json",
    "planner_toy/schemas/toy_quality_per_seed_summary.schema.json",
    "planner_toy/schemas/toy_quality_structural_breakdown.schema.json",
    "docs/architecture/planner_module_inventory_v1.yaml",
    "docs/architecture/task_encoding_v1.yaml",
)
SOURCE_FILES = tuple(sorted(set(SOURCE_FILES) | {
    str(path.relative_to(ROOT)) for path in (ROOT / "planner_toy/schemas").glob("toy_*.schema.json")
}))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _torch_object_sha256(value) -> str:
    digest = hashlib.sha256()
    def visit(item) -> None:
        if torch.is_tensor(item):
            digest.update(b"tensor\0" + str(item.dtype).encode() + b"\0")
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(b"key\0" + str(key).encode() + b"\0"); visit(item[key])
        elif isinstance(item, list | tuple):
            digest.update(f"sequence:{len(item)}\0".encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode() + b"\0")
    visit(value)
    return "sha256:" + digest.hexdigest()


def source_identity() -> dict:
    entries = [{"path": name, "sha256": _file_hash(ROOT / name)} for name in SOURCE_FILES]
    return {"evaluator_source_files": entries, "evaluator_source_sha256": sha256(entries)}


def canonical_replay_payload(root: Path, manifest: dict, rows: list[dict]) -> dict:
    """Deterministic cross-run identity; raw hashes remain in ``manifest`` for integrity."""
    checkpoint_identities = {}
    for path in sorted(root.glob("training-runs/*/seed-*/checkpoint-manifest.json")):
        checkpoint = json.loads(path.read_bytes())
        semantic = {
            key: value for key, value in checkpoint.items()
            if key not in {
                "initialization_file_sha256", "trained_file_sha256",
                "optimizer_state_file_sha256", "reuse_source_manifest_hash",
            }
        }
        checkpoint_identities[str(path.relative_to(root))] = sha256(semantic)
    return {
        "schema_version": "toy-quality-canonical-replay/1.0",
        "evaluator_version": manifest["evaluator_version"],
        "evaluator_source_sha256": manifest["evaluator_source_sha256"],
        "variants": manifest["variants"], "seeds": manifest["seeds"],
        "variant_mapping": manifest["variant_mapping"],
        "artifact_hashes": manifest["artifact_hashes"],
        "checkpoint_semantic_hashes": checkpoint_identities,
        "evidence_semantic_hashes": {
            f"{row['variant']}/seed-{row['seed']}/{row['task_id']}": row["evidence_hash"]
            for row in rows
        },
    }


def canonical_replay_hash(root: Path, manifest: dict, rows: list[dict]) -> str:
    return sha256(canonical_replay_payload(root, manifest, rows))


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def source_identity_at_commit(commit: str) -> dict:
    entries = []
    for name in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{name}")
        if result.returncode:
            raise ValueError(f"implementation commit does not contain {name}")
        entries.append({"path": name, "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest()})
    return {"evaluator_source_files": entries, "evaluator_source_sha256": sha256(entries)}


def implementation_provenance(commit: str | None = None) -> dict:
    result = _git("rev-parse", commit or "HEAD")
    implementation_commit = result.stdout.strip() if result.returncode == 0 else None
    if implementation_commit is None or _git("cat-file", "-e", f"{implementation_commit}^{{commit}}").returncode:
        raise ValueError("implementation commit must identify an existing commit")
    requirements = _git_bytes("show", f"{implementation_commit}:requirements.lock")
    if requirements.returncode:
        raise ValueError("implementation commit does not contain requirements.lock")
    return {
        "implementation_commit": implementation_commit,
        "requirements_lock_sha256": "sha256:" + hashlib.sha256(requirements.stdout).hexdigest(),
        "runtime_versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "execution_device": "cpu",
        },
    }


def _train(rows: list[dict], variant: str, seed: int, output: Path, dataset_hash: str) -> tuple[LockedPlanner, dict]:
    """Train in canonical task order, with no evaluation inputs or selection."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = LockedPlanner(seed, variant).cpu()
    output.mkdir(parents=True, exist_ok=True)
    initial_path = output / "initialization.pt"
    torch.save(model.state_dict(), initial_path)
    initialization_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
    )
    for _ in range(EPOCHS):
        for row in sorted(rows, key=lambda item: item["task_id"]):
            action, arg1, arg2 = labels(row)
            valid = len(row["oracle_work_plan"])
            target = targets(row)
            shifted = torch.cat([torch.zeros_like(target[:, :1]), target[:, :-1]], 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                canonical_task_encoding(row), action, arg1, arg2,
                semantic_feedback=shifted if variant in {"A3", "A4"} else None,
            )
            flat = action[:, :valid].flatten()
            loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
            one = flat != ACTIONS["END"]
            two = (flat == ACTIONS["UNSTACK"]) | (flat == ACTIONS["STACK"])
            if one.any():
                loss += F.cross_entropy(logits.arg1[:, :valid].flatten(0, 1)[one], arg1[:, :valid].flatten()[one])
            if two.any():
                loss += F.cross_entropy(logits.arg2[:, :valid].flatten(0, 1)[two], arg2[:, :valid].flatten()[two])
            if variant in {"A3", "A4"}:
                loss += (1 - (logits.z_semantic[:, :valid] * target[:, :valid]).sum(-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    trained_path = output / "trained.pt"
    torch.save(model.state_dict(), trained_path)
    optimizer_path = output / "optimizer-state.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    optimizer_evidence = {
        "schema_version": "toy-quality-optimizer-evidence/0.1",
        "optimizer_state_sha256": _torch_object_sha256(optimizer.state_dict()),
        "active_parameter_names": sorted(model.active_names),
        "parameter_group_count": len(optimizer.param_groups),
        "update_count": EPOCHS * len(rows),
    }
    _write(output / "optimizer-evidence.json", optimizer_evidence)
    checkpoint_hash = state_dict_sha256(model.state_dict())
    identity = {"architecture_stage": MAPPING[variant][0], "implementation_variant": variant,
                "experimental_arm": MAPPING[variant][1], "target_type": MAPPING[variant][2]}
    config = {"schema_version": "toy-quality-training-config/0.1", "variant_identity": identity,
              "seed": seed, "dataset_hash": dataset_hash,
              "train_task_ids": [r["task_id"] for r in rows], "epochs": EPOCHS,
              "updates": EPOCHS * len(rows), "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "betas": [0.9, .95], "eps": 1e-8, "weight_decay": .01, "gradient_clip_norm": 1.0},
              "checkpoint_policy": "final_epoch_only_no_heldout_selection",
              "inventory_hash": _file_hash(ROOT / "docs/architecture/planner_module_inventory_v1.yaml"),
              "task_encoding_hash": _file_hash(ROOT / "docs/architecture/task_encoding_v1.yaml")}
    _write(output / "training-config.json", config)
    report = {"schema_version": "toy-quality-training-report/0.1", "updates": EPOCHS * len(rows), "final_checkpoint_selected_without_heldout": True}
    _write(output / "training-report.json", report)
    manifest = {"schema_version": "toy-quality-checkpoint/0.1", "variant_identity": identity,
                "seed": seed, "dataset_hash": dataset_hash, "train_task_ids": config["train_task_ids"],
                "epochs": EPOCHS, "updates": EPOCHS * len(rows), "config_hash": _file_hash(output / "training-config.json"),
                "initialization_path": "initialization.pt", "initialization_file_sha256": _file_hash(initial_path),
                "initialization_state_dict_sha256": initialization_hash,
                "trained_path": "trained.pt", "trained_file_sha256": _file_hash(trained_path),
                "trained_state_dict_sha256": checkpoint_hash,
                "optimizer_state_path": "optimizer-state.pt", "optimizer_state_file_sha256": _file_hash(optimizer_path),
                "optimizer_state_sha256": _torch_object_sha256(optimizer.state_dict()),
                "optimizer_evidence_path": "optimizer-evidence.json",
                "optimizer_evidence_file_sha256": _file_hash(output / "optimizer-evidence.json"),
                "training_report_path": "training-report.json", "training_report_file_sha256": _file_hash(output / "training-report.json"),
                "checkpoint_policy": config["checkpoint_policy"],
                "training_execution_mode": "TRAINED_IN_RUN",
                "checkpoint_origin_run_hash": None, "reuse_source_manifest_hash": None,
                "deterministic_training_replay_status": "CANONICAL_DETERMINISTIC",
                "active_parameter_names": sorted(model.active_names),
                "dormant_parameter_names": sorted(name for name, p in model.named_parameters() if not p.requires_grad)}
    _write(output / "checkpoint-manifest.json", manifest)
    return model, manifest


def _evaluate(row: dict, variant: str, seed: int, model: LockedPlanner, checkpoint: dict, evidence_root: Path) -> dict:
    planner = A2Planner(model)
    evidence = evaluate_frozen_plan(row=row, planner=planner, output=evidence_root, checkpoint_binding=checkpoint)
    raw = evidence["raw_output"]
    terminal = evidence["terminal_end"]
    parse_valid = evidence["generation_failure"] is None
    trace = evidence["events"]
    applicable = [event["status"] == "APPLIED" for event in trace]
    state = validate_state(tuple(row["blocks"]), tuple(tuple(x) for x in row["initial"]))
    initial_goal = goal_satisfied(state, tuple(tuple(x) for x in row["goal"]))
    generated = len(raw) - (1 if terminal else 0)
    attempted, applied = len(trace), sum(applicable)
    completed = parse_valid and terminal and attempted == applied
    executable = completed and (generated > 0 or initial_goal)
    reached = evidence["goal_reached"]
    failure = evidence["failure_code"]
    final_state = row["initial"]
    if trace:
        # Existing executor evidence is hash-based; semantic validator replays it.
        from .domain import apply_action
        current = state
        for action in evidence["parsed_actions"][:applied]:
            current = apply_action(tuple(row["blocks"]), current, tuple(action))
        final_state = [list(x) for x in current]
    gold = row["oracle_work_plan"][:-1]
    predicted = raw[:-1] if terminal else raw
    return {
        "variant": variant, "architecture_stage": MAPPING[variant][0], "implementation_variant": variant,
        "experimental_arm": MAPPING[variant][1], "target_type": MAPPING[variant][2], "seed": seed,
        "task_id": row["task_id"], "split": "validation", "initial_state": row["initial"],
        "goal": row["goal"], "gold_actions": gold, "predicted_actions": predicted,
        "gold_plan_length": len(gold), "predicted_plan_length": len(predicted),
        "terminal_end_produced": terminal,
        "plan_generation_success": evidence["generation_failure"] is None,
        "action_parse_valid": parse_valid, "action_applicability": applicable,
        "generated_action_count": generated, "action_attempt_count": attempted,
        "applicable_action_count": applied, "nonempty_plan": generated > 0,
        "initial_goal_satisfied": initial_goal, "end_only_plan": terminal and generated == 0,
        "execution_completed_without_precondition_failure": completed,
        "full_plan_executable": executable, "goal_reached": reached,
        "exact_plan_match": predicted == gold, "planner_call_count": planner.calls,
        "replanning_count": evidence["replanning_count"], "model_forward_count": evidence["model_forward_count"],
        "checkpoint_hash": checkpoint["trained_state_dict_sha256"], "failure_code": failure,
        "execution_trace": trace, "final_state": final_state,
        "evidence_root": str(evidence_root), "evidence_hash": evidence["evidence_hash"],
        "work_plan_hash": evidence["work_plan_hash"],
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    attempted = sum(row["action_attempt_count"] for row in rows)
    applicable = sum(row["applicable_action_count"] for row in rows)
    return {
        "unique_task_count": len({r["task_id"] for r in rows}),
        "seed_count": len({r["seed"] for r in rows}), "evaluation_unit_count": n,
        "heldout_task_count": n, "success_count": sum(r["goal_reached"] for r in rows),
        "task_success_rate": sum(r["goal_reached"] for r in rows) / n,
        "plan_generation_success_rate": sum(r["plan_generation_success"] for r in rows) / n,
        "end_rate": sum(r["terminal_end_produced"] for r in rows) / n,
        "fully_executable_plan_rate": sum(r["full_plan_executable"] for r in rows) / n,
        "action_parse_valid_rate": sum(r["action_parse_valid"] for r in rows) / n,
        "attempted_action_count": attempted, "applicable_action_count": applicable,
        "action_applicable_rate": applicable / attempted if attempted else None,
        "nonempty_plan_rate": sum(r["nonempty_plan"] for r in rows) / n,
        "end_only_rate": sum(r["end_only_plan"] for r in rows) / n,
        "execution_completed_rate": sum(r["execution_completed_without_precondition_failure"] for r in rows) / n,
        "exact_plan_match_rate": sum(r["exact_plan_match"] for r in rows) / n,
        "mean_predicted_plan_length": statistics.mean(r["predicted_plan_length"] for r in rows),
        "median_predicted_plan_length": statistics.median(r["predicted_plan_length"] for r in rows),
        "mean_gold_plan_length": statistics.mean(r["gold_plan_length"] for r in rows),
        "mean_absolute_plan_length_difference": statistics.mean(abs(r["predicted_plan_length"] - r["gold_plan_length"]) for r in rows),
        "failure_code_distribution": dict(sorted(Counter(r["failure_code"] or "NONE" for r in rows).items())),
    }


def paired(rows: list[dict], first: str, second: str) -> dict:
    index = {(r["variant"], r["seed"], r["task_id"]): r for r in rows}
    pairs = [(index[(first, s, t)], index[(second, s, t)]) for s in sorted({r["seed"] for r in rows}) for t in sorted({r["task_id"] for r in rows})]
    return {"first": first, "second": second, "pair_count": len(pairs),
            "both_succeed": sum(a["goal_reached"] and b["goal_reached"] for a, b in pairs),
            "only_first_succeeds": sum(a["goal_reached"] and not b["goal_reached"] for a, b in pairs),
            "only_second_succeeds": sum(not a["goal_reached"] and b["goal_reached"] for a, b in pairs),
            "both_fail": sum(not a["goal_reached"] and not b["goal_reached"] for a, b in pairs)}


def validate_task_result_semantics(persisted: dict, reproduced: dict) -> None:
    """Reject a persisted task claim that differs from independent replay."""
    if persisted != reproduced:
        raise ValueError("SEMANTIC_REPLAY_MISMATCH")


def _breakdowns(rows: list[dict]) -> dict:
    out = {}
    for label, low, high in (("1-2", 1, 2), ("3-4", 3, 4), ("5+", 5, 10**9)):
        subset = [r for r in rows if low <= r["gold_plan_length"] <= high]
        if subset:
            summary = summarize(subset)
            out[label] = {k: summary[k] for k in ("unique_task_count", "seed_count", "evaluation_unit_count", "task_success_rate", "fully_executable_plan_rate", "mean_predicted_plan_length", "failure_code_distribution")}
    return out


def aggregate_summaries(per_seed: dict, rows: list[dict], variants, seeds) -> dict:
    return {v: {"mean_success_rate": statistics.mean(per_seed[f"{v}/seed-{s}"]["task_success_rate"] for s in seeds),
                "min_success_rate": min(per_seed[f"{v}/seed-{s}"]["task_success_rate"] for s in seeds),
                "max_success_rate": max(per_seed[f"{v}/seed-{s}"]["task_success_rate"] for s in seeds),
                "success_counts": {str(s): per_seed[f"{v}/seed-{s}"]["success_count"] for s in seeds},
                "mean_executability": statistics.mean(per_seed[f"{v}/seed-{s}"]["fully_executable_plan_rate"] for s in seeds),
                "attempted_action_count": sum(per_seed[f"{v}/seed-{s}"]["attempted_action_count"] for s in seeds),
                "applicable_action_count": sum(per_seed[f"{v}/seed-{s}"]["applicable_action_count"] for s in seeds),
                "action_applicable_rate": (lambda attempted, applicable: applicable / attempted if attempted else None)(sum(per_seed[f"{v}/seed-{s}"]["attempted_action_count"] for s in seeds), sum(per_seed[f"{v}/seed-{s}"]["applicable_action_count"] for s in seeds)),
                "nonempty_plan_rate": statistics.mean(per_seed[f"{v}/seed-{s}"]["nonempty_plan_rate"] for s in seeds),
                "end_only_rate": statistics.mean(per_seed[f"{v}/seed-{s}"]["end_only_rate"] for s in seeds),
                "mean_plan_length_difference": statistics.mean(per_seed[f"{v}/seed-{s}"]["mean_absolute_plan_length_difference"] for s in seeds),
                "structural_breakdown": _breakdowns([r for r in rows if r["variant"] == v])} for v in variants}


def validate_evaluation(root: Path) -> dict:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    schema_root = Path(__file__).with_name("schemas")
    def validate_schema(name: str, value) -> None:
        Draft202012Validator(json.loads((schema_root / name).read_bytes())).validate(value)

    config = json.loads((root / "evaluation-config.json").read_bytes())
    validate_schema("toy_quality_evaluation.schema.json", config)
    canonical = generate(17)
    expected_train = sorted(r["task_id"] for r in canonical["train"])
    expected_eval = sorted(r["task_id"] for r in canonical["validation"])
    if config["dataset_manifest_hash"] != canonical["dataset_hash"]:
        raise ValueError("CANONICAL_DATASET_HASH_MISMATCH")
    if config["train_task_ids"] != expected_train:
        raise ValueError("CANONICAL_TRAIN_SPLIT_MISMATCH")
    if config["diagnostic_complete"] and config["eval_task_ids"] != expected_eval:
        raise ValueError("CANONICAL_EVAL_SPLIT_MISMATCH")
    semantic_complete = (
        tuple(config["variants"]) == VARIANTS
        and tuple(config["seeds"]) == SEEDS
        and config["train_task_ids"] == expected_train
        and config["eval_task_ids"] == expected_eval
        and config["training_execution_mode"] == "TRAINED_IN_RUN"
        and config["checkpoint_origin_run_hash"] is None
        and config["reuse_source_manifest_hash"] is None
        and config["deterministic_training_replay_status"] == "CANONICAL_DETERMINISTIC"
    )
    if config["diagnostic_complete"] != semantic_complete:
        raise ValueError("DIAGNOSTIC_COMPLETE_SEMANTIC_MISMATCH")
    if config["diagnostic_complete"] and (
        config["training_execution_mode"] != "TRAINED_IN_RUN"
        or config["checkpoint_origin_run_hash"] is not None
        or config["reuse_source_manifest_hash"] is not None
        or config["deterministic_training_replay_status"] != "CANONICAL_DETERMINISTIC"
    ):
        raise ValueError("COMPLETE_RUN_TRAINING_PROVENANCE_MISMATCH")
    if config["evaluator_source_files"] != source_identity()["evaluator_source_files"] or config["evaluator_source_sha256"] != source_identity()["evaluator_source_sha256"]:
        raise ValueError("EVALUATOR_SOURCE_STALE")
    implementation_commit = config["implementation_commit"]
    if not implementation_commit or _git("cat-file", "-e", f"{implementation_commit}^{{commit}}").returncode:
        raise ValueError("IMPLEMENTATION_COMMIT_MISSING")
    if _git("merge-base", "--is-ancestor", implementation_commit, "HEAD").returncode:
        raise ValueError("IMPLEMENTATION_COMMIT_NOT_ANCESTOR")
    if config["diagnostic_complete"] and source_identity_at_commit(implementation_commit) != source_identity():
        raise ValueError("IMPLEMENTATION_COMMIT_SOURCE_MISMATCH")
    current_provenance = implementation_provenance(implementation_commit)
    if config["requirements_lock_sha256"] != current_provenance["requirements_lock_sha256"] or config["runtime_versions"] != current_provenance["runtime_versions"]:
        raise ValueError("IMPLEMENTATION_RUNTIME_PROVENANCE_MISMATCH")
    expected_dataset_manifest = {"dataset_hash": canonical["dataset_hash"], "train_task_ids": expected_train, "eval_task_ids": config["eval_task_ids"]}
    dataset_manifest = json.loads((root / "dataset-manifest.json").read_bytes())
    validate_schema("toy_quality_dataset_manifest.schema.json", dataset_manifest)
    if dataset_manifest != expected_dataset_manifest:
        raise ValueError("DATASET_MANIFEST_SEMANTIC_MISMATCH")
    rows = [json.loads(line) for line in (root / "task-results.jsonl").read_text().splitlines()]
    for row in rows:
        validate_schema("toy_quality_task_result.schema.json", row)
    keys = [(r["variant"], r["seed"], r["task_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(config["train_task_ids"]) & set(config["eval_task_ids"]):
        raise ValueError("DUPLICATE_OR_SPLIT_OVERLAP")
    expected = {(v, s, t) for v in config["variants"] for s in config["seeds"] for t in config["eval_task_ids"]}
    if set(keys) != expected:
        raise ValueError("TASK_RESULT_COVERAGE_MISMATCH")
    expected_order = [(v, s, t) for v in config["variants"] for s in config["seeds"] for t in config["eval_task_ids"]]
    if keys != expected_order:
        raise ValueError("TASK_RESULT_CANONICAL_ORDER_MISMATCH")
    for row in rows:
        if (row["architecture_stage"], row["implementation_variant"], row["experimental_arm"], row["target_type"]) != (MAPPING[row["variant"]][0], row["variant"], MAPPING[row["variant"]][1], MAPPING[row["variant"]][2]):
            raise ValueError("VARIANT_MAPPING_MISMATCH")
    dataset_rows = {r["task_id"]: r for r in canonical["validation"]}
    loaded = {}
    for variant in config["variants"]:
        for seed in config["seeds"]:
            run_dir = root / "training-runs" / variant / f"seed-{seed}"
            checkpoint = json.loads((run_dir / "checkpoint-manifest.json").read_bytes())
            training_config = json.loads((run_dir / "training-config.json").read_bytes())
            validate_schema("toy_quality_checkpoint_manifest.schema.json", checkpoint)
            validate_schema("toy_quality_training_config.schema.json", training_config)
            if _file_hash(run_dir / "training-config.json") != checkpoint["config_hash"]:
                raise ValueError("TRAINING_CONFIG_HASH_MISMATCH")
            if training_config["variant_identity"] != config["variant_mapping"][variant] or training_config["seed"] != seed or training_config["dataset_hash"] != canonical["dataset_hash"] or training_config["train_task_ids"] != expected_train or training_config["epochs"] != 3 or training_config["updates"] != 9 or training_config["optimizer"] != config["optimizer"] or training_config["checkpoint_policy"] != config["checkpoint_policy"]:
                raise ValueError("TRAINING_CONFIG_SEMANTIC_MISMATCH")
            for name, field, state_field in (("initialization.pt", "initialization_file_sha256", "initialization_state_dict_sha256"), ("trained.pt", "trained_file_sha256", "trained_state_dict_sha256")):
                path = run_dir / name
                if _file_hash(path) != checkpoint[field]:
                    raise ValueError("CHECKPOINT_FILE_HASH_MISMATCH")
                state = torch.load(path, map_location="cpu", weights_only=True)
                if state_dict_sha256(state) != checkpoint[state_field]:
                    raise ValueError("CHECKPOINT_STATE_HASH_MISMATCH")
            for path_field, hash_field in (("optimizer_state_path", "optimizer_state_file_sha256"), ("optimizer_evidence_path", "optimizer_evidence_file_sha256"), ("training_report_path", "training_report_file_sha256")):
                if _file_hash(run_dir / checkpoint[path_field]) != checkpoint[hash_field]:
                    raise ValueError("TRAINING_EVIDENCE_HASH_MISMATCH")
            optimizer_state = torch.load(
                run_dir / checkpoint["optimizer_state_path"], map_location="cpu", weights_only=True
            )
            if _torch_object_sha256(optimizer_state) != checkpoint["optimizer_state_sha256"]:
                raise ValueError("OPTIMIZER_STATE_SEMANTIC_HASH_MISMATCH")
            if len(optimizer_state.get("param_groups", [])) != 1 or len(
                optimizer_state["param_groups"][0].get("params", [])
            ) != len(checkpoint["active_parameter_names"]):
                raise ValueError("OPTIMIZER_PARAMETER_GROUP_MISMATCH")
            optimizer_evidence = json.loads(
                (run_dir / checkpoint["optimizer_evidence_path"]).read_bytes()
            )
            validate_schema("toy_quality_optimizer_evidence.schema.json", optimizer_evidence)
            if optimizer_evidence != {
                "schema_version": "toy-quality-optimizer-evidence/0.1",
                "optimizer_state_sha256": checkpoint["optimizer_state_sha256"],
                "active_parameter_names": checkpoint["active_parameter_names"],
                "parameter_group_count": 1,
                "update_count": 9,
            }:
                raise ValueError("OPTIMIZER_EVIDENCE_SEMANTIC_MISMATCH")
            if checkpoint["seed"] != seed or checkpoint["variant_identity"] != config["variant_mapping"][variant] or checkpoint["train_task_ids"] != expected_train or checkpoint["updates"] != 9:
                raise ValueError("CHECKPOINT_LINEAGE_MISMATCH")
            training_report = json.loads((run_dir / checkpoint["training_report_path"]).read_bytes())
            validate_schema("toy_quality_training_report.schema.json", training_report)
            if training_report != {"schema_version": "toy-quality-training-report/0.1", "updates": 9, "final_checkpoint_selected_without_heldout": True}:
                raise ValueError("TRAINING_REPORT_SEMANTIC_MISMATCH")
            expected_mode = config["training_execution_mode"]
            if checkpoint["training_execution_mode"] != expected_mode or checkpoint["checkpoint_origin_run_hash"] != config["checkpoint_origin_run_hash"] or checkpoint["reuse_source_manifest_hash"] != config["reuse_source_manifest_hash"] or checkpoint["deterministic_training_replay_status"] != config["deterministic_training_replay_status"]:
                raise ValueError("CHECKPOINT_TRAINING_ORIGIN_MISMATCH")
            model = LockedPlanner(seed, variant).cpu()
            canonical_initial = LockedPlanner(seed, variant).state_dict()
            persisted_initial = torch.load(run_dir / "initialization.pt", map_location="cpu", weights_only=True)
            if state_dict_sha256(canonical_initial) != state_dict_sha256(persisted_initial):
                raise ValueError("CANONICAL_INITIALIZATION_MISMATCH")
            model.load_state_dict(torch.load(run_dir / "trained.pt", map_location="cpu", weights_only=True)); model.eval()
            if checkpoint["active_parameter_names"] != sorted(model.active_names) or checkpoint["dormant_parameter_names"] != sorted(name for name, parameter in model.named_parameters() if not parameter.requires_grad):
                raise ValueError("PARAMETER_POLICY_MISMATCH")
            trained_state = model.state_dict()
            if any(not torch.equal(trained_state[name], persisted_initial[name]) for name in checkpoint["dormant_parameter_names"]):
                raise ValueError("DORMANT_PARAMETER_CHANGED")
            if config["diagnostic_complete"] or config["training_execution_mode"] == "REUSED":
                with tempfile.TemporaryDirectory() as training_replay_dir:
                    replay_model, replay_manifest = _train(
                        sorted(canonical["train"], key=lambda row: row["task_id"]),
                        variant, seed, Path(training_replay_dir), canonical["dataset_hash"],
                    )
                if (
                    state_dict_sha256(replay_model.state_dict())
                    != checkpoint["trained_state_dict_sha256"]
                    or replay_manifest["optimizer_state_sha256"]
                    != checkpoint["optimizer_state_sha256"]
                ):
                    raise ValueError("DETERMINISTIC_TRAINING_REPLAY_MISMATCH")
                if checkpoint["trained_state_dict_sha256"] == checkpoint["initialization_state_dict_sha256"]:
                    raise ValueError("TRAINED_CHECKPOINT_UNCHANGED")
            loaded[(variant, seed)] = (model, checkpoint)
    for seed in config["seeds"]:
        initialization_hashes = {loaded[(variant, seed)][1]["initialization_state_dict_sha256"] for variant in config["variants"]}
        if len(initialization_hashes) != 1:
            raise ValueError("COMMON_INITIALIZATION_MISMATCH")
    for persisted in rows:
        evidence_root = root / persisted["evidence_root"]
        required = {"planner-request.json", "attempt-log.jsonl", "episode-plan-manifest.json", "episode-log.json", "evaluation-result.json"}
        if persisted["plan_generation_success"]:
            required.add("work-plan.json")
        if persisted["variant"] in {"A3", "A4"}:
            required.update({"semantic-trace.json", "semantic-latents.f32"})
        actual = {path.name for path in evidence_root.iterdir() if path.is_file()}
        if actual != required:
            raise ValueError("EVIDENCE_COVERAGE_MISMATCH")
        model, checkpoint = loaded[(persisted["variant"], persisted["seed"])]
        evidence_objects = validate_persisted_quality_evidence(
            root=evidence_root,
            task=dataset_rows[persisted["task_id"]],
            checkpoint=checkpoint,
        )
        actual_evidence_hash = quality_evidence_semantic_hash(**evidence_objects)
        if actual_evidence_hash != persisted["evidence_hash"]:
            raise ValueError("EVIDENCE_SEMANTIC_HASH_MISMATCH")
        with tempfile.TemporaryDirectory() as directory:
            reproduced = _evaluate(dataset_rows[persisted["task_id"]], persisted["variant"], persisted["seed"], model, checkpoint, Path(directory))
        reproduced["evidence_root"] = persisted["evidence_root"]
        validate_task_result_semantics(persisted, reproduced)
    per_seed = json.loads((root / "per-seed-summary.json").read_bytes())
    validate_schema("toy_quality_per_seed_summary.schema.json", per_seed)
    recalculated = {f"{v}/seed-{s}": summarize([r for r in rows if r["variant"] == v and r["seed"] == s]) for v in config["variants"] for s in config["seeds"]}
    if list(per_seed) != list(recalculated):
        raise ValueError("PER_SEED_CANONICAL_ORDER_MISMATCH")
    if per_seed != recalculated:
        raise ValueError("SUMMARY_MISMATCH")
    aggregate = json.loads((root / "aggregate-summary.json").read_bytes())
    validate_schema("toy_quality_aggregate_summary.schema.json", aggregate)
    for value in aggregate.values():
        validate_schema("toy_quality_structural_breakdown.schema.json", value["structural_breakdown"])
    if aggregate != aggregate_summaries(recalculated, rows, config["variants"], config["seeds"]):
        raise ValueError("AGGREGATE_MISMATCH")
    comparisons = [paired(rows, a, b) for a, b in (("A3", "A2"), ("A4", "A2"), ("A3", "A4")) if a in config["variants"] and b in config["variants"]]
    persisted_comparisons = json.loads((root / "paired-comparisons.json").read_bytes())
    for comparison in persisted_comparisons:
        validate_schema("toy_quality_paired_comparison.schema.json", comparison)
    if persisted_comparisons != comparisons:
        raise ValueError("PAIRED_MISMATCH")
    selected_rows = [dataset_rows[task_id] for task_id in config["eval_task_ids"]]
    if (root / "human-readable-examples.md").read_text() != _examples(rows, selected_rows, config["variants"], config["seeds"]):
        raise ValueError("HUMAN_EXAMPLES_SEMANTIC_MISMATCH")
    manifest = json.loads((root / "evaluation-manifest.json").read_bytes())
    validate_schema("toy_quality_evaluation_manifest.schema.json", manifest)
    if manifest["evaluator_source_sha256"] != config["evaluator_source_sha256"]:
        raise ValueError("EVALUATOR_SOURCE_MANIFEST_MISMATCH")
    if manifest["evaluator_version"] != config["evaluator_version"] or manifest["variants"] != config["variants"] or manifest["seeds"] != config["seeds"] or manifest["variant_mapping"] != config["variant_mapping"]:
        raise ValueError("EVALUATION_MANIFEST_IDENTITY_MISMATCH")
    expected_artifacts = {"evaluation-config.json", "dataset-manifest.json", "task-results.jsonl", "per-seed-summary.json", "aggregate-summary.json", "paired-comparisons.json", "human-readable-examples.md"}
    if set(manifest["artifact_hashes"]) != expected_artifacts:
        raise ValueError("TOP_LEVEL_ARTIFACT_COVERAGE_MISMATCH")
    for name, digest in manifest["artifact_hashes"].items():
        if _file_hash(root / name) != digest:
            raise ValueError("ARTIFACT_HASH_MISMATCH")
    for collection in ("checkpoint_manifest_hashes", "evidence_artifact_hashes"):
        expected_paths = (
            {str(path.relative_to(root)) for path in root.glob("training-runs/*/seed-*/*") if path.is_file()}
            if collection == "checkpoint_manifest_hashes"
            else {str(path.relative_to(root)) for path in root.glob("evidence/*/seed-*/*/*") if path.is_file()}
        )
        if set(manifest[collection]) != expected_paths:
            raise ValueError("RECURSIVE_MANIFEST_COVERAGE_MISMATCH")
        for name, digest in manifest[collection].items():
            if _file_hash(root / name) != digest:
                raise ValueError("RECURSIVE_ARTIFACT_HASH_MISMATCH")
    replay = canonical_replay_hash(root, manifest, rows)
    if (root / "replay-hash.txt").read_text().strip() != replay:
        raise ValueError("REPLAY_HASH_MISMATCH")
    return {"valid": True, "task_results": len(rows), "replay_hash": replay}


def export_compact(root: Path, destination: Path) -> None:
    """Generate the two reviewable reference artifacts exclusively from a validated run."""
    validate_evaluation(root)
    config = json.loads((root / "evaluation-config.json").read_bytes())
    if not config["diagnostic_complete"] or tuple(config["variants"]) != VARIANTS or tuple(config["seeds"]) != SEEDS or len(config["eval_task_ids"]) != 2 or config["training_execution_mode"] != "TRAINED_IN_RUN" or config["deterministic_training_replay_status"] != "CANONICAL_DETERMINISTIC" or config["reuse_source_manifest_hash"] is not None:
        raise ValueError("COMPACT_EXPORT_REQUIRES_COMPLETE_CANONICAL_RUN")
    per_seed = json.loads((root / "per-seed-summary.json").read_bytes())
    aggregate = json.loads((root / "aggregate-summary.json").read_bytes())
    comparisons = json.loads((root / "paired-comparisons.json").read_bytes())
    rows = [json.loads(line) for line in (root / "task-results.jsonl").read_text().splitlines()]
    semantic_traces = [
        json.loads(path.read_bytes())
        for variant in ("A3", "A4")
        for path in sorted(root.glob(f"evidence/{variant}/seed-*/*/semantic-trace.json"))
    ]
    feedback_positions = sum(
        trace["feedback_application_count"] for trace in semantic_traces
    )
    nonzero_feedback_positions = sum(
        trace["nonzero_feedback_application_count"] for trace in semantic_traces
    )
    influenced_positions = sum(
        trace["feedback_influenced_decoding_position_count"] for trace in semantic_traces
    )
    compact = {
        "status": "development-only-diagnostic",
        "evaluator_source_files": config["evaluator_source_files"],
        "evaluator_source_sha256": config["evaluator_source_sha256"],
        "implementation_commit": config["implementation_commit"],
        "requirements_lock_sha256": config["requirements_lock_sha256"],
        "runtime_versions": config["runtime_versions"],
        "evaluator_version": VERSION,
        "dataset_hash": config["dataset_manifest_hash"],
        "train_task_count": len(config["train_task_ids"]),
        "heldout_task_count": len(config["eval_task_ids"]),
        "seeds": config["seeds"],
        "training_budget": {"epochs": config["epochs"], "updates_per_run": config["epochs"] * len(config["train_task_ids"]), "checkpoint_policy": config["checkpoint_policy"]},
        "per_seed": per_seed, "aggregate": aggregate, "paired_comparisons": comparisons,
        "observed_failure_codes": sorted({r["failure_code"] for r in rows if r["failure_code"]}),
        "observed_feedback_application_positions": feedback_positions,
        "observed_nonzero_feedback_application_positions": nonzero_feedback_positions,
        "observed_feedback_influenced_positions": influenced_positions,
        "replay_hash": (root / "replay-hash.txt").read_text().strip(),
    }
    Draft202012Validator(json.loads((Path(__file__).with_name("schemas") / "toy_quality_compact_summary.schema.json").read_bytes())).validate(compact)
    data = destination / "data" / "a2_a3_a4_heldout_summary.json"
    _write(data, compact)
    lines = [
        "# Диагностика качества A2/A3/A4 на held-out задачах", "",
        "> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.", "",
        f"- Evaluator source SHA256: `{config['evaluator_source_sha256']}`",
        f"- Implementation commit: `{config['implementation_commit']}`",
        f"- Requirements lock: `{config['requirements_lock_sha256']}`",
        f"- Runtime: `{json.dumps(config['runtime_versions'], sort_keys=True)}`",
        f"- Evaluator: `{VERSION}`", f"- Dataset hash: `{config['dataset_manifest_hash']}`",
        f"- Train tasks: {len(config['train_task_ids'])}; held-out tasks: {len(config['eval_task_ids'])}",
        f"- Seeds: {', '.join(map(str, config['seeds']))}",
        f"- Budget: {config['epochs']} epochs × {len(config['train_task_ids'])} canonical train tasks = {config['epochs'] * len(config['train_task_ids'])} updates/run; final checkpoint only", "",
        "## Variant × seed", "", "| Variant | Seed | Success | Rate | Executable | Action applicable | Mean predicted length |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, summary in per_seed.items():
        variant, seed = key.split("/seed-")
        applicable = "null" if summary["action_applicable_rate"] is None else f"{summary['action_applicable_rate']:.3f}"
        lines.append(f"| {MAPPING[variant][1]} | {seed} | {summary['success_count']}/{summary['heldout_task_count']} | {summary['task_success_rate']:.3f} | {summary['fully_executable_plan_rate']:.3f} | {applicable} | {summary['mean_predicted_plan_length']:.2f} |")
    limitation = (
        "Все failures включены в denominator. Hyperparameters не подбирались после "
        "просмотра held-out результата. A3a-shuffled, A3a-foreign, A3s, A3b и "
        f"Verbalizer не реализованы. Observed feedback application positions: {feedback_positions}. "
        f"Observed nonzero feedback application positions: {nonzero_feedback_positions}. "
        f"Observed feedback-influenced positions: {influenced_positions}."
    )
    if influenced_positions == 0:
        limitation += (
            " Because every run terminated before feedback influenced a downstream token, "
            "this diagnostic is non-diagnostic for feedback-channel causality and for the "
            "difference between active feedback and compute-then-zero at downstream decoding "
            "positions."
        )
    lines += ["", "## Aggregate", "", "```json", json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True), "```", "", "## Paired comparisons", "", "```json", json.dumps(comparisons, ensure_ascii=False, indent=2, sort_keys=True), "```", "", "## Детерминированно выбранные примеры", "", (root / "human-readable-examples.md").read_text(), "", "## Ограничения", "", limitation, "", "## Воспроизведение", "", "```bash", "python -m scripts.run_toy_quality_evaluation \\", "  --output-dir .quality-eval \\", "  --implementation-commit <IMPLEMENTATION_SHA>", "", "python -m scripts.run_toy_quality_evaluation \\", "  --output-dir .quality-eval \\", "  --validate-only \\", "  --compact-dir docs/evaluations", "```", ""]
    report = destination / "A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def run(
    output: Path,
    *,
    variants=VARIANTS,
    seeds=SEEDS,
    max_eval_tasks: int | None = None,
    reuse_checkpoint_root: Path | None = None,
    implementation_commit: str | None = None,
) -> dict:
    if max_eval_tasks is not None and max_eval_tasks <= 0:
        raise ValueError("max_eval_tasks must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    dataset = generate(17)
    train_rows = sorted(dataset["train"], key=lambda r: r["task_id"])
    eval_rows = sorted(dataset["validation"], key=lambda r: r["task_id"])
    if max_eval_tasks is not None:
        eval_rows = eval_rows[:max_eval_tasks]
    variants, seeds = tuple(variants), tuple(seeds)
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError("only predeclared development seeds 17, 29, and 43 are supported")
    reuse_validation = None
    if reuse_checkpoint_root is not None:
        reuse_validation = validate_evaluation(reuse_checkpoint_root)
        reuse_config = json.loads(
            (reuse_checkpoint_root / "evaluation-config.json").read_bytes()
        )
        if (
            not reuse_config["diagnostic_complete"]
            or tuple(reuse_config["variants"]) != VARIANTS
            or tuple(reuse_config["seeds"]) != SEEDS
            or reuse_config["training_execution_mode"] != "TRAINED_IN_RUN"
            or reuse_config["deterministic_training_replay_status"]
            != "CANONICAL_DETERMINISTIC"
        ):
            raise ValueError("REUSE_REQUIRES_COMPLETE_CANONICAL_SOURCE")
    complete = variants == VARIANTS and seeds == SEEDS and max_eval_tasks is None and reuse_checkpoint_root is None
    if complete and implementation_commit is None:
        raise ValueError("complete canonical generation requires --implementation-commit")
    provenance = implementation_provenance(implementation_commit)
    if complete:
        if _git("diff", "--quiet").returncode or _git("diff", "--cached", "--quiet").returncode:
            raise ValueError("complete canonical generation requires a clean tracked working tree")
        if _git("merge-base", "--is-ancestor", provenance["implementation_commit"], "HEAD").returncode:
            raise ValueError("implementation commit must be an ancestor of the checkout")
        if source_identity_at_commit(provenance["implementation_commit"]) != source_identity():
            raise ValueError("working evaluator sources differ from implementation commit")
    config = {"schema_version": VERSION, "evaluator_version": VERSION, "variants": list(variants), "seeds": list(seeds),
              "variant_mapping": {v: {"architecture_stage": MAPPING[v][0], "implementation_variant": v, "experimental_arm": MAPPING[v][1], "target_type": MAPPING[v][2]} for v in variants},
              "train_task_ids": [r["task_id"] for r in train_rows], "eval_task_ids": [r["task_id"] for r in eval_rows],
              "epochs": EPOCHS, "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "betas": [0.9, .95], "eps": 1e-8, "weight_decay": .01, "gradient_clip_norm": 1.0},
              "checkpoint_policy": "final_epoch_only_no_heldout_selection", "max_decoding_steps": MAX_STEPS,
              "dataset_manifest_hash": dataset["dataset_hash"], "diagnostic_complete": complete,
              "budget_version": "quality-v0.1-fixed-2026-07", "updates_per_run": EPOCHS * len(train_rows),
              "training_execution_mode": "REUSED" if reuse_checkpoint_root is not None else "TRAINED_IN_RUN",
              "checkpoint_origin_run_hash": (
                  reuse_validation["replay_hash"]
                  if reuse_checkpoint_root is not None else None
              ),
              "reuse_source_manifest_hash": (
                  _file_hash(reuse_checkpoint_root / "evaluation-manifest.json")
                  if reuse_checkpoint_root is not None else None
              ),
              "deterministic_training_replay_status": (
                  "REUSED_VALIDATED" if reuse_checkpoint_root is not None else "CANONICAL_DETERMINISTIC"
              ),
              **source_identity(), **provenance}
    _write(output / "evaluation-config.json", config)
    _write(output / "dataset-manifest.json", {"dataset_hash": dataset["dataset_hash"], "train_task_ids": config["train_task_ids"], "eval_task_ids": config["eval_task_ids"]})
    rows = []
    for variant in variants:
        for seed in seeds:
            run_dir = output / "training-runs" / variant / f"seed-{seed}"
            if reuse_checkpoint_root is None:
                model, checkpoint = _train(train_rows, variant, seed, run_dir, dataset["dataset_hash"])
            else:
                source = reuse_checkpoint_root / "training-runs" / variant / f"seed-{seed}"
                manifest = json.loads((source / "checkpoint-manifest.json").read_bytes())
                manifest = dict(manifest)
                if manifest["variant_identity"] != config["variant_mapping"][variant] or manifest["seed"] != seed:
                    raise ValueError("REUSED_CHECKPOINT_IDENTITY_MISMATCH")
                model = LockedPlanner(seed, variant).cpu()
                model.load_state_dict(torch.load(source / "trained.pt", map_location="cpu", weights_only=True))
                checkpoint = manifest
                if state_dict_sha256(model.state_dict()) != manifest["trained_state_dict_sha256"]:
                    raise ValueError("REUSED_CHECKPOINT_HASH_MISMATCH")
                if manifest["dataset_hash"] != dataset["dataset_hash"] or manifest["train_task_ids"] != config["train_task_ids"] or manifest["epochs"] != EPOCHS or manifest["updates"] != 9:
                    raise ValueError("REUSED_CHECKPOINT_LINEAGE_MISMATCH")
                shutil.copytree(source, run_dir)
                manifest.update({"training_execution_mode": "REUSED",
                                 "checkpoint_origin_run_hash": config["checkpoint_origin_run_hash"],
                                 "reuse_source_manifest_hash": config["reuse_source_manifest_hash"],
                                 "deterministic_training_replay_status": "REUSED_VALIDATED"})
                _write(run_dir / "checkpoint-manifest.json", manifest)
            for row in eval_rows:
                evidence = output / "evidence" / variant / f"seed-{seed}" / row["task_id"]
                result = _evaluate(row, variant, seed, model, checkpoint, evidence)
                result["evidence_root"] = str(evidence.relative_to(output))
                rows.append(result)
    rows.sort(key=lambda r: (r["variant"], r["seed"], r["task_id"]))
    (output / "task-results.jsonl").write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))
    per_seed = {f"{v}/seed-{s}": summarize([r for r in rows if r["variant"] == v and r["seed"] == s]) for v in variants for s in seeds}
    aggregate = aggregate_summaries(per_seed, rows, variants, seeds)
    comparisons = [paired(rows, a, b) for a, b in (("A3", "A2"), ("A4", "A2"), ("A3", "A4")) if a in variants and b in variants]
    _write(output / "per-seed-summary.json", per_seed); _write(output / "aggregate-summary.json", aggregate); _write(output / "paired-comparisons.json", comparisons)
    examples = _examples(rows, eval_rows, variants, seeds)
    (output / "human-readable-examples.md").write_text(examples)
    names = ["evaluation-config.json", "dataset-manifest.json", "task-results.jsonl", "per-seed-summary.json", "aggregate-summary.json", "paired-comparisons.json", "human-readable-examples.md"]
    checkpoint_files = sorted(path for path in output.glob("training-runs/*/seed-*/*") if path.is_file())
    evidence_files = sorted(output.glob("evidence/*/seed-*/*/*"))
    manifest = {"evaluator_version": VERSION, "seeds": list(seeds), "variants": list(variants), "variant_mapping": config["variant_mapping"],
                "evaluator_source_sha256": config["evaluator_source_sha256"],
                "artifact_hashes": {n: _file_hash(output / n) for n in names},
                "checkpoint_manifest_hashes": {str(p.relative_to(output)): _file_hash(p) for p in checkpoint_files},
                "evidence_artifact_hashes": {str(p.relative_to(output)): _file_hash(p) for p in evidence_files}}
    _write(output / "evaluation-manifest.json", manifest)
    (output / "replay-hash.txt").write_text(
        canonical_replay_hash(output, manifest, rows) + "\n"
    )
    return validate_evaluation(output)


def _examples(rows: list[dict], eval_rows: list[dict], variants, seeds) -> str:
    lines = ["# Human-readable held-out examples", "", "Development diagnostic only; not a Stage 2A semantic gate.", ""]
    index = {(r["variant"], r["seed"], r["task_id"]): r for r in rows}
    for seed in seeds:
        for task in eval_rows[:5]:
            lines += [f"## {task['task_id']} (seed {seed})", "", f"Initial state: `{json.dumps(task['initial'])}`", f"Goal: `{json.dumps(task['goal'])}`", f"Gold/reference plan: `{json.dumps(task['oracle_work_plan'][:-1])}`"]
            for variant in variants:
                r = index[(variant, seed, task["task_id"])]
                lines += [f"### {MAPPING[variant][1]}", f"Predicted plan: `{json.dumps(r['predicted_actions'])}`", f"END-only: `{str(r['end_only_plan']).lower()}`", f"Model forwards: `{r['model_forward_count']}`", f"Generated/attempted/applicable: `{r['generated_action_count']}/{r['action_attempt_count']}/{r['applicable_action_count']}`", f"Execution evidence hash: `{r['evidence_hash']}`", f"Checkpoint hash: `{r['checkpoint_hash']}`", f"Execution: `{json.dumps(r['execution_trace'])}`", f"Initial goal satisfied: `{str(r['initial_goal_satisfied']).lower()}`", f"Failure: `{r['failure_code']}`", f"Final state: `{json.dumps(r['final_state'])}`", f"Goal reached: `{str(r['goal_reached']).lower()}`"]
            lines.append("")
    return "\n\n".join(lines) + "\n"
