from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from validation.hashing import hash_json

TRAIN_VARIANTS = ("A1", "A2", "A2b", "A2c", "A3", "A3r")
STAGE1B_ARMS = (
    "E0_EQUAL_TOKENS_RAW",
    "E1_A3_FULL_PLAN_RAW",
    "E2_SHUFFLED_A3_FULL_PLAN_RAW",
    "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW",
    "E4_A2C_STRUCTURED_FULL_PLAN_RAW",
    "E5_SELF_PLAN_RAW",
    "P_FULL_PLAN_REPLAY_RAW",
)
FINAL_SEEDS = (101, 202, 303, 404, 505)
DEV_SEED = 17
BATCH_SIZE = 128
FINAL_UPDATES = 12000
FLOPS_SENSITIVITY_VARIANTS = ("A2c", "A3")
STAGE1B_TASK_CAPACITY = 4000


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def canonical_report_hash(obj: dict[str, Any]) -> str:
    value = dict(obj)
    value.pop("report_hash", None)
    return hash_json(value)


def _unique_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field))
        if key in out:
            raise ValueError(f"duplicate {field}: {key}")
        out[key] = row
    return out


def _expected_grid(hyper: dict[str, Any]) -> list[str]:
    grid = hyper.get("grid", {})
    lrs = grid.get("learning_rate", [])
    dropouts = grid.get("dropout", [])
    configs = [f"lr={lr}|dropout={dropout}" for lr in lrs for dropout in dropouts]
    if len(configs) != 4 or hyper.get("max_configs") != 4:
        raise ValueError("development grid must contain exactly four configurations")
    if tuple(hyper.get("run_each_config_seeds", [])) != (DEV_SEED,):
        raise ValueError("development grid must use only seed 17")
    return configs


def validate_compute_profile(profile: dict[str, Any], root: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        variants = _unique_by(profile.get("variants", []), "variant")
        if set(variants) != set(TRAIN_VARIANTS):
            errors.append("compute profile must contain exactly A1/A2/A2b/A2c/A3/A3r")
            return errors
        eq = profile.get("equal_data_schedule", {})
        if eq.get("optimizer_updates_exact") != FINAL_UPDATES or eq.get("batch_size_examples") != BATCH_SIZE:
            errors.append("equal-data schedule must be exactly 12000 updates x batch 128")
        flops = profile.get("flops_matched_schedule", {})
        if flops.get("comparison_id") != "A3_vs_A2c":
            errors.append("FLOPs-matched schedule must be the locked A3_vs_A2c comparison")
        per_step = {variant: float(variants[variant]["train_flops_per_optimizer_step"]) for variant in FLOPS_SENSITIVITY_VARIANTS}
        if any(value <= 0 for value in per_step.values()):
            errors.append("FLOPs sensitivity variants require positive measured per-step FLOPs")
        else:
            source = min(FLOPS_SENSITIVITY_VARIANTS, key=lambda variant: (per_step[variant], variant))
            cap = per_step[source] * FINAL_UPDATES
            if flops.get("cap_source_variant") != source:
                errors.append("FLOPs cap source must be the cheaper measured variant")
            if not math.isclose(float(flops.get("cap_flops_per_workload", -1)), cap, rel_tol=1e-12, abs_tol=1e-6):
                errors.append("FLOPs cap must equal cheaper A3/A2c variant at 12000 updates")
            tolerance = float(flops.get("tolerance_fraction", -1))
            if not math.isclose(tolerance, 0.02, rel_tol=0.0, abs_tol=1e-12):
                errors.append("FLOPs-matched tolerance must be exactly 0.02")
            updates = flops.get("optimizer_updates_by_variant", {})
            if set(updates) != set(FLOPS_SENSITIVITY_VARIANTS):
                errors.append("FLOPs sensitivity update map must contain exactly A2c and A3")
            else:
                for variant in FLOPS_SENSITIVITY_VARIANTS:
                    expected = math.floor(cap / per_step[variant])
                    if expected < 1 or expected > FINAL_UPDATES:
                        errors.append(f"invalid FLOPs-matched optimizer update count for {variant}")
                        continue
                    if updates.get(variant) != expected:
                        errors.append(f"FLOPs-matched optimizer updates mismatch for {variant}")
                    matched_total = expected * per_step[variant]
                    if abs(matched_total - cap) / cap > tolerance + 1e-12:
                        errors.append(f"FLOPs-matched workload for {variant} is outside tolerance")
        stage = profile.get("stage1b_inference", {})
        if stage.get("confirmatory_task_capacity") != STAGE1B_TASK_CAPACITY:
            errors.append("Stage1B compute profile must cover 4000 confirmatory tasks")
        arm_rows = _unique_by(stage.get("arms", []), "arm")
        if set(arm_rows) != set(STAGE1B_ARMS):
            errors.append("Stage1B compute profile must contain exactly seven arms")
        cap = float(stage.get("flops_cap_per_task", -1))
        if cap <= 0:
            errors.append("Stage1B FLOPs cap must be positive")
        for arm, row in arm_rows.items():
            if float(row.get("estimated_flops_per_task", 0)) > cap:
                errors.append(f"{arm} exceeds the locked pre-outcome FLOPs cap")
        if stage.get("budget_exhaustion_policy") != "PAIRED_FAILURE_NO_TASK_EXCLUSION":
            errors.append("budget exhaustion must remain a paired failure without task exclusion")
        if profile.get("report_hash") != canonical_report_hash(profile):
            errors.append("compute profile report_hash mismatch")
        for arm, row in arm_rows.items():
            coeffs = ("fixed_flops_per_task", "planner_flops_per_call", "plan_token_flops", "executor_input_token_flops", "executor_output_token_flops")
            if all(float(row.get(name, 0)) == 0 for name in coeffs):
                errors.append(f"{arm} has no non-zero locked FLOPs accounting coefficient")
        if root is not None:
            evidence = profile.get("measurement_evidence", {})
            raw_rel = str(evidence.get("raw_artifact_path", ""))
            raw_path = (root / raw_rel).resolve()
            if root.resolve() not in raw_path.parents or not raw_path.is_file():
                errors.append("compute measurement raw artifact is missing or escapes repository")
            elif evidence.get("raw_artifact_sha256") != file_digest(raw_path):
                errors.append("compute measurement raw artifact hash mismatch")
            else:
                try:
                    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    repetitions = int(evidence.get("repetitions", 0))
                    run_id = profile.get("run_id")
                    device_name = profile.get("hardware", {}).get("device_name")
                    trial_ids = [row.get("trial_id") for row in rows]
                    if len(trial_ids) != len(set(trial_ids)):
                        errors.append("compute measurement trial_id values must be unique")
                    expected_row_count = repetitions * (len(TRAIN_VARIANTS) + len(STAGE1B_ARMS))
                    if len(rows) != expected_row_count:
                        errors.append(f"compute measurement must contain exactly {expected_row_count} locked trials")
                    if any(row.get("run_id") != run_id or row.get("device_name") != device_name for row in rows):
                        errors.append("compute measurement raw lineage differs from profile run/device")
                    if any(
                        (row.get("measurement_type") == "TRAIN_OPTIMIZER_STEP" and (row.get("variant") not in TRAIN_VARIANTS or row.get("arm") is not None))
                        or (row.get("measurement_type") == "STAGE1B_TASK" and (row.get("arm") not in STAGE1B_ARMS or row.get("variant") is not None))
                        for row in rows
                    ):
                        errors.append("compute measurement row type/variant/arm identity mismatch")
                    train_rows = {variant: [row for row in rows if row.get("measurement_type") == "TRAIN_OPTIMIZER_STEP" and row.get("variant") == variant] for variant in TRAIN_VARIANTS}
                    arm_measurements = {arm: [row for row in rows if row.get("measurement_type") == "STAGE1B_TASK" and row.get("arm") == arm] for arm in STAGE1B_ARMS}
                    for variant, group in train_rows.items():
                        if len(group) != repetitions:
                            errors.append(f"compute measurement must have exactly {repetitions} trials for {variant}")
                            continue
                        expected = variants[variant]
                        if not math.isclose(median(float(row["measured_flops"]) for row in group), float(expected["train_flops_per_optimizer_step"]), rel_tol=1e-12, abs_tol=1e-6):
                            errors.append(f"compute measurement FLOPs median mismatch for {variant}")
                        if not math.isclose(median(float(row["elapsed_seconds"]) for row in group), float(expected["train_seconds_per_optimizer_step"]), rel_tol=1e-12, abs_tol=1e-9):
                            errors.append(f"compute measurement time median mismatch for {variant}")
                        if int(median(int(row["peak_vram_bytes"]) for row in group)) != int(expected["peak_vram_bytes"]):
                            errors.append(f"compute measurement VRAM median mismatch for {variant}")
                    for arm, group in arm_measurements.items():
                        if len(group) != repetitions:
                            errors.append(f"compute measurement must have exactly {repetitions} trials for {arm}")
                            continue
                        expected = arm_rows[arm]
                        if not math.isclose(median(float(row["measured_flops"]) for row in group), float(expected["estimated_flops_per_task"]), rel_tol=1e-12, abs_tol=1e-6):
                            errors.append(f"compute measurement FLOPs median mismatch for {arm}")
                        if not math.isclose(median(float(row["elapsed_seconds"]) for row in group), float(expected["estimated_seconds_per_task"]), rel_tol=1e-12, abs_tol=1e-9):
                            errors.append(f"compute measurement time median mismatch for {arm}")
                except Exception as exc:
                    errors.append(f"invalid compute measurement raw artifact: {exc}")
            environment = root / "locks/environment.lock.json"
            if not environment.is_file() or evidence.get("environment_lock_sha256") != file_digest(environment):
                errors.append("compute measurement environment lock hash mismatch")
            if int(evidence.get("repetitions", 0)) < 3:
                errors.append("compute measurement requires at least three repetitions")
            rel = str(profile.get("measurement_code_path", ""))
            path = (root / rel).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                errors.append("measurement_code_path is missing or escapes repository")
            elif profile.get("measurement_code_sha256") != file_digest(path):
                errors.append("measurement_code_sha256 does not match measurement_code_path")
    except Exception as exc:
        errors.append(str(exc))
    return errors


def validate_capacity_preflight(root: Path, obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    paths = {
        "training_contract_sha256": root / "docs/training/planner_training_contract_v1.yaml",
        "hyperparameter_contract_sha256": root / "docs/training/hyperparameter_search_v1.yaml",
        "dataset_contract_sha256": root / "docs/data/dataset_split_contract_v1.yaml",
        "compute_profile_sha256": root / "reports/compute-profile.json",
        "resource_plan_sha256": root / "reports/resource-plan.json",
        "corpus_manifest_sha256": root / "data/manifests/training-corpus.json",
    }
    for field, path in paths.items():
        if not path.is_file():
            errors.append(f"capacity evidence missing: {path.relative_to(root)}")
        elif obj.get(field) != file_digest(path):
            errors.append(f"{field} does not match evidence file")
    if errors:
        return errors

    try:
        training = load_yaml(paths["training_contract_sha256"])
        hyper = load_yaml(paths["hyperparameter_contract_sha256"])
        dataset = load_yaml(paths["dataset_contract_sha256"])
        compute = load_json(paths["compute_profile_sha256"])
        resource = load_json(paths["resource_plan_sha256"])
        corpus = load_json(paths["corpus_manifest_sha256"])
        errors.extend(validate_compute_profile(compute, root))

        if training.get("schema_version") != "work-planner/1.17":
            errors.append("training contract is not v1.17")
        final = training.get("training", {}).get("final_confirmatory_models", {})
        if final.get("optimizer_updates_exact") != FINAL_UPDATES or final.get("same_batch_size") != BATCH_SIZE:
            errors.append("final training contract is not exactly 12000 updates x batch 128")
        if final.get("final_checkpoint_optimizer_step_exact") != FINAL_UPDATES or final.get("intermediate_checkpoint_selection") != "forbidden":
            errors.append("final checkpoint must be the exact step-12000 checkpoint")
        development = training.get("training", {}).get("development", {})
        if development.get("retained_checkpoint_count_exact") != 2:
            errors.append("development must retain exactly best and last checkpoints")
        if final.get("retained_checkpoint_count_exact") != 1 or final.get("intermediate_checkpoint_retention_after_completion") != "forbidden":
            errors.append("final training must retain only the exact step-12000 checkpoint")
        if tuple(training.get("final_seeds", [])) != FINAL_SEEDS:
            errors.append("final training must contain exactly five frozen seeds")

        configs = _expected_grid(hyper)
        train_rows = int(corpus.get("counts_by_split", {}).get("train", 0))
        if train_rows <= 0:
            errors.append("training corpus must expose a positive train row count")
        variant_profile = _unique_by(compute.get("variants", []), "variant")

        expected_dev: dict[str, dict[str, Any]] = {}
        dev_updates = math.ceil(train_rows / BATCH_SIZE) * int(training["training"]["development"]["max_epochs"])
        for config_id in configs:
            for variant in TRAIN_VARIANTS:
                wid = f"development|{config_id}|{variant}|seed={DEV_SEED}"
                cp = variant_profile[variant]
                expected_dev[wid] = {
                    "variant": variant,
                    "seed": DEV_SEED,
                    "config_id": config_id,
                    "optimizer_updates": dev_updates,
                    "batch_size_examples": BATCH_SIZE,
                    "unique_training_rows": train_rows,
                    "processed_examples": dev_updates * BATCH_SIZE,
                    "estimated_gpu_seconds": dev_updates * float(cp["train_seconds_per_optimizer_step"]),
                    "peak_vram_bytes": cp["peak_vram_bytes"],
                    "checkpoint_bytes": cp["checkpoint_bytes"] * 2,
                }

        expected_final: dict[str, dict[str, Any]] = {}
        for seed in FINAL_SEEDS:
            for variant in TRAIN_VARIANTS:
                wid = f"final|{variant}|seed={seed}"
                cp = variant_profile[variant]
                expected_final[wid] = {
                    "variant": variant,
                    "seed": seed,
                    "config_id": None,
                    "optimizer_updates": FINAL_UPDATES,
                    "batch_size_examples": BATCH_SIZE,
                    "unique_training_rows": train_rows,
                    "processed_examples": FINAL_UPDATES * BATCH_SIZE,
                    "estimated_gpu_seconds": FINAL_UPDATES * float(cp["train_seconds_per_optimizer_step"]),
                    "peak_vram_bytes": cp["peak_vram_bytes"],
                    "checkpoint_bytes": cp["checkpoint_bytes"],
                }

        expected_sensitivity: dict[str, dict[str, Any]] = {}
        sensitivity_updates = compute["flops_matched_schedule"]["optimizer_updates_by_variant"]
        for seed in FINAL_SEEDS:
            for variant in FLOPS_SENSITIVITY_VARIANTS:
                wid = f"flops-sensitivity|{variant}|seed={seed}"
                cp = variant_profile[variant]
                optimizer_updates = int(sensitivity_updates[variant])
                expected_sensitivity[wid] = {
                    "variant": variant,
                    "seed": seed,
                    "config_id": None,
                    "optimizer_updates": optimizer_updates,
                    "batch_size_examples": BATCH_SIZE,
                    "unique_training_rows": train_rows,
                    "processed_examples": optimizer_updates * BATCH_SIZE,
                    "estimated_gpu_seconds": optimizer_updates * float(cp["train_seconds_per_optimizer_step"]),
                    "peak_vram_bytes": cp["peak_vram_bytes"],
                    "checkpoint_bytes": cp["checkpoint_bytes"],
                }

        def check_training_rows(label: str, rows: list[dict[str, Any]], expected: dict[str, dict[str, Any]]) -> None:
            try:
                actual = _unique_by(rows, "workload_id")
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                return
            if set(actual) != set(expected):
                errors.append(f"{label} workload identity set mismatch")
                return
            for wid, exp in expected.items():
                row = actual[wid]
                for field, value in exp.items():
                    got = row.get(field)
                    if isinstance(value, float):
                        if not math.isclose(float(got), value, rel_tol=1e-12, abs_tol=1e-9):
                            errors.append(f"{label} {wid} {field} mismatch")
                    elif got != value:
                        errors.append(f"{label} {wid} {field} mismatch")

        check_training_rows("development", obj.get("development_workloads", []), expected_dev)
        check_training_rows("final", obj.get("final_training_workloads", []), expected_final)
        check_training_rows("flops-sensitivity", obj.get("flops_sensitivity_workloads", []), expected_sensitivity)

        stage_profile = _unique_by(compute["stage1b_inference"]["arms"], "arm")
        actual_stage = _unique_by(obj.get("stage1b_inference_workloads", []), "arm")
        if set(actual_stage) != set(STAGE1B_ARMS):
            errors.append("Stage1B preflight must contain exactly seven arms")
        else:
            for arm in STAGE1B_ARMS:
                cp = stage_profile[arm]
                row = actual_stage[arm]
                expected = {
                    "task_capacity": STAGE1B_TASK_CAPACITY,
                    "estimated_flops": float(cp["estimated_flops_per_task"]) * STAGE1B_TASK_CAPACITY,
                    "estimated_gpu_seconds": float(cp["estimated_seconds_per_task"]) * STAGE1B_TASK_CAPACITY,
                    "maximum_plan_calls_per_task": cp["maximum_plan_calls"],
                    "maximum_executor_calls_per_task": cp["maximum_executor_calls"],
                }
                for field, value in expected.items():
                    got = row.get(field)
                    if isinstance(value, float):
                        if not math.isclose(float(got), value, rel_tol=1e-12, abs_tol=1e-9):
                            errors.append(f"Stage1B {arm} {field} mismatch")
                    elif got != value:
                        errors.append(f"Stage1B {arm} {field} mismatch")

        summary = obj.get("summary", {})
        all_training = list(expected_dev.values()) + list(expected_final.values()) + list(expected_sensitivity.values())
        expected_gpu_seconds = sum(float(x["estimated_gpu_seconds"]) for x in all_training) + sum(float(x.get("estimated_gpu_seconds", 0)) for x in actual_stage.values())
        expected_storage = sum(int(x["checkpoint_bytes"]) for x in all_training)
        limits = resource.get("capacity_limits", {})
        expected_summary = {
            "development_workload_count": 24,
            "final_training_workload_count": 30,
            "flops_sensitivity_workload_count": 10,
            "total_training_workload_count": 64,
            "total_sensitivity_processed_examples": sum(int(x["processed_examples"]) for x in expected_sensitivity.values()),
            "stage1b_arm_count": 7,
            "final_optimizer_updates_per_workload": FINAL_UPDATES,
            "final_batch_size_examples": BATCH_SIZE,
            "final_processed_examples_per_workload": FINAL_UPDATES * BATCH_SIZE,
            "total_final_processed_examples": len(expected_final) * FINAL_UPDATES * BATCH_SIZE,
            "estimated_gpu_seconds": expected_gpu_seconds,
            "maximum_gpu_seconds": float(limits.get("maximum_gpu_seconds", -1)),
            "estimated_storage_bytes": expected_storage,
            "maximum_storage_bytes": int(limits.get("maximum_storage_bytes", -1)),
        }
        for field, value in expected_summary.items():
            got = summary.get(field)
            if isinstance(value, float):
                if not math.isclose(float(got), value, rel_tol=1e-12, abs_tol=1e-9):
                    errors.append(f"summary {field} mismatch")
            elif got != value:
                errors.append(f"summary {field} mismatch")
        affordable = expected_gpu_seconds <= float(limits.get("maximum_gpu_seconds", -1)) and expected_storage <= int(limits.get("maximum_storage_bytes", -1))
        gpu_cost = expected_gpu_seconds / 3600.0 * float(limits.get("gpu_hour_cost", -1))
        if gpu_cost > float(resource.get("estimated_cost", -1)) + 1e-9:
            errors.append("resource-plan estimated_cost is below recomputed GPU cost")
            affordable = False
        expected_status = "PASS" if affordable else "BLOCKED_PROTOCOL_CAPACITY"
        if obj.get("status") != expected_status:
            errors.append(f"capacity status must be {expected_status}")
        reserve = dataset.get("partitions", {}).get("stage1b_confirmatory_reserve", {}).get("target_base_tasks")
        if reserve != STAGE1B_TASK_CAPACITY:
            errors.append("dataset contract Stage1B reserve must equal 4000")
        if obj.get("report_hash") != canonical_report_hash(obj):
            errors.append("capacity preflight report_hash mismatch")
    except Exception as exc:
        errors.append(str(exc))
    return errors
