from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import yaml

from validation.capacity_validator import (
    BATCH_SIZE,
    DEV_SEED,
    FINAL_SEEDS,
    FINAL_UPDATES,
    FLOPS_SENSITIVITY_VARIANTS,
    STAGE1B_ARMS,
    TRAIN_VARIANTS,
    canonical_report_hash,
    file_digest,
    validate_capacity_preflight,
    validate_compute_profile,
)
from validation.hashing import hash_json

ROOT = Path(__file__).resolve().parents[1]
H = "sha256:" + "1" * 64


def _write(path: Path, value, *, yaml_file: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml_file:
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _compute() -> dict:
    variant_rows = []
    for index, variant in enumerate(TRAIN_VARIANTS, start=1):
        variant_rows.append({
            "variant": variant,
            "train_flops_per_optimizer_step": float(index * 1000),
            "train_seconds_per_optimizer_step": float(index) / 100.0,
            "peak_vram_bytes": index * 1_000_000,
            "checkpoint_bytes": index * 100_000,
        })
    per_step = {row["variant"]: row["train_flops_per_optimizer_step"] for row in variant_rows}
    source = min(FLOPS_SENSITIVITY_VARIANTS, key=lambda variant: (per_step[variant], variant))
    cap = per_step[source] * FINAL_UPDATES
    arm_rows = []
    for index, arm in enumerate(STAGE1B_ARMS, start=1):
        arm_rows.append({
            "arm": arm,
            "estimated_flops_per_task": float(index * 100),
            "estimated_seconds_per_task": float(index) / 1000.0,
            "maximum_plan_calls": 0 if arm in {"E0_EQUAL_TOKENS_RAW", "E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"} else 1,
            "maximum_executor_calls": 16,
            "fixed_flops_per_task": 10.0,
            "planner_flops_per_call": 20.0,
            "plan_token_flops": 1.0,
            "executor_input_token_flops": 1.0,
            "executor_output_token_flops": 2.0,
        })
    obj = {
        "schema_version": "work-planner-compute/1.1",
        "run_id": "run-capacity",
        "hardware": {"device_name": "test-gpu", "device_count": 1},
        "variants": variant_rows,
        "equal_data_schedule": {"optimizer_updates_exact": FINAL_UPDATES, "batch_size_examples": BATCH_SIZE},
        "flops_matched_schedule": {
            "comparison_id": "A3_vs_A2c",
            "cap_source_variant": source,
            "cap_flops_per_workload": cap,
            "tolerance_fraction": 0.02,
            "optimizer_updates_by_variant": {
                variant: math.floor(cap / per_step[variant])
                for variant in FLOPS_SENSITIVITY_VARIANTS
            },
        },
        "stage1b_inference": {
            "confirmatory_task_capacity": 4000,
            "arms": arm_rows,
            "flops_cap_per_task": 1000.0,
            "budget_exhaustion_policy": "PAIRED_FAILURE_NO_TASK_EXCLUSION",
        },
        "measurement_evidence": {
            "command": ["python", "validation/measurement_probe.py"],
            "repetitions": 3,
            "raw_artifact_path": "reports/compute-evidence/raw.jsonl",
            "raw_artifact_sha256": H,
            "environment_lock_sha256": H,
        },
        "measurement_code_path": "validation/measurement_probe.py",
        "measurement_code_sha256": H,
        "report_hash": H,
    }
    obj["report_hash"] = canonical_report_hash(obj)
    return obj


def _tree(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path
    training = yaml.safe_load((ROOT / "docs/training/planner_training_contract_v1.yaml").read_text())
    hyper = yaml.safe_load((ROOT / "docs/training/hyperparameter_search_v1.yaml").read_text())
    hyper["schema_version"] = "work-planner/1.16"
    dataset = yaml.safe_load((ROOT / "docs/data/dataset_split_contract_v1.yaml").read_text())
    dataset["schema_version"] = "work-planner/1.16"
    compute = _compute()
    measurement = root / compute["measurement_code_path"]
    measurement.parent.mkdir(parents=True, exist_ok=True)
    measurement.write_text("# deterministic measurement probe\n", encoding="utf-8")
    raw = root / compute["measurement_evidence"]["raw_artifact_path"]
    raw.parent.mkdir(parents=True, exist_ok=True)
    measurement_rows = []
    for row in compute["variants"]:
        for trial in range(3):
            measurement_rows.append({
                "schema_version":"work-planner-compute-measurement/1.0","run_id":compute["run_id"],
                "trial_id":f"train-{row['variant']}-{trial}","measurement_type":"TRAIN_OPTIMIZER_STEP",
                "device_name":compute["hardware"]["device_name"],"variant":row["variant"],"arm":None,
                "measured_flops":row["train_flops_per_optimizer_step"],"elapsed_seconds":row["train_seconds_per_optimizer_step"],
                "peak_vram_bytes":row["peak_vram_bytes"]})
    for row in compute["stage1b_inference"]["arms"]:
        for trial in range(3):
            measurement_rows.append({
                "schema_version":"work-planner-compute-measurement/1.0","run_id":compute["run_id"],
                "trial_id":f"arm-{row['arm']}-{trial}","measurement_type":"STAGE1B_TASK",
                "device_name":compute["hardware"]["device_name"],"variant":None,"arm":row["arm"],
                "measured_flops":row["estimated_flops_per_task"],"elapsed_seconds":row["estimated_seconds_per_task"],
                "peak_vram_bytes":1})
    raw.write_text("".join(json.dumps(row)+"\n" for row in measurement_rows), encoding="utf-8")
    environment = root / "locks/environment.lock.json"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text("{}\n", encoding="utf-8")
    compute["measurement_code_sha256"] = file_digest(measurement)
    compute["measurement_evidence"]["raw_artifact_sha256"] = file_digest(raw)
    compute["measurement_evidence"]["environment_lock_sha256"] = file_digest(environment)
    compute["report_hash"] = canonical_report_hash(compute)
    resource = {
        "schema_version": "work-planner-infra/1.0", "run_id": "run-capacity",
        "capacity_limits": {"maximum_gpu_seconds": 1_000_000.0, "maximum_storage_bytes": 1_000_000_000, "gpu_hour_cost": 0.1},
        "estimated_cost": 100.0,
    }
    corpus = {"counts_by_split": {"train": 1000}}
    files = {
        "docs/training/planner_training_contract_v1.yaml": (training, True),
        "docs/training/hyperparameter_search_v1.yaml": (hyper, True),
        "docs/data/dataset_split_contract_v1.yaml": (dataset, True),
        "reports/compute-profile.json": (compute, False),
        "reports/resource-plan.json": (resource, False),
        "data/manifests/training-corpus.json": (corpus, False),
    }
    for rel, (value, is_yaml) in files.items():
        _write(root / rel, value, yaml_file=is_yaml)

    train_rows = 1000
    dev_updates = math.ceil(train_rows / BATCH_SIZE) * training["training"]["development"]["max_epochs"]
    vp = {x["variant"]: x for x in compute["variants"]}
    configs = [f"lr={lr}|dropout={drop}" for lr in hyper["grid"]["learning_rate"] for drop in hyper["grid"]["dropout"]]
    dev = []
    for config in configs:
        for variant in TRAIN_VARIANTS:
            cp = vp[variant]
            dev.append({
                "workload_id": f"development|{config}|{variant}|seed={DEV_SEED}", "variant": variant, "seed": DEV_SEED, "config_id": config,
                "optimizer_updates": dev_updates, "batch_size_examples": BATCH_SIZE, "unique_training_rows": train_rows,
                "processed_examples": dev_updates * BATCH_SIZE, "estimated_gpu_seconds": dev_updates * cp["train_seconds_per_optimizer_step"],
                "peak_vram_bytes": cp["peak_vram_bytes"], "checkpoint_bytes": cp["checkpoint_bytes"] * 2,
            })
    final = []
    for seed in FINAL_SEEDS:
        for variant in TRAIN_VARIANTS:
            cp = vp[variant]
            final.append({
                "workload_id": f"final|{variant}|seed={seed}", "variant": variant, "seed": seed, "config_id": None,
                "optimizer_updates": FINAL_UPDATES, "batch_size_examples": BATCH_SIZE, "unique_training_rows": train_rows,
                "processed_examples": FINAL_UPDATES * BATCH_SIZE, "estimated_gpu_seconds": FINAL_UPDATES * cp["train_seconds_per_optimizer_step"],
                "peak_vram_bytes": cp["peak_vram_bytes"], "checkpoint_bytes": cp["checkpoint_bytes"],
            })
    sensitivity = []
    sensitivity_updates = compute["flops_matched_schedule"]["optimizer_updates_by_variant"]
    for seed in FINAL_SEEDS:
        for variant in FLOPS_SENSITIVITY_VARIANTS:
            cp = vp[variant]
            updates = sensitivity_updates[variant]
            sensitivity.append({
                "workload_id": f"flops-sensitivity|{variant}|seed={seed}", "variant": variant, "seed": seed, "config_id": None,
                "optimizer_updates": updates, "batch_size_examples": BATCH_SIZE, "unique_training_rows": train_rows,
                "processed_examples": updates * BATCH_SIZE, "estimated_gpu_seconds": updates * cp["train_seconds_per_optimizer_step"],
                "peak_vram_bytes": cp["peak_vram_bytes"], "checkpoint_bytes": cp["checkpoint_bytes"],
            })
    stage = []
    for cp in compute["stage1b_inference"]["arms"]:
        stage.append({
            "arm": cp["arm"], "task_capacity": 4000,
            "estimated_flops": cp["estimated_flops_per_task"] * 4000,
            "estimated_gpu_seconds": cp["estimated_seconds_per_task"] * 4000,
            "maximum_plan_calls_per_task": cp["maximum_plan_calls"],
            "maximum_executor_calls_per_task": cp["maximum_executor_calls"],
        })
    gpu = sum(x["estimated_gpu_seconds"] for x in dev + final + sensitivity + stage)
    storage = sum(x["checkpoint_bytes"] for x in dev + final + sensitivity)
    report = {
        "schema_version": "work-planner-capacity/1.0", "run_id": "run-capacity", "status": "PASS",
        "training_contract_sha256": file_digest(root / "docs/training/planner_training_contract_v1.yaml"),
        "hyperparameter_contract_sha256": file_digest(root / "docs/training/hyperparameter_search_v1.yaml"),
        "dataset_contract_sha256": file_digest(root / "docs/data/dataset_split_contract_v1.yaml"),
        "compute_profile_sha256": file_digest(root / "reports/compute-profile.json"),
        "resource_plan_sha256": file_digest(root / "reports/resource-plan.json"),
        "corpus_manifest_sha256": file_digest(root / "data/manifests/training-corpus.json"),
        "test_status": {k: "PASS" for k in ("clean_checkout", "all_arms_forward_backward", "two_batch_overfit", "serialization", "fake_episode", "statistics_golden")},
        "development_workloads": dev, "final_training_workloads": final, "flops_sensitivity_workloads": sensitivity, "stage1b_inference_workloads": stage,
        "summary": {
            "development_workload_count": 24, "final_training_workload_count": 30, "flops_sensitivity_workload_count": 10, "total_training_workload_count": 64,
            "total_sensitivity_processed_examples": sum(x["processed_examples"] for x in sensitivity),
            "stage1b_arm_count": 7, "final_optimizer_updates_per_workload": 12000, "final_batch_size_examples": 128,
            "final_processed_examples_per_workload": 1536000, "total_final_processed_examples": 46080000,
            "estimated_gpu_seconds": gpu, "maximum_gpu_seconds": 1_000_000.0,
            "estimated_storage_bytes": storage, "maximum_storage_bytes": 1_000_000_000,
        },
        "report_hash": H,
    }
    report["report_hash"] = canonical_report_hash(report)
    return root, report


def test_capacity_preflight_recomputes_all_64_training_workloads_and_seven_arms(tmp_path: Path):
    root, report = _tree(tmp_path)
    assert validate_capacity_preflight(root, report) == []


def test_capacity_rejects_one_pass_training_rows_instead_of_exact_updates(tmp_path: Path):
    root, report = _tree(tmp_path)
    report["final_training_workloads"][0]["processed_examples"] = 1000
    report["report_hash"] = canonical_report_hash(report)
    assert any("processed_examples mismatch" in x for x in validate_capacity_preflight(root, report))


def test_capacity_rejects_missing_development_grid_workload(tmp_path: Path):
    root, report = _tree(tmp_path)
    report["development_workloads"].pop()
    report["report_hash"] = canonical_report_hash(report)
    assert any("workload identity set mismatch" in x for x in validate_capacity_preflight(root, report))


def test_capacity_rejects_intermediate_final_checkpoint_contract(tmp_path: Path):
    root, report = _tree(tmp_path)
    path = root / "docs/training/planner_training_contract_v1.yaml"
    training = yaml.safe_load(path.read_text())
    training["training"]["final_confirmatory_models"]["final_checkpoint_optimizer_step_exact"] = 11000
    _write(path, training, yaml_file=True)
    report["training_contract_sha256"] = file_digest(path)
    report["report_hash"] = canonical_report_hash(report)
    assert any("step-12000" in x for x in validate_capacity_preflight(root, report))


def test_compute_profile_rejects_arm_above_preoutcome_flops_cap():
    profile = _compute()
    profile["stage1b_inference"]["arms"][0]["estimated_flops_per_task"] = 1001.0
    profile["report_hash"] = canonical_report_hash(profile)
    assert any("exceeds" in x for x in validate_compute_profile(profile))


def test_compute_profile_rejects_tampered_measurement_evidence(tmp_path: Path):
    root, report = _tree(tmp_path)
    profile_path = root / "reports/compute-profile.json"
    profile = json.loads(profile_path.read_text())
    raw = root / profile["measurement_evidence"]["raw_artifact_path"]
    raw.write_text(raw.read_text() + "{\"trial\":999}\n")
    assert any("raw artifact hash mismatch" in x for x in validate_compute_profile(profile, root))


def test_compute_profile_rejects_tampered_measurement_code(tmp_path: Path):
    root, report = _tree(tmp_path)
    profile = json.loads((root / "reports/compute-profile.json").read_text())
    code = root / profile["measurement_code_path"]
    code.write_text(code.read_text() + "# tamper\n")
    assert any("measurement_code_sha256" in x for x in validate_compute_profile(profile, root))


def test_compute_profile_rejects_duplicate_or_extra_measurement_trials(tmp_path: Path):
    root, report = _tree(tmp_path)
    profile_path = root / "reports/compute-profile.json"
    profile = json.loads(profile_path.read_text())
    raw = root / profile["measurement_evidence"]["raw_artifact_path"]
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    rows.append(copy.deepcopy(rows[0]))
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    profile["measurement_evidence"]["raw_artifact_sha256"] = file_digest(raw)
    profile["report_hash"] = canonical_report_hash(profile)
    errors = validate_compute_profile(profile, root)
    assert any("trial_id" in x for x in errors)
    assert any("exactly" in x and "trials" in x for x in errors)
