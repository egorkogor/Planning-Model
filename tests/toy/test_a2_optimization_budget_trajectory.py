from __future__ import annotations

import copy
import math
import subprocess
from pathlib import Path

import pytest

from planner_toy.a2_optimization_budget_trajectory import (
    CANONICAL_ORDER,
    CHECKPOINT_EPOCHS,
    EXPECTED_TRAIN_TASK_IDS,
    INTERPRETATION_LABEL,
    MAX_EPOCH,
    SEEDS,
    SOURCE_FILES,
    VERSION,
    _aggregate_teacher_summary,
    _assert_prefix_equivalence,
    _control_training,
    _train_rows,
    _train_trajectory,
)
from planner_toy.a2_optimization_budget_trajectory_validator import (
    _aggregate_teacher_summary_from_raw,
    _first_rescue_from_raw,
    _validate_aggregate_teacher_claim,
    _validate_checkpoint,
    _validate_position0_epoch_record,
    _validate_prefix_equivalence,
    _validate_trajectory_claim,
)

REPOSITORY = Path(__file__).parents[2]
IMPLEMENTATION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
).strip()


@pytest.fixture(scope="session")
def prefix_fixture():
    dataset, rows = _train_rows()
    control = _control_training(
        rows,
        seed=17,
        dataset_hash=dataset["frozen_dataset_lineage_hash"],
    )
    trajectory = _train_trajectory(
        rows,
        seed=17,
        control=control,
        max_epoch=3,
        checkpoint_epochs=(3,),
    )
    return rows, control, trajectory


def _cross_seed_teacher_records(prefix_fixture) -> list[dict]:
    _rows, _control, trajectory = prefix_fixture
    base = trajectory["checkpoints"][0]["teacher_forced"]
    records = []
    probabilities = {17: 0.2, 29: 0.5, 43: 0.8}
    end_probabilities = {17: 0.75, 29: 0.45, 43: 0.15}
    for seed in SEEDS:
        for task in base:
            item = copy.deepcopy(task)
            item["seed"] = seed
            if item["task_id"] == "bw-00000002":
                position = item["positions"][0]
                probability = probabilities[seed]
                position["probability_gold_operator"] = probability
                position["operator_nll"] = -math.log(probability)
                position["probability_end"] = end_probabilities[seed]
                position["predicted_operator"] = "UNSTACK" if seed == 43 else "END"
                position["operator_correct"] = seed == 43
                position["joint_step_correct"] = (
                    position["operator_correct"]
                    and position["arg1_correct"] is not False
                    and position["arg2_correct"] is not False
                )
            records.append(item)
    return records


def test_budget_contract_is_prefix_preserving_and_train_only() -> None:
    assert VERSION == "development-a2-optimization-budget-trajectory/0.2"
    assert EXPECTED_TRAIN_TASK_IDS == (
        "bw-00000001",
        "bw-00000002",
        "bw-00000003",
    )
    assert CANONICAL_ORDER == EXPECTED_TRAIN_TASK_IDS
    assert CHECKPOINT_EPOCHS == (3, 10, 30, 100)
    assert MAX_EPOCH == 100
    assert SEEDS == (17, 29, 43)
    assert INTERPRETATION_LABEL == "SUPPORTED HYPOTHESIS / NOT PROVEN"


def test_epoch3_prefix_exactly_matches_frozen_control(prefix_fixture) -> None:
    _rows, control, trajectory = prefix_fixture
    record = trajectory["prefix_equivalence"]
    assert record["status"] == "PASS"
    assert record["control"] == record["trajectory_prefix"]
    assert len(record["control"]["updates"]) == 9
    assert [item["task_id"] for item in record["control"]["updates"]] == (
        list(CANONICAL_ORDER) * 3
    )
    prefix = {
        "initialization_canonical_sha256": trajectory[
            "initialization_canonical_sha256"
        ],
        "trained_canonical_sha256": trajectory["checkpoints"][0][
            "trained_canonical_sha256"
        ],
        "optimizer_canonical_sha256": trajectory["checkpoints"][0][
            "optimizer_canonical_sha256"
        ],
        "updates": trajectory["updates"][:9],
    }
    _assert_prefix_equivalence(control, prefix, seed=17)
    _validate_prefix_equivalence(record, trajectory, seed=17)


def test_prefix_equivalence_tamper_is_rejected(prefix_fixture) -> None:
    _rows, _control, trajectory = prefix_fixture
    record = copy.deepcopy(trajectory["prefix_equivalence"])
    record["trajectory_prefix"]["updates"][0]["gradient_norm"] += 1.0
    with pytest.raises(ValueError, match="PREFIX_"):
        _validate_prefix_equivalence(record, trajectory, seed=17)


def test_resigned_both_sides_prefix_tamper_is_rejected(prefix_fixture) -> None:
    _rows, _control, trajectory = prefix_fixture
    record = copy.deepcopy(trajectory["prefix_equivalence"])
    record["control"]["updates"][0]["gradient_norm"] += 1.0
    record["trajectory_prefix"]["updates"][0]["gradient_norm"] += 1.0
    assert record["control"] == record["trajectory_prefix"]
    with pytest.raises(ValueError, match="PREFIX_TRAJECTORY_BINDING"):
        _validate_prefix_equivalence(record, trajectory, seed=17)


def test_checkpoint_metric_tamper_is_rejected(prefix_fixture) -> None:
    rows, _control, trajectory = prefix_fixture
    row_by_id = {row["task_id"]: row for row in rows}
    checkpoint = copy.deepcopy(trajectory["checkpoints"][0])
    checkpoint["teacher_forced_summary"]["aggregate"]["operator_accuracy"] = 0.123
    with pytest.raises(ValueError, match="CHECKPOINT_METRIC"):
        _validate_checkpoint(checkpoint, row_by_id, seed=17, epoch=3)


def test_raw_position_evidence_tamper_is_rejected(prefix_fixture) -> None:
    rows, _control, trajectory = prefix_fixture
    row_by_id = {row["task_id"]: row for row in rows}
    record = copy.deepcopy(trajectory["position0_epoch_evidence"][0])
    record["tasks"][0]["operator_correct"] = not record["tasks"][0]["operator_correct"]
    with pytest.raises(ValueError, match="POS0_RAW_CLAIM"):
        _validate_position0_epoch_record(record, row_by_id, seed=17, epoch=1)


def test_first_rescue_claim_tamper_is_rejected(prefix_fixture) -> None:
    rows, _control, trajectory = prefix_fixture
    row_by_id = {row["task_id"]: row for row in rows}
    records = []
    template = trajectory["position0_epoch_evidence"][-1]
    for epoch in range(1, 101):
        record = copy.deepcopy(template)
        record["epoch"] = epoch
        record["update_count"] = epoch * 3
        records.append(record)
    expected = _first_rescue_from_raw(records)
    tampered = {"epoch": 1, "update_count": 3, "task_ids": ["bw-00000002"]}
    if expected == tampered:
        tampered["task_ids"] = ["bw-00000003"]
    with pytest.raises(ValueError, match="FIRST_RESCUE_CLAIM"):
        _validate_trajectory_claim(records, tampered, row_by_id, seed=17)


def test_cross_seed_task_aggregates_pool_repeated_task_ids(prefix_fixture) -> None:
    tasks = _cross_seed_teacher_records(prefix_fixture)
    produced = _aggregate_teacher_summary(tasks)
    independently_recomputed = _aggregate_teacher_summary_from_raw(tasks)
    assert produced == independently_recomputed

    task02 = produced["per_task"]["bw-00000002"]
    assert task02["seed_count"] == 3
    assert task02["task_record_count"] == 3
    assert task02["operator_target_count"] == 15
    assert task02["operator_accuracy"] != produced["per_task"]["bw-00000003"][
        "operator_accuracy"
    ]

    position0 = produced["position0_by_task"]["bw-00000002"]
    assert position0["seed_count"] == 3
    assert position0["target_count"] == 3
    assert position0["operator_accuracy"] == pytest.approx(1 / 3)
    assert position0["mean_gold_operator_probability"] == pytest.approx(0.5)
    assert position0["mean_end_probability"] == pytest.approx(0.45)
    assert "predicted_operator" not in position0
    assert "operator_correct" not in position0


def test_cross_seed_discrimination_is_contrast_of_true_means(prefix_fixture) -> None:
    tasks = _cross_seed_teacher_records(prefix_fixture)
    summary = _aggregate_teacher_summary(tasks)
    by_task = summary["position0_by_task"]
    discrimination = summary["position0_task_discrimination"]
    expected_nontrivial = (
        by_task["bw-00000002"]["mean_end_probability"]
        + by_task["bw-00000003"]["mean_end_probability"]
    ) / 2
    assert discrimination["aggregation"] == "contrast_of_cross_seed_means"
    assert discrimination["task02_mean_end_probability"] == pytest.approx(0.45)
    assert discrimination["nontrivial_mean_end_probability"] == pytest.approx(
        expected_nontrivial
    )
    assert discrimination["task01_minus_nontrivial_mean_end_probability"] == pytest.approx(
        by_task["bw-00000001"]["mean_end_probability"] - expected_nontrivial
    )


def test_last_seed_overwrite_aggregate_claim_is_rejected(prefix_fixture) -> None:
    tasks = _cross_seed_teacher_records(prefix_fixture)
    correct = _aggregate_teacher_summary_from_raw(tasks)
    tainted = copy.deepcopy(correct)
    task02_seed43 = next(
        task
        for task in tasks
        if task["task_id"] == "bw-00000002" and task["seed"] == 43
    )
    position = task02_seed43["positions"][0]
    tainted["position0_by_task"]["bw-00000002"] = {
        "gold_operator": position["gold_operator"],
        "seed_count": 1,
        "target_count": 1,
        "operator_accuracy": float(position["operator_correct"]),
        "mean_gold_operator_probability": position["probability_gold_operator"],
        "mean_operator_nll": position["operator_nll"],
        "mean_end_probability": position["probability_end"],
    }
    with pytest.raises(ValueError, match="AGGREGATE_TEACHER_CLAIM"):
        _validate_aggregate_teacher_claim(tasks, tainted, epoch=3)


def test_raw_seed_probability_mutation_breaks_aggregate_binding(prefix_fixture) -> None:
    tasks = _cross_seed_teacher_records(prefix_fixture)
    original_claim = _aggregate_teacher_summary_from_raw(tasks)
    mutated = copy.deepcopy(tasks)
    task = next(
        item
        for item in mutated
        if item["task_id"] == "bw-00000002" and item["seed"] == 17
    )
    position = task["positions"][0]
    position["probability_gold_operator"] = 0.3
    position["operator_nll"] = -math.log(0.3)
    position["probability_end"] = 0.65
    with pytest.raises(ValueError, match="AGGREGATE_TEACHER_CLAIM"):
        _validate_aggregate_teacher_claim(mutated, original_claim, epoch=3)


def test_validator_does_not_import_producer_aggregate_helpers() -> None:
    validator = (
        REPOSITORY / "planner_toy/a2_optimization_budget_trajectory_validator.py"
    ).read_text()
    assert "from .a2_optimization_budget_trajectory import" not in validator
    assert "_aggregate_teacher_summary_from_raw" in validator


def test_source_inventory_closes_transitive_claim_sources() -> None:
    required = {
        "planner_toy/dataset.py",
        "planner_toy/domain.py",
        "planner_toy/e2e.py",
        "planner_toy/learnability.py",
        "planner_toy/model.py",
        "planner_toy/quality.py",
        "planner_toy/semantic.py",
        "planner_toy/training.py",
        "planner_toy/train_only_dataset.py",
        "planner_toy/a2_optimization_budget_trajectory.py",
        "planner_toy/a2_optimization_budget_trajectory_validator.py",
        ".github/workflows/a2-optimization-budget-trajectory.yml",
        "scripts/run_a2_optimization_budget_trajectory.py",
    }
    assert required <= set(SOURCE_FILES)


def test_cli_validate_only_binds_requested_implementation_commit() -> None:
    cli = (REPOSITORY / "scripts/run_a2_optimization_budget_trajectory.py").read_text()
    assert "validate_trajectory(" in cli
    assert "implementation_commit=args.implementation_commit" in cli


def test_workflow_is_manual_fixed_target_and_hidden_artifact_only() -> None:
    workflow = (
        REPOSITORY / ".github/workflows/a2-optimization-budget-trajectory.yml"
    ).read_text()
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "\npush:" not in workflow
    assert "planning-model-canonical-cpu-v1" in workflow
    assert "self-hosted" in workflow
    assert "validate-trusted-commit" in workflow
    assert "run_a2_optimization_budget_trajectory" in workflow
    assert "--validate-only" in workflow
    assert '--implementation-commit "$A2_BUDGET_IMPLEMENTATION_SHA"' in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "include-hidden-files: true" in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 30" in workflow
