from __future__ import annotations

import copy
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
    _assert_prefix_equivalence,
    _control_training,
    _train_rows,
    _train_trajectory,
)
from planner_toy.a2_optimization_budget_trajectory_validator import (
    _first_rescue_from_raw,
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


def test_budget_contract_is_prefix_preserving_and_train_only() -> None:
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
