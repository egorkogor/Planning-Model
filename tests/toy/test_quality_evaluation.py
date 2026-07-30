from __future__ import annotations

import json
import shutil

import pytest

from planner_toy.model import LockedPlanner
from planner_toy.quality import (
    MAPPING,
    paired,
    run,
    state_dict_sha256,
    summarize,
    validate_evaluation,
)


def test_common_initialization_is_byte_identical() -> None:
    hashes = [state_dict_sha256(LockedPlanner(29, variant).state_dict()) for variant in MAPPING]
    assert len(set(hashes)) == 1


def test_handcrafted_metrics_and_paired_matrix() -> None:
    base = {
        "predicted_plan_length": 2, "gold_plan_length": 1, "plan_generation_success": True,
        "terminal_end_produced": True, "action_parse_valid": True,
        "action_applicability": [True, True], "full_plan_executable": True,
        "exact_plan_match": False, "failure_code": None,
    }
    rows = [
        {**base, "variant": "A3", "seed": 17, "task_id": "x", "goal_reached": True},
        {**base, "variant": "A2", "seed": 17, "task_id": "x", "goal_reached": False,
         "failure_code": "GOAL_NOT_ACHIEVED"},
    ]
    assert summarize(rows[:1])["mean_absolute_plan_length_difference"] == 1
    assert paired(rows, "A3", "A2")["only_first_succeeds"] == 1


def test_deterministic_smoke_and_public_mutation_rejection(tmp_path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    run(first, variants=("A2",), seeds=(17,), max_eval_tasks=1)
    run(second, variants=("A2",), seeds=(17,), max_eval_tasks=1)
    deterministic = [
        "evaluation-config.json", "dataset-manifest.json", "task-results.jsonl",
        "per-seed-summary.json", "aggregate-summary.json", "paired-comparisons.json",
        "human-readable-examples.md", "evaluation-manifest.json", "replay-hash.txt",
    ]
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes() for name in deterministic
    )
    assert validate_evaluation(first)["valid"]
    damaged = tmp_path / "damaged"
    shutil.copytree(first, damaged)
    row = json.loads((damaged / "task-results.jsonl").read_text())
    row["goal_reached"] = not row["goal_reached"]
    (damaged / "task-results.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        validate_evaluation(damaged)


def test_split_overlap_fails_closed(tmp_path) -> None:
    root = tmp_path / "run"
    run(root, variants=("A2",), seeds=(17,), max_eval_tasks=1)
    config_path = root / "evaluation-config.json"
    config = json.loads(config_path.read_text())
    config["train_task_ids"].append(config["eval_task_ids"][0])
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_evaluation(root)
