from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from planner_toy.e2e import A2Planner
from planner_toy.model import LockedPlanner
from planner_toy.quality import (
    MAPPING,
    paired,
    run,
    state_dict_sha256,
    summarize,
    validate_evaluation,
)


@pytest.fixture(scope="session")
def canonical_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("quality-base") / "run"
    run(root, variants=("A2",), seeds=(17,), max_eval_tasks=1)
    return root


def copied_run(tmp_path, canonical_smoke):
    root = tmp_path / "run"
    shutil.copytree(canonical_smoke, root)
    return root


def test_common_initialization_is_byte_identical() -> None:
    hashes = [state_dict_sha256(LockedPlanner(29, variant).state_dict()) for variant in MAPPING]
    assert len(set(hashes)) == 1


def test_handcrafted_metrics_and_paired_matrix() -> None:
    base = {
        "predicted_plan_length": 2, "gold_plan_length": 1, "plan_generation_success": True,
        "terminal_end_produced": True, "action_parse_valid": True,
        "action_applicability": [True, True], "full_plan_executable": True,
        "exact_plan_match": False, "failure_code": None,
        "action_attempt_count": 2, "applicable_action_count": 2,
        "nonempty_plan": True, "end_only_plan": False,
        "execution_completed_without_precondition_failure": True,
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


class _ScriptedModel:
    def __init__(self, actions: list[int], variant: str = "A2"):
        self.actions, self.variant, self.index = actions, variant, 0

    def __call__(self, *args, **kwargs):
        action = torch.full((1, 17, 5), -10.0)
        action[0, self.index, self.actions[min(self.index, len(self.actions) - 1)]] = 10.0
        self.index += 1
        z = (
            torch.nn.functional.normalize(torch.ones(1, 17, 384), dim=-1)
            if self.variant != "A2"
            else None
        )
        return SimpleNamespace(
            action=action,
            arg1=torch.zeros(1, 17, 3),
            arg2=torch.zeros(1, 17, 3),
            z_semantic=z,
        )


@pytest.mark.parametrize("actions,expected", [([4], 1), ([0, 4], 2), ([0, 2, 4], 3)])
def test_model_forward_count(actions, expected) -> None:
    from planner_toy.dataset import generate
    planner = A2Planner(_ScriptedModel(actions))
    planner.plan(generate()["validation"][0])
    assert planner.model_forward_count == expected


@pytest.mark.parametrize("variant", ["A2", "A3", "A4"])
def test_forward_count_policy_same_for_variants(variant) -> None:
    from planner_toy.dataset import generate
    planner = A2Planner(_ScriptedModel([4], variant))
    planner.plan(generate()["validation"][0])
    assert planner.model_forward_count == 1


def test_no_end_counts_all_17_forwards() -> None:
    from planner_toy.dataset import generate
    planner = A2Planner(_ScriptedModel([0]))
    with pytest.raises(ValueError, match="PLAN_NO_END"):
        planner.plan(generate()["validation"][0])
    assert planner.model_forward_count == 17


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal_reached", True), ("terminal_end_produced", False),
        ("generated_action_count", 1), ("action_attempt_count", 1),
        ("applicable_action_count", 1), ("model_forward_count", 99),
        ("planner_call_count", 2), ("replanning_count", 1),
        ("experimental_arm", "A3a-zero"), ("architecture_stage", "STAGE_1"),
        ("target_type", "NONE"), ("seed", 43), ("task_id", "unknown"),
    ],
)
def test_semantic_task_result_mutations_fail(tmp_path, canonical_smoke, field, value) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "task-results.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises((ValueError, KeyError)):
        validate_evaluation(root)


@pytest.mark.parametrize("name", ["initialization.pt", "trained.pt", "training-report.json"])
def test_missing_training_lineage_fails(tmp_path, canonical_smoke, name) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    (root / "training-runs/A2/seed-17" / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_evaluation(root)


@pytest.mark.parametrize("field", ["dataset_manifest_hash", "train_task_ids", "eval_task_ids"])
def test_canonical_dataset_mutations_fail(tmp_path, canonical_smoke, field) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "evaluation-config.json"
    config = json.loads(path.read_text())
    config[field] = (
        "sha256:" + "0" * 64
        if field == "dataset_manifest_hash"
        else list(reversed(config[field])) + (["unknown"] if len(config[field]) == 1 else [])
    )
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_evaluation(root)
