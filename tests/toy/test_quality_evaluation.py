from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from jsonschema import ValidationError

from planner_toy.canonical import sha256
from planner_toy.e2e import A2Planner
from planner_toy.model import LockedPlanner
from planner_toy.quality import (
    MAPPING,
    export_compact,
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


@pytest.fixture(scope="session")
def all_three_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("quality-all-three") / "run"
    run(root, variants=("A2", "A3", "A4"), seeds=(17,), max_eval_tasks=1)
    return root


def copied_run(tmp_path, canonical_smoke):
    root = tmp_path / "run"
    shutil.copytree(canonical_smoke, root)
    return root


def rehash_run(root, name):
    manifest_path = root / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib
    manifest["artifact_hashes"][name] = (
        "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    (root / "replay-hash.txt").write_text(sha256(manifest) + "\n")


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


def test_deterministic_smoke_and_public_mutation_rejection(tmp_path, canonical_smoke) -> None:
    assert validate_evaluation(canonical_smoke)["valid"]
    damaged = tmp_path / "damaged"
    shutil.copytree(canonical_smoke, damaged)
    row = json.loads((damaged / "task-results.jsonl").read_text())
    row["goal_reached"] = not row["goal_reached"]
    (damaged / "task-results.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        validate_evaluation(damaged)


def test_split_overlap_fails_closed(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
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
            projected_semantic=(torch.ones(1, 17, 256) if z is not None else None),
            semantic_component=(
                torch.zeros(1, 17, 256)
                if self.variant == "A4"
                else torch.ones(1, 17, 256)
                if z is not None
                else None
            ),
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
    with pytest.raises((ValueError, KeyError, ValidationError)):
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


def test_all_three_variant_real_smoke(all_three_smoke) -> None:
    root = all_three_smoke
    config = json.loads((root / "evaluation-config.json").read_text())
    assert config["diagnostic_complete"] is False
    rows = [json.loads(line) for line in (root / "task-results.jsonl").read_text().splitlines()]
    assert {row["variant"] for row in rows} == {"A2", "A3", "A4"}
    assert all(row["planner_call_count"] == 1 and row["replanning_count"] == 0 for row in rows)
    assert not (root / rows[0]["evidence_root"] / "semantic-trace.json").exists()
    for row in rows[1:]:
        trace = json.loads((root / row["evidence_root"] / "semantic-trace.json").read_text())
        assert trace["feedback_source"] == "predicted"
        assert trace["feedback_application_count"] == 0
        assert trace["nonzero_feedback_application_count"] == 0
        assert trace["feedback_influenced_decoding_position_count"] == 0
        assert trace["feedback_applied"] is False
        assert trace["feedback_mode_enabled"] == (row["variant"] == "A3")
        assert trace["compute_then_zero"] == (row["variant"] == "A4")


def test_incomplete_export_rejected(tmp_path, canonical_smoke) -> None:
    with pytest.raises(ValueError, match="COMPLETE_CANONICAL"):
        export_compact(canonical_smoke, tmp_path / "docs")


def test_reuse_rejects_incomplete_source(tmp_path, canonical_smoke) -> None:
    reused = tmp_path / "reused"
    with pytest.raises(ValueError, match="COMPLETE_CANONICAL_SOURCE"):
        run(
            reused, variants=("A2",), seeds=(17,), max_eval_tasks=1,
            reuse_checkpoint_root=canonical_smoke,
        )


def test_self_consistent_markdown_claim_rejected(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "human-readable-examples.md"
    path.write_text(path.read_text().replace("Goal reached: `false`", "Goal reached: `true`"))
    rehash_run(root, "human-readable-examples.md")
    with pytest.raises(ValueError, match="HUMAN_EXAMPLES"):
        validate_evaluation(root)


def test_self_consistent_dataset_manifest_rejected(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "dataset-manifest.json"
    value = json.loads(path.read_text())
    value["dataset_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value) + "\n")
    rehash_run(root, "dataset-manifest.json")
    with pytest.raises(ValueError, match="DATASET_MANIFEST"):
        validate_evaluation(root)


def test_plan_no_end_retains_partial_output() -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import PlannerGenerationFailure
    planner = A2Planner(_ScriptedModel([0]))
    with pytest.raises(PlannerGenerationFailure) as caught:
        planner.plan(generate()["validation"][0])
    assert len(caught.value.partial_raw_output) == 17
    assert caught.value.model_forward_count == 17


def test_plan_no_end_persists_generation_failure_without_workplan(tmp_path) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan
    planner = A2Planner(_ScriptedModel([0]))
    result = evaluate_frozen_plan(
        row=generate()["validation"][0], planner=planner, output=tmp_path / "evidence",
        checkpoint_binding={"trained_state_dict_sha256": "sha256:" + "0" * 64,
                            "trained_file_sha256": "sha256:" + "1" * 64},
    )
    assert result["failure_code"] == "PLAN_NO_END"
    assert len(result["raw_output"]) == 17
    assert result["model_forward_count"] == 17
    assert not (tmp_path / "evidence/work-plan.json").exists()
    assert (tmp_path / "evidence/attempt-log.jsonl").read_text() == ""


def test_quality_adapter_cannot_bypass_shared_lineage_core(tmp_path, monkeypatch) -> None:
    from planner_toy import e2e
    from planner_toy.dataset import generate
    called = []
    original = e2e.validate_frozen_plan_lineage_core
    def spy(**kwargs):
        called.append(kwargs["variant"])
        return original(**kwargs)
    monkeypatch.setattr(e2e, "validate_frozen_plan_lineage_core", spy)
    e2e.evaluate_frozen_plan(
        row=generate()["validation"][0], planner=A2Planner(_ScriptedModel([4])),
        output=tmp_path / "evidence",
        checkpoint_binding={"trained_state_dict_sha256": "sha256:" + "0" * 64,
                            "trained_file_sha256": "sha256:" + "1" * 64},
    )
    assert called == ["A2"]


def test_task_result_reordering_rejected(tmp_path, all_three_smoke) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    path = root / "task-results.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n")
    rehash_run(root, "task-results.jsonl")
    with pytest.raises(ValueError, match="CANONICAL_ORDER"):
        validate_evaluation(root)


def test_zero_max_eval_tasks_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        run(tmp_path / "run", max_eval_tasks=0)


def test_cli_rejects_noncanonical_seed(tmp_path) -> None:
    import subprocess
    result = subprocess.run(
        ["python", "-m", "scripts.run_toy_quality_evaluation", "--output-dir",
         str(tmp_path / "run"), "--seeds", "99"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_missing_implementation_commit_rejected_for_complete(tmp_path, monkeypatch) -> None:
    from planner_toy import quality
    monkeypatch.setattr(quality, "implementation_provenance", lambda commit: (_ for _ in ()).throw(
        AssertionError("provenance must not be resolved before explicit commit check")
    ))
    with pytest.raises(ValueError, match="requires --implementation-commit"):
        run(tmp_path / "run", implementation_commit=None)


@pytest.mark.parametrize("commit", ["not-a-sha", "0" * 40], ids=["malformed", "missing"])
def test_invalid_implementation_commit_rejected(commit) -> None:
    from planner_toy.quality import implementation_provenance
    with pytest.raises(ValueError, match="existing commit"):
        implementation_provenance(commit)


def test_requirements_hash_is_read_from_implementation_tree() -> None:
    import hashlib
    import subprocess

    from planner_toy.quality import implementation_provenance
    expected = subprocess.run(
        ["git", "show", "HEAD:requirements.lock"], capture_output=True, check=True,
    ).stdout
    provenance = implementation_provenance("HEAD")
    expected_hash = "sha256:" + hashlib.sha256(expected).hexdigest()
    assert provenance["requirements_lock_sha256"] == expected_hash
    assert set(provenance["runtime_versions"]) == {
        "python", "torch", "numpy", "cuda_version", "cuda_available", "execution_device",
    }


def test_ci_uses_full_history_and_explicit_provenance_preflight() -> None:
    workflow = (Path(__file__).parents[2] / ".github/workflows/ci.yml").read_text()
    assert "fetch-depth: 0" in workflow
    assert 'git cat-file -e "${IMPLEMENTATION_SHA}^{commit}"' in workflow
    assert 'git merge-base --is-ancestor "$IMPLEMENTATION_SHA" HEAD' in workflow
    assert 'git show "$IMPLEMENTATION_SHA:requirements.lock"' in workflow


def test_shared_lineage_core_rejects_reordered_attempts() -> None:
    from planner_toy.e2e import validate_frozen_plan_lineage_core

    plan = {"steps": [
        {"step_index": 0, "action": "PICK_UP", "args": ["A"]},
        {"step_index": 1, "action": "PUT_DOWN", "args": ["A"]},
        {"step_index": 2, "action": "END", "args": []},
    ]}
    attempts = [
        {"step_index": 1, "candidate_action": ["PUT_DOWN", "A"],
         "state_before_hash": "one", "state_after_hash": "two", "status": "APPLIED"},
        {"step_index": 0, "candidate_action": ["PICK_UP", "A"],
         "state_before_hash": "two", "state_after_hash": "three", "status": "APPLIED"},
    ]
    with pytest.raises(ValueError, match="frozen WorkPlan"):
        validate_frozen_plan_lineage_core(
            variant="A2", planner_calls=1, replanning_count=0,
            work_plan=plan, attempts=attempts, evaluation={"replanning_count": 0},
            semantic_trace=None,
        )


def test_shared_lineage_core_rejects_broken_state_chain() -> None:
    from planner_toy.e2e import validate_frozen_plan_lineage_core

    plan = {"steps": [
        {"step_index": 0, "action": "PICK_UP", "args": ["A"]},
        {"step_index": 1, "action": "PUT_DOWN", "args": ["A"]},
        {"step_index": 2, "action": "END", "args": []},
    ]}
    attempts = [
        {"step_index": 0, "candidate_action": ["PICK_UP", "A"],
         "state_before_hash": "one", "state_after_hash": "two", "status": "APPLIED"},
        {"step_index": 1, "candidate_action": ["PUT_DOWN", "A"],
         "state_before_hash": "not-two", "state_after_hash": "three", "status": "APPLIED"},
    ]
    with pytest.raises(ValueError, match="state transition"):
        validate_frozen_plan_lineage_core(
            variant="A2", planner_calls=1, replanning_count=0,
            work_plan=plan, attempts=attempts, evaluation={"replanning_count": 0},
            semantic_trace=None,
        )
