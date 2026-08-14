from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from planner_toy.learnability import OUTPUT_JSON as CORE_OUTPUT_JSON
from planner_toy.learnability import _dataset_context
from planner_toy.learnability_v0_2 import (
    CORE_DIRECTORY,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    SOURCE_FILES,
    _interpretation,
    run,
    source_identity,
    validate_diagnostic,
    validate_payload,
)
from planner_toy.numeric_identity import (
    canonical_state_dict_sha256,
    canonical_torch_object_sha256,
)
from planner_toy.quality import _train as quality_train

REPOSITORY = Path(__file__).parents[2]
IMPLEMENTATION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
).strip()


@pytest.fixture(scope="session")
def v0_2_smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("learnability-v0-2-smoke") / "run"
    run(
        root,
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000003",),
    )
    return root


def test_v0_2_bundle_validates_and_binds_core(v0_2_smoke_root) -> None:
    result = validate_diagnostic(v0_2_smoke_root)
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    core_payload = json.loads(
        (v0_2_smoke_root / CORE_DIRECTORY / CORE_OUTPUT_JSON).read_text()
    )
    assert result["valid"] is True
    assert result["heldout_accessed"] is False
    assert result["training_policy_changed"] is False
    assert payload["core_canonical_identity"] == core_payload["canonical_identity"]
    assert payload["heldout_accessed"] is False
    assert payload["training_policy_changed"] is False
    assert payload["model_changed"] is False
    assert payload["gate_decision"] is None
    assert payload["learnability_thresholds"] is None


def test_v0_2_observes_exact_nine_real_updates(v0_2_smoke_root) -> None:
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    rows = payload["per_update_training_observation"]
    assert [(row["update_index"], row["epoch_index"], row["task_id"]) for row in rows] == [
        (0, 0, "bw-00000001"),
        (1, 0, "bw-00000002"),
        (2, 0, "bw-00000003"),
        (3, 1, "bw-00000001"),
        (4, 1, "bw-00000002"),
        (5, 1, "bw-00000003"),
        (6, 2, "bw-00000001"),
        (7, 2, "bw-00000002"),
        (8, 2, "bw-00000003"),
    ]
    assert all(row["all_gradients_finite"] for row in rows)
    assert any(row["nonzero_gradient_tensor_count"] > 0 for row in rows)
    assert all(row["gradient_norm_post_clip"] >= 0 for row in rows)
    assert all(row["teacher_forced"]["confusion_matrix"] for row in rows)
    assert all(row["teacher_forced_by_gold_position"] for row in rows)


def test_outer_observer_preserves_frozen_quality_checkpoint_and_optimizer(
    v0_2_smoke_root, tmp_path
) -> None:
    _, train_rows = _dataset_context()
    observed_root = v0_2_smoke_root / CORE_DIRECTORY / "training-runs" / "A2" / "seed-17"
    plain_root = tmp_path / "plain"
    model, manifest = quality_train(
        train_rows,
        "A2",
        17,
        plain_root,
        json.loads((observed_root / "training-config.json").read_text())["dataset_hash"],
    )
    observed_trained = torch.load(
        observed_root / "trained.pt", map_location="cpu", weights_only=True
    )
    observed_optimizer = torch.load(
        observed_root / "optimizer-state.pt", map_location="cpu", weights_only=True
    )
    assert canonical_state_dict_sha256(model.state_dict()) == canonical_state_dict_sha256(
        observed_trained
    )
    assert manifest["canonical_trained_state_dict_sha256"] == canonical_state_dict_sha256(
        observed_trained
    )
    assert manifest["canonical_optimizer_state_sha256"] == canonical_torch_object_sha256(
        observed_optimizer
    )


def test_checkpoint_delta_evidence_is_read_only_and_complete(v0_2_smoke_root) -> None:
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    record = payload["checkpoint_deltas"][0]
    assert record["seed"] == 17
    assert record["active_parameter_tensor_count"] > 0
    assert record["active_parameter_tensors_changed"] > 0
    assert record["fraction_active_tensors_changed"] > 0
    assert record["active_parameter_delta_l2"] > 0
    assert record["dormant_parameter_tensors_changed"] == 0
    assert record["all_checkpoint_values_finite"] is True
    assert record["optimizer_state_finite"] is True
    assert record["optimizer_nonzero_moment_tensor_count"] > 0
    assert record["action_head_weight_delta_l2"] >= 0
    assert record["end_action_weight_row_delta_l2"] >= 0
    assert record["arg1_pointer_weight_delta_l2"] >= 0
    assert record["arg2_pointer_weight_delta_l2"] >= 0


def test_free_running_end_observability_matches_core_plan(v0_2_smoke_root) -> None:
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    core_payload = json.loads(
        (v0_2_smoke_root / CORE_DIRECTORY / CORE_OUTPUT_JSON).read_text()
    )
    observed = payload["free_running"][0]
    core_task = core_payload["free_running"][0]
    assert observed["predicted_plan"] == core_task["predicted_plan"]
    assert observed["predicted_plan_length"] == core_task["predicted_plan_length"]
    assert observed["zero_action_plan"] == (core_task["predicted_action_count"] == 0)
    assert all("end_vs_best_non_end_logit_margin" in row for row in observed["positions"])
    aggregate = payload["free_running_aggregate"]["overall"]
    assert sum(aggregate["plan_length_distribution"].values()) == aggregate["task_count"]
    assert sum(aggregate["first_predicted_end_position_distribution"].values()) == aggregate[
        "task_count"
    ]


def test_pointer_metrics_remain_independent_of_operator_in_v0_2(v0_2_smoke_root) -> None:
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    final = payload["final_teacher_forced"]["overall"]
    assert final["arg1_target_count"] > 0
    assert final["arg2_target_count"] > 0
    assert final["arg1_accuracy"] is not None
    assert final["arg2_accuracy"] is not None
    assert final["joint_step_accuracy"] is not None


def test_train_only_identities_remain_distinct(v0_2_smoke_root) -> None:
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    assert payload["dataset_lineage_order"] == [
        "bw-00000001", "bw-00000003", "bw-00000002"
    ]
    assert payload["optimizer_execution_task_order"] == [
        "bw-00000001", "bw-00000002", "bw-00000003"
    ]
    assert not set(payload["evaluated_task_ids"]) & {"bw-00000004", "bw-00000005"}


def test_v0_2_source_inventory_binds_execution_workflow() -> None:
    required = {
        "planner_toy/learnability.py",
        "planner_toy/learnability_v0_2.py",
        "planner_toy/train_only_dataset.py",
        "planner_toy/schemas/learnability_diagnostic_v0_2.schema.json",
        "scripts/run_toy_learnability_diagnostic_v0_2.py",
        ".github/workflows/a2-learnability-diagnostic.yml",
    }
    assert required <= set(SOURCE_FILES)
    identity = source_identity()
    assert {entry["path"] for entry in identity["source_files"]} == set(SOURCE_FILES)


def test_schema_rejects_scope_or_gate_mutation(v0_2_smoke_root) -> None:
    payload = json.loads((v0_2_smoke_root / OUTPUT_JSON).read_text())
    core_payload = json.loads(
        (v0_2_smoke_root / CORE_DIRECTORY / CORE_OUTPUT_JSON).read_text()
    )
    payload["heldout_accessed"] = True
    with pytest.raises(ValueError, match="SCHEMA_VALIDATION|SCOPE_FLAG"):
        validate_payload(payload, core_payload=core_payload)


def test_interpretation_is_noncausal_and_threshold_free() -> None:
    core_payload = {
        "seeds": [17],
        "aggregates": {
            "free_running": {
                "overall": {"exact_plan_rate": 0.0},
                "by_seed": {"17": {"exact_plan_rate": 0.0}},
            }
        },
    }
    teacher = {"operator_accuracy": 1.0, "arg1_accuracy": 1.0, "arg2_accuracy": 1.0}
    final_teacher = {"overall": teacher, "by_seed": {"17": teacher}}
    interpretation = _interpretation(core_payload, final_teacher)
    assert interpretation == {
        "first_major_failure_stage": "FREE_RUNNING_ONLY_ERRORS_PRESENT",
        "supported_hypothesis": "EXPOSURE_OR_ROLLOUT_FAILURE",
        "interpretation_label": "SUPPORTED HYPOTHESIS / NOT PROVEN",
        "by_seed": {
            "17": {
                "first_major_failure_stage": "FREE_RUNNING_ONLY_ERRORS_PRESENT",
                "supported_hypothesis": "EXPOSURE_OR_ROLLOUT_FAILURE",
            }
        },
        "cross_seed_localization_consistent": True,
    }


def test_markdown_answers_required_questions_without_gate(v0_2_smoke_root) -> None:
    markdown = (v0_2_smoke_root / OUTPUT_MARKDOWN).read_text()
    for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7."):
        assert prefix in markdown
    assert "SUPPORTED HYPOTHESIS / NOT PROVEN" in markdown
    assert "No gate, threshold, model change, or training-policy change" in markdown


def test_dedicated_target_workflow_is_manual_nonformal_diagnostic() -> None:
    workflow = (REPOSITORY / ".github/workflows/a2-learnability-diagnostic.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "planning-model-canonical-cpu-v1" in workflow
    assert "self-hosted" in workflow
    assert "validate-trusted-commit" in workflow
    assert "run_toy_learnability_diagnostic_v0_2" in workflow
    assert "--validate-only" in workflow
    assert "--formal" not in workflow
    assert "validate-bundle" not in workflow
    assert "final-gate" not in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
