from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from planner_toy.a2_causal_microexperiment import (
    ARM_CONTRACT,
    ARM_CONTROL,
    ARM_EQUAL_POSITION,
    ARM_ORDER_ONLY,
    ARMS,
    CANONICAL_ORDER,
    EXPECTED_TRAIN_TASK_IDS,
    INTERPRETATION_LABEL,
    ORDER_ONLY_ORDER,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    SEEDS,
    SOURCE_FILES,
    _assert_control_equivalence,
    _control_equivalence_probe,
    _control_training,
    _operator_loss,
    _train_rows,
    _weighting_contract,
    operator_position_weight,
    render_markdown,
    run,
    validate_microexperiment,
)
from planner_toy.canonical import canonical_bytes, sha256
from planner_toy.train_only_dataset import generate_train_only

REPOSITORY = Path(__file__).parents[2]
IMPLEMENTATION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
).strip()


def test_arm_contract_is_minimal_and_causal() -> None:
    assert ARMS == (ARM_CONTROL, ARM_EQUAL_POSITION, ARM_ORDER_ONLY)
    assert ARM_CONTRACT[ARM_CONTROL]["changed_dimension"] == "NONE"
    assert ARM_CONTRACT[ARM_CONTROL]["task_order"] == list(CANONICAL_ORDER)
    assert ARM_CONTRACT[ARM_CONTROL]["operator_loss"] == "per-task-mean-cross-entropy"
    assert (
        ARM_CONTRACT[ARM_EQUAL_POSITION]["changed_dimension"]
        == "OPERATOR_LOSS_NORMALIZATION_ONLY"
    )
    assert ARM_CONTRACT[ARM_EQUAL_POSITION]["task_order"] == list(CANONICAL_ORDER)
    assert ARM_CONTRACT[ARM_ORDER_ONLY]["changed_dimension"] == "TASK_ORDER_ONLY"
    assert ARM_CONTRACT[ARM_ORDER_ONLY]["operator_loss"] == "per-task-mean-cross-entropy"
    assert ARM_CONTRACT[ARM_ORDER_ONLY]["task_order"] == list(ORDER_ONLY_ORDER)
    assert ORDER_ONLY_ORDER == (
        "bw-00000002",
        "bw-00000003",
        "bw-00000001",
    )


def test_equal_position_operator_weight_removes_short_task_overweighting() -> None:
    dataset = generate_train_only()
    contract = _weighting_contract(dataset["train"])
    assert contract["valid_operator_positions_by_task"] == {
        "bw-00000001": 1,
        "bw-00000003": 5,
        "bw-00000002": 5,
    }
    assert contract["canonical_end_to_unstack_position0_weight_ratio"] == pytest.approx(2.5)
    assert contract["equal_position_end_to_unstack_position0_weight_ratio"] == pytest.approx(
        0.5
    )
    assert contract["equal_position_weight_each"] == pytest.approx(3 / 11)
    assert contract["epoch_scale_preservation"][
        "canonical_uniform_loss_operator_weight_sum"
    ] == pytest.approx(3.0)
    assert contract["epoch_scale_preservation"][
        "equal_position_uniform_loss_operator_weight_sum"
    ] == pytest.approx(3.0)


def test_operator_loss_modes_have_declared_position_weights() -> None:
    logits = torch.zeros((5, 5))
    target = torch.tensor([0, 1, 2, 3, 4])
    canonical, canonical_weight = _operator_loss(
        logits,
        target,
        mode="per-task-mean-cross-entropy",
        total_positions=11,
        task_count=3,
    )
    equal, equal_weight = _operator_loss(
        logits,
        target,
        mode="equal-per-position-cross-entropy-epoch-scale-preserved",
        total_positions=11,
        task_count=3,
    )
    assert canonical_weight == pytest.approx(1 / 5)
    assert equal_weight == pytest.approx(3 / 11)
    assert float(equal) == pytest.approx(float(canonical) * 5 * 3 / 11)
    assert operator_position_weight(
        valid_positions=5, total_positions=11, task_count=3
    ) == pytest.approx(3 / 11)


@pytest.fixture(scope="session")
def causal_smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("a2-causal-smoke") / "run"
    run(root, implementation_commit=IMPLEMENTATION, seeds=(17,))
    return root


def _load(root: Path) -> dict:
    return json.loads((root / OUTPUT_JSON).read_text())


def _resign_and_write(root: Path, payload: dict) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    payload["canonical_identity"] = sha256(unsigned)
    (root / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (root / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")


def _tampered_copy(source: Path, target: Path) -> tuple[Path, dict]:
    shutil.copytree(source, target)
    return target, _load(target)


def test_microexperiment_is_train_only_and_independently_validated(
    causal_smoke_root,
) -> None:
    result = validate_microexperiment(
        causal_smoke_root, implementation_commit=IMPLEMENTATION
    )
    payload = _load(causal_smoke_root)
    assert result["valid"] is True
    assert result["heldout_accessed"] is False
    assert result["automatic_gate"] is None
    assert result["control_equivalence"] == "PASS"
    assert result["claim_validation"] == "INDEPENDENT_FROM_PERSISTED_RAW_EVIDENCE"
    assert payload["heldout_accessed"] is False
    assert payload["frozen_science_changed"] is False
    assert payload["go_latent"] == "NOT EVALUATED"
    assert payload["hypothesis"]["label"] == INTERPRETATION_LABEL
    assert payload["dataset"]["evaluated_task_ids"] == list(EXPECTED_TRAIN_TASK_IDS)
    assert not set(payload["dataset"]["evaluated_task_ids"]) & {
        "bw-00000004",
        "bw-00000005",
    }
    assert set(payload["aggregates"]) == set(ARMS)
    assert set(payload["contrasts_vs_control"]) == {
        ARM_EQUAL_POSITION,
        ARM_ORDER_ONLY,
    }


def test_control_equivalence_probe_covers_all_frozen_seeds() -> None:
    dataset, rows = _train_rows()
    for seed in SEEDS:
        _model, control = _control_training(
            rows,
            seed=seed,
            dataset_hash=dataset["frozen_dataset_lineage_hash"],
        )
        record = _control_equivalence_probe(control, rows, seed=seed)
        assert record["status"] == "PASS"
        assert record["control"] == record["probe"]
        assert [row["task_id"] for row in record["control"]["updates"]] == list(
            CANONICAL_ORDER
        ) * 3


def test_non_scientific_control_equivalence_probe_is_exact(causal_smoke_root) -> None:
    payload = _load(causal_smoke_root)
    records = payload["control_equivalence"]
    assert len(records) == 1
    record = records[0]
    assert record["seed"] == 17
    assert record["status"] == "PASS"
    assert record["control"] == record["probe"]
    assert len(record["control"]["updates"]) == 9
    assert [row["task_id"] for row in record["control"]["updates"]] == list(
        CANONICAL_ORDER
    ) * 3


def test_control_equivalence_guard_rejects_exact_hash_or_trace_drift(
    causal_smoke_root,
) -> None:
    record = _load(causal_smoke_root)["control_equivalence"][0]
    changed_hash = copy.deepcopy(record["probe"])
    changed_hash["trained_canonical_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="CONTROL_EQUIVALENCE_TRAINED"):
        _assert_control_equivalence(record["control"], changed_hash, seed=17)
    changed_trace = copy.deepcopy(record["probe"])
    changed_trace["updates"][0]["gradient_norm"] += 1.0
    with pytest.raises(RuntimeError, match="CONTROL_EQUIVALENCE_TRACE"):
        _assert_control_equivalence(record["control"], changed_trace, seed=17)


def test_all_arms_share_exact_initialization_per_seed(causal_smoke_root) -> None:
    payload = _load(causal_smoke_root)
    hashes = {
        row["training"]["initialization_canonical_sha256"]
        for row in payload["seed_results"]
    }
    assert len(hashes) == 1
    for row in payload["seed_results"]:
        assert len(row["training"]["updates"]) == 9
        expected_order = row["contract"]["task_order"] * 3
        assert [update["task_id"] for update in row["training"]["updates"]] == expected_order


def test_required_causal_metrics_are_persisted(causal_smoke_root) -> None:
    payload = _load(causal_smoke_root)
    for arm in ARMS:
        teacher = payload["aggregates"][arm]["teacher_forced"]
        free = payload["aggregates"][arm]["free_running"]
        assert teacher["position0_unstack"]["target_count"] == 2
        assert teacher["position0_unstack"]["accuracy"] is not None
        assert teacher["position0_unstack"]["mean_end_probability"] is not None
        assert teacher["position4_end"]["target_count"] == 2
        assert teacher["position4_end"]["accuracy"] is not None
        for key in (
            "operator_accuracy",
            "non_end_operator_accuracy",
            "end_accuracy",
            "arg1_accuracy",
            "arg2_accuracy",
            "joint_step_accuracy",
        ):
            assert teacher[key] is not None
        for key in (
            "exact_plan_rate",
            "goal_success_rate",
            "zero_action_rate",
            "initially_unsatisfied_goal_success_rate",
        ):
            assert free[key] is not None


def test_validator_rejects_resigned_aggregate_tamper(causal_smoke_root, tmp_path) -> None:
    root, payload = _tampered_copy(causal_smoke_root, tmp_path / "aggregate")
    payload["aggregates"][ARM_EQUAL_POSITION]["teacher_forced"]["operator_accuracy"] = 0.123
    _resign_and_write(root, payload)
    with pytest.raises(ValueError, match="VALIDATOR_AGGREGATES"):
        validate_microexperiment(root, implementation_commit=IMPLEMENTATION)


def test_validator_rejects_resigned_contrast_tamper(causal_smoke_root, tmp_path) -> None:
    root, payload = _tampered_copy(causal_smoke_root, tmp_path / "contrast")
    payload["contrasts_vs_control"][ARM_ORDER_ONLY]["operator_accuracy_delta"] = 0.25
    _resign_and_write(root, payload)
    with pytest.raises(ValueError, match="VALIDATOR_CONTRASTS"):
        validate_microexperiment(root, implementation_commit=IMPLEMENTATION)


def test_validator_rejects_resigned_raw_evidence_tamper(causal_smoke_root, tmp_path) -> None:
    root, payload = _tampered_copy(causal_smoke_root, tmp_path / "raw")
    result = next(row for row in payload["seed_results"] if row["arm"] == ARM_CONTROL)
    position = result["teacher_forced"][0]["positions"][0]
    position["operator_correct"] = not position["operator_correct"]
    _resign_and_write(root, payload)
    with pytest.raises(ValueError, match="VALIDATOR_TEACHER_OPERATOR_CLAIM"):
        validate_microexperiment(root, implementation_commit=IMPLEMENTATION)


def test_validator_rejects_resigned_raw_claim_and_summary_tamper(
    causal_smoke_root, tmp_path
) -> None:
    root, payload = _tampered_copy(causal_smoke_root, tmp_path / "raw-consistent")
    result = next(row for row in payload["seed_results"] if row["arm"] == ARM_CONTROL)
    position = result["teacher_forced"][0]["positions"][0]
    position["operator_correct"] = not position["operator_correct"]
    result["teacher_forced_metrics"]["operator_accuracy"] = 0.0
    payload["aggregates"][ARM_CONTROL]["teacher_forced"]["operator_accuracy"] = 0.0
    _resign_and_write(root, payload)
    with pytest.raises(ValueError, match="VALIDATOR_TEACHER_OPERATOR_CLAIM"):
        validate_microexperiment(root, implementation_commit=IMPLEMENTATION)


def test_validator_rejects_resigned_equivalence_tamper(causal_smoke_root, tmp_path) -> None:
    root, payload = _tampered_copy(causal_smoke_root, tmp_path / "equivalence")
    payload["control_equivalence"][0]["probe"]["optimizer_canonical_sha256"] = (
        "sha256:" + "0" * 64
    )
    _resign_and_write(root, payload)
    with pytest.raises(ValueError, match="VALIDATOR_EQUIVALENCE_EXACT_MISMATCH"):
        validate_microexperiment(root, implementation_commit=IMPLEMENTATION)


def test_validate_only_binds_requested_implementation_commit(causal_smoke_root) -> None:
    with pytest.raises(ValueError, match="IMPLEMENTATION_COMMIT_MISMATCH"):
        validate_microexperiment(causal_smoke_root, implementation_commit="0" * 40)


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
        "planner_toy/a2_causal_microexperiment.py",
        "planner_toy/a2_causal_microexperiment_validator.py",
        ".github/workflows/a2-causal-microexperiment.yml",
        "scripts/run_a2_causal_microexperiment.py",
    }
    assert required <= set(SOURCE_FILES)


def test_cli_validate_only_passes_requested_implementation_commit() -> None:
    cli = (REPOSITORY / "scripts/run_a2_causal_microexperiment.py").read_text()
    assert "validate_microexperiment(" in cli
    assert "implementation_commit=args.implementation_commit" in cli


def test_workflow_is_manual_fixed_target_and_persists_hidden_evidence() -> None:
    workflow = (
        REPOSITORY / ".github/workflows/a2-causal-microexperiment.yml"
    ).read_text()
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "\npush:" not in workflow
    assert "planning-model-canonical-cpu-v1" in workflow
    assert "self-hosted" in workflow
    assert "validate-trusted-commit" in workflow
    assert "run_a2_causal_microexperiment" in workflow
    assert "--validate-only" in workflow
    assert '--implementation-commit "$A2_CAUSAL_IMPLEMENTATION_SHA"' in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "include-hidden-files: true" in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 30" in workflow
    assert "A3" not in workflow
    assert "GO_LATENT" not in workflow
