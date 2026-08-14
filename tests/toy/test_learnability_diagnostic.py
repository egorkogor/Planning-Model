from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from planner_toy.canonical import canonical_bytes, sha256
from planner_toy.dataset import generate
from planner_toy.learnability import (
    FIRST_ERROR_CATEGORIES,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    SOURCE_FILES,
    TRAINING_REQUIRED_FILES,
    _artifact_identity,
    _dataset_context,
    _gold_history_projection,
    _pipeline_replay,
    _train_a2_with_loss_trace,
    _TrainingLossObserver,
    aggregate_free_running,
    aggregate_history_modes,
    aggregate_loss_breakdown,
    aggregate_teacher_forced,
    classify_first_error,
    diagnostic_source_identity,
    diagnostic_source_identity_at_commit,
    executable_prefix,
    free_running_task,
    implementation_provenance,
    render_markdown,
    run,
    teacher_forced_task,
    validate_diagnostic,
    validate_payload,
)
from planner_toy.model import LockedPlanner, canonical_task_encoding
from planner_toy.numeric_identity import (
    canonical_state_dict_sha256,
    canonical_torch_object_sha256,
)
from planner_toy.quality import (
    _optimizer_named_parameters,
)
from planner_toy.quality import (
    _train as quality_train,
)
from planner_toy.train_only_dataset import (
    FROZEN_DATASET_LINEAGE_HASH_V1,
    generate_train_only,
)
from planner_toy.training import ACTIONS, labels, state_dict_sha256

REPOSITORY = Path(__file__).parents[2]
IMPLEMENTATION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
).strip()
HISTORICAL_QUALITY_IMPLEMENTATION = "779172c3bbca3d03552deaed6421e82fcf19a932"


def _reseal(payload: dict) -> None:
    payload["aggregates"]["teacher_forced"] = aggregate_teacher_forced(
        payload["teacher_forced"]
    )
    payload["aggregates"]["free_running"] = aggregate_free_running(
        payload["free_running"]
    )
    payload["aggregates"]["history_mode_comparison"] = aggregate_history_modes(
        payload["teacher_forced"], payload["free_running"]
    )
    payload["aggregates"]["loss_breakdown"] = aggregate_loss_breakdown(
        payload["per_update_loss_breakdown"]
    )
    payload["first_error_distribution"] = payload["aggregates"]["free_running"][
        "overall"
    ]["first_error_distribution"]
    payload["canonical_identity"] = _artifact_identity(payload)


def _write_payload(root: Path, payload: dict) -> None:
    payload["canonical_identity"] = _artifact_identity(payload)
    (root / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (root / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")


def _copy_bundle(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


class ScriptedA2(torch.nn.Module):
    variant = "A2"

    def __init__(
        self,
        action_ids: list[int],
        *,
        arg1_ids: list[int] | None = None,
        arg2_ids: list[int] | None = None,
        end_probabilities: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.action_ids = action_ids
        self.arg1_ids = arg1_ids or [0] * len(action_ids)
        self.arg2_ids = arg2_ids or [0] * len(action_ids)
        self.end_probabilities = end_probabilities
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, _encoded, actions, arg1, arg2, **_kwargs):
        self.calls.append((actions.clone(), arg1.clone(), arg2.clone()))
        call = len(self.calls) - 1
        action = torch.full((1, 17, 5), -20.0)
        pointer1 = torch.full((1, 17, 3), -20.0)
        pointer2 = torch.full((1, 17, 3), -20.0)
        for position in range(17):
            index = min(position if len(self.calls) == 1 else call, len(self.action_ids) - 1)
            action_id = self.action_ids[index]
            action[0, position, action_id] = 20.0
            if self.end_probabilities is not None and action_id != ACTIONS["END"]:
                probability = self.end_probabilities[min(index, len(self.end_probabilities) - 1)]
                probability = min(max(probability, 1e-6), 1 - 1e-6)
                action[0, position, ACTIONS["END"]] = torch.log(
                    torch.tensor(probability / (1 - probability))
                )
            pointer1[0, position, self.arg1_ids[index]] = 20.0
            pointer2[0, position, self.arg2_ids[index]] = 20.0
        return SimpleNamespace(
            action=action,
            arg1=pointer1,
            arg2=pointer2,
            z_semantic=None,
            projected_semantic=None,
            semantic_component=None,
        )


@pytest.fixture(scope="session")
def train_rows():
    return sorted(generate_train_only(17)["train"], key=lambda row: row["task_id"])


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("learnability-smoke") / "run"
    run(
        root,
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000003",),
    )
    return root


@pytest.fixture(scope="session")
def second_smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("learnability-smoke-second") / "run"
    run(
        root,
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000003",),
    )
    return root


def test_teacher_forced_uses_gold_prefix(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    action, arg1, arg2 = labels(row)
    model = ScriptedA2([ACTIONS[step[0]] for step in row["oracle_work_plan"]])
    teacher_forced_task(model, row, split="train", seed=17)
    captured_action, captured_arg1, captured_arg2 = model.calls[0]
    assert torch.equal(captured_action, action)
    assert torch.equal(captured_arg1, arg1)
    assert torch.equal(captured_arg2, arg2)


def test_free_running_never_receives_gold_prefix(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    model = ScriptedA2([ACTIONS["PICK_UP"], ACTIONS["END"]])
    result = free_running_task(model, row, split="train", seed=17)
    assert result["predicted_plan"][0][0] == "PICK_UP"
    assert model.calls[0][0][0, 0].item() == ACTIONS["END"]
    assert model.calls[1][0][0, 0].item() == ACTIONS["PICK_UP"]
    assert model.calls[1][0][0, 0].item() != ACTIONS[row["oracle_work_plan"][0][0]]


def test_dataset_context_uses_train_only_helper(monkeypatch, train_rows) -> None:
    calls = []

    def fake_train_only(seed):
        calls.append(seed)
        return {
            "schema_version": "toy-a2-train-only-dataset/1.0",
            "seed": 17,
            "train": train_rows,
            "train_task_ids": [row["task_id"] for row in generate(17)["train"]],
            "frozen_dataset_lineage_hash": FROZEN_DATASET_LINEAGE_HASH_V1,
            "evaluated_train_split_hash": sha256(
                {
                    "schema_version": "toy-a2-evaluated-train-split-hash/1.0",
                    "seed": 17,
                    "train": [
                        next(row for row in train_rows if row["task_id"] == task_id)
                        for task_id in ("bw-00000001", "bw-00000003", "bw-00000002")
                    ],
                }
            ),
        }

    monkeypatch.setattr("planner_toy.learnability.generate_train_only", fake_train_only)
    _dataset_context()
    assert calls == [17]


def test_operator_accuracy_and_end_probability_are_position_specific(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    actions = [ACTIONS[step[0]] for step in row["oracle_work_plan"]]
    model = ScriptedA2(actions, end_probabilities=[0.1, 0.2, 0.3, 0.4, 0.8])
    result = teacher_forced_task(model, row, split="train", seed=17)
    positions = result["positions"]
    assert all(position["operator_correct"] for position in positions)
    assert positions[0]["probability_end"] != positions[-1]["probability_end"]
    assert positions[-1]["gold_operator"] == "END"


def test_pointer_denominators_exclude_end_positions(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    actions = [ACTIONS[step[0]] for step in row["oracle_work_plan"]]
    task = teacher_forced_task(ScriptedA2(actions), row, split="train", seed=17)
    summary = aggregate_teacher_forced([task])["overall"]
    assert summary["position_count"] == len(row["oracle_work_plan"])
    assert summary["arg1_target_count"] == len(row["oracle_work_plan"]) - 1
    assert summary["arg2_target_count"] == 2
    end = task["positions"][-1]
    assert end["arg1_head_prediction"] is None
    assert end["arg1_correct"] is None
    assert end["arg2_head_prediction"] is None
    assert end["arg2_correct"] is None


def test_joint_step_requires_operator_and_arguments(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    actions = [ACTIONS[step[0]] for step in row["oracle_work_plan"]]
    model = ScriptedA2(actions, arg1_ids=[0, 0, 0, 0, 0], arg2_ids=[0, 0, 0, 0, 0])
    task = teacher_forced_task(model, row, split="train", seed=17)
    assert any(
        position["operator_correct"] and not position["joint_step_correct"]
        for position in task["positions"]
    )


@pytest.mark.parametrize(
    "expected,gold,predicted,parse_pos,precondition_pos,goal_success,position",
    [
        ("NONE", [["END"]], [["END"]], None, None, True, None),
        ("EARLY_END", [["PICK_UP", "@B0"], ["END"]], [["END"]], None, None, False, 0),
        (
            "WRONG_OPERATOR",
            [["PICK_UP", "@B0"], ["END"]],
            [["PUT_DOWN", "@B0"], ["END"]],
            None, None, False, 0,
        ),
        (
            "WRONG_ARG1",
            [["PICK_UP", "@B0"], ["END"]],
            [["PICK_UP", "@B1"], ["END"]],
            None, None, False, 0,
        ),
        (
            "WRONG_ARG2",
            [["STACK", "@B0", "@B1"], ["END"]],
            [["STACK", "@B0", "@B2"], ["END"]],
            None, None, False, 0,
        ),
        (
            "EXTRA_ACTION_AFTER_GOLD_END", [["END"]],
            [["PICK_UP", "@B0"], ["END"]], None, None, False, 0,
        ),
        ("PARSE_FAILURE", [["END"]], [["BROKEN"]], 0, None, False, 0),
        (
            "PRECONDITION_FAILURE",
            [["PICK_UP", "@B0"], ["END"]],
            [["PUT_DOWN", "@B0"], ["END"]],
            None, 0, False, 0,
        ),
        ("GOAL_NOT_ACHIEVED", [["END"]], [["END"]], None, None, False, None),
    ],
)
def test_first_error_taxonomy(
    expected, gold, predicted, parse_pos, precondition_pos, goal_success, position
) -> None:
    category, actual_position = classify_first_error(
        gold=gold,
        predicted=predicted,
        parser_failure_position=parse_pos,
        precondition_failure_position=precondition_pos,
        goal_success=goal_success,
    )
    assert category == expected
    assert actual_position == position
    assert category in FIRST_ERROR_CATEGORIES



def test_first_error_uses_earliest_causal_position() -> None:
    category, position = classify_first_error(
        gold=[["PICK_UP", "@B0"], ["END"]],
        predicted=[["PUT_DOWN", "@B0"], ["BROKEN"]],
        parser_failure_position=1,
        precondition_failure_position=None,
        goal_success=False,
    )
    assert (category, position) == ("WRONG_OPERATOR", 0)

def test_empty_predicted_plan_is_not_executable_when_goal_unsatisfied() -> None:
    row = next(row for row in generate(17)["train"] if row["task_id"] == "bw-00000003")
    result = executable_prefix(row, [["END"]])
    assert result["executable_prefix_length"] == 0
    assert result["executable_prefix_fraction_of_predicted"] is None
    assert result["full_plan_executable"] is False
    assert result["final_goal_success"] is False


def test_executable_prefix_uses_evolving_state(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    result = executable_prefix(row, row["oracle_work_plan"])
    assert result["executable_prefix_length"] == 4
    assert result["full_plan_executable"] is True
    assert result["final_goal_success"] is True
    broken = executable_prefix(
        row,
        [["UNSTACK", "@B1", "@B0"], ["PICK_UP", "@B0"], ["END"]],
    )
    assert broken["executable_prefix_length"] == 1
    assert broken["full_plan_executable"] is False


def test_read_only_pass_restores_parameters_and_rng_in_subprocess() -> None:
    code = """
import torch
import torch.nn.functional as F
from planner_toy.learnability import _read_only_diagnostic_pass
model = torch.nn.Linear(2, 2)
before = {key: value.clone() for key, value in model.state_dict().items()}
torch.manual_seed(123)
state = torch.get_rng_state().clone()
with _read_only_diagnostic_pass(model):
    _ = model(torch.ones(1, 2))
    _ = torch.rand(3)
assert all(torch.equal(before[key], model.state_dict()[key]) for key in before)
assert torch.equal(state, torch.get_rng_state())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_artifact_passes_schema_and_semantic_validation(smoke_root) -> None:
    result = validate_diagnostic(smoke_root)
    assert result["valid"] is True
    assert result["diagnostic_complete"] is False


def test_schema_violation_uses_public_value_error_contract(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    payload["heldout_accessed"] = True
    with pytest.raises(ValueError, match="LEARNABILITY_SCHEMA_VALIDATION_FAILED"):
        validate_payload(payload, root=smoke_root)


def test_semantic_violation_uses_public_value_error_contract(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    payload["canonical_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="LEARNABILITY_CANONICAL_IDENTITY_MISMATCH"):
        validate_payload(payload, root=smoke_root)


def test_diagnostic_does_not_change_checkpoint_optimizer_or_pipeline(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    record = payload["diagnostic_invariance"][0]
    assert record["model_parameters_unchanged"] is True
    assert record["checkpoint_bytes_unchanged"] is True
    assert record["optimizer_state_unchanged"] is True
    assert record["pipeline_replay_unchanged"] is True
    assert record["rng_state_restored"] is True
    assert record["checkpoint_canonical_before"] == record["checkpoint_canonical_after"]
    assert record["optimizer_canonical_before"] == record["optimizer_canonical_after"]


def test_loss_observer_preserves_scalar_gradients_and_update(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000001")
    plain = LockedPlanner(17, "A2").cpu()
    observed = LockedPlanner(17, "A2").cpu()
    observed.load_state_dict(copy.deepcopy(plain.state_dict()))

    def one_update(model, observer=None):
        named = _optimizer_named_parameters(model)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
        action, arg1, arg2 = labels(row)
        valid = len(row["oracle_work_plan"])
        optimizer.zero_grad(set_to_none=True)

        def compute_and_step():
            logits = model(canonical_task_encoding(row), action, arg1, arg2)
            flat = action[:, :valid].flatten()
            loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
            one = flat != ACTIONS["END"]
            two = (flat == ACTIONS["UNSTACK"]) | (flat == ACTIONS["STACK"])
            if one.any():
                loss = loss + F.cross_entropy(
                    logits.arg1[:, :valid].flatten(0, 1)[one],
                    arg1[:, :valid].flatten()[one],
                )
            if two.any():
                loss = loss + F.cross_entropy(
                    logits.arg2[:, :valid].flatten(0, 1)[two],
                    arg2[:, :valid].flatten()[two],
                )
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in named
                if parameter.grad is not None
            }
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], 1.0
            )
            optimizer.step()
            return (
                float(loss.detach()),
                gradients,
                float(norm),
                canonical_state_dict_sha256(model.state_dict()),
                canonical_torch_object_sha256(optimizer.state_dict()),
            )

        if observer is None:
            return compute_and_step()
        observer.schedule = [(0, row)]
        with observer:
            return compute_and_step()

    plain_result = one_update(plain)
    observed_result = one_update(observed, _TrainingLossObserver([row]))
    assert plain_result[0] == pytest.approx(observed_result[0], rel=0, abs=0)
    assert plain_result[1].keys() == observed_result[1].keys()
    assert all(
        torch.equal(plain_result[1][name], observed_result[1][name])
        for name in plain_result[1]
    )
    assert plain_result[2] == pytest.approx(observed_result[2], rel=0, abs=0)
    assert plain_result[3:] == observed_result[3:]


def test_training_observer_preserves_checkpoint_and_optimizer(
    tmp_path, smoke_root, train_rows
) -> None:
    plain_root = tmp_path / "plain"
    model, manifest = quality_train(
        train_rows,
        "A2",
        17,
        plain_root,
        generate(17)["dataset_hash"],
    )
    observed_payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    observed = observed_payload["checkpoints"][0]
    assert state_dict_sha256(model.state_dict()) == manifest["trained_state_dict_sha256"]
    assert canonical_state_dict_sha256(model.state_dict()) == observed[
        "canonical_trained_state_dict_sha256"
    ]
    assert manifest["canonical_optimizer_state_sha256"] == observed[
        "canonical_optimizer_state_sha256"
    ]
    observed_model = LockedPlanner(17, "A2").cpu()
    observed_model.load_state_dict(
        torch.load(
            smoke_root / observed["trained_checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )
    )
    selected = [row for row in train_rows if row["task_id"] == "bw-00000001"]
    assert _pipeline_replay(model, selected) == _pipeline_replay(observed_model, selected)


def test_loss_breakdown_matches_existing_total_without_new_targets(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    rows = payload["per_update_loss_breakdown"]
    assert len(rows) == 9
    for row in rows:
        total = row["operator_loss"]
        if row["arg1_pointer_loss"] is not None:
            total += row["arg1_pointer_loss"]
        if row["arg2_pointer_loss"] is not None:
            total += row["arg2_pointer_loss"]
        assert total == pytest.approx(row["total_loss"], rel=1e-6, abs=1e-7)
        assert row["operator_target_count"] > 0


def test_repeated_smoke_runs_are_byte_identical(smoke_root, second_smoke_root) -> None:
    assert (smoke_root / OUTPUT_JSON).read_bytes() == (second_smoke_root / OUTPUT_JSON).read_bytes()
    assert (smoke_root / OUTPUT_MARKDOWN).read_bytes() == (
        second_smoke_root / OUTPUT_MARKDOWN
    ).read_bytes()


def test_mutated_artifact_is_rejected(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    payload["teacher_forced"][0]["positions"][-1]["arg1_head_prediction"] = "@B0"
    payload["canonical_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        validate_payload(payload, root=smoke_root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"heldout_accessed": True}),
        lambda payload: payload["evaluated_task_ids"].append("bw-00000004"),
        lambda payload: payload["teacher_forced"].append(
            copy.deepcopy(payload["teacher_forced"][0])
        ),
        lambda payload: payload["free_running"][0].update(
            {"executable_prefix_length": 99}
        ),
        lambda payload: payload["training_config"].update({"epochs": 4}),
    ],
)
def test_scope_and_semantic_mutations_are_rejected(smoke_root, mutate) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    mutate(payload)
    payload["canonical_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        validate_payload(payload, root=smoke_root)


def test_nonfinite_value_is_rejected(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    payload["per_update_loss_breakdown"][0]["operator_loss"] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE"):
        validate_payload(payload, root=smoke_root)


def test_markdown_renderer_is_deterministic(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    assert render_markdown(payload) == render_markdown(copy.deepcopy(payload))
    assert (smoke_root / OUTPUT_MARKDOWN).read_text() == render_markdown(payload)


def test_a3_and_a4_are_never_trained(monkeypatch, tmp_path, train_rows) -> None:
    variants: list[str] = []

    def fake_train(rows, variant, seed, output, dataset_hash):
        variants.append(variant)
        model = ScriptedA2([ACTIONS["END"]])
        output.mkdir(parents=True, exist_ok=True)
        state = model.state_dict()
        torch.save(state, output / "trained.pt")
        optimizer = {"state": {}, "param_groups": []}
        torch.save(optimizer, output / "optimizer-state.pt")
        manifest = {
            "trained_state_dict_sha256": state_dict_sha256(state),
            "canonical_trained_state_dict_sha256": canonical_state_dict_sha256(state),
            "optimizer_state_sha256": "sha256:" + "1" * 64,
            "canonical_optimizer_state_sha256": "sha256:" + "2" * 64,
        }
        return model, manifest

    monkeypatch.setattr("planner_toy.learnability._quality_train", fake_train)
    with pytest.raises(RuntimeError, match="LOSS_COMPONENT_COUNT|UPDATE_COUNT"):
        _train_a2_with_loss_trace(train_rows, 17, tmp_path / "run", "sha256:" + "3" * 64)
    assert variants == ["A2"]


def test_frozen_v0_1_artifacts_have_expected_bytes() -> None:
    expected = {
        "docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md":
            "133b3be3ec0d1a9d1025c32ddd60413344cea03bd4acb84eb8b60cc0e3a46df1",
        "docs/evaluations/data/a2_a3_a4_heldout_summary.json":
            "0a9c0cb0ee98fa4d458cb0a6fafa2e9e78bb0771a223d46a035f458c566df050",
        "docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md":
            "a1392b21961c728c7b4fa8686d6d095a345dc319fdfeec2b1e47eaaccb9b754a",
    }
    root = Path(__file__).parents[2]
    for relative, digest in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest


def test_canonical_json_has_no_nonfinite_values(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    assert canonical_bytes(payload) + b"\n" == (smoke_root / OUTPUT_JSON).read_bytes()


def _gold_pointer_ids(row: dict) -> tuple[list[int], list[int]]:
    blocks = row["blocks"]
    arg1_ids = []
    arg2_ids = []
    for step in row["oracle_work_plan"]:
        arg1_ids.append(blocks.index(step[1]) if len(step) > 1 else 0)
        arg2_ids.append(blocks.index(step[2]) if len(step) > 2 else 0)
    return arg1_ids, arg2_ids


def test_train_only_rows_match_historical_train_bytes() -> None:
    full = generate(17)
    train_only = generate_train_only(17)
    assert canonical_bytes(train_only["train"]) == canonical_bytes(full["train"])
    assert [row["task_id"] for row in train_only["train"]] == [
        "bw-00000001",
        "bw-00000003",
        "bw-00000002",
    ]


def test_full_dataset_identity_is_unchanged() -> None:
    full = generate(17)
    assert full["dataset_hash"] == FROZEN_DATASET_LINEAGE_HASH_V1
    assert hashlib.sha256(canonical_bytes(full) + b"\n").hexdigest() == (
        "e5911f6cd80288083ba9ab56a9d8bde41bb1aa8e93f391e44ba9960c83dbe8d4"
    )


def test_train_only_hash_is_versioned_and_stable() -> None:
    train_only = generate_train_only(17)
    assert train_only["evaluated_train_split_hash"] == (
        "sha256:4bbbc91e7e2561c91b26cddb1ccf995df7850646150937cf4f71607dfcf7f9a6"
    )
    assert train_only["frozen_dataset_lineage_hash"] == FROZEN_DATASET_LINEAGE_HASH_V1


def test_train_only_never_builds_heldout_tasks(monkeypatch) -> None:
    import planner_toy.train_only_dataset as dataset_module

    calls: list[int] = []
    original = dataset_module._build_train_row

    def observed(serial: int):
        calls.append(serial)
        return original(serial)

    monkeypatch.setattr(dataset_module, "_build_train_row", observed)
    generate_train_only(17)
    assert calls == [1, 3, 2]
    assert not set(calls) & {4, 5}


def test_train_only_never_calls_shortest_plan_for_heldout(monkeypatch) -> None:
    import planner_toy.train_only_dataset as dataset_module

    task_ids: list[str] = []
    original = dataset_module.shortest_plan

    def observed(task):
        task_ids.append(task.task_id)
        return original(task)

    monkeypatch.setattr(dataset_module, "shortest_plan", observed)
    generate_train_only(17)
    assert task_ids == ["bw-00000001", "bw-00000003", "bw-00000002"]
    assert "bw-00000004" not in task_ids
    assert "bw-00000005" not in task_ids


def test_heldout_definition_mutation_does_not_change_train_hash(monkeypatch) -> None:
    import planner_toy.train_only_dataset as dataset_module

    before = generate_train_only(17)["evaluated_train_split_hash"]
    assert set(dataset_module._TASK_SPECS_V1) == {1, 2, 3}
    assert generate_train_only(17)["evaluated_train_split_hash"] == before


def test_train_row_mutation_changes_train_hash(monkeypatch) -> None:
    import planner_toy.train_only_dataset as dataset_module

    before = generate_train_only(17)["evaluated_train_split_hash"]
    original = dataset_module._build_train_row

    def mutated(serial: int):
        row = copy.deepcopy(original(serial))
        if serial == 2:
            row["oracle_work_plan"][0][0] = "PUT_DOWN"
        return row

    monkeypatch.setattr(dataset_module, "_build_train_row", mutated)
    after = generate_train_only(17)["evaluated_train_split_hash"]
    assert after != before


def test_diagnostic_run_does_not_materialize_heldout(monkeypatch, tmp_path) -> None:
    import planner_toy.train_only_dataset as dataset_module

    calls: list[int] = []
    original = dataset_module._build_train_row

    def observed(serial: int):
        calls.append(serial)
        if serial in {4, 5}:
            raise AssertionError("held-out task was materialized")
        return original(serial)

    monkeypatch.setattr(dataset_module, "_build_train_row", observed)
    run(
        tmp_path / "run",
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000001",),
    )
    assert calls and set(calls) == {1, 2, 3}


def test_heldout_access_flag_is_bound_to_train_only_hash(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    train_only = generate_train_only(17)
    assert payload["heldout_accessed"] is False
    assert payload["frozen_dataset_lineage_hash"] == FROZEN_DATASET_LINEAGE_HASH_V1
    assert payload["evaluated_train_split_hash"] == train_only[
        "evaluated_train_split_hash"
    ]


def test_pointer_heads_remain_visible_when_operator_predicts_end(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    arg1_ids, arg2_ids = _gold_pointer_ids(row)
    action_ids = [ACTIONS["END"]] + [ACTIONS[step[0]] for step in row["oracle_work_plan"][1:]]
    task = teacher_forced_task(
        ScriptedA2(action_ids, arg1_ids=arg1_ids, arg2_ids=arg2_ids),
        row,
        split="train",
        seed=17,
    )
    first = task["positions"][0]
    assert first["gold_operator"] == "UNSTACK"
    assert first["predicted_operator"] == "END"
    assert first["operator_correct"] is False
    assert first["arg1_head_prediction"] == first["gold_arg1"]
    assert first["arg2_head_prediction"] == first["gold_arg2"]
    assert first["arg1_correct"] is True
    assert first["arg2_correct"] is True
    assert first["decoded_arg1"] is None
    assert first["decoded_arg2"] is None
    assert first["joint_step_correct"] is False


def test_pointer_accuracy_can_be_perfect_with_zero_operator_accuracy(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    arg1_ids, arg2_ids = _gold_pointer_ids(row)
    action_ids = [
        ACTIONS["END"] if step[0] != "END" else ACTIONS["PICK_UP"]
        for step in row["oracle_work_plan"]
    ]
    task = teacher_forced_task(
        ScriptedA2(action_ids, arg1_ids=arg1_ids, arg2_ids=arg2_ids),
        row,
        split="train",
        seed=17,
    )
    summary = aggregate_teacher_forced([task])["overall"]
    assert summary["operator_accuracy"] == 0.0
    assert summary["arg1_accuracy"] == 1.0
    assert summary["arg2_accuracy"] == 1.0
    assert summary["joint_step_accuracy"] == 0.0
    assert summary["arg1_target_count"] == 4
    assert summary["arg2_target_count"] == 2


def test_gold_end_position_has_null_pointer_head_metrics(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    actions = [ACTIONS[step[0]] for step in row["oracle_work_plan"]]
    task = teacher_forced_task(ScriptedA2(actions), row, split="train", seed=17)
    end = task["positions"][-1]
    assert end["gold_operator"] == "END"
    assert end["gold_arg1"] is None
    assert end["gold_arg2"] is None
    assert end["arg1_head_prediction"] is None
    assert end["arg2_head_prediction"] is None
    assert end["arg1_correct"] is None
    assert end["arg2_correct"] is None


def test_gold_history_metrics_keep_all_positions_after_early_end(train_rows) -> None:
    row = next(row for row in train_rows if row["task_id"] == "bw-00000003")
    arg1_ids, arg2_ids = _gold_pointer_ids(row)
    action_ids = [ACTIONS["END"]] + [
        ACTIONS[step[0]] for step in row["oracle_work_plan"][1:]
    ]
    teacher = teacher_forced_task(
        ScriptedA2(action_ids, arg1_ids=arg1_ids, arg2_ids=arg2_ids),
        row,
        split="train",
        seed=17,
    )
    projected = _gold_history_projection(teacher, row)
    free = free_running_task(
        ScriptedA2([ACTIONS["END"]]), row, split="train", seed=17
    )
    comparison = aggregate_history_modes([teacher], [free])["gold_history"]
    expected_operator_correct = sum(
        position["operator_correct"] for position in teacher["positions"]
    )
    expected_joint_correct = sum(
        position["joint_step_correct"] for position in teacher["positions"]
    )
    assert comparison["predicted_position_count"] == len(row["oracle_work_plan"])
    assert comparison["operator_accuracy"] == pytest.approx(
        expected_operator_correct / len(row["oracle_work_plan"])
    )
    assert comparison["joint_step_accuracy"] == pytest.approx(
        expected_joint_correct / len(row["oracle_work_plan"])
    )
    assert projected["predicted_plan"] == [["END"]]
    assert projected["first_mismatch_type"] == "EARLY_END"
    assert projected["first_mismatch_position"] == 0


def test_markdown_separates_operator_pointer_and_joint_metrics(smoke_root) -> None:
    markdown = (smoke_root / OUTPUT_MARKDOWN).read_text()
    assert "Operator-head accuracy" in markdown
    assert "Arg1-head accuracy" in markdown
    assert "Arg2-head accuracy" in markdown
    assert "Joint decoded-step accuracy" in markdown


EXPECTED_DIAGNOSTIC_SOURCE_FILES = (
    "docs/architecture/planner_module_inventory_v1.yaml",
    "docs/architecture/task_encoding_v1.yaml",
    "docs/evaluations/A2_END_COLLAPSE_DIAGNOSTIC_SPEC_RU.md",
    "planner_toy/canonical.py",
    "planner_toy/canonical_runtime.py",
    "planner_toy/dataset.py",
    "planner_toy/domain.py",
    "planner_toy/e2e.py",
    "planner_toy/learnability.py",
    "planner_toy/model.py",
    "planner_toy/numeric_identity.py",
    "planner_toy/quality.py",
    "planner_toy/schemas/learnability_diagnostic.schema.json",
    "planner_toy/schemas/toy_quality_checkpoint_manifest.schema.json",
    "planner_toy/schemas/toy_quality_optimizer_evidence.schema.json",
    "planner_toy/schemas/toy_quality_training_config.schema.json",
    "planner_toy/schemas/toy_quality_training_report.schema.json",
    "planner_toy/semantic.py",
    "planner_toy/training.py",
    "planner_toy/train_only_dataset.py",
    "scripts/run_toy_learnability_diagnostic.py",
)


def test_exact_diagnostic_source_inventory() -> None:
    assert SOURCE_FILES == EXPECTED_DIAGNOSTIC_SOURCE_FILES


@pytest.mark.parametrize(
    "required",
    [
        "docs/architecture/planner_module_inventory_v1.yaml",
        "docs/architecture/task_encoding_v1.yaml",
        "planner_toy/semantic.py",
        "planner_toy/quality.py",
        "planner_toy/model.py",
        "planner_toy/training.py",
        "planner_toy/dataset.py",
    ],
)
def test_runtime_dependency_is_in_source_inventory(required) -> None:
    assert required in SOURCE_FILES


@pytest.mark.parametrize(
    "relative",
    [
        "docs/architecture/planner_module_inventory_v1.yaml",
        "docs/architecture/task_encoding_v1.yaml",
        "planner_toy/semantic.py",
    ],
)
def test_transitive_dependency_mutation_changes_source_hash(
    monkeypatch, tmp_path, relative
) -> None:
    import planner_toy.learnability as learnability_module

    root = tmp_path / "repository"
    for source in SOURCE_FILES:
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / source, target)
    monkeypatch.setattr(learnability_module, "ROOT", root)
    before = diagnostic_source_identity()["diagnostic_source_sha256"]
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n# mutation\n")
    after = diagnostic_source_identity()["diagnostic_source_sha256"]
    assert after != before


@pytest.mark.parametrize(
    "missing",
    [
        "docs/architecture/planner_module_inventory_v1.yaml",
        "docs/architecture/task_encoding_v1.yaml",
        "planner_toy/semantic.py",
    ],
)
def test_missing_transitive_dependency_at_commit_is_rejected(
    monkeypatch, missing
) -> None:
    import planner_toy.learnability as learnability_module

    original = learnability_module._git_bytes

    def missing_file(*args):
        if args == ("show", f"{IMPLEMENTATION}:{missing}"):
            return subprocess.CompletedProcess(args, 1, b"", b"missing")
        return original(*args)

    monkeypatch.setattr(learnability_module, "_git_bytes", missing_file)
    with pytest.raises(ValueError, match="LEARNABILITY_IMPLEMENTATION_SOURCE_MISSING"):
        diagnostic_source_identity_at_commit(IMPLEMENTATION)


def test_nonexistent_full_sha_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="LEARNABILITY_IMPLEMENTATION_COMMIT_NOT_FOUND"
    ):
        diagnostic_source_identity_at_commit("f" * 40)


def test_reachable_commit_without_diagnostic_sources_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="LEARNABILITY_IMPLEMENTATION_SOURCE_MISSING"
    ):
        diagnostic_source_identity_at_commit(HISTORICAL_QUALITY_IMPLEMENTATION)


def test_source_identity_and_requirements_are_read_from_implementation_tree() -> None:
    provenance = implementation_provenance(IMPLEMENTATION)
    for entry in provenance["diagnostic_source_files"]:
        source_bytes = subprocess.check_output(
            ["git", "show", f"{IMPLEMENTATION}:{entry['path']}"], cwd=REPOSITORY
        )
        assert entry["sha256"] == "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    requirements = subprocess.check_output(
        ["git", "show", f"{IMPLEMENTATION}:requirements.lock"], cwd=REPOSITORY
    )
    assert provenance["requirements_lock_sha256"] == (
        "sha256:" + hashlib.sha256(requirements).hexdigest()
    )


def test_generation_rejects_working_tree_source_mismatch(monkeypatch, tmp_path) -> None:
    current = diagnostic_source_identity()
    mutated = copy.deepcopy(current)
    mutated["diagnostic_source_sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        "planner_toy.learnability.diagnostic_source_identity", lambda: mutated
    )
    with pytest.raises(ValueError, match="LEARNABILITY_WORKING_TREE_SOURCE_MISMATCH"):
        run(
            tmp_path / "run",
            implementation_commit=IMPLEMENTATION,
            seeds=(17,),
            task_ids=("bw-00000001",),
        )


def test_validator_uses_implementation_tree_not_working_tree(
    smoke_root, monkeypatch
) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())

    def forbidden_current_identity():
        raise AssertionError("working-tree identity must not be consulted")

    monkeypatch.setattr(
        "planner_toy.learnability.diagnostic_source_identity",
        forbidden_current_identity,
    )
    validate_payload(payload, root=smoke_root)


def test_resealed_implementation_commit_substitution_is_rejected(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    payload["implementation_commit"] = HISTORICAL_QUALITY_IMPLEMENTATION
    _reseal(payload)
    with pytest.raises(
        ValueError, match="LEARNABILITY_IMPLEMENTATION_SOURCE_MISSING"
    ):
        validate_payload(payload, root=smoke_root)


@pytest.mark.parametrize(
    "mutation,error_code",
    [
        (
            lambda payload: payload["teacher_forced"][0]["positions"][0].update(
                {
                    "operator_correct": not payload["teacher_forced"][0][
                        "positions"
                    ][0]["operator_correct"]
                }
            ),
            "LEARNABILITY_TEACHER_OPERATOR_CORRECT_MISMATCH",
        ),
        (
            lambda payload: payload["teacher_forced"][0]["positions"][0].update(
                {
                    "arg1_correct": not payload["teacher_forced"][0][
                        "positions"
                    ][0]["arg1_correct"]
                }
            ),
            "LEARNABILITY_TEACHER_ARG1_CORRECT_MISMATCH",
        ),
        (
            lambda payload: payload["teacher_forced"][0]["positions"][0].update(
                {
                    "joint_step_correct": not payload["teacher_forced"][0][
                        "positions"
                    ][0]["joint_step_correct"]
                }
            ),
            "LEARNABILITY_TEACHER_JOINT_CORRECT_MISMATCH",
        ),
        (
            lambda payload: payload["free_running"][0].update(
                {
                    "predicted_pre_end_action_count": payload["free_running"][0][
                        "predicted_pre_end_action_count"
                    ]
                    + 1
                }
            ),
            "LEARNABILITY_PREDICTED_PRE_END_COUNT_MISMATCH",
        ),
        (
            lambda payload: payload["free_running"][0].update(
                {"failure_code": "PLAN_PARSE_ERROR"}
            ),
            "LEARNABILITY_FAILURE_CODE_MISMATCH",
        ),
        (
            lambda payload: payload["free_running"][0].update(
                {"parser_success": not payload["free_running"][0]["parser_success"]}
            ),
            "LEARNABILITY_PARSER_SUCCESS_MISMATCH",
        ),
    ],
)
def test_resealed_derived_mutations_are_rejected(
    smoke_root, mutation, error_code
) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    mutation(payload)
    _reseal(payload)
    assert payload["canonical_identity"] == _artifact_identity(payload)
    with pytest.raises(ValueError, match=error_code):
        validate_payload(payload, root=smoke_root)


def test_resealed_pointer_head_prediction_mutation_is_rejected(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    position = next(
        item
        for item in payload["teacher_forced"][0]["positions"]
        if item["gold_arg1"] is not None and item["predicted_operator"] != "END"
    )
    blocks = next(
        row["blocks"]
        for row in generate_train_only(17)["train"]
        if row["task_id"] == payload["teacher_forced"][0]["task_id"]
    )
    replacement = next(block for block in blocks if block != position["arg1_head_prediction"])
    position["arg1_head_prediction"] = replacement
    position["arg1_correct"] = replacement == position["gold_arg1"]
    position["joint_step_correct"] = (
        position["operator_correct"]
        and position["arg1_correct"] is not False
        and position["arg2_correct"] is not False
    )
    _reseal(payload)
    with pytest.raises(ValueError, match="LEARNABILITY_ARG1_HEAD_DECODED_MISMATCH"):
        validate_payload(payload, root=smoke_root)


def _refresh_training_surface(root: Path, payload: dict) -> None:
    training_root = root / "training-runs"
    payload["training_artifact_hashes"] = {
        str(path.relative_to(root)): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(training_root.rglob("*"))
        if path.is_file()
    }
    for record in payload["checkpoints"]:
        manifest_path = root / record["checkpoint_manifest_path"]
        if manifest_path.exists():
            record["checkpoint_manifest_file_sha256"] = (
                "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            )
    _write_payload(root, payload)


def _mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    path.write_bytes(canonical_bytes(value) + b"\n")


def test_training_lineage_required_file_set_is_exact(smoke_root) -> None:
    payload = json.loads((smoke_root / OUTPUT_JSON).read_text())
    expected = {
        f"training-runs/A2/seed-17/{name}" for name in TRAINING_REQUIRED_FILES
    }
    assert set(payload["training_artifact_hashes"]) == expected


@pytest.mark.parametrize(
    "case,error_code",
    [
        ("delete_initialization", "LEARNABILITY_TRAINING_FILE_COVERAGE_MISMATCH"),
        ("training_config", "LEARNABILITY_TRAINING_SCHEMA_INVALID"),
        ("training_report", "LEARNABILITY_TRAINING_SCHEMA_INVALID"),
        ("optimizer_evidence", "LEARNABILITY_TRAINING_SCHEMA_INVALID"),
        ("checkpoint_manifest", "LEARNABILITY_CHECKPOINT_LINEAGE_MISMATCH"),
        ("replace_checkpoint", "LEARNABILITY_TRAINED_CHECKPOINT_UNCHANGED"),
        ("extra_file", "LEARNABILITY_TRAINING_FILE_COVERAGE_MISMATCH"),
        ("missing_report", "LEARNABILITY_TRAINING_FILE_COVERAGE_MISMATCH"),
    ],
)
def test_resealed_training_lineage_mutations_are_rejected(
    smoke_root, tmp_path, case, error_code
) -> None:
    root = _copy_bundle(smoke_root, tmp_path / case)
    payload = json.loads((root / OUTPUT_JSON).read_text())
    run_dir = root / "training-runs" / "A2" / "seed-17"
    manifest_path = run_dir / "checkpoint-manifest.json"

    if case == "delete_initialization":
        (run_dir / "initialization.pt").unlink()
    elif case == "missing_report":
        (run_dir / "training-report.json").unlink()
    elif case == "extra_file":
        (run_dir / "unexpected.bin").write_bytes(b"unexpected")
    elif case == "training_config":
        _mutate_json(run_dir / "training-config.json", lambda value: value.update(epochs=4))
        _mutate_json(
            manifest_path,
            lambda value: value.update(
                config_hash="sha256:"
                + hashlib.sha256((run_dir / "training-config.json").read_bytes()).hexdigest()
            ),
        )
    elif case == "training_report":
        _mutate_json(
            run_dir / "training-report.json",
            lambda value: value.update(updates=8),
        )
        _mutate_json(
            manifest_path,
            lambda value: value.update(
                training_report_file_sha256="sha256:"
                + hashlib.sha256((run_dir / "training-report.json").read_bytes()).hexdigest()
            ),
        )
    elif case == "optimizer_evidence":
        _mutate_json(
            run_dir / "optimizer-evidence.json",
            lambda value: value.update(update_count=8),
        )
        _mutate_json(
            manifest_path,
            lambda value: value.update(
                optimizer_evidence_file_sha256="sha256:"
                + hashlib.sha256((run_dir / "optimizer-evidence.json").read_bytes()).hexdigest()
            ),
        )
    elif case == "checkpoint_manifest":
        _mutate_json(
            manifest_path,
            lambda value: value.update(checkpoint_origin_run_hash="sha256:" + "1" * 64),
        )
    elif case == "replace_checkpoint":
        initialization_path = run_dir / "initialization.pt"
        trained_path = run_dir / "trained.pt"
        trained_path.write_bytes(initialization_path.read_bytes())
        state = torch.load(initialization_path, map_location="cpu", weights_only=True)
        exact = state_dict_sha256(state)
        canonical = canonical_state_dict_sha256(state)
        _mutate_json(
            manifest_path,
            lambda value: value.update(
                trained_file_sha256="sha256:"
                + hashlib.sha256(trained_path.read_bytes()).hexdigest(),
                trained_state_dict_sha256=exact,
                canonical_trained_state_dict_sha256=canonical,
            ),
        )
        payload["checkpoints"][0][
            "canonical_trained_state_dict_sha256"
        ] = canonical
    _refresh_training_surface(root, payload)
    if case == "extra_file":
        payload = json.loads((root / OUTPUT_JSON).read_text())
        payload["training_artifact_hashes"].pop(
            "training-runs/A2/seed-17/unexpected.bin"
        )
        _write_payload(root, payload)
    resealed = json.loads((root / OUTPUT_JSON).read_text())
    assert resealed["canonical_identity"] == _artifact_identity(resealed)
    with pytest.raises(ValueError, match=error_code):
        validate_diagnostic(root)


def test_two_clean_process_bundles_are_recursive_byte_identical(tmp_path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.run_toy_learnability_diagnostic",
                "--output-dir",
                str(root),
                "--implementation-commit",
                IMPLEMENTATION,
                "--seeds",
                "17",
                "--task-ids",
                "bw-00000003",
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
    first_files = {
        str(path.relative_to(roots[0])): path.read_bytes()
        for path in roots[0].rglob("*")
        if path.is_file()
    }
    second_files = {
        str(path.relative_to(roots[1])): path.read_bytes()
        for path in roots[1].rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert validate_diagnostic(roots[0]) == validate_diagnostic(roots[1])


def test_dataset_lineage_and_optimizer_execution_orders_are_distinct() -> None:
    dataset, optimizer_rows = _dataset_context()
    assert dataset["train_task_ids"] == [
        "bw-00000001", "bw-00000003", "bw-00000002"
    ]
    assert [row["task_id"] for row in optimizer_rows] == [
        "bw-00000001", "bw-00000002", "bw-00000003"
    ]
