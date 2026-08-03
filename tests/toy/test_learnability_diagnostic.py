from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from planner_toy.canonical import canonical_bytes
from planner_toy.dataset import generate
from planner_toy.learnability import (
    FIRST_ERROR_CATEGORIES,
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    _dataset_context,
    _pipeline_replay,
    _train_a2_with_loss_trace,
    _TrainingLossObserver,
    aggregate_teacher_forced,
    classify_first_error,
    executable_prefix,
    free_running_task,
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
from planner_toy.training import ACTIONS, labels, state_dict_sha256

IMPLEMENTATION = "a" * 40


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
    return sorted(generate(17)["train"], key=lambda row: row["task_id"])


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("learnability-smoke") / "run"
    run(
        root,
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000001",),
    )
    return root


@pytest.fixture(scope="session")
def second_smoke_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("learnability-smoke-second") / "run"
    run(
        root,
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000001",),
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


def test_dataset_context_does_not_read_validation(monkeypatch, train_rows) -> None:
    accesses: list[str] = []

    class GuardedDataset(dict):
        def __getitem__(self, key):
            accesses.append(key)
            if key == "validation":
                raise AssertionError("held-out validation was read")
            return super().__getitem__(key)

    guarded = GuardedDataset(train=train_rows, dataset_hash="sha256:" + "1" * 64)
    monkeypatch.setattr("planner_toy.learnability.generate", lambda _seed: guarded)
    _dataset_context()
    assert "validation" not in accesses


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
    assert end["predicted_arg1"] is None
    assert end["arg1_correct"] is None
    assert end["predicted_arg2"] is None
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
    payload["teacher_forced"][0]["positions"][-1]["predicted_arg1"] = "@B0"
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
