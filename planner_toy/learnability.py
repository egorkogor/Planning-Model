"""Read-only A2 learnability diagnostics for the frozen quality-v0.1 training setup."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .canonical import canonical_bytes, sha256
from .canonical_runtime import configure_canonical_cpu_runtime
from .dataset import generate, task_from_row
from .domain import apply_action, goal_satisfied, validate_state
from .e2e import (
    ACTION_NAMES,
    A2Planner,
    PlannerGenerationFailure,
    PlanParseFailure,
    parse_nonterminal_step,
    parse_work_plan,
)
from .model import LockedPlanner, canonical_task_encoding
from .numeric_identity import (
    canonical_state_dict_sha256,
    canonical_torch_object_sha256,
    exact_torch_object_sha256,
)
from .quality import _train as _quality_train
from .training import ACTIONS, labels, state_dict_sha256

VERSION = "development-learnability-diagnostic/0.1"
STATUS = "development-only-diagnostic"
SEEDS = (17, 29, 43)
EPOCHS = 3
UPDATES_PER_RUN = 9
MAX_STEPS = 17
VARIANT = "A2"
FIRST_ERROR_CATEGORIES = (
    "NONE",
    "EARLY_END",
    "WRONG_OPERATOR",
    "WRONG_ARG1",
    "WRONG_ARG2",
    "EXTRA_ACTION_AFTER_GOLD_END",
    "PARSE_FAILURE",
    "PRECONDITION_FAILURE",
    "GOAL_NOT_ACHIEVED",
)
FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS = ("bw-00000004", "bw-00000005")
ROOT = Path(__file__).parents[1]
SCHEMA_PATH = Path(__file__).parent / "schemas" / "toy_learnability_diagnostic.schema.json"
OUTPUT_JSON = "a2-end-collapse-diagnostic.json"
OUTPUT_MARKDOWN = "A2_END_COLLAPSE_DIAGNOSTIC.md"
TRAINING_CONFIG = {
    "epochs": EPOCHS,
    "updates_per_run": UPDATES_PER_RUN,
    "train_task_order": "task_id-ascending-per-epoch",
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
    },
    "checkpoint_policy": "final_epoch_only_no_heldout_selection",
    "max_decoding_steps": MAX_STEPS,
}
SOURCE_FILES = (
    "docs/evaluations/A2_END_COLLAPSE_DIAGNOSTIC_SPEC_RU.md",
    "planner_toy/canonical.py",
    "planner_toy/canonical_runtime.py",
    "planner_toy/dataset.py",
    "planner_toy/domain.py",
    "planner_toy/e2e.py",
    "planner_toy/learnability.py",
    "planner_toy/schemas/toy_learnability_diagnostic.schema.json",
    "planner_toy/model.py",
    "planner_toy/numeric_identity.py",
    "planner_toy/quality.py",
    "planner_toy/training.py",
    "scripts/run_toy_learnability_diagnostic.py",
)

_TRAINING_OBSERVER_LOCK = threading.Lock()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic_source_identity() -> dict[str, Any]:
    files = [{"path": path, "sha256": _file_hash(ROOT / path)} for path in SOURCE_FILES]
    return {"diagnostic_source_files": files, "diagnostic_source_sha256": sha256(files)}


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "execution_device": "cpu",
    }


def _rate(correct: int, total: int) -> float | None:
    return correct / total if total else None


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _rng_snapshot() -> tuple[object, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_rng(snapshot: tuple[object, tuple[Any, ...], torch.Tensor]) -> None:
    python_state, numpy_state, torch_state = snapshot
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def _rng_equal(
    left: tuple[object, tuple[Any, ...], torch.Tensor],
    right: tuple[object, tuple[Any, ...], torch.Tensor],
) -> bool:
    if left[0] != right[0]:
        return False
    left_np, right_np = left[1], right[1]
    if left_np[0] != right_np[0] or left_np[2:] != right_np[2:]:
        return False
    if not np.array_equal(left_np[1], right_np[1]):
        return False
    return torch.equal(left[2], right[2])


@contextmanager
def _read_only_diagnostic_pass(model: LockedPlanner):
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    exact_before = state_dict_sha256(state_before)
    canonical_before = canonical_state_dict_sha256(state_before)
    rng_before = _rng_snapshot()
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(training_mode)
        _restore_rng(rng_before)
    state_after = model.state_dict()
    exact_after = state_dict_sha256(state_after)
    canonical_after = canonical_state_dict_sha256(state_after)
    rng_after = _rng_snapshot()
    if exact_after != exact_before or canonical_after != canonical_before:
        raise RuntimeError("LEARNABILITY_DIAGNOSTIC_MUTATED_MODEL")
    if not _rng_equal(rng_before, rng_after):
        raise RuntimeError("LEARNABILITY_DIAGNOSTIC_RNG_RESTORE_FAILED")


class _TrainingLossObserver:
    """Transparent instrumentation around the existing quality-v0.1 training loop."""

    def __init__(self, rows: list[dict[str, Any]]):
        ordered = sorted(rows, key=lambda row: row["task_id"])
        self.schedule = [
            (epoch, row)
            for epoch in range(EPOCHS)
            for row in ordered
        ]
        self.records: list[dict[str, Any]] = []
        self._components: list[tuple[torch.Tensor, int]] = []
        self._original_cross_entropy = F.cross_entropy
        self._original_clip = torch.nn.utils.clip_grad_norm_
        self._entered = False

    def __enter__(self) -> _TrainingLossObserver:
        if not _TRAINING_OBSERVER_LOCK.acquire(blocking=False):
            raise RuntimeError("LEARNABILITY_TRAINING_OBSERVER_CONCURRENT")
        self._entered = True

        def observed_cross_entropy(*args, **kwargs):
            value = self._original_cross_entropy(*args, **kwargs)
            target = args[1] if len(args) > 1 else kwargs["target"]
            self._components.append((value.detach().clone(), int(target.numel())))
            return value

        def observed_clip(parameters, max_norm, *args, **kwargs):
            norm = self._original_clip(parameters, max_norm, *args, **kwargs)
            self._finish_update(float(norm), float(max_norm))
            return norm

        F.cross_entropy = observed_cross_entropy
        torch.nn.utils.clip_grad_norm_ = observed_clip
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        F.cross_entropy = self._original_cross_entropy
        torch.nn.utils.clip_grad_norm_ = self._original_clip
        if self._entered:
            _TRAINING_OBSERVER_LOCK.release()
        self._entered = False
        if exc_type is None:
            if self._components:
                raise RuntimeError("LEARNABILITY_TRAINING_OBSERVER_INCOMPLETE_UPDATE")
            if len(self.records) != len(self.schedule):
                raise RuntimeError("LEARNABILITY_TRAINING_OBSERVER_UPDATE_COUNT")

    def _finish_update(self, gradient_norm: float, clip_norm: float) -> None:
        update_index = len(self.records)
        if update_index >= len(self.schedule):
            raise RuntimeError("LEARNABILITY_TRAINING_OBSERVER_EXTRA_UPDATE")
        epoch_index, row = self.schedule[update_index]
        gold = row["oracle_work_plan"]
        expected_names = ["operator"]
        if any(step[0] != "END" for step in gold):
            expected_names.append("arg1")
        if any(step[0] in {"UNSTACK", "STACK"} for step in gold):
            expected_names.append("arg2")
        if len(self._components) != len(expected_names):
            raise RuntimeError("LEARNABILITY_TRAINING_LOSS_COMPONENT_COUNT")
        component_values: dict[str, float] = {}
        target_counts = {"operator": 0, "arg1": 0, "arg2": 0}
        total = self._components[0][0].clone()
        for index, (name, (value, target_count)) in enumerate(
            zip(expected_names, self._components, strict=True)
        ):
            component_values[name] = float(value)
            target_counts[name] = target_count
            if index:
                total = total + value
        self.records.append(
            {
                "update_index": update_index,
                "epoch_index": epoch_index,
                "task_id": row["task_id"],
                "operator_loss": component_values["operator"],
                "arg1_pointer_loss": component_values.get("arg1"),
                "arg2_pointer_loss": component_values.get("arg2"),
                "total_loss": float(total),
                "operator_target_count": target_counts["operator"],
                "arg1_target_count": target_counts["arg1"],
                "arg2_target_count": target_counts["arg2"],
                "gradient_norm": gradient_norm,
                "gradient_clip_norm": clip_norm,
                "clipping_occurred": gradient_norm > clip_norm,
            }
        )
        self._components.clear()


def _train_a2_with_loss_trace(
    rows: list[dict[str, Any]], seed: int, output: Path, dataset_hash: str
) -> tuple[LockedPlanner, dict[str, Any], list[dict[str, Any]]]:
    with _TrainingLossObserver(rows) as observer:
        model, checkpoint = _quality_train(rows, VARIANT, seed, output, dataset_hash)
    return model, checkpoint, observer.records


def _target_metadata(row: dict[str, Any], position: int) -> tuple[str, str | None, str | None]:
    step = row["oracle_work_plan"][position]
    return step[0], step[1] if len(step) > 1 else None, step[2] if len(step) > 2 else None


def teacher_forced_task(
    model: LockedPlanner, row: dict[str, Any], *, split: str, seed: int
) -> dict[str, Any]:
    if model.variant != VARIANT:
        raise ValueError("LEARNABILITY_A2_ONLY")
    action, arg1, arg2 = labels(row)
    valid = len(row["oracle_work_plan"])
    logits = model(canonical_task_encoding(row), action, arg1, arg2)
    positions = []
    for position in range(valid):
        gold_operator, gold_arg1, gold_arg2 = _target_metadata(row, position)
        operator_probs = torch.softmax(logits.action[0, position].float(), dim=-1)
        predicted_operator_id = int(operator_probs.argmax())
        predicted_operator = ACTION_NAMES[predicted_operator_id]
        gold_operator_id = ACTIONS[gold_operator]
        has_arg1 = gold_operator != "END"
        has_arg2 = gold_operator in {"UNSTACK", "STACK"}
        predicted_arg1 = None
        predicted_arg2 = None
        arg1_correct = None
        arg2_correct = None
        if has_arg1:
            predicted_arg1_id = int(logits.arg1[0, position, : len(row["blocks"])].argmax())
            predicted_arg1 = row["blocks"][predicted_arg1_id]
            arg1_correct = predicted_arg1 == gold_arg1
        if has_arg2:
            predicted_arg2_id = int(logits.arg2[0, position, : len(row["blocks"])].argmax())
            predicted_arg2 = row["blocks"][predicted_arg2_id]
            arg2_correct = predicted_arg2 == gold_arg2
        operator_correct = predicted_operator == gold_operator
        joint = operator_correct and (arg1_correct is not False) and (arg2_correct is not False)
        probability_gold = float(operator_probs[gold_operator_id])
        positions.append(
            {
                "split": split,
                "task_id": row["task_id"],
                "seed": seed,
                "position_index": position,
                "gold_operator": gold_operator,
                "predicted_operator": predicted_operator,
                "operator_correct": operator_correct,
                "end_correct": operator_correct if gold_operator == "END" else None,
                "probability_gold_operator": probability_gold,
                "probability_end": float(operator_probs[ACTIONS["END"]]),
                "operator_nll": -math.log(max(probability_gold, torch.finfo(torch.float32).tiny)),
                "has_arg1_target": has_arg1,
                "has_arg2_target": has_arg2,
                "gold_arg1": gold_arg1,
                "gold_arg2": gold_arg2,
                "predicted_arg1": predicted_arg1,
                "predicted_arg2": predicted_arg2,
                "arg1_correct": arg1_correct,
                "arg2_correct": arg2_correct,
                "joint_step_correct": joint,
            }
        )
    return {"split": split, "task_id": row["task_id"], "seed": seed, "positions": positions}


def _first_parse_failure_position(raw: list[list[str]], blocks: list[str]) -> int | None:
    if not isinstance(raw, list) or not raw:
        return 0
    for index, step in enumerate(raw):
        try:
            if step == ["END"]:
                continue
            parse_nonterminal_step(step, blocks)
        except PlanParseFailure:
            return index
    if raw[-1] != ["END"]:
        return len(raw)
    if any(step == ["END"] for step in raw[:-1]):
        return next(index for index, step in enumerate(raw[:-1]) if step == ["END"])
    return None


def _content_mismatch(
    gold: list[list[str]], predicted: list[list[str]]
) -> tuple[str | None, int | None]:
    for position in range(max(len(gold), len(predicted))):
        gold_step = gold[position] if position < len(gold) else None
        predicted_step = predicted[position] if position < len(predicted) else None
        if predicted_step is None:
            return "EARLY_END", position
        if gold_step is None:
            return "EXTRA_ACTION_AFTER_GOLD_END", position
        gold_operator = gold_step[0]
        predicted_operator = predicted_step[0]
        if predicted_operator == "END" and gold_operator != "END":
            return "EARLY_END", position
        if gold_operator == "END" and predicted_operator != "END":
            return "EXTRA_ACTION_AFTER_GOLD_END", position
        if predicted_operator != gold_operator:
            return "WRONG_OPERATOR", position
        if gold_operator == "END":
            continue
        if len(predicted_step) < 2 or predicted_step[1] != gold_step[1]:
            return "WRONG_ARG1", position
        if len(gold_step) > 2 and (
            len(predicted_step) < 3 or predicted_step[2] != gold_step[2]
        ):
            return "WRONG_ARG2", position
    return None, None


def classify_first_error(
    *,
    gold: list[list[str]],
    predicted: list[list[str]],
    parser_failure_position: int | None,
    precondition_failure_position: int | None,
    goal_success: bool,
) -> tuple[str, int | None]:
    mismatch, mismatch_position = _content_mismatch(gold, predicted)
    if parser_failure_position is not None and (
        mismatch_position is None or parser_failure_position <= mismatch_position
    ):
        return "PARSE_FAILURE", parser_failure_position
    if precondition_failure_position is not None and (
        mismatch_position is None or precondition_failure_position <= mismatch_position
    ):
        return "PRECONDITION_FAILURE", precondition_failure_position
    if mismatch is not None:
        return mismatch, mismatch_position
    if not goal_success:
        return "GOAL_NOT_ACHIEVED", None
    return "NONE", None


def executable_prefix(row: dict[str, Any], raw: list[list[str]]) -> dict[str, Any]:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    initial_goal_satisfied = goal_satisfied(state, task.goal)
    terminal_end = bool(raw and raw[-1] == ["END"])
    action_steps = raw[:-1] if terminal_end else raw
    prefix_length = 0
    parser_failure_position = None
    precondition_failure_position = None
    for position, step in enumerate(action_steps):
        try:
            action = parse_nonterminal_step(step, list(task.blocks))
        except PlanParseFailure:
            parser_failure_position = position
            break
        try:
            state = apply_action(task.blocks, state, action)
        except ValueError:
            precondition_failure_position = position
            break
        prefix_length += 1
    predicted_count = len(action_steps)
    gold_count = sum(step[0] != "END" for step in row["oracle_work_plan"])
    all_actions_applied = prefix_length == predicted_count
    parser_success = parser_failure_position is None and terminal_end
    full_plan_executable = (
        parser_success
        and precondition_failure_position is None
        and all_actions_applied
        and (predicted_count > 0 or initial_goal_satisfied)
    )
    final_goal_success = full_plan_executable and goal_satisfied(state, task.goal)
    return {
        "executable_prefix_length": prefix_length,
        "predicted_action_count": predicted_count,
        "gold_action_count": gold_count,
        "executable_prefix_fraction_of_predicted": (
            prefix_length / predicted_count if predicted_count else None
        ),
        "executable_prefix_fraction_of_gold": prefix_length / gold_count if gold_count else None,
        "full_plan_executable": full_plan_executable,
        "initial_goal_satisfied": initial_goal_satisfied,
        "final_goal_success": final_goal_success,
        "final_state": [list(fact) for fact in state],
        "parser_failure_position": parser_failure_position,
        "precondition_failure_position": precondition_failure_position,
    }


def free_running_task(
    model: LockedPlanner, row: dict[str, Any], *, split: str, seed: int
) -> dict[str, Any]:
    if model.variant != VARIANT:
        raise ValueError("LEARNABILITY_A2_ONLY")
    trace: list[dict[str, Any]] = []

    def capture(_module, _args, _kwargs, output) -> None:
        position = len(trace)
        probabilities = torch.softmax(output.action[0, position].float(), dim=-1)
        operator_id = int(probabilities.argmax())
        arg1_id = int(output.arg1[0, position, : len(row["blocks"])].argmax())
        arg2_id = int(output.arg2[0, position, : len(row["blocks"])].argmax())
        operator = ACTION_NAMES[operator_id]
        trace.append(
            {
                "position_index": position,
                "predicted_operator": operator,
                "predicted_arg1": row["blocks"][arg1_id] if operator != "END" else None,
                "predicted_arg2": (
                    row["blocks"][arg2_id] if operator in {"UNSTACK", "STACK"} else None
                ),
                "probability_end": float(probabilities[ACTIONS["END"]]),
            }
        )

    handle = model.register_forward_hook(capture, with_kwargs=True)
    planner = A2Planner(model)
    inference_row = {key: value for key, value in row.items() if key != "oracle_work_plan"}
    raw: list[list[str]] = []
    generation_failure = None
    try:
        raw = planner.plan(inference_row)
    except PlannerGenerationFailure as error:
        raw = error.partial_raw_output
        generation_failure = error.code
    finally:
        handle.remove()
    parser_failure_position = _first_parse_failure_position(raw, row["blocks"])
    parser_success = parser_failure_position is None and generation_failure is None
    if parser_success:
        try:
            parse_work_plan(raw, row["blocks"])
        except PlanParseFailure:
            parser_success = False
            parser_failure_position = _first_parse_failure_position(raw, row["blocks"])
    prefix = executable_prefix(row, raw)
    first_error, first_error_position = classify_first_error(
        gold=row["oracle_work_plan"],
        predicted=raw,
        parser_failure_position=parser_failure_position,
        precondition_failure_position=prefix["precondition_failure_position"],
        goal_success=prefix["final_goal_success"],
    )
    if generation_failure is not None:
        failure_code = generation_failure
    elif parser_failure_position is not None:
        failure_code = "PLAN_PARSE_ERROR"
    elif prefix["precondition_failure_position"] is not None:
        failure_code = "EXECUTOR_PRECONDITION_FAILED"
    elif not prefix["final_goal_success"]:
        failure_code = "GOAL_NOT_ACHIEVED"
    else:
        failure_code = None
    gold = row["oracle_work_plan"]
    predicted_positions = []
    for item in trace:
        position = item["position_index"]
        gold_step = gold[position] if position < len(gold) else None
        gold_operator = gold_step[0] if gold_step is not None else None
        gold_arg1 = gold_step[1] if gold_step is not None and len(gold_step) > 1 else None
        gold_arg2 = gold_step[2] if gold_step is not None and len(gold_step) > 2 else None
        operator_correct = item["predicted_operator"] == gold_operator
        arg1_correct = (
            item["predicted_arg1"] == gold_arg1 if gold_arg1 is not None else None
        )
        arg2_correct = (
            item["predicted_arg2"] == gold_arg2 if gold_arg2 is not None else None
        )
        predicted_positions.append(
            {
                **item,
                "gold_operator": gold_operator,
                "gold_arg1": gold_arg1,
                "gold_arg2": gold_arg2,
                "operator_correct": operator_correct,
                "arg1_correct": arg1_correct,
                "arg2_correct": arg2_correct,
                "joint_step_correct": (
                    operator_correct and arg1_correct is not False and arg2_correct is not False
                ),
            }
        )
    predicted_action_count = prefix["predicted_action_count"]
    end_only = raw == [["END"]]
    return {
        "split": split,
        "task_id": row["task_id"],
        "seed": seed,
        "predicted_plan": raw,
        "predicted_history_positions": predicted_positions,
        "predicted_pre_end_action_count": predicted_action_count,
        "first_predicted_operator": trace[0]["predicted_operator"] if trace else None,
        "first_step_end_probability": trace[0]["probability_end"] if trace else None,
        "predicted_plan_length": predicted_action_count,
        "end_only": end_only,
        "parser_success": parser_success,
        "exact_plan_match": raw == gold,
        "first_mismatch_position": first_error_position,
        "first_mismatch_type": first_error,
        **{key: value for key, value in prefix.items() if key not in {
            "parser_failure_position", "precondition_failure_position"
        }},
        "failure_code": failure_code,
        "generation_failure": generation_failure,
        "model_forward_count": planner.model_forward_count,
        "planner_call_count": planner.calls,
        "replanning_count": 0,
    }


def _teacher_summary(position_rows: list[dict[str, Any]]) -> dict[str, Any]:
    operator_count = len(position_rows)
    operator_correct = sum(row["operator_correct"] for row in position_rows)
    arg1_rows = [row for row in position_rows if row["has_arg1_target"]]
    arg2_rows = [row for row in position_rows if row["has_arg2_target"]]
    end_rows = [row for row in position_rows if row["gold_operator"] == "END"]
    return {
        "position_count": operator_count,
        "operator_correct_count": operator_correct,
        "operator_accuracy": _rate(operator_correct, operator_count),
        "joint_step_correct_count": sum(row["joint_step_correct"] for row in position_rows),
        "joint_step_accuracy": _rate(
            sum(row["joint_step_correct"] for row in position_rows), operator_count
        ),
        "arg1_target_count": len(arg1_rows),
        "arg1_correct_count": sum(row["arg1_correct"] for row in arg1_rows),
        "arg1_accuracy": _rate(sum(row["arg1_correct"] for row in arg1_rows), len(arg1_rows)),
        "arg2_target_count": len(arg2_rows),
        "arg2_correct_count": sum(row["arg2_correct"] for row in arg2_rows),
        "arg2_accuracy": _rate(sum(row["arg2_correct"] for row in arg2_rows), len(arg2_rows)),
        "end_target_count": len(end_rows),
        "end_correct_count": sum(row["end_correct"] for row in end_rows),
        "end_accuracy": _rate(sum(row["end_correct"] for row in end_rows), len(end_rows)),
        "mean_probability_gold_operator": _mean(
            [row["probability_gold_operator"] for row in position_rows]
        ),
        "mean_probability_end": _mean([row["probability_end"] for row in position_rows]),
        "mean_operator_nll": _mean([row["operator_nll"] for row in position_rows]),
    }


def aggregate_teacher_forced(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [position for task in tasks for position in task["positions"]]

    def grouped(key) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(key(row))].append(row)
        return {name: _teacher_summary(groups[name]) for name in sorted(groups)}

    return {
        "overall": _teacher_summary(rows),
        "by_split": grouped(lambda row: row["split"]),
        "by_seed": grouped(lambda row: row["seed"]),
        "by_position_index": grouped(lambda row: row["position_index"]),
        "by_operator": grouped(lambda row: row["gold_operator"]),
        "non_end": _teacher_summary([row for row in rows if row["gold_operator"] != "END"]),
        "end": _teacher_summary([row for row in rows if row["gold_operator"] == "END"]),
    }


def _free_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [position for task in tasks for position in task["predicted_history_positions"]]
    error_distribution = Counter(task["first_mismatch_type"] for task in tasks)
    return {
        "task_count": len(tasks),
        "predicted_position_count": len(positions),
        "operator_accuracy": _rate(
            sum(position["operator_correct"] for position in positions), len(positions)
        ),
        "joint_step_accuracy": _rate(
            sum(position["joint_step_correct"] for position in positions), len(positions)
        ),
        "mean_end_probability": _mean(
            [position["probability_end"] for position in positions]
        ),
        "nonempty_plan_rate": _rate(
            sum(task["predicted_action_count"] > 0 for task in tasks), len(tasks)
        ),
        "exact_plan_rate": _rate(sum(task["exact_plan_match"] for task in tasks), len(tasks)),
        "mean_executable_prefix_length": _mean(
            [task["executable_prefix_length"] for task in tasks]
        ),
        "mean_executable_prefix_fraction_of_predicted": _mean(
            [task["executable_prefix_fraction_of_predicted"] for task in tasks]
        ),
        "mean_executable_prefix_fraction_of_gold": _mean(
            [task["executable_prefix_fraction_of_gold"] for task in tasks]
        ),
        "full_plan_executable_rate": _rate(
            sum(task["full_plan_executable"] for task in tasks), len(tasks)
        ),
        "goal_success_rate": _rate(sum(task["final_goal_success"] for task in tasks), len(tasks)),
        "mean_first_error_position": _mean(
            [task["first_mismatch_position"] for task in tasks]
        ),
        "first_error_distribution": {
            category: error_distribution.get(category, 0) for category in FIRST_ERROR_CATEGORIES
        },
    }


def _gold_history_projection(
    teacher_task: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    raw: list[list[str]] = []
    positions = []
    for position in teacher_task["positions"]:
        operator = position["predicted_operator"]
        step = [operator]
        if operator != "END":
            step.append(position["predicted_arg1"])
        if operator in {"UNSTACK", "STACK"}:
            step.append(position["predicted_arg2"])
        raw.append(step)
        positions.append(
            {
                "operator_correct": position["operator_correct"],
                "joint_step_correct": position["joint_step_correct"],
                "probability_end": position["probability_end"],
            }
        )
        if operator == "END":
            break
    prefix = executable_prefix(row, raw)
    parser_failure = _first_parse_failure_position(raw, row["blocks"])
    category, mismatch_position = classify_first_error(
        gold=row["oracle_work_plan"],
        predicted=raw,
        parser_failure_position=parser_failure,
        precondition_failure_position=prefix["precondition_failure_position"],
        goal_success=prefix["final_goal_success"],
    )
    return {
        "predicted_history_positions": positions,
        "predicted_action_count": prefix["predicted_action_count"],
        "exact_plan_match": raw == row["oracle_work_plan"],
        "executable_prefix_length": prefix["executable_prefix_length"],
        "executable_prefix_fraction_of_predicted": prefix[
            "executable_prefix_fraction_of_predicted"
        ],
        "executable_prefix_fraction_of_gold": prefix[
            "executable_prefix_fraction_of_gold"
        ],
        "full_plan_executable": prefix["full_plan_executable"],
        "final_goal_success": prefix["final_goal_success"],
        "first_mismatch_position": mismatch_position,
        "first_mismatch_type": category,
    }


def aggregate_history_modes(
    teacher_tasks: list[dict[str, Any]], free_tasks: list[dict[str, Any]]
) -> dict[str, Any]:
    _, train_rows = _dataset_context()
    train_by_id = {row["task_id"]: row for row in train_rows}
    gold_history = [
        _gold_history_projection(task, train_by_id[task["task_id"]])
        for task in teacher_tasks
    ]
    return {
        "gold_history": _free_summary(gold_history),
        "predicted_history": _free_summary(free_tasks),
    }


def _loss_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "update_count": len(rows),
        "mean_operator_loss": _mean([row["operator_loss"] for row in rows]),
        "mean_arg1_pointer_loss": _mean([row["arg1_pointer_loss"] for row in rows]),
        "mean_arg2_pointer_loss": _mean([row["arg2_pointer_loss"] for row in rows]),
        "mean_total_loss": _mean([row["total_loss"] for row in rows]),
        "operator_target_count": sum(row["operator_target_count"] for row in rows),
        "arg1_target_count": sum(row["arg1_target_count"] for row in rows),
        "arg2_target_count": sum(row["arg2_target_count"] for row in rows),
        "mean_gradient_norm": _mean([row["gradient_norm"] for row in rows]),
        "clipping_occurrence_count": sum(row["clipping_occurred"] for row in rows),
    }


def aggregate_loss_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[str(row["seed"])].append(row)
    return {
        "overall": _loss_summary(rows),
        "by_seed": {seed: _loss_summary(by_seed[seed]) for seed in sorted(by_seed)},
    }


def aggregate_free_running(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_split[task["split"]].append(task)
        by_seed[str(task["seed"])].append(task)
    return {
        "overall": _free_summary(tasks),
        "by_split": {key: _free_summary(by_split[key]) for key in sorted(by_split)},
        "by_seed": {key: _free_summary(by_seed[key]) for key in sorted(by_seed)},
    }


def _pipeline_replay(model: LockedPlanner, rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for row in rows:
        planner = A2Planner(model)
        raw = []
        failure = None
        try:
            inference_row = {
                key: value for key, value in row.items() if key != "oracle_work_plan"
            }
            raw = planner.plan(inference_row)
        except PlannerGenerationFailure as error:
            raw = error.partial_raw_output
            failure = error.code
        results.append(
            {
                "task_id": row["task_id"],
                "raw_output": raw,
                "failure": failure,
                "model_forward_count": planner.model_forward_count,
            }
        )
    return {"results": results, "hash": sha256(results)}


def _checkpoint_record(seed: int, run_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "checkpoint_manifest_path": str(
            (run_dir / "checkpoint-manifest.json").relative_to(run_dir.parents[2])
        ),
        "trained_checkpoint_path": str((run_dir / "trained.pt").relative_to(run_dir.parents[2])),
        "optimizer_state_path": str(
            (run_dir / "optimizer-state.pt").relative_to(run_dir.parents[2])
        ),
        "canonical_trained_state_dict_sha256": checkpoint[
            "canonical_trained_state_dict_sha256"
        ],
        "canonical_optimizer_state_sha256": checkpoint[
            "canonical_optimizer_state_sha256"
        ],
    }


def _artifact_identity(payload: dict[str, Any]) -> str:
    value = copy.deepcopy(payload)
    value.pop("canonical_identity", None)
    return sha256(
        {
            "schema_version": "toy-learnability-diagnostic-hash/1.0",
            "payload": value,
        }
    )


def _dataset_context() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = generate(17)
    train_rows = sorted(dataset["train"], key=lambda row: row["task_id"])
    if any(row["task_id"] in FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS for row in train_rows):
        raise ValueError("LEARNABILITY_TRAIN_HELDOUT_OVERLAP")
    return dataset, train_rows


def run(
    output: Path,
    *,
    implementation_commit: str,
    seeds: tuple[int, ...] = SEEDS,
    task_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    runtime_fingerprint = configure_canonical_cpu_runtime()
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ValueError("LEARNABILITY_IMPLEMENTATION_COMMIT_INVALID")
    if output.exists() and any(output.iterdir()):
        raise ValueError("LEARNABILITY_OUTPUT_NOT_CLEAN")
    output.mkdir(parents=True, exist_ok=True)
    dataset, train_rows = _dataset_context()
    known_task_ids = tuple(row["task_id"] for row in train_rows)
    requested_ids = task_ids or known_task_ids
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("LEARNABILITY_TASK_SELECTION_INVALID")
    if any(task_id not in known_task_ids for task_id in requested_ids):
        raise ValueError("LEARNABILITY_NONTRAIN_TASK_FORBIDDEN")
    selected_ids = tuple(task_id for task_id in known_task_ids if task_id in requested_ids)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed not in SEEDS for seed in seeds):
        raise ValueError("LEARNABILITY_SEED_SELECTION_INVALID")
    selected_seeds = tuple(seed for seed in SEEDS if seed in seeds)
    selected_rows = [row for row in train_rows if row["task_id"] in selected_ids]
    selected_rows.sort(key=lambda row: row["task_id"])
    teacher_tasks: list[dict[str, Any]] = []
    free_tasks: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    invariance: list[dict[str, Any]] = []
    training_root = output / "training-runs"
    for seed in selected_seeds:
        run_dir = training_root / "A2" / f"seed-{seed}"
        model, checkpoint, observed_losses = _train_a2_with_loss_trace(
            train_rows, seed, run_dir, dataset["dataset_hash"]
        )
        for row in observed_losses:
            loss_rows.append({"seed": seed, **row})
        checkpoint_record = _checkpoint_record(seed, run_dir, checkpoint)
        checkpoints.append(checkpoint_record)
        checkpoint_bytes_before = (run_dir / "trained.pt").read_bytes()
        optimizer_bytes_before = (run_dir / "optimizer-state.pt").read_bytes()
        optimizer_state_before = torch.load(
            run_dir / "optimizer-state.pt", map_location="cpu", weights_only=True
        )
        checkpoint_exact_before = state_dict_sha256(model.state_dict())
        checkpoint_canonical_before = canonical_state_dict_sha256(model.state_dict())
        pipeline_before = _pipeline_replay(model, selected_rows)
        rng_before = _rng_snapshot()
        with _read_only_diagnostic_pass(model):
            for row in selected_rows:
                teacher_tasks.append(teacher_forced_task(model, row, split="train", seed=seed))
                free_tasks.append(free_running_task(model, row, split="train", seed=seed))
        rng_after = _rng_snapshot()
        pipeline_after = _pipeline_replay(model, selected_rows)
        checkpoint_bytes_after = (run_dir / "trained.pt").read_bytes()
        optimizer_bytes_after = (run_dir / "optimizer-state.pt").read_bytes()
        optimizer_state_after = torch.load(
            run_dir / "optimizer-state.pt", map_location="cpu", weights_only=True
        )
        checkpoint_exact_after = state_dict_sha256(model.state_dict())
        checkpoint_canonical_after = canonical_state_dict_sha256(model.state_dict())
        invariance.append(
            {
                "seed": seed,
                "checkpoint_canonical_before": checkpoint_canonical_before,
                "checkpoint_canonical_after": checkpoint_canonical_after,
                "optimizer_canonical_before": canonical_torch_object_sha256(
                    optimizer_state_before
                ),
                "optimizer_canonical_after": canonical_torch_object_sha256(
                    optimizer_state_after
                ),
                "pipeline_replay_hash_before": pipeline_before["hash"],
                "pipeline_replay_hash_after": pipeline_after["hash"],
                "model_parameters_unchanged": checkpoint_exact_before == checkpoint_exact_after,
                "checkpoint_bytes_unchanged": (
                    checkpoint_bytes_before == checkpoint_bytes_after
                ),
                "optimizer_state_unchanged": optimizer_bytes_before == optimizer_bytes_after,
                "pipeline_replay_unchanged": pipeline_before == pipeline_after,
                "rng_state_restored": _rng_equal(rng_before, rng_after),
            }
        )
    teacher_tasks.sort(key=lambda item: (item["seed"], item["split"], item["task_id"]))
    free_tasks.sort(key=lambda item: (item["seed"], item["split"], item["task_id"]))
    loss_rows.sort(key=lambda item: (item["seed"], item["update_index"]))
    checkpoints.sort(key=lambda item: item["seed"])
    invariance.sort(key=lambda item: item["seed"])
    complete = selected_seeds == SEEDS and selected_ids == known_task_ids
    payload: dict[str, Any] = {
        "diagnostic_version": VERSION,
        "status": STATUS,
        "diagnostic_complete": complete,
        "variant": VARIANT,
        "implementation_commit": implementation_commit,
        **diagnostic_source_identity(),
        "requirements_lock_sha256": _file_hash(ROOT / "requirements.lock"),
        "runtime_versions": _runtime_versions(),
        "canonical_cpu_runtime": runtime_fingerprint,
        "dataset_hash": dataset["dataset_hash"],
        "evaluated_splits": ["train"],
        "evaluated_task_ids": list(selected_ids),
        "excluded_splits": [
            {
                "split": "validation",
                "reason": "FROZEN_QUALITY_V0_1_HELDOUT",
                "task_ids": list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS),
            }
        ],
        "seeds": list(selected_seeds),
        "training_config": TRAINING_CONFIG,
        "heldout_accessed": False,
        "training_policy_changed": False,
        "gate_decision": None,
        "learnability_thresholds": None,
        "checkpoints": checkpoints,
        "per_update_loss_breakdown": loss_rows,
        "teacher_forced": teacher_tasks,
        "free_running": free_tasks,
        "aggregates": {
            "teacher_forced": aggregate_teacher_forced(teacher_tasks),
            "free_running": aggregate_free_running(free_tasks),
            "history_mode_comparison": aggregate_history_modes(teacher_tasks, free_tasks),
            "loss_breakdown": aggregate_loss_breakdown(loss_rows),
        },
        "first_error_distribution": aggregate_free_running(free_tasks)["overall"][
            "first_error_distribution"
        ],
        "diagnostic_invariance": invariance,
    }
    payload["canonical_identity"] = _artifact_identity(payload)
    validate_payload(payload, root=output)
    (output / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (output / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"LEARNABILITY_NONFINITE_VALUE:{path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, f"{path}[{index}]")


def _assert_pointer_semantics(position: dict[str, Any]) -> None:
    has_arg1 = position["has_arg1_target"]
    has_arg2 = position["has_arg2_target"]
    arg1_fields = ("gold_arg1", "predicted_arg1", "arg1_correct")
    arg2_fields = ("gold_arg2", "predicted_arg2", "arg2_correct")
    if has_arg1 != all(position[field] is not None for field in arg1_fields):
        raise ValueError("LEARNABILITY_ARG1_TARGET_SEMANTICS")
    if has_arg2 != all(position[field] is not None for field in arg2_fields):
        raise ValueError("LEARNABILITY_ARG2_TARGET_SEMANTICS")
    if position["gold_operator"] == "END" and any(
        position[field] is not None for field in (*arg1_fields, *arg2_fields)
    ):
        raise ValueError("LEARNABILITY_END_POINTER_METRIC_NON_NULL")


def validate_payload(payload: dict[str, Any], *, root: Path | None = None) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
    except SchemaError as exc:
        raise ValueError("LEARNABILITY_SCHEMA_INVALID") from exc
    except ValidationError as exc:
        raise ValueError("LEARNABILITY_SCHEMA_VALIDATION_FAILED") from exc
    _assert_finite_json(payload)
    if (
        payload["diagnostic_version"] != VERSION
        or payload["status"] != STATUS
        or payload["variant"] != VARIANT
        or payload["gate_decision"] is not None
    ):
        raise ValueError("LEARNABILITY_IDENTITY_MISMATCH")
    if payload["learnability_thresholds"] is not None:
        raise ValueError("LEARNABILITY_THRESHOLDS_FORBIDDEN")
    if payload["training_policy_changed"] or payload["heldout_accessed"]:
        raise ValueError("LEARNABILITY_SCOPE_FLAG_MISMATCH")
    if payload["training_config"] != TRAINING_CONFIG:
        raise ValueError("LEARNABILITY_TRAINING_CONFIG_CHANGED")
    if payload["diagnostic_source_files"] != diagnostic_source_identity()[
        "diagnostic_source_files"
    ] or payload["diagnostic_source_sha256"] != diagnostic_source_identity()[
        "diagnostic_source_sha256"
    ]:
        raise ValueError("LEARNABILITY_SOURCE_IDENTITY_STALE")
    if payload["requirements_lock_sha256"] != _file_hash(ROOT / "requirements.lock"):
        raise ValueError("LEARNABILITY_REQUIREMENTS_HASH_MISMATCH")
    if payload["canonical_cpu_runtime"] != configure_canonical_cpu_runtime():
        raise ValueError("LEARNABILITY_RUNTIME_PROFILE_MISMATCH")
    if payload["runtime_versions"] != _runtime_versions():
        raise ValueError("LEARNABILITY_RUNTIME_VERSIONS_MISMATCH")
    if payload["evaluated_splits"] != ["train"]:
        raise ValueError("LEARNABILITY_SPLIT_POLICY_MISMATCH")
    expected_excluded = [{
        "split": "validation",
        "reason": "FROZEN_QUALITY_V0_1_HELDOUT",
        "task_ids": list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS),
    }]
    if payload["excluded_splits"] != expected_excluded:
        raise ValueError("LEARNABILITY_EXCLUDED_SPLIT_MISMATCH")
    if payload["seeds"] != [seed for seed in SEEDS if seed in payload["seeds"]]:
        raise ValueError("LEARNABILITY_SEED_ORDER_MISMATCH")
    dataset, train_rows = _dataset_context()
    train_by_id = {row["task_id"]: row for row in train_rows}
    if payload["dataset_hash"] != dataset["dataset_hash"]:
        raise ValueError("LEARNABILITY_DATASET_HASH_MISMATCH")
    evaluated_ids = payload["evaluated_task_ids"]
    canonical_evaluated_ids = [task_id for task_id in train_by_id if task_id in evaluated_ids]
    if evaluated_ids != canonical_evaluated_ids or not evaluated_ids:
        raise ValueError("LEARNABILITY_TASK_ORDER_MISMATCH")
    expected_complete = payload["seeds"] == list(SEEDS) and evaluated_ids == list(train_by_id)
    if payload["diagnostic_complete"] != expected_complete:
        raise ValueError("LEARNABILITY_COMPLETE_FLAG_MISMATCH")
    if any(task_id not in train_by_id for task_id in evaluated_ids):
        raise ValueError("LEARNABILITY_HELDOUT_OR_UNKNOWN_TASK")
    if set(evaluated_ids) & set(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS):
        raise ValueError("LEARNABILITY_HELDOUT_TASK_PRESENT")
    expected_keys = {
        (seed, task_id) for seed in payload["seeds"] for task_id in evaluated_ids
    }
    teacher_keys = [(task["seed"], task["task_id"]) for task in payload["teacher_forced"]]
    free_keys = [(task["seed"], task["task_id"]) for task in payload["free_running"]]
    if len(teacher_keys) != len(set(teacher_keys)) or set(teacher_keys) != expected_keys:
        raise ValueError("LEARNABILITY_TEACHER_COVERAGE_MISMATCH")
    if len(free_keys) != len(set(free_keys)) or set(free_keys) != expected_keys:
        raise ValueError("LEARNABILITY_FREE_COVERAGE_MISMATCH")
    for task in payload["teacher_forced"]:
        row = train_by_id[task["task_id"]]
        positions = task["positions"]
        if [position["position_index"] for position in positions] != list(
            range(len(row["oracle_work_plan"]))
        ):
            raise ValueError("LEARNABILITY_GOLD_POSITION_COVERAGE_MISMATCH")
        for position in positions:
            gold_operator, gold_arg1, gold_arg2 = _target_metadata(
                row, position["position_index"]
            )
            if (
                position["split"] != "train"
                or position["task_id"] != task["task_id"]
                or position["seed"] != task["seed"]
                or position["gold_operator"] != gold_operator
                or position["gold_arg1"] != gold_arg1
                or position["gold_arg2"] != gold_arg2
            ):
                raise ValueError("LEARNABILITY_GOLD_TARGET_MISMATCH")
            _assert_pointer_semantics(position)
    for task in payload["free_running"]:
        row = train_by_id[task["task_id"]]
        if task["end_only"] != (task["predicted_plan"] == [["END"]]):
            raise ValueError("LEARNABILITY_END_ONLY_CONTRADICTION")
        if task["predicted_plan_length"] != task["predicted_action_count"]:
            raise ValueError("LEARNABILITY_PLAN_LENGTH_CONTRADICTION")
        if task["executable_prefix_length"] > task["predicted_action_count"]:
            raise ValueError("LEARNABILITY_EXECUTABLE_PREFIX_OVERFLOW")
        positions = task["predicted_history_positions"]
        if [position["position_index"] for position in positions] != list(range(len(positions))):
            raise ValueError("LEARNABILITY_PREDICTED_POSITION_ORDER")
        if len(positions) != task["model_forward_count"]:
            raise ValueError("LEARNABILITY_MODEL_FORWARD_COUNT_MISMATCH")
        if task["planner_call_count"] != 1 or task["replanning_count"] != 0:
            raise ValueError("LEARNABILITY_PLANNER_CALL_POLICY_MISMATCH")
        first_operator = positions[0]["predicted_operator"] if positions else None
        first_end_probability = positions[0]["probability_end"] if positions else None
        if (
            task["first_predicted_operator"] != first_operator
            or task["first_step_end_probability"] != first_end_probability
        ):
            raise ValueError("LEARNABILITY_FIRST_POSITION_MISMATCH")
        for position_row in positions:
            index = position_row["position_index"]
            gold_step = (
                row["oracle_work_plan"][index]
                if index < len(row["oracle_work_plan"])
                else None
            )
            expected_gold_operator = gold_step[0] if gold_step else None
            expected_gold_arg1 = gold_step[1] if gold_step and len(gold_step) > 1 else None
            expected_gold_arg2 = gold_step[2] if gold_step and len(gold_step) > 2 else None
            actual_gold = (
                position_row["gold_operator"],
                position_row["gold_arg1"],
                position_row["gold_arg2"],
            )
            expected_gold = (expected_gold_operator, expected_gold_arg1, expected_gold_arg2)
            if actual_gold != expected_gold:
                raise ValueError("LEARNABILITY_PREDICTED_HISTORY_GOLD_MISMATCH")
        recalculated = executable_prefix(row, task["predicted_plan"])
        for field in (
            "executable_prefix_length",
            "predicted_action_count",
            "gold_action_count",
            "executable_prefix_fraction_of_predicted",
            "executable_prefix_fraction_of_gold",
            "full_plan_executable",
            "initial_goal_satisfied",
            "final_goal_success",
            "final_state",
        ):
            if task[field] != recalculated[field]:
                raise ValueError("LEARNABILITY_EXECUTABLE_PREFIX_MISMATCH")
        parser_failure = _first_parse_failure_position(task["predicted_plan"], row["blocks"])
        category, position = classify_first_error(
            gold=row["oracle_work_plan"],
            predicted=task["predicted_plan"],
            parser_failure_position=parser_failure,
            precondition_failure_position=recalculated["precondition_failure_position"],
            goal_success=recalculated["final_goal_success"],
        )
        if task["first_mismatch_type"] != category or task["first_mismatch_position"] != position:
            raise ValueError("LEARNABILITY_FIRST_ERROR_MISMATCH")
        if task["final_goal_success"]:
            canonical_state = tuple(tuple(fact) for fact in task["final_state"])
            if not goal_satisfied(canonical_state, task_from_row(row).goal):
                raise ValueError("LEARNABILITY_FALSE_GOAL_SUCCESS")
    if payload["aggregates"]["teacher_forced"] != aggregate_teacher_forced(
        payload["teacher_forced"]
    ):
        raise ValueError("LEARNABILITY_TEACHER_AGGREGATE_MISMATCH")
    expected_free = aggregate_free_running(payload["free_running"])
    if payload["aggregates"]["free_running"] != expected_free:
        raise ValueError("LEARNABILITY_FREE_AGGREGATE_MISMATCH")
    if payload["aggregates"]["history_mode_comparison"] != aggregate_history_modes(
        payload["teacher_forced"], payload["free_running"]
    ):
        raise ValueError("LEARNABILITY_HISTORY_COMPARISON_MISMATCH")
    if payload["aggregates"]["loss_breakdown"] != aggregate_loss_breakdown(
        payload["per_update_loss_breakdown"]
    ):
        raise ValueError("LEARNABILITY_LOSS_AGGREGATE_MISMATCH")
    if payload["first_error_distribution"] != expected_free["overall"][
        "first_error_distribution"
    ]:
        raise ValueError("LEARNABILITY_FIRST_ERROR_DISTRIBUTION_MISMATCH")
    if payload["canonical_identity"] != _artifact_identity(payload):
        raise ValueError("LEARNABILITY_CANONICAL_IDENTITY_MISMATCH")
    expected_loss_keys = {
        (seed, update) for seed in payload["seeds"] for update in range(UPDATES_PER_RUN)
    }
    loss_keys = [
        (row["seed"], row["update_index"]) for row in payload["per_update_loss_breakdown"]
    ]
    if len(loss_keys) != len(set(loss_keys)) or set(loss_keys) != expected_loss_keys:
        raise ValueError("LEARNABILITY_LOSS_BREAKDOWN_COVERAGE")
    for row in payload["per_update_loss_breakdown"]:
        components = [row["operator_loss"]]
        if row["arg1_pointer_loss"] is not None:
            components.append(row["arg1_pointer_loss"])
        if row["arg2_pointer_loss"] is not None:
            components.append(row["arg2_pointer_loss"])
        if not math.isclose(sum(components), row["total_loss"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("LEARNABILITY_TOTAL_LOSS_BREAKDOWN_MISMATCH")
    checkpoint_seeds = [record["seed"] for record in payload["checkpoints"]]
    invariance_seeds = [record["seed"] for record in payload["diagnostic_invariance"]]
    if checkpoint_seeds != payload["seeds"] or invariance_seeds != payload["seeds"]:
        raise ValueError("LEARNABILITY_SEED_ARTIFACT_COVERAGE")
    expected_schedule = [(epoch, task_id) for epoch in range(EPOCHS) for task_id in train_by_id]
    for seed in payload["seeds"]:
        actual_schedule = [
            (row["epoch_index"], row["task_id"])
            for row in payload["per_update_loss_breakdown"]
            if row["seed"] == seed
        ]
        if actual_schedule != expected_schedule:
            raise ValueError("LEARNABILITY_LOSS_SCHEDULE_MISMATCH")
    if any(
        not record["model_parameters_unchanged"]
        or not record["checkpoint_bytes_unchanged"]
        or not record["optimizer_state_unchanged"]
        or not record["pipeline_replay_unchanged"]
        or not record["rng_state_restored"]
        or record["checkpoint_canonical_before"] != record["checkpoint_canonical_after"]
        or record["optimizer_canonical_before"] != record["optimizer_canonical_after"]
        or record["pipeline_replay_hash_before"] != record["pipeline_replay_hash_after"]
        for record in payload["diagnostic_invariance"]
    ):
        raise ValueError("LEARNABILITY_DIAGNOSTIC_INVARIANCE_FAILED")
    if root is not None:
        for checkpoint in payload["checkpoints"]:
            seed = checkpoint["seed"]
            run_dir = root / "training-runs" / "A2" / f"seed-{seed}"
            manifest = json.loads((run_dir / "checkpoint-manifest.json").read_text())
            state = torch.load(run_dir / "trained.pt", map_location="cpu", weights_only=True)
            optimizer = torch.load(
                run_dir / "optimizer-state.pt", map_location="cpu", weights_only=True
            )
            expected_paths = {
                "checkpoint_manifest_path": (
                    f"training-runs/A2/seed-{seed}/checkpoint-manifest.json"
                ),
                "trained_checkpoint_path": f"training-runs/A2/seed-{seed}/trained.pt",
                "optimizer_state_path": f"training-runs/A2/seed-{seed}/optimizer-state.pt",
            }
            if any(checkpoint[key] != value for key, value in expected_paths.items()):
                raise ValueError("LEARNABILITY_CHECKPOINT_PATH_MISMATCH")
            if manifest["trained_state_dict_sha256"] != state_dict_sha256(state):
                raise ValueError("LEARNABILITY_CHECKPOINT_HASH_MISMATCH")
            if checkpoint["canonical_trained_state_dict_sha256"] != canonical_state_dict_sha256(
                state
            ):
                raise ValueError("LEARNABILITY_CHECKPOINT_CANONICAL_HASH_MISMATCH")
            if checkpoint["canonical_optimizer_state_sha256"] != canonical_torch_object_sha256(
                optimizer
            ):
                raise ValueError("LEARNABILITY_OPTIMIZER_CANONICAL_HASH_MISMATCH")
            if manifest["optimizer_state_sha256"] != exact_torch_object_sha256(optimizer):
                raise ValueError("LEARNABILITY_OPTIMIZER_EXACT_HASH_MISMATCH")
            if (
                manifest["canonical_optimizer_state_sha256"]
                != checkpoint["canonical_optimizer_state_sha256"]
            ):
                raise ValueError("LEARNABILITY_OPTIMIZER_MANIFEST_HASH_MISMATCH")
            if (
                manifest["canonical_trained_state_dict_sha256"]
                != checkpoint["canonical_trained_state_dict_sha256"]
            ):
                raise ValueError("LEARNABILITY_CHECKPOINT_MANIFEST_HASH_MISMATCH")
            if (
                manifest["seed"] != seed
                or manifest["variant_identity"]["implementation_variant"] != "A2"
            ):
                raise ValueError("LEARNABILITY_CHECKPOINT_IDENTITY_MISMATCH")
            if manifest["epochs"] != EPOCHS or manifest["updates"] != UPDATES_PER_RUN:
                raise ValueError("LEARNABILITY_CHECKPOINT_TRAINING_CONFIG_MISMATCH")


def validate_diagnostic(root: Path) -> dict[str, Any]:
    payload = json.loads((root / OUTPUT_JSON).read_text(encoding="utf-8"))
    validate_payload(payload, root=root)
    if (root / OUTPUT_MARKDOWN).read_text(encoding="utf-8") != render_markdown(payload):
        raise ValueError("LEARNABILITY_MARKDOWN_MISMATCH")
    return {
        "valid": True,
        "diagnostic_complete": payload["diagnostic_complete"],
        "task_count": len(payload["free_running"]),
        "canonical_identity": payload["canonical_identity"],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    teacher = payload["aggregates"]["teacher_forced"]["overall"]
    free = payload["aggregates"]["free_running"]["overall"]
    history = payload["aggregates"]["history_mode_comparison"]
    loss = payload["aggregates"]["loss_breakdown"]["overall"]
    lines = [
        "# A2 END-only learnability diagnostic",
        "",
        "> Development-only diagnostic. No gate threshold or causal decision is encoded.",
        "",
        f"- Version: `{payload['diagnostic_version']}`",
        f"- Status: `{payload['status']}`",
        f"- Implementation commit: `{payload['implementation_commit']}`",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        f"- Evaluated splits: `{', '.join(payload['evaluated_splits'])}`",
        f"- Seeds: `{', '.join(map(str, payload['seeds']))}`",
        f"- Held-out accessed: `{str(payload['heldout_accessed']).lower()}`",
        f"- Training policy changed: `{str(payload['training_policy_changed']).lower()}`",
        f"- Gate decision: `{payload['gate_decision']}`",
        "",
        "## Split boundary",
        "",
        "The repository has no separate non-held-out development validation split. The existing",
        "`validation` rows are the frozen quality-v0.1 held-out tasks and are excluded from this",
        "diagnostic. Metrics therefore cover the existing `train` split only.",
        "",
        "## Teacher-forced summary",
        "",
        f"- Positions: `{teacher['position_count']}`",
        f"- Operator accuracy: `{teacher['operator_accuracy']}`",
        f"- Joint-step accuracy: `{teacher['joint_step_accuracy']}`",
        (
            f"- Arg1 accuracy: `{teacher['arg1_accuracy']}` over "
            f"`{teacher['arg1_target_count']}` targets"
        ),
        (
            f"- Arg2 accuracy: `{teacher['arg2_accuracy']}` over "
            f"`{teacher['arg2_target_count']}` targets"
        ),
        f"- END accuracy: `{teacher['end_accuracy']}`",
        f"- Mean END probability: `{teacher['mean_probability_end']}`",
        "",
        "## Free-running summary",
        "",
        f"- Tasks: `{free['task_count']}`",
        f"- Operator accuracy on evaluated autoregressive positions: `{free['operator_accuracy']}`",
        f"- Joint-step accuracy: `{free['joint_step_accuracy']}`",
        f"- Nonempty-plan rate: `{free['nonempty_plan_rate']}`",
        f"- Exact-plan rate: `{free['exact_plan_rate']}`",
        f"- Full-plan executable rate: `{free['full_plan_executable_rate']}`",
        f"- Goal-success rate: `{free['goal_success_rate']}`",
        "",
        "## Gold-history versus predicted-history",
        "",
        f"- Gold-history operator accuracy: `{history['gold_history']['operator_accuracy']}`",
        (
            "- Predicted-history operator accuracy: "
            f"`{history['predicted_history']['operator_accuracy']}`"
        ),
        "",
        "## Loss breakdown",
        "",
        f"- Mean total loss: `{loss['mean_total_loss']}`",
        f"- Clipping occurrences: `{loss['clipping_occurrence_count']}`",
        "",
        "## First-error distribution",
        "",
    ]
    for category in FIRST_ERROR_CATEGORIES:
        lines.append(f"- `{category}`: `{payload['first_error_distribution'][category]}`")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The metrics do not identify a cause automatically. Strong teacher-forced performance",
            "with weak rollout supports an exposure/rollout hypothesis; weak teacher-forced",
            "performance supports a basic learnability hypothesis. Any intervention requires a",
            "separate decision record. This diagnostic does not define thresholds, issue `GO`, or",
            "unlock A3b.",
            "",
            f"Canonical identity: `{payload['canonical_identity']}`",
            "",
        ]
    )
    return "\n".join(lines)
