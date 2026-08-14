"""Extended read-only observability for the A2 END-only learnability diagnostic.

This module wraps the existing development-learnability-diagnostic/0.1 implementation.
It does not implement an alternative trainer. The frozen quality-v0.1 `_train` function
remains the only training producer; instrumentation observes its real gradient clipping and
AdamW steps and restores every patched callable before returning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import learnability as core
from .canonical import canonical_bytes, sha256
from .e2e import ACTION_NAMES
from .model import LockedPlanner, canonical_task_encoding
from .quality import _optimizer_named_parameters, _optimizer_parameter_policy
from .training import ACTIONS, labels

VERSION = "development-learnability-diagnostic/0.2"
STATUS = "development-only-diagnostic"
CORE_DIRECTORY = "core-v0.1"
OUTPUT_JSON = "a2-end-collapse-diagnostic-v0.2.json"
OUTPUT_MARKDOWN = "A2_END_COLLAPSE_DIAGNOSTIC_V0_2.md"
SCHEMA_PATH = Path(__file__).parent / "schemas" / "learnability_diagnostic_v0_2.schema.json"
ROOT = Path(__file__).parents[1]

SOURCE_FILES = tuple(
    sorted(
        set(core.SOURCE_FILES)
        | {
            ".github/workflows/a2-learnability-diagnostic.yml",
            "docs/evaluations/A2_END_COLLAPSE_DIAGNOSTIC_SPEC_RU_v0.2.md",
            "planner_toy/learnability_v0_2.py",
            "planner_toy/schemas/learnability_diagnostic_v0_2.schema.json",
            "scripts/run_toy_learnability_diagnostic_v0_2.py",
        }
    )
)

_OBSERVER_LOCK = threading.Lock()
_ACTIONS = tuple(name for name, _ in sorted(ACTIONS.items(), key=lambda item: item[1]))


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _validate_implementation_commit(commit: str) -> None:
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("LEARNABILITY_V0_2_IMPLEMENTATION_COMMIT_FORMAT")
    if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("LEARNABILITY_V0_2_IMPLEMENTATION_COMMIT_NOT_FOUND")
    if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError("LEARNABILITY_V0_2_IMPLEMENTATION_COMMIT_NOT_ANCESTOR")
    for path in SOURCE_FILES:
        if _git_bytes("show", f"{commit}:{path}").returncode:
            raise ValueError(f"LEARNABILITY_V0_2_IMPLEMENTATION_SOURCE_MISSING:{path}")


def source_identity() -> dict[str, Any]:
    files = [{"path": path, "sha256": _file_hash(ROOT / path)} for path in SOURCE_FILES]
    return {"source_files": files, "source_sha256": sha256(files)}


def source_identity_at_commit(commit: str) -> dict[str, Any]:
    _validate_implementation_commit(commit)
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        if result.returncode:
            raise ValueError(f"LEARNABILITY_V0_2_IMPLEMENTATION_SOURCE_MISSING:{path}")
        files.append(
            {
                "path": path,
                "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest(),
            }
        )
    return {"source_files": files, "source_sha256": sha256(files)}


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _tensor_l2(tensor: torch.Tensor) -> float:
    value = tensor.detach().to(dtype=torch.float64, device="cpu")
    return float(torch.linalg.vector_norm(value))


def _delta_l2(before: torch.Tensor, after: torch.Tensor) -> float:
    return _tensor_l2(after.detach().cpu() - before.detach().cpu())


def _gradient_l2(parameters: list[torch.Tensor]) -> float:
    squares = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        value = gradient.detach().to(dtype=torch.float64, device="cpu")
        squares += float(torch.sum(value * value))
    return math.sqrt(squares)


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {gold: {predicted: 0 for predicted in _ACTIONS} for gold in _ACTIONS}


def _teacher_positions(
    model: LockedPlanner, rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for row in rows:
        action, arg1, arg2 = labels(row)
        valid = len(row["oracle_work_plan"])
        logits = model(canonical_task_encoding(row), action, arg1, arg2)
        for position_index in range(valid):
            gold_step = row["oracle_work_plan"][position_index]
            gold_operator = gold_step[0]
            gold_arg1 = gold_step[1] if len(gold_step) > 1 else None
            gold_arg2 = gold_step[2] if len(gold_step) > 2 else None
            action_logits = logits.action[0, position_index].float()
            probabilities = torch.softmax(action_logits, dim=-1)
            predicted_id = int(action_logits.argmax())
            predicted_operator = ACTION_NAMES[predicted_id]
            end_logit = float(action_logits[ACTIONS["END"]])
            non_end_logits = torch.stack(
                [action_logits[ACTIONS[name]] for name in _ACTIONS if name != "END"]
            )
            margin = end_logit - float(non_end_logits.max())
            has_arg1 = gold_arg1 is not None
            has_arg2 = gold_arg2 is not None
            arg1_prediction = None
            arg2_prediction = None
            if has_arg1:
                arg1_id = int(logits.arg1[0, position_index, : len(row["blocks"])].argmax())
                arg1_prediction = row["blocks"][arg1_id]
            if has_arg2:
                arg2_id = int(logits.arg2[0, position_index, : len(row["blocks"])].argmax())
                arg2_prediction = row["blocks"][arg2_id]
            arg1_correct = arg1_prediction == gold_arg1 if has_arg1 else None
            arg2_correct = arg2_prediction == gold_arg2 if has_arg2 else None
            operator_correct = predicted_operator == gold_operator
            positions.append(
                {
                    "seed": seed,
                    "task_id": row["task_id"],
                    "position_index": position_index,
                    "gold_operator": gold_operator,
                    "predicted_operator": predicted_operator,
                    "operator_correct": operator_correct,
                    "gold_arg1": gold_arg1,
                    "gold_arg2": gold_arg2,
                    "arg1_head_prediction": arg1_prediction,
                    "arg2_head_prediction": arg2_prediction,
                    "arg1_correct": arg1_correct,
                    "arg2_correct": arg2_correct,
                    "joint_step_correct": (
                        operator_correct
                        and arg1_correct is not False
                        and arg2_correct is not False
                    ),
                    "probability_end": float(probabilities[ACTIONS["END"]]),
                    "end_vs_best_non_end_logit_margin": margin,
                }
            )
    return positions


def _teacher_summary(positions: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = _empty_confusion()
    predicted_counts = Counter()
    for row in positions:
        confusion[row["gold_operator"]][row["predicted_operator"]] += 1
        predicted_counts[row["predicted_operator"]] += 1
    arg1 = [row for row in positions if row["gold_arg1"] is not None]
    arg2 = [row for row in positions if row["gold_arg2"] is not None]
    end = [row for row in positions if row["gold_operator"] == "END"]
    non_end = [row for row in positions if row["gold_operator"] != "END"]
    max_non_end_prediction_count = max(
        (predicted_counts[name] for name in _ACTIONS if name != "END"), default=0
    )
    return {
        "position_count": len(positions),
        "operator_accuracy": _rate(
            sum(row["operator_correct"] for row in positions), len(positions)
        ),
        "arg1_accuracy": _rate(sum(row["arg1_correct"] for row in arg1), len(arg1)),
        "arg1_target_count": len(arg1),
        "arg2_accuracy": _rate(sum(row["arg2_correct"] for row in arg2), len(arg2)),
        "arg2_target_count": len(arg2),
        "joint_step_accuracy": _rate(
            sum(row["joint_step_correct"] for row in positions), len(positions)
        ),
        "end_accuracy": _rate(sum(row["operator_correct"] for row in end), len(end)),
        "non_end_accuracy": _rate(
            sum(row["operator_correct"] for row in non_end), len(non_end)
        ),
        "predicted_end_count": predicted_counts["END"],
        "predicted_end_rate": _rate(predicted_counts["END"], len(positions)),
        "mean_end_probability": _mean([row["probability_end"] for row in positions]),
        "mean_end_vs_best_non_end_logit_margin": _mean(
            [row["end_vs_best_non_end_logit_margin"] for row in positions]
        ),
        "end_is_modal_prediction": (
            bool(positions) and predicted_counts["END"] > max_non_end_prediction_count
        ),
        "all_positions_predict_end": (
            bool(positions) and predicted_counts["END"] == len(positions)
        ),
        "predicted_operator_counts": {name: predicted_counts[name] for name in _ACTIONS},
        "confusion_matrix": confusion,
    }


def _group_teacher_by_position(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in positions:
        groups[row["position_index"]].append(row)
    return {str(index): _teacher_summary(groups[index]) for index in sorted(groups)}


def _copy_optimizer_weights_to_model(
    optimizer: torch.optim.Optimizer, seed: int
) -> LockedPlanner:
    model = LockedPlanner(seed, core.VARIANT).cpu()
    ordered = _optimizer_named_parameters(model)
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(ordered) != len(optimizer_parameters):
        raise RuntimeError("LEARNABILITY_V0_2_OPTIMIZER_PARAMETER_COUNT_MISMATCH")
    with torch.no_grad():
        for (_, destination), source in zip(ordered, optimizer_parameters, strict=True):
            destination.copy_(source.detach())
    return model


class _TrainingTrajectoryObserver:
    """Read-only observer around the real quality-v0.1 clip and AdamW step calls."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._original_clip = torch.nn.utils.clip_grad_norm_
        self._original_step = torch.optim.AdamW.step
        self._seed: int | None = None
        self._rows: list[dict[str, Any]] = []
        self._pending_gradient: dict[str, Any] | None = None
        self._update_index = 0
        self._entered = False

    def begin_seed(self, seed: int, rows: list[dict[str, Any]]) -> None:
        if self._seed is not None:
            raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_SEED_OVERLAP")
        self._seed = seed
        self._rows = sorted(rows, key=lambda row: row["task_id"])
        self._pending_gradient = None
        self._update_index = 0

    def end_seed(self) -> None:
        if self._pending_gradient is not None:
            raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_PENDING_GRADIENT")
        if self._update_index != core.UPDATES_PER_RUN:
            raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_UPDATE_COUNT")
        self._seed = None
        self._rows = []

    def __enter__(self) -> _TrainingTrajectoryObserver:
        if not _OBSERVER_LOCK.acquire(blocking=False):
            raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_CONCURRENT")
        self._entered = True

        def observed_clip(parameters, max_norm, *args, **kwargs):
            parameter_list = list(parameters)
            gradients = [
                parameter.grad
                for parameter in parameter_list
                if parameter.grad is not None
            ]
            all_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            nonzero_count = sum(bool(torch.any(gradient != 0)) for gradient in gradients)
            norm = self._original_clip(parameter_list, max_norm, *args, **kwargs)
            self._pending_gradient = {
                "gradient_norm_pre_clip": float(norm),
                "gradient_norm_post_clip": _gradient_l2(parameter_list),
                "gradient_tensor_count": len(gradients),
                "nonzero_gradient_tensor_count": nonzero_count,
                "all_gradients_finite": all_finite,
                "gradient_clip_norm": float(max_norm),
                "clipping_occurred": float(norm) > float(max_norm),
            }
            return norm

        def observed_step(optimizer, *args, **kwargs):
            if self._seed is None or not self._rows:
                raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_STEP_OUTSIDE_SEED")
            if self._pending_gradient is None:
                raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_STEP_WITHOUT_CLIP")
            optimizer_parameters = [
                parameter for group in optimizer.param_groups for parameter in group["params"]
            ]
            before = [parameter.detach().clone() for parameter in optimizer_parameters]
            result = self._original_step(optimizer, *args, **kwargs)
            after = [parameter.detach() for parameter in optimizer_parameters]
            deltas = [
                _delta_l2(left, right)
                for left, right in zip(before, after, strict=True)
            ]
            model = _copy_optimizer_weights_to_model(optimizer, self._seed)
            with core._read_only_diagnostic_pass(model):
                positions = _teacher_positions(model, self._rows, self._seed)
            summary = _teacher_summary(positions)
            epoch_index = self._update_index // len(self._rows)
            task_id = self._rows[self._update_index % len(self._rows)]["task_id"]
            self.records.append(
                {
                    "seed": self._seed,
                    "update_index": self._update_index,
                    "epoch_index": epoch_index,
                    "task_id": task_id,
                    **self._pending_gradient,
                    "active_parameter_tensor_count": len(deltas),
                    "active_parameter_tensors_changed": sum(
                        delta > 0.0 for delta in deltas
                    ),
                    "active_parameter_update_l2": math.sqrt(
                        sum(delta * delta for delta in deltas)
                    ),
                    "teacher_forced": summary,
                    "teacher_forced_by_gold_position": _group_teacher_by_position(positions),
                }
            )
            self._pending_gradient = None
            self._update_index += 1
            return result

        torch.nn.utils.clip_grad_norm_ = observed_clip
        torch.optim.AdamW.step = observed_step
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        torch.nn.utils.clip_grad_norm_ = self._original_clip
        torch.optim.AdamW.step = self._original_step
        if self._entered:
            _OBSERVER_LOCK.release()
        self._entered = False
        if exc_type is None and self._seed is not None:
            raise RuntimeError("LEARNABILITY_V0_2_OBSERVER_SEED_NOT_CLOSED")


@contextmanager
def _instrument_core_training(observer: _TrainingTrajectoryObserver):
    original = core._train_a2_with_loss_trace

    def observed_train(rows, seed, output, dataset_hash):
        observer.begin_seed(seed, rows)
        try:
            result = original(rows, seed, output, dataset_hash)
        except Exception:
            observer._seed = None
            observer._rows = []
            observer._pending_gradient = None
            raise
        observer.end_seed()
        return result

    core._train_a2_with_loss_trace = observed_train
    try:
        with observer:
            yield
    finally:
        core._train_a2_with_loss_trace = original


def _checkpoint_delta(root: Path, seed: int) -> dict[str, Any]:
    run_dir = root / "training-runs" / "A2" / f"seed-{seed}"
    initialization = torch.load(
        run_dir / "initialization.pt", map_location="cpu", weights_only=True
    )
    trained = torch.load(run_dir / "trained.pt", map_location="cpu", weights_only=True)
    optimizer_state = torch.load(
        run_dir / "optimizer-state.pt", map_location="cpu", weights_only=True
    )
    model = LockedPlanner(seed, core.VARIANT).cpu()
    active, dormant = _optimizer_parameter_policy(model)
    active_deltas = {name: _delta_l2(initialization[name], trained[name]) for name in active}
    dormant_changed = [
        name for name in dormant if not torch.equal(initialization[name], trained[name])
    ]
    state_entries = optimizer_state["state"] if isinstance(optimizer_state, dict) else {}
    nonzero_moment_entries = 0
    finite_optimizer_state = True
    for entry in state_entries.values():
        if not isinstance(entry, dict):
            finite_optimizer_state = False
            continue
        moments = [entry.get("exp_avg"), entry.get("exp_avg_sq")]
        for moment in moments:
            if not torch.is_tensor(moment) or not bool(torch.isfinite(moment).all()):
                finite_optimizer_state = False
            elif bool(torch.any(moment != 0)):
                nonzero_moment_entries += 1

    def named_delta(name: str) -> float:
        return _delta_l2(initialization[name], trained[name])

    end_index = ACTIONS["END"]
    action_weight_delta = trained["heads.action.weight"] - initialization["heads.action.weight"]
    action_bias_delta = trained["heads.action.bias"] - initialization["heads.action.bias"]
    all_finite = all(
        bool(torch.isfinite(value).all()) for value in initialization.values()
    ) and all(
        bool(torch.isfinite(value).all()) for value in trained.values()
    )
    changed_count = sum(delta > 0.0 for delta in active_deltas.values())
    return {
        "seed": seed,
        "active_parameter_tensor_count": len(active),
        "active_parameter_tensors_changed": changed_count,
        "fraction_active_tensors_changed": _rate(changed_count, len(active)),
        "active_parameter_delta_l2": math.sqrt(
            sum(delta * delta for delta in active_deltas.values())
        ),
        "active_parameter_delta_l2_by_name": active_deltas,
        "dormant_parameter_tensor_count": len(dormant),
        "dormant_parameter_tensors_changed": len(dormant_changed),
        "all_checkpoint_values_finite": all_finite,
        "optimizer_state_parameter_count": len(state_entries),
        "optimizer_nonzero_moment_tensor_count": nonzero_moment_entries,
        "optimizer_state_finite": finite_optimizer_state,
        "action_head_weight_delta_l2": _tensor_l2(action_weight_delta),
        "action_head_bias_delta_l2": _tensor_l2(action_bias_delta),
        "end_action_weight_row_delta_l2": _tensor_l2(action_weight_delta[end_index]),
        "end_action_bias_delta": float(action_bias_delta[end_index]),
        "arg1_pointer_weight_delta_l2": named_delta("heads.arg1_pointer.weight"),
        "arg2_pointer_weight_delta_l2": named_delta("heads.arg2_pointer.weight"),
    }


def _load_trained_model(core_root: Path, seed: int) -> LockedPlanner:
    model = LockedPlanner(seed, core.VARIANT).cpu()
    state = torch.load(
        core_root / "training-runs" / "A2" / f"seed-{seed}" / "trained.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model


def _free_running_observation(
    model: LockedPlanner, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []

    def capture(_module, _args, _kwargs, output) -> None:
        position = len(trace)
        action_logits = output.action[0, position].float()
        probabilities = torch.softmax(action_logits, dim=-1)
        predicted_id = int(action_logits.argmax())
        non_end_logits = torch.stack(
            [action_logits[ACTIONS[name]] for name in _ACTIONS if name != "END"]
        )
        trace.append(
            {
                "position_index": position,
                "predicted_operator": ACTION_NAMES[predicted_id],
                "probability_end": float(probabilities[ACTIONS["END"]]),
                "end_vs_best_non_end_logit_margin": (
                    float(action_logits[ACTIONS["END"]]) - float(non_end_logits.max())
                ),
            }
        )

    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        core_result = core.free_running_task(model, row, split="train", seed=seed)
    finally:
        handle.remove()
    if len(trace) != core_result["model_forward_count"]:
        raise RuntimeError("LEARNABILITY_V0_2_FREE_TRACE_COUNT_MISMATCH")
    first_end = next(
        (item["position_index"] for item in trace if item["predicted_operator"] == "END"),
        None,
    )
    gold_end_position = len(row["oracle_work_plan"]) - 1
    return {
        "seed": seed,
        "task_id": row["task_id"],
        "predicted_plan": core_result["predicted_plan"],
        "predicted_plan_length": core_result["predicted_plan_length"],
        "zero_action_plan": core_result["predicted_action_count"] == 0,
        "end_only": core_result["end_only"],
        "initial_goal_satisfied": core_result["initial_goal_satisfied"],
        "first_predicted_end_position": first_end,
        "early_end": first_end is not None and first_end < gold_end_position,
        "predicted_end_count": sum(item["predicted_operator"] == "END" for item in trace),
        "predicted_end_rate": _rate(
            sum(item["predicted_operator"] == "END" for item in trace), len(trace)
        ),
        "mean_end_probability": _mean([item["probability_end"] for item in trace]),
        "mean_end_vs_best_non_end_logit_margin": _mean(
            [item["end_vs_best_non_end_logit_margin"] for item in trace]
        ),
        "positions": trace,
    }


def _free_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [position for task in tasks for position in task["positions"]]
    lengths = Counter(task["predicted_plan_length"] for task in tasks)
    first_end = Counter(
        (
            "none"
            if task["first_predicted_end_position"] is None
            else str(task["first_predicted_end_position"])
        )
        for task in tasks
    )
    return {
        "task_count": len(tasks),
        "predicted_position_count": len(positions),
        "predicted_end_count": sum(item["predicted_operator"] == "END" for item in positions),
        "predicted_end_rate": _rate(
            sum(item["predicted_operator"] == "END" for item in positions), len(positions)
        ),
        "mean_end_probability": _mean([item["probability_end"] for item in positions]),
        "mean_end_vs_best_non_end_logit_margin": _mean(
            [item["end_vs_best_non_end_logit_margin"] for item in positions]
        ),
        "early_end_count": sum(task["early_end"] for task in tasks),
        "early_end_rate": _rate(sum(task["early_end"] for task in tasks), len(tasks)),
        "zero_action_plan_count": sum(task["zero_action_plan"] for task in tasks),
        "zero_action_plan_rate": _rate(
            sum(task["zero_action_plan"] for task in tasks), len(tasks)
        ),
        "plan_length_distribution": {
            str(length): lengths[length] for length in sorted(lengths)
        },
        "first_predicted_end_position_distribution": {
            key: first_end[key]
            for key in sorted(first_end, key=lambda value: (value == "none", value))
        },
    }


def _aggregate_free(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_initial: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_seed[task["seed"]].append(task)
        by_initial[task["initial_goal_satisfied"]].append(task)
    return {
        "overall": _free_summary(tasks),
        "by_seed": {str(seed): _free_summary(by_seed[seed]) for seed in sorted(by_seed)},
        "by_initial_goal_satisfied": {
            str(value).lower(): _free_summary(by_initial[value]) for value in sorted(by_initial)
        },
    }


def _first_update(records: list[dict[str, Any]], field: str) -> int | None:
    return next((record["update_index"] for record in records if record[field]), None)


def _trajectory_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[record["seed"]].append(record)
    result = {}
    for seed in sorted(by_seed):
        rows = sorted(by_seed[seed], key=lambda record: record["update_index"])
        epoch_loss_means = []
        for epoch_index in range(core.EPOCHS):
            epoch_rows = [row for row in rows if row["epoch_index"] == epoch_index]
            epoch_loss_means.append(
                {
                    "epoch_index": epoch_index,
                    "mean_total_loss": _mean(
                        [row["losses"]["total_loss"] for row in epoch_rows]
                    ),
                    "mean_operator_loss": _mean(
                        [row["losses"]["operator_loss"] for row in epoch_rows]
                    ),
                    "mean_arg1_pointer_loss": _mean(
                        [row["losses"]["arg1_pointer_loss"] for row in epoch_rows]
                    ),
                    "mean_arg2_pointer_loss": _mean(
                        [row["losses"]["arg2_pointer_loss"] for row in epoch_rows]
                    ),
                }
            )
        result[str(seed)] = {
            "first_end_modal_update_index": _first_update(
                [
                    {**record, "value": record["teacher_forced"]["end_is_modal_prediction"]}
                    for record in rows
                ],
                "value",
            ),
            "first_all_positions_end_update_index": _first_update(
                [
                    {
                        **record,
                        "value": record["teacher_forced"]["all_positions_predict_end"],
                    }
                    for record in rows
                ],
                "value",
            ),
            "epoch_end_updates": [
                record["update_index"]
                for record in rows
                if record["update_index"] in {2, 5, 8}
            ],
            "nonzero_gradient_update_count": sum(
                record["nonzero_gradient_tensor_count"] > 0 for record in rows
            ),
            "clipping_occurrence_count": sum(
                record["clipping_occurred"] for record in rows
            ),
            "epoch_loss_means": epoch_loss_means,
        }
    return result


def _localize(teacher: dict[str, Any], free_exact: float | None) -> dict[str, str]:
    if teacher["operator_accuracy"] != 1.0:
        stage = "TEACHER_FORCED_OPERATOR_ERRORS_PRESENT"
        hypothesis = "BASIC_LEARNABILITY_OR_OPTIMIZATION_FAILURE_PRESENT"
    elif teacher["arg1_accuracy"] != 1.0 or teacher["arg2_accuracy"] != 1.0:
        stage = "TEACHER_FORCED_POINTER_ERRORS_PRESENT"
        hypothesis = "POINTER_HEAD_LEARNABILITY_FAILURE_PRESENT"
    elif free_exact != 1.0:
        stage = "FREE_RUNNING_ONLY_ERRORS_PRESENT"
        hypothesis = "EXPOSURE_OR_ROLLOUT_FAILURE"
    else:
        stage = "NO_ERROR_LOCALIZED_ON_EVALUATED_TRAIN_TASKS"
        hypothesis = "NO_SINGLE_FAILURE_HYPOTHESIS_SELECTED"
    return {
        "first_major_failure_stage": stage,
        "supported_hypothesis": hypothesis,
    }


def _interpretation(
    core_payload: dict[str, Any], final_teacher: dict[str, Any]
) -> dict[str, Any]:
    by_seed = {}
    for seed in core_payload["seeds"]:
        seed_key = str(seed)
        by_seed[seed_key] = _localize(
            final_teacher["by_seed"][seed_key],
            core_payload["aggregates"]["free_running"]["by_seed"][seed_key][
                "exact_plan_rate"
            ],
        )
    overall = _localize(
        final_teacher["overall"],
        core_payload["aggregates"]["free_running"]["overall"]["exact_plan_rate"],
    )
    stages = {value["first_major_failure_stage"] for value in by_seed.values()}
    return {
        **overall,
        "interpretation_label": "SUPPORTED HYPOTHESIS / NOT PROVEN",
        "by_seed": by_seed,
        "cross_seed_localization_consistent": len(stages) == 1,
    }


def _projected_gold_history(
    core_payload: dict[str, Any], train_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    train_by_id = {row["task_id"]: row for row in train_rows}
    result = []
    for task in core_payload["teacher_forced"]:
        projected = core._gold_history_projection(task, train_by_id[task["task_id"]])
        result.append(
            {
                "seed": task["seed"],
                "task_id": task["task_id"],
                "predicted_plan": projected["predicted_plan"],
                "predicted_action_count": projected["predicted_action_count"],
                "exact_plan_match": projected["exact_plan_match"],
                "full_plan_executable": projected["full_plan_executable"],
                "final_goal_success": projected["final_goal_success"],
                "first_mismatch_position": projected["first_mismatch_position"],
                "first_mismatch_type": projected["first_mismatch_type"],
            }
        )
    result.sort(key=lambda row: (row["seed"], row["task_id"]))
    return result


def _history_mode_summary(
    core_payload: dict[str, Any], projected: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "teacher_forced_operator_accuracy": core_payload["aggregates"]["teacher_forced"][
            "overall"
        ]["operator_accuracy"],
        "teacher_forced_joint_step_accuracy": core_payload["aggregates"]["teacher_forced"][
            "overall"
        ]["joint_step_accuracy"],
        "gold_history_projected_exact_plan_rate": _rate(
            sum(row["exact_plan_match"] for row in projected), len(projected)
        ),
        "gold_history_projected_goal_success_rate": _rate(
            sum(row["final_goal_success"] for row in projected), len(projected)
        ),
        "free_running_exact_plan_rate": core_payload["aggregates"]["free_running"][
            "overall"
        ]["exact_plan_rate"],
        "free_running_goal_success_rate": core_payload["aggregates"]["free_running"][
            "overall"
        ]["goal_success_rate"],
    }


def _artifact_identity(payload: dict[str, Any]) -> str:
    value = copy.deepcopy(payload)
    value.pop("canonical_identity", None)
    return sha256({"schema_version": "toy-learnability-diagnostic-v0.2-hash/1.0", "payload": value})


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"LEARNABILITY_V0_2_NONFINITE:{path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _join_losses(
    records: list[dict[str, Any]], core_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    losses = {
        (row["seed"], row["update_index"]): row
        for row in core_payload["per_update_loss_breakdown"]
    }
    joined = []
    for record in records:
        key = (record["seed"], record["update_index"])
        if key not in losses:
            raise RuntimeError("LEARNABILITY_V0_2_LOSS_JOIN_MISSING")
        loss = losses[key]
        if record["epoch_index"] != loss["epoch_index"] or record["task_id"] != loss["task_id"]:
            raise RuntimeError("LEARNABILITY_V0_2_LOSS_SCHEDULE_MISMATCH")
        joined.append({**record, "losses": loss})
    return joined


def run(
    output: Path,
    *,
    implementation_commit: str,
    seeds: tuple[int, ...] = core.SEEDS,
    task_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    provenance = source_identity_at_commit(implementation_commit)
    if provenance != source_identity():
        raise ValueError("LEARNABILITY_V0_2_WORKING_TREE_SOURCE_MISMATCH")
    if output.exists() and any(output.iterdir()):
        raise ValueError("LEARNABILITY_V0_2_OUTPUT_NOT_CLEAN")
    output.mkdir(parents=True, exist_ok=True)
    core_root = output / CORE_DIRECTORY
    observer = _TrainingTrajectoryObserver()
    with _instrument_core_training(observer):
        core_payload = core.run(
            core_root,
            implementation_commit=implementation_commit,
            seeds=seeds,
            task_ids=task_ids,
        )
    core.validate_diagnostic(core_root)
    trajectory = _join_losses(observer.records, core_payload)
    dataset, train_rows = core._dataset_context()
    selected_ids = set(core_payload["evaluated_task_ids"])
    selected_rows = [row for row in train_rows if row["task_id"] in selected_ids]
    final_teacher_positions = []
    free_tasks = []
    checkpoint_deltas = []
    core_free = {
        (task["seed"], task["task_id"]): task for task in core_payload["free_running"]
    }
    for seed in core_payload["seeds"]:
        model = _load_trained_model(core_root, seed)
        with core._read_only_diagnostic_pass(model):
            final_teacher_positions.extend(_teacher_positions(model, selected_rows, seed))
            for row in selected_rows:
                observed = _free_running_observation(model, row, seed)
                expected = core_free[(seed, row["task_id"])]
                if observed["predicted_plan"] != expected["predicted_plan"]:
                    raise RuntimeError("LEARNABILITY_V0_2_FREE_PLAN_CORE_MISMATCH")
                free_tasks.append(observed)
        checkpoint_deltas.append(_checkpoint_delta(core_root, seed))
    final_teacher = {
        "overall": _teacher_summary(final_teacher_positions),
        "by_seed": {
            str(seed): _teacher_summary(
                [row for row in final_teacher_positions if row["seed"] == seed]
            )
            for seed in core_payload["seeds"]
        },
        "by_gold_position": _group_teacher_by_position(final_teacher_positions),
    }
    free_aggregate = _aggregate_free(free_tasks)
    projected = _projected_gold_history(core_payload, selected_rows)
    history_modes = _history_mode_summary(core_payload, projected)
    payload: dict[str, Any] = {
        "diagnostic_version": VERSION,
        "status": STATUS,
        "implementation_commit": implementation_commit,
        **provenance,
        "core_diagnostic_version": core_payload["diagnostic_version"],
        "core_canonical_identity": core_payload["canonical_identity"],
        "evaluated_splits": ["train"],
        "dataset_lineage_order": core_payload["dataset_lineage_order"],
        "optimizer_execution_task_order": core_payload["optimizer_execution_task_order"],
        "evaluated_task_ids": core_payload["evaluated_task_ids"],
        "seeds": core_payload["seeds"],
        "frozen_dataset_lineage_hash": dataset["frozen_dataset_lineage_hash"],
        "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
        "heldout_accessed": False,
        "training_policy_changed": False,
        "model_changed": False,
        "gate_decision": None,
        "learnability_thresholds": None,
        "per_update_training_observation": trajectory,
        "training_trajectory_summary": _trajectory_summary(trajectory),
        "final_teacher_forced": final_teacher,
        "gold_history_projected": projected,
        "history_mode_summary": history_modes,
        "free_running": free_tasks,
        "free_running_aggregate": free_aggregate,
        "checkpoint_deltas": checkpoint_deltas,
        "interpretation": _interpretation(core_payload, final_teacher),
    }
    payload["canonical_identity"] = _artifact_identity(payload)
    validate_payload(payload, core_payload=core_payload)
    (output / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (output / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def validate_payload(payload: dict[str, Any], *, core_payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
    except SchemaError as exc:
        raise ValueError("LEARNABILITY_V0_2_SCHEMA_INVALID") from exc
    except ValidationError as exc:
        raise ValueError("LEARNABILITY_V0_2_SCHEMA_VALIDATION_FAILED") from exc
    _assert_finite(payload)
    if payload["diagnostic_version"] != VERSION or payload["status"] != STATUS:
        raise ValueError("LEARNABILITY_V0_2_IDENTITY_MISMATCH")
    if (
        payload["heldout_accessed"]
        or payload["training_policy_changed"]
        or payload["model_changed"]
    ):
        raise ValueError("LEARNABILITY_V0_2_SCOPE_FLAG_MISMATCH")
    if payload["gate_decision"] is not None or payload["learnability_thresholds"] is not None:
        raise ValueError("LEARNABILITY_V0_2_GATE_OR_THRESHOLD_FORBIDDEN")
    expected_source = source_identity_at_commit(payload["implementation_commit"])
    if (
        payload["source_files"] != expected_source["source_files"]
        or payload["source_sha256"] != expected_source["source_sha256"]
    ):
        raise ValueError("LEARNABILITY_V0_2_SOURCE_IDENTITY_MISMATCH")
    if payload["core_diagnostic_version"] != core.VERSION:
        raise ValueError("LEARNABILITY_V0_2_CORE_VERSION_MISMATCH")
    if payload["core_canonical_identity"] != core_payload["canonical_identity"]:
        raise ValueError("LEARNABILITY_V0_2_CORE_IDENTITY_MISMATCH")
    for field in (
        "evaluated_splits",
        "dataset_lineage_order",
        "optimizer_execution_task_order",
        "evaluated_task_ids",
        "seeds",
        "frozen_dataset_lineage_hash",
        "evaluated_train_split_hash",
    ):
        if payload[field] != core_payload[field]:
            raise ValueError(f"LEARNABILITY_V0_2_CORE_BINDING_MISMATCH:{field}")
    if set(payload["evaluated_task_ids"]) & set(
        core.FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS
    ):
        raise ValueError("LEARNABILITY_V0_2_HELDOUT_TASK_PRESENT")
    expected_updates = len(payload["seeds"]) * core.UPDATES_PER_RUN
    if len(payload["per_update_training_observation"]) != expected_updates:
        raise ValueError("LEARNABILITY_V0_2_UPDATE_COVERAGE")
    if any(
        not row["all_gradients_finite"]
        for row in payload["per_update_training_observation"]
    ):
        raise ValueError("LEARNABILITY_V0_2_NONFINITE_GRADIENT")
    if any(
        row["dormant_parameter_tensors_changed"] != 0
        for row in payload["checkpoint_deltas"]
    ):
        raise ValueError("LEARNABILITY_V0_2_DORMANT_PARAMETER_CHANGED")
    if any(
        not row["all_checkpoint_values_finite"] or not row["optimizer_state_finite"]
        for row in payload["checkpoint_deltas"]
    ):
        raise ValueError("LEARNABILITY_V0_2_CHECKPOINT_OR_OPTIMIZER_NONFINITE")
    if payload["canonical_identity"] != _artifact_identity(payload):
        raise ValueError("LEARNABILITY_V0_2_CANONICAL_IDENTITY_MISMATCH")


def validate_diagnostic(root: Path) -> dict[str, Any]:
    expected = {CORE_DIRECTORY, OUTPUT_JSON, OUTPUT_MARKDOWN}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("LEARNABILITY_V0_2_TOP_LEVEL_COVERAGE_MISMATCH")
    core_root = root / CORE_DIRECTORY
    core_result = core.validate_diagnostic(core_root)
    core_payload = json.loads((core_root / core.OUTPUT_JSON).read_text(encoding="utf-8"))
    payload = json.loads((root / OUTPUT_JSON).read_text(encoding="utf-8"))
    validate_payload(payload, core_payload=core_payload)
    if (root / OUTPUT_MARKDOWN).read_text(encoding="utf-8") != render_markdown(payload):
        raise ValueError("LEARNABILITY_V0_2_MARKDOWN_MISMATCH")
    return {
        "valid": True,
        "diagnostic_complete": core_result["diagnostic_complete"],
        "canonical_identity": payload["canonical_identity"],
        "core_canonical_identity": payload["core_canonical_identity"],
        "heldout_accessed": False,
        "training_policy_changed": False,
    }


def _answer(value: Any) -> str:
    return "YES" if value else "NO"


def render_markdown(payload: dict[str, Any]) -> str:
    teacher = payload["final_teacher_forced"]["overall"]
    free = payload["free_running_aggregate"]["overall"]
    updates = payload["per_update_training_observation"]
    checkpoints = payload["checkpoint_deltas"]
    any_nonzero_gradient = any(
        row["nonzero_gradient_tensor_count"] > 0 for row in updates
    )
    all_seeds_changed = all(
        row["active_parameter_tensors_changed"] > 0 for row in checkpoints
    )
    model_trained = any_nonzero_gradient and all_seeds_changed
    end_teacher = teacher["predicted_end_rate"]
    end_free = free["predicted_end_rate"]
    lines = [
        "# A2 END-only learnability diagnostic v0.2",
        "",
        "> Development-only, read-only diagnostic. No gate, threshold, model change, "
        "or training-policy change.",
        "",
        f"- Implementation: `{payload['implementation_commit']}`",
        f"- Canonical identity: `{payload['canonical_identity']}`",
        f"- Core v0.1 identity: `{payload['core_canonical_identity']}`",
        f"- Held-out accessed: `{str(payload['heldout_accessed']).lower()}`",
        "",
        "## Diagnostic questions",
        "",
        (
            f"1. Модель вообще обучалась? **{_answer(model_trained)}** — nonzero "
            "gradients observed; active checkpoint tensors changed: "
            f"`{[row['active_parameter_tensors_changed'] for row in checkpoints]}`."
        ),
        (
            "2. Action head обучался? Final teacher-forced operator accuracy: "
            f"`{teacher['operator_accuracy']}`; END accuracy: `{teacher['end_accuracy']}`; "
            f"non-END accuracy: `{teacher['non_end_accuracy']}`."
        ),
        (
            f"3. Pointer heads обучались? Arg1: `{teacher['arg1_accuracy']}` over "
            f"`{teacher['arg1_target_count']}` gold targets; Arg2: "
            f"`{teacher['arg2_accuracy']}` over `{teacher['arg2_target_count']}` gold "
            "targets; independent of predicted operator."
        ),
        (
            f"4. END dominance: teacher-forced predicted END rate `{end_teacher}` vs "
            f"free-running `{end_free}`; free early-END rate `{free['early_end_rate']}`."
        ),
        (
            "5. Первый major failure: "
            f"`{payload['interpretation']['first_major_failure_stage']}`."
        ),
        (
            "6. Seeds 17/29/43: localization consistent across evaluated seeds: "
            f"`{str(payload['interpretation']['cross_seed_localization_consistent']).lower()}`; "
            "per-seed localizations remain explicit in canonical JSON."
        ),
        (
            f"7. `{payload['interpretation']['interpretation_label']}`: "
            f"`{payload['interpretation']['supported_hypothesis']}`."
        ),
        "",
        "## History-mode comparison",
        "",
        (
            "- Teacher-forced operator accuracy: "
            f"`{payload['history_mode_summary']['teacher_forced_operator_accuracy']}`"
        ),
        (
            "- Gold-history projected exact-plan rate: "
            f"`{payload['history_mode_summary']['gold_history_projected_exact_plan_rate']}`"
        ),
        (
            "- True free-running exact-plan rate: "
            f"`{payload['history_mode_summary']['free_running_exact_plan_rate']}`"
        ),
        "",
        "## END-specific observability",
        "",
        f"- Teacher mean END probability: `{teacher['mean_end_probability']}`",
        (
            "- Teacher mean END-vs-best-non-END logit margin: "
            f"`{teacher['mean_end_vs_best_non_end_logit_margin']}`"
        ),
        f"- Free mean END probability: `{free['mean_end_probability']}`",
        (
            "- Free mean END-vs-best-non-END logit margin: "
            f"`{free['mean_end_vs_best_non_end_logit_margin']}`"
        ),
        f"- Early-END count/rate: `{free['early_end_count']}` / `{free['early_end_rate']}`",
        f"- Zero-action plans: `{free['zero_action_plan_count']}`",
        (
            "- Plan-length distribution: "
            f"`{json.dumps(free['plan_length_distribution'], sort_keys=True)}`"
        ),
        (
            "- First predicted END positions: "
            f"`{json.dumps(free['first_predicted_end_position_distribution'], sort_keys=True)}`"
        ),
        "",
        "## Optimization observation",
        "",
        f"- Updates observed: `{len(updates)}`",
        f"- Clipping occurrences: `{sum(row['clipping_occurred'] for row in updates)}`",
        (
            "- Nonzero-gradient updates: "
            f"`{sum(row['nonzero_gradient_tensor_count'] > 0 for row in updates)}`"
        ),
        (
            "- Per-update total/action/pointer losses, pre/post clipping norms, "
            "teacher-forced confusion matrices, by-position END metrics, and active "
            "parameter updates are in the JSON."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "The selected hypothesis is descriptive evidence only. Exact teacher-forced "
            "errors establish that learnability/optimization failure is already present "
            "under gold history; exact teacher-forced success with free-running errors is "
            "compatible with exposure/rollout failure. Neither pattern proves causality "
            "or authorizes an intervention."
        ),
        "",
    ]
    return "\n".join(lines)
