"""Train-only A2 causal microexperiment for operator-loss weighting vs task order.

Development-only scientific experiment. It does not alter the frozen quality trainer,
the accepted A2 diagnostic, A3/A4, held-out data, or any formal acceptance gate.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .canonical import canonical_bytes, sha256
from .canonical_runtime import configure_canonical_cpu_runtime
from .learnability import (
    FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS,
    SEEDS,
    _read_only_diagnostic_pass,
    _train_a2_with_loss_trace,
    free_running_task,
    teacher_forced_task,
)
from .learnability import (
    SOURCE_FILES as LEARNABILITY_SOURCE_FILES,
)
from .model import LockedPlanner, canonical_task_encoding
from .numeric_identity import (
    canonical_state_dict_sha256,
    canonical_torch_object_sha256,
)
from .quality import _optimizer_named_parameters
from .semantic import targets
from .train_only_dataset import (
    FROZEN_DATASET_LINEAGE_HASH_V1,
    generate_train_only,
)
from .training import ACTIONS, labels

VERSION = "development-a2-causal-microexperiment/0.1"
STATUS = "development-only-scientific-microexperiment"
VARIANT = "A2"
EPOCHS = 3
EXPECTED_TRAIN_TASK_IDS = ("bw-00000001", "bw-00000002", "bw-00000003")
CANONICAL_ORDER = EXPECTED_TRAIN_TASK_IDS
ORDER_ONLY_ORDER = ("bw-00000002", "bw-00000003", "bw-00000001")
OUTPUT_JSON = "a2-causal-microexperiment.json"
OUTPUT_MARKDOWN = "A2_CAUSAL_MICROEXPERIMENT.md"
INTERPRETATION_LABEL = "SUPPORTED HYPOTHESIS / NOT PROVEN"

ARM_CONTROL = "canonical_control"
ARM_EQUAL_POSITION = "equal_position_operator_loss"
ARM_ORDER_ONLY = "task_order_only"
ARMS = (ARM_CONTROL, ARM_EQUAL_POSITION, ARM_ORDER_ONLY)

ROOT = Path(__file__).parents[1]
SOURCE_FILES = tuple(
    sorted(
        set(LEARNABILITY_SOURCE_FILES)
        | {
            ".github/workflows/a2-causal-microexperiment.yml",
            "docs/evaluations/A2_CAUSAL_MICROEXPERIMENT_SPEC_RU.md",
            "planner_toy/a2_causal_microexperiment.py",
            "planner_toy/a2_causal_microexperiment_validator.py",
            "requirements.lock",
            "scripts/run_a2_causal_microexperiment.py",
        }
    )
)

ARM_CONTRACT = {
    ARM_CONTROL: {
        "trainer": (
            "planner_toy.learnability._train_a2_with_loss_trace -> "
            "planner_toy.quality._train"
        ),
        "operator_loss": "per-task-mean-cross-entropy",
        "task_order": list(CANONICAL_ORDER),
        "changed_dimension": "NONE",
    },
    ARM_EQUAL_POSITION: {
        "trainer": "microexperiment-intervention",
        "operator_loss": "equal-per-position-cross-entropy-epoch-scale-preserved",
        "task_order": list(CANONICAL_ORDER),
        "changed_dimension": "OPERATOR_LOSS_NORMALIZATION_ONLY",
    },
    ARM_ORDER_ONLY: {
        "trainer": "microexperiment-intervention",
        "operator_loss": "per-task-mean-cross-entropy",
        "task_order": list(ORDER_ONLY_ORDER),
        "changed_dimension": "TASK_ORDER_ONLY",
    },
}

EQUIVALENCE_TRACE_FIELDS = (
    "update_index",
    "epoch_index",
    "task_id",
    "operator_loss",
    "arg1_pointer_loss",
    "arg2_pointer_loss",
    "total_loss",
    "operator_target_count",
    "arg1_target_count",
    "arg2_target_count",
    "gradient_norm",
    "gradient_clip_norm",
    "clipping_occurred",
    "operator_position_weight",
)


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _validate_commit(commit: str) -> None:
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("A2_CAUSAL_IMPLEMENTATION_COMMIT_FORMAT")
    if _git_bytes("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("A2_CAUSAL_IMPLEMENTATION_COMMIT_NOT_FOUND")
    for path in SOURCE_FILES:
        if _git_bytes("show", f"{commit}:{path}").returncode:
            raise ValueError(f"A2_CAUSAL_IMPLEMENTATION_SOURCE_MISSING:{path}")


def source_identity_at_commit(commit: str) -> dict[str, Any]:
    _validate_commit(commit)
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        files.append(
            {
                "path": path,
                "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest(),
            }
        )
    return {"source_files": files, "source_sha256": sha256(files)}


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _train_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = generate_train_only()
    rows = list(dataset["train"])
    ids = {row["task_id"] for row in rows}
    if ids != set(EXPECTED_TRAIN_TASK_IDS) or len(rows) != len(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError("A2_CAUSAL_TRAIN_TASK_COVERAGE_MISMATCH")
    if ids & set(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS):
        raise ValueError("A2_CAUSAL_HELDOUT_ACCESS")
    return dataset, rows


def _ordered_rows(rows: list[dict[str, Any]], order: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {row["task_id"]: row for row in rows}
    if set(by_id) != set(order) or len(by_id) != len(order):
        raise ValueError("A2_CAUSAL_TASK_ORDER_COVERAGE_MISMATCH")
    return [by_id[task_id] for task_id in order]


def operator_position_weight(
    *, valid_positions: int, total_positions: int, task_count: int
) -> float:
    """Weight of one operator position under the equal-position intervention."""
    if valid_positions <= 0 or total_positions <= 0 or task_count <= 0:
        raise ValueError("A2_CAUSAL_OPERATOR_WEIGHT_INVALID")
    return task_count / total_positions


def _operator_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    total_positions: int,
    task_count: int,
) -> tuple[torch.Tensor, float]:
    if mode == "per-task-mean-cross-entropy":
        return F.cross_entropy(logits, target), 1.0 / int(target.numel())
    if mode == "equal-per-position-cross-entropy-epoch-scale-preserved":
        coefficient = operator_position_weight(
            valid_positions=int(target.numel()),
            total_positions=total_positions,
            task_count=task_count,
        )
        return F.cross_entropy(logits, target, reduction="sum") * coefficient, coefficient
    raise ValueError(f"A2_CAUSAL_OPERATOR_LOSS_MODE:{mode}")


def _train_intervention(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    order: tuple[str, ...],
    operator_loss_mode: str,
) -> tuple[LockedPlanner, dict[str, Any]]:
    """Mirror the frozen A2 trainer and change only the declared intervention knob."""
    configure_canonical_cpu_runtime(seed)
    model = LockedPlanner(seed, VARIANT).cpu()
    initialization = canonical_state_dict_sha256(model.state_dict())
    optimizer_named = _optimizer_named_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in optimizer_named],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    ordered = _ordered_rows(rows, order)
    total_positions = sum(len(row["oracle_work_plan"]) for row in rows)
    task_count = len(rows)
    updates: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        for row in ordered:
            action, arg1, arg2 = labels(row)
            valid = len(row["oracle_work_plan"])
            target = targets(row)
            _shifted = torch.cat([torch.zeros_like(target[:, :1]), target[:, :-1]], 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(canonical_task_encoding(row), action, arg1, arg2)
            flat = action[:, :valid].flatten()
            action_loss, position_weight = _operator_loss(
                logits.action[:, :valid].flatten(0, 1),
                flat,
                mode=operator_loss_mode,
                total_positions=total_positions,
                task_count=task_count,
            )
            operator_loss_value = float(action_loss.detach())
            arg1_loss = torch.zeros((), dtype=action_loss.dtype)
            arg2_loss = torch.zeros((), dtype=action_loss.dtype)
            loss = action_loss
            one = flat != ACTIONS["END"]
            two = (flat == ACTIONS["UNSTACK"]) | (flat == ACTIONS["STACK"])
            if one.any():
                arg1_loss = F.cross_entropy(
                    logits.arg1[:, :valid].flatten(0, 1)[one],
                    arg1[:, :valid].flatten()[one],
                )
                loss += arg1_loss
            if two.any():
                arg2_loss = F.cross_entropy(
                    logits.arg2[:, :valid].flatten(0, 1)[two],
                    arg2[:, :valid].flatten()[two],
                )
                loss += arg2_loss
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
            )
            optimizer.step()
            updates.append(
                {
                    "update_index": len(updates),
                    "epoch_index": epoch,
                    "task_id": row["task_id"],
                    "operator_loss": operator_loss_value,
                    "arg1_pointer_loss": float(arg1_loss.detach()) if one.any() else None,
                    "arg2_pointer_loss": float(arg2_loss.detach()) if two.any() else None,
                    "total_loss": float(loss.detach()),
                    "operator_target_count": int(flat.numel()),
                    "arg1_target_count": int(one.sum()),
                    "arg2_target_count": int(two.sum()),
                    "gradient_norm": gradient_norm,
                    "gradient_clip_norm": 1.0,
                    "clipping_occurred": gradient_norm > 1.0,
                    "operator_position_weight": position_weight,
                }
            )
    optimizer_state = optimizer.state_dict()
    return model, {
        "initialization_canonical_sha256": initialization,
        "trained_canonical_sha256": canonical_state_dict_sha256(model.state_dict()),
        "optimizer_canonical_sha256": canonical_torch_object_sha256(optimizer_state),
        "updates": updates,
    }


def _control_training(
    rows: list[dict[str, Any]], *, seed: int, dataset_hash: str
) -> tuple[LockedPlanner, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="a2-causal-control-") as temp:
        model, checkpoint, trace = _train_a2_with_loss_trace(
            rows, seed, Path(temp), dataset_hash
        )
    updates = []
    for row in trace:
        target_count = int(row["operator_target_count"])
        updates.append({**row, "operator_position_weight": 1.0 / target_count})
    return model, {
        "initialization_canonical_sha256": checkpoint[
            "canonical_initialization_state_dict_sha256"
        ],
        "trained_canonical_sha256": checkpoint["canonical_trained_state_dict_sha256"],
        "optimizer_canonical_sha256": checkpoint["canonical_optimizer_state_sha256"],
        "updates": updates,
    }


def _equivalence_projection(training: dict[str, Any]) -> dict[str, Any]:
    return {
        "initialization_canonical_sha256": training["initialization_canonical_sha256"],
        "trained_canonical_sha256": training["trained_canonical_sha256"],
        "optimizer_canonical_sha256": training["optimizer_canonical_sha256"],
        "updates": [
            {field: update[field] for field in EQUIVALENCE_TRACE_FIELDS}
            for update in training["updates"]
        ],
    }


def _assert_control_equivalence(
    control: dict[str, Any], probe: dict[str, Any], *, seed: int
) -> None:
    left = _equivalence_projection(control)
    right = _equivalence_projection(probe)
    if left["initialization_canonical_sha256"] != right["initialization_canonical_sha256"]:
        raise RuntimeError(f"A2_CAUSAL_CONTROL_EQUIVALENCE_INITIALIZATION:{seed}")
    if left["trained_canonical_sha256"] != right["trained_canonical_sha256"]:
        raise RuntimeError(f"A2_CAUSAL_CONTROL_EQUIVALENCE_TRAINED:{seed}")
    if left["optimizer_canonical_sha256"] != right["optimizer_canonical_sha256"]:
        raise RuntimeError(f"A2_CAUSAL_CONTROL_EQUIVALENCE_OPTIMIZER:{seed}")
    if left["updates"] != right["updates"]:
        raise RuntimeError(f"A2_CAUSAL_CONTROL_EQUIVALENCE_TRACE:{seed}")


def _control_equivalence_probe(
    control: dict[str, Any], rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    _model, probe = _train_intervention(
        rows,
        seed=seed,
        order=CANONICAL_ORDER,
        operator_loss_mode="per-task-mean-cross-entropy",
    )
    _assert_control_equivalence(control, probe, seed=seed)
    return {
        "seed": seed,
        "status": "PASS",
        "purpose": "NON_SCIENTIFIC_INTERVENTION_TRAINER_CONTROL_EQUIVALENCE",
        "task_order": list(CANONICAL_ORDER),
        "operator_loss": "per-task-mean-cross-entropy",
        "trace_fields": list(EQUIVALENCE_TRACE_FIELDS),
        "control": _equivalence_projection(control),
        "probe": _equivalence_projection(probe),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _teacher_metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [position for task in observations for position in task["positions"]]
    end_positions = [row for row in positions if row["gold_operator"] == "END"]
    non_end_positions = [row for row in positions if row["gold_operator"] != "END"]
    arg1 = [row for row in positions if row["has_arg1_target"]]
    arg2 = [row for row in positions if row["has_arg2_target"]]
    pos0_unstack = [
        row
        for row in positions
        if row["position_index"] == 0 and row["gold_operator"] == "UNSTACK"
    ]
    pos4_end = [
        row
        for row in positions
        if row["position_index"] == 4 and row["gold_operator"] == "END"
    ]
    return {
        "operator_accuracy": _rate([row["operator_correct"] for row in positions]),
        "non_end_operator_accuracy": _rate(
            [row["operator_correct"] for row in non_end_positions]
        ),
        "end_accuracy": _rate([row["operator_correct"] for row in end_positions]),
        "arg1_accuracy": _rate([bool(row["arg1_correct"]) for row in arg1]),
        "arg2_accuracy": _rate([bool(row["arg2_correct"]) for row in arg2]),
        "joint_step_accuracy": _rate([row["joint_step_correct"] for row in positions]),
        "predicted_end_rate": _rate(
            [row["predicted_operator"] == "END" for row in positions]
        ),
        "position0_unstack": {
            "target_count": len(pos0_unstack),
            "accuracy": _rate([row["operator_correct"] for row in pos0_unstack]),
            "mean_end_probability": _mean(
                [row["probability_end"] for row in pos0_unstack]
            ),
        },
        "position4_end": {
            "target_count": len(pos4_end),
            "accuracy": _rate([row["operator_correct"] for row in pos4_end]),
            "mean_end_probability": _mean(
                [row["probability_end"] for row in pos4_end]
            ),
        },
    }


def _free_metrics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(observations),
        "exact_plan_rate": _rate([row["exact_plan_match"] for row in observations]),
        "goal_success_rate": _rate([row["final_goal_success"] for row in observations]),
        "zero_action_rate": _rate(
            [row["predicted_plan_length"] == 0 for row in observations]
        ),
        "initially_unsatisfied_goal_success_rate": _rate(
            [
                row["final_goal_success"]
                for row in observations
                if not row["initial_goal_satisfied"]
            ]
        ),
    }


def _seed_arm_result(
    arm: str,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    dataset_hash: str,
) -> dict[str, Any]:
    contract = ARM_CONTRACT[arm]
    if arm == ARM_CONTROL:
        model, training = _control_training(rows, seed=seed, dataset_hash=dataset_hash)
    else:
        model, training = _train_intervention(
            rows,
            seed=seed,
            order=tuple(contract["task_order"]),
            operator_loss_mode=str(contract["operator_loss"]),
        )
    ordered_eval = sorted(rows, key=lambda item: item["task_id"])
    with _read_only_diagnostic_pass(model):
        teacher = [
            teacher_forced_task(model, row, split="train", seed=seed)
            for row in ordered_eval
        ]
        free = [
            free_running_task(model, row, split="train", seed=seed)
            for row in ordered_eval
        ]
    return {
        "arm": arm,
        "seed": seed,
        "contract": contract,
        "training": training,
        "teacher_forced": teacher,
        "teacher_forced_metrics": _teacher_metrics(teacher),
        "free_running": free,
        "free_running_metrics": _free_metrics(free),
    }


def _aggregate_arm(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    teacher = [task for result in seed_results for task in result["teacher_forced"]]
    free = [task for result in seed_results for task in result["free_running"]]
    return {
        "seed_count": len(seed_results),
        "teacher_forced": _teacher_metrics(teacher),
        "free_running": _free_metrics(free),
    }


def _metric_delta(intervention: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    def delta(left: float | None, right: float | None) -> float | None:
        return left - right if left is not None and right is not None else None

    it = intervention["teacher_forced"]
    ct = control["teacher_forced"]
    iff = intervention["free_running"]
    cff = control["free_running"]
    return {
        "position0_unstack_accuracy_delta": delta(
            it["position0_unstack"]["accuracy"], ct["position0_unstack"]["accuracy"]
        ),
        "position0_unstack_end_probability_delta": delta(
            it["position0_unstack"]["mean_end_probability"],
            ct["position0_unstack"]["mean_end_probability"],
        ),
        "position4_end_accuracy_delta": delta(
            it["position4_end"]["accuracy"], ct["position4_end"]["accuracy"]
        ),
        "operator_accuracy_delta": delta(it["operator_accuracy"], ct["operator_accuracy"]),
        "joint_step_accuracy_delta": delta(
            it["joint_step_accuracy"], ct["joint_step_accuracy"]
        ),
        "free_running_goal_success_rate_delta": delta(
            iff["goal_success_rate"], cff["goal_success_rate"]
        ),
        "free_running_initially_unsatisfied_goal_success_rate_delta": delta(
            iff["initially_unsatisfied_goal_success_rate"],
            cff["initially_unsatisfied_goal_success_rate"],
        ),
    }


def _weighting_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_positions = sum(len(row["oracle_work_plan"]) for row in rows)
    task_count = len(rows)
    lengths = {row["task_id"]: len(row["oracle_work_plan"]) for row in rows}
    canonical_pos0_end_weight = 1.0 / lengths["bw-00000001"]
    canonical_pos0_unstack_weight = (
        1.0 / lengths["bw-00000002"] + 1.0 / lengths["bw-00000003"]
    )
    equal_weight = operator_position_weight(
        valid_positions=1, total_positions=total_positions, task_count=task_count
    )
    return {
        "operator_position_count_per_epoch": total_positions,
        "task_count": task_count,
        "valid_operator_positions_by_task": lengths,
        "canonical_position0_end_weight_per_epoch": canonical_pos0_end_weight,
        "canonical_position0_unstack_weight_per_epoch": canonical_pos0_unstack_weight,
        "canonical_end_to_unstack_position0_weight_ratio": (
            canonical_pos0_end_weight / canonical_pos0_unstack_weight
        ),
        "equal_position_weight_each": equal_weight,
        "equal_position_position0_end_weight_per_epoch": equal_weight,
        "equal_position_position0_unstack_weight_per_epoch": 2 * equal_weight,
        "equal_position_end_to_unstack_position0_weight_ratio": 0.5,
        "epoch_scale_preservation": {
            "canonical_uniform_loss_operator_weight_sum": float(task_count),
            "equal_position_uniform_loss_operator_weight_sum": equal_weight * total_positions,
        },
    }


def _produce_payload(*, implementation_commit: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    if not seeds or any(seed not in SEEDS for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("A2_CAUSAL_SEEDS_INVALID")
    configure_canonical_cpu_runtime()
    dataset, rows = _train_rows()
    source = source_identity_at_commit(implementation_commit)
    results: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for seed in seeds:
        control = _seed_arm_result(
            ARM_CONTROL,
            rows,
            seed=seed,
            dataset_hash=dataset["frozen_dataset_lineage_hash"],
        )
        equivalence.append(_control_equivalence_probe(control["training"], rows, seed=seed))
        equal_position = _seed_arm_result(
            ARM_EQUAL_POSITION,
            rows,
            seed=seed,
            dataset_hash=dataset["frozen_dataset_lineage_hash"],
        )
        order_only = _seed_arm_result(
            ARM_ORDER_ONLY,
            rows,
            seed=seed,
            dataset_hash=dataset["frozen_dataset_lineage_hash"],
        )
        seed_rows = [control, equal_position, order_only]
        initializations = {
            row["training"]["initialization_canonical_sha256"] for row in seed_rows
        }
        if len(initializations) != 1:
            raise RuntimeError(f"A2_CAUSAL_INITIALIZATION_NOT_MATCHED:{seed}")
        results.extend(seed_rows)

    aggregates = {
        arm: _aggregate_arm([row for row in results if row["arm"] == arm]) for arm in ARMS
    }
    payload: dict[str, Any] = {
        "schema_version": VERSION,
        "status": STATUS,
        "implementation_commit": implementation_commit,
        **source,
        "runtime": _runtime(),
        "variant": VARIANT,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "arm_contract": ARM_CONTRACT,
        "control_equivalence": equivalence,
        "hypothesis": {
            "label": INTERPRETATION_LABEL,
            "tested": (
                "Per-task mean operator loss overweights the one-position trivial END task "
                "relative to individual positions in the two five-position nontrivial tasks."
            ),
            "task_order_alternative": (
                "The canonical first-update placement of task 01 materially drives the "
                "position-0 END pathology independently of loss normalization."
            ),
            "causal_result": None,
        },
        "heldout_accessed": False,
        "heldout_task_ids": list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS),
        "training_policy_changed": True,
        "frozen_science_changed": False,
        "go_latent": "NOT EVALUATED",
        "dataset": {
            "schema_version": dataset["schema_version"],
            "frozen_dataset_lineage_hash": FROZEN_DATASET_LINEAGE_HASH_V1,
            "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
            "dataset_lineage_order": dataset["train_task_ids"],
            "evaluated_task_ids": sorted(row["task_id"] for row in rows),
        },
        "weighting_contract": _weighting_contract(rows),
        "seed_results": results,
        "aggregates": aggregates,
        "contrasts_vs_control": {
            ARM_EQUAL_POSITION: _metric_delta(
                aggregates[ARM_EQUAL_POSITION], aggregates[ARM_CONTROL]
            ),
            ARM_ORDER_ONLY: _metric_delta(aggregates[ARM_ORDER_ONLY], aggregates[ARM_CONTROL]),
        },
        "interpretation_policy": {
            "label": INTERPRETATION_LABEL,
            "automatic_gate": None,
            "scientific_status": "REDESIGN",
        },
    }
    payload["canonical_identity"] = sha256(payload)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# A2 causal microexperiment",
        "",
        f"- Version: `{payload['schema_version']}`",
        f"- Implementation: `{payload['implementation_commit']}`",
        f"- Source: `{payload['source_sha256']}`",
        f"- Seeds: `{payload['seeds']}`",
        "- Held-out accessed: `false`",
        "- GO_LATENT: `NOT EVALUATED`",
        "- Intervention trainer control-equivalence: `PASS` for every seed",
        "",
        "## Arms",
        "",
    ]
    for arm in ARMS:
        contract = payload["arm_contract"][arm]
        lines.append(
            f"- `{arm}` — {contract['changed_dimension']}; "
            f"operator loss `{contract['operator_loss']}`; order `{contract['task_order']}`"
        )
    lines.extend(["", "## Aggregate outcome", ""])
    for arm in ARMS:
        agg = payload["aggregates"][arm]
        tf = agg["teacher_forced"]
        fr = agg["free_running"]
        lines.extend(
            [
                f"### {arm}",
                f"- position-0 UNSTACK accuracy: `{tf['position0_unstack']['accuracy']}`",
                "- position-0 UNSTACK mean END probability: "
                f"`{tf['position0_unstack']['mean_end_probability']}`",
                f"- position-4 END accuracy: `{tf['position4_end']['accuracy']}`",
                f"- teacher-forced operator accuracy: `{tf['operator_accuracy']}`",
                f"- teacher-forced joint-step accuracy: `{tf['joint_step_accuracy']}`",
                f"- arg1 accuracy: `{tf['arg1_accuracy']}`",
                f"- arg2 accuracy: `{tf['arg2_accuracy']}`",
                f"- free-running goal success: `{fr['goal_success_rate']}`",
                "- free-running initially-unsatisfied goal success: "
                f"`{fr['initially_unsatisfied_goal_success_rate']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            f"`{INTERPRETATION_LABEL}`",
            "",
            "This artifact does not make an automatic causal decision. It reports the",
            "objective-normalization and task-order contrasts for independent scientific review.",
            "A3/latent and GO_LATENT are outside this experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_payload_invariants(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != VERSION or payload.get("status") != STATUS:
        raise ValueError("A2_CAUSAL_VERSION_OR_STATUS")
    if payload.get("variant") != VARIANT or payload.get("arms") != list(ARMS):
        raise ValueError("A2_CAUSAL_ARM_CONTRACT")
    if payload.get("heldout_accessed") is not False:
        raise ValueError("A2_CAUSAL_HELDOUT_ACCESS")
    if payload.get("frozen_science_changed") is not False:
        raise ValueError("A2_CAUSAL_FROZEN_SCIENCE_FLAG")
    if payload.get("go_latent") != "NOT EVALUATED":
        raise ValueError("A2_CAUSAL_GO_LATENT_SCOPE")
    if payload.get("dataset", {}).get("evaluated_task_ids") != list(EXPECTED_TRAIN_TASK_IDS):
        raise ValueError("A2_CAUSAL_TRAIN_TASK_COVERAGE")
    if set(payload["dataset"]["evaluated_task_ids"]) & set(
        FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS
    ):
        raise ValueError("A2_CAUSAL_HELDOUT_TASK_PRESENT")
    if payload.get("arm_contract") != ARM_CONTRACT:
        raise ValueError("A2_CAUSAL_ARM_CONTRACT_MUTATED")
    identity = payload.get("canonical_identity")
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    if identity != sha256(unsigned):
        raise ValueError("A2_CAUSAL_CANONICAL_IDENTITY")


def run(
    output: Path,
    *,
    implementation_commit: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("A2_CAUSAL_OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    payload = _produce_payload(implementation_commit=implementation_commit, seeds=seeds)
    _validate_payload_invariants(payload)
    from .a2_causal_microexperiment_validator import validate_claims_from_evidence

    validate_claims_from_evidence(payload)
    (output / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (output / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def validate_microexperiment(
    output: Path, *, implementation_commit: str
) -> dict[str, Any]:
    expected_names = {OUTPUT_JSON, OUTPUT_MARKDOWN}
    if not output.is_dir() or {path.name for path in output.iterdir()} != expected_names:
        raise ValueError("A2_CAUSAL_OUTPUT_COVERAGE")
    payload = json.loads((output / OUTPUT_JSON).read_text(encoding="utf-8"))
    _validate_payload_invariants(payload)
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_CAUSAL_IMPLEMENTATION_COMMIT_MISMATCH")
    expected_source = source_identity_at_commit(implementation_commit)
    if payload["source_sha256"] != expected_source["source_sha256"]:
        raise ValueError("A2_CAUSAL_SOURCE_IDENTITY_MISMATCH")
    if payload["source_files"] != expected_source["source_files"]:
        raise ValueError("A2_CAUSAL_SOURCE_FILES_MISMATCH")
    if (output / OUTPUT_MARKDOWN).read_text(encoding="utf-8") != render_markdown(payload):
        raise ValueError("A2_CAUSAL_MARKDOWN_MISMATCH")
    from .a2_causal_microexperiment_validator import validate_claims_from_evidence

    validation = validate_claims_from_evidence(payload)
    return {
        "valid": True,
        "canonical_identity": payload["canonical_identity"],
        "source_sha256": payload["source_sha256"],
        "heldout_accessed": False,
        "arms": list(ARMS),
        "seeds": payload["seeds"],
        "automatic_gate": None,
        "control_equivalence": validation["control_equivalence"],
        "claim_validation": validation["claim_validation"],
    }
