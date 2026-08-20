"""Independent persisted-evidence validator for A2 gradient-clipping causal experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .a2_gradient_clipping_reference import reconstruct_reference
from .a2_optimization_budget_trajectory import SOURCE_FILES as BUDGET_SOURCE_FILES
from .canonical import sha256
from .canonical_runtime import canonical_cpu_runtime_fingerprint, configure_canonical_cpu_runtime
from .dataset import task_from_row
from .domain import apply_action, goal_satisfied, validate_state
from .e2e import parse_nonterminal_step
from .model import LockedPlanner
from .quality import _optimizer_named_parameters
from .train_only_dataset import generate_train_only

VERSION = "development-a2-gradient-clipping/0.1"
STATUS = "development-only-scientific-microexperiment"
TASK_ID = "a2-gradient-clipping-v1"
SEEDS = (17, 29, 43)
TASKS = ("bw-00000001", "bw-00000002", "bw-00000003")
NONTRIVIAL = ("bw-00000002", "bw-00000003")
ARMS: dict[str, float | None] = {"clip_1_0": 1.0, "clip_5_0": 5.0, "no_clip": None}
CHECKPOINTS = (3, 10, 30, 100)
PERSISTENCE_CHECKPOINTS = (10, 30, 100)
MAX_EPOCH = 100
EXPECTED_UPDATES = 300
GRADIENT_HASH_VERSION = "a2-named-gradients-exact/1.1"
GRADIENT_EVIDENCE_COMMITMENT_VERSION = "a2-gradient-evidence-commitment/1.1"
GRADIENT_ACTIVITY_STATES = ("GRAD", "NO_GRAD")
HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
ROOT = Path(__file__).parents[1]
SOURCE_FILES = tuple(
    sorted(
        set(BUDGET_SOURCE_FILES)
        | {
            ".github/workflows/reviewer-execution-bridge.yml",
            "configs/fixed-cpu-target-1.0.json",
            "docs/evaluations/A2_GRADIENT_CLIPPING_CAUSAL_SPEC_RU.md",
            "planner_toy/a2_gradient_clipping.py",
            "planner_toy/a2_gradient_clipping_reference.py",
            "planner_toy/a2_gradient_clipping_validator.py",
            "planner_toy/a2_sufficient_budget_task_order.py",
            "planner_toy/a2_sufficient_budget_task_order_validator.py",
            "scripts/run_a2_gradient_clipping.py",
            "scripts/run_reviewer_execution_bridge.py",
        }
    )
)
REFERENCE_FIELDS = (
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

CLIPPING_CONTRACT = {
    "only_intended_causal_intervention": "gradient clipping policy",
    "clip_1_0": {"primitive": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0},
    "clip_5_0": {"primitive": "torch.nn.utils.clip_grad_norm_", "max_norm": 5.0},
    "no_clip": {"primitive": None, "max_norm": None},
    "gradient_hash_version": GRADIENT_HASH_VERSION,
    "gradient_hash_semantics": {
        "domain_separator": "ASCII gradient_hash_version followed by one NUL byte",
        "parameter_order": "canonical optimizer parameter order",
        "parameter_name": "u64be UTF-8 byte length followed by UTF-8 parameter-name bytes",
        "activity_marker": "u64be ASCII byte length followed by exactly GRAD or NO_GRAD",
        "grad_payload": (
            "for GRAD only: u64be dtype ASCII length + dtype ASCII; u64be ndim; "
            "each shape dimension as u64be; u64be tensor-byte length; exact "
            "detach().cpu().contiguous().numpy().tobytes() bytes"
        ),
        "no_grad_payload": (
            "for NO_GRAD: no dtype, ndim, shape, tensor-byte length, or tensor bytes; "
            "NO_GRAD is not equivalent to a real zero-valued gradient tensor"
        ),
    },
    "gradient_activity_field": "gradient_activity",
    "gradient_activity_states": list(GRADIENT_ACTIVITY_STATES),
    "gradient_activity_semantics": (
        "ordered [{index,name,state}] aligned exactly with gradient_parameter_manifest; "
        "state is actual autograd presence before intervention and is unchanged by clipping"
    ),
    "gradient_evidence_commitment_version": GRADIENT_EVIDENCE_COMMITMENT_VERSION,
    "common_pre_intervention_norm_field": "pre_intervention_global_l2_norm",
    "common_pre_intervention_norm_semantics": (
        "read-only global L2 over the same ordered active named gradients before intervention"
    ),
    "legacy_reference_norm_field": "gradient_norm",
    "legacy_reference_norm_semantics": (
        "clip primitive return for clipped arms; retained only for accepted reference equivalence"
    ),
    "threshold_predicate_field": "threshold_exceeded",
    "threshold_predicate_semantics": (
        "pre_intervention_global_l2_norm > configured threshold; not proof of actual mutation"
    ),
    "actual_mutation_field": "gradient_mutated",
    "actual_mutation_semantics": (
        "gradient_before_sha256 != gradient_after_sha256 under the versioned named-gradient encoding"
    ),
    "actual_intervention_field": "intervention_applied",
    "actual_intervention_semantics": (
        "clipping arm and actual before/after gradient commitment difference"
    ),
}


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _source_identity_at_commit(commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("A2_CLIP_VALIDATOR_IMPLEMENTATION_FORMAT")
    if _git_bytes("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("A2_CLIP_VALIDATOR_IMPLEMENTATION_NOT_FOUND")
    if _git_bytes("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError("A2_CLIP_VALIDATOR_IMPLEMENTATION_NOT_ANCESTOR")
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        if result.returncode:
            raise ValueError(f"A2_CLIP_VALIDATOR_SOURCE_MISSING:{path}")
        files.append(
            {"path": path, "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest()}
        )
    return {"source_files": files, "source_sha256": sha256(files)}


def _fixed_target_contract_identity() -> dict[str, Any]:
    contract = json.loads((ROOT / "configs/fixed-cpu-target-1.0.json").read_text(encoding="utf-8"))
    fields = (
        "target_contract_version",
        "target_name",
        "runner_image",
        "python_implementation",
        "python_version",
        "torch_version",
        "ATEN_CPU_CAPABILITY",
        "actual_atten_cpu_capability",
        "MKL_CBWR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "torch_num_threads",
        "torch_num_interop_threads",
        "mkldnn_enabled",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "optimizer_foreach",
        "optimizer_fused",
    )
    return {"fields": {field: contract[field] for field in fields}, "sha256": sha256(contract)}


def _runtime_identity() -> dict[str, Any]:
    configure_canonical_cpu_runtime()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "canonical_cpu_runtime": canonical_cpu_runtime_fingerprint(),
        "fixed_target_contract": _fixed_target_contract_identity(),
    }


def _expected_gradient_parameter_manifest(seed: int) -> list[dict[str, Any]]:
    configure_canonical_cpu_runtime(seed)
    model = LockedPlanner(seed, "A2").cpu()
    return [
        {
            "index": index,
            "name": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
        }
        for index, (name, parameter) in enumerate(_optimizer_named_parameters(model))
    ]


def _task_counts(row: dict[str, Any]) -> tuple[int, int, int]:
    plan = row["oracle_work_plan"]
    return (
        len(plan),
        sum(step[0] != "END" for step in plan),
        sum(step[0] in {"UNSTACK", "STACK"} for step in plan),
    )


def _expected_gradient_activity(
    expected_gradient_manifest: list[dict[str, Any]],
    *,
    arg1_target_count: int,
    arg2_target_count: int,
) -> list[dict[str, Any]]:
    """Derive A2 autograd activity from the frozen objective graph, not producer evidence."""
    activity = []
    for entry in expected_gradient_manifest:
        name = entry["name"]
        if name == "heads.arg1_pointer.weight":
            state = "GRAD" if arg1_target_count > 0 else "NO_GRAD"
        elif name == "heads.arg2_pointer.weight":
            state = "GRAD" if arg2_target_count > 0 else "NO_GRAD"
        else:
            # Every other optimizer parameter lies on the always-present action-loss path.
            state = "GRAD"
        activity.append({"index": entry["index"], "name": name, "state": state})
    return activity


def _float32_total(operator: float, arg1: float | None, arg2: float | None) -> float:
    total = torch.tensor(operator, dtype=torch.float32)
    if arg1 is not None:
        total = total + torch.tensor(arg1, dtype=torch.float32)
    if arg2 is not None:
        total = total + torch.tensor(arg2, dtype=torch.float32)
    return float(total)


def _validate_gradient_activity(
    update: dict[str, Any],
    expected_gradient_manifest: list[dict[str, Any]],
    *,
    arg1_target_count: int,
    arg2_target_count: int,
    context: str,
) -> None:
    manifest_hash = sha256(expected_gradient_manifest)
    if update.get("gradient_parameter_manifest_sha256") != manifest_hash:
        raise ValueError(f"A2_CLIP_VALIDATOR_UPDATE_GRADIENT_MANIFEST_HASH:{context}")
    activity = update.get("gradient_activity")
    if not isinstance(activity, list) or len(activity) != len(expected_gradient_manifest):
        raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_LENGTH:{context}")
    for entry, manifest_entry in zip(activity, expected_gradient_manifest, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"index", "name", "state"}:
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_ENTRY:{context}")
        if entry.get("index") != manifest_entry["index"] or entry.get("name") != manifest_entry["name"]:
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_ORDER:{context}")
        if entry.get("state") not in GRADIENT_ACTIVITY_STATES:
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_STATE:{context}")
    expected_activity = _expected_gradient_activity(
        expected_gradient_manifest,
        arg1_target_count=arg1_target_count,
        arg2_target_count=arg2_target_count,
    )
    if activity != expected_activity:
        raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_SEMANTICS:{context}")
    if update.get("gradient_activity_sha256") != sha256(activity):
        raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_HASH:{context}")


def _validate_updates(
    result: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    expected_gradient_manifest: list[dict[str, Any]],
) -> None:
    arm = result.get("arm")
    seed = result.get("seed")
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("A2_CLIP_VALIDATOR_RESULT_SCOPE")
    threshold = ARMS[arm]
    if result.get("task_order") != list(TASKS):
        raise ValueError(f"A2_CLIP_VALIDATOR_TASK_ORDER:{arm}:{seed}")
    if result.get("clipping_policy") != arm or result.get("clip_threshold") != threshold:
        raise ValueError(f"A2_CLIP_VALIDATOR_POLICY_METADATA:{arm}:{seed}")
    if result.get("gradient_parameter_manifest") != expected_gradient_manifest:
        raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_PARAMETER_MANIFEST:{arm}:{seed}")
    manifest_hash = sha256(expected_gradient_manifest)
    if result.get("gradient_parameter_manifest_sha256") != manifest_hash:
        raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_PARAMETER_MANIFEST_HASH:{arm}:{seed}")
    updates = result.get("updates")
    if not isinstance(updates, list) or len(updates) != EXPECTED_UPDATES:
        raise ValueError(f"A2_CLIP_VALIDATOR_UPDATE_COUNT:{arm}:{seed}")

    for index, update in enumerate(updates):
        task_id = TASKS[index % 3]
        context = f"{arm}:{seed}:{index}"
        if (
            update.get("update_index") != index
            or update.get("epoch_index") != index // 3
            or update.get("task_id") != task_id
        ):
            raise ValueError(f"A2_CLIP_VALIDATOR_UPDATE_SCHEDULE:{context}")
        operator_count, arg1_count, arg2_count = _task_counts(rows[task_id])
        if (
            update.get("operator_target_count") != operator_count
            or update.get("arg1_target_count") != arg1_count
            or update.get("arg2_target_count") != arg2_count
            or update.get("operator_position_weight") != 1.0 / operator_count
        ):
            raise ValueError(f"A2_CLIP_VALIDATOR_TARGETS:{context}")
        _validate_gradient_activity(
            update,
            expected_gradient_manifest,
            arg1_target_count=arg1_count,
            arg2_target_count=arg2_count,
            context=context,
        )
        operator = float(update["operator_loss"])
        arg1 = update.get("arg1_pointer_loss")
        arg2 = update.get("arg2_pointer_loss")
        if not math.isfinite(operator):
            raise ValueError(f"A2_CLIP_VALIDATOR_LOSS:{context}")
        if (arg1_count == 0) != (arg1 is None):
            raise ValueError(f"A2_CLIP_VALIDATOR_ARG1:{context}")
        if (arg2_count == 0) != (arg2 is None):
            raise ValueError(f"A2_CLIP_VALIDATOR_ARG2:{context}")
        if arg1 is not None and not math.isfinite(float(arg1)):
            raise ValueError(f"A2_CLIP_VALIDATOR_ARG1:{context}")
        if arg2 is not None and not math.isfinite(float(arg2)):
            raise ValueError(f"A2_CLIP_VALIDATOR_ARG2:{context}")
        expected_total = _float32_total(
            operator,
            float(arg1) if arg1 is not None else None,
            float(arg2) if arg2 is not None else None,
        )
        if float(update["total_loss"]) != expected_total:
            raise ValueError(f"A2_CLIP_VALIDATOR_LOSS_DECOMPOSITION:{context}")
        if update.get("clipping_policy") != arm or update.get("clip_threshold") != threshold:
            raise ValueError(f"A2_CLIP_VALIDATOR_POLICY:{context}")
        if update.get("gradient_clip_norm") != threshold:
            raise ValueError(f"A2_CLIP_VALIDATOR_REFERENCE_CLIP_FIELD:{context}")

        legacy_pre = float(update["gradient_norm"])
        common_pre = float(update["pre_intervention_global_l2_norm"])
        post = float(update["post_clip_global_l2_norm"])
        if any(not math.isfinite(value) or value < 0 for value in (legacy_pre, common_pre, post)):
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_NORM:{context}")
        before = update.get("gradient_before_sha256")
        after = update.get("gradient_after_sha256")
        if (
            not isinstance(before, str)
            or not HASH.fullmatch(before)
            or not isinstance(after, str)
            or not HASH.fullmatch(after)
        ):
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_HASH:{context}")
        if update.get("gradient_hash_version") != GRADIENT_HASH_VERSION:
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_HASH_VERSION:{context}")

        expected_legacy_clipped = threshold is not None and legacy_pre > threshold
        if update.get("clipping_occurred") is not expected_legacy_clipped:
            raise ValueError(f"A2_CLIP_VALIDATOR_LEGACY_CLIPPING_FLAG:{context}")
        expected_threshold = threshold is not None and common_pre > threshold
        if update.get("threshold_exceeded") is not expected_threshold:
            raise ValueError(f"A2_CLIP_VALIDATOR_THRESHOLD_PREDICATE:{context}")
        mutated = before != after
        if update.get("gradient_mutated") is not mutated:
            raise ValueError(f"A2_CLIP_VALIDATOR_GRADIENT_MUTATION:{context}")
        intervention = threshold is not None and mutated
        if update.get("intervention_applied") is not intervention:
            raise ValueError(f"A2_CLIP_VALIDATOR_INTERVENTION_APPLIED:{context}")

        primitive_return = update.get("clip_primitive_return_norm")
        if threshold is None:
            if primitive_return is not None or legacy_pre != common_pre:
                raise ValueError(f"A2_CLIP_VALIDATOR_NO_CLIP_NORM_SEMANTICS:{seed}:{index}")
            if mutated or post != common_pre:
                raise ValueError(f"A2_CLIP_VALIDATOR_NO_CLIP_MUTATION:{seed}:{index}")
        else:
            if primitive_return is None or float(primitive_return) != legacy_pre:
                raise ValueError(f"A2_CLIP_VALIDATOR_PRIMITIVE_RETURN:{context}")
            if not math.isfinite(float(primitive_return)) or float(primitive_return) < 0:
                raise ValueError(f"A2_CLIP_VALIDATOR_PRIMITIVE_RETURN:{context}")
            if intervention and post > threshold * (1.0 + 1e-5):
                raise ValueError(f"A2_CLIP_VALIDATOR_CLIP_TRANSFORM:{context}")
            if not intervention and post != common_pre:
                raise ValueError(f"A2_CLIP_VALIDATOR_INACTIVE_CLIP_MUTATION:{context}")


def _validate_position0(item: dict[str, Any], row: dict[str, Any], context: str) -> None:
    gold = row["oracle_work_plan"][0][0]
    if item.get("task_id") != row["task_id"] or item.get("gold_operator") != gold:
        raise ValueError(f"A2_CLIP_VALIDATOR_POSITION0_GOLD:{context}")
    if item.get("operator_correct") != (item.get("predicted_operator") == gold):
        raise ValueError(f"A2_CLIP_VALIDATOR_POSITION0_CORRECT:{context}")
    probability = float(item["probability_gold_operator"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"A2_CLIP_VALIDATOR_POSITION0_PROB:{context}")
    expected_nll = -math.log(max(probability, torch.finfo(torch.float32).tiny))
    if float(item["operator_nll"]) != expected_nll:
        raise ValueError(f"A2_CLIP_VALIDATOR_POSITION0_NLL:{context}")


def _free_goal(row: dict[str, Any], predicted_plan: Any) -> tuple[bool, bool, int]:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    initial = goal_satisfied(state, task.goal)
    if not isinstance(predicted_plan, list):
        return initial, False, 0
    terminal_end = bool(predicted_plan and predicted_plan[-1] == ["END"])
    action_steps = predicted_plan[:-1] if terminal_end else predicted_plan
    for step in action_steps:
        try:
            action = parse_nonterminal_step(step, list(task.blocks))
            state = apply_action(task.blocks, state, action)
        except (ValueError, TypeError, IndexError):
            return initial, False, len(action_steps)
    success = bool(terminal_end and (action_steps or initial) and goal_satisfied(state, task.goal))
    return initial, success, len(action_steps)


def _validate_free(item: dict[str, Any], row: dict[str, Any], context: str) -> None:
    if item.get("task_id") != row["task_id"]:
        raise ValueError(f"A2_CLIP_VALIDATOR_FREE_TASK:{context}")
    initial, success, length = _free_goal(row, item.get("predicted_plan"))
    if item.get("initial_goal_satisfied") != initial or item.get("final_goal_success") != success:
        raise ValueError(f"A2_CLIP_VALIDATOR_FREE_GOAL:{context}")
    if item.get("predicted_plan_length") != length:
        raise ValueError(f"A2_CLIP_VALIDATOR_FREE_LENGTH:{context}")
    if item.get("exact_plan_match") != (item.get("predicted_plan") == row["oracle_work_plan"]):
        raise ValueError(f"A2_CLIP_VALIDATOR_FREE_EXACT:{context}")


def _validate_epochs(result: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    arm, seed = result["arm"], result["seed"]
    records = result.get("epoch_evidence")
    if not isinstance(records, list) or len(records) != MAX_EPOCH:
        raise ValueError(f"A2_CLIP_VALIDATOR_EPOCH_COUNT:{arm}:{seed}")
    for epoch, record in enumerate(records, 1):
        if record.get("epoch") != epoch or record.get("update_count") != epoch * 3:
            raise ValueError(f"A2_CLIP_VALIDATOR_EPOCH_INDEX:{arm}:{seed}:{epoch}")
        p0 = record.get("position0")
        free = record.get("free_running")
        if not isinstance(p0, list) or not isinstance(free, list) or len(p0) != 3 or len(free) != 3:
            raise ValueError(f"A2_CLIP_VALIDATOR_EPOCH_SHAPE:{arm}:{seed}:{epoch}")
        p0_by_id = {item.get("task_id"): item for item in p0}
        free_by_id = {item.get("task_id"): item for item in free}
        if (
            set(p0_by_id) != set(TASKS)
            or len(p0_by_id) != 3
            or set(free_by_id) != set(TASKS)
            or len(free_by_id) != 3
        ):
            raise ValueError(f"A2_CLIP_VALIDATOR_EPOCH_TASKS:{arm}:{seed}:{epoch}")
        for task_id in TASKS:
            context = f"{arm}:{seed}:{epoch}:{task_id}"
            _validate_position0(p0_by_id[task_id], rows[task_id], context)
            _validate_free(free_by_id[task_id], rows[task_id], context)


def _position0_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["position0"]}
    return all(
        by_id[task]["gold_operator"] == "UNSTACK"
        and bool(by_id[task]["operator_correct"])
        for task in NONTRIVIAL
    )


def _free_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["free_running"]}
    return all(
        by_id[task]["initial_goal_satisfied"] is False
        and bool(by_id[task]["final_goal_success"])
        for task in NONTRIVIAL
    )


def _first(records: list[dict[str, Any]], predicate) -> dict[str, int] | None:
    for record in records:
        if predicate(record):
            return {"epoch": int(record["epoch"]), "update_count": int(record["update_count"])}
    return None


def _persistence(
    records: list[dict[str, Any]], event: dict[str, int] | None, predicate
) -> dict[str, bool | None]:
    by_epoch = {int(record["epoch"]): record for record in records}
    return {
        str(epoch): (
            None
            if event is None or epoch < int(event["epoch"])
            else bool(predicate(by_epoch[epoch]))
        )
        for epoch in PERSISTENCE_CHECKPOINTS
    }


def _validate_outcomes(result: dict[str, Any]) -> None:
    records = result["epoch_evidence"]
    position0 = _first(records, _position0_rescued)
    free = _first(records, _free_rescued)
    expected_events = {
        "first_position0_operator_rescue": position0,
        "first_full_free_running_rescue": free,
    }
    if result.get("rescue_events") != expected_events:
        raise ValueError(f"A2_CLIP_VALIDATOR_RESCUE_EVENT:{result['arm']}:{result['seed']}")
    expected_persistence = {
        "position0_operator_rescue": _persistence(records, position0, _position0_rescued),
        "full_free_running_rescue": _persistence(records, free, _free_rescued),
    }
    if result.get("rescue_persistence") != expected_persistence:
        raise ValueError(f"A2_CLIP_VALIDATOR_PERSISTENCE:{result['arm']}:{result['seed']}")


def _validate_checkpoints(result: dict[str, Any]) -> None:
    checkpoints = result.get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or [item.get("epoch") for item in checkpoints] != list(CHECKPOINTS)
    ):
        raise ValueError(f"A2_CLIP_VALIDATOR_CHECKPOINTS:{result['arm']}:{result['seed']}")
    for checkpoint in checkpoints:
        if checkpoint.get("update_count") != int(checkpoint["epoch"]) * 3:
            raise ValueError("A2_CLIP_VALIDATOR_CHECKPOINT_UPDATE_COUNT")
        for key in ("trained_canonical_sha256", "optimizer_canonical_sha256"):
            if not isinstance(checkpoint.get(key), str) or not HASH.fullmatch(checkpoint[key]):
                raise ValueError("A2_CLIP_VALIDATOR_CHECKPOINT_HASH")
    for key in (
        "initialization_canonical_sha256",
        "final_trained_canonical_sha256",
        "final_optimizer_canonical_sha256",
    ):
        if not isinstance(result.get(key), str) or not HASH.fullmatch(result[key]):
            raise ValueError(f"A2_CLIP_VALIDATOR_STATE_HASH:{key}")
    final_checkpoint = checkpoints[-1]
    if (
        final_checkpoint["trained_canonical_sha256"] != result["final_trained_canonical_sha256"]
        or final_checkpoint["optimizer_canonical_sha256"]
        != result["final_optimizer_canonical_sha256"]
    ):
        raise ValueError(f"A2_CLIP_VALIDATOR_FINAL_CHECKPOINT:{result['arm']}:{result['seed']}")


def _control_projection(result: dict[str, Any]) -> dict[str, Any]:
    updates = []
    for update in result["updates"]:
        record = {}
        for field in REFERENCE_FIELDS:
            if field == "gradient_clip_norm":
                record[field] = update["clip_threshold"]
            else:
                record[field] = update[field]
        updates.append(record)
    return {
        "seed": int(result["seed"]),
        "initialization_canonical_sha256": result["initialization_canonical_sha256"],
        "updates": updates,
        "checkpoints": result["checkpoints"],
        "final_trained_canonical_sha256": result["final_trained_canonical_sha256"],
        "final_optimizer_canonical_sha256": result["final_optimizer_canonical_sha256"],
        "rescue_events": result["rescue_events"],
        "rescue_persistence": result["rescue_persistence"],
    }


def _control_prefix_projection(projection: dict[str, Any]) -> dict[str, Any]:
    checkpoint = next(item for item in projection["checkpoints"] if item["epoch"] == 3)
    return {
        "seed": projection["seed"],
        "initialization_canonical_sha256": projection["initialization_canonical_sha256"],
        "updates": projection["updates"][:9],
        "checkpoint_epoch_3": checkpoint,
    }


def _gradient_evidence_projection(result: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "update_index",
        "epoch_index",
        "task_id",
        "clipping_policy",
        "clip_threshold",
        "gradient_norm",
        "pre_intervention_global_l2_norm",
        "clip_primitive_return_norm",
        "threshold_exceeded",
        "gradient_mutated",
        "intervention_applied",
        "post_clip_global_l2_norm",
        "clipping_occurred",
        "gradient_before_sha256",
        "gradient_after_sha256",
        "gradient_hash_version",
        "gradient_parameter_manifest_sha256",
        "gradient_activity",
        "gradient_activity_sha256",
    )
    return [{field: update[field] for field in fields} for update in result["updates"]]


def _gradient_evidence_commitment(result: dict[str, Any]) -> dict[str, Any]:
    projection = _gradient_evidence_projection(result)
    return {
        "version": GRADIENT_EVIDENCE_COMMITMENT_VERSION,
        "gradient_hash_version": GRADIENT_HASH_VERSION,
        "gradient_parameter_manifest_sha256": result["gradient_parameter_manifest_sha256"],
        "arm": result["arm"],
        "seed": result["seed"],
        "update_count": len(projection),
        "sha256": sha256(projection),
    }


def _validate_control_equivalence(
    payload: dict[str, Any],
    indexed: dict[tuple[str, int], dict[str, Any]],
    rows: list[dict[str, Any]],
    dataset_hash: str,
) -> None:
    control = payload.get("control_equivalence")
    if not isinstance(control, dict) or control.get("required_status") != "PASS":
        raise ValueError("A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_METADATA")
    by_seed = control.get("by_seed")
    if not isinstance(by_seed, dict) or set(by_seed) != {str(seed) for seed in SEEDS}:
        raise ValueError("A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_COVERAGE")

    for seed in SEEDS:
        record = by_seed[str(seed)]
        candidate = _control_projection(indexed[("clip_1_0", seed)])
        reference = record.get("reference_projection")
        independent, frozen_prefix = reconstruct_reference(
            rows, seed=seed, dataset_hash=dataset_hash
        )
        if candidate != independent:
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_INDEPENDENT_CANDIDATE:{seed}")
        if not isinstance(reference, dict) or reference != independent:
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_INDEPENDENT_REFERENCE:{seed}")

        candidate_hash = sha256(candidate)
        reference_hash = sha256(reference)
        if record.get("status") != "PASS" or record.get("scope") != "WHOLE_300_UPDATE_TRAJECTORY":
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_STATUS:{seed}")
        if record.get("candidate_projection_sha256") != candidate_hash:
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_CANDIDATE:{seed}")
        if record.get("reference_projection_sha256") != reference_hash:
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_REFERENCE:{seed}")
        expected_prefix_hash = sha256(_control_prefix_projection(independent))
        if record.get("prefix_9_update_projection_sha256") != expected_prefix_hash:
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_PREFIX:{seed}")
        if record.get("trace_fields") != list(REFERENCE_FIELDS):
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_FIELDS:{seed}")
        if (
            record.get("candidate_checkpoints") != independent["checkpoints"]
            or record.get("reference_checkpoints") != independent["checkpoints"]
        ):
            raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_CHECKPOINTS:{seed}")
        for prefix in ("candidate", "reference"):
            if record.get(f"{prefix}_final_trained_canonical_sha256") != independent[
                "final_trained_canonical_sha256"
            ]:
                raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_MODEL:{seed}:{prefix}")
            if record.get(f"{prefix}_final_optimizer_canonical_sha256") != independent[
                "final_optimizer_canonical_sha256"
            ]:
                raise ValueError(
                    f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_OPTIMIZER:{seed}:{prefix}"
                )
            if record.get(f"{prefix}_rescue_events") != independent["rescue_events"]:
                raise ValueError(f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_RESCUE:{seed}:{prefix}")
            if record.get(f"{prefix}_rescue_persistence") != independent["rescue_persistence"]:
                raise ValueError(
                    f"A2_CLIP_VALIDATOR_CONTROL_EQUIVALENCE_PERSISTENCE:{seed}:{prefix}"
                )
        historical = record.get("reference_historical_prefix_equivalence")
        if (
            not isinstance(historical, dict)
            or historical.get("status") != "PASS"
            or historical.get("control") != frozen_prefix
            or historical.get("arm_prefix") != frozen_prefix
        ):
            raise ValueError(f"A2_CLIP_VALIDATOR_HISTORICAL_PREFIX_EQUIVALENCE:{seed}")


def _window_metrics(updates: list[dict[str, Any]]) -> dict[str, Any]:
    norms = [float(update["pre_intervention_global_l2_norm"]) for update in updates]
    interventions = [bool(update["intervention_applied"]) for update in updates]
    effects = [
        max(
            0.0,
            float(update["pre_intervention_global_l2_norm"])
            - float(update["post_clip_global_l2_norm"]),
        )
        for update in updates
    ]
    return {
        "update_count": len(updates),
        "clipped_update_count": sum(interventions),
        "clipped_update_fraction": sum(interventions) / len(updates) if updates else None,
        "max_pre_clip_global_l2_norm": max(norms) if norms else None,
        "mean_pre_clip_global_l2_norm": statistics.fmean(norms) if norms else None,
        "median_pre_clip_global_l2_norm": statistics.median(norms) if norms else None,
        "mean_clipping_effect_l2": statistics.fmean(effects) if effects else None,
        "max_clipping_effect_l2": max(effects) if effects else None,
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    events = result["rescue_events"]

    def through(event: dict[str, int] | None) -> dict[str, Any]:
        if event is None:
            return {"censored_at_update": EXPECTED_UPDATES, "event": None, "metrics": None}
        count = int(event["update_count"])
        return {
            "censored_at_update": None,
            "event": event,
            "metrics": _window_metrics(result["updates"][:count]),
        }

    return {
        "arm": result["arm"],
        "seed": result["seed"],
        "first_9_updates": _window_metrics(result["updates"][:9]),
        "through_first_position0_rescue_observation": through(
            events["first_position0_operator_rescue"]
        ),
        "through_first_full_free_running_rescue_observation": through(
            events["first_full_free_running_rescue"]
        ),
        "full_trajectory": _window_metrics(result["updates"]),
    }


def _cross_seed_clipping_aggregates(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        arm_results = sorted(
            (result for result in results if result["arm"] == arm),
            key=lambda result: int(result["seed"]),
        )
        if [int(result["seed"]) for result in arm_results] != list(SEEDS):
            raise ValueError(f"A2_CLIP_VALIDATOR_AGGREGATE_SEED_COVERAGE:{arm}")
        updates = [update for result in arm_results for update in result["updates"]]
        output[arm] = {
            "seed_count": len(arm_results),
            "total_update_count": len(updates),
            "full_trajectory": _window_metrics(updates),
            "clipped_update_count_by_seed": {
                str(result["seed"]): sum(
                    bool(update["intervention_applied"]) for update in result["updates"]
                )
                for result in arm_results
            },
        }
    return output


def _event_delta(other: dict[str, int] | None, control: dict[str, int] | None) -> int | None:
    if other is None or control is None:
        return None
    return int(other["update_count"]) - int(control["update_count"])


def _contrasts(indexed: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ("clip_5_0", "no_clip"):
        by_seed = {}
        for seed in SEEDS:
            control = indexed[("clip_1_0", seed)]
            other = indexed[(arm, seed)]
            persistence = {
                key: {
                    str(epoch): {
                        "control": control["rescue_persistence"][key][str(epoch)],
                        "intervention": other["rescue_persistence"][key][str(epoch)],
                    }
                    for epoch in PERSISTENCE_CHECKPOINTS
                }
                for key in ("position0_operator_rescue", "full_free_running_rescue")
            }
            control_count = sum(
                bool(update["intervention_applied"]) for update in control["updates"]
            )
            other_count = sum(
                bool(update["intervention_applied"]) for update in other["updates"]
            )
            by_seed[str(seed)] = {
                "delta_first_position0_rescue_update_vs_clip_1_0": _event_delta(
                    other["rescue_events"]["first_position0_operator_rescue"],
                    control["rescue_events"]["first_position0_operator_rescue"],
                ),
                "delta_first_full_free_running_rescue_update_vs_clip_1_0": _event_delta(
                    other["rescue_events"]["first_full_free_running_rescue"],
                    control["rescue_events"]["first_full_free_running_rescue"],
                ),
                "control_clipped_update_count": control_count,
                "intervention_clipped_update_count": other_count,
                "intervention_clipped_update_fraction": other_count / EXPECTED_UPDATES,
                "persistence": persistence,
                "gradient_norm_full_trajectory": _window_metrics(other["updates"]),
            }
        result[arm] = {"by_seed": by_seed}
    return result


def _pre_intervention_update_projection(update: dict[str, Any]) -> dict[str, Any]:
    return {
        key: update[key]
        for key in (
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
            "pre_intervention_global_l2_norm",
            "gradient_before_sha256",
            "gradient_hash_version",
            "gradient_parameter_manifest_sha256",
            "gradient_activity",
            "gradient_activity_sha256",
            "operator_position_weight",
        )
    }


def _validate_intervention_consistency(
    payload: dict[str, Any], indexed: dict[tuple[str, int], dict[str, Any]]
) -> None:
    actual = payload.get("intervention_consistency")
    if not isinstance(actual, dict) or set(actual) != {str(seed) for seed in SEEDS}:
        raise ValueError("A2_CLIP_VALIDATOR_INTERVENTION_CONSISTENCY_COVERAGE")
    for seed in SEEDS:
        by_arm = {arm: indexed[(arm, seed)] for arm in ARMS}
        first_intervention = None
        identical_before_first_intervention = 0
        for index in range(EXPECTED_UPDATES):
            projections = [
                _pre_intervention_update_projection(by_arm[arm]["updates"][index])
                for arm in ARMS
            ]
            if any(projection != projections[0] for projection in projections[1:]):
                raise ValueError(f"A2_CLIP_VALIDATOR_PREINTERVENTION_DIVERGENCE:{seed}:{index}")
            identical_before_first_intervention += 1
            if any(by_arm[arm]["updates"][index]["intervention_applied"] for arm in ARMS):
                first_intervention = index
                break

        clip5 = by_arm["clip_5_0"]
        no_clip = by_arm["no_clip"]
        clip5_count = sum(bool(update["intervention_applied"]) for update in clip5["updates"])
        clip5_first_intervention = None
        clip5_noclip_identical = 0
        for index in range(EXPECTED_UPDATES):
            left_pre = _pre_intervention_update_projection(clip5["updates"][index])
            right_pre = _pre_intervention_update_projection(no_clip["updates"][index])
            if left_pre != right_pre:
                raise ValueError(
                    f"A2_CLIP_VALIDATOR_CLIP5_NOCLIP_PREINTERVENTION:{seed}:{index}"
                )
            clip5_noclip_identical += 1
            if clip5["updates"][index]["intervention_applied"]:
                clip5_first_intervention = index
                break

        expected_common = {
            "first_actual_or_pregradient_difference_update_index": first_intervention,
            "all_arm_identical_preintervention_update_count": identical_before_first_intervention,
            "clip_5_0_clipped_update_count": clip5_count,
            "clip_5_0_first_actual_intervention_update_index": clip5_first_intervention,
            "clip_5_0_vs_no_clip_identical_preintervention_update_count": clip5_noclip_identical,
        }
        for key, expected in expected_common.items():
            if actual[str(seed)].get(key) != expected:
                raise ValueError(f"A2_CLIP_VALIDATOR_INTERVENTION_FIELD:{seed}:{key}")

        if clip5_count == 0:
            left = {
                "losses": [
                    (
                        update["operator_loss"],
                        update["arg1_pointer_loss"],
                        update["arg2_pointer_loss"],
                        update["total_loss"],
                        update["gradient_before_sha256"],
                    )
                    for update in clip5["updates"]
                ],
                "final_model": clip5["final_trained_canonical_sha256"],
                "final_optimizer": clip5["final_optimizer_canonical_sha256"],
                "rescue_events": clip5["rescue_events"],
            }
            right = {
                "losses": [
                    (
                        update["operator_loss"],
                        update["arg1_pointer_loss"],
                        update["arg2_pointer_loss"],
                        update["total_loss"],
                        update["gradient_before_sha256"],
                    )
                    for update in no_clip["updates"]
                ],
                "final_model": no_clip["final_trained_canonical_sha256"],
                "final_optimizer": no_clip["final_optimizer_canonical_sha256"],
                "rescue_events": no_clip["rescue_events"],
            }
            if left != right:
                raise ValueError(f"A2_CLIP_VALIDATOR_CLIP5_NOCLIP_DIVERGENCE:{seed}")
            expected = {"status": "PASS", "projection_sha256": sha256(left)}
            if actual[str(seed)].get("clip_5_0_vs_no_clip_no_effect_equivalence") != expected:
                raise ValueError(f"A2_CLIP_VALIDATOR_CLIP5_NOCLIP_EQUIVALENCE:{seed}")
        elif actual[str(seed)].get("clip_5_0_vs_no_clip_no_effect_equivalence") is not None:
            raise ValueError(f"A2_CLIP_VALIDATOR_CLIP5_NOCLIP_EQUIVALENCE_UNEXPECTED:{seed}")


def validate_claims_from_evidence(
    payload: dict[str, Any], *, implementation_commit: str
) -> dict[str, Any]:
    if (
        payload.get("experiment_version") != VERSION
        or payload.get("schema_version") != VERSION
        or payload.get("status") != STATUS
        or payload.get("task_id") != TASK_ID
    ):
        raise ValueError("A2_CLIP_VALIDATOR_VERSION")
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_CLIP_VALIDATOR_IMPLEMENTATION_MISMATCH")
    if payload.get("runtime") != _runtime_identity():
        raise ValueError("A2_CLIP_VALIDATOR_RUNTIME_IDENTITY")
    source = _source_identity_at_commit(implementation_commit)
    if (
        payload.get("source_files") != source["source_files"]
        or payload.get("source_sha256") != source["source_sha256"]
    ):
        raise ValueError("A2_CLIP_VALIDATOR_SOURCE_IDENTITY")
    if payload.get("seeds") != list(SEEDS) or payload.get("canonical_task_order") != list(TASKS):
        raise ValueError("A2_CLIP_VALIDATOR_SCOPE")
    expected_arms = {name: {"clip_threshold": threshold} for name, threshold in ARMS.items()}
    if payload.get("arms") != expected_arms:
        raise ValueError("A2_CLIP_VALIDATOR_ARMS")
    if (
        payload.get("max_epoch") != MAX_EPOCH
        or payload.get("updates_per_epoch") != 3
        or payload.get("optimizer_updates_per_seed_arm") != EXPECTED_UPDATES
    ):
        raise ValueError("A2_CLIP_VALIDATOR_BUDGET")
    if (
        payload.get("checkpoint_epochs") != list(CHECKPOINTS)
        or payload.get("persistence_checkpoint_epochs") != list(PERSISTENCE_CHECKPOINTS)
    ):
        raise ValueError("A2_CLIP_VALIDATOR_CHECKPOINT_METADATA")
    optimizer = payload.get("optimizer_contract")
    if optimizer != {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "parameter_order": "planner_toy.quality._optimizer_named_parameters",
    }:
        raise ValueError("A2_CLIP_VALIDATOR_OPTIMIZER")
    if payload.get("clipping_contract") != CLIPPING_CONTRACT:
        raise ValueError("A2_CLIP_VALIDATOR_CLIPPING_CONTRACT")
    if payload.get("heldout_accessed") is not False or payload.get("go_latent") != "NOT EVALUATED":
        raise ValueError("A2_CLIP_VALIDATOR_SCIENCE_BOUNDARY")
    if payload.get("heldout_task_ids") != ["bw-00000004", "bw-00000005"]:
        raise ValueError("A2_CLIP_VALIDATOR_HELDOUT_TASK_IDS")

    dataset = generate_train_only()
    rows_list = list(dataset["train"])
    rows = {row["task_id"]: row for row in rows_list}
    if set(rows) != set(TASKS) or len(rows) != 3:
        raise ValueError("A2_CLIP_VALIDATOR_DATASET_TASKS")
    if payload.get("dataset") != {
        "schema_version": dataset["schema_version"],
        "frozen_dataset_lineage_hash": dataset["frozen_dataset_lineage_hash"],
        "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
        "dataset_lineage_order": dataset["train_task_ids"],
        "evaluated_task_ids": list(TASKS),
    }:
        raise ValueError("A2_CLIP_VALIDATOR_DATASET_LINEAGE")
    if set(payload.get("heldout_task_ids", ())) & set(TASKS):
        raise ValueError("A2_CLIP_VALIDATOR_HELDOUT_OVERLAP")

    results = payload.get("arm_seed_results")
    if not isinstance(results, list) or len(results) != len(ARMS) * len(SEEDS):
        raise ValueError("A2_CLIP_VALIDATOR_RESULT_COUNT")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        key = (result.get("arm"), result.get("seed"))
        if key in indexed:
            raise ValueError("A2_CLIP_VALIDATOR_DUPLICATE_ARM_SEED")
        indexed[key] = result
    if set(indexed) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("A2_CLIP_VALIDATOR_ARM_SEED_COVERAGE")
    manifest_by_seed = {seed: _expected_gradient_parameter_manifest(seed) for seed in SEEDS}
    for result in results:
        _validate_updates(result, rows, manifest_by_seed[int(result["seed"])])
        _validate_epochs(result, rows)
        _validate_outcomes(result)
        _validate_checkpoints(result)
    for seed in SEEDS:
        initializations = {
            indexed[(arm, seed)]["initialization_canonical_sha256"] for arm in ARMS
        }
        if len(initializations) != 1:
            raise ValueError(f"A2_CLIP_VALIDATOR_INITIALIZATION_MISMATCH:{seed}")

    _validate_control_equivalence(
        payload,
        indexed,
        rows_list,
        dataset["frozen_dataset_lineage_hash"],
    )
    expected_commitments = [_gradient_evidence_commitment(result) for result in results]
    if payload.get("gradient_evidence_commitments") != expected_commitments:
        raise ValueError("A2_CLIP_VALIDATOR_GRADIENT_EVIDENCE_COMMITMENTS")
    expected_summaries = [_summary(result) for result in results]
    if payload.get("clipping_summaries") != expected_summaries:
        raise ValueError("A2_CLIP_VALIDATOR_CLIPPING_SUMMARIES")
    expected_aggregates = _cross_seed_clipping_aggregates(results)
    if payload.get("cross_seed_clipping_aggregates") != expected_aggregates:
        raise ValueError("A2_CLIP_VALIDATOR_CROSS_SEED_AGGREGATES")
    expected_contrasts = _contrasts(indexed)
    if payload.get("paired_causal_contrasts") != expected_contrasts:
        raise ValueError("A2_CLIP_VALIDATOR_PAIRED_CONTRASTS")
    _validate_intervention_consistency(payload, indexed)
    policy = payload.get("interpretation_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("producer_scientific_verdict") is not None
        or policy.get("validator_scientific_verdict") is not None
        or policy.get("reviewer_owns_interpretation") is not True
    ):
        raise ValueError("A2_CLIP_VALIDATOR_INTERPRETATION_POLICY")
    identity = payload.get("canonical_identity")
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    if identity != sha256(unsigned):
        raise ValueError("A2_CLIP_VALIDATOR_CANONICAL_IDENTITY")
    return {
        "valid": True,
        "validated_arm_seed_count": len(results),
        "validated_updates_per_arm_seed": EXPECTED_UPDATES,
        "control_equivalence": "PASS",
        "heldout_accessed": False,
        "source_sha256": source["source_sha256"],
        "go_latent": "NOT EVALUATED",
    }
