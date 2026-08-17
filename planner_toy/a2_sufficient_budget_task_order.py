"""Development-only train-only A2 sufficient-budget task-order causal microexperiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .a2_optimization_budget_trajectory import (
    PREFIX_TRACE_FIELDS,
    _checkpoint_evidence,
    _control_training,
    _train_rows,
)
from .a2_optimization_budget_trajectory import (
    SOURCE_FILES as BUDGET_SOURCE_FILES,
)
from .canonical import canonical_bytes, sha256
from .canonical_runtime import configure_canonical_cpu_runtime
from .learnability import (
    FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS,
    SEEDS,
    _read_only_diagnostic_pass,
    free_running_task,
    teacher_forced_task,
)
from .model import LockedPlanner, canonical_task_encoding
from .numeric_identity import (
    canonical_state_dict_sha256,
    canonical_torch_object_sha256,
)
from .quality import _optimizer_named_parameters
from .training import ACTIONS, labels

VERSION = "development-a2-sufficient-budget-task-order/0.1"
STATUS = "development-only-scientific-microexperiment"
VARIANT = "A2"
EXPECTED_TRAIN_TASK_IDS = ("bw-00000001", "bw-00000002", "bw-00000003")
NONTRIVIAL_TASK_IDS = ("bw-00000002", "bw-00000003")
ARMS = {
    "canonical_order": ("bw-00000001", "bw-00000002", "bw-00000003"),
    "task01_middle": ("bw-00000002", "bw-00000001", "bw-00000003"),
    "task01_last": ("bw-00000002", "bw-00000003", "bw-00000001"),
}
CHECKPOINT_EPOCHS = (3, 10, 30, 100)
PERSISTENCE_CHECKPOINT_EPOCHS = (10, 30, 100)
MAX_EPOCH = 100
OUTPUT_JSON = "a2-sufficient-budget-task-order.json"
OUTPUT_MARKDOWN = "A2_SUFFICIENT_BUDGET_TASK_ORDER.md"
INTERPRETATION_LABEL = "SUPPORTED HYPOTHESIS / NOT PROVEN"
ROOT = Path(__file__).parents[1]
SOURCE_FILES = tuple(
    sorted(
        set(BUDGET_SOURCE_FILES)
        | {
            ".github/workflows/a2-sufficient-budget-task-order.yml",
            "docs/evaluations/A2_SUFFICIENT_BUDGET_TASK_ORDER_SPEC_RU.md",
            "planner_toy/a2_sufficient_budget_task_order.py",
            "planner_toy/a2_sufficient_budget_task_order_validator.py",
            "scripts/run_a2_sufficient_budget_task_order.py",
        }
    )
)


def _git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def _validate_commit(commit: str) -> None:
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("A2_ORDER_IMPLEMENTATION_COMMIT_FORMAT")
    if _git_bytes("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError("A2_ORDER_IMPLEMENTATION_COMMIT_NOT_FOUND")
    for path in SOURCE_FILES:
        if _git_bytes("show", f"{commit}:{path}").returncode:
            raise ValueError(f"A2_ORDER_IMPLEMENTATION_SOURCE_MISSING:{path}")


def source_identity_at_commit(commit: str) -> dict[str, Any]:
    _validate_commit(commit)
    files = []
    for path in SOURCE_FILES:
        result = _git_bytes("show", f"{commit}:{path}")
        files.append(
            {"path": path, "sha256": "sha256:" + hashlib.sha256(result.stdout).hexdigest()}
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


def _prefix_projection(
    training: dict[str, Any],
    *,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint is None:
        trained = training["trained_canonical_sha256"]
        optimizer = training["optimizer_canonical_sha256"]
    else:
        trained = checkpoint["trained_canonical_sha256"]
        optimizer = checkpoint["optimizer_canonical_sha256"]
    return {
        "initialization_canonical_sha256": training["initialization_canonical_sha256"],
        "trained_canonical_sha256": trained,
        "optimizer_canonical_sha256": optimizer,
        "updates": [
            {field: update[field] for field in PREFIX_TRACE_FIELDS}
            for update in training["updates"][:9]
        ],
    }


def _assert_canonical_prefix(
    control: dict[str, Any],
    arm_training: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    left = _prefix_projection(control)
    right = _prefix_projection(arm_training, checkpoint=checkpoint)
    if left != right:
        raise RuntimeError(f"A2_ORDER_CANONICAL_PREFIX_EQUIVALENCE:{seed}")
    return {
        "seed": seed,
        "status": "PASS",
        "purpose": "NON_SCIENTIFIC_FROZEN_3_EPOCH_PREFIX_EQUIVALENCE",
        "trace_fields": list(PREFIX_TRACE_FIELDS),
        "control": left,
        "arm_prefix": right,
    }


def _ordered_rows(rows: list[dict[str, Any]], order: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {row["task_id"]: row for row in rows}
    if set(by_id) != set(EXPECTED_TRAIN_TASK_IDS) or len(by_id) != 3:
        raise ValueError("A2_ORDER_TRAIN_TASK_COVERAGE")
    if tuple(order) not in tuple(ARMS.values()):
        raise ValueError("A2_ORDER_UNKNOWN_SCHEDULE")
    return [by_id[task_id] for task_id in order]


def _epoch_evidence(
    model: LockedPlanner,
    canonical_rows: list[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    with _read_only_diagnostic_pass(model):
        teacher = [
            teacher_forced_task(model, row, split="train", seed=seed)
            for row in canonical_rows
        ]
        free = [
            free_running_task(model, row, split="train", seed=seed)
            for row in canonical_rows
        ]
    position0 = []
    for task in teacher:
        first = task["positions"][0]
        position0.append(
            {
                "task_id": task["task_id"],
                "gold_operator": first["gold_operator"],
                "predicted_operator": first["predicted_operator"],
                "operator_correct": first["operator_correct"],
                "probability_gold_operator": first["probability_gold_operator"],
                "operator_nll": first["operator_nll"],
                "probability_end": first["probability_end"],
            }
        )
    free_light = [
        {
            "task_id": task["task_id"],
            "initial_goal_satisfied": task["initial_goal_satisfied"],
            "predicted_plan": task["predicted_plan"],
            "predicted_plan_length": task["predicted_plan_length"],
            "exact_plan_match": task["exact_plan_match"],
            "final_goal_success": task["final_goal_success"],
            "failure_code": task["failure_code"],
        }
        for task in free
    ]
    return {
        "epoch": epoch,
        "update_count": epoch * len(canonical_rows),
        "position0": position0,
        "free_running": free_light,
    }


def _position0_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["position0"]}
    return all(
        by_id[task_id]["gold_operator"] == "UNSTACK"
        and bool(by_id[task_id]["operator_correct"])
        for task_id in NONTRIVIAL_TASK_IDS
    )


def _free_rescued(record: dict[str, Any]) -> bool:
    by_id = {item["task_id"]: item for item in record["free_running"]}
    return all(
        by_id[task_id]["initial_goal_satisfied"] is False
        and bool(by_id[task_id]["final_goal_success"])
        for task_id in NONTRIVIAL_TASK_IDS
    )


def _first_event(records: list[dict[str, Any]], predicate) -> dict[str, int] | None:
    for record in records:
        if predicate(record):
            return {"epoch": int(record["epoch"]), "update_count": int(record["update_count"])}
    return None


def _persistence(
    records: list[dict[str, Any]],
    event: dict[str, int] | None,
    predicate,
) -> dict[str, bool | None]:
    by_epoch = {int(record["epoch"]): record for record in records}
    result: dict[str, bool | None] = {}
    for epoch in PERSISTENCE_CHECKPOINT_EPOCHS:
        if event is None or epoch < int(event["epoch"]):
            result[str(epoch)] = None
        else:
            result[str(epoch)] = bool(predicate(by_epoch[epoch]))
    return result


def _train_arm(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    arm: str,
    order: tuple[str, ...],
    control: dict[str, Any] | None,
    max_epoch: int = MAX_EPOCH,
    checkpoint_epochs: tuple[int, ...] = CHECKPOINT_EPOCHS,
) -> dict[str, Any]:
    configure_canonical_cpu_runtime(seed)
    canonical_rows = sorted(rows, key=lambda row: row["task_id"])
    training_rows = _ordered_rows(canonical_rows, order)
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
    updates: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    prefix_equivalence = None

    for epoch_index in range(max_epoch):
        for row in training_rows:
            action, arg1, arg2 = labels(row)
            valid = len(row["oracle_work_plan"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(canonical_task_encoding(row), action, arg1, arg2)
            flat = action[:, :valid].flatten()
            action_loss = F.cross_entropy(logits.action[:, :valid].flatten(0, 1), flat)
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
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
            )
            optimizer.step()
            updates.append(
                {
                    "update_index": len(updates),
                    "epoch_index": epoch_index,
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
                    "operator_position_weight": 1.0 / int(flat.numel()),
                }
            )

        epoch = epoch_index + 1
        epochs.append(_epoch_evidence(model, canonical_rows, seed=seed, epoch=epoch))
        if epoch in checkpoint_epochs:
            checkpoint = _checkpoint_evidence(
                model, optimizer, canonical_rows, seed=seed, epoch=epoch
            )
            checkpoints.append(checkpoint)
            if arm == "canonical_order" and epoch == 3:
                if control is None:
                    raise RuntimeError(f"A2_ORDER_CANONICAL_CONTROL_MISSING:{seed}")
                prefix_training = {
                    "initialization_canonical_sha256": initialization,
                    "updates": updates,
                }
                prefix_equivalence = _assert_canonical_prefix(
                    control, prefix_training, checkpoint, seed=seed
                )

    if max_epoch >= 3 and arm == "canonical_order" and prefix_equivalence is None:
        raise RuntimeError(f"A2_ORDER_CANONICAL_PREFIX_NOT_EVALUATED:{seed}")

    position0_event = _first_event(epochs, _position0_rescued)
    free_event = _first_event(epochs, _free_rescued)
    return {
        "arm": arm,
        "task_order": list(order),
        "seed": seed,
        "initialization_canonical_sha256": initialization,
        "final_trained_canonical_sha256": canonical_state_dict_sha256(model.state_dict()),
        "final_optimizer_canonical_sha256": canonical_torch_object_sha256(
            optimizer.state_dict()
        ),
        "updates": updates,
        "checkpoints": checkpoints,
        "epoch_evidence": epochs,
        "rescue_events": {
            "first_position0_operator_rescue": position0_event,
            "first_full_free_running_rescue": free_event,
        },
        "rescue_persistence": {
            "position0_operator_rescue": _persistence(
                epochs, position0_event, _position0_rescued
            ),
            "full_free_running_rescue": _persistence(epochs, free_event, _free_rescued),
        },
        "prefix_equivalence": prefix_equivalence,
    }


def _mean_complete(values: list[int | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None) / len(values)


def _arm_summaries(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for arm in ARMS:
        arm_results = [result for result in results if result["arm"] == arm]
        if {int(result["seed"]) for result in arm_results} != set(SEEDS):
            raise ValueError(f"A2_ORDER_ARM_SEED_COVERAGE:{arm}")
        p0_updates = [
            (
                result["rescue_events"]["first_position0_operator_rescue"]["update_count"]
                if result["rescue_events"]["first_position0_operator_rescue"] is not None
                else None
            )
            for result in arm_results
        ]
        free_updates = [
            (
                result["rescue_events"]["first_full_free_running_rescue"]["update_count"]
                if result["rescue_events"]["first_full_free_running_rescue"] is not None
                else None
            )
            for result in arm_results
        ]
        summaries[arm] = {
            "seed_count": len(arm_results),
            "task_order": list(ARMS[arm]),
            "first_position0_operator_rescue_update_by_seed": {
                str(result["seed"]): (
                    result["rescue_events"]["first_position0_operator_rescue"]["update_count"]
                    if result["rescue_events"]["first_position0_operator_rescue"] is not None
                    else None
                )
                for result in arm_results
            },
            "first_position0_operator_rescue_mean_update_if_all_seeds": _mean_complete(
                p0_updates
            ),
            "first_full_free_running_rescue_update_by_seed": {
                str(result["seed"]): (
                    result["rescue_events"]["first_full_free_running_rescue"]["update_count"]
                    if result["rescue_events"]["first_full_free_running_rescue"] is not None
                    else None
                )
                for result in arm_results
            },
            "first_full_free_running_rescue_mean_update_if_all_seeds": _mean_complete(
                free_updates
            ),
        }
    return summaries


def _cross_arm_deltas(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm_seed = {(result["arm"], int(result["seed"])): result for result in results}
    output = {}
    for arm in ("task01_middle", "task01_last"):
        by_seed = {}
        for seed in SEEDS:
            canonical = by_arm_seed[("canonical_order", seed)]["rescue_events"]
            other = by_arm_seed[(arm, seed)]["rescue_events"]
            seed_record = {}
            for key in (
                "first_position0_operator_rescue",
                "first_full_free_running_rescue",
            ):
                left = canonical[key]
                right = other[key]
                seed_record[f"{key}_update_delta_vs_canonical"] = (
                    int(right["update_count"]) - int(left["update_count"])
                    if left is not None and right is not None
                    else None
                )
            by_seed[str(seed)] = seed_record
        output[arm] = {"by_seed": by_seed}
    return output


def _produce_payload(*, implementation_commit: str) -> dict[str, Any]:
    configure_canonical_cpu_runtime()
    dataset, rows = _train_rows()
    if set(row["task_id"] for row in rows) & set(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS):
        raise ValueError("A2_ORDER_HELDOUT_ACCESS")
    source = source_identity_at_commit(implementation_commit)
    results = []
    init_by_seed: dict[int, str] = {}
    for seed in SEEDS:
        control = _control_training(
            rows, seed=seed, dataset_hash=dataset["frozen_dataset_lineage_hash"]
        )
        for arm, order in ARMS.items():
            result = _train_arm(
                rows,
                seed=seed,
                arm=arm,
                order=order,
                control=control if arm == "canonical_order" else None,
            )
            if seed in init_by_seed and init_by_seed[seed] != result[
                "initialization_canonical_sha256"
            ]:
                raise RuntimeError(f"A2_ORDER_INITIALIZATION_MISMATCH:{seed}")
            init_by_seed[seed] = result["initialization_canonical_sha256"]
            results.append(result)

    payload: dict[str, Any] = {
        "schema_version": VERSION,
        "status": STATUS,
        "implementation_commit": implementation_commit,
        **source,
        "runtime": _runtime(),
        "variant": VARIANT,
        "seeds": list(SEEDS),
        "arms": {name: list(order) for name, order in ARMS.items()},
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "max_epoch": MAX_EPOCH,
        "optimizer_updates_per_seed_arm": MAX_EPOCH * len(EXPECTED_TRAIN_TASK_IDS),
        "canonical_objective": "per-task-mean operator/arg1/arg2 cross-entropy",
        "canonical_optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "hypothesis": {
            "label": INTERPRETATION_LABEL,
            "question": (
                "Does within-epoch placement of the trivial END-only task materially control "
                "rescue timing once A2 receives a sufficient optimization budget?"
            ),
            "causal_result": None,
        },
        "rescue_definitions": {
            "first_position0_operator_rescue": (
                "earliest completed epoch where both train tasks 02 and 03 decode gold "
                "UNSTACK at position 0"
            ),
            "first_full_free_running_rescue": (
                "earliest completed epoch where both initially-unsatisfied train tasks "
                "02 and 03 are solved by true free-running execution"
            ),
        },
        "heldout_accessed": False,
        "heldout_task_ids": list(FROZEN_QUALITY_V0_1_HELDOUT_TASK_IDS),
        "training_policy_changed": True,
        "frozen_science_changed": False,
        "go_latent": "NOT EVALUATED",
        "dataset": {
            "schema_version": dataset["schema_version"],
            "frozen_dataset_lineage_hash": dataset["frozen_dataset_lineage_hash"],
            "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
            "dataset_lineage_order": dataset["train_task_ids"],
            "evaluated_task_ids": [row["task_id"] for row in rows],
        },
        "arm_seed_results": results,
        "cross_seed_arm_summaries": _arm_summaries(results),
        "cross_arm_rescue_deltas": _cross_arm_deltas(results),
        "interpretation_policy": {
            "label": INTERPRETATION_LABEL,
            "automatic_gate": None,
            "scientific_status": "REDESIGN",
            "similar_timing_supports": "BUDGET_DOMINANT_ORDER_SECONDARY",
            "large_order_shift_supports": "SEQUENTIAL_TASK_ORDER_MATERIAL_CONTRIBUTOR",
            "a3_latent_conclusion": None,
        },
    }
    payload["canonical_identity"] = sha256(payload)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# A2 sufficient-budget task-order causal discrimination",
        "",
        f"- Version: `{payload['schema_version']}`",
        f"- Implementation: `{payload['implementation_commit']}`",
        f"- Source: `{payload['source_sha256']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- Arms: `{payload['arms']}`",
        "- Budget: `100 epochs / 300 updates per seed-arm`",
        "- Canonical frozen first-9-update equivalence: `PASS` for every seed",
        "- Held-out accessed: `false`",
        "- GO_LATENT: `NOT EVALUATED`",
        "",
        "## Rescue events",
        "",
    ]
    for arm, summary in payload["cross_seed_arm_summaries"].items():
        lines.extend(
            [
                f"### {arm}",
                f"- order: `{summary['task_order']}`",
                "- first position-0 rescue by seed: "
                f"`{summary['first_position0_operator_rescue_update_by_seed']}`",
                "- first full free-running rescue by seed: "
                f"`{summary['first_full_free_running_rescue_update_by_seed']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            f"`{INTERPRETATION_LABEL}`",
            "",
            "This round tests only whether sufficient-budget rescue depends materially on",
            "within-epoch placement of task01. It does not support A3/latent/semantic claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_payload_invariants(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != VERSION or payload.get("status") != STATUS:
        raise ValueError("A2_ORDER_VERSION_OR_STATUS")
    if payload.get("variant") != VARIANT or payload.get("seeds") != list(SEEDS):
        raise ValueError("A2_ORDER_SCOPE")
    if payload.get("arms") != {name: list(order) for name, order in ARMS.items()}:
        raise ValueError("A2_ORDER_ARMS")
    if payload.get("checkpoint_epochs") != list(CHECKPOINT_EPOCHS):
        raise ValueError("A2_ORDER_CHECKPOINTS")
    if payload.get("max_epoch") != MAX_EPOCH:
        raise ValueError("A2_ORDER_MAX_EPOCH")
    if payload.get("heldout_accessed") is not False:
        raise ValueError("A2_ORDER_HELDOUT_ACCESS")
    if payload.get("go_latent") != "NOT EVALUATED":
        raise ValueError("A2_ORDER_GO_LATENT")
    if payload.get("dataset", {}).get("evaluated_task_ids") != list(
        EXPECTED_TRAIN_TASK_IDS
    ):
        raise ValueError("A2_ORDER_TRAIN_TASK_COVERAGE")
    identity = payload.get("canonical_identity")
    unsigned = {key: value for key, value in payload.items() if key != "canonical_identity"}
    if identity != sha256(unsigned):
        raise ValueError("A2_ORDER_CANONICAL_IDENTITY")


def run(output: Path, *, implementation_commit: str) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("A2_ORDER_OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    payload = _produce_payload(implementation_commit=implementation_commit)
    _validate_payload_invariants(payload)
    from .a2_sufficient_budget_task_order_validator import validate_claims_from_evidence

    validate_claims_from_evidence(payload, implementation_commit=implementation_commit)
    (output / OUTPUT_JSON).write_bytes(canonical_bytes(payload) + b"\n")
    (output / OUTPUT_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


def validate_experiment(output: Path, *, implementation_commit: str) -> dict[str, Any]:
    expected_names = {OUTPUT_JSON, OUTPUT_MARKDOWN}
    if not output.is_dir() or {path.name for path in output.iterdir()} != expected_names:
        raise ValueError("A2_ORDER_OUTPUT_COVERAGE")
    payload = json.loads((output / OUTPUT_JSON).read_text(encoding="utf-8"))
    _validate_payload_invariants(payload)
    if payload.get("implementation_commit") != implementation_commit:
        raise ValueError("A2_ORDER_IMPLEMENTATION_COMMIT_MISMATCH")
    expected_source = source_identity_at_commit(implementation_commit)
    if payload.get("source_sha256") != expected_source["source_sha256"]:
        raise ValueError("A2_ORDER_SOURCE_IDENTITY_MISMATCH")
    if payload.get("source_files") != expected_source["source_files"]:
        raise ValueError("A2_ORDER_SOURCE_FILES_MISMATCH")
    if (output / OUTPUT_MARKDOWN).read_text(encoding="utf-8") != render_markdown(payload):
        raise ValueError("A2_ORDER_MARKDOWN_MISMATCH")

    from .a2_sufficient_budget_task_order_validator import validate_claims_from_evidence

    validation = validate_claims_from_evidence(
        payload, implementation_commit=implementation_commit
    )
    return {
        "valid": True,
        "canonical_identity": payload["canonical_identity"],
        "source_sha256": payload["source_sha256"],
        "heldout_accessed": False,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "automatic_gate": None,
        **validation,
    }
