"""Fail-closed public surface for the A2 END-only learnability diagnostic v0.2.

The implementation lives in ``learnability_v0_2_impl``.  This wrapper keeps the
producer semantics unchanged while making validation independently replay and
recompute every claim-bearing v0.2 section from the sealed core-v0.1 evidence.
"""

from __future__ import annotations

import contextvars
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from . import learnability as core
from . import learnability_v0_2_impl as _impl
from .numeric_identity import canonical_state_dict_sha256, canonical_torch_object_sha256

# Re-export the implementation surface, including intentionally private helpers used by
# the diagnostic tests.  The public module remains planner_toy.learnability_v0_2.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

SOURCE_FILES = tuple(
    sorted(set(_impl.SOURCE_FILES) | {"planner_toy/learnability_v0_2_impl.py"})
)
_impl.SOURCE_FILES = SOURCE_FILES

_WEAK_VALIDATE_PAYLOAD = _impl.validate_payload
_CURRENT_CORE_ROOT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "learnability_v0_2_core_root", default=None
)
_CORE_ROOT_BY_IDENTITY: dict[str, Path] = {}

_CLAIM_FIELDS = (
    "per_update_training_observation",
    "training_trajectory_summary",
    "final_teacher_forced",
    "gold_history_projected",
    "history_mode_summary",
    "free_running",
    "free_running_aggregate",
    "checkpoint_deltas",
    "interpretation",
)


def _register_core_root(core_payload: dict[str, Any], core_root: Path) -> None:
    _CORE_ROOT_BY_IDENTITY[core_payload["canonical_identity"]] = core_root.resolve()


def _resolve_core_root(core_payload: dict[str, Any]) -> Path:
    current = _CURRENT_CORE_ROOT.get()
    if current is not None:
        return current.resolve()
    known = _CORE_ROOT_BY_IDENTITY.get(core_payload["canonical_identity"])
    if known is None:
        raise ValueError("LEARNABILITY_V0_2_SEALED_CORE_ROOT_REQUIRED")
    return known


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(torch.is_tensor(item) for item in value.values()):
        raise ValueError("LEARNABILITY_V0_2_REPLAY_STATE_DICT_INVALID")
    return value


def _assert_exact_state_dict(
    replay: dict[str, torch.Tensor], sealed: dict[str, torch.Tensor], *, seed: int
) -> None:
    if replay.keys() != sealed.keys() or any(
        not torch.equal(replay[name], sealed[name]) for name in replay
    ):
        raise ValueError(f"LEARNABILITY_V0_2_REPLAY_TRAINED_MISMATCH:{seed}")


def _replay_training_observation(
    *,
    core_root: Path,
    core_payload: dict[str, Any],
    train_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observer = _impl._TrainingTrajectoryObserver()
    replay_losses: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="a2-learnability-v0-2-validate-") as temp:
        replay_root = Path(temp)
        with observer:
            for seed in core_payload["seeds"]:
                replay_dir = replay_root / f"seed-{seed}"
                observer.begin_seed(seed, train_rows)
                try:
                    _model, _checkpoint, losses = core._train_a2_with_loss_trace(
                        train_rows,
                        seed,
                        replay_dir,
                        core_payload["frozen_dataset_lineage_hash"],
                    )
                except Exception:
                    observer._seed = None
                    observer._rows = []
                    observer._pending_gradient = None
                    raise
                observer.end_seed()
                replay_losses.extend({"seed": seed, **row} for row in losses)

                sealed_dir = core_root / "training-runs" / "A2" / f"seed-{seed}"
                replay_initialization = _load_state_dict(replay_dir / "initialization.pt")
                sealed_initialization = _load_state_dict(sealed_dir / "initialization.pt")
                if (
                    canonical_state_dict_sha256(replay_initialization)
                    != canonical_state_dict_sha256(sealed_initialization)
                ):
                    raise ValueError(
                        f"LEARNABILITY_V0_2_REPLAY_INITIALIZATION_MISMATCH:{seed}"
                    )
                replay_trained = _load_state_dict(replay_dir / "trained.pt")
                sealed_trained = _load_state_dict(sealed_dir / "trained.pt")
                _assert_exact_state_dict(replay_trained, sealed_trained, seed=seed)
                replay_optimizer = torch.load(
                    replay_dir / "optimizer-state.pt", map_location="cpu", weights_only=True
                )
                sealed_optimizer = torch.load(
                    sealed_dir / "optimizer-state.pt", map_location="cpu", weights_only=True
                )
                if (
                    canonical_torch_object_sha256(replay_optimizer)
                    != canonical_torch_object_sha256(sealed_optimizer)
                ):
                    raise ValueError(
                        f"LEARNABILITY_V0_2_REPLAY_OPTIMIZER_MISMATCH:{seed}"
                    )

    replay_losses.sort(key=lambda row: (row["seed"], row["update_index"]))
    if replay_losses != core_payload["per_update_loss_breakdown"]:
        raise ValueError("LEARNABILITY_V0_2_REPLAY_CORE_LOSS_MISMATCH")
    return _impl._join_losses(observer.records, core_payload)


def _recompute_claims(
    *, core_root: Path, core_payload: dict[str, Any]
) -> dict[str, Any]:
    core.validate_diagnostic(core_root)
    sealed_core_payload = json.loads(
        (core_root / core.OUTPUT_JSON).read_text(encoding="utf-8")
    )
    if sealed_core_payload != core_payload:
        raise ValueError("LEARNABILITY_V0_2_CORE_ROOT_PAYLOAD_MISMATCH")

    _dataset, train_rows = core._dataset_context()
    selected_ids = set(core_payload["evaluated_task_ids"])
    selected_rows = [row for row in train_rows if row["task_id"] in selected_ids]
    trajectory = _replay_training_observation(
        core_root=core_root,
        core_payload=core_payload,
        train_rows=train_rows,
    )

    final_teacher_positions: list[dict[str, Any]] = []
    free_tasks: list[dict[str, Any]] = []
    checkpoint_deltas: list[dict[str, Any]] = []
    core_free = {
        (task["seed"], task["task_id"]): task for task in core_payload["free_running"]
    }
    for seed in core_payload["seeds"]:
        model = _impl._load_trained_model(core_root, seed)
        with core._read_only_diagnostic_pass(model):
            final_teacher_positions.extend(
                _impl._teacher_positions(model, selected_rows, seed)
            )
            for row in selected_rows:
                observed = _impl._free_running_observation(model, row, seed)
                expected = core_free[(seed, row["task_id"])]
                if observed["predicted_plan"] != expected["predicted_plan"]:
                    raise ValueError("LEARNABILITY_V0_2_FREE_PLAN_CORE_MISMATCH")
                free_tasks.append(observed)
        checkpoint_deltas.append(_impl._checkpoint_delta(core_root, seed))

    final_teacher = {
        "overall": _impl._teacher_summary(final_teacher_positions),
        "by_seed": {
            str(seed): _impl._teacher_summary(
                [row for row in final_teacher_positions if row["seed"] == seed]
            )
            for seed in core_payload["seeds"]
        },
        "by_gold_position": _impl._group_teacher_by_position(final_teacher_positions),
    }
    projected = _impl._projected_gold_history(core_payload, selected_rows)
    return {
        "per_update_training_observation": trajectory,
        "training_trajectory_summary": _impl._trajectory_summary(trajectory),
        "final_teacher_forced": final_teacher,
        "gold_history_projected": projected,
        "history_mode_summary": _impl._history_mode_summary(core_payload, projected),
        "free_running": free_tasks,
        "free_running_aggregate": _impl._aggregate_free(free_tasks),
        "checkpoint_deltas": checkpoint_deltas,
        "interpretation": _impl._interpretation(core_payload, final_teacher),
    }


def validate_payload(payload: dict[str, Any], *, core_payload: dict[str, Any]) -> None:
    """Validate v0.2 claims against independently replayed sealed core evidence."""

    _WEAK_VALIDATE_PAYLOAD(payload, core_payload=core_payload)
    core_root = _resolve_core_root(core_payload)
    expected = _recompute_claims(core_root=core_root, core_payload=core_payload)
    for field in _CLAIM_FIELDS:
        if payload[field] != expected[field]:
            raise ValueError(f"LEARNABILITY_V0_2_CLAIM_EVIDENCE_MISMATCH:{field}")


def run(
    output: Path,
    *,
    implementation_commit: str,
    seeds: tuple[int, ...] = core.SEEDS,
    task_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the unchanged producer, then independently validate its v0.2 claims."""

    core_root = output / _impl.CORE_DIRECTORY
    token = _CURRENT_CORE_ROOT.set(core_root)
    original_validator = _impl.validate_payload
    _impl.validate_payload = _WEAK_VALIDATE_PAYLOAD
    try:
        payload = _impl.run(
            output,
            implementation_commit=implementation_commit,
            seeds=seeds,
            task_ids=task_ids,
        )
    finally:
        _impl.validate_payload = original_validator
        _CURRENT_CORE_ROOT.reset(token)

    core_payload = json.loads(
        (core_root / core.OUTPUT_JSON).read_text(encoding="utf-8")
    )
    _register_core_root(core_payload, core_root)
    token = _CURRENT_CORE_ROOT.set(core_root)
    try:
        validate_payload(payload, core_payload=core_payload)
    finally:
        _CURRENT_CORE_ROOT.reset(token)
    return payload


def validate_diagnostic(root: Path) -> dict[str, Any]:
    expected_top_level = {
        _impl.CORE_DIRECTORY,
        _impl.OUTPUT_JSON,
        _impl.OUTPUT_MARKDOWN,
    }
    if {path.name for path in root.iterdir()} != expected_top_level:
        raise ValueError("LEARNABILITY_V0_2_TOP_LEVEL_COVERAGE_MISMATCH")

    core_root = root / _impl.CORE_DIRECTORY
    core_result = core.validate_diagnostic(core_root)
    core_payload = json.loads(
        (core_root / core.OUTPUT_JSON).read_text(encoding="utf-8")
    )
    _register_core_root(core_payload, core_root)
    payload = json.loads((root / _impl.OUTPUT_JSON).read_text(encoding="utf-8"))
    token = _CURRENT_CORE_ROOT.set(core_root)
    try:
        validate_payload(payload, core_payload=core_payload)
    finally:
        _CURRENT_CORE_ROOT.reset(token)
    if (root / _impl.OUTPUT_MARKDOWN).read_text(encoding="utf-8") != _impl.render_markdown(
        payload
    ):
        raise ValueError("LEARNABILITY_V0_2_MARKDOWN_MISMATCH")
    return {
        "valid": True,
        "diagnostic_complete": core_result["diagnostic_complete"],
        "canonical_identity": payload["canonical_identity"],
        "core_canonical_identity": payload["core_canonical_identity"],
        "heldout_accessed": False,
        "training_policy_changed": False,
    }


# Ensure calls made from the implementation module itself resolve to the fail-closed
# validator after import, while ``run`` above deliberately uses the original structural
# validator only during production and performs the independent validation immediately after.
_impl.validate_payload = validate_payload
_impl.validate_diagnostic = validate_diagnostic
