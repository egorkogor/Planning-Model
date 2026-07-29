"""Fail-closed, single-call toy A2 planning and lineage pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .canonical import artifact_hash, canonical_bytes, sha256
from .dataset import generate, task_from_row
from .domain import ACTION_RANK, apply_action, goal_satisfied, validate_state
from .model import LockedA2, canonical_task_encoding
from .training import train

ACTION_NAMES = tuple(ACTION_RANK)


@dataclass
class A2Planner:
    model: LockedA2
    calls: int = 0

    def plan(self, row: dict) -> list[list[str]]:
        if self.calls:
            raise RuntimeError("replanning is forbidden")
        self.calls += 1
        # A complete sequence is emitted by one API invocation. Autoregressive model
        # forwards inside it are decoder computation, never external Planner calls.
        action_ids = [4] * 17
        arg1_ids = [0] * 17
        arg2_ids = [0] * 17
        decoded: list[list[str]] = []
        for index in range(17):
            probe_actions = torch.tensor([action_ids])
            probe_arg1 = torch.tensor([arg1_ids])
            probe_arg2 = torch.tensor([arg2_ids])
            with torch.no_grad():
                logits = self.model(
                    canonical_task_encoding(row), probe_actions, probe_arg1, probe_arg2
                )
            action_id = int(logits.action[0, index].argmax())
            if action_id == 4:
                return decoded + [["END"]]
            arg1_id = int(logits.arg1[0, index, : len(row["blocks"])].argmax())
            arg2_id = int(logits.arg2[0, index, : len(row["blocks"])].argmax())
            step = [ACTION_NAMES[action_id], row["blocks"][arg1_id]]
            if action_id in (1, 3):
                step.append(row["blocks"][arg2_id])
            decoded.append(step)
            action_ids[index] = action_id
            arg1_ids[index] = arg1_id
            arg2_ids[index] = arg2_id
        raise ValueError("planner did not emit END within normative limit")


def parse_work_plan(raw: list[list[str]], blocks: list[str]) -> tuple[tuple[str, ...], ...]:
    if not raw or raw[-1] != ["END"] or any(step == ["END"] for step in raw[:-1]):
        raise ValueError("plan must contain exactly one terminal END")
    result = []
    arity = {"PICK_UP": 1, "UNSTACK": 2, "PUT_DOWN": 1, "STACK": 2}
    for step in raw[:-1]:
        if step[0] not in arity or len(step) != arity[step[0]] + 1:
            raise ValueError("invalid typed action")
        if any(arg not in blocks for arg in step[1:]):
            raise ValueError("unknown object reference")
        result.append(tuple(step))
    return tuple(result)


def execute(row: dict, plan: tuple[tuple[str, ...], ...]) -> tuple[list[dict], bool]:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    attempts = []
    for index, action in enumerate(plan):
        before = sha256([list(f) for f in state])
        try:
            state = apply_action(task.blocks, state, action)
        except ValueError as exc:
            attempts.append(
                {
                    "step": index,
                    "action": list(action),
                    "before": before,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )
            return attempts, False
        attempts.append(
            {
                "step": index,
                "action": list(action),
                "before": before,
                "after": sha256([list(f) for f in state]),
                "status": "APPLIED",
            }
        )
    return attempts, goal_satisfied(state, task.goal)


def _plan_content_hash(plan: dict) -> str:
    content = {
        k: v for k, v in plan.items() if k not in {"plan_content_hash", "plan_artifact_hash"}
    }
    return sha256(content)


def validate_lineage(
    task: dict, request: dict, plan: dict, manifest: dict, attempts: dict, evaluation: dict
) -> None:
    """Recompute every content edge; never trust a stored lineage hash."""
    task_hash = sha256({k: task[k] for k in ("domain_id", "blocks", "initial", "goal")})
    if task_hash != request["canonical_task_hash"] or task_hash != plan["canonical_task_hash"]:
        raise ValueError("task lineage mismatch")
    if artifact_hash(request, "request_hash") != request["request_hash"]:
        raise ValueError("PlannerRequest content mutation")
    if plan["planner_checkpoint_sha256"] != request["planner_checkpoint_sha256"]:
        raise ValueError("checkpoint lineage mismatch")
    if plan["planner_config_sha256"] != request["planner_config_sha256"]:
        raise ValueError("config lineage mismatch")
    if _plan_content_hash(plan) != plan["plan_content_hash"]:
        raise ValueError("WorkPlan content mutation")
    if artifact_hash(plan, "plan_artifact_hash") != plan["plan_artifact_hash"]:
        raise ValueError("WorkPlan artifact mutation")
    if manifest["run_class"] != "DEVELOPMENT_TOY" or manifest["planner_seed"] != 17:
        raise ValueError("seed 17 is restricted to DEVELOPMENT_TOY")
    if manifest["work_plan_content_hash"] != plan["plan_content_hash"]:
        raise ValueError("EpisodePlanManifest content lineage mismatch")
    if manifest["work_plan_artifact_hash"] != plan["plan_artifact_hash"]:
        raise ValueError("EpisodePlanManifest artifact lineage mismatch")
    if artifact_hash(manifest, "manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("EpisodePlanManifest content mutation")
    if attempts["episode_plan_manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("AttemptLog manifest lineage mismatch")
    if attempts["plan_artifact_hash"] != plan["plan_artifact_hash"]:
        raise ValueError("AttemptLog plan lineage mismatch")
    if artifact_hash(attempts, "attempt_log_hash") != attempts["attempt_log_hash"]:
        raise ValueError("AttemptLog content mutation")
    if evaluation["attempt_log_hash"] != attempts["attempt_log_hash"]:
        raise ValueError("EvaluationResult attempt lineage mismatch")
    if artifact_hash(evaluation, "evaluation_result_hash") != evaluation["evaluation_result_hash"]:
        raise ValueError("EvaluationResult content mutation")


def run(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    dataset = generate(17)
    row = next(r for r in dataset["train"] if len(r["oracle_work_plan"]) > 1)
    model, training = train(row, output / "model", steps=50)
    planner = A2Planner(model)
    raw = planner.plan(row)
    plan = parse_work_plan(raw, row["blocks"])
    checkpoint_hash = training["trained_sha256"]
    config_hash = sha256({"variant": "A2", "seed": 17, "training_steps": 50})
    request = {
        "schema_version": "work-planner/1.21",
        "request_id": "request-toy-a2-0001",
        "run_class": "DEVELOPMENT_TOY",
        "run_id": "run-toy-a2",
        "task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "planner_variant": "A2",
        "planner_seed": 17,
        "planner_checkpoint_sha256": checkpoint_hash,
        "planner_config_sha256": config_hash,
        "request_hash": "",
    }
    request["request_hash"] = artifact_hash(request, "request_hash")
    steps = []
    for index, action in enumerate(raw):
        if len(action) == 2:
            typed_args = [{"role": "block", "ref": action[1]}]
        elif len(action) == 3:
            typed_args = [
                {"role": "moving", "ref": action[1]},
                {"role": "support", "ref": action[2]},
            ]
        else:
            typed_args = []
        steps.append(
            {
                "schema_version": "work-planner/1.21",
                "step_id": f"S{index:02d}",
                "step_index": index,
                "representation": "TYPED_ONLY",
                "planner_variant": "A2",
                "typed_action": {
                    "schema_version": "work-planner/1.21",
                    "action": action[0],
                    "args": typed_args,
                },
                "status": "VALIDATED",
                "semantic_ref": None,
                "intent_id": None,
                "semantic_signature": None,
                "semantic_similarity": None,
                "semantic_margin": None,
            }
        )
    plan_payload = {
        "schema_version": "work-planner/1.21",
        "plan_id": "plan-toy-a2-0001",
        "plan_version": 1,
        "run_id": "run-toy-a2",
        "stage": "PLANNER_ONLY",
        "task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "planner_checkpoint_sha256": checkpoint_hash,
        "planner_config_sha256": config_hash,
        "planner_seed": 17,
        "semantic_artifact_manifest_sha256": None,
        "state_hash": sha256(row["initial"]),
        "steps": steps,
        "created_at": "2026-01-01T00:00:00Z",
        "planner_variant": "A2",
        "representation": "TYPED_ONLY",
        "plan_content_hash": "",
        "plan_artifact_hash": "",
    }
    plan_payload["plan_content_hash"] = _plan_content_hash(plan_payload)
    plan_payload["plan_artifact_hash"] = artifact_hash(plan_payload, "plan_artifact_hash")
    manifest = {
        "schema_version": "work-planner/1.21",
        "run_class": "DEVELOPMENT_TOY",
        "run_id": "run-toy-a2",
        "episode_id": "episode-toy-a2-0001",
        "trajectory_id": "trajectory-toy-a2-0001",
        "stage": "PLANNER_ONLY",
        "task_id": row["task_id"],
        "base_task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "split": "development",
        "arm": "PLANNER_A2_RAW",
        "planner_seed": 17,
        "planner_checkpoint_sha256": checkpoint_hash,
        "generator_type": "MICRO_PLANNER_A2",
        "plan_status": "READY",
        "planner_call_count": planner.calls,
        "generated_before_execution": True,
        "generation_started_at": "2026-01-01T00:00:00Z",
        "generation_completed_at": "2026-01-01T00:00:00Z",
        "initial_state_hash": sha256(row["initial"]),
        "goal_hash": sha256(row["goal"]),
        "work_plan_content_hash": plan_payload["plan_content_hash"],
        "work_plan_artifact_hash": plan_payload["plan_artifact_hash"],
        "work_plan_path": "results/development/plans/work-plan.json",
        "source_episode_plan_manifest_hash": None,
        "source_arm": None,
        "semantic_signature_bank_hash": None,
        "generation_failure_code": None,
        "control_artifact_path": None,
        "control_artifact_hash": None,
        "control_status": "NOT_APPLICABLE",
        "actual_cost": {"tokens_in": 192, "tokens_out": len(raw), "latency_ms": 0},
        "attributed_cost": {"tokens_in": 192, "tokens_out": len(raw), "latency_ms": 0},
        "replay_context": None,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = artifact_hash(manifest, "manifest_hash")
    events, success = execute(row, plan)
    attempts = {
        "schema_version": "work-planner/1.21",
        "run_id": "run-toy-a2",
        "task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "planner_request_hash": request["request_hash"],
        "episode_plan_manifest_hash": manifest["manifest_hash"],
        "plan_content_hash": plan_payload["plan_content_hash"],
        "plan_artifact_hash": plan_payload["plan_artifact_hash"],
        "planner_checkpoint_sha256": checkpoint_hash,
        "planner_config_sha256": config_hash,
        "attempts": events,
        "replanning_observed": False,
        "attempt_log_hash": "",
    }
    attempts["attempt_log_hash"] = artifact_hash(attempts, "attempt_log_hash")
    evaluation = {
        "schema_version": "work-planner/1.21",
        "run_id": "run-toy-a2",
        "task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "attempt_log_hash": attempts["attempt_log_hash"],
        "success": success,
        "executed_action_count": len(events),
        "goal_hash": sha256(row["goal"]),
        "replanning_count": 0,
        "evaluation_result_hash": "",
    }
    evaluation["evaluation_result_hash"] = artifact_hash(evaluation, "evaluation_result_hash")
    validate_lineage(row, request, plan_payload, manifest, attempts, evaluation)
    artifacts = {
        "planner-request.json": request,
        "work-plan.json": plan_payload,
        "episode-plan-manifest.json": manifest,
        "attempt-log.json": attempts,
        "evaluation-result.json": evaluation,
        "training-summary.json": training,
    }
    for name, payload in artifacts.items():
        (output / name).write_bytes(canonical_bytes(payload) + b"\n")
    replay_hash = sha256(
        {name: json.loads((output / name).read_bytes()) for name in sorted(artifacts)}
    )
    result = {
        "success": success,
        "planner_call_count": planner.calls,
        "tensor_count": training["tensor_count"],
        "replay_hash": replay_hash,
    }
    (output / "run-result.json").write_bytes(canonical_bytes(result) + b"\n")
    return result
