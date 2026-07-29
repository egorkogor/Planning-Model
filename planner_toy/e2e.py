"""Fail-closed, single-call toy A2 planning and lineage pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import torch

from .canonical import (
    artifact_hash,
    canonical_bytes,
    canonical_task_hash,
    goal_hash,
    plan_artifact_hash,
    plan_content_hash,
    sha256,
    state_hash,
)
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
        before = state_hash(state)
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
                "after": state_hash(state),
                "status": "APPLIED",
            }
        )
    return attempts, goal_satisfied(state, task.goal)


def _template(name: str) -> dict:
    return json.loads(files("planner_toy").joinpath("templates", name).read_text())


def normative_attempts(row, events, plan, manifest, checkpoint_hash, config_hash) -> list[dict]:
    records = []
    for index, event in enumerate(events):
        record = _template("attempt_log.json")
        action = event["action"]
        typed_args = (
            [{"role": "block", "ref": action[1]}]
            if len(action) == 2
            else [{"role": "moving", "ref": action[1]}, {"role": "support", "ref": action[2]}]
        )
        record.update(
            run_id="run-toy-a2",
            episode_id="episode-toy-a2-0001",
            trajectory_id="trajectory-toy-a2-0001",
            stage="PLANNER_ONLY",
            task_id=row["task_id"],
            base_task_id=row["task_id"],
            canonical_task_hash=row["canonical_task_hash"],
            split="development",
            arm="PLANNER_A2_RAW",
            planner_seed=17,
            planner_checkpoint_sha256=checkpoint_hash,
            planner_config_sha256=config_hash,
            episode_plan_manifest_hash=manifest["manifest_hash"],
            plan_generation_status="READY",
            plan_content_hash=plan["plan_content_hash"],
            plan_artifact_hash=plan["plan_artifact_hash"],
            plan_step_id=f"S{index:02d}",
            step_index=index,
            plan_position_index=index,
            state_before_hash=event["before"],
            state_after_hash=event["after"],
            goal_hash=goal_hash(row["goal"]),
            candidate_source="PLAN_REPLAY",
            candidate_typed_action={
                "schema_version": "work-planner/1.21",
                "action": action[0],
                "args": typed_args,
            },
            raw_unmasked_action=action[0],
            raw_unmasked_args=action[1:],
            parsed_typed_action=None,
            parsed_llm_response=None,
            parse_status="NOT_APPLICABLE",
            validation_status="VALID",
            replanning_observed=False,
            guidance_source_position_index=None,
            guidance_source_step_id=None,
            guidance_source_semantic_ref=None,
            tokens_in=0,
            tokens_out=0,
            queue_ms=0.0,
            inference_ms=0.0,
            total_ms=0.0,
        )
        records.append(record)
    return records


def normative_episode(row, attempts, manifest, success, final_state) -> dict:
    episode = _template("episode_log.json")
    episode.update(
        run_id="run-toy-a2",
        episode_id="episode-toy-a2-0001",
        trajectory_id="trajectory-toy-a2-0001",
        stage="PLANNER_ONLY",
        task_id=row["task_id"],
        base_task_id=row["task_id"],
        canonical_task_hash=row["canonical_task_hash"],
        split="development",
        arm="PLANNER_A2_RAW",
        planner_seed=17,
        trajectory_policy="frozen_full_plan",
        goal_success=success,
        steps_accepted=len(attempts),
        attempts_total=len(attempts),
        planner_calls=1,
        oracle_length=len(row["oracle_work_plan"]) - 1,
        executed_length=len(attempts),
        final_state_hash=final_state,
        episode_plan_manifest_hash=manifest["manifest_hash"],
        plan_generation_status="READY",
        plan_positions_consumed=len(attempts),
        total_tokens_in=0,
        total_attended_tokens=0,
        total_tokens_out=0,
        total_latency_ms=0.0,
        executor_tokens_in=0,
        executor_tokens_out=0,
        executor_latency_ms=0.0,
        plan_tokens_in_actual=192,
        plan_tokens_out_actual=len(row["oracle_work_plan"]),
        plan_latency_ms_actual=0.0,
        plan_tokens_in_attributed=192,
        plan_tokens_out_attributed=len(row["oracle_work_plan"]),
        plan_latency_ms_attributed=0.0,
    )
    return episode


def validate_lineage(
    task: dict,
    request: dict,
    plan: dict,
    manifest: dict,
    attempts: list[dict],
    episode: dict,
    evaluation: dict,
) -> None:
    """Recompute every content edge; never trust a stored lineage hash."""
    development_objects = [manifest, episode, *attempts]
    for value in development_objects:
        if value.get("planner_seed") != 17 or value.get("split") != "development":
            raise ValueError("seed 17 requires development split")
        if value.get("stage") != "PLANNER_ONLY" or value.get("arm") != "PLANNER_A2_RAW":
            raise ValueError("seed 17 requires toy A2 planner-only profile")
    if not manifest.get("work_plan_path", "").startswith("results/development/plans/"):
        raise ValueError("seed 17 requires development-only artifact path")
    task_hash = canonical_task_hash(task)
    if task_hash != request["canonical_task_hash"] or task_hash != plan["canonical_task_hash"]:
        raise ValueError("task lineage mismatch")
    if artifact_hash(request, "request_hash") != request["request_hash"]:
        raise ValueError("PlannerRequest content mutation")
    if plan["planner_checkpoint_sha256"] != request["planner_checkpoint_sha256"]:
        raise ValueError("checkpoint lineage mismatch")
    if plan["planner_config_sha256"] != request["planner_config_sha256"]:
        raise ValueError("config lineage mismatch")
    if plan_content_hash(plan) != plan["plan_content_hash"]:
        raise ValueError("WorkPlan content mutation")
    if plan_artifact_hash(plan) != plan["plan_artifact_hash"]:
        raise ValueError("WorkPlan artifact mutation")
    if manifest["planner_seed"] != 17:
        raise ValueError("development planner seed mismatch")
    if manifest["work_plan_content_hash"] != plan["plan_content_hash"]:
        raise ValueError("EpisodePlanManifest content lineage mismatch")
    if manifest["work_plan_artifact_hash"] != plan["plan_artifact_hash"]:
        raise ValueError("EpisodePlanManifest artifact lineage mismatch")
    if artifact_hash(manifest, "manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("EpisodePlanManifest content mutation")
    state = validate_state(tuple(task["blocks"]), tuple(tuple(x) for x in task["initial"]))
    non_end = [step for step in plan["steps"] if step["typed_action"]["action"] != "END"]
    if len(attempts) != len(non_end):
        raise ValueError("AttemptLog length mismatch")
    for index, (attempt, step) in enumerate(zip(attempts, non_end, strict=True)):
        if attempt["episode_plan_manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("AttemptLog manifest lineage mismatch")
        if attempt["plan_artifact_hash"] != plan["plan_artifact_hash"]:
            raise ValueError("AttemptLog plan lineage mismatch")
        expected = step["typed_action"]
        if (
            attempt["candidate_typed_action"] != expected
            or attempt["plan_step_id"] != step["step_id"]
        ):
            raise ValueError("AttemptLog frozen action mismatch")
        if attempt["state_before_hash"] != state_hash(state):
            raise ValueError("AttemptLog state-before mismatch")
        action = (expected["action"], *(arg["ref"] for arg in expected["args"]))
        state = apply_action(tuple(task["blocks"]), state, action)
        if attempt["state_after_hash"] != state_hash(state) or attempt["step_index"] != index:
            raise ValueError("AttemptLog transition mismatch")
    attempts_hash = sha256(
        {"schema": "work-planner-hash/1.0", "kind": "attempt_logs", "value": attempts}
    )
    if evaluation["attempt_log_hash"] != attempts_hash:
        raise ValueError("EvaluationResult attempt lineage mismatch")
    if episode["final_state_hash"] != state_hash(state) or episode[
        "goal_success"
    ] != goal_satisfied(state, tuple(tuple(x) for x in task["goal"])):
        raise ValueError("EpisodeLog semantic outcome mismatch")
    if episode["episode_plan_manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("EpisodeLog manifest lineage mismatch")
    if artifact_hash(evaluation, "evaluation_result_hash") != evaluation["evaluation_result_hash"]:
        raise ValueError("EvaluationResult content mutation")


def run(output: Path, *, failure_mode: str | None = None, reuse_from: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    dataset = generate(17)
    row = next(r for r in dataset["train"] if len(r["oracle_work_plan"]) > 1)
    if reuse_from is None:
        trained_model, training = train(row, output / "model", steps=30)
        del trained_model
    else:
        shutil.copytree(reuse_from / "model", output / "model")
        training = json.loads((reuse_from / "training-summary.json").read_bytes())
    checkpoint_path = output / "model/trained.pt"
    checkpoint_hash = "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != training["trained_file_sha256"]:
        raise ValueError("persisted checkpoint hash mismatch")
    model = LockedA2(17)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    config = {
        "schema_version": "toy-a2-config/1.0",
        "variant": "A2",
        "seed": 17,
        "training_steps": 30,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
    }
    config_path = output / "planner-config.json"
    config_path.write_bytes(canonical_bytes(config))
    config_hash = "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest()
    request = {
        "schema_version": "work-planner/1.21",
        "request_id": "request-toy-a2-0001",
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
    planner = A2Planner(model)
    try:
        raw = planner.plan(row)
        if failure_mode == "NO_END":
            raw = raw[:-1]
        plan = parse_work_plan(raw, row["blocks"])
    except (RuntimeError, ValueError) as error:
        code = "PLAN_NO_END" if failure_mode == "NO_END" else "PLAN_PARSE_ERROR"
        manifest = {
            "schema_version": "work-planner/1.21",
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
            "plan_status": "FAILED",
            "planner_call_count": planner.calls,
            "generated_before_execution": True,
            "generation_started_at": "2026-01-01T00:00:00Z",
            "generation_completed_at": "2026-01-01T00:00:00Z",
            "initial_state_hash": state_hash(row["initial"]),
            "goal_hash": goal_hash(row["goal"]),
            "work_plan_content_hash": None,
            "work_plan_artifact_hash": None,
            "work_plan_path": None,
            "source_episode_plan_manifest_hash": None,
            "source_arm": None,
            "semantic_signature_bank_hash": None,
            "generation_failure_code": code,
            "control_artifact_path": None,
            "control_artifact_hash": None,
            "control_status": "NOT_APPLICABLE",
            "actual_cost": {"tokens_in": 192, "tokens_out": 0, "latency_ms": 0},
            "attributed_cost": {"tokens_in": 192, "tokens_out": 0, "latency_ms": 0},
            "replay_context": None,
            "manifest_hash": "",
        }
        manifest["manifest_hash"] = artifact_hash(manifest, "manifest_hash")
        episode = normative_episode(row, [], manifest, False, state_hash(row["initial"]))
        episode.update(
            plan_generation_status="FAILED",
            terminal_error=code,
            error_tags=[code],
            planner_calls=planner.calls,
            plan_tokens_out_actual=0,
            plan_tokens_out_attributed=0,
        )
        attempts: list[dict] = []
        attempts_hash = sha256(
            {"schema": "work-planner-hash/1.0", "kind": "attempt_logs", "value": attempts}
        )
        evaluation = {
            "schema_version": "work-planner/1.21",
            "run_id": "run-toy-a2",
            "task_id": row["task_id"],
            "canonical_task_hash": row["canonical_task_hash"],
            "attempt_log_hash": attempts_hash,
            "success": False,
            "executed_action_count": 0,
            "goal_hash": goal_hash(row["goal"]),
            "replanning_count": 0,
            "failure_code": code,
            "failure_detail": str(error),
            "evaluation_result_hash": "",
        }
        evaluation["evaluation_result_hash"] = artifact_hash(evaluation, "evaluation_result_hash")
        artifacts = {
            "planner-request.json": request,
            "episode-plan-manifest.json": manifest,
            "attempt-log.json": attempts,
            "episode-log.json": episode,
            "evaluation-result.json": evaluation,
            "training-summary.json": training,
        }
        for name, payload in artifacts.items():
            (output / name).write_bytes(canonical_bytes(payload) + b"\n")
        result = {
            "success": False,
            "planner_call_count": planner.calls,
            "executed_action_count": 0,
            "failure_code": code,
            "tensor_count": training["tensor_count"],
        }
        (output / "run-result.json").write_bytes(canonical_bytes(result) + b"\n")
        return result
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
        "state_hash": state_hash(row["initial"]),
        "steps": steps,
        "created_at": "2026-01-01T00:00:00Z",
        "planner_variant": "A2",
        "representation": "TYPED_ONLY",
        "plan_content_hash": "",
        "plan_artifact_hash": "",
    }
    plan_payload["plan_content_hash"] = plan_content_hash(plan_payload)
    plan_payload["plan_artifact_hash"] = plan_artifact_hash(plan_payload)
    manifest = {
        "schema_version": "work-planner/1.21",
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
        "initial_state_hash": state_hash(row["initial"]),
        "goal_hash": goal_hash(row["goal"]),
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
    attempts = normative_attempts(row, events, plan_payload, manifest, checkpoint_hash, config_hash)
    final_state = events[-1]["after"] if events else state_hash(row["initial"])
    episode = normative_episode(row, attempts, manifest, success, final_state)
    attempts_hash = sha256(
        {"schema": "work-planner-hash/1.0", "kind": "attempt_logs", "value": attempts}
    )
    evaluation = {
        "schema_version": "work-planner/1.21",
        "run_id": "run-toy-a2",
        "task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "attempt_log_hash": attempts_hash,
        "success": success,
        "executed_action_count": len(events),
        "goal_hash": goal_hash(row["goal"]),
        "replanning_count": 0,
        "evaluation_result_hash": "",
    }
    evaluation["evaluation_result_hash"] = artifact_hash(evaluation, "evaluation_result_hash")
    validate_lineage(row, request, plan_payload, manifest, attempts, episode, evaluation)
    artifacts = {
        "planner-request.json": request,
        "results/development/plans/work-plan.json": plan_payload,
        "episode-plan-manifest.json": manifest,
        "attempt-log.json": attempts,
        "episode-log.json": episode,
        "evaluation-result.json": evaluation,
        "training-summary.json": training,
    }
    for name, payload in artifacts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(payload) + b"\n")
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
