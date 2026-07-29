"""Fail-closed, single-call toy A2 planning and lineage pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .canonical import canonical_bytes, sha256
from .dataset import generate, task_from_row
from .domain import ACTION_RANK, apply_action, goal_satisfied, validate_state
from .model import LockedA2, canonical_token_ids
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
        action_ids: list[int] = []
        arg1_ids: list[int] = []
        arg2_ids: list[int] = []
        decoded: list[list[str]] = []
        for _ in range(17):
            probe_actions = torch.tensor([action_ids + [4]])
            probe_arg1 = torch.tensor([arg1_ids + [0]])
            probe_arg2 = torch.tensor([arg2_ids + [0]])
            with torch.no_grad():
                logits = self.model(canonical_token_ids(row), probe_actions, probe_arg1, probe_arg2)
            action_id = int(logits.action[0, -1].argmax())
            if action_id == 4:
                return decoded + [["END"]]
            arg1_id = int(logits.arg1[0, -1, : len(row["blocks"])].argmax())
            arg2_id = int(logits.arg2[0, -1, : len(row["blocks"])].argmax())
            step = [ACTION_NAMES[action_id], row["blocks"][arg1_id]]
            if action_id in (1, 3):
                step.append(row["blocks"][arg2_id])
            decoded.append(step)
            action_ids.append(action_id)
            arg1_ids.append(arg1_id)
            arg2_ids.append(arg2_id)
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


def validate_lineage(manifest: dict, attempts: dict, evaluation: dict) -> None:
    if attempts["plan_sha256"] != manifest["work_plan_sha256"]:
        raise ValueError("AttemptLog plan lineage mismatch")
    if evaluation["attempt_log_sha256"] != sha256(attempts):
        raise ValueError("EvaluationResult attempt lineage mismatch")
    if manifest["run_class"] != "DEVELOPMENT/TOY" or manifest["seed"] != 17:
        raise ValueError("seed 17 is restricted to DEVELOPMENT/TOY")


def run(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    dataset = generate(17)
    row = next(
        r for r in dataset["train"] + dataset["validation"] if r["oracle_work_plan"] == [["END"]]
    )
    model, training = train(row, output / "model")
    planner = A2Planner(model)
    raw = planner.plan(row)
    plan = parse_work_plan(raw, row["blocks"])
    plan_payload = {
        "schema_version": "toy-work-plan/1.0",
        "task_id": row["task_id"],
        "steps": raw,
        "frozen": True,
    }
    manifest = {
        "schema_version": "toy-episode-plan-manifest/1.0",
        "run_class": "DEVELOPMENT/TOY",
        "confirmatory": False,
        "seed": 17,
        "task_hash": row["canonical_task_hash"],
        "work_plan_sha256": sha256(plan_payload),
        "planner_call_count": planner.calls,
    }
    events, success = execute(row, plan)
    attempts = {
        "schema_version": "toy-attempt-log/1.0",
        "task_id": row["task_id"],
        "plan_sha256": manifest["work_plan_sha256"],
        "attempts": events,
    }
    evaluation = {
        "schema_version": "toy-evaluation-result/1.0",
        "success": success,
        "attempt_log_sha256": sha256(attempts),
        "replanning_count": 0,
    }
    validate_lineage(manifest, attempts, evaluation)
    artifacts = {
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
