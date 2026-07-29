"""Fail-closed, development-only toy A2 planning and lineage pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from .canonical import (
    artifact_hash,
    canonical_bytes,
    canonical_task_hash,
    goal_hash,
    sha256,
    state_hash,
)
from .dataset import generate, task_from_row
from .domain import ACTION_RANK, apply_action, goal_satisfied, validate_state
from .model import LockedA2, canonical_task_encoding
from .training import state_dict_sha256, train

ACTION_NAMES = tuple(ACTION_RANK)
ROOT = Path(__file__).parents[1]
DEV = {"seed": 17, "split": "development", "stage": "PLANNER_ONLY", "arm": "PLANNER_A2_RAW"}


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def toy_hash(kind: str, value) -> str:
    return sha256({"schema": "toy-planner-hash/1.0", "kind": kind, "value": value})


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


@dataclass
class A2Planner:
    model: LockedA2
    calls: int = 0

    def plan(self, row: dict) -> list[list[str]]:
        if self.calls:
            raise RuntimeError("replanning is forbidden")
        self.calls += 1
        action_ids, arg1_ids, arg2_ids = [4] * 17, [0] * 17, [0] * 17
        decoded: list[list[str]] = []
        encoded = canonical_task_encoding(row)
        for index in range(17):
            with torch.no_grad():
                logits = self.model(
                    encoded,
                    torch.tensor([action_ids]),
                    torch.tensor([arg1_ids]),
                    torch.tensor([arg2_ids]),
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
            action_ids[index], arg1_ids[index], arg2_ids[index] = action_id, arg1_id, arg2_id
        raise ValueError("PLAN_NO_END")


def parse_work_plan(raw: list[list[str]], blocks: list[str]) -> tuple[tuple[str, ...], ...]:
    if not raw or raw[-1] != ["END"] or any(step == ["END"] for step in raw[:-1]):
        raise ValueError("PLAN_NO_END")
    arity = {"PICK_UP": 1, "UNSTACK": 2, "PUT_DOWN": 1, "STACK": 2}
    parsed = []
    for step in raw[:-1]:
        if step[0] not in arity or len(step) != arity[step[0]] + 1:
            raise ValueError("PLAN_PARSE_ERROR")
        if any(arg not in blocks for arg in step[1:]):
            raise ValueError("PLAN_UNKNOWN_REF")
        parsed.append(tuple(step))
    return tuple(parsed)


def execute(row: dict, plan: tuple[tuple[str, ...], ...]) -> tuple[list[dict], bool]:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    events = []
    for index, action in enumerate(plan):
        before = state_hash(state)
        state = apply_action(task.blocks, state, action)
        events.append(
            {"index": index, "action": list(action), "before": before, "after": state_hash(state)}
        )
    return events, goal_satisfied(state, task.goal)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _config(row: dict, dataset: dict) -> dict:
    return {
        "schema_version": "toy-development-config/1.0",
        "variant": "A2",
        "seed": 17,
        "training": {
            "steps": 30,
            "learning_rate": 3e-4,
            "adamw_betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "architecture": {
            "d_model": 256,
            "encoder_layers": 4,
            "decoder_layers": 4,
            "attention_heads": 8,
            "ffn_dim": 1024,
        },
        "inventory_sha256": file_hash(ROOT / "docs/architecture/planner_module_inventory_v1.yaml"),
        "task_encoding_sha256": file_hash(ROOT / "docs/architecture/task_encoding_v1.yaml"),
        "dataset_hash": dataset["dataset_hash"],
        "training_task_id": row["task_id"],
        "training_task_hash": row["canonical_task_hash"],
        "runtime": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
        },
        "code_commit": _git_commit(),
        "confirmatory": False,
        "sealed_data": False,
    }


def _common(row, config_hash, checkpoint_manifest_hash, checkpoint_file_hash):
    return {
        "task_id": row["task_id"],
        "canonical_task_hash": row["canonical_task_hash"],
        "config_hash": config_hash,
        "checkpoint_manifest_hash": checkpoint_manifest_hash,
        "checkpoint_file_hash": checkpoint_file_hash,
    }


def _write(path: Path, value, *, jsonl=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes(value) if jsonl else canonical_bytes(value) + b"\n")


def validate_lineage(
    *,
    root: Path,
    task: dict,
    request: dict,
    config: dict,
    checkpoint: dict,
    work_plan: dict | None,
    manifest: dict,
    attempts: list[dict],
    episode: dict,
    evaluation: dict,
) -> None:
    """Validate files, hashes, profile constraints, and replay semantics end to end."""
    task_hash = canonical_task_hash(task)
    if request["task_id"] != task["task_id"] or request["canonical_task_hash"] != task_hash:
        raise ValueError("request task mismatch")
    if file_hash(root / "development-config.json") != request["config_hash"]:
        raise ValueError("config file mismatch")
    if json.loads((root / "development-config.json").read_bytes()) != config:
        raise ValueError("config content mismatch")
    if file_hash(root / "checkpoint-manifest.json") != request["checkpoint_manifest_hash"]:
        raise ValueError("checkpoint manifest file mismatch")
    if json.loads((root / "checkpoint-manifest.json").read_bytes()) != checkpoint:
        raise ValueError("checkpoint manifest content mismatch")
    if file_hash(root / checkpoint["model_path"]) != checkpoint["model_file_sha256"]:
        raise ValueError("checkpoint file mismatch")
    persisted_state = torch.load(
        root / checkpoint["model_path"], map_location="cpu", weights_only=True
    )
    if state_dict_sha256(persisted_state) != checkpoint["state_dict_sha256"]:
        raise ValueError("checkpoint state_dict mismatch")
    if file_hash(root / checkpoint["config_path"]) != checkpoint["config_hash"]:
        raise ValueError("checkpoint config mismatch")
    if file_hash(root / checkpoint["training_report_path"]) != checkpoint["training_report_hash"]:
        raise ValueError("training report mismatch")
    bindings = _common(
        task,
        request["config_hash"],
        request["checkpoint_manifest_hash"],
        checkpoint["model_file_sha256"],
    )
    for artifact in [manifest, episode, evaluation, *attempts]:
        for field, expected in bindings.items():
            if artifact.get(field) != expected:
                raise ValueError(f"{artifact['schema_version']} {field} mismatch")
    for artifact in (manifest, episode, evaluation, *attempts):
        for field, expected in DEV.items():
            if artifact.get(field) != expected:
                raise ValueError(f"development profile {field} mismatch")
        if artifact.get("confirmatory", False) or artifact.get("sealed_data", False):
            raise ValueError("toy artifacts cannot claim confirmatory or sealed evidence")
    if manifest["planner_call_count"] != 1 or episode["planner_calls"] != 1:
        raise ValueError("planner call count mismatch")
    if (
        episode["episode_plan_manifest_hash"] != manifest["manifest_hash"]
        or evaluation["episode_plan_manifest_hash"] != manifest["manifest_hash"]
    ):
        raise ValueError("manifest downstream binding mismatch")
    if artifact_hash(request, "request_hash") != request["request_hash"]:
        raise ValueError("request hash mismatch")
    if artifact_hash(manifest, "manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("manifest hash mismatch")
    if evaluation["replanning_count"] != 0 or any(a["replanning_observed"] for a in attempts):
        raise ValueError("replanning forbidden")
    if work_plan is None:
        if manifest["plan_status"] != "FAILED" or manifest["work_plan_path"] is not None:
            raise ValueError("failure manifest mismatch")
        if (
            manifest["work_plan_content_hash"] is not None
            or manifest["work_plan_artifact_hash"] is not None
        ):
            raise ValueError("failed generation cannot bind WorkPlan")
        if attempts or episode["attempts_total"] or episode["executed_length"]:
            raise ValueError("failure must execute zero attempts")
        if evaluation["success"] or evaluation["executed_action_count"]:
            raise ValueError("failure evaluation mismatch")
        if episode["plan_generation_status"] != "FAILED" or not manifest["failure_code"]:
            raise ValueError("typed failure missing")
        if not (
            manifest["failure_code"] == episode["terminal_error"] == evaluation["failure_code"]
        ):
            raise ValueError("failure code lineage mismatch")
    else:
        for field, expected in bindings.items():
            if work_plan.get(field) != expected:
                raise ValueError(f"WorkPlan {field} mismatch")
        content = {
            k: v
            for k, v in work_plan.items()
            if k not in {"plan_content_hash", "plan_artifact_hash"}
        }
        if toy_hash("plan_content", content) != work_plan["plan_content_hash"]:
            raise ValueError("plan content mismatch")
        artifact = {k: v for k, v in work_plan.items() if k != "plan_artifact_hash"}
        if toy_hash("plan_artifact", artifact) != work_plan["plan_artifact_hash"]:
            raise ValueError("plan artifact mismatch")
        if (
            manifest["work_plan_content_hash"] != work_plan["plan_content_hash"]
            or manifest["work_plan_artifact_hash"] != work_plan["plan_artifact_hash"]
        ):
            raise ValueError("manifest plan mismatch")
        if manifest["failure_code"] is not None or evaluation["failure_code"] is not None:
            raise ValueError("successful generation cannot carry failure")
        if not (root / manifest["work_plan_path"]).is_file():
            raise ValueError("declared WorkPlan missing")
        non_end = [step for step in work_plan["steps"] if step["action"] != "END"]
        if len(attempts) != len(non_end):
            raise ValueError("attempt count mismatch")
        state = validate_state(tuple(task["blocks"]), tuple(tuple(x) for x in task["initial"]))
        for index, (attempt, step) in enumerate(zip(attempts, non_end, strict=True)):
            if attempt["step_index"] != index or attempt["plan_position_index"] != index:
                raise ValueError("attempt order mismatch")
            if attempt["episode_plan_manifest_hash"] != manifest["manifest_hash"]:
                raise ValueError("attempt manifest mismatch")
            if (
                attempt["checkpoint_manifest_hash"] != request["checkpoint_manifest_hash"]
                or attempt["config_hash"] != request["config_hash"]
            ):
                raise ValueError("attempt model binding mismatch")
            if attempt["candidate_action"] != {"action": step["action"], "args": step["args"]}:
                raise ValueError("frozen action mismatch")
            if attempt["plan_content_hash"] != work_plan["plan_content_hash"]:
                raise ValueError("attempt plan mismatch")
            if attempt["state_before_hash"] != state_hash(state):
                raise ValueError("state before mismatch")
            state = apply_action(tuple(task["blocks"]), state, (step["action"], *step["args"]))
            if attempt["state_after_hash"] != state_hash(state):
                raise ValueError("state after mismatch")
        success = goal_satisfied(state, tuple(tuple(x) for x in task["goal"]))
        count = len(attempts)
        if (
            episode["attempts_total"],
            episode["executed_length"],
            episode["plan_positions_consumed"],
        ) != (count, count, count):
            raise ValueError("episode counts mismatch")
        if episode["final_state_hash"] != state_hash(state) or episode["goal_success"] != success:
            raise ValueError("episode outcome mismatch")
        if evaluation["success"] != success or evaluation["executed_action_count"] != count:
            raise ValueError("evaluation outcome mismatch")
        if evaluation["goal_hash"] != goal_hash(task["goal"]):
            raise ValueError("evaluation goal mismatch")
    attempt_hash = toy_hash("attempt_jsonl", attempts)
    if evaluation["attempt_log_hash"] != attempt_hash:
        raise ValueError("attempt log hash mismatch")
    if artifact_hash(evaluation, "evaluation_result_hash") != evaluation["evaluation_result_hash"]:
        raise ValueError("evaluation hash mismatch")


def run(output: Path, *, failure_mode: str | None = None, reuse_from: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    dataset = generate(17)
    row = next(r for r in dataset["train"] if len(r["oracle_work_plan"]) > 1)
    if reuse_from is None:
        trained, training = train(row, output / "model", steps=30)
        del trained
    else:
        import shutil

        shutil.copytree(reuse_from / "model", output / "model")
        training = json.loads((reuse_from / "training-summary.json").read_bytes())
    config = _config(row, dataset)
    _write(output / "development-config.json", config)
    config_hash = file_hash(output / "development-config.json")
    checkpoint_path = output / "model/trained.pt"
    checkpoint = {
        "schema_version": "toy-checkpoint-manifest/1.0",
        "model_path": "model/trained.pt",
        "model_file_sha256": file_hash(checkpoint_path),
        "state_dict_sha256": training["trained_sha256"],
        "config_path": "development-config.json",
        "config_hash": config_hash,
        "inventory_sha256": config["inventory_sha256"],
        "initialization_sha256": training["initialization_sha256"],
        "training_report_path": "model/training-report.json",
        "training_report_hash": file_hash(output / "model/training-report.json"),
        "runtime": config["runtime"],
        "confirmatory": False,
        "sealed_data": False,
    }
    _write(output / "checkpoint-manifest.json", checkpoint)
    checkpoint_manifest_hash = file_hash(output / "checkpoint-manifest.json")
    common = _common(row, config_hash, checkpoint_manifest_hash, checkpoint["model_file_sha256"])
    request = {
        "schema_version": "toy-planner-request/1.0",
        "request_id": "toy-request-0001",
        **common,
        **DEV,
        "confirmatory": False,
        "sealed_data": False,
        "request_hash": "",
    }
    request["request_hash"] = artifact_hash(request, "request_hash")
    _write(output / "planner-request.json", request)
    model = LockedA2(17)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    planner = A2Planner(model)
    work_plan = None
    failure_code = None
    try:
        raw = planner.plan(row)
        if failure_mode == "NO_END":
            raw = raw[:-1]
        parsed = parse_work_plan(raw, row["blocks"])
        steps = [
            {"step_index": i, "action": step[0], "args": step[1:]} for i, step in enumerate(raw)
        ]
        work_plan = {
            "schema_version": "toy-work-plan/1.0",
            **common,
            **DEV,
            "state_hash": state_hash(row["initial"]),
            "steps": steps,
            "plan_content_hash": "",
            "plan_artifact_hash": "",
        }
        content = {
            k: v
            for k, v in work_plan.items()
            if k not in {"plan_content_hash", "plan_artifact_hash"}
        }
        work_plan["plan_content_hash"] = toy_hash("plan_content", content)
        artifact = {k: v for k, v in work_plan.items() if k != "plan_artifact_hash"}
        work_plan["plan_artifact_hash"] = toy_hash("plan_artifact", artifact)
        events, success = execute(row, parsed)
    except (RuntimeError, ValueError) as error:
        failure_code = str(error) if str(error).startswith("PLAN_") else "PLAN_GENERATION_ERROR"
        events, success = [], False
    plan_path = "results/development/plans/work-plan.json" if work_plan else None
    manifest = {
        "schema_version": "toy-episode-plan-manifest/1.0",
        **common,
        **DEV,
        "confirmatory": False,
        "sealed_data": False,
        "plan_status": "READY" if work_plan else "FAILED",
        "planner_call_count": 1,
        "work_plan_path": plan_path,
        "work_plan_content_hash": work_plan["plan_content_hash"] if work_plan else None,
        "work_plan_artifact_hash": work_plan["plan_artifact_hash"] if work_plan else None,
        "failure_code": failure_code,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = artifact_hash(manifest, "manifest_hash")
    attempts = []
    if work_plan:
        for event in events:
            step = work_plan["steps"][event["index"]]
            attempts.append(
                {
                    "schema_version": "toy-attempt-log/1.0",
                    **common,
                    **DEV,
                    "confirmatory": False,
                    "sealed_data": False,
                    "episode_plan_manifest_hash": manifest["manifest_hash"],
                    "plan_content_hash": work_plan["plan_content_hash"],
                    "plan_artifact_hash": work_plan["plan_artifact_hash"],
                    "step_index": event["index"],
                    "plan_position_index": event["index"],
                    "candidate_action": {"action": step["action"], "args": step["args"]},
                    "state_before_hash": event["before"],
                    "state_after_hash": event["after"],
                    "replanning_observed": False,
                }
            )
    final_state = events[-1]["after"] if events else state_hash(row["initial"])
    episode = {
        "schema_version": "toy-episode-log/1.0",
        **common,
        **DEV,
        "episode_plan_manifest_hash": manifest["manifest_hash"],
        "planner_calls": 1,
        "attempts_total": len(attempts),
        "executed_length": len(attempts),
        "plan_positions_consumed": len(attempts),
        "final_state_hash": final_state,
        "goal_success": success,
        "plan_generation_status": "READY" if work_plan else "FAILED",
        "terminal_error": failure_code,
    }
    evaluation = {
        "schema_version": "toy-evaluation-result/1.0",
        **common,
        **DEV,
        "confirmatory": False,
        "sealed_data": False,
        "episode_plan_manifest_hash": manifest["manifest_hash"],
        "attempt_log_hash": toy_hash("attempt_jsonl", attempts),
        "success": success,
        "goal_hash": goal_hash(row["goal"]),
        "executed_action_count": len(attempts),
        "replanning_count": 0,
        "failure_code": failure_code,
        "evaluation_result_hash": "",
    }
    evaluation["evaluation_result_hash"] = artifact_hash(evaluation, "evaluation_result_hash")
    if work_plan:
        _write(output / plan_path, work_plan)
    validate_lineage(
        root=output,
        task=row,
        request=request,
        config=config,
        checkpoint=checkpoint,
        work_plan=work_plan,
        manifest=manifest,
        attempts=attempts,
        episode=episode,
        evaluation=evaluation,
    )
    _write(output / "episode-plan-manifest.json", manifest)
    _write(output / "attempt-log.jsonl", attempts, jsonl=True)
    _write(output / "episode-log.json", episode)
    _write(output / "evaluation-result.json", evaluation)
    _write(output / "training-summary.json", training)
    artifacts = sorted(p for p in output.rglob("*") if p.is_file())
    replay_hash = toy_hash(
        "run_files", {str(p.relative_to(output)): file_hash(p) for p in artifacts}
    )
    result = {
        "success": success,
        "planner_call_count": 1,
        "tensor_count": training["tensor_count"],
        "failure_code": failure_code,
        "replay_hash": replay_hash,
    }
    _write(output / "run-result.json", result)
    return result
