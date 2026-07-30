"""Fail-closed, development-only toy A2 planning and lineage pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import struct
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
from .model import LockedPlanner, canonical_task_encoding
from .training import optimizer_state_sha256, state_dict_sha256, train

ACTION_NAMES = tuple(ACTION_RANK)
ROOT = Path(__file__).parents[1]
DEV = {"seed": 17, "split": "development", "stage": "PLANNER_ONLY", "arm": "PLANNER_A2_RAW"}
FAILURE_CODES = frozenset(
    {
        "PLAN_NO_END",
        "PLAN_PARSE_ERROR",
        "PLAN_UNKNOWN_REF",
        "PLAN_GENERATION_ERROR",
        "EXECUTOR_PRECONDITION_FAILED",
        "GOAL_NOT_ACHIEVED",
        "LATENT_NONFINITE",
        "LATENT_ZERO_NORM",
        "LATENT_DIMENSION",
        "SEMANTIC_TRACE_MISSING",
        "SEMANTIC_TRACE_LENGTH",
        "VARIANT_MISMATCH",
    }
)


class PlannerGenerationFailure(ValueError):
    """Typed generation failure retaining inference evidence before termination."""

    def __init__(self, code: str, partial_raw_output: list[list[str]], model_forward_count: int):
        super().__init__(code)
        self.code = code
        self.partial_raw_output = partial_raw_output
        self.model_forward_count = model_forward_count
PRESERVED_GENERATION_CODES = FAILURE_CODES - {
    "EXECUTOR_PRECONDITION_FAILED",
    "GOAL_NOT_ACHIEVED",
    "PLAN_GENERATION_ERROR",
}


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def toy_hash(kind: str, value) -> str:
    return sha256({"schema": "toy-planner-hash/1.0", "kind": kind, "value": value})


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


@dataclass
class A2Planner:
    model: LockedPlanner
    calls: int = 0
    semantic_steps: list[torch.Tensor] | None = None
    injected_failure: str | None = None
    model_forward_count: int = 0
    semantic_audit: list[dict] | None = None

    def plan(self, row: dict) -> list[list[str]]:
        if self.calls:
            raise RuntimeError("replanning is forbidden")
        self.calls += 1
        self.model_forward_count = 0
        if self.injected_failure in {"LATENT_NONFINITE", "LATENT_ZERO_NORM", "LATENT_DIMENSION"}:
            raise ValueError(self.injected_failure)
        action_ids, arg1_ids, arg2_ids = [4] * 17, [0] * 17, [0] * 17
        decoded: list[list[str]] = []
        self.semantic_steps = []
        self.semantic_audit = []
        feedback = torch.zeros(1, 17, 384)
        previous_z_hash = None
        def tensor_hash(tensor):
            return "sha256:" + hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        encoded = canonical_task_encoding(row)
        for index in range(17):
            with torch.no_grad():
                self.model_forward_count += 1
                logits = self.model(
                    encoded,
                    torch.tensor([action_ids]),
                    torch.tensor([arg1_ids]),
                    torch.tensor([arg2_ids]),
                    semantic_feedback=feedback if self.model.variant in {"A3", "A4"} else None,
                )
            if logits.z_semantic is not None:
                z = logits.z_semantic[0, index].detach().clone()
                input_feedback = feedback[0, index].detach().cpu().contiguous()
                projected = logits.projected_semantic[0, index].detach().cpu().contiguous()
                downstream = logits.semantic_component[0, index].detach().cpu().contiguous()
                self.semantic_steps.append(z)
                if index + 1 < 17:
                    feedback[0, index + 1] = z
                self.semantic_audit.append({
                    "step_index": index,
                    "input_feedback_sha256": tensor_hash(input_feedback),
                    "expected_previous_z_sha256": previous_z_hash,
                    "projected_feedback_sha256": tensor_hash(projected),
                    "downstream_semantic_component_sha256": tensor_hash(downstream),
                    "input_feedback_norm": float(input_feedback.norm()),
                    "projected_feedback_norm": float(projected.norm()),
                    "downstream_semantic_component_norm": float(downstream.norm()),
                    "source": "BOS_ZERO" if index == 0 else "PREVIOUS_PREDICTED_LATENT",
                    "projected_feedback_present": (
                        getattr(logits, "projected_semantic", None) is not None
                    ),
                    "downstream_component_zero": bool(
                        torch.all(
                            getattr(logits, "semantic_component", torch.zeros(1, 17, 1))[
                                0, index
                            ]
                            == 0
                        )
                    ),
                })
                previous_z_hash = tensor_hash(z.detach().cpu().contiguous())
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
        raise PlannerGenerationFailure("PLAN_NO_END", decoded, self.model_forward_count)


def evaluate_frozen_plan(
    *, row: dict, planner: A2Planner, output: Path, checkpoint_binding: dict
) -> dict:
    """Run one development inference through a persisted frozen-plan evidence chain.

    Gold plan information is deliberately removed before the single Planner call.
    The resulting WorkPlan is written before any action is given to the executor.
    """
    if output.exists() and any(output.iterdir()):
        raise ValueError("evidence output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    inference_row = {key: value for key, value in row.items() if key != "oracle_work_plan"}
    request = {
        "schema_version": "toy-quality-planner-request/1.0",
        "variant": planner.model.variant,
        "task_id": row["task_id"],
        "task_hash": row["canonical_task_hash"],
        "checkpoint_state_dict_sha256": checkpoint_binding["trained_state_dict_sha256"],
        "oracle_fields_removed": ["oracle_work_plan"],
        "teacher_forcing": False,
        "target_latent": False,
        "max_decoding_steps": 17,
    }
    _write(output / "planner-request.json", request)
    raw = []
    generation_failure = None
    try:
        raw = planner.plan(inference_row)
        parsed = parse_work_plan(raw, row["blocks"])
    except PlannerGenerationFailure as error:
        raw = error.partial_raw_output
        parsed = ()
        generation_failure = error.code
    except ValueError as error:
        parsed = ()
        generation_failure = str(error)
    terminal_end = bool(raw and raw[-1] == ["END"])
    work_plan = None
    if generation_failure is None:
        work_plan = {
            "schema_version": "toy-work-plan/1.0",
            "variant": planner.model.variant,
            "task_id": row["task_id"],
            "state_hash": state_hash(row["initial"]),
            "steps": [
                {"step_index": index, "action": step[0], "args": step[1:]}
                for index, step in enumerate(raw)
            ],
        }
        work_plan["plan_content_hash"] = toy_hash("quality_frozen_plan", work_plan)
        _write(output / "work-plan.json", work_plan)  # freeze before execution
    events, success, execution_failure = ([], False, None)
    if work_plan is not None:
        events, success, execution_failure = execute(row, parsed)
    attempts = [
        {
            "step_index": event["index"], "candidate_action": event["action"],
            "state_before_hash": event["before"], "state_after_hash": event["after"],
            "status": event["status"], "error": event["error"], "replanning_observed": False,
        }
        for event in events
    ]
    _write(output / "attempt-log.jsonl", attempts, jsonl=True)
    failure = generation_failure or execution_failure
    if work_plan is not None and not success and failure is None:
        failure = "GOAL_NOT_ACHIEVED"
    manifest = {
        "schema_version": "toy-quality-episode-plan-manifest/1.0",
        "planner_call_count": planner.calls, "replanning_count": 0,
        "model_forward_count": planner.model_forward_count,
        "plan_status": "READY" if work_plan else "FAILED",
        "work_plan_path": "work-plan.json" if work_plan else None,
        "work_plan_hash": work_plan["plan_content_hash"] if work_plan else None,
        "failure_code": generation_failure,
    }
    _write(output / "episode-plan-manifest.json", manifest)
    final_hash = events[-1]["after"] if events else state_hash(row["initial"])
    episode = {
        "schema_version": "toy-quality-episode-log/1.0", "planner_calls": planner.calls,
        "replanning_count": 0, "attempts_total": len(attempts),
        "executed_length": sum(a["status"] == "APPLIED" for a in attempts),
        "final_state_hash": final_hash, "goal_success": success, "terminal_error": failure,
    }
    _write(output / "episode-log.json", episode)
    evaluation = {
        "schema_version": "toy-quality-evaluation-result/1.0", "success": success,
        "failure_code": failure, "planner_call_count": planner.calls,
        "replanning_count": 0, "model_forward_count": planner.model_forward_count,
        "attempt_log_hash": file_hash(output / "attempt-log.jsonl"),
        "episode_log_hash": file_hash(output / "episode-log.json"),
    }
    _write(output / "evaluation-result.json", evaluation)
    semantic_trace = None
    if planner.model.variant in {"A3", "A4"}:
        import struct

        latents = planner.semantic_steps or []
        latent_path = output / "semantic-latents.f32"
        latent_path.write_bytes(b"".join(struct.pack("<384f", *z.tolist()) for z in latents))
        previous = None
        trace_steps = []
        for index, z in enumerate(latents):
            payload = struct.pack("<384f", *z.tolist())
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            action = raw[index] if index < len(raw) else ["END"]
            trace_steps.append({"step_index": index, "action": action[0], "args": action[1:],
                                "z_sha256": digest, "previous_z_sha256": previous,
                                "latent_norm": float(z.norm()), "source": "predicted"})
            previous = digest
        trace = {
            "schema_version": "toy-semantic-trace/1.0",
            "variant": planner.model.variant, "feedback_source": "predicted",
            "feedback_applied": planner.model.variant == "A3",
            "compute_then_zero": planner.model.variant == "A4",
            "checkpoint_file_hash": checkpoint_binding["trained_file_sha256"],
            "planner_request_hash": file_hash(output / "planner-request.json"),
            "work_plan_artifact_hash": work_plan["plan_content_hash"] if work_plan else None,
            "latent_path": "semantic-latents.f32", "latent_file_sha256": file_hash(latent_path),
            "steps": trace_steps,
            "control_audit": planner.semantic_audit,
        }
        _write(output / "semantic-trace.json", trace)
        semantic_trace = trace
    validate_frozen_plan_lineage_core(
        variant=planner.model.variant, planner_calls=planner.calls, replanning_count=0,
        work_plan=work_plan, attempts=attempts, evaluation=evaluation,
        semantic_trace=semantic_trace,
    )
    evidence_files = sorted(path for path in output.iterdir() if path.is_file())
    evidence_hash = toy_hash(
        "quality_evidence",
        {path.name: file_hash(path) for path in evidence_files},
    )
    return {
        "raw_output": raw, "parsed_actions": [list(action) for action in parsed],
        "terminal_end": terminal_end, "generation_failure": generation_failure,
        "execution_failure": execution_failure, "failure_code": failure,
        "events": events, "goal_reached": success, "planner_call_count": planner.calls,
        "replanning_count": 0, "model_forward_count": planner.model_forward_count,
        "work_plan_hash": work_plan["plan_content_hash"] if work_plan else None,
        "evidence_hash": evidence_hash,
    }


def parse_work_plan(raw: list[list[str]], blocks: list[str]) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("PLAN_PARSE_ERROR")
    if any(
        not isinstance(step, list) or not step or any(not isinstance(value, str) for value in step)
        for step in raw
    ):
        raise ValueError("PLAN_PARSE_ERROR")
    if raw[-1] != ["END"] or any(step == ["END"] for step in raw[:-1]):
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


def execute(row: dict, plan: tuple[tuple[str, ...], ...]) -> tuple[list[dict], bool, str | None]:
    task = task_from_row(row)
    state = validate_state(task.blocks, task.initial)
    events = []
    for index, action in enumerate(plan):
        before = state_hash(state)
        try:
            state = apply_action(task.blocks, state, action)
        except ValueError as error:
            events.append(
                {
                    "index": index,
                    "action": list(action),
                    "before": before,
                    "after": before,
                    "status": "FAILED",
                    "error": str(error),
                }
            )
            return events, False, "EXECUTOR_PRECONDITION_FAILED"
        events.append(
            {
                "index": index,
                "action": list(action),
                "before": before,
                "after": state_hash(state),
                "status": "APPLIED",
                "error": None,
            }
        )
    return events, goal_satisfied(state, task.goal), None


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _config(row: dict, dataset: dict, *, variant: str = "A2") -> dict:
    if variant not in {"A2", "A3", "A4"}:
        raise ValueError("VARIANT_UNSUPPORTED")
    return {
        "schema_version": "toy-development-config/1.0",
        "variant": variant,
        **DEV,
        "training": {
            "steps": 30,
            "learning_rate": 3e-4,
            "adamw_betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "semantic_loss_weight": 1.0 if variant in {"A3", "A4"} else 0.0,
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


def validate_frozen_plan_lineage_core(
    *, variant: str, planner_calls: int, replanning_count: int,
    work_plan: dict | None, attempts: list[dict], evaluation: dict,
    semantic_trace: dict | None,
) -> None:
    """Shared profile-independent invariants for legacy and quality evidence."""
    if planner_calls != 1:
        raise ValueError("planner call count mismatch")
    if replanning_count != 0 or any(a.get("replanning_observed") for a in attempts):
        raise ValueError("replanning forbidden")
    if work_plan is not None:
        steps = work_plan.get("steps", [])
        indices = [step.get("step_index") for step in steps]
        if indices != list(range(len(steps))):
            raise ValueError("WorkPlan step indices mismatch")
        if not steps or steps[-1].get("action") != "END":
            raise ValueError("WorkPlan terminal END mismatch")
    if variant == "A2" and semantic_trace is not None:
        raise ValueError("A2_SEMANTIC_ARTIFACT_FORBIDDEN")
    if variant in {"A3", "A4"} and (work_plan is not None or semantic_trace is not None):
        if semantic_trace is None:
            raise ValueError("SEMANTIC_TRACE_MISSING")
        if semantic_trace.get("feedback_source") != "predicted":
            raise ValueError("INFERENCE_FEEDBACK_NOT_PREDICTED")
        if variant == "A3" and semantic_trace.get("feedback_applied") is not True:
            raise ValueError("A3_INFERENCE_FEEDBACK_NOT_APPLIED")
        if variant == "A4" and semantic_trace.get("feedback_applied") is not False:
            raise ValueError("A4_FEEDBACK_APPLIED")
    if evaluation.get("replanning_count", 0) != 0:
        raise ValueError("evaluation replanning mismatch")


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
    semantic_trace: dict | None = None,
) -> None:
    """Validate files, hashes, profile constraints, and replay semantics end to end."""
    validate_frozen_plan_lineage_core(
        variant=config.get("variant", "A2"), planner_calls=episode["planner_calls"],
        replanning_count=evaluation["replanning_count"], work_plan=work_plan,
        attempts=attempts, evaluation=evaluation, semantic_trace=semantic_trace,
    )
    task_hash = canonical_task_hash(task)
    variant = config.get("variant", "A2")
    for artifact in (request, checkpoint, work_plan, manifest, episode, evaluation, *attempts):
        if artifact is not None and artifact.get("variant", "A2") != variant:
            raise ValueError("VARIANT_MISMATCH")
    if variant == "A2" and semantic_trace is not None:
        raise ValueError("A2_SEMANTIC_ARTIFACT_FORBIDDEN")
    if variant in {"A3", "A4"} and work_plan is not None:
        if semantic_trace is None:
            raise ValueError("SEMANTIC_TRACE_MISSING")
        if semantic_trace.get("variant") != variant:
            raise ValueError("VARIANT_MISMATCH")
        if variant == "A3" and semantic_trace.get("feedback_source") != "predicted":
            raise ValueError("A3_INFERENCE_FEEDBACK_NOT_PREDICTED")
        if variant == "A4" and semantic_trace.get("feedback_applied") is not False:
            raise ValueError("A4_FEEDBACK_APPLIED")
        if len(semantic_trace.get("steps", [])) != len(work_plan["steps"]):
            raise ValueError("SEMANTIC_TRACE_LENGTH")
        if [row.get("step_index") for row in semantic_trace["steps"]] != list(
            range(len(work_plan["steps"]))
        ):
            raise ValueError("SEMANTIC_TRACE_ORDER")
        if semantic_trace.get("config_hash") != request["config_hash"]:
            raise ValueError("SEMANTIC_TRACE_CONFIG_MISMATCH")
        if semantic_trace.get("work_plan_artifact_hash") != work_plan["plan_artifact_hash"]:
            raise ValueError("SEMANTIC_TRACE_PLAN_MISMATCH")
        if semantic_trace.get("checkpoint_manifest_hash") != request["checkpoint_manifest_hash"]:
            raise ValueError("SEMANTIC_TRACE_CHECKPOINT_MISMATCH")
        if semantic_trace.get("checkpoint_file_hash") != checkpoint["model_file_sha256"]:
            raise ValueError("SEMANTIC_TRACE_CHECKPOINT_MISMATCH")
        if semantic_trace.get("planner_request_hash") != request["request_hash"]:
            raise ValueError("SEMANTIC_TRACE_REQUEST_MISMATCH")
        latent_path = root / semantic_trace.get("latent_path", "")
        if not latent_path.is_file() or file_hash(latent_path) != semantic_trace.get(
            "latent_file_sha256"
        ):
            raise ValueError("SEMANTIC_LATENT_FILE_MISMATCH")
        if latent_path.stat().st_size != len(work_plan["steps"]) * 384 * 4:
            raise ValueError("LATENT_DIMENSION")
        payload = latent_path.read_bytes()
        previous = None
        for index, (trace_step, plan_step) in enumerate(
            zip(semantic_trace["steps"], work_plan["steps"], strict=True)
        ):
            if (trace_step.get("action"), trace_step.get("args")) != (
                plan_step["action"],
                plan_step["args"],
            ) or trace_step.get("previous_z_sha256") != previous:
                raise ValueError("SEMANTIC_TRACE_STEP_MISMATCH")
            if not (0.999 <= trace_step.get("latent_norm", 0) <= 1.001):
                raise ValueError("LATENT_NORM")
            chunk = payload[index * 1536 : (index + 1) * 1536]
            if "sha256:" + hashlib.sha256(chunk).hexdigest() != trace_step.get("z_sha256"):
                raise ValueError("SEMANTIC_STEP_HASH")
            values = struct.unpack("<384f", chunk)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("LATENT_NONFINITE")
            actual_norm = math.sqrt(sum(value * value for value in values))
            if not 0.999 <= actual_norm <= 1.001:
                raise ValueError("LATENT_NORM")
            if not math.isclose(trace_step["latent_norm"], actual_norm, rel_tol=0, abs_tol=1e-6):
                raise ValueError("LATENT_NORM_EVIDENCE_MISMATCH")
            previous = trace_step.get("z_sha256")
    if request["task_id"] != task["task_id"] or request["canonical_task_hash"] != task_hash:
        raise ValueError("request task mismatch")
    if file_hash(root / "development-config.json") != request["config_hash"]:
        raise ValueError("config file mismatch")
    if json.loads((root / "development-config.json").read_bytes()) != config:
        raise ValueError("config content mismatch")
    if config != _config(task, generate(17), variant=variant):
        raise ValueError("config provenance mismatch")
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
    if variant in {"A3", "A4"} and work_plan is not None:
        replay_model = LockedPlanner(17, variant)
        replay_model.load_state_dict(persisted_state)
        replay_model.eval()
        replay_planner = A2Planner(replay_model)
        replay_raw = replay_planner.plan(task)
        if replay_raw != [[step["action"], *step["args"]] for step in work_plan["steps"]]:
            raise ValueError("CHECKPOINT_PLAN_REPLAY_MISMATCH")
        reproduced = b"".join(
            struct.pack("<384f", *z.tolist()) for z in (replay_planner.semantic_steps or [])
        )
        if reproduced != payload:
            raise ValueError("CHECKPOINT_LATENT_REPLAY_MISMATCH")
    if file_hash(root / checkpoint["config_path"]) != checkpoint["config_hash"]:
        raise ValueError("checkpoint config mismatch")
    if file_hash(root / checkpoint["training_report_path"]) != checkpoint["training_report_hash"]:
        raise ValueError("training report mismatch")
    if (
        file_hash(root / checkpoint["optimizer_evidence_path"])
        != checkpoint["optimizer_evidence_hash"]
    ):
        raise ValueError("optimizer evidence mismatch")
    report = json.loads((root / checkpoint["training_report_path"]).read_bytes())
    if (
        checkpoint["state_dict_sha256"] != report["trained_sha256"]
        or checkpoint["model_file_sha256"] != report["trained_file_sha256"]
    ):
        raise ValueError("checkpoint differs from training report")
    if (
        file_hash(root / checkpoint["initialization_path"])
        != checkpoint["initialization_file_sha256"]
        or checkpoint["initialization_file_sha256"] != report["initialization_file_sha256"]
    ):
        raise ValueError("initialization file mismatch")
    initialization = torch.load(
        root / checkpoint["initialization_path"], map_location="cpu", weights_only=True
    )
    if (
        state_dict_sha256(initialization) != checkpoint["initialization_sha256"]
        or checkpoint["initialization_sha256"] != report["initialization_sha256"]
    ):
        raise ValueError("initialization content mismatch")
    for field in (
        "config_hash",
        "dataset_hash",
        "training_task_id",
        "training_task_hash",
        "inventory_sha256",
        "task_encoding_sha256",
        "runtime",
        "code_commit",
    ):
        expected = request["config_hash"] if field == "config_hash" else config[field]
        if report.get(field) != expected:
            raise ValueError(f"training report {field} mismatch")
    optimizer_evidence = json.loads((root / checkpoint["optimizer_evidence_path"]).read_bytes())
    audit_model = LockedPlanner(17, variant)
    inventory_active = set(audit_model.active_names)
    allowed_no_gradient = (
        {name for name in inventory_active if name.startswith("semantic.latent_feedback.")}
        if variant == "A4"
        else set()
    )
    expected_gradient = inventory_active - allowed_no_gradient
    evidence = optimizer_evidence.get("active_gradient_evidence", {})
    if set(evidence) != inventory_active:
        raise ValueError("gradient evidence active set mismatch")
    if (
        set(optimizer_evidence.get("active_parameter_names", [])) != inventory_active
        or set(optimizer_evidence.get("expected_gradient_parameter_names", [])) != expected_gradient
        or set(optimizer_evidence.get("allowed_no_gradient_parameter_names", []))
        != allowed_no_gradient
        or set(optimizer_evidence.get("optimizer_state_parameter_names", [])) != expected_gradient
        or not optimizer_evidence.get("optimizer_state_matches_expected_gradient_set")
    ):
        raise ValueError("optimizer gradient policy mismatch")
    for name in expected_gradient:
        row = evidence[name]
        if not (row.get("grad_present") and row.get("finite") and row.get("nonzero")):
            raise ValueError("required active gradient missing")
    for name in allowed_no_gradient:
        if evidence[name].get("grad_present"):
            raise ValueError("A4 allowed-no-gradient policy mismatch")
    if (
        checkpoint["optimizer_evidence_hash"] != report["optimizer_evidence_hash"]
        or checkpoint["optimizer_state_path"] != report["optimizer_state_path"]
        or checkpoint["optimizer_state_file_sha256"] != report["optimizer_state_file_sha256"]
        or checkpoint["optimizer_state_sha256"] != report["optimizer_state_sha256"]
        or checkpoint["optimizer_state_sha256"] != optimizer_evidence["optimizer_state_sha256"]
    ):
        raise ValueError("optimizer report binding mismatch")
    if (
        file_hash(root / checkpoint["optimizer_state_path"])
        != checkpoint["optimizer_state_file_sha256"]
    ):
        raise ValueError("optimizer state file mismatch")
    audit_parameters = [
        parameter for parameter in audit_model.parameters() if parameter.requires_grad
    ]
    training = config["training"]
    audit_optimizer = torch.optim.AdamW(
        audit_parameters,
        lr=training["learning_rate"],
        betas=tuple(training["adamw_betas"]),
        eps=training["eps"],
        weight_decay=training["weight_decay"],
    )
    audit_optimizer.load_state_dict(
        torch.load(root / checkpoint["optimizer_state_path"], map_location="cpu", weights_only=True)
    )
    if (
        optimizer_state_sha256(audit_optimizer, dict(audit_model.named_parameters()))
        != checkpoint["optimizer_state_sha256"]
    ):
        raise ValueError("optimizer state content mismatch")
    if (
        checkpoint["inventory_sha256"] != config["inventory_sha256"]
        or checkpoint["runtime"] != config["runtime"]
    ):
        raise ValueError("checkpoint provenance mismatch")
    expected_feedback = (
        "none" if variant == "A2" else "teacher-forced-current-step-target-shifted-to-next-position"
    )
    from .semantic import DIMENSION, TARGET_CONFIG_SHA256, TARGET_SOURCE

    if (
        checkpoint.get("variant") != variant
        or report.get("variant") != variant
        or report.get("feedback_source") != expected_feedback
        or report.get("latent_dimension") != DIMENSION
        or report.get("semantic_loss_weight") != config["training"]["semantic_loss_weight"]
        or report.get("latent_target_source")
        != (TARGET_SOURCE if variant in {"A3", "A4"} else None)
        or report.get("latent_target_config_hash")
        != (TARGET_CONFIG_SHA256 if variant in {"A3", "A4"} else None)
    ):
        raise ValueError("training variant semantics mismatch")
    bindings = _common(
        task,
        request["config_hash"],
        request["checkpoint_manifest_hash"],
        checkpoint["model_file_sha256"],
    )
    for artifact in [work_plan, manifest, episode, evaluation, *attempts]:
        if artifact is None:
            continue
        for field, expected in bindings.items():
            if artifact.get(field) != expected:
                raise ValueError(f"{artifact['schema_version']} {field} mismatch")
    for artifact in (
        request,
        config,
        checkpoint,
        work_plan,
        manifest,
        episode,
        evaluation,
        *attempts,
    ):
        if artifact is None:
            continue
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
    for artifact in [work_plan, manifest, episode, evaluation, *attempts]:
        if artifact is not None and artifact.get("planner_request_hash") != request["request_hash"]:
            raise ValueError("planner request downstream binding mismatch")
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
        if episode["plan_positions_consumed"] != 0:
            raise ValueError("generation failure consumed plan positions")
        if episode["final_state_hash"] != state_hash(task["initial"]):
            raise ValueError("generation failure final state mismatch")
        if episode["goal_success"] is not False:
            raise ValueError("generation failure cannot claim goal success")
        if evaluation["goal_hash"] != goal_hash(task["goal"]):
            raise ValueError("generation failure goal hash mismatch")
        if evaluation["success"] or evaluation["executed_action_count"]:
            raise ValueError("failure evaluation mismatch")
        if episode["plan_generation_status"] != "FAILED" or not manifest["failure_code"]:
            raise ValueError("typed failure missing")
        if not (
            manifest["failure_code"] == episode["terminal_error"] == evaluation["failure_code"]
        ):
            raise ValueError("failure code lineage mismatch")
    else:
        if manifest["plan_status"] != "READY" or episode["plan_generation_status"] != "READY":
            raise ValueError("ready state mismatch")
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
        if work_plan["state_hash"] != state_hash(task["initial"]):
            raise ValueError("WorkPlan initial state mismatch")
        indices = [step["step_index"] for step in work_plan["steps"]]
        if indices != list(range(len(indices))):
            raise ValueError("WorkPlan step indices mismatch")
        end_steps = [step for step in work_plan["steps"] if step["action"] == "END"]
        if len(end_steps) != 1 or work_plan["steps"][-1] != end_steps[0] or end_steps[0]["args"]:
            raise ValueError("WorkPlan terminal END mismatch")
        if (
            manifest["work_plan_content_hash"] != work_plan["plan_content_hash"]
            or manifest["work_plan_artifact_hash"] != work_plan["plan_artifact_hash"]
        ):
            raise ValueError("manifest plan mismatch")
        if manifest["failure_code"] is not None:
            raise ValueError("READY generation cannot carry generation failure")
        if not (root / manifest["work_plan_path"]).is_file():
            raise ValueError("declared WorkPlan missing")
        if json.loads((root / manifest["work_plan_path"]).read_bytes()) != work_plan:
            raise ValueError("declared WorkPlan content mismatch")
        non_end = [step for step in work_plan["steps"] if step["action"] != "END"]
        if len(attempts) > len(non_end):
            raise ValueError("attempt count exceeds frozen plan")
        if len(attempts) < len(non_end) and (not attempts or attempts[-1]["status"] != "FAILED"):
            raise ValueError("truncated execution requires terminal failed attempt")
        state = validate_state(tuple(task["blocks"]), tuple(tuple(x) for x in task["initial"]))
        for index, (attempt, step) in enumerate(
            zip(attempts, non_end[: len(attempts)], strict=True)
        ):
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
            if attempt["plan_artifact_hash"] != work_plan["plan_artifact_hash"]:
                raise ValueError("attempt plan artifact mismatch")
            if attempt["state_before_hash"] != state_hash(state):
                raise ValueError("state before mismatch")
            action = (step["action"], *step["args"])
            try:
                successor = apply_action(tuple(task["blocks"]), state, action)
            except ValueError:
                if attempt["status"] != "FAILED" or not attempt["error"]:
                    raise ValueError("executor failure evidence mismatch") from None
                if attempt["state_after_hash"] != state_hash(state) or index != len(attempts) - 1:
                    raise ValueError("executor failure state mismatch") from None
                continue
            if attempt["status"] != "APPLIED" or attempt["error"] is not None:
                raise ValueError("applied attempt status mismatch")
            state = successor
            if attempt["state_after_hash"] != state_hash(state):
                raise ValueError("state after mismatch")
        success = goal_satisfied(state, tuple(tuple(x) for x in task["goal"]))
        count = len(attempts)
        executed = sum(attempt["status"] == "APPLIED" for attempt in attempts)
        if (
            episode["attempts_total"],
            episode["executed_length"],
            episode["plan_positions_consumed"],
        ) != (count, executed, count):
            raise ValueError("episode counts mismatch")
        if episode["final_state_hash"] != state_hash(state) or episode["goal_success"] != success:
            raise ValueError("episode outcome mismatch")
        if evaluation["success"] != success or evaluation["executed_action_count"] != executed:
            raise ValueError("evaluation outcome mismatch")
        if evaluation["goal_hash"] != goal_hash(task["goal"]):
            raise ValueError("evaluation goal mismatch")
        failed_attempts = [attempt for attempt in attempts if attempt["status"] == "FAILED"]
        if failed_attempts:
            if (
                episode["terminal_error"] != "EXECUTOR_PRECONDITION_FAILED"
                or evaluation["failure_code"] != "EXECUTOR_PRECONDITION_FAILED"
            ):
                raise ValueError("executor failure outcome mismatch")
        elif success:
            if episode["terminal_error"] is not None or evaluation["failure_code"] is not None:
                raise ValueError("successful execution cannot carry failure code")
        elif not (
            episode["terminal_error"] == "GOAL_NOT_ACHIEVED"
            and evaluation["failure_code"] == "GOAL_NOT_ACHIEVED"
        ):
            raise ValueError("unsatisfied goal requires typed failure")
    if evaluation["attempt_log_hash"] != file_hash(root / "attempt-log.jsonl"):
        raise ValueError("attempt log hash mismatch")
    if evaluation["episode_log_hash"] != file_hash(root / "episode-log.json"):
        raise ValueError("episode log hash mismatch")
    if artifact_hash(evaluation, "evaluation_result_hash") != evaluation["evaluation_result_hash"]:
        raise ValueError("evaluation hash mismatch")


def validate_run_directory(root: Path, task: dict) -> None:
    """Reload every emitted lineage artifact and validate the on-disk chain."""

    from jsonschema import Draft202012Validator, ValidationError

    def load(name):
        return json.loads((root / name).read_bytes())

    manifest = load("episode-plan-manifest.json")
    attempt_path = root / "attempt-log.jsonl"
    attempts = [json.loads(line) for line in attempt_path.read_text().splitlines()]
    work_plan = load(manifest["work_plan_path"]) if manifest["work_plan_path"] else None
    semantic_trace = (
        load("semantic-trace.json") if (root / "semantic-trace.json").is_file() else None
    )
    persisted = {
        "toy_development_config": load("development-config.json"),
        "toy_checkpoint_manifest": load("checkpoint-manifest.json"),
        "toy_planner_request": load("planner-request.json"),
        "toy_episode_plan_manifest": manifest,
        "toy_episode_log": load("episode-log.json"),
        "toy_evaluation_result": load("evaluation-result.json"),
        "toy_optimizer_evidence": load("model/optimizer-evidence.json"),
        "toy_training_report": load("model/training-report.json"),
        "toy_run_result": load("run-result.json"),
    }
    if work_plan is not None:
        persisted["toy_work_plan"] = work_plan
    if semantic_trace is not None:
        persisted["toy_semantic_trace"] = semantic_trace
    schema_root = Path(__file__).with_name("schemas")
    try:
        for schema_name, artifact in persisted.items():
            schema = json.loads((schema_root / f"{schema_name}.schema.json").read_bytes())
            Draft202012Validator(schema).validate(artifact)
        attempt_schema = json.loads((schema_root / "toy_attempt_log.schema.json").read_bytes())
        for attempt in attempts:
            Draft202012Validator(attempt_schema).validate(attempt)
    except ValidationError as error:
        raise ValueError(f"RUNTIME_SCHEMA_INVALID: {error.message}") from error
    request = persisted["toy_planner_request"]
    config = persisted["toy_development_config"]
    checkpoint = persisted["toy_checkpoint_manifest"]
    evaluation = persisted["toy_evaluation_result"]
    result = persisted["toy_run_result"]
    validate_lineage(
        root=root,
        task=task,
        request=request,
        config=config,
        checkpoint=checkpoint,
        work_plan=work_plan,
        manifest=manifest,
        attempts=attempts,
        episode=load("episode-log.json"),
        evaluation=load("evaluation-result.json"),
        semantic_trace=semantic_trace,
    )
    artifacts = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "run-result.json"
    )
    expected_replay = toy_hash(
        "run_files", {str(path.relative_to(root)): file_hash(path) for path in artifacts}
    )
    if artifact_hash(result, "run_result_hash") != result["run_result_hash"]:
        raise ValueError("run result hash mismatch")
    if (
        result["replay_hash"] != expected_replay
        or result["variant"] != config["variant"]
        or result["success"] != evaluation["success"]
        or result["failure_code"] != evaluation["failure_code"]
        or result["planner_call_count"] != 1
        or result["tensor_count"] != 177
        or result["config_hash"] != request["config_hash"]
        or result["request_hash"] != request["request_hash"]
        or result["checkpoint_manifest_hash"] != request["checkpoint_manifest_hash"]
        or result["checkpoint_file_hash"] != checkpoint["model_file_sha256"]
        or result["evaluation_result_hash"] != evaluation["evaluation_result_hash"]
    ):
        raise ValueError("run result semantics mismatch")


def verify_replay(root: Path, task: dict) -> None:
    """Recompute the complete non-circular run hash after all files are durable."""
    validate_run_directory(root, task)


def run(
    output: Path,
    *,
    variant: str = "A2",
    failure_mode: str | None = None,
    reuse_from: Path | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be clean")
    output.mkdir(parents=True, exist_ok=True)
    dataset = generate(17)
    row = next(r for r in dataset["train"] if len(r["oracle_work_plan"]) > 1)
    expected_config = _config(row, dataset, variant=variant)
    if reuse_from is None:
        config = expected_config
        _write(output / "development-config.json", config)
        config_hash = file_hash(output / "development-config.json")
        trained, training = train(row, output / "model", config=config, config_hash=config_hash)
        del trained
        checkpoint_path = output / "model/trained.pt"
        checkpoint = {
            "schema_version": "toy-checkpoint-manifest/1.0",
            "variant": variant,
            **DEV,
            "model_path": "model/trained.pt",
            "model_file_sha256": file_hash(checkpoint_path),
            "state_dict_sha256": training["trained_sha256"],
            "config_path": "development-config.json",
            "config_hash": config_hash,
            "inventory_sha256": config["inventory_sha256"],
            "initialization_sha256": training["initialization_sha256"],
            "initialization_path": "model/initialization.pt",
            "initialization_file_sha256": training["initialization_file_sha256"],
            "training_report_path": "model/training-report.json",
            "training_report_hash": file_hash(output / "model/training-report.json"),
            "optimizer_evidence_path": training["optimizer_evidence_path"],
            "optimizer_evidence_hash": training["optimizer_evidence_hash"],
            "optimizer_state_path": training["optimizer_state_path"],
            "optimizer_state_file_sha256": training["optimizer_state_file_sha256"],
            "optimizer_state_sha256": training["optimizer_state_sha256"],
            "runtime": config["runtime"],
            "confirmatory": False,
            "sealed_data": False,
        }
        _write(output / "checkpoint-manifest.json", checkpoint)
    else:
        import shutil

        shutil.copy2(reuse_from / "development-config.json", output / "development-config.json")
        shutil.copytree(reuse_from / "model", output / "model")
        shutil.copy2(reuse_from / "checkpoint-manifest.json", output / "checkpoint-manifest.json")
        config = json.loads((output / "development-config.json").read_bytes())
        if config != expected_config:
            raise ValueError("reused config provenance differs from requested run")
        checkpoint = json.loads((output / "checkpoint-manifest.json").read_bytes())
        training = json.loads((output / "model/training-report.json").read_bytes())
    config_hash = file_hash(output / "development-config.json")
    checkpoint_path = output / "model/trained.pt"
    checkpoint_manifest_hash = file_hash(output / "checkpoint-manifest.json")
    common = _common(row, config_hash, checkpoint_manifest_hash, checkpoint["model_file_sha256"])
    request = {
        "schema_version": "toy-planner-request/1.0",
        "variant": variant,
        "request_id": "toy-request-0001",
        **common,
        **DEV,
        "confirmatory": False,
        "sealed_data": False,
        "request_hash": "",
    }
    request["request_hash"] = artifact_hash(request, "request_hash")
    _write(output / "planner-request.json", request)
    downstream = {**common, "planner_request_hash": request["request_hash"]}
    model = LockedPlanner(17, variant)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    planner = A2Planner(
        model,
        injected_failure=failure_mode
        if failure_mode in {"LATENT_NONFINITE", "LATENT_ZERO_NORM", "LATENT_DIMENSION"}
        else None,
    )
    work_plan = None
    generation_failure_code = None
    try:
        raw = planner.plan(row)
        if failure_mode == "NO_END":
            raw = raw[:-1]
        elif failure_mode == "INAPPLICABLE":
            raw = [
                raw[0],
                ["STACK", row["blocks"][0], row["blocks"][0]],
                raw[1],
                ["END"],
            ]
        parsed = parse_work_plan(raw, row["blocks"])
        steps = [
            {"step_index": i, "action": step[0], "args": step[1:]} for i, step in enumerate(raw)
        ]
        work_plan = {
            "schema_version": "toy-work-plan/1.0",
            "variant": variant,
            **downstream,
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
    except (RuntimeError, ValueError) as error:
        code = str(error)
        generation_failure_code = (
            code if code in PRESERVED_GENERATION_CODES else "PLAN_GENERATION_ERROR"
        )
        events, success = [], False
    execution_failure_code = None
    if work_plan is not None:
        events, success, execution_failure_code = execute(row, parsed)
    plan_path = "results/development/plans/work-plan.json" if work_plan else None
    manifest = {
        "schema_version": "toy-episode-plan-manifest/1.0",
        "variant": variant,
        **downstream,
        **DEV,
        "confirmatory": False,
        "sealed_data": False,
        "plan_status": "READY" if work_plan else "FAILED",
        "planner_call_count": 1,
        "work_plan_path": plan_path,
        "work_plan_content_hash": work_plan["plan_content_hash"] if work_plan else None,
        "work_plan_artifact_hash": work_plan["plan_artifact_hash"] if work_plan else None,
        "failure_code": generation_failure_code,
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
                    "variant": variant,
                    **downstream,
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
                    "status": event["status"],
                    "error": event["error"],
                    "replanning_observed": False,
                }
            )
    final_state = events[-1]["after"] if events else state_hash(row["initial"])
    executed_count = sum(event["status"] == "APPLIED" for event in events)
    outcome_failure_code = generation_failure_code or execution_failure_code
    if work_plan is not None and not success and outcome_failure_code is None:
        outcome_failure_code = "GOAL_NOT_ACHIEVED"
    episode = {
        "schema_version": "toy-episode-log/1.0",
        "variant": variant,
        **downstream,
        **DEV,
        "episode_plan_manifest_hash": manifest["manifest_hash"],
        "planner_calls": 1,
        "attempts_total": len(attempts),
        "executed_length": executed_count,
        "plan_positions_consumed": len(attempts),
        "final_state_hash": final_state,
        "goal_success": success,
        "plan_generation_status": "READY" if work_plan else "FAILED",
        "terminal_error": outcome_failure_code,
    }
    if work_plan:
        _write(output / plan_path, work_plan)
    _write(output / "episode-plan-manifest.json", manifest)
    _write(output / "attempt-log.jsonl", attempts, jsonl=True)
    _write(output / "episode-log.json", episode)
    evaluation = {
        "schema_version": "toy-evaluation-result/1.0",
        "variant": variant,
        **downstream,
        **DEV,
        "confirmatory": False,
        "sealed_data": False,
        "episode_plan_manifest_hash": manifest["manifest_hash"],
        "attempt_log_hash": file_hash(output / "attempt-log.jsonl"),
        "episode_log_hash": file_hash(output / "episode-log.json"),
        "success": success,
        "goal_hash": goal_hash(row["goal"]),
        "executed_action_count": executed_count,
        "replanning_count": 0,
        "failure_code": outcome_failure_code,
        "evaluation_result_hash": "",
    }
    evaluation["evaluation_result_hash"] = artifact_hash(evaluation, "evaluation_result_hash")
    _write(output / "evaluation-result.json", evaluation)
    if variant in {"A3", "A4"} and work_plan is not None:
        import struct

        latent_path = output / "semantic-latents.f32"
        latents = planner.semantic_steps or []
        latent_path.write_bytes(b"".join(struct.pack("<384f", *z.tolist()) for z in latents))
        entries = []
        previous = None
        for index, (step, z) in enumerate(zip(work_plan["steps"], latents, strict=True)):
            z_hash = "sha256:" + hashlib.sha256(z.numpy().tobytes()).hexdigest()
            entries.append(
                {
                    "step_index": index,
                    "action": step["action"],
                    "args": step["args"],
                    "z_sha256": z_hash,
                    "latent_norm": float(z.norm()),
                    "source": "predicted",
                    "previous_z_sha256": previous,
                }
            )
            previous = z_hash
        semantic_trace = {
            "schema_version": "toy-semantic-trace/1.0",
            "variant": variant,
            "feedback_source": "predicted",
            "feedback_applied": variant == "A3",
            "config_hash": config_hash,
            "checkpoint_manifest_hash": checkpoint_manifest_hash,
            "checkpoint_file_hash": checkpoint["model_file_sha256"],
            "planner_request_hash": request["request_hash"],
            "work_plan_artifact_hash": work_plan["plan_artifact_hash"],
            "latent_path": "semantic-latents.f32",
            "latent_file_sha256": file_hash(latent_path),
            "steps": entries,
        }
        _write(output / "semantic-trace.json", semantic_trace)
    artifacts = sorted(p for p in output.rglob("*") if p.is_file())
    replay_hash = toy_hash(
        "run_files", {str(p.relative_to(output)): file_hash(p) for p in artifacts}
    )
    result = {
        "schema_version": "toy-run-result/1.0",
        "success": success,
        "planner_call_count": 1,
        "tensor_count": training["tensor_count"],
        "failure_code": outcome_failure_code,
        "variant": variant,
        "replay_hash": replay_hash,
        "config_hash": config_hash,
        "request_hash": request["request_hash"],
        "checkpoint_manifest_hash": checkpoint_manifest_hash,
        "checkpoint_file_hash": checkpoint["model_file_sha256"],
        "evaluation_result_hash": evaluation["evaluation_result_hash"],
        "run_result_hash": "",
    }
    result["run_result_hash"] = artifact_hash(result, "run_result_hash")
    _write(output / "run-result.json", result)
    verify_replay(output, row)
    return result
