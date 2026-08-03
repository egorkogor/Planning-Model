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
from .numeric_identity import canonical_float32_sha256, canonical_norm
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


class PlanParseFailure(ValueError):
    """Typed failure produced only by the quality WorkPlan parser."""


PLAN_FAILURE_CODES = frozenset({
    "PLAN_NO_END",
    "PLAN_PARSE_ERROR",
    "PLAN_UNKNOWN_REF",
    "PLAN_GENERATION_ERROR",
})
LATENT_FAILURE_CODES = frozenset({
    "LATENT_NONFINITE",
    "LATENT_ZERO_NORM",
    "LATENT_DIMENSION",
})
GENERATION_FAILURE_CODES_BY_VARIANT = {
    "A2": PLAN_FAILURE_CODES,
    "A3": PLAN_FAILURE_CODES | LATENT_FAILURE_CODES,
    "A4": PLAN_FAILURE_CODES | LATENT_FAILURE_CODES,
}
GENERATION_FAILURE_CODES = PLAN_FAILURE_CODES | LATENT_FAILURE_CODES
PRE_FORWARD_FAILURE_CODES = frozenset({
    "PLAN_GENERATION_ERROR", "LATENT_NONFINITE", "LATENT_ZERO_NORM", "LATENT_DIMENSION",
})
LEGACY_PRESERVED_GENERATION_CODES = frozenset({
    "PLAN_NO_END", "PLAN_PARSE_ERROR", "PLAN_UNKNOWN_REF",
    "LATENT_NONFINITE", "LATENT_ZERO_NORM", "LATENT_DIMENSION",
})


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def toy_hash(kind: str, value) -> str:
    return sha256({"schema": "toy-planner-hash/1.0", "kind": kind, "value": value})


def quality_evidence_semantic_hash(
    *, request: dict, work_plan: dict | None, attempts: list[dict], manifest: dict,
    episode: dict, evaluation: dict, semantic_trace: dict | None,
) -> str:
    """Cross-run identity excluding run-local serialization/file hashes."""
    evaluation_semantic = {
        key: value for key, value in evaluation.items()
        if key not in {"attempt_log_hash", "episode_log_hash"}
    }
    request_semantic = {
        key: value for key, value in request.items()
        if key != "checkpoint_state_dict_sha256"
    }
    trace_semantic = None
    if semantic_trace is not None:
        trace_semantic = {
            key: value for key, value in semantic_trace.items()
            if key not in {
                "checkpoint_file_hash", "planner_request_hash", "latent_file_sha256",
                "projected_file_sha256",
            }
        }
        trace_semantic["steps"] = [{
            key: value for key, value in step.items()
            if key not in {"z_sha256", "previous_z_sha256", "latent_norm"}
        } for step in semantic_trace["steps"]]
        trace_semantic["control_audit"] = [{
            key: value for key, value in row.items()
            if key not in {
                "input_feedback_sha256", "projected_feedback_sha256",
                "downstream_semantic_component_sha256", "input_feedback_norm",
                "projected_feedback_norm", "downstream_semantic_component_norm",
                "expected_previous_z_sha256",
            }
        } for row in semantic_trace["control_audit"]]
    return toy_hash(
        "quality_evidence_semantic",
        {
            "request": request_semantic, "work_plan": work_plan, "attempts": attempts,
            "manifest": manifest, "episode": episode,
            "evaluation": evaluation_semantic, "semantic_trace": trace_semantic,
        },
    )


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
    projected_steps: list[torch.Tensor] | None = None

    def plan(self, row: dict) -> list[list[str]]:
        if self.calls:
            raise RuntimeError("replanning is forbidden")
        self.calls += 1
        self.model_forward_count = 0
        if self.injected_failure in {"LATENT_NONFINITE", "LATENT_ZERO_NORM", "LATENT_DIMENSION"}:
            raise PlannerGenerationFailure(self.injected_failure, [], 0)
        action_ids, arg1_ids, arg2_ids = [4] * 17, [0] * 17, [0] * 17
        decoded: list[list[str]] = []
        self.semantic_steps = []
        self.semantic_audit = []
        self.projected_steps = []
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
                self.projected_steps.append(projected.clone())
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
    except PlanParseFailure as error:
        parsed = ()
        generation_failure = str(error)
    except ValueError as error:
        raise ValueError("QUALITY_UNKNOWN_GENERATION_EXCEPTION") from error
    if generation_failure is not None and generation_failure not in (
        GENERATION_FAILURE_CODES_BY_VARIANT[planner.model.variant]
    ):
        raise ValueError("QUALITY_VARIANT_FAILURE_CODE_MISMATCH")
    terminal_end = bool(raw and raw[-1] == ["END"])
    work_plan = None
    if generation_failure is None:
        work_plan = {
            "schema_version": "toy-quality-work-plan/1.0",
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
        "variant": planner.model.variant,
        "planner_call_count": planner.calls, "replanning_count": 0,
        "model_forward_count": planner.model_forward_count,
        "plan_status": "READY" if work_plan else "FAILED",
        "work_plan_path": "work-plan.json" if work_plan else None,
        "work_plan_hash": work_plan["plan_content_hash"] if work_plan else None,
        "failure_code": generation_failure,
        "partial_raw_output": raw if generation_failure is not None else None,
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
        projected_path = output / "projected-feedback.f32"
        projected = getattr(planner, "projected_steps", None) or []
        projected_path.write_bytes(
            b"".join(struct.pack("<256f", *value.tolist()) for value in projected)
        )
        previous = None
        trace_steps = []
        for index, z in enumerate(latents):
            payload = struct.pack("<384f", *z.tolist())
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            action = raw[index] if index < len(raw) else ["END"]
            trace_steps.append({"step_index": index, "action": action[0], "args": action[1:],
                                "z_sha256": digest, "previous_z_sha256": previous,
                                "canonical_z_sha256": canonical_float32_sha256(payload),
                                "latent_norm": float(z.norm()),
                                "canonical_latent_norm": canonical_norm(payload),
                                "source": "predicted"})
            previous = digest
        trace = {
            "schema_version": "toy-quality-semantic-trace/1.0",
            "variant": planner.model.variant, "feedback_source": "predicted",
            "downstream_semantic_component_observed": any(
                row["step_index"] > 0 and row["downstream_semantic_component_norm"] > 0
                for row in (planner.semantic_audit or [])
            ),
            "feedback_mode_enabled": planner.model.variant == "A3",
            "feedback_application_count": sum(
                row["step_index"] > 0 for row in (planner.semantic_audit or [])
            ),
            "nonzero_feedback_application_count": sum(
                row["step_index"] > 0 and row["input_feedback_norm"] > 0
                for row in (planner.semantic_audit or [])
            ),
            "nonzero_downstream_semantic_component_count": sum(
                row["step_index"] > 0 and row["downstream_semantic_component_norm"] > 0
                for row in (planner.semantic_audit or [])
            ),
            "compute_then_zero": planner.model.variant == "A4",
            "checkpoint_file_hash": checkpoint_binding["trained_file_sha256"],
            "planner_request_hash": file_hash(output / "planner-request.json"),
            "work_plan_artifact_hash": work_plan["plan_content_hash"] if work_plan else None,
            "latent_path": "semantic-latents.f32", "latent_file_sha256": file_hash(latent_path),
            "projected_path": "projected-feedback.f32",
            "projected_file_sha256": file_hash(projected_path),
            "steps": trace_steps,
            "control_audit": planner.semantic_audit or [],
        }
        for index, audit_row in enumerate(trace["control_audit"]):
            projected_block = projected_path.read_bytes()[index * 1024 : (index + 1) * 1024]
            audit_row["canonical_projected_feedback_sha256"] = canonical_float32_sha256(
                projected_block
            )
            audit_row["canonical_projected_feedback_norm"] = canonical_norm(projected_block)
            # A3 downstream equals projected; A4 downstream is canonical zero.
            downstream_payload = (
                projected_block if planner.model.variant == "A3" else bytes(256 * 4)
            )
            audit_row["canonical_downstream_semantic_component_sha256"] = (
                canonical_float32_sha256(downstream_payload)
            )
            audit_row["canonical_downstream_semantic_component_norm"] = canonical_norm(
                downstream_payload
            )
        _write(output / "semantic-trace.json", trace)
        semantic_trace = trace
    validate_frozen_plan_lineage_core(
        variant=planner.model.variant, planner_calls=planner.calls, replanning_count=0,
        work_plan=work_plan, attempts=attempts, evaluation=evaluation,
        semantic_trace=semantic_trace,
    )
    evidence_hash = quality_evidence_semantic_hash(
        request=request, work_plan=work_plan, attempts=attempts, manifest=manifest,
        episode=episode, evaluation=evaluation, semantic_trace=semantic_trace,
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
        raise PlanParseFailure("PLAN_PARSE_ERROR")
    if any(
        not isinstance(step, list) or not step or any(not isinstance(value, str) for value in step)
        for step in raw
    ):
        raise PlanParseFailure("PLAN_PARSE_ERROR")
    parsed = []
    for step in raw:
        if step[0] == "END":
            if len(step) != 1:
                raise PlanParseFailure("PLAN_PARSE_ERROR")
        else:
            parsed.append(parse_nonterminal_step(step, blocks))
    if raw[-1] != ["END"] or any(step == ["END"] for step in raw[:-1]):
        raise PlanParseFailure("PLAN_NO_END")
    return tuple(parsed)


def parse_nonterminal_step(step: object, blocks: list[str]) -> tuple[str, ...]:
    """Validate one decoded action without allowing or requiring terminal END."""
    if (
        not isinstance(step, list) or not step
        or any(not isinstance(value, str) for value in step)
    ):
        raise PlanParseFailure("PLAN_PARSE_ERROR")
    arity = {"PICK_UP": 1, "UNSTACK": 2, "PUT_DOWN": 1, "STACK": 2}
    if step[0] not in arity or len(step) != arity[step[0]] + 1:
        raise PlanParseFailure("PLAN_PARSE_ERROR")
    if any(arg not in blocks for arg in step[1:]):
        raise PlanParseFailure("PLAN_UNKNOWN_REF")
    return tuple(step)


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
        end_indices = [index for index, step in enumerate(steps) if step.get("action") == "END"]
        if end_indices != [len(steps) - 1]:
            raise ValueError("WorkPlan terminal END mismatch")
        arity = {"PICK_UP": 1, "PUT_DOWN": 1, "UNSTACK": 2, "STACK": 2, "END": 0}
        for step in steps:
            if (
                step.get("action") not in arity
                or len(step.get("args", [])) != arity[step["action"]]
            ):
                raise ValueError("QUALITY_ACTION_ARITY_MISMATCH")
        executable_steps = [step for step in steps if step.get("action") != "END"]
        if len(attempts) > len(executable_steps):
            raise ValueError("attempt count exceeds frozen WorkPlan")
        previous_after = None
        for index, (attempt, step) in enumerate(
            zip(attempts, executable_steps, strict=False)
        ):
            candidate = attempt.get("candidate_action")
            expected = [step["action"], *step.get("args", [])]
            if isinstance(candidate, dict):
                candidate = [candidate.get("action"), *candidate.get("args", [])]
            if attempt.get("step_index") != index or candidate != expected:
                raise ValueError("attempt does not match frozen WorkPlan")
            if previous_after is not None and attempt.get("state_before_hash") != previous_after:
                raise ValueError("attempt state transition chain broken")
            previous_after = attempt.get("state_after_hash")
        if len(attempts) < len(executable_steps) and (
            not attempts or attempts[-1].get("status") != "FAILED"
        ):
            raise ValueError("execution stopped before frozen WorkPlan terminal condition")
    if variant == "A2" and semantic_trace is not None:
        raise ValueError("A2_SEMANTIC_ARTIFACT_FORBIDDEN")
    if variant in {"A3", "A4"} and (work_plan is not None or semantic_trace is not None):
        if semantic_trace is None:
            raise ValueError("SEMANTIC_TRACE_MISSING")
        if semantic_trace.get("feedback_source") != "predicted":
            raise ValueError("INFERENCE_FEEDBACK_NOT_PREDICTED")
        if "feedback_application_count" in semantic_trace:
            count = semantic_trace["feedback_application_count"]
            downstream_count = semantic_trace["nonzero_downstream_semantic_component_count"]
            if count < 0 or downstream_count < 0 or downstream_count > count:
                raise ValueError("SEMANTIC_FEEDBACK_COUNT_INVALID")
            if semantic_trace.get("downstream_semantic_component_observed") != (
                downstream_count > 0
            ):
                raise ValueError("SEMANTIC_DOWNSTREAM_COMPONENT_COUNT_MISMATCH")
            if variant == "A3" and semantic_trace.get("feedback_mode_enabled") is not True:
                raise ValueError("A3_FEEDBACK_MODE_DISABLED")
            if variant == "A4" and downstream_count != 0:
                raise ValueError("A4_NONZERO_DOWNSTREAM_COMPONENT")
            audit = semantic_trace.get("control_audit", [])
            if [row.get("step_index") for row in audit] != list(range(len(audit))):
                raise ValueError("SEMANTIC_CONTROL_AUDIT_ORDER")
            steps = semantic_trace.get("steps", [])
            zero_feedback_hash = "sha256:" + hashlib.sha256(
                torch.zeros(384).numpy().tobytes()
            ).hexdigest()
            zero_component_hash = "sha256:" + hashlib.sha256(
                torch.zeros(256).numpy().tobytes()
            ).hexdigest()
            for index, row in enumerate(audit):
                expected_previous = None if index == 0 else steps[index - 1]["z_sha256"]
                expected_input = zero_feedback_hash if index == 0 else expected_previous
                expected_norm = 0.0 if index == 0 else steps[index - 1]["latent_norm"]
                if (
                    row.get("expected_previous_z_sha256") != expected_previous
                    or row.get("input_feedback_sha256") != expected_input
                    or row.get("source")
                    != ("BOS_ZERO" if index == 0 else "PREVIOUS_PREDICTED_LATENT")
                    or not math.isclose(
                        row.get("input_feedback_norm", math.inf), expected_norm,
                        rel_tol=0, abs_tol=1e-6,
                    )
                ):
                    raise ValueError("QUALITY_INPUT_FEEDBACK_NORM_MISMATCH")
                if variant == "A4":
                    if not row.get("projected_feedback_present"):
                        raise ValueError("QUALITY_PROJECTED_FEEDBACK_MISMATCH")
                    if not row.get("downstream_component_zero"):
                        raise ValueError("QUALITY_DOWNSTREAM_ZERO_FLAG_MISMATCH")
                    if (
                        row.get("downstream_semantic_component_sha256")
                        != zero_component_hash
                        or row.get("downstream_semantic_component_norm") != 0.0
                    ):
                        raise ValueError("A4_COMPUTE_THEN_ZERO_MISMATCH")
        else:
            if variant == "A3" and semantic_trace.get("feedback_applied") is not True:
                raise ValueError("A3_INFERENCE_FEEDBACK_NOT_APPLIED")
            if variant == "A4" and semantic_trace.get("feedback_applied") is not False:
                raise ValueError("A4_FEEDBACK_APPLIED")
    if evaluation.get("replanning_count", 0) != 0:
        raise ValueError("evaluation replanning mismatch")


def validate_persisted_quality_evidence(
    *, root: Path, task: dict, checkpoint: dict
) -> dict:
    """Independently validate persisted quality artifacts before model replay."""
    from jsonschema import Draft202012Validator, ValidationError

    schema_root = Path(__file__).with_name("schemas")

    def load_json(name: str, schema_name: str) -> dict:
        try:
            value = json.loads((root / name).read_bytes())
            schema = json.loads((schema_root / schema_name).read_bytes())
            Draft202012Validator(schema).validate(value)
            return value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"QUALITY_SCHEMA_INVALID:{name}") from error

    request = load_json("planner-request.json", "toy_quality_planner_request.schema.json")
    manifest = load_json(
        "episode-plan-manifest.json", "toy_quality_episode_plan_manifest.schema.json"
    )
    episode = load_json("episode-log.json", "toy_quality_episode_log.schema.json")
    evaluation = load_json(
        "evaluation-result.json", "toy_quality_evaluation_result.schema.json"
    )
    try:
        attempt_schema = json.loads(
            (schema_root / "toy_quality_attempt_log.schema.json").read_bytes()
        )
        attempts = [
            json.loads(line) for line in (root / "attempt-log.jsonl").read_text().splitlines()
        ]
        for attempt in attempts:
            Draft202012Validator(attempt_schema).validate(attempt)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError("QUALITY_SCHEMA_INVALID:attempt-log.jsonl") from error
    work_plan = None
    if manifest["work_plan_path"] is not None:
        work_plan = load_json("work-plan.json", "toy_quality_work_plan.schema.json")
    semantic_trace = None
    if request["variant"] in {"A3", "A4"}:
        semantic_trace = load_json(
            "semantic-trace.json", "toy_quality_semantic_trace.schema.json"
        )
    try:
        expected_variant = request["variant"]
        if (
            request["task_id"] != task["task_id"]
            or request["task_hash"] != task["canonical_task_hash"]
            or request["checkpoint_state_dict_sha256"]
            != checkpoint["trained_state_dict_sha256"]
            or checkpoint["variant_identity"]["implementation_variant"] != expected_variant
        ):
            raise ValueError("QUALITY_REQUEST_BINDING_MISMATCH")
        if (
            manifest["variant"] != expected_variant
            or (work_plan is not None and work_plan["variant"] != expected_variant)
            or (semantic_trace is not None and semantic_trace["variant"] != expected_variant)
            or (expected_variant == "A2" and (root / "semantic-trace.json").exists())
        ):
            raise ValueError("QUALITY_VARIANT_BINDING_MISMATCH")
        if evaluation["attempt_log_hash"] != file_hash(root / "attempt-log.jsonl"):
            raise ValueError("QUALITY_ATTEMPT_LOG_HASH_MISMATCH")
        if evaluation["episode_log_hash"] != file_hash(root / "episode-log.json"):
            raise ValueError("QUALITY_EPISODE_LOG_HASH_MISMATCH")
        if work_plan is not None:
            raw_plan = [[step["action"], *step["args"]] for step in work_plan["steps"]]
            try:
                independently_parsed = parse_work_plan(raw_plan, task["blocks"])
            except PlanParseFailure as error:
                code = (
                    "QUALITY_WORK_PLAN_UNKNOWN_REF"
                    if str(error) == "PLAN_UNKNOWN_REF"
                    else "QUALITY_READY_PLAN_PARSE_MISMATCH"
                )
                raise ValueError(code) from None
            if len(independently_parsed) != len(work_plan["steps"]) - 1:
                raise ValueError("QUALITY_READY_PLAN_PARSE_MISMATCH")
        validate_frozen_plan_lineage_core(
            variant=request["variant"], planner_calls=manifest["planner_call_count"],
            replanning_count=manifest["replanning_count"], work_plan=work_plan,
            attempts=attempts, evaluation=evaluation, semantic_trace=semantic_trace,
        )
        state = validate_state(tuple(task["blocks"]), tuple(tuple(x) for x in task["initial"]))
        initial_hash = state_hash(state)
        applied = 0
        terminal_failure = False
        for index, attempt in enumerate(attempts):
            if attempt["state_before_hash"] != state_hash(state):
                raise ValueError("QUALITY_EXECUTOR_STATE_BEFORE_MISMATCH")
            action = tuple(attempt["candidate_action"])
            try:
                successor = apply_action(tuple(task["blocks"]), state, action)
            except ValueError as error:
                if (
                    attempt["status"] != "FAILED"
                    or attempt["error"] != str(error)
                    or attempt["state_after_hash"] != state_hash(state)
                    or index != len(attempts) - 1
                ):
                    raise ValueError("QUALITY_EXECUTOR_FAILURE_MISMATCH") from None
                terminal_failure = True
                break
            if (
                attempt["status"] != "APPLIED"
                or attempt["error"] is not None
                or attempt["state_after_hash"] != state_hash(successor)
            ):
                raise ValueError("QUALITY_EXECUTOR_APPLIED_MISMATCH")
            state = successor
            applied += 1
        final_hash = state_hash(state)
        final_state_goal_satisfied = goal_satisfied(
            state, tuple(tuple(x) for x in task["goal"])
        )
        expected_execution_success = (
            work_plan is not None and final_state_goal_satisfied and not terminal_failure
        )
        if work_plan is not None and (
            work_plan["task_id"] != task["task_id"]
            or work_plan["state_hash"] != initial_hash
            or manifest["work_plan_hash"] != work_plan["plan_content_hash"]
        ):
            raise ValueError("QUALITY_WORK_PLAN_BINDING_MISMATCH")
        if work_plan is not None:
            plan_payload = {
                key: value for key, value in work_plan.items()
                if key != "plan_content_hash"
            }
            if toy_hash("quality_frozen_plan", plan_payload) != work_plan["plan_content_hash"]:
                raise ValueError("QUALITY_WORK_PLAN_HASH_MISMATCH")
        if manifest["plan_status"] == "READY":
            if (
                work_plan is None or manifest["work_plan_path"] != "work-plan.json"
                or manifest["work_plan_hash"] is None or manifest["failure_code"] is not None
                or manifest["partial_raw_output"] is not None
                or manifest["variant"] != request["variant"]
            ):
                raise ValueError("QUALITY_READY_MANIFEST_MISMATCH")
        elif (
            work_plan is not None or manifest["work_plan_path"] is not None
            or manifest["work_plan_hash"] is not None
            or manifest["failure_code"] not in GENERATION_FAILURE_CODES_BY_VARIANT[
                request["variant"]
            ]
            or manifest["variant"] != request["variant"]
            or attempts
        ):
            raise ValueError("QUALITY_FAILED_MANIFEST_MISMATCH")
        if work_plan is not None:
            if not 1 <= len(work_plan["steps"]) <= request["max_decoding_steps"]:
                raise ValueError("QUALITY_DECODING_BUDGET_EXCEEDED")
            expected_forward_count = len(work_plan["steps"])
        elif manifest["failure_code"] == "PLAN_NO_END":
            expected_forward_count = request["max_decoding_steps"]
            if len(manifest["partial_raw_output"]) != expected_forward_count:
                raise ValueError("QUALITY_FAILURE_OUTPUT_COUNT_MISMATCH")
            for step in manifest["partial_raw_output"]:
                try:
                    parse_nonterminal_step(step, task["blocks"])
                except PlanParseFailure as error:
                    raise ValueError("QUALITY_PARTIAL_OUTPUT_INVALID") from error
        elif manifest["failure_code"] in {"PLAN_PARSE_ERROR", "PLAN_UNKNOWN_REF"}:
            partial = manifest["partial_raw_output"]
            if not 1 <= len(partial) <= request["max_decoding_steps"]:
                raise ValueError("QUALITY_DECODING_BUDGET_EXCEEDED")
            expected_forward_count = len(partial)
            try:
                parse_work_plan(partial, task["blocks"])
            except ValueError as error:
                if str(error) != manifest["failure_code"]:
                    raise ValueError("QUALITY_FAILURE_OUTPUT_CODE_MISMATCH") from None
            else:
                raise ValueError("QUALITY_FAILURE_OUTPUT_CODE_MISMATCH")
        elif manifest["failure_code"] in PRE_FORWARD_FAILURE_CODES:
            expected_forward_count = 0
            if manifest["partial_raw_output"] != []:
                raise ValueError("QUALITY_FAILURE_OUTPUT_COUNT_MISMATCH")
        else:  # guarded by the explicit generation-failure set above
            raise ValueError("QUALITY_FAILED_MANIFEST_MISMATCH")
        if (
            manifest["model_forward_count"] != expected_forward_count
            or evaluation["model_forward_count"] != expected_forward_count
        ):
            raise ValueError("QUALITY_MODEL_FORWARD_COUNT_MISMATCH")
        if attempts and attempts[0]["state_before_hash"] != initial_hash:
            raise ValueError("QUALITY_INITIAL_STATE_MISMATCH")
        if (
            episode["attempts_total"] != len(attempts)
            or episode["executed_length"] != applied
            or episode["final_state_hash"] != final_hash
            or episode["goal_success"] != expected_execution_success
            or evaluation["success"] != expected_execution_success
        ):
            raise ValueError("QUALITY_EPISODE_COUNTS_MISMATCH")
        if (
            manifest["model_forward_count"] != evaluation["model_forward_count"]
            or manifest["planner_call_count"] != evaluation["planner_call_count"]
            or episode["planner_calls"] != evaluation["planner_call_count"]
        ):
            raise ValueError("QUALITY_PLANNER_COUNT_MISMATCH")
        expected_failure = (
            "EXECUTOR_PRECONDITION_FAILED" if terminal_failure
            else None if evaluation["success"] else "GOAL_NOT_ACHIEVED"
        )
        if work_plan is None:
            expected_failure = manifest["failure_code"]
        if not (
            evaluation["failure_code"] == episode["terminal_error"] == expected_failure
        ):
            raise ValueError("QUALITY_FAILURE_LINEAGE_MISMATCH")
        if semantic_trace is not None:
            if (
                semantic_trace["checkpoint_file_hash"] != checkpoint["trained_file_sha256"]
                or semantic_trace["planner_request_hash"]
                != file_hash(root / "planner-request.json")
                or semantic_trace["work_plan_artifact_hash"]
                != (work_plan["plan_content_hash"] if work_plan else None)
            ):
                raise ValueError("QUALITY_SEMANTIC_BINDING_MISMATCH")
            steps = semantic_trace["steps"]
            audit = semantic_trace["control_audit"]
            if len(steps) != evaluation["model_forward_count"] or len(audit) != len(steps):
                raise ValueError("QUALITY_SEMANTIC_LENGTH_MISMATCH")
            if work_plan is not None and [
                (step["action"], step["args"]) for step in steps
            ] != [
                (step["action"], step["args"]) for step in work_plan["steps"]
            ]:
                raise ValueError("QUALITY_SEMANTIC_PLAN_MISMATCH")
            if work_plan is None and [
                [step["action"], *step["args"]] for step in steps
            ] != manifest["partial_raw_output"]:
                raise ValueError("QUALITY_PARTIAL_TRACE_MISMATCH")
            expected_applications = max(len(steps) - 1, 0)
            actual_nonzero = sum(
                row["step_index"] > 0 and row["input_feedback_norm"] > 0
                for row in audit
            )
            actual_downstream_nonzero = sum(
                row["step_index"] > 0
                and row["downstream_semantic_component_norm"] > 0
                for row in audit
            )
            if (
                semantic_trace["feedback_application_count"] != expected_applications
                or semantic_trace["nonzero_feedback_application_count"]
                != actual_nonzero
                or semantic_trace["nonzero_downstream_semantic_component_count"]
                != actual_downstream_nonzero
                or semantic_trace["downstream_semantic_component_observed"]
                != (actual_downstream_nonzero > 0)
                or semantic_trace["feedback_mode_enabled"]
                != (request["variant"] == "A3")
                or semantic_trace["compute_then_zero"]
                != (request["variant"] == "A4")
            ):
                raise ValueError("QUALITY_SEMANTIC_COUNT_MISMATCH")
            latent_path = root / semantic_trace["latent_path"]
            try:
                payload = latent_path.read_bytes()
            except OSError as error:
                raise ValueError("QUALITY_LATENT_FILE_MISMATCH") from error
            if (
                len(payload) != len(steps) * 384 * 4
                or file_hash(latent_path) != semantic_trace["latent_file_sha256"]
            ):
                raise ValueError("QUALITY_LATENT_FILE_MISMATCH")
            previous = None
            for index, step in enumerate(steps):
                block = payload[index * 1536 : (index + 1) * 1536]
                values = struct.unpack("<384f", block)
                norm = math.sqrt(sum(value * value for value in values))
                if (
                    step["step_index"] != index
                    or step["previous_z_sha256"] != previous
                    or step["source"] != "predicted"
                    or "sha256:" + hashlib.sha256(block).hexdigest() != step["z_sha256"]
                    or canonical_float32_sha256(block) != step["canonical_z_sha256"]
                    or not all(math.isfinite(value) for value in values)
                    or not math.isclose(step["latent_norm"], norm, rel_tol=0, abs_tol=1e-6)
                    or not math.isclose(
                        step["canonical_latent_norm"], canonical_norm(block),
                        rel_tol=0, abs_tol=1e-6,
                    )
                ):
                    raise ValueError("QUALITY_LATENT_BLOCK_MISMATCH")
                previous = step["z_sha256"]
            projected_path = root / semantic_trace["projected_path"]
            projected_payload = projected_path.read_bytes()
            if (
                len(projected_payload) != len(steps) * 256 * 4
                or file_hash(projected_path) != semantic_trace["projected_file_sha256"]
            ):
                raise ValueError("QUALITY_PROJECTED_FEEDBACK_MISMATCH")
            for index, row in enumerate(audit):
                block = projected_payload[index * 1024 : (index + 1) * 1024]
                values = struct.unpack("<256f", block)
                norm = math.sqrt(sum(value * value for value in values))
                digest = "sha256:" + hashlib.sha256(block).hexdigest()
                if (
                    not all(math.isfinite(value) for value in values)
                    or digest != row["projected_feedback_sha256"]
                    or canonical_float32_sha256(block)
                    != row["canonical_projected_feedback_sha256"]
                    or not math.isclose(
                        norm, row["projected_feedback_norm"], rel_tol=0, abs_tol=1e-6
                    )
                    or not math.isclose(
                        canonical_norm(block), row["canonical_projected_feedback_norm"],
                        rel_tol=0, abs_tol=1e-6,
                    )
                ):
                    raise ValueError("QUALITY_PROJECTED_FEEDBACK_MISMATCH")
                if row["projected_feedback_present"] is not True:
                    raise ValueError("QUALITY_PROJECTED_FEEDBACK_MISMATCH")
                zero_component_hash = "sha256:" + hashlib.sha256(
                    torch.zeros(256).numpy().tobytes()
                ).hexdigest()
                actual_downstream_zero = (
                    row["downstream_semantic_component_sha256"] == zero_component_hash
                    and math.isclose(
                        row["downstream_semantic_component_norm"], 0.0,
                        rel_tol=0, abs_tol=1e-6,
                    )
                )
                if row["downstream_component_zero"] != actual_downstream_zero:
                    raise ValueError("QUALITY_DOWNSTREAM_ZERO_FLAG_MISMATCH")
                downstream_payload = block if request["variant"] == "A3" else bytes(1024)
                if (
                    row["canonical_downstream_semantic_component_sha256"]
                    != canonical_float32_sha256(downstream_payload)
                    or not math.isclose(
                        row["canonical_downstream_semantic_component_norm"],
                        canonical_norm(downstream_payload), rel_tol=0, abs_tol=1e-6,
                    )
                ):
                    raise ValueError("QUALITY_PROJECTED_FEEDBACK_MISMATCH")
                if request["variant"] == "A3" and (
                    digest != row["downstream_semantic_component_sha256"]
                    or not math.isclose(
                        norm, row["downstream_semantic_component_norm"],
                        rel_tol=0, abs_tol=1e-6,
                    )
                ):
                    raise ValueError("QUALITY_PROJECTED_FEEDBACK_MISMATCH")
    except (
        KeyError, IndexError, TypeError, struct.error, OverflowError,
        UnicodeDecodeError, OSError, StopIteration,
    ) as error:
        raise ValueError("QUALITY_LINEAGE_STRUCTURE_INVALID") from error
    return {
        "request": request, "work_plan": work_plan, "manifest": manifest,
        "attempts": attempts, "episode": episode, "evaluation": evaluation,
        "semantic_trace": semantic_trace,
    }


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
            code if code in LEGACY_PRESERVED_GENERATION_CODES else "PLAN_GENERATION_ERROR"
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
