from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hashing import (
    canonical_task_hash,
    goal_hash,
    manifest_artifact_hash,
    manifest_content_hash,
    pair_group_hash,
    plan_artifact_hash,
    plan_content_hash,
    prompt_bytes_hash,
    state_hash,
)

V = "work-planner/1.18"
H1 = "sha256:" + "1" * 64
H2 = "sha256:" + "2" * 64
H3 = "sha256:" + "3" * 64
H4 = "sha256:" + "4" * 64
H5 = "sha256:" + "5" * 64


def semantic_signature(intent_id: int = 4) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "hand_mode": "HAND_EMPTY",
        "goal_relation": "ON",
        "moving_clear": "YES",
        "support_clear": "YES",
        "obstruction_depth_bucket": "ZERO",
        "remaining_distance_bucket": "ONE_TWO",
    }


def task() -> dict[str, Any]:
    obj = {
        "schema_version": V,
        "task_id": "bw-00000001",
        "base_task_id": "bw-00000001",
        "domain": "blocks_world_v1",
        "split": "planner_pilot_horizon",
        "ledger": {
            "@B0": {"type": "BLOCK", "surface_alias": "block_0", "display_name": "amber"},
            "@B1": {"type": "BLOCK", "surface_alias": "block_1", "display_name": "teal"},
        },
        "initial": [
            ["ON_TABLE", "@B0"],
            ["ON_TABLE", "@B1"],
            ["CLEAR", "@B0"],
            ["CLEAR", "@B1"],
            ["HAND_EMPTY"],
        ],
        "goal": [["ON", "@B0", "@B1"]],
        "canonical_task_hash": H1,
        "generator_seed": 1,
        "generator_version": "blocks-generator/1.6",
        "difficulty": {"block_count": 2, "oracle_length": 2, "horizon_bucket": "short_1_5"},
        "permutation_id": None,
    }
    obj["canonical_task_hash"] = canonical_task_hash(obj)
    return obj


def typed(action: str = "PICK_UP", refs: tuple[str, ...] = ("@B0",)) -> dict[str, Any]:
    roles = {
        "PICK_UP": ["block"],
        "PUT_DOWN": ["block"],
        "UNSTACK": ["moving", "support"],
        "STACK": ["moving", "support"],
        "END": [],
    }[action]
    return {
        "schema_version": V,
        "action": action,
        "args": [{"role": role, "ref": ref} for role, ref in zip(roles, refs)],
    }


def step(index: int = 0, action: str = "PICK_UP", refs: tuple[str, ...] = ("@B0",), variant: str = "A3") -> dict[str, Any]:
    rep = {
        "A1": "TOKEN_GRAMMAR",
        "A2": "TYPED_ONLY",
        "A2b": "DISCRETE_INTENT",
        "A2c": "STRUCTURED_DISCRETE",
        "A3": "CONTINUOUS_LATENT",
        "A3r": "CONTINUOUS_LATENT",
        "SELF_PLAN": "DISCRETE_INTENT",
        "A4": "CONTINUOUS_LATENT",
        "A5": "CONTINUOUS_LATENT",
    }[variant]
    is_end = action == "END"
    digest = ("a" if index == 0 else "b") * 64
    obj = {
        "schema_version": V,
        "step_id": f"S{index:02d}",
        "step_index": index,
        "representation": rep,
        "planner_variant": variant,
        "typed_action": typed(action, refs),
        "status": "PREDICTED",
        "semantic_ref": None,
        "intent_id": None,
        "semantic_signature": None,
        "semantic_similarity": None,
        "semantic_margin": None,
    }
    if not is_end and rep == "CONTINUOUS_LATENT":
        obj.update(
            semantic_ref=f"latent+sha256://{digest}#S{index:02d}",
            intent_id=4,
            semantic_signature=semantic_signature(4),
            semantic_similarity=0.8,
            semantic_margin=0.1,
        )
    elif not is_end and rep == "STRUCTURED_DISCRETE":
        obj.update(intent_id=4, semantic_signature=semantic_signature(4))
    elif not is_end and rep == "DISCRETE_INTENT":
        obj.update(intent_id=4)
    return obj


def artifact(step_obj: dict[str, Any], task_obj: dict[str, Any]) -> dict[str, Any]:
    tensor_sha = "sha256:" + step_obj["semantic_ref"].split("//", 1)[1].split("#", 1)[0]
    return {
        "schema_version": V,
        "artifact_id": "sem-" + step_obj["step_id"].lower().replace("s", "0") + "0" * 13,
        "semantic_ref": step_obj["semantic_ref"],
        "task_id": task_obj["task_id"],
        "step_id": step_obj["step_id"],
        "tensor_artifact": "semantic.safetensors",
        "tensor_key": f"{task_obj['task_id']}/{step_obj['step_id']}",
        "shape": [384],
        "dtype": "float32",
        "planner_checkpoint_sha256": H1,
        "planner_config_sha256": H2,
        "state_hash": state_hash(task_obj["initial"]),
        "decoder_state_hash": H3,
        "tensor_sha256": tensor_sha,
        "created_at": "2026-07-24T12:00:00Z",
    }


def plan_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    t = task()
    steps = [
        step(0, "PICK_UP", ("@B0",)),
        step(1, "STACK", ("@B0", "@B1")),
        step(2, "END", ()),
    ]
    manifest = {
        "schema_version": V,
        "manifest_id": "manifest-1",
        "task_id": t["task_id"],
        "canonical_task_hash": t["canonical_task_hash"],
        "state_hash": state_hash(t["initial"]),
        "planner_checkpoint_sha256": H1,
        "planner_config_sha256": H2,
        "artifacts": [artifact(steps[0], t), artifact(steps[1], t)],
        "manifest_content_hash": H3,
        "manifest_hash": H4,
    }
    manifest["manifest_content_hash"] = manifest_content_hash(manifest)
    manifest["manifest_hash"] = manifest_artifact_hash(manifest)
    plan = {
        "schema_version": V,
        "plan_id": "plan-run1",
        "plan_version": 1,
        "task_id": t["task_id"],
        "canonical_task_hash": t["canonical_task_hash"],
        "planner_checkpoint_sha256": H1,
        "planner_config_sha256": H2,
        "planner_seed": 202,
        "semantic_artifact_manifest_sha256": manifest["manifest_hash"],
        "state_hash": state_hash(t["initial"]),
        "steps": steps,
        "created_at": "2026-07-24T12:00:00Z",
        "planner_variant": "A3",
        "representation": "CONTINUOUS_LATENT",
        "plan_content_hash": H3,
        "plan_artifact_hash": H4,
    }
    plan["plan_content_hash"] = plan_content_hash(plan)
    plan["plan_artifact_hash"] = plan_artifact_hash(plan)
    return plan, manifest


def _response(action: str = "PICK_UP", aliases: tuple[str, ...] = ("block_0",)) -> dict[str, Any]:
    return {"schema_version": V, "action": action, "args": list(aliases)}


def attempt(arm: str = "I1_ORACLE_CURRENT_RAW", stage: str = "STAGE1A_INTERFACE") -> dict[str, Any]:
    is_interface = stage == "STAGE1A_INTERFACE"
    split = "stage1a_pilot" if is_interface else "stage1b_pilot"
    snapshot_id = "snap-0123456789abcdef" if is_interface else None
    t = task()
    source = {
        "I0_EQUAL_TOKENS_RAW": "EQUAL_TOKEN_LLM",
        "I1_ORACLE_CURRENT_RAW": "ORACLE_INTENT_LLM",
        "I2_SHUFFLED_RAW": "SHUFFLED_INTENT_LLM",
        "I3_PLANNER_CURRENT_RAW": "PLANNER_INTENT_LLM",
        "E0_EQUAL_TOKENS_RAW": "EQUAL_TOKEN_LLM",
        "E1_A3_FULL_PLAN_RAW": "A3_FULL_PLAN_LLM",
        "E2_SHUFFLED_A3_FULL_PLAN_RAW": "SHUFFLED_A3_PLAN_LLM",
        "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW": "A3R_RANDOM_CODE_PLAN_LLM",
        "E4_A2C_STRUCTURED_FULL_PLAN_RAW": "A2C_STRUCTURED_PLAN_LLM",
        "E5_SELF_PLAN_RAW": "SELF_PLAN_LLM",
        "P_FULL_PLAN_REPLAY_RAW": "PLAN_REPLAY",
    }[arm]
    prompt_text = "rendered prompt"
    raw_text = '{"schema_version":"work-planner/1.18","action":"PICK_UP","args":["block_0"]}'
    prompt_total = 128 if is_interface else 180
    attended = prompt_total
    candidate = typed("PICK_UP", ("@B0",))
    obj: dict[str, Any] = {
        "schema_version": V,
        "run_id": "run-1",
        "episode_id": "episode-1",
        "trajectory_id": "trajectory-1",
        "stage": stage,
        "task_id": t["task_id"],
        "base_task_id": t["base_task_id"],
        "snapshot_id": snapshot_id,
        "canonical_task_hash": t["canonical_task_hash"],
        "trajectory_policy": "oracle_snapshot" if is_interface else "frozen_full_plan",
        "experiment_freeze_hash": H5,
        "episode_plan_manifest_hash": None if is_interface or arm in {"I0_EQUAL_TOKENS_RAW", "I1_ORACLE_CURRENT_RAW", "I2_SHUFFLED_RAW", "I3_PLANNER_CURRENT_RAW", "E0_EQUAL_TOKENS_RAW"} else H4,
        "plan_generation_status": "NONE" if is_interface or arm == "E0_EQUAL_TOKENS_RAW" else "READY",
        "replay_context": "STAGE1B_E1" if arm == "P_FULL_PLAN_REPLAY_RAW" else None,
        "plan_position_index": None if is_interface or arm == "E0_EQUAL_TOKENS_RAW" else 0,
        "guidance_source_position_index": None if is_interface or arm in {"E0_EQUAL_TOKENS_RAW", "P_FULL_PLAN_REPLAY_RAW"} else 0,
        "guidance_source_step_id": None if is_interface or arm in {"E0_EQUAL_TOKENS_RAW", "P_FULL_PLAN_REPLAY_RAW"} else "S00",
        "guidance_source_semantic_ref": None if is_interface or arm in {"E0_EQUAL_TOKENS_RAW", "P_FULL_PLAN_REPLAY_RAW"} else "latent+sha256://" + "a" * 64 + "#S00",
        "replanning_observed": False,
        "pair_group_hash": pair_group_hash(
            stage=stage,
            task_id=t["task_id"],
            base_task_id=t["base_task_id"],
            split=split,
            snapshot_id=snapshot_id,
            trajectory_policy="oracle_snapshot" if is_interface else "frozen_full_plan",
            experiment_freeze_hash=H5,
        ),
        "goal_hash": goal_hash(t["goal"]),
        "split": split,
        "arm": arm,
        "rollout_mode": "RAW",
        "state_source": "ORACLE_SNAPSHOT" if is_interface else "ACTUAL_TRAJECTORY",
        "step_index": 0,
        "attempt_index": 0,
        "planner_checkpoint_sha256": None,
        "planner_config_sha256": None,
        "planner_seed": None,
        "planner_support_status": "NOT_APPLICABLE",
        "planner_support_signature_hash": None,
        "plan_content_hash": None,
        "plan_artifact_hash": None,
        "plan_step_id": None,
        "semantic_ref": None,
        "semantic_artifact_hash": None,
        "semantic_resolution_status": "NOT_APPLICABLE",
        "semantic_resolution_method": None,
        "semantic_bank_hash": None,
        "semantic_top1_intent_id": None,
        "semantic_top1_similarity": None,
        "semantic_top2_intent_id": None,
        "semantic_top2_similarity": None,
        "semantic_margin": None,
        "semantic_min_similarity": None,
        "semantic_min_margin": None,
        "intent_text_hash": None if arm in {"I0_EQUAL_TOKENS_RAW", "E0_EQUAL_TOKENS_RAW"} else H2,
        "control_source_intent_id": None,
        "compatible_intent_ids": None,
        "control_mapping_hash": None,
        "intent_compatibility_hash": None,
        "control_certification_hash": None,
        "prompt_hash": prompt_bytes_hash(prompt_text.encode()),
        "base_prompt_hash": H2,
        "prompt_artifact": "artifacts/prompts/prompt.txt",
        "prompt_token_ids_hash": H3,
        "attention_mask_hash": H4,
        "position_ids_hash": H5,
        "guidance_block_token_ids_hash": H1,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "chat_template_hash": H1,
        "prompt_tokens_total": prompt_total,
        "attended_prompt_tokens": attended,
        "padded_sequence_length": prompt_total,
        "padding_side": "NONE",
        "added_block_tokens": 32,
        "matched_control_tokens": prompt_total if is_interface else 32,
        "state_before_hash": state_hash(t["initial"]),
        "state_after_hash": H2,
        "oracle_distance_before": 2 if is_interface else None,
        "oracle_distance_after": 1 if is_interface else None,
        "progress_success": True if is_interface else None,
        "raw_output": raw_text,
        "raw_output_bytes_hash": prompt_bytes_hash(raw_text.encode()),
        "parsed_llm_response": _response(),
        "raw_unmasked_action": None,
        "raw_unmasked_args": None,
        "mask_applied": False,
        "mask_hash": None,
        "parsed_typed_action": candidate,
        "candidate_typed_action": candidate,
        "parse_status": "PARSED",
        "validation_status": "VALID",
        "issue_code": None,
        "candidate_source": source,
        "tokens_in": attended,
        "tokens_out": 8,
        "queue_ms": 1.0,
        "inference_ms": 10.0,
        "total_ms": 11.0,
        "timestamp": "2026-07-24T12:00:00Z",
    }
    if not is_interface and arm != "E0_EQUAL_TOKENS_RAW":
        obj.update(
            planner_checkpoint_sha256=H1, planner_config_sha256=H2, planner_seed=202,
            planner_support_status="STRUCTURAL_SIGNATURE_SEEN", planner_support_signature_hash=H3,
            plan_content_hash=H2, plan_artifact_hash=H3, plan_step_id="S00",
        )
    if arm in {"I3_PLANNER_CURRENT_RAW"}:
        obj.update(
            planner_checkpoint_sha256=H1,
            planner_config_sha256=H2,
            planner_seed=202,
            planner_support_status="STRUCTURAL_SIGNATURE_SEEN",
            planner_support_signature_hash=H3,
            plan_step_id="S00",
            semantic_ref="latent+sha256://" + "a" * 64 + "#S00",
            semantic_artifact_hash=H4,
            semantic_resolution_status="RESOLVED",
            semantic_resolution_method="cosine-v1",
            semantic_bank_hash=H2,
            semantic_top1_intent_id=4,
            semantic_top1_similarity=0.8,
            semantic_top2_intent_id=3,
            semantic_top2_similarity=0.7,
            semantic_margin=0.1,
            semantic_min_similarity=0.5,
            semantic_min_margin=0.05,
        )
    if arm == "I2_SHUFFLED_RAW":
        obj.update(
            control_source_intent_id=3,
            compatible_intent_ids=[4],
            control_mapping_hash=H1,
            intent_compatibility_hash=H2,
            control_certification_hash=H3,
        )
    if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW":
        obj.update(control_mapping_hash=H1)
    if arm == "P_FULL_PLAN_REPLAY_RAW":
        obj.update(
            planner_checkpoint_sha256=H1,
            planner_config_sha256=H2,
            planner_seed=202,
            planner_support_status="STRUCTURAL_SIGNATURE_SEEN",
            planner_support_signature_hash=H3,
            plan_step_id="S00",
            semantic_resolution_status="NOT_APPLICABLE",
            raw_unmasked_action="PICK_UP",
            raw_unmasked_args=["@B0"],
            candidate_typed_action=candidate,
            parsed_typed_action=None,
            parsed_llm_response=None,
            parse_status="NOT_APPLICABLE",
            intent_text_hash=None,
        )
        for field in (
            "raw_output",
            "raw_output_bytes_hash",
            "prompt_hash",
            "base_prompt_hash",
            "prompt_artifact",
            "prompt_token_ids_hash",
            "attention_mask_hash",
            "position_ids_hash",
            "guidance_block_token_ids_hash",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "chat_template_hash",
            "prompt_tokens_total",
            "attended_prompt_tokens",
            "padded_sequence_length",
            "padding_side",
            "added_block_tokens",
            "matched_control_tokens",
            "tokens_in",
            "tokens_out",
            "queue_ms",
            "inference_ms",
            "total_ms",
        ):
            obj[field] = None
    return obj


def unresolved_attempt() -> dict[str, Any]:
    obj = attempt("I3_PLANNER_CURRENT_RAW", "STAGE1A_INTERFACE")
    obj.update(
        semantic_resolution_status="UNRESOLVED",
        candidate_source="PLANNER_RESOLUTION_FAILURE",
        issue_code="SEMANTIC_UNRESOLVED",
        raw_output=None,
        raw_output_bytes_hash=None,
        parsed_llm_response=None,
        parsed_typed_action=None,
        candidate_typed_action=None,
        parse_status="NOT_APPLICABLE",
        validation_status="NOT_APPLICABLE",
        state_after_hash=None,
    )
    for field in (
        "prompt_hash",
        "base_prompt_hash",
        "prompt_artifact",
        "prompt_token_ids_hash",
        "attention_mask_hash",
        "position_ids_hash",
        "guidance_block_token_ids_hash",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_hash",
        "prompt_tokens_total",
        "attended_prompt_tokens",
        "padded_sequence_length",
        "added_block_tokens",
        "matched_control_tokens",
        "tokens_in",
        "tokens_out",
        "queue_ms",
        "inference_ms",
        "total_ms",
    ):
        obj[field] = None
    return obj


def episode_from_attempt(a: dict[str, Any], *, goal_success: bool | None = None, terminal_error: str | None = None) -> dict[str, Any]:
    is_interface = a["stage"] == "STAGE1A_INTERFACE"
    executed = int(
        a["validation_status"] == "VALID"
        and a["candidate_typed_action"] is not None
        and a["candidate_typed_action"]["action"] != "END"
    )
    if not is_interface and goal_success is None:
        goal_success = True
    return {
        "schema_version": V,
        "run_id": a["run_id"],
        "episode_id": a["episode_id"],
        "trajectory_id": a["trajectory_id"],
        "stage": a["stage"],
        "task_id": a["task_id"],
        "base_task_id": a["base_task_id"],
        "snapshot_id": a["snapshot_id"],
        "canonical_task_hash": a["canonical_task_hash"],
        "split": a["split"],
        "arm": a["arm"],
        "planner_seed": a.get("planner_seed"),
        "rollout_mode": a["rollout_mode"],
        "trajectory_policy": a["trajectory_policy"],
        "experiment_freeze_hash": a["experiment_freeze_hash"],
        "goal_success": None if is_interface else goal_success,
        "progress_success": a["progress_success"] if is_interface else None,
        "steps_accepted": executed,
        "attempts_total": 1,
        "retries_total": 0,
        "planner_calls": int(a["arm"] in CrossObjectContractValidator.PLANNER_INTENT_ARMS) if is_interface else (0 if a["arm"] in {"E0_EQUAL_TOKENS_RAW", "E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"} else 1),
        "oracle_length": 2,
        "executed_length": executed,
        "optimality_ratio": None,
        "semantic_resolver_calls": int(a["semantic_resolution_status"] in {"RESOLVED", "UNRESOLVED"}),
        "semantic_unresolved_count": int(a["semantic_resolution_status"] == "UNRESOLVED"),
        "unseen_support_signature_count": int(a["planner_support_status"] == "UNSEEN_SIGNATURE"),
        "terminal_error": terminal_error,
        "error_tags": [] if terminal_error is None else [terminal_error],
        "total_tokens_in": a["tokens_in"] or 0,
        "total_attended_tokens": a["attended_prompt_tokens"] or 0,
        "total_tokens_out": a["tokens_out"] or 0,
        "total_latency_ms": a["total_ms"] or 0,
        "final_state_hash": a["state_after_hash"] or a["state_before_hash"],
        "pair_group_hash": a["pair_group_hash"],
        "prompt_budget_violation_count": 0,
        "contract_violation_count": 0,
        "hash_violation_count": 0,
        "episode_plan_manifest_hash": a.get("episode_plan_manifest_hash"),
        "plan_generation_status": a.get("plan_generation_status", "NONE"),
        "replay_context": a.get("replay_context"),
        "plan_positions_consumed": 0 if is_interface or a["arm"] == "E0_EQUAL_TOKENS_RAW" else executed,
        "plan_tokens_in_actual": 0,
        "plan_tokens_out_actual": 0,
        "plan_latency_ms_actual": 0.0,
        "plan_tokens_in_attributed": 0,
        "plan_tokens_out_attributed": 0,
        "plan_latency_ms_attributed": 0.0,
        "executor_tokens_in": a["tokens_in"] or 0,
        "executor_tokens_out": a["tokens_out"] or 0,
        "executor_latency_ms": a["total_ms"] or 0,
        "flops_actual": 0.0,
        "flops_attributed": 0.0,
        "flops_cap_per_task": None,
        "flops_budget_exhausted": False,
    }


# Late import avoids circular type use at function definition time.
from .cross_object_validator import CrossObjectContractValidator  # noqa: E402
