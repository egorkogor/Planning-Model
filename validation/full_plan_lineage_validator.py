"""Fail-closed validation of frozen full-plan execution lineage (v1.14).

This module verifies relations that local JSON Schemas cannot express:
one plan-generation call before execution, immutable WorkPlan reuse, sequential
position consumption, no replanning, phase-specific replay provenance, and
separate actual/attributed planning cost.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from validation.hashing import hash_json, plan_artifact_hash, plan_content_hash
from validation.capacity_validator import validate_compute_profile
from validation.random_codebook_validator import validate_random_codebook

STAGE1B_ARMS = {
    "E0_EQUAL_TOKENS_RAW",
    "E1_A3_FULL_PLAN_RAW",
    "E2_SHUFFLED_A3_FULL_PLAN_RAW",
    "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW",
    "E4_A2C_STRUCTURED_FULL_PLAN_RAW",
    "E5_SELF_PLAN_RAW",
    "P_FULL_PLAN_REPLAY_RAW",
}
PLAN_ARMS = STAGE1B_ARMS - {"E0_EQUAL_TOKENS_RAW"}
GENERATING_ARMS = {
    "PLANNER_A3_RAW": "MICRO_PLANNER_A3",
    "E1_A3_FULL_PLAN_RAW": "MICRO_PLANNER_A3",
    "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW": "MICRO_PLANNER_A3R",
    "E4_A2C_STRUCTURED_FULL_PLAN_RAW": "MICRO_PLANNER_A2C",
    "E5_SELF_PLAN_RAW": "SELF_PLAN_LLM",
}
REUSED_ARMS = {"E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"}
ROOT = Path(__file__).resolve().parents[1]
SHUFFLE_CONTRACT = ROOT / "docs/controls/full_plan_shuffle_contract_v1.yaml"


def _err(path: str, message: str) -> str:
    return f"{path}: {message}"


def _without(obj: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in keys}


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cost_tuple(cost: Mapping[str, Any]) -> tuple[int, int, float]:
    return int(cost["tokens_in"]), int(cost["tokens_out"]), float(cost["latency_ms"])


def _signature_key(signature: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(signature.items()))


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_flops_accounting(episode: Mapping[str, Any], arm_profile: Mapping[str, Any], cap: float) -> list[str]:
    errors: list[str] = []
    arm = str(episode.get("arm"))
    planner_calls = int(episode.get("planner_calls", 0))
    plan_actual_tokens = int(episode.get("plan_tokens_in_actual", 0)) + int(episode.get("plan_tokens_out_actual", 0))
    plan_attributed_tokens = int(episode.get("plan_tokens_in_attributed", 0)) + int(episode.get("plan_tokens_out_attributed", 0))
    executor_in = int(episode.get("executor_tokens_in", 0))
    executor_out = int(episode.get("executor_tokens_out", 0))
    fixed = float(arm_profile["fixed_flops_per_task"])
    planner = float(arm_profile["planner_flops_per_call"])
    plan_token = float(arm_profile["plan_token_flops"])
    executor_input = float(arm_profile["executor_input_token_flops"])
    executor_output = float(arm_profile["executor_output_token_flops"])
    actual = fixed + planner_calls * planner + plan_actual_tokens * plan_token + executor_in * executor_input + executor_out * executor_output
    attributed_plan_calls = 0 if arm == "E0_EQUAL_TOKENS_RAW" else 1
    attributed = fixed + attributed_plan_calls * planner + plan_attributed_tokens * plan_token + executor_in * executor_input + executor_out * executor_output
    for field, expected in (("flops_actual", actual), ("flops_attributed", attributed), ("flops_cap_per_task", cap)):
        try:
            if not math.isclose(float(episode.get(field)), expected, rel_tol=1e-12, abs_tol=1e-6):
                errors.append(_err(field, f"must equal recomputed locked value {expected}"))
        except Exception:
            errors.append(_err(field, "missing or non-numeric"))
    if actual > attributed + 1e-6:
        errors.append(_err("flops_actual", "cannot exceed attributed FLOPs"))
    exhausted = attributed > cap + 1e-6
    if episode.get("flops_budget_exhausted") is not exhausted:
        errors.append(_err("flops_budget_exhausted", "does not match recomputed attributed FLOPs cap"))
    if exhausted:
        if episode.get("terminal_error") != "FLOPS_BUDGET_EXHAUSTED" or episode.get("goal_success") is not False:
            errors.append(_err("terminal_error", "cap exhaustion must remain a typed paired failure"))
    elif episode.get("terminal_error") == "FLOPS_BUDGET_EXHAUSTED":
        errors.append(_err("terminal_error", "FLOPS_BUDGET_EXHAUSTED declared below cap"))
    return errors


def _shuffle_mapping_errors(
    mapping: Mapping[str, Any], *, manifest: Mapping[str, Any], work_plan: Mapping[str, Any],
    episode: Mapping[str, Any], attempts: list[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if mapping.get("mapping_hash") != hash_json(_without(mapping, "mapping_hash")):
        errors.append(_err("control_artifact.mapping_hash", "canonical hash mismatch"))
    contract_hash = _file_digest(SHUFFLE_CONTRACT)
    if mapping.get("contract_sha256") != contract_hash:
        errors.append(_err("control_artifact.contract_sha256", "shuffle contract hash mismatch"))
    for field in ("run_id", "task_id"):
        if mapping.get(field) != manifest.get(field):
            errors.append(_err(f"control_artifact.{field}", "differs from EpisodePlanManifest"))
    if mapping.get("source_episode_plan_manifest_hash") != manifest.get("source_episode_plan_manifest_hash"):
        errors.append(_err("control_artifact.source_episode_plan_manifest_hash", "source manifest mismatch"))
    if mapping.get("source_plan_content_hash") != manifest.get("work_plan_content_hash"):
        errors.append(_err("control_artifact.source_plan_content_hash", "source plan mismatch"))
    seed_payload = {
        "run_id": manifest.get("run_id"),
        "task_id": manifest.get("task_id"),
        "source_episode_plan_manifest_hash": manifest.get("source_episode_plan_manifest_hash"),
        "source_plan_content_hash": manifest.get("work_plan_content_hash"),
        "contract_sha256": contract_hash,
    }
    seed_hash = hash_json(seed_payload)
    if mapping.get("seed_payload_hash") != seed_hash:
        errors.append(_err("control_artifact.seed_payload_hash", "deterministic payload hash mismatch"))
    non_end = [row for row in work_plan.get("steps", []) if row.get("typed_action", {}).get("action") != "END"]
    n = len(non_end)
    if mapping.get("plan_non_end_length") != n:
        errors.append(_err("control_artifact.plan_non_end_length", "differs from frozen WorkPlan"))
    permutation = mapping.get("permutation", [])
    if n < 2:
        if mapping.get("status") != "DEGENERATE" or mapping.get("rotation_offset") is not None or permutation != []:
            errors.append(_err("control_artifact", "length zero/one must be deterministic DEGENERATE mapping"))
        if manifest.get("control_status") != "DEGENERATE":
            errors.append(_err("control_status", "must be DEGENERATE for unshufflable plan"))
        if attempts or episode.get("attempts_total") != 0 or episode.get("goal_success") is not False or episode.get("terminal_error") != "CONTROL_UNAVAILABLE":
            errors.append(_err("episode", "degenerate shuffle must remain zero-execution CONTROL_UNAVAILABLE paired failure"))
    else:
        offset = 1 + int.from_bytes(hashlib.sha256(seed_hash.encode("utf-8")).digest()[:8], "big") % (n - 1)
        expected = [(i + offset) % n for i in range(n)]
        if mapping.get("status") != "VALID" or mapping.get("rotation_offset") != offset or permutation != expected:
            errors.append(_err("control_artifact.permutation", "does not equal locked nonzero cyclic rotation"))
        if manifest.get("control_status") != "VALID":
            errors.append(_err("control_status", "must be VALID for deterministic derangement"))
        for i, attempt in enumerate(attempts):
            if attempt.get("control_mapping_hash") != mapping.get("mapping_hash"):
                errors.append(_err(f"attempts[{i}].control_mapping_hash", "does not bind frozen shuffle mapping"))
            if any(attempt.get(field) is not None for field in ("control_source_intent_id", "compatible_intent_ids", "intent_compatibility_hash", "control_certification_hash")):
                errors.append(_err(f"attempts[{i}].control_*", "position shuffle must not claim incompatible-intent control fields"))
            if i < len(expected) and attempt.get("guidance_source_position_index") != expected[i]:
                errors.append(_err(f"attempts[{i}].guidance_source_position_index", "does not follow frozen shuffle permutation"))
    return errors


def _bank_signatures(bank: Mapping[str, Any] | None) -> set[tuple[tuple[str, Any], ...]] | None:
    if bank is None:
        return None
    rows = bank.get("signatures") or bank.get("entries")
    if not isinstance(rows, list):
        return set()
    out = set()
    for row in rows:
        sig = row.get("semantic_signature", row) if isinstance(row, dict) else None
        if isinstance(sig, dict):
            out.add(_signature_key(sig))
    return out


def validate_episode_plan_manifest(
    manifest: Mapping[str, Any],
    *,
    work_plan: Mapping[str, Any] | None,
    episode: Mapping[str, Any],
    attempts: list[Mapping[str, Any]],
    source_manifest: Mapping[str, Any] | None = None,
    signature_bank: Mapping[str, Any] | None = None,
    control_artifact: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    arm = str(manifest.get("arm"))

    expected_hash = hash_json(_without(manifest, "manifest_hash"))
    if manifest.get("manifest_hash") != expected_hash:
        errors.append(_err("manifest_hash", "does not match canonical manifest payload"))

    for field in (
        "run_id", "episode_id", "trajectory_id", "stage", "task_id",
        "base_task_id", "canonical_task_hash", "split", "arm",
    ):
        if manifest.get(field) != episode.get(field):
            errors.append(_err(field, "EpisodePlanManifest differs from EpisodeLog"))

    try:
        started = _parse_time(str(manifest["generation_started_at"]))
        completed = _parse_time(str(manifest["generation_completed_at"]))
        if completed < started:
            errors.append(_err("generation_completed_at", "precedes generation_started_at"))
        if attempts and _parse_time(str(attempts[0]["timestamp"])) < completed:
            errors.append(_err("generated_before_execution", "first attempt predates completed plan generation"))
    except Exception as exc:
        errors.append(_err("generation timestamps", str(exc)))

    if manifest.get("generated_before_execution") is not True:
        errors.append(_err("generated_before_execution", "must be true"))

    if arm in GENERATING_ARMS:
        if manifest.get("generator_type") != GENERATING_ARMS[arm]:
            errors.append(_err("generator_type", f"expected {GENERATING_ARMS[arm]}"))
        if manifest.get("planner_call_count") != 1:
            errors.append(_err("planner_call_count", "generated plan requires exactly one pre-execution call"))
        if source_manifest is not None or manifest.get("source_episode_plan_manifest_hash") is not None:
            errors.append(_err("source_episode_plan_manifest_hash", "newly generated plan cannot claim reused source"))
    elif arm in REUSED_ARMS:
        if manifest.get("generator_type") != "REUSED_E1_A3":
            errors.append(_err("generator_type", "reused control/replay must use REUSED_E1_A3"))
        if manifest.get("planner_call_count") != 0:
            errors.append(_err("planner_call_count", "reused plan must not call Planner again"))
        if source_manifest is None:
            errors.append(_err("source_episode_plan_manifest_hash", "source manifest is required"))
        else:
            if manifest.get("source_episode_plan_manifest_hash") != source_manifest.get("manifest_hash"):
                errors.append(_err("source_episode_plan_manifest_hash", "does not bind the supplied source manifest"))
            for field in ("run_id", "task_id", "base_task_id", "canonical_task_hash", "split"):
                if manifest.get(field) != source_manifest.get(field):
                    errors.append(_err(f"source.{field}", "reused source has different identity"))
            for field in ("work_plan_content_hash", "work_plan_artifact_hash"):
                if manifest.get(field) != source_manifest.get(field):
                    errors.append(_err(field, "reused arm must consume the exact source WorkPlan"))
    else:
        errors.append(_err("arm", f"unsupported full-plan arm {arm}"))

    replay_context = manifest.get("replay_context")
    if arm == "P_FULL_PLAN_REPLAY_RAW":
        if replay_context == "PLANNER_CONFIRMATORY_A3":
            expected_source = "PLANNER_A3_RAW"
        elif replay_context == "STAGE1B_E1":
            expected_source = "E1_A3_FULL_PLAN_RAW"
        else:
            expected_source = None
            errors.append(_err("replay_context", "must identify Planner-confirmatory A3 or Stage1B E1"))
        if expected_source and manifest.get("source_arm") != expected_source:
            errors.append(_err("source_arm", f"expected {expected_source} for replay context"))
    elif replay_context is not None:
        errors.append(_err("replay_context", "only replay arm may carry replay context"))

    if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW":
        if manifest.get("source_arm") != "E1_A3_FULL_PLAN_RAW":
            errors.append(_err("source_arm", "shuffled plan must derive from E1 without task exclusion"))
        if manifest.get("control_status") not in {"VALID", "DEGENERATE"}:
            errors.append(_err("control_status", "shuffled arm must log VALID or DEGENERATE"))
    elif manifest.get("control_status") != "NOT_APPLICABLE":
        errors.append(_err("control_status", "non-shuffled arm must be NOT_APPLICABLE"))

    if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW", "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW"}:
        if control_artifact is None:
            errors.append(_err("control_artifact", "required for E2/E3"))
        else:
            artifact_hash = control_artifact.get("mapping_hash") if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW" else control_artifact.get("manifest_hash")
            if manifest.get("control_artifact_hash") != artifact_hash:
                errors.append(_err("control_artifact_hash", "does not bind supplied control artifact"))
            expected_path = "semantic_bank/random-codebook/manifest.json" if arm == "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW" else manifest.get("control_artifact_path")
            if arm == "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW" and manifest.get("control_artifact_path") != expected_path:
                errors.append(_err("control_artifact_path", "A3r must bind frozen random codebook manifest"))
    elif control_artifact is not None or manifest.get("control_artifact_path") is not None or manifest.get("control_artifact_hash") is not None:
        errors.append(_err("control_artifact", "only E2/E3 may bind a control artifact"))

    if episode.get("episode_plan_manifest_hash") != manifest.get("manifest_hash"):
        errors.append(_err("episode.episode_plan_manifest_hash", "does not bind manifest"))
    if episode.get("planner_calls") != manifest.get("planner_call_count"):
        errors.append(_err("episode.planner_calls", "does not equal EpisodePlanManifest planner_call_count"))

    status = manifest.get("plan_status")
    if status == "FAILED":
        if work_plan is not None:
            errors.append(_err("work_plan", "FAILED plan generation must not produce a WorkPlan"))
        if attempts:
            errors.append(_err("attempts", "FAILED plan generation must produce zero executor attempts"))
        if episode.get("attempts_total") != 0 or episode.get("steps_accepted") != 0 or episode.get("executed_length") != 0:
            errors.append(_err("episode", "FAILED plan must remain a paired zero-execution outcome"))
        if episode.get("goal_success") is not False:
            errors.append(_err("goal_success", "FAILED plan must be counted as failure, not excluded"))
        if manifest.get("generation_failure_code") is None or episode.get("terminal_error") != manifest.get("generation_failure_code"):
            errors.append(_err("generation_failure_code", "must equal EpisodeLog terminal_error"))
        if manifest.get("work_plan_content_hash") is not None or manifest.get("work_plan_artifact_hash") is not None:
            errors.append(_err("work_plan_*", "FAILED plan hashes must be null"))
        if episode.get("plan_generation_status") != "FAILED" or episode.get("plan_positions_consumed") != 0:
            errors.append(_err("episode.plan_generation_status", "FAILED manifest requires FAILED zero-position EpisodeLog"))
        if any(int(episode.get(field, -1)) != 0 for field in ("executor_tokens_in", "executor_tokens_out")) or float(episode.get("executor_latency_ms", -1)) != 0.0:
            errors.append(_err("episode.executor_*", "FAILED plan must have zero executor cost"))
        actual = _cost_tuple(manifest["actual_cost"]); attributed = _cost_tuple(manifest["attributed_cost"])
        episode_actual = (int(episode.get("plan_tokens_in_actual", -1)), int(episode.get("plan_tokens_out_actual", -1)), float(episode.get("plan_latency_ms_actual", -1)))
        episode_attr = (int(episode.get("plan_tokens_in_attributed", -1)), int(episode.get("plan_tokens_out_attributed", -1)), float(episode.get("plan_latency_ms_attributed", -1)))
        if actual != episode_actual or attributed != episode_attr:
            errors.append(_err("episode.plan_cost", "FAILED plan cost does not match EpisodePlanManifest"))
        return errors

    if status != "READY":
        errors.append(_err("plan_status", "must be READY or FAILED"))
        return errors
    if work_plan is None:
        errors.append(_err("work_plan", "READY manifest requires WorkPlan"))
        return errors

    if work_plan.get("plan_content_hash") != plan_content_hash(work_plan):
        errors.append(_err("work_plan.plan_content_hash", "hash mismatch"))
    if work_plan.get("plan_artifact_hash") != plan_artifact_hash(work_plan):
        errors.append(_err("work_plan.plan_artifact_hash", "hash mismatch"))
    if manifest.get("work_plan_content_hash") != work_plan.get("plan_content_hash"):
        errors.append(_err("work_plan_content_hash", "manifest differs from WorkPlan"))
    if manifest.get("work_plan_artifact_hash") != work_plan.get("plan_artifact_hash"):
        errors.append(_err("work_plan_artifact_hash", "manifest differs from WorkPlan"))
    for field in ("task_id", "canonical_task_hash"):
        if manifest.get(field) != work_plan.get(field):
            errors.append(_err(f"work_plan.{field}", "identity mismatch"))
    if manifest.get("initial_state_hash") != work_plan.get("state_hash"):
        errors.append(_err("initial_state_hash", "plan was not generated from episode initial state"))

    if arm == "E4_A2C_STRUCTURED_FULL_PLAN_RAW":
        if manifest.get("semantic_signature_bank_hash") is None:
            errors.append(_err("semantic_signature_bank_hash", "A2c plan requires frozen signature bank"))
        if signature_bank is None:
            errors.append(_err("semantic_signature_bank", "A2c signature bank artifact must be supplied to the lineage validator"))
        else:
            if manifest.get("semantic_signature_bank_hash") != hash_json(signature_bank):
                errors.append(_err("semantic_signature_bank_hash", "does not match the supplied frozen signature bank"))
            known = _bank_signatures(signature_bank)
            for i, step in enumerate(work_plan.get("steps", [])):
                if step.get("typed_action", {}).get("action") == "END":
                    continue
                sig = step.get("semantic_signature")
                if not isinstance(sig, dict) or _signature_key(sig) not in known:
                    errors.append(_err(f"work_plan.steps[{i}].semantic_signature", "unknown A2c combination must fail SEMANTIC_UNRESOLVED, never heuristic-map"))
    elif signature_bank is not None or manifest.get("semantic_signature_bank_hash") is not None:
        errors.append(_err("semantic_signature_bank", "only A2c may bind the structured signature bank"))

    expected_plan = {
        "E1_A3_FULL_PLAN_RAW": ("A3", "CONTINUOUS_LATENT"),
        "E2_SHUFFLED_A3_FULL_PLAN_RAW": ("A3", "CONTINUOUS_LATENT"),
        "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW": ("A3r", "CONTINUOUS_LATENT"),
        "E4_A2C_STRUCTURED_FULL_PLAN_RAW": ("A2c", "STRUCTURED_DISCRETE"),
        "E5_SELF_PLAN_RAW": ("SELF_PLAN", "DISCRETE_INTENT"),
        "P_FULL_PLAN_REPLAY_RAW": ("A3", "CONTINUOUS_LATENT"),
        "PLANNER_A3_RAW": ("A3", "CONTINUOUS_LATENT"),
    }.get(arm)
    if expected_plan and (work_plan.get("planner_variant"), work_plan.get("representation")) != expected_plan:
        errors.append(_err("work_plan.variant", f"expected {expected_plan}"))

    non_end_steps = [s for s in work_plan.get("steps", []) if s.get("typed_action", {}).get("action") != "END"]
    if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW" and control_artifact is not None:
        errors.extend(_shuffle_mapping_errors(control_artifact, manifest=manifest, work_plan=work_plan, episode=episode, attempts=attempts))
    if arm == "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW" and control_artifact is not None:
        known = {row.get("signature_sha256") for row in control_artifact.get("entries", [])}
        for i, step in enumerate(non_end_steps):
            signature = step.get("semantic_signature")
            if not isinstance(signature, dict) or hash_json(signature) not in known:
                errors.append(_err(f"work_plan.steps[{i}].semantic_signature", "A3r signature missing from frozen random codebook"))

    positions = [a.get("plan_position_index") for a in attempts]
    if positions != list(range(len(attempts))):
        errors.append(_err("attempts.plan_position_index", "positions must be contiguous from zero"))
    if len(attempts) > len(non_end_steps):
        errors.append(_err("attempts", "execution consumed beyond frozen plan length"))
    for i, attempt in enumerate(attempts):
        for field in ("run_id", "episode_id", "trajectory_id", "stage", "task_id", "base_task_id", "canonical_task_hash", "split", "arm"):
            if attempt.get(field) != episode.get(field):
                errors.append(_err(f"attempts[{i}].{field}", "differs from episode"))
        if attempt.get("step_index") != i or attempt.get("attempt_index") != 0:
            errors.append(_err(f"attempts[{i}].step_index|attempt_index", "full-plan raw execution requires one attempt per contiguous position"))
        if attempt.get("episode_plan_manifest_hash") != manifest.get("manifest_hash"):
            errors.append(_err(f"attempts[{i}].episode_plan_manifest_hash", "does not bind frozen manifest"))
        if attempt.get("plan_content_hash") != work_plan.get("plan_content_hash") or attempt.get("plan_artifact_hash") != work_plan.get("plan_artifact_hash"):
            errors.append(_err(f"attempts[{i}].plan_*", "does not bind frozen WorkPlan"))
        if attempt.get("replanning_observed") is not False:
            errors.append(_err(f"attempts[{i}].replanning_observed", "must be false"))
        if i < len(non_end_steps):
            step = non_end_steps[i]
            if attempt.get("plan_step_id") != step.get("step_id"):
                errors.append(_err(f"attempts[{i}].plan_step_id", "does not match frozen plan position"))
            if arm == "P_FULL_PLAN_REPLAY_RAW" and attempt.get("candidate_typed_action") != step.get("typed_action"):
                errors.append(_err(f"attempts[{i}].candidate_typed_action", "replay must execute exact frozen typed action"))
            if arm == "P_FULL_PLAN_REPLAY_RAW":
                if any(attempt.get(field) is not None for field in ("guidance_source_position_index", "guidance_source_step_id", "guidance_source_semantic_ref")):
                    errors.append(_err(f"attempts[{i}].guidance_source_*", "raw replay has no LLM guidance"))
            else:
                source_position = attempt.get("guidance_source_position_index")
                if arm != "E2_SHUFFLED_A3_FULL_PLAN_RAW" and source_position != i:
                    errors.append(_err(f"attempts[{i}].guidance_source_position_index", "must equal its own frozen plan position"))
                if isinstance(source_position, int) and 0 <= source_position < len(non_end_steps):
                    source_step = non_end_steps[source_position]
                    if attempt.get("guidance_source_step_id") != source_step.get("step_id"):
                        errors.append(_err(f"attempts[{i}].guidance_source_step_id", "does not match frozen source plan position"))
                    if attempt.get("guidance_source_semantic_ref") != source_step.get("semantic_ref"):
                        errors.append(_err(f"attempts[{i}].guidance_source_semantic_ref", "does not match frozen source plan position"))
                else:
                    errors.append(_err(f"attempts[{i}].guidance_source_position_index", "is outside frozen non-END plan"))

    if episode.get("plan_generation_status") != "READY":
        errors.append(_err("episode.plan_generation_status", "must be READY"))
    if episode.get("plan_positions_consumed") != len(attempts):
        errors.append(_err("episode.plan_positions_consumed", "does not equal attempt positions"))

    executor_cost = (
        sum(int(a.get("tokens_in") or 0) for a in attempts),
        sum(int(a.get("tokens_out") or 0) for a in attempts),
        sum(float(a.get("total_ms") or 0.0) for a in attempts),
    )
    logged_executor = (
        int(episode.get("executor_tokens_in", -1)),
        int(episode.get("executor_tokens_out", -1)),
        float(episode.get("executor_latency_ms", -1)),
    )
    if executor_cost != logged_executor:
        errors.append(_err("episode.executor_*", "does not equal executor attempt cost"))

    actual = _cost_tuple(manifest["actual_cost"])
    attributed = _cost_tuple(manifest["attributed_cost"])
    episode_actual = (
        int(episode.get("plan_tokens_in_actual", -1)),
        int(episode.get("plan_tokens_out_actual", -1)),
        float(episode.get("plan_latency_ms_actual", -1)),
    )
    episode_attr = (
        int(episode.get("plan_tokens_in_attributed", -1)),
        int(episode.get("plan_tokens_out_attributed", -1)),
        float(episode.get("plan_latency_ms_attributed", -1)),
    )
    if actual != episode_actual:
        errors.append(_err("episode.plan_*_actual", "does not equal manifest actual planning cost"))
    if attributed != episode_attr:
        errors.append(_err("episode.plan_*_attributed", "does not equal manifest attributed planning cost"))
    if arm in REUSED_ARMS:
        if actual != (0, 0, 0.0):
            errors.append(_err("actual_cost", "reused plan must have zero new planning cost"))
        if source_manifest is not None and attributed != _cost_tuple(source_manifest["actual_cost"]):
            errors.append(_err("attributed_cost", "must attribute source E1/A3 plan cost"))
    elif actual != attributed:
        errors.append(_err("attributed_cost", "newly generated plan actual and attributed cost must match"))
    if arm == "E5_SELF_PLAN_RAW" and actual[0] + actual[1] <= 0:
        errors.append(_err("actual_cost", "self-plan tokens must be logged separately from executor tokens"))
    return errors


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"path escapes repository: {relative}")
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_lineage_index(root: Path, index: Mapping[str, Any], *, expected_stage: str) -> list[str]:
    errors: list[str] = []
    if index.get("stage") != expected_stage:
        errors.append(_err("index.stage", f"expected {expected_stage}"))
    if index.get("index_hash") != hash_json(_without(index, "index_hash")):
        errors.append(_err("index_hash", "does not match canonical index payload"))

    arm_compute: dict[str, Mapping[str, Any]] = {}
    flops_cap: float | None = None
    if expected_stage == "STAGE1B":
        try:
            rel = index.get("compute_profile")
            if rel != "reports/compute-profile.json":
                raise ValueError("Stage1B lineage must use reports/compute-profile.json")
            path = _safe_path(root, rel)
            if index.get("compute_profile_sha256") != _file_digest(path):
                errors.append(_err("compute_profile_sha256", "does not match compute profile artifact"))
            profile = json.loads(path.read_text(encoding="utf-8"))
            errors.extend(_err("compute_profile", e) for e in validate_compute_profile(profile, root))
            arm_compute = {str(row["arm"]): row for row in profile["stage1b_inference"]["arms"]}
            flops_cap = float(profile["stage1b_inference"]["flops_cap_per_task"])
        except Exception as exc:
            errors.append(_err("compute_profile", str(exc)))

    loaded: list[tuple[Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None]] = []
    manifests_by_hash: dict[str, Mapping[str, Any]] = {}
    for i, row in enumerate(index.get("records", [])):
        try:
            manifest = None if row.get("episode_plan_manifest") is None else json.loads(_safe_path(root, row["episode_plan_manifest"]).read_text(encoding="utf-8"))
            plan = None if row.get("work_plan") is None else json.loads(_safe_path(root, row["work_plan"]).read_text(encoding="utf-8"))
            episode = json.loads(_safe_path(root, row["episode_log"]).read_text(encoding="utf-8"))
            attempts = _load_jsonl(_safe_path(root, row["attempt_log"]))
            signature_bank = None if row.get("semantic_signature_bank") is None else json.loads(_safe_path(root, row["semantic_signature_bank"]).read_text(encoding="utf-8"))
            control_artifact = None if row.get("control_artifact") is None else json.loads(_safe_path(root, row["control_artifact"]).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(_err(f"records[{i}]", str(exc)))
            continue
        if manifest is not None:
            manifests_by_hash[str(manifest.get("manifest_hash"))] = manifest
        if row.get("arm") == "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW" and control_artifact is not None:
            errors.extend(_err(f"records[{i}].control_artifact", e) for e in validate_random_codebook(root, control_artifact))
        loaded.append((row, manifest, episode, attempts, plan, signature_bank, control_artifact))

    if expected_stage == "STAGE1B":
        by_task: dict[str, list[tuple[Any, ...]]] = {}
        for item in loaded:
            by_task.setdefault(str(item[0].get("task_id")), []).append(item)
        for task_id, rows in by_task.items():
            arms = [str(row[0].get("arm")) for row in rows]
            if set(arms) != STAGE1B_ARMS or len(arms) != len(STAGE1B_ARMS):
                errors.append(_err(f"task[{task_id}].arms", f"expected exactly {sorted(STAGE1B_ARMS)}; post-treatment exclusion is forbidden"))

    if expected_stage == "PLANNER":
        by_task: dict[str, list[tuple[Any, ...]]] = {}
        for item in loaded:
            by_task.setdefault(str(item[0].get("task_id")), []).append(item)
        for task_id, rows in by_task.items():
            arms = {str(row[0].get("arm")) for row in rows}
            if not {"PLANNER_A3_RAW", "P_FULL_PLAN_REPLAY_RAW"}.issubset(arms):
                errors.append(_err(f"task[{task_id}].arms", "Planner confirmatory replay requires precomputed A3 plan and P replay"))

    for i, (row, manifest, episode, attempts, plan, signature_bank, control_artifact) in enumerate(loaded):
        arm = str(row.get("arm"))
        if arm != episode.get("arm") or row.get("task_id") != episode.get("task_id"):
            errors.append(_err(f"records[{i}]", "index identity differs from EpisodeLog"))
        if expected_stage == "STAGE1B" and flops_cap is not None:
            profile_row = arm_compute.get(arm)
            if profile_row is None:
                errors.append(_err(f"records[{i}].arm", "missing from locked compute profile"))
            else:
                errors.extend(f"records[{i}].{e}" for e in _validate_flops_accounting(episode, profile_row, flops_cap))
        if arm == "E0_EQUAL_TOKENS_RAW":
            if manifest is not None or plan is not None:
                errors.append(_err(f"records[{i}]", "E0 must not carry a plan"))
            if episode.get("episode_plan_manifest_hash") is not None or episode.get("planner_calls") != 0:
                errors.append(_err(f"records[{i}].episode", "E0 plan lineage must be null"))
            continue
        if manifest is None:
            errors.append(_err(f"records[{i}]", "plan arm missing EpisodePlanManifest"))
            continue
        source = manifests_by_hash.get(str(manifest.get("source_episode_plan_manifest_hash")))
        errors.extend(f"records[{i}].{e}" for e in validate_episode_plan_manifest(
            manifest, work_plan=plan, episode=episode, attempts=attempts, source_manifest=source, signature_bank=signature_bank, control_artifact=control_artifact
        ))
    return errors
