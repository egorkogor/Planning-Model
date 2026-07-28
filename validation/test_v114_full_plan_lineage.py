from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from validation.fixtures import H1, H2, H3, H4, attempt, episode_from_attempt, plan_manifest, step, task
from validation.full_plan_lineage_validator import (
    STAGE1B_ARMS,
    validate_episode_plan_manifest,
    validate_lineage_index,
)
from validation.hashing import hash_json, plan_artifact_hash, plan_content_hash
from validation.random_codebook_validator import code_hex
from validation.test_v118_architecture_evidence import _write_final_matrix, _write_flops_sensitivity_matrix


def _plan(variant: str = "A3") -> dict:
    base, _ = plan_manifest()
    base["run_id"] = "run-1"
    base["stage"] = "STAGE1B_END_TO_END"
    if variant != "A3":
        base["planner_variant"] = variant
        rep = {
            "A1": "TOKEN_GRAMMAR",
            "A2": "TYPED_ONLY",
            "A2b": "DISCRETE_INTENT",
            "A2c": "STRUCTURED_DISCRETE",
            "A3r": "CONTINUOUS_LATENT",
            "A4": "CONTINUOUS_LATENT",
            "A5": "CONTINUOUS_LATENT",
            "SELF_PLAN": "DISCRETE_INTENT",
        }[variant]
        base["representation"] = rep
        base["steps"] = [
            step(0, "PICK_UP", ("@B0",), variant),
            step(1, "STACK", ("@B0", "@B1"), variant),
            step(2, "END", (), variant),
        ]
        base["semantic_artifact_manifest_sha256"] = H4 if rep == "CONTINUOUS_LATENT" else None
    base["plan_content_hash"] = plan_content_hash(base)
    base["plan_artifact_hash"] = plan_artifact_hash(base)
    return base



def _signature_bank(plan: dict) -> dict:
    return {
        "schema_version": "work-planner/1.20",
        "bank_id": "structured-signatures-v1",
        "signatures": [
            {"semantic_signature": deepcopy(step["semantic_signature"])}
            for step in plan["steps"]
            if step.get("semantic_signature") is not None
        ],
    }

SEED = "905098775febd6fe3b64b9ab8ddb4262f569dcb156915e7c805cef54e14314ca"


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _json_digest(obj: dict) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(obj)).hexdigest()


def _random_bank_and_codebook(plan: dict) -> tuple[dict, dict]:
    signatures = {hash_json(step["semantic_signature"]): deepcopy(step["semantic_signature"]) for step in plan["steps"] if step.get("semantic_signature") is not None}
    entries = [{"signature_sha256": key, "semantic_signature": signatures[key]} for key in sorted(signatures)]
    bank = {"schema_version":"work-planner-signature-bank/1.0","run_id":"run-1","source_split":"train",
            "intent_labeler_contract_sha256":H1,"entries":entries,"bank_hash":H1}
    bank["bank_hash"] = hash_json({k:v for k,v in bank.items() if k!="bank_hash"})
    codes = [{"signature_sha256":row["signature_sha256"],"code_hex":code_hex(SEED,row["signature_sha256"])} for row in entries]
    from validation.random_codebook_validator import abs_cosine_bits, file_digest, CONTRACT
    observed = max((abs_cosine_bits(a["code_hex"], b["code_hex"]) for i,a in enumerate(codes) for b in codes[i+1:]), default=0.0)
    codebook={"schema_version":"work-planner-random-codebook/1.0","run_id":"run-1","contract_sha256":file_digest(CONTRACT),
              "source_signature_bank_path":"semantic_bank/signatures/manifest.json","source_signature_bank_sha256":_json_digest(bank),
              "algorithm":"SHAKE256_BITS_V1","seed_hex":SEED,"dimension":384,"entries":codes,
              "maximum_abs_pairwise_cosine":observed,"created_at":"2026-07-24T11:00:00Z","manifest_hash":H1}
    codebook["manifest_hash"] = hash_json({k:v for k,v in codebook.items() if k!="manifest_hash"})
    return bank, codebook


def _shuffle_mapping(manifest: dict, plan: dict) -> dict:
    from validation.random_codebook_validator import file_digest
    contract = Path("docs/controls/full_plan_shuffle_contract_v1.yaml")
    contract_hash=file_digest(contract)
    n=sum(step.get("typed_action",{}).get("action")!="END" for step in plan["steps"])
    payload={"run_id":manifest["run_id"],"task_id":manifest["task_id"],
             "source_episode_plan_manifest_hash":manifest["source_episode_plan_manifest_hash"],
             "source_plan_content_hash":manifest["work_plan_content_hash"],"contract_sha256":contract_hash}
    seed_hash=hash_json(payload)
    if n<2:
        status,offset,permutation="DEGENERATE",None,[]
    else:
        offset=1+int.from_bytes(hashlib.sha256(seed_hash.encode()).digest()[:8],"big")%(n-1)
        status,permutation="VALID",[(i+offset)%n for i in range(n)]
    mapping={"schema_version":"work-planner-shuffled-plan/1.0","run_id":manifest["run_id"],"task_id":manifest["task_id"],
             "source_episode_plan_manifest_hash":manifest["source_episode_plan_manifest_hash"],
             "source_plan_content_hash":manifest["work_plan_content_hash"],"contract_sha256":contract_hash,
             "plan_non_end_length":n,"status":status,"seed_payload_hash":seed_hash,"rotation_offset":offset,
             "permutation":permutation,"mapping_hash":H1}
    mapping["mapping_hash"]=hash_json({k:v for k,v in mapping.items() if k!="mapping_hash"})
    return mapping


def _cost(tokens_in=10, tokens_out=5, latency=3.0):
    return {"tokens_in": tokens_in, "tokens_out": tokens_out, "latency_ms": latency}


FLOPS_CAP = 1000.0
FLOPS_COEFF = {
    "fixed_flops_per_task": 10.0,
    "planner_flops_per_call": 20.0,
    "plan_token_flops": 1.0,
    "executor_input_token_flops": 1.0,
    "executor_output_token_flops": 2.0,
}


def _set_flops(episode: dict) -> None:
    actual = (FLOPS_COEFF["fixed_flops_per_task"]
              + episode["planner_calls"] * FLOPS_COEFF["planner_flops_per_call"]
              + (episode["plan_tokens_in_actual"] + episode["plan_tokens_out_actual"]) * FLOPS_COEFF["plan_token_flops"]
              + episode["executor_tokens_in"] * FLOPS_COEFF["executor_input_token_flops"]
              + episode["executor_tokens_out"] * FLOPS_COEFF["executor_output_token_flops"])
    attributed_calls = 0 if episode["arm"] == "E0_EQUAL_TOKENS_RAW" else 1
    attributed = (FLOPS_COEFF["fixed_flops_per_task"]
                  + attributed_calls * FLOPS_COEFF["planner_flops_per_call"]
                  + (episode["plan_tokens_in_attributed"] + episode["plan_tokens_out_attributed"]) * FLOPS_COEFF["plan_token_flops"]
                  + episode["executor_tokens_in"] * FLOPS_COEFF["executor_input_token_flops"]
                  + episode["executor_tokens_out"] * FLOPS_COEFF["executor_output_token_flops"])
    episode.update(flops_actual=actual, flops_attributed=attributed, flops_cap_per_task=FLOPS_CAP, flops_budget_exhausted=attributed > FLOPS_CAP)


def _write_compute_profile(root: Path) -> tuple[str, str]:
    from validation.capacity_validator import canonical_report_hash, file_digest
    evidence = root / "reports/compute-evidence/raw.jsonl"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    code = root / "validation/measurement_probe.py"
    code.parent.mkdir(parents=True, exist_ok=True); code.write_text("# probe\n")
    env = root / "locks/environment.lock.json"
    env.parent.mkdir(parents=True, exist_ok=True); env.write_text("{}\n")
    variants=[]
    for i,v in enumerate(("A1","A2","A2b","A2c","A3","A3r"),1):
        variants.append({"variant":v,"train_flops_per_optimizer_step":float(i*100),"train_seconds_per_optimizer_step":0.01*i,"peak_vram_bytes":1000*i,"checkpoint_bytes":100*i})
    per_step={row["variant"]:row["train_flops_per_optimizer_step"] for row in variants}
    source=min(("A2c","A3"),key=lambda variant:(per_step[variant],variant)); cap=per_step[source]*12000
    profile={
      "schema_version":"work-planner-compute/1.1","run_id":"run-1","hardware":{"device_name":"test","device_count":1},
      "variants":variants,"equal_data_schedule":{"optimizer_updates_exact":12000,"batch_size_examples":128},
      "flops_matched_schedule":{"comparison_id":"A3_vs_A2c","cap_source_variant":source,"cap_flops_per_workload":cap,"tolerance_fraction":0.02,
        "optimizer_updates_by_variant":{variant:int(cap//per_step[variant]) for variant in ("A2c","A3")}},
      "stage1b_inference":{"confirmatory_task_capacity":4000,"flops_cap_per_task":FLOPS_CAP,"budget_exhaustion_policy":"PAIRED_FAILURE_NO_TASK_EXCLUSION",
        "arms":[{"arm":arm,"estimated_flops_per_task":100.0,"estimated_seconds_per_task":0.01,
          "maximum_plan_calls":0 if arm in {"E0_EQUAL_TOKENS_RAW","E2_SHUFFLED_A3_FULL_PLAN_RAW","P_FULL_PLAN_REPLAY_RAW"} else 1,
          "maximum_executor_calls":16,**FLOPS_COEFF} for arm in sorted(STAGE1B_ARMS)]},
      "measurement_evidence":{"command":["python","validation/measurement_probe.py"],"repetitions":3,"raw_artifact_path":"reports/compute-evidence/raw.jsonl",
        "raw_artifact_sha256":H1,"environment_lock_sha256":file_digest(env)},
      "measurement_code_path":"validation/measurement_probe.py","measurement_code_sha256":file_digest(code),"report_hash":H1}
    rows=[]
    for row in variants:
        for trial in range(3):
            rows.append({"schema_version":"work-planner-compute-measurement/1.0","run_id":"run-1","trial_id":f"train-{row['variant']}-{trial}",
                "measurement_type":"TRAIN_OPTIMIZER_STEP","device_name":"test","variant":row["variant"],"arm":None,
                "measured_flops":row["train_flops_per_optimizer_step"],"elapsed_seconds":row["train_seconds_per_optimizer_step"],"peak_vram_bytes":row["peak_vram_bytes"]})
    for row in profile["stage1b_inference"]["arms"]:
        for trial in range(3):
            rows.append({"schema_version":"work-planner-compute-measurement/1.0","run_id":"run-1","trial_id":f"arm-{row['arm']}-{trial}",
                "measurement_type":"STAGE1B_TASK","device_name":"test","variant":None,"arm":row["arm"],
                "measured_flops":row["estimated_flops_per_task"],"elapsed_seconds":row["estimated_seconds_per_task"],"peak_vram_bytes":1})
    evidence.write_text("".join(json.dumps(row)+"\n" for row in rows))
    profile["measurement_evidence"]["raw_artifact_sha256"]=file_digest(evidence)
    profile["report_hash"]=canonical_report_hash(profile)
    path=root/"reports/compute-profile.json"; path.write_text(json.dumps(profile))
    return "reports/compute-profile.json", file_digest(path)


def _bundle(arm: str, *, source: dict | None = None, failed: bool = False):
    variant = {
        "E1_A3_FULL_PLAN_RAW": "A3",
        "E2_SHUFFLED_A3_FULL_PLAN_RAW": "A3",
        "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW": "A3r",
        "E4_A2C_STRUCTURED_FULL_PLAN_RAW": "A2c",
        "E5_SELF_PLAN_RAW": "SELF_PLAN",
        "P_FULL_PLAN_REPLAY_RAW": "A3",
    }[arm]
    plan = None if failed else (_plan(variant) if source is None else source[1])
    a = attempt(arm, "STAGE1B_END_TO_END")
    e = episode_from_attempt(a, goal_success=False, terminal_error="GOAL_NOT_REACHED")
    generator = {
        "E1_A3_FULL_PLAN_RAW": "MICRO_PLANNER_A3",
        "E2_SHUFFLED_A3_FULL_PLAN_RAW": "REUSED_E1_A3",
        "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW": "MICRO_PLANNER_A3R",
        "E4_A2C_STRUCTURED_FULL_PLAN_RAW": "MICRO_PLANNER_A2C",
        "E5_SELF_PLAN_RAW": "SELF_PLAN_LLM",
        "P_FULL_PLAN_REPLAY_RAW": "REUSED_E1_A3",
    }[arm]
    actual = _cost(0, 0, 0) if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"} else _cost(20 if arm == "E5_SELF_PLAN_RAW" else 10)
    attributed = deepcopy(source[0]["actual_cost"]) if source is not None else deepcopy(actual)
    replay = "STAGE1B_E1" if arm == "P_FULL_PLAN_REPLAY_RAW" else None
    manifest = {
        "schema_version": "work-planner/1.20",
        "run_id": a["run_id"], "episode_id": a["episode_id"], "trajectory_id": a["trajectory_id"],
        "stage": a["stage"], "task_id": a["task_id"], "base_task_id": a["base_task_id"],
        "canonical_task_hash": a["canonical_task_hash"], "split": a["split"], "arm": arm,
        "planner_seed": plan.get("planner_seed") if plan is not None else a.get("planner_seed"),
        "planner_checkpoint_sha256": plan.get("planner_checkpoint_sha256") if plan is not None else H1,
        "generator_type": generator, "plan_status": "FAILED" if failed else "READY",
        "planner_call_count": 0 if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"} else 1,
        "generated_before_execution": True,
        "generation_started_at": "2026-07-24T11:59:58Z", "generation_completed_at": "2026-07-24T11:59:59Z",
        "initial_state_hash": a["state_before_hash"], "goal_hash": a["goal_hash"],
        "work_plan_content_hash": None if failed else plan["plan_content_hash"],
        "work_plan_artifact_hash": None if failed else plan["plan_artifact_hash"],
        "work_plan_path": None if failed else f"results/stage1b-confirmatory/plans/{arm}.json",
        "source_episode_plan_manifest_hash": source[0]["manifest_hash"] if source is not None else None,
        "source_arm": source[0]["arm"] if source is not None else None,
        "semantic_signature_bank_hash": hash_json(_signature_bank(plan)) if arm == "E4_A2C_STRUCTURED_FULL_PLAN_RAW" and plan is not None else None,
        "generation_failure_code": "SEMANTIC_UNRESOLVED" if failed else None,
        "control_status": "NOT_APPLICABLE",
        "control_artifact_path": None, "control_artifact_hash": None,
        "actual_cost": actual, "attributed_cost": attributed, "replay_context": replay, "manifest_hash": H1,
    }
    control = None
    if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW":
        control = _shuffle_mapping(manifest, plan)
        manifest["control_status"] = control["status"]
        manifest["control_artifact_path"] = f"controls/stage1b/shuffled-plan/{manifest['episode_id']}.json"
        manifest["control_artifact_hash"] = control["mapping_hash"]
    elif arm == "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW":
        _, control = _random_bank_and_codebook(plan)
        manifest["control_artifact_path"] = "semantic_bank/random-codebook/manifest.json"
        manifest["control_artifact_hash"] = control["manifest_hash"]
    manifest["manifest_hash"] = hash_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW" and control is not None and control["status"] == "DEGENERATE":
        e.update(goal_success=False, terminal_error="CONTROL_UNAVAILABLE", attempts_total=0, steps_accepted=0,
                 executed_length=0, planner_calls=0, final_state_hash=a["state_before_hash"],
                 episode_plan_manifest_hash=manifest["manifest_hash"], plan_generation_status="READY",
                 plan_positions_consumed=0, executor_tokens_in=0, executor_tokens_out=0, executor_latency_ms=0,
                 plan_tokens_in_actual=0, plan_tokens_out_actual=0, plan_latency_ms_actual=0,
                 plan_tokens_in_attributed=attributed["tokens_in"], plan_tokens_out_attributed=attributed["tokens_out"],
                 plan_latency_ms_attributed=attributed["latency_ms"])
        _set_flops(e)
        return manifest, plan, e, [], control
    if failed:
        e.update(goal_success=False, terminal_error="SEMANTIC_UNRESOLVED", attempts_total=0, steps_accepted=0,
                 executed_length=0, planner_calls=manifest["planner_call_count"], final_state_hash=a["state_before_hash"],
                 episode_plan_manifest_hash=manifest["manifest_hash"], plan_generation_status="FAILED",
                 plan_positions_consumed=0, executor_tokens_in=0, executor_tokens_out=0, executor_latency_ms=0,
                 plan_tokens_in_actual=actual["tokens_in"], plan_tokens_out_actual=actual["tokens_out"],
                 plan_latency_ms_actual=actual["latency_ms"], plan_tokens_in_attributed=attributed["tokens_in"],
                 plan_tokens_out_attributed=attributed["tokens_out"], plan_latency_ms_attributed=attributed["latency_ms"])
        _set_flops(e)
        return manifest, None, e, [], control
    a.update(
        episode_plan_manifest_hash=manifest["manifest_hash"], plan_generation_status="READY",
        plan_content_hash=plan["plan_content_hash"], plan_artifact_hash=plan["plan_artifact_hash"],
        plan_step_id=plan["steps"][0]["step_id"], plan_position_index=0, replanning_observed=False,
        replay_context=replay,
    )
    if arm == "P_FULL_PLAN_REPLAY_RAW":
        a["candidate_typed_action"] = deepcopy(plan["steps"][0]["typed_action"])
    if arm == "E2_SHUFFLED_A3_FULL_PLAN_RAW":
        source_position = control["permutation"][0]
        source_step = [row for row in plan["steps"] if row["typed_action"]["action"] != "END"][source_position]
        a["guidance_source_position_index"] = source_position
        a["guidance_source_step_id"] = source_step["step_id"]
        a["guidance_source_semantic_ref"] = source_step["semantic_ref"]
        a["control_mapping_hash"] = control["mapping_hash"]
    elif arm != "P_FULL_PLAN_REPLAY_RAW":
        a["guidance_source_step_id"] = plan["steps"][0]["step_id"]
        a["guidance_source_semantic_ref"] = plan["steps"][0]["semantic_ref"]
    e.update(
        planner_calls=manifest["planner_call_count"], episode_plan_manifest_hash=manifest["manifest_hash"],
        plan_generation_status="READY", replay_context=replay, plan_positions_consumed=1,
        plan_tokens_in_actual=actual["tokens_in"], plan_tokens_out_actual=actual["tokens_out"],
        plan_latency_ms_actual=actual["latency_ms"], plan_tokens_in_attributed=attributed["tokens_in"],
        plan_tokens_out_attributed=attributed["tokens_out"], plan_latency_ms_attributed=attributed["latency_ms"],
        executor_tokens_in=a["tokens_in"] or 0, executor_tokens_out=a["tokens_out"] or 0,
        executor_latency_ms=a["total_ms"] or 0,
    )
    _set_flops(e)
    return manifest, plan, e, [a], control


def test_all_full_plan_arms_validate_and_reused_cost_is_attributed():
    e1 = _bundle("E1_A3_FULL_PLAN_RAW")
    for arm in sorted(STAGE1B_ARMS - {"E0_EQUAL_TOKENS_RAW"}):
        bundle = _bundle(arm, source=e1 if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"} else None)
        bank = _signature_bank(bundle[1]) if arm == "E4_A2C_STRUCTURED_FULL_PLAN_RAW" else None
        assert not validate_episode_plan_manifest(bundle[0], work_plan=bundle[1], episode=bundle[2], attempts=bundle[3], source_manifest=e1[0] if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"} else None, signature_bank=bank, control_artifact=bundle[4]), arm


def test_one_call_before_execution_and_no_replanning_are_enforced():
    m, p, e, attempts, control = _bundle("E1_A3_FULL_PLAN_RAW")
    bad = deepcopy(m); bad["planner_call_count"] = 2; bad["manifest_hash"] = hash_json({k:v for k,v in bad.items() if k != "manifest_hash"})
    assert any("exactly one" in x for x in validate_episode_plan_manifest(bad, work_plan=p, episode=e, attempts=attempts, control_artifact=control))
    bad_attempts = deepcopy(attempts); bad_attempts[0]["replanning_observed"] = True
    assert any("replanning" in x for x in validate_episode_plan_manifest(m, work_plan=p, episode=e, attempts=bad_attempts, control_artifact=control))
    bad_attempts = deepcopy(attempts); bad_attempts[0]["timestamp"] = "2026-07-24T11:59:00Z"
    assert any("predates" in x for x in validate_episode_plan_manifest(m, work_plan=p, episode=e, attempts=bad_attempts, control_artifact=control))


def test_frozen_positions_and_workplan_hash_cannot_be_mutated():
    m, p, e, attempts, control = _bundle("E1_A3_FULL_PLAN_RAW")
    bad = deepcopy(attempts); bad[0]["plan_position_index"] = 1
    assert any("contiguous" in x for x in validate_episode_plan_manifest(m, work_plan=p, episode=e, attempts=bad, control_artifact=control))
    bad_plan = deepcopy(p); bad_plan["steps"][0]["step_id"] = "S99"
    assert any("hash mismatch" in x for x in validate_episode_plan_manifest(m, work_plan=bad_plan, episode=e, attempts=attempts, control_artifact=control))


def test_failed_plan_is_paired_failure_not_hidden_exclusion():
    m, p, e, attempts, control = _bundle("E4_A2C_STRUCTURED_FULL_PLAN_RAW", failed=True)
    assert not validate_episode_plan_manifest(m, work_plan=p, episode=e, attempts=attempts, control_artifact=control)
    bad_e = deepcopy(e); bad_e["goal_success"] = None
    assert validate_episode_plan_manifest(m, work_plan=p, episode=bad_e, attempts=attempts, control_artifact=control)


def test_a2c_unknown_signature_must_fail_not_heuristic_map():
    m, p, e, attempts, control = _bundle("E4_A2C_STRUCTURED_FULL_PLAN_RAW")
    bank = {"signatures": [{"semantic_signature": {"intent_id": 999}}]}
    assert any("unknown A2c" in x for x in validate_episode_plan_manifest(m, work_plan=p, episode=e, attempts=attempts, signature_bank=bank, control_artifact=control))


def test_replay_context_prevents_phase_order_substitution():
    e1 = _bundle("E1_A3_FULL_PLAN_RAW")
    m, p, e, attempts, control = _bundle("P_FULL_PLAN_REPLAY_RAW", source=e1)
    bad = deepcopy(m); bad["replay_context"] = "PLANNER_CONFIRMATORY_A3"; bad["manifest_hash"] = hash_json({k:v for k,v in bad.items() if k != "manifest_hash"})
    assert any("PLANNER_A3_RAW" in x for x in validate_episode_plan_manifest(bad, work_plan=p, episode=e, attempts=attempts, source_manifest=e1[0], control_artifact=control))


def _selection_fields(root: Path, stage: str, task_ids: list[str]) -> dict:
    slug = stage.lower()
    selected_rel = f"sealed/{slug}/selected-task-manifest.json"
    selected = {
        "schema_version": "work-planner-selected-tasks/1.0",
        "run_id": "run-1",
        "stage": stage,
        "task_ids": sorted(task_ids),
        "task_count": len(set(task_ids)),
        "created_at": "2026-07-24T10:00:00Z",
        "manifest_hash": H1,
    }
    selected["manifest_hash"] = hash_json({k: v for k, v in selected.items() if k != "manifest_hash"})
    selected_path = root / selected_rel
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps(selected))
    selected_sha = _json_digest(selected)
    sealer_rel = f"sealed/{slug}/sealer-manifest.json"
    sealer = {"run_id": "run-1", "stage": stage, "task_count": len(set(task_ids)),
              "selected_task_manifest_path": selected_rel, "selected_task_manifest_sha256": selected_sha}
    if stage == "STAGE1B":
        sealer["control_certification"] = {"task_only_selection_manifest_sha256": selected_sha}
    sealer_path = root / sealer_rel
    sealer_path.write_text(json.dumps(sealer))
    return {
        "selected_task_manifest": selected_rel,
        "selected_task_manifest_sha256": selected_sha,
        "sealer_manifest": sealer_rel,
        "sealer_manifest_sha256": _json_digest(sealer),
        "expected_task_count": len(set(task_ids)),
    }


def _write_bundle(root: Path, arm: str, bundle, source_hash=None, *, identity_suffix: str = ""):
    m, p, e, attempts, control = bundle
    safe = arm.lower() + identity_suffix
    paths = {
        "episode_plan_manifest": None if m is None else f"results/stage1b-confirmatory/episode-plan-manifests/{safe}.json",
        "work_plan": None if p is None else f"results/stage1b-confirmatory/plans/{safe}.json",
        "episode_log": f"results/stage1b-confirmatory/episodes/{safe}.json",
        "attempt_log": f"results/stage1b-confirmatory/attempts/{safe}.jsonl",
        "semantic_signature_bank": f"semantic_bank/{safe}-signatures.json" if arm == "E4_A2C_STRUCTURED_FULL_PLAN_RAW" and p is not None else None,
        "control_artifact": None if control is None else m["control_artifact_path"],
    }
    for key,obj in (("episode_plan_manifest",m),("work_plan",p),("episode_log",e)):
        if obj is not None:
            path=root/paths[key]; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj))
    ap=root/paths["attempt_log"]; ap.parent.mkdir(parents=True,exist_ok=True); ap.write_text("".join(json.dumps(x)+"\n" for x in attempts))
    if paths["semantic_signature_bank"] is not None:
        bp=root/paths["semantic_signature_bank"]; bp.parent.mkdir(parents=True,exist_ok=True); bp.write_text(json.dumps(_signature_bank(p)))
    if control is not None:
        cp=root/paths["control_artifact"]; cp.parent.mkdir(parents=True,exist_ok=True)
        if arm == "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW":
            bank, control = _random_bank_and_codebook(p)
            bp=root/"semantic_bank/signatures/manifest.json"; bp.parent.mkdir(parents=True,exist_ok=True); bp.write_bytes(_json_bytes(bank))
            # Source hash is based on these exact bytes; rebuild manifest after source write for safety.
            control["source_signature_bank_sha256"]=_json_digest(bank)
            control["manifest_hash"]=hash_json({k:v for k,v in control.items() if k!="manifest_hash"})
            m["control_artifact_hash"]=control["manifest_hash"]
            m["manifest_hash"]=hash_json({k:v for k,v in m.items() if k!="manifest_hash"})
            e["episode_plan_manifest_hash"]=m["manifest_hash"]
            for attempt_row in attempts: attempt_row["episode_plan_manifest_hash"]=m["manifest_hash"]
            (root/paths["episode_plan_manifest"]).write_text(json.dumps(m))
            ap.write_text("".join(json.dumps(x)+"\n" for x in attempts))
            (root/paths["episode_log"]).write_text(json.dumps(e))
        cp.write_text(json.dumps(control))
    planner_seed = p.get("planner_seed") if p is not None else (attempts[0].get("planner_seed") if attempts else m.get("planner_seed") if m else None)
    checkpoint_manifest = None
    checkpoint_manifest_sha256 = None
    if e.get("stage") == "PLANNER_ONLY":
        expected = {
            "PLANNER_A1_RAW": ("A1", "FINAL_EQUAL_DATA"),
            "PLANNER_A2_RAW": ("A2", "FINAL_EQUAL_DATA"),
            "PLANNER_A2B_RAW": ("A2b", "FINAL_EQUAL_DATA"),
            "PLANNER_A2C_RAW": ("A2c", "FINAL_EQUAL_DATA"),
            "PLANNER_A3_RAW": ("A3", "FINAL_EQUAL_DATA"),
            "PLANNER_A3R_RAW": ("A3r", "FINAL_EQUAL_DATA"),
            "PLANNER_A4_RAW": ("A3", "FINAL_EQUAL_DATA"),
            "PLANNER_A5_RAW": ("A3", "FINAL_EQUAL_DATA"),
            "PLANNER_A2C_FLOPS_RAW": ("A2c", "FLOPS_SENSITIVITY"),
            "PLANNER_A3_FLOPS_RAW": ("A3", "FLOPS_SENSITIVITY"),
            "P_FULL_PLAN_REPLAY_RAW": ("A3", "FINAL_EQUAL_DATA"),
        }[arm]
        prefix = "final" if expected[1] == "FINAL_EQUAL_DATA" else "flops"
        checkpoint_manifest = f"reports/training/checkpoints/{prefix}-{expected[0]}-seed-{planner_seed}.json"
        cp_path = root / checkpoint_manifest
        checkpoint_manifest_sha256 = "sha256:" + hashlib.sha256(cp_path.read_bytes()).hexdigest()
    return {
        "task_id": e["task_id"], "arm": arm, "planner_seed": planner_seed,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        **paths,
    }


def test_lineage_index_requires_all_seven_arms_even_degenerate(tmp_path: Path):
    e1 = _bundle("E1_A3_FULL_PLAN_RAW")
    rows=[]
    e0a=attempt("E0_EQUAL_TOKENS_RAW","STAGE1B_END_TO_END"); e0e=episode_from_attempt(e0a,goal_success=False,terminal_error="GOAL_NOT_REACHED")
    e0e.update(episode_plan_manifest_hash=None,plan_generation_status="NONE",replay_context=None,plan_positions_consumed=0,
               plan_tokens_in_actual=0,plan_tokens_out_actual=0,plan_latency_ms_actual=0,plan_tokens_in_attributed=0,
               plan_tokens_out_attributed=0,plan_latency_ms_attributed=0,executor_tokens_in=e0a["tokens_in"],
               executor_tokens_out=e0a["tokens_out"],executor_latency_ms=e0a["total_ms"],planner_calls=0)
    _set_flops(e0e)
    rows.append(_write_bundle(tmp_path,"E0_EQUAL_TOKENS_RAW",(None,None,e0e,[e0a],None)))
    for arm in sorted(STAGE1B_ARMS-{"E0_EQUAL_TOKENS_RAW"}):
        rows.append(_write_bundle(tmp_path,arm,_bundle(arm,source=e1 if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW","P_FULL_PLAN_REPLAY_RAW"} else None)))
    compute_path, compute_hash = _write_compute_profile(tmp_path)
    index={"schema_version":"work-planner-lineage/1.0","run_id":"run-1","stage":"STAGE1B","compute_profile":compute_path,"compute_profile_sha256":compute_hash,"records":rows,"index_hash":H1, **_selection_fields(tmp_path, "STAGE1B", ["bw-00000001"])}
    index["index_hash"]=hash_json({k:v for k,v in index.items() if k!="index_hash"})
    assert not validate_lineage_index(tmp_path,index,expected_stage="STAGE1B")
    bad=deepcopy(index); bad["records"]=[r for r in bad["records"] if r["arm"]!="E2_SHUFFLED_A3_FULL_PLAN_RAW"]; bad["index_hash"]=hash_json({k:v for k,v in bad.items() if k!="index_hash"})
    assert any("post-treatment exclusion" in x for x in validate_lineage_index(tmp_path,bad,expected_stage="STAGE1B"))


def test_runtime_generation_profiles_are_exact_and_remote_code_is_forbidden():
    schema=json.load(open('docs/schemas/runtime_stack_lock.schema.json'))
    mr=schema['properties']['model_runtime']
    assert mr['properties']['trust_remote_code']=={'const': False}
    profiles=mr['properties']['generation_profiles']['properties']
    assert profiles['stage1a_executor']['properties']['max_new_tokens']['const']==64
    assert profiles['stage1b_executor']['properties']['max_new_tokens']['const']==128
    assert profiles['self_plan_generator']['properties']['max_new_tokens']['const']==128


def test_lineage_recomputes_flops_and_enforces_cap(tmp_path: Path):
    e1 = _bundle("E1_A3_FULL_PLAN_RAW")
    rows=[]
    e0a=attempt("E0_EQUAL_TOKENS_RAW","STAGE1B_END_TO_END"); e0e=episode_from_attempt(e0a,goal_success=False,terminal_error="GOAL_NOT_REACHED")
    e0e.update(episode_plan_manifest_hash=None,plan_generation_status="NONE",replay_context=None,plan_positions_consumed=0,
               plan_tokens_in_actual=0,plan_tokens_out_actual=0,plan_latency_ms_actual=0,plan_tokens_in_attributed=0,
               plan_tokens_out_attributed=0,plan_latency_ms_attributed=0,executor_tokens_in=e0a["tokens_in"],
               executor_tokens_out=e0a["tokens_out"],executor_latency_ms=e0a["total_ms"],planner_calls=0)
    _set_flops(e0e); rows.append(_write_bundle(tmp_path,"E0_EQUAL_TOKENS_RAW",(None,None,e0e,[e0a],None)))
    for arm in sorted(STAGE1B_ARMS-{"E0_EQUAL_TOKENS_RAW"}):
        rows.append(_write_bundle(tmp_path,arm,_bundle(arm,source=e1 if arm in {"E2_SHUFFLED_A3_FULL_PLAN_RAW","P_FULL_PLAN_REPLAY_RAW"} else None)))
    compute_path, compute_hash = _write_compute_profile(tmp_path)
    index={"schema_version":"work-planner-lineage/1.0","run_id":"run-1","stage":"STAGE1B","compute_profile":compute_path,"compute_profile_sha256":compute_hash,"records":rows,"index_hash":H1, **_selection_fields(tmp_path, "STAGE1B", ["bw-00000001"])}
    index["index_hash"]=hash_json({k:v for k,v in index.items() if k!="index_hash"})
    assert not validate_lineage_index(tmp_path,index,expected_stage="STAGE1B")
    target=tmp_path/next(r["episode_log"] for r in rows if r["arm"]=="E5_SELF_PLAN_RAW")
    obj=json.loads(target.read_text()); obj["flops_attributed"] += 1; target.write_text(json.dumps(obj))
    assert any("flops_attributed" in x for x in validate_lineage_index(tmp_path,index,expected_stage="STAGE1B"))


def test_shuffled_mapping_and_attempt_position_are_immutable():
    e1 = _bundle("E1_A3_FULL_PLAN_RAW")
    m, p, e, attempts, control = _bundle("E2_SHUFFLED_A3_FULL_PLAN_RAW", source=e1)

    bad_control = deepcopy(control)
    bad_control["permutation"] = list(range(bad_control["plan_non_end_length"]))
    bad_control["mapping_hash"] = hash_json({k: v for k, v in bad_control.items() if k != "mapping_hash"})
    bad_manifest = deepcopy(m)
    bad_manifest["control_artifact_hash"] = bad_control["mapping_hash"]
    bad_manifest["manifest_hash"] = hash_json({k: v for k, v in bad_manifest.items() if k != "manifest_hash"})
    bad_episode = deepcopy(e)
    bad_episode["episode_plan_manifest_hash"] = bad_manifest["manifest_hash"]
    bad_attempts = deepcopy(attempts)
    bad_attempts[0]["episode_plan_manifest_hash"] = bad_manifest["manifest_hash"]
    assert any("locked nonzero cyclic rotation" in x for x in validate_episode_plan_manifest(
        bad_manifest, work_plan=p, episode=bad_episode, attempts=bad_attempts,
        source_manifest=e1[0], control_artifact=bad_control,
    ))

    bad_attempts = deepcopy(attempts)
    bad_attempts[0]["guidance_source_position_index"] = 0
    assert any("shuffle permutation" in x for x in validate_episode_plan_manifest(
        m, work_plan=p, episode=e, attempts=bad_attempts,
        source_manifest=e1[0], control_artifact=control,
    ))


def test_degenerate_shuffle_stays_zero_execution_paired_failure():
    source = list(_bundle("E1_A3_FULL_PLAN_RAW"))
    plan = deepcopy(source[1])
    plan["steps"] = [deepcopy(plan["steps"][0]), deepcopy(plan["steps"][-1])]
    plan["steps"][1]["step_index"] = 1
    plan["steps"][1]["step_id"] = "S01"
    plan["plan_content_hash"] = plan_content_hash(plan)
    plan["plan_artifact_hash"] = plan_artifact_hash(plan)
    source_manifest = deepcopy(source[0])
    source_manifest["work_plan_content_hash"] = plan["plan_content_hash"]
    source_manifest["work_plan_artifact_hash"] = plan["plan_artifact_hash"]
    source_manifest["manifest_hash"] = hash_json({k: v for k, v in source_manifest.items() if k != "manifest_hash"})
    source[0], source[1] = source_manifest, plan
    source = tuple(source)

    m, p, e, attempts, control = _bundle("E2_SHUFFLED_A3_FULL_PLAN_RAW", source=source)
    assert control["status"] == "DEGENERATE"
    e = deepcopy(e)
    e.update(
        attempts_total=0, steps_accepted=0, executed_length=0,
        goal_success=False, terminal_error="CONTROL_UNAVAILABLE",
        plan_positions_consumed=0, executor_tokens_in=0,
        executor_tokens_out=0, executor_latency_ms=0,
    )
    _set_flops(e)
    assert not validate_episode_plan_manifest(
        m, work_plan=p, episode=e, attempts=[],
        source_manifest=source_manifest, control_artifact=control,
    )


def test_a3r_requires_frozen_codebook_and_known_signatures():
    m, p, e, attempts, control = _bundle("E3_A3R_RANDOM_CODE_FULL_PLAN_RAW")
    assert any("control_artifact" in x for x in validate_episode_plan_manifest(
        m, work_plan=p, episode=e, attempts=attempts, control_artifact=None,
    ))

    bad_plan = deepcopy(p)
    bad_plan["steps"][0]["semantic_signature"]["intent_id"] = 999
    bad_plan["plan_content_hash"] = plan_content_hash(bad_plan)
    bad_plan["plan_artifact_hash"] = plan_artifact_hash(bad_plan)
    bad_manifest = deepcopy(m)
    bad_manifest["work_plan_content_hash"] = bad_plan["plan_content_hash"]
    bad_manifest["work_plan_artifact_hash"] = bad_plan["plan_artifact_hash"]
    bad_manifest["manifest_hash"] = hash_json({k: v for k, v in bad_manifest.items() if k != "manifest_hash"})
    bad_episode = deepcopy(e)
    bad_episode["episode_plan_manifest_hash"] = bad_manifest["manifest_hash"]
    bad_attempts = deepcopy(attempts)
    bad_attempts[0]["episode_plan_manifest_hash"] = bad_manifest["manifest_hash"]
    bad_attempts[0]["plan_content_hash"] = bad_plan["plan_content_hash"]
    bad_attempts[0]["plan_artifact_hash"] = bad_plan["plan_artifact_hash"]
    assert any("missing from frozen random codebook" in x for x in validate_episode_plan_manifest(
        bad_manifest, work_plan=bad_plan, episode=bad_episode,
        attempts=bad_attempts, control_artifact=control,
    ))


def _planner_confirmatory_bundle(arm: str, seed: int, *, source=None):
    variant_by_arm = {
        "PLANNER_A1_RAW": "A1",
        "PLANNER_A2_RAW": "A2",
        "PLANNER_A2B_RAW": "A2b",
        "PLANNER_A2C_RAW": "A2c",
        "PLANNER_A3_RAW": "A3",
        "PLANNER_A3R_RAW": "A3r",
        "PLANNER_A4_RAW": "A4",
        "PLANNER_A5_RAW": "A5",
        "PLANNER_A2C_FLOPS_RAW": "A2c",
        "PLANNER_A3_FLOPS_RAW": "A3",
    }
    generator_by_arm = {
        "PLANNER_A1_RAW": "MICRO_PLANNER_A1",
        "PLANNER_A2_RAW": "MICRO_PLANNER_A2",
        "PLANNER_A2B_RAW": "MICRO_PLANNER_A2B",
        "PLANNER_A2C_RAW": "MICRO_PLANNER_A2C",
        "PLANNER_A3_RAW": "MICRO_PLANNER_A3",
        "PLANNER_A3R_RAW": "MICRO_PLANNER_A3R",
        "PLANNER_A4_RAW": "MICRO_PLANNER_A4",
        "PLANNER_A5_RAW": "MICRO_PLANNER_A5",
        "PLANNER_A2C_FLOPS_RAW": "MICRO_PLANNER_A2C",
        "PLANNER_A3_FLOPS_RAW": "MICRO_PLANNER_A3",
    }
    if arm == "P_FULL_PLAN_REPLAY_RAW":
        if source is None:
            raise ValueError("planner replay requires same-seed PLANNER_A3_RAW source")
        rm, rp, re, rattempts, rcontrol = deepcopy(_bundle("P_FULL_PLAN_REPLAY_RAW", source=source))
        unique = f"{seed}-{arm.lower()}"
        rm.update(
            run_id="run-1", episode_id=f"episode-{unique}", trajectory_id=f"trajectory-{unique}",
            stage="PLANNER_ONLY", split="planner_confirmatory_horizon", planner_seed=seed, replay_context="PLANNER_CONFIRMATORY_A3",
            source_arm="PLANNER_A3_RAW", work_plan_path=f"results/planner-confirmatory/plans/{unique}.json",
        )
        rp = deepcopy(source[1])
        re.update(
            run_id="run-1", episode_id=f"episode-{unique}", trajectory_id=f"trajectory-{unique}",
            stage="PLANNER_ONLY", split="planner_confirmatory_horizon", arm=arm, planner_seed=seed, replay_context="PLANNER_CONFIRMATORY_A3",
        )
        for row in rattempts:
            row.update(
                run_id="run-1", episode_id=f"episode-{unique}", trajectory_id=f"trajectory-{unique}",
                stage="PLANNER_ONLY", split="planner_confirmatory_horizon", arm=arm, planner_seed=seed,
                replay_context="PLANNER_CONFIRMATORY_A3",
            )
        rm["manifest_hash"] = hash_json({k: v for k, v in rm.items() if k != "manifest_hash"})
        re["episode_plan_manifest_hash"] = rm["manifest_hash"]
        for row in rattempts:
            row["episode_plan_manifest_hash"] = rm["manifest_hash"]
        return rm, rp, re, rattempts, rcontrol

    base = _bundle("E1_A3_FULL_PLAN_RAW")
    m, _, e, attempts, control = deepcopy(base)
    variant = variant_by_arm[arm]
    p = _plan(variant)
    unique = f"{seed}-{arm.lower()}"
    p.update(
        plan_id=f"plan-{unique}", planner_seed=seed, run_id="run-1", stage="PLANNER_ONLY",
    )
    p["plan_content_hash"] = plan_content_hash(p)
    p["plan_artifact_hash"] = plan_artifact_hash(p)
    m.update(
        run_id="run-1", episode_id=f"episode-{unique}", trajectory_id=f"trajectory-{unique}",
        arm=arm, stage="PLANNER_ONLY", split="planner_confirmatory_horizon", planner_seed=seed, generator_type=generator_by_arm[arm],
        work_plan_content_hash=p["plan_content_hash"], work_plan_artifact_hash=p["plan_artifact_hash"],
        work_plan_path=f"results/planner-confirmatory/plans/{unique}.json", source_episode_plan_manifest_hash=None,
        source_arm=None, replay_context=None, control_status="NOT_APPLICABLE", control_artifact_path=None,
        control_artifact_hash=None, semantic_signature_bank_hash=None,
    )
    m["manifest_hash"] = hash_json({k: v for k, v in m.items() if k != "manifest_hash"})
    e.update(
        run_id="run-1", episode_id=f"episode-{unique}", trajectory_id=f"trajectory-{unique}",
        arm=arm, stage="PLANNER_ONLY", split="planner_confirmatory_horizon", planner_seed=seed,
        episode_plan_manifest_hash=m["manifest_hash"], replay_context=None,
    )
    for row in attempts:
        row.update(
            run_id="run-1", episode_id=f"episode-{unique}", trajectory_id=f"trajectory-{unique}",
            arm=arm, stage="PLANNER_ONLY", split="planner_confirmatory_horizon", planner_seed=seed,
            episode_plan_manifest_hash=m["manifest_hash"], plan_content_hash=p["plan_content_hash"],
            plan_artifact_hash=p["plan_artifact_hash"], plan_step_id=p["steps"][0]["step_id"],
            guidance_source_position_index=0, guidance_source_step_id=p["steps"][0]["step_id"],
            guidance_source_semantic_ref=p["steps"][0]["semantic_ref"], replay_context=None,
        )
    return m, p, e, attempts, control


def _bind_planner_bundle_checkpoint(bundle, model_sha256: str):
    m, p, e, attempts, control = bundle
    if p is not None:
        p["planner_checkpoint_sha256"] = model_sha256
        p["plan_content_hash"] = plan_content_hash(p)
        p["plan_artifact_hash"] = plan_artifact_hash(p)
    if m is not None:
        m["planner_checkpoint_sha256"] = model_sha256
        if p is not None:
            m["work_plan_content_hash"] = p["plan_content_hash"]
            m["work_plan_artifact_hash"] = p["plan_artifact_hash"]
        m["manifest_hash"] = hash_json({k: v for k, v in m.items() if k != "manifest_hash"})
    if m is not None:
        e["episode_plan_manifest_hash"] = m["manifest_hash"]
    for row in attempts:
        if m is not None:
            row["episode_plan_manifest_hash"] = m["manifest_hash"]
        if p is not None:
            row["plan_content_hash"] = p["plan_content_hash"]
            row["plan_artifact_hash"] = p["plan_artifact_hash"]
    return m, p, e, attempts, control


def _canonical_checkpoint_model_sha(root: Path, arm: str, seed: int) -> str:
    expected = {
        "PLANNER_A1_RAW": ("A1", "final"),
        "PLANNER_A2_RAW": ("A2", "final"),
        "PLANNER_A2B_RAW": ("A2b", "final"),
        "PLANNER_A2C_RAW": ("A2c", "final"),
        "PLANNER_A3_RAW": ("A3", "final"),
        "PLANNER_A3R_RAW": ("A3r", "final"),
        "PLANNER_A4_RAW": ("A3", "final"),
        "PLANNER_A5_RAW": ("A3", "final"),
        "PLANNER_A2C_FLOPS_RAW": ("A2c", "flops"),
        "PLANNER_A3_FLOPS_RAW": ("A3", "flops"),
        "P_FULL_PLAN_REPLAY_RAW": ("A3", "final"),
    }[arm]
    cp = json.loads((root / f"reports/training/checkpoints/{expected[1]}-{expected[0]}-seed-{seed}.json").read_text())
    return cp["model_file_sha256"]


def _planner_confirmatory_bundle_pair():
    source = _planner_confirmatory_bundle("PLANNER_A3_RAW", 202)
    replay = _planner_confirmatory_bundle("P_FULL_PLAN_REPLAY_RAW", 202, source=source)
    return source, replay


def _planner_confirmatory_matrix_rows(root: Path):
    from validation.full_plan_lineage_validator import PLANNER_ARMS, PLANNER_SEEDS
    _write_final_matrix(root)
    _write_flops_sensitivity_matrix(root)
    rows = []
    for seed in PLANNER_SEEDS:
        source = _bind_planner_bundle_checkpoint(
            _planner_confirmatory_bundle("PLANNER_A3_RAW", seed),
            _canonical_checkpoint_model_sha(root, "PLANNER_A3_RAW", seed),
        )
        bundles = {"PLANNER_A3_RAW": source}
        for arm in sorted(PLANNER_ARMS - {"PLANNER_A3_RAW", "P_FULL_PLAN_REPLAY_RAW"}):
            bundles[arm] = _bind_planner_bundle_checkpoint(
                _planner_confirmatory_bundle(arm, seed),
                _canonical_checkpoint_model_sha(root, arm, seed),
            )
        replay = _planner_confirmatory_bundle("P_FULL_PLAN_REPLAY_RAW", seed, source=source)
        bundles["P_FULL_PLAN_REPLAY_RAW"] = _bind_planner_bundle_checkpoint(
            replay, _canonical_checkpoint_model_sha(root, "P_FULL_PLAN_REPLAY_RAW", seed)
        )
        for arm in sorted(PLANNER_ARMS):
            rows.append(_write_bundle(root, arm, bundles[arm], identity_suffix=f"-{seed}"))
    return rows


def test_planner_confirmatory_p08_uses_precomputed_a3_plan(tmp_path: Path):
    rows = _planner_confirmatory_matrix_rows(tmp_path)
    index = {
        "schema_version": "work-planner-lineage/1.0", "run_id": "run-1", "stage": "PLANNER",
        "compute_profile": None, "compute_profile_sha256": None,
        "records": rows, "index_hash": H1,
        **_selection_fields(tmp_path, "PLANNER", ["bw-00000001"]),
    }
    index["index_hash"] = hash_json({k: v for k, v in index.items() if k != "index_hash"})
    assert not validate_lineage_index(tmp_path, index, expected_stage="PLANNER")

    bad = deepcopy(index)
    bad["records"] = [row for row in bad["records"] if row["arm"] != "PLANNER_A3_RAW"]
    bad["index_hash"] = hash_json({k: v for k, v in bad.items() if k != "index_hash"})
    assert any("expected exactly" in x for x in validate_lineage_index(tmp_path, bad, expected_stage="PLANNER"))

    wrong = deepcopy(index)
    target = next(row for row in wrong["records"] if row["arm"] == "PLANNER_A2C_FLOPS_RAW" and row["planner_seed"] == 101)
    checkpoint_path = tmp_path / target["checkpoint_manifest"]
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["checkpoint_kind"] = "FINAL_EQUAL_DATA"
    checkpoint["manifest_hash"] = hash_json({k: v for k, v in checkpoint.items() if k != "manifest_hash"})
    checkpoint_path.write_text(json.dumps(checkpoint))
    target["checkpoint_manifest_sha256"] = "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    wrong["index_hash"] = hash_json({k: v for k, v in wrong.items() if k != "index_hash"})
    assert any("expected A2c/FLOPS_SENSITIVITY" in x for x in validate_lineage_index(tmp_path, wrong, expected_stage="PLANNER"))


def test_episode_call_count_and_single_attempt_per_position_are_enforced():
    m, p, e, attempts, control = _bundle("E1_A3_FULL_PLAN_RAW")
    bad_episode = deepcopy(e)
    bad_episode["planner_calls"] = 0
    assert any("planner_calls" in x for x in validate_episode_plan_manifest(
        m, work_plan=p, episode=bad_episode, attempts=attempts, control_artifact=control,
    ))
    bad_attempts = deepcopy(attempts)
    bad_attempts[0]["attempt_index"] = 1
    assert any("one attempt" in x for x in validate_episode_plan_manifest(
        m, work_plan=p, episode=e, attempts=bad_attempts, control_artifact=control,
    ))
