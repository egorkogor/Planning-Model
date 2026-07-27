from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from validation.cross_object_validator import CrossObjectContractValidator
from validation.fixtures import (
    H1,
    attempt,
    episode_from_attempt,
    plan_manifest,
    semantic_signature,
    step,
    task,
    typed,
    unresolved_attempt,
)
from validation.hashing import canonical_task_hash, pair_group_hash, plan_artifact_hash, plan_content_hash

ROOT = Path(__file__).resolve().parents[1]
V = CrossObjectContractValidator()


def codes(violations):
    return {v.code for v in violations}


def test_normative_domain_contract_is_semantically_valid():
    domain = yaml.safe_load((ROOT / "docs/domain/blocks_world_v1.yaml").read_text(encoding="utf-8"))
    assert not V.validate_domain_contract(domain)
    assert all(isinstance(p["name"], str) for p in domain["state_representation"]["positive_predicates"])
    on = next(p for p in domain["state_representation"]["positive_predicates"] if p["name"] == "ON")
    assert on == {"name": "ON", "arity": 2, "roles": ["moving", "support"]}


def test_broken_structured_predicate_is_rejected():
    domain = yaml.safe_load((ROOT / "docs/domain/blocks_world_v1.yaml").read_text(encoding="utf-8"))
    bad = deepcopy(domain)
    unstack = next(a for a in bad["actions"] if a["name"] == "UNSTACK")
    unstack["preconditions"][0] = {"predicate": "ON", "args": ["moving"]}
    assert "CONTRACT_VIOLATION" in codes(V.validate_domain_contract(bad))


def test_valid_task_and_hash():
    assert not V.validate_task(task())


def test_ledger_refs_must_be_contiguous_and_aliases_unique():
    bad = task()
    bad["ledger"] = {
        "@B0": {"type": "BLOCK", "surface_alias": "block_0", "display_name": None},
        "@B2": {"type": "BLOCK", "surface_alias": "block_0", "display_name": None},
    }
    bad["difficulty"]["block_count"] = 2
    violations = V.validate_task(bad, verify_hash=False)
    assert {"CONTRACT_VIOLATION", "AMBIGUOUS_ALIAS"} <= codes(violations)


def test_task_hash_detects_semantic_change():
    bad = task()
    bad["goal"] = [["ON", "@B0", "@B1"], ["ON_TABLE", "@B1"]]
    assert canonical_task_hash(bad) != bad["canonical_task_hash"]
    assert "HASH_MISMATCH" in codes(V.validate_task(bad))


def test_multiple_supports_and_support_collision_are_rejected():
    bad = task()
    bad["ledger"]["@B2"] = {"type": "BLOCK", "surface_alias": "block_2", "display_name": None}
    bad["difficulty"]["block_count"] = 3
    bad["initial"] = [
        ["ON", "@B0", "@B1"],
        ["ON", "@B0", "@B2"],
        ["ON_TABLE", "@B1"],
        ["ON_TABLE", "@B2"],
        ["CLEAR", "@B0"],
        ["HAND_EMPTY"],
    ]
    assert "CONTRACT_VIOLATION" in codes(V.validate_task(bad, verify_hash=False))
    bad["initial"] = [
        ["ON", "@B0", "@B2"],
        ["ON", "@B1", "@B2"],
        ["ON_TABLE", "@B2"],
        ["CLEAR", "@B0"],
        ["CLEAR", "@B1"],
        ["HAND_EMPTY"],
    ]
    assert "CONTRACT_VIOLATION" in codes(V.validate_task(bad, verify_hash=False))


def test_goal_contract_forbids_non_location_predicates():
    bad = task()
    bad["goal"] = [["CLEAR", "@B0"]]
    assert "CONTRACT_VIOLATION" in codes(V.validate_task(bad, verify_hash=False))


def test_typed_action_self_argument_rejected():
    assert "INVALID_ACTION" in codes(V.validate_typed_action(typed("STACK", ("@B0", "@B0"))))


def test_a2c_signature_and_continuous_step_contracts():
    a2c = step(0, "PICK_UP", ("@B0",), "A2c")
    assert not V.validate_planner_step(a2c)
    a2c["intent_id"] = 3
    assert "CONTRACT_VIOLATION" in codes(V.validate_planner_step(a2c))
    a3 = step()
    a3["semantic_ref"] = a3["semantic_ref"].replace("#S00", "#S01")
    assert "CONTRACT_VIOLATION" in codes(V.validate_planner_step(a3))


def test_plan_has_terminal_end_and_hashes():
    plan, manifest = plan_manifest()
    assert not V.validate_plan_manifest(plan, manifest)
    bad = deepcopy(plan)
    bad["steps"] = bad["steps"][:-1]
    bad["plan_content_hash"] = plan_content_hash(bad)
    bad["plan_artifact_hash"] = plan_artifact_hash(bad)
    assert "CONTRACT_VIOLATION" in codes(V.validate_plan_manifest(bad, manifest))


def test_plan_supports_sixteen_actions_plus_end():
    plan, manifest = plan_manifest()
    # Use non-continuous A2 to avoid creating 16 semantic artifacts in this boundary test.
    steps = [step(i, "PICK_UP", ("@B0",), "A2") for i in range(16)] + [step(16, "END", (), "A2")]
    plan.update(
        planner_variant="A2",
        representation="TYPED_ONLY",
        semantic_artifact_manifest_sha256=None,
        steps=steps,
    )
    plan["plan_content_hash"] = plan_content_hash(plan)
    plan["plan_artifact_hash"] = plan_artifact_hash(plan)
    assert not V.validate_plan_manifest(plan, None)
    bad = deepcopy(plan)
    bad["steps"].insert(16, step(16, "PICK_UP", ("@B0",), "A2"))
    for i, s in enumerate(bad["steps"]):
        s["step_index"] = i
        s["step_id"] = f"S{i:02d}"
    assert "HORIZON_EXCEEDED" in codes(V.validate_plan_manifest(bad, None))


def test_plan_manifest_hash_and_tensor_uri_are_verified():
    plan, manifest = plan_manifest()
    bad = deepcopy(plan)
    bad["plan_content_hash"] = H1
    assert "HASH_MISMATCH" in codes(V.validate_plan_manifest(bad, manifest))
    bad_plan, bad_manifest = plan_manifest()
    bad_manifest["artifacts"][0]["tensor_sha256"] = H1
    assert "HASH_MISMATCH" in codes(V.validate_plan_manifest(bad_plan, bad_manifest))


def test_valid_attempts_for_every_raw_arm():
    for arm in sorted(V.INTERFACE_ARMS):
        assert not V.validate_attempt(attempt(arm, "STAGE1A_INTERFACE")), arm
    for arm in sorted({"E0_EQUAL_TOKENS_RAW", "E1_A3_FULL_PLAN_RAW", "E2_SHUFFLED_A3_FULL_PLAN_RAW", "E3_A3R_RANDOM_CODE_FULL_PLAN_RAW", "E4_A2C_STRUCTURED_FULL_PLAN_RAW", "E5_SELF_PLAN_RAW", "P_FULL_PLAN_REPLAY_RAW"}):
        assert not V.validate_attempt(attempt(arm, "STAGE1B_END_TO_END")), arm


def test_stage1b_requires_max_512_unpadded_and_32_guidance_tokens():
    bad = attempt("E1_A3_FULL_PLAN_RAW", "STAGE1B_END_TO_END")
    bad["prompt_tokens_total"] = 513
    bad["attended_prompt_tokens"] = 513
    bad["tokens_in"] = 513
    bad["padded_sequence_length"] = 513
    assert "PROMPT_BUDGET_EXCEEDED" in codes(V.validate_attempt(bad))
    bad = attempt("E1_A3_FULL_PLAN_RAW", "STAGE1B_END_TO_END")
    bad["added_block_tokens"] = 31
    assert "CONTRACT_VIOLATION" in codes(V.validate_attempt(bad))


def test_p_replay_requires_planner_identity_and_candidate():
    bad = attempt("P_FULL_PLAN_REPLAY_RAW", "STAGE1B_END_TO_END")
    bad["planner_config_sha256"] = None
    bad["candidate_typed_action"] = None
    assert "CONTRACT_VIOLATION" in codes(V.validate_attempt(bad))


def test_shuffled_requires_pre_outcome_certification_and_incompatibility():
    bad = attempt("I2_SHUFFLED_RAW", "STAGE1A_INTERFACE")
    bad["control_certification_hash"] = None
    assert "CONTRACT_VIOLATION" in codes(V.validate_attempt(bad))
    bad = attempt("I2_SHUFFLED_RAW", "STAGE1A_INTERFACE")
    bad["compatible_intent_ids"] = [3, 4]
    assert "CONTRACT_VIOLATION" in codes(V.validate_attempt(bad))


def test_semantic_unresolved_has_no_llm_call():
    assert not V.validate_attempt(unresolved_attempt())
    bad = unresolved_attempt()
    bad["prompt_hash"] = H1
    assert "CONTRACT_VIOLATION" in codes(V.validate_attempt(bad))


def test_stage1a_pair_group_is_exact_and_complete():
    group = [attempt(arm, "STAGE1A_INTERFACE") for arm in sorted(V.INTERFACE_ARMS)]
    assert not V.validate_pair_group(group)
    bad = deepcopy(group)
    bad[-1]["base_prompt_hash"] = H1
    assert "CONTRACT_VIOLATION" in codes(V.validate_pair_group(bad))
    assert "INCOMPLETE_PAIR" in codes(V.validate_pair_group(group[:-1]))


def test_episode_aggregates_attempts_exactly():
    a = attempt("E1_A3_FULL_PLAN_RAW", "STAGE1B_END_TO_END")
    e = episode_from_attempt(a)
    assert not V.validate_episode_attempts(e, [a])
    bad = deepcopy(e)
    bad["planner_calls"] = 0
    assert "CONTRACT_VIOLATION" in codes(V.validate_episode_attempts(bad, [a]))
    bad = deepcopy(e)
    bad["total_attended_tokens"] += 1
    assert "CONTRACT_VIOLATION" in codes(V.validate_episode_attempts(bad, [a]))


def test_pair_group_hash_is_not_arbitrary():
    bad = attempt()
    bad["pair_group_hash"] = H1
    expected = pair_group_hash(
        stage=bad["stage"], task_id=bad["task_id"], base_task_id=bad["base_task_id"],
        split=bad["split"], snapshot_id=bad["snapshot_id"],
        trajectory_policy=bad["trajectory_policy"], experiment_freeze_hash=bad["experiment_freeze_hash"]
    )
    assert expected != H1
    assert "HASH_MISMATCH" in codes(V.validate_attempt(bad))
