from __future__ import annotations

import copy

import pytest
import torch

from planner_toy import a2_gradient_clipping as producer
from planner_toy import a2_gradient_clipping_validator as validator
from planner_toy.canonical import sha256

FAKE_HASH_A = "sha256:" + "1" * 64
FAKE_HASH_B = "sha256:" + "2" * 64
FAKE_HASH_C = "sha256:" + "3" * 64
FAKE_SOURCE = {
    "source_files": [{"path": "fixture", "sha256": FAKE_HASH_A}],
    "source_sha256": FAKE_HASH_B,
}
IMPLEMENTATION = "a" * 40
GRADIENT_MANIFEST = [
    {"index": 0, "name": "shared.weight", "dtype": "torch.float32", "shape": [1]},
    {
        "index": 1,
        "name": "heads.arg1_pointer.weight",
        "dtype": "torch.float32",
        "shape": [1],
    },
    {
        "index": 2,
        "name": "heads.arg2_pointer.weight",
        "dtype": "torch.float32",
        "shape": [1],
    },
]


def _rows():
    return {
        "schema_version": "fixture",
        "frozen_dataset_lineage_hash": FAKE_HASH_A,
        "evaluated_train_split_hash": FAKE_HASH_B,
        "train_task_ids": list(validator.TASKS),
        "train": [
            {"task_id": "bw-00000001", "oracle_work_plan": [["END"]]},
            {"task_id": "bw-00000002", "oracle_work_plan": [["UNSTACK", "A", "B"], ["END"]]},
            {"task_id": "bw-00000003", "oracle_work_plan": [["UNSTACK", "A", "B"], ["END"]]},
        ],
    }


def _losses(task_id: str):
    if task_id == "bw-00000001":
        return 1.0, None, None, 1.0
    a = torch.tensor(1.0, dtype=torch.float32)
    b = torch.tensor(0.5, dtype=torch.float32)
    c = torch.tensor(0.25, dtype=torch.float32)
    return 1.0, 0.5, 0.25, float(a + b + c)


def _activity(task_id: str):
    pointer_state = "NO_GRAD" if task_id == "bw-00000001" else "GRAD"
    return [
        {"index": 0, "name": "shared.weight", "state": "GRAD"},
        {"index": 1, "name": "heads.arg1_pointer.weight", "state": pointer_state},
        {"index": 2, "name": "heads.arg2_pointer.weight", "state": pointer_state},
    ]


def _update(index: int, arm: str):
    task_id = validator.TASKS[index % 3]
    op, arg1, arg2, total = _losses(task_id)
    threshold = validator.ARMS[arm]
    clipped = arm == "clip_1_0"
    activity = _activity(task_id)
    return {
        "update_index": index,
        "epoch_index": index // 3,
        "task_id": task_id,
        "operator_loss": op,
        "arg1_pointer_loss": arg1,
        "arg2_pointer_loss": arg2,
        "total_loss": total,
        "operator_target_count": 1 if task_id == "bw-00000001" else 2,
        "arg1_target_count": 0 if task_id == "bw-00000001" else 1,
        "arg2_target_count": 0 if task_id == "bw-00000001" else 1,
        "gradient_norm": 2.0,
        "gradient_clip_norm": threshold,
        "clipping_policy": arm,
        "clip_threshold": threshold,
        "clipping_occurred": clipped,
        "pre_intervention_global_l2_norm": 2.0,
        "clip_primitive_return_norm": 2.0 if threshold is not None else None,
        "threshold_exceeded": clipped,
        "gradient_mutated": clipped,
        "intervention_applied": clipped,
        "post_clip_global_l2_norm": 1.0 if clipped else 2.0,
        "gradient_before_sha256": FAKE_HASH_A,
        "gradient_after_sha256": FAKE_HASH_B if clipped else FAKE_HASH_A,
        "gradient_hash_version": validator.GRADIENT_HASH_VERSION,
        "gradient_parameter_manifest_sha256": sha256(GRADIENT_MANIFEST),
        "gradient_activity": activity,
        "gradient_activity_sha256": sha256(activity),
        "operator_position_weight": 1.0 if task_id == "bw-00000001" else 0.5,
    }


def _epoch(epoch: int):
    rescued = epoch >= 10
    return {
        "epoch": epoch,
        "update_count": epoch * 3,
        "position0": [
            {
                "task_id": task,
                "gold_operator": "END" if task.endswith("1") else "UNSTACK",
                "predicted_operator": (
                    "END" if task.endswith("1") else ("UNSTACK" if rescued else "END")
                ),
                "operator_correct": True if task.endswith("1") else rescued,
                "probability_gold_operator": 1.0,
                "operator_nll": -0.0,
                "probability_end": 1.0 if task.endswith("1") else 0.0,
            }
            for task in validator.TASKS
        ],
        "free_running": [
            {
                "task_id": task,
                "initial_goal_satisfied": task.endswith("1"),
                "predicted_plan": [["END"]],
                "predicted_plan_length": 0,
                "exact_plan_match": False,
                "final_goal_success": True if task.endswith("1") else rescued,
                "failure_code": None,
            }
            for task in validator.TASKS
        ],
    }


def _result(arm: str, seed: int):
    epochs = [_epoch(epoch) for epoch in range(1, 101)]
    event = {"epoch": 10, "update_count": 30}
    return {
        "arm": arm,
        "clipping_policy": arm,
        "clip_threshold": validator.ARMS[arm],
        "task_order": list(validator.TASKS),
        "seed": seed,
        "initialization_canonical_sha256": FAKE_HASH_A,
        "final_trained_canonical_sha256": FAKE_HASH_B if arm == "clip_1_0" else FAKE_HASH_C,
        "final_optimizer_canonical_sha256": (
            FAKE_HASH_B if arm == "clip_1_0" else FAKE_HASH_C
        ),
        "gradient_parameter_manifest": copy.deepcopy(GRADIENT_MANIFEST),
        "gradient_parameter_manifest_sha256": sha256(GRADIENT_MANIFEST),
        "updates": [_update(index, arm) for index in range(300)],
        "checkpoints": [
            {
                "epoch": epoch,
                "update_count": epoch * 3,
                "trained_canonical_sha256": (
                    FAKE_HASH_B if arm == "clip_1_0" else FAKE_HASH_C
                ),
                "optimizer_canonical_sha256": (
                    FAKE_HASH_B if arm == "clip_1_0" else FAKE_HASH_C
                ),
            }
            for epoch in validator.CHECKPOINTS
        ],
        "epoch_evidence": epochs,
        "rescue_events": {
            "first_position0_operator_rescue": event,
            "first_full_free_running_rescue": event,
        },
        "rescue_persistence": {
            "position0_operator_rescue": {"10": True, "30": True, "100": True},
            "full_free_running_rescue": {"10": True, "30": True, "100": True},
        },
    }


def _synthetic_reference(seed: int):
    return validator._control_projection(_result("clip_1_0", seed))


def _synthetic_frozen_prefix(seed: int):
    projection = _synthetic_reference(seed)
    checkpoint = next(item for item in projection["checkpoints"] if item["epoch"] == 3)
    return {
        "initialization_canonical_sha256": projection["initialization_canonical_sha256"],
        "trained_canonical_sha256": checkpoint["trained_canonical_sha256"],
        "optimizer_canonical_sha256": checkpoint["optimizer_canonical_sha256"],
        "updates": projection["updates"][:9],
    }


def _payload():
    results = [_result(arm, seed) for seed in validator.SEEDS for arm in validator.ARMS]
    indexed = {(r["arm"], r["seed"]): r for r in results}
    equivalence = {}
    for seed in validator.SEEDS:
        projection = validator._control_projection(indexed[("clip_1_0", seed)])
        projection_hash = sha256(projection)
        frozen_prefix = _synthetic_frozen_prefix(seed)
        equivalence[str(seed)] = {
            "seed": seed,
            "status": "PASS",
            "scope": "WHOLE_300_UPDATE_TRAJECTORY",
            "trace_fields": list(validator.REFERENCE_FIELDS),
            "reference_projection": copy.deepcopy(projection),
            "reference_projection_sha256": projection_hash,
            "candidate_projection_sha256": projection_hash,
            "prefix_9_update_projection_sha256": sha256(
                validator._control_prefix_projection(projection)
            ),
            "reference_historical_prefix_equivalence": {
                "status": "PASS",
                "control": copy.deepcopy(frozen_prefix),
                "arm_prefix": copy.deepcopy(frozen_prefix),
            },
            "checkpoint_epochs": list(validator.CHECKPOINTS),
            "reference_checkpoints": projection["checkpoints"],
            "candidate_checkpoints": projection["checkpoints"],
            "reference_final_trained_canonical_sha256": projection[
                "final_trained_canonical_sha256"
            ],
            "candidate_final_trained_canonical_sha256": projection[
                "final_trained_canonical_sha256"
            ],
            "reference_final_optimizer_canonical_sha256": projection[
                "final_optimizer_canonical_sha256"
            ],
            "candidate_final_optimizer_canonical_sha256": projection[
                "final_optimizer_canonical_sha256"
            ],
            "reference_rescue_events": projection["rescue_events"],
            "candidate_rescue_events": projection["rescue_events"],
            "reference_rescue_persistence": projection["rescue_persistence"],
            "candidate_rescue_persistence": projection["rescue_persistence"],
        }
    consistency = {
        str(seed): {
            "first_actual_or_pregradient_difference_update_index": 0,
            "all_arm_identical_preintervention_update_count": 1,
            "clip_5_0_clipped_update_count": 0,
            "clip_5_0_first_actual_intervention_update_index": None,
            "clip_5_0_vs_no_clip_identical_preintervention_update_count": 300,
            "clip_5_0_vs_no_clip_no_effect_equivalence": {
                "status": "PASS",
                "projection_sha256": sha256(
                    {
                        "losses": [
                            (
                                u["operator_loss"],
                                u["arg1_pointer_loss"],
                                u["arg2_pointer_loss"],
                                u["total_loss"],
                                u["gradient_before_sha256"],
                            )
                            for u in indexed[("clip_5_0", seed)]["updates"]
                        ],
                        "final_model": FAKE_HASH_C,
                        "final_optimizer": FAKE_HASH_C,
                        "rescue_events": indexed[("clip_5_0", seed)]["rescue_events"],
                    }
                ),
            },
        }
        for seed in validator.SEEDS
    }
    dataset = _rows()
    payload = {
        "experiment_version": validator.VERSION,
        "schema_version": validator.VERSION,
        "status": validator.STATUS,
        "task_id": validator.TASK_ID,
        "implementation_commit": IMPLEMENTATION,
        **FAKE_SOURCE,
        "runtime": {},
        "variant": "A2",
        "seeds": list(validator.SEEDS),
        "arms": {name: {"clip_threshold": threshold} for name, threshold in validator.ARMS.items()},
        "canonical_task_order": list(validator.TASKS),
        "max_epoch": 100,
        "updates_per_epoch": 3,
        "optimizer_updates_per_seed_arm": 300,
        "checkpoint_epochs": list(validator.CHECKPOINTS),
        "persistence_checkpoint_epochs": list(validator.PERSISTENCE_CHECKPOINTS),
        "optimizer_contract": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "parameter_order": "planner_toy.quality._optimizer_named_parameters",
        },
        "clipping_contract": copy.deepcopy(validator.CLIPPING_CONTRACT),
        "control_equivalence": {
            "reference": "fixture",
            "required_status": "PASS",
            "by_seed": equivalence,
        },
        "rescue_definitions": {},
        "heldout_accessed": False,
        "heldout_task_ids": ["bw-00000004", "bw-00000005"],
        "dataset": {
            "schema_version": dataset["schema_version"],
            "frozen_dataset_lineage_hash": dataset["frozen_dataset_lineage_hash"],
            "evaluated_train_split_hash": dataset["evaluated_train_split_hash"],
            "dataset_lineage_order": dataset["train_task_ids"],
            "evaluated_task_ids": list(validator.TASKS),
        },
        "arm_seed_results": results,
        "gradient_evidence_commitments": [
            validator._gradient_evidence_commitment(r) for r in results
        ],
        "clipping_summaries": [validator._summary(r) for r in results],
        "cross_seed_clipping_aggregates": validator._cross_seed_clipping_aggregates(results),
        "paired_causal_contrasts": validator._contrasts(indexed),
        "intervention_consistency": consistency,
        "interpretation_policy": {
            "producer_scientific_verdict": None,
            "validator_scientific_verdict": None,
            "reviewer_owns_interpretation": True,
        },
        "go_latent": "NOT EVALUATED",
    }
    payload["canonical_identity"] = sha256(payload)
    return payload


@pytest.fixture
def validator_env(monkeypatch):
    monkeypatch.setattr(validator, "_runtime_identity", lambda: {})
    monkeypatch.setattr(validator, "_source_identity_at_commit", lambda commit: FAKE_SOURCE)
    monkeypatch.setattr(validator, "generate_train_only", _rows)
    monkeypatch.setattr(
        validator,
        "_expected_gradient_parameter_manifest",
        lambda seed: copy.deepcopy(GRADIENT_MANIFEST),
    )
    monkeypatch.setattr(validator, "_validate_position0", lambda *args, **kwargs: None)
    monkeypatch.setattr(validator, "_validate_free", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        validator,
        "reconstruct_reference",
        lambda rows, *, seed, dataset_hash: (
            _synthetic_reference(seed),
            _synthetic_frozen_prefix(seed),
        ),
    )


def _reject(payload, validator_env):
    with pytest.raises(ValueError):
        validator.validate_claims_from_evidence(payload, implementation_commit=IMPLEMENTATION)


def _resign(payload):
    results = payload["arm_seed_results"]
    indexed = {(r["arm"], r["seed"]): r for r in results}
    payload["gradient_evidence_commitments"] = [
        validator._gradient_evidence_commitment(result) for result in results
    ]
    payload["intervention_consistency"] = producer._intervention_consistency(results)
    payload["paired_causal_contrasts"] = validator._contrasts(indexed)
    payload["clipping_summaries"] = [validator._summary(result) for result in results]
    payload["cross_seed_clipping_aggregates"] = validator._cross_seed_clipping_aggregates(results)
    payload["canonical_identity"] = sha256(
        {key: value for key, value in payload.items() if key != "canonical_identity"}
    )


def test_synthetic_contract_validates(validator_env):
    result = validator.validate_claims_from_evidence(
        _payload(), implementation_commit=IMPLEMENTATION
    )
    assert result["valid"] is True
    assert result["control_equivalence"] == "PASS"
    assert result["go_latent"] == "NOT EVALUATED"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["arm_seed_results"].pop(),
        lambda p: p["arm_seed_results"].append(copy.deepcopy(p["arm_seed_results"][0])),
        lambda p: p["arm_seed_results"].__setitem__(
            0, {**p["arm_seed_results"][0], "arm": "clip_9_0"}
        ),
        lambda p: p["arm_seed_results"].__setitem__(0, {**p["arm_seed_results"][0], "seed": 999}),
        lambda p: p["arm_seed_results"].__setitem__(0, {**p["arm_seed_results"][0], "seed": 29}),
        lambda p: p["arm_seed_results"][0].__setitem__(
            "task_order", list(reversed(validator.TASKS))
        ),
        lambda p: p["arm_seed_results"][0]["updates"][0].__setitem__("task_id", "bw-00000003"),
        lambda p: p["arm_seed_results"][0]["updates"][0].__setitem__("task_id", "bw-00000004"),
        lambda p: p["arm_seed_results"][0]["updates"][0].__setitem__("clip_threshold", 2.0),
        lambda p: p["clipping_contract"]["clip_1_0"].__setitem__("max_norm", 2.0),
        lambda p: next(
            r for r in p["arm_seed_results"] if r["arm"] == "no_clip"
        )["updates"][0].__setitem__("intervention_applied", True),
        lambda p: p["clipping_summaries"][0]["full_trajectory"].__setitem__(
            "clipped_update_count", 0
        ),
        lambda p: p["arm_seed_results"][0]["updates"][0].__setitem__(
            "pre_intervention_global_l2_norm", 0.25
        ),
        lambda p: p["arm_seed_results"][0]["updates"][0].__setitem__(
            "gradient_before_sha256", FAKE_HASH_C
        ),
        lambda p: p["arm_seed_results"][0]["gradient_parameter_manifest"][0].__setitem__(
            "name", "tampered.weight"
        ),
        lambda p: p["arm_seed_results"][0]["rescue_events"].__setitem__(
            "first_position0_operator_rescue", None
        ),
        lambda p: p["arm_seed_results"][0]["rescue_events"][
            "first_full_free_running_rescue"
        ].__setitem__("update_count", 31),
        lambda p: p["arm_seed_results"][0]["rescue_persistence"][
            "position0_operator_rescue"
        ].__setitem__("10", False),
        lambda p: p["paired_causal_contrasts"]["clip_5_0"]["by_seed"][
            "17"
        ].__setitem__("control_clipped_update_count", 0),
        lambda p: p["cross_seed_clipping_aggregates"]["clip_1_0"].__setitem__(
            "total_update_count", 899
        ),
        lambda p: p["gradient_evidence_commitments"][0].__setitem__("sha256", FAKE_HASH_C),
        lambda p: p.__setitem__("implementation_commit", "b" * 40),
        lambda p: p.__setitem__("source_sha256", FAKE_HASH_C),
        lambda p: p.__setitem__("heldout_task_ids", ["bw-00000004"]),
        lambda p: p["control_equivalence"]["by_seed"]["17"].__setitem__(
            "reference_projection_sha256", FAKE_HASH_C
        ),
        lambda p: p["control_equivalence"]["by_seed"]["17"][
            "reference_projection"
        ]["updates"][0].__setitem__("total_loss", 99.0),
        lambda p: p["arm_seed_results"][0]["updates"].pop(),
        lambda p: p["arm_seed_results"][0]["updates"].append(
            copy.deepcopy(p["arm_seed_results"][0]["updates"][-1])
        ),
        lambda p: p["clipping_summaries"][0].__setitem__("first_9_updates", {}),
        lambda p: p["arm_seed_results"][0]["updates"][0]["gradient_activity"][1].__setitem__(
            "state", "GRAD"
        ),
        lambda p: p["arm_seed_results"][0]["updates"][0]["gradient_activity"].reverse(),
        lambda p: p["arm_seed_results"][0]["updates"][0]["gradient_activity"].pop(),
        lambda p: p["arm_seed_results"][0]["updates"][0]["gradient_activity"].append(
            {"index": 3, "name": "extra.weight", "state": "GRAD"}
        ),
        lambda p: p["arm_seed_results"][0]["updates"][0]["gradient_activity"][1].__setitem__(
            "state", "ZERO"
        ),
        lambda p: p["arm_seed_results"][0]["updates"][0].__setitem__(
            "gradient_hash_version", "a2-named-gradients-exact/1.0"
        ),
    ],
)
def test_adversarial_tampering_rejected(validator_env, mutation):
    payload = _payload()
    mutation(payload)
    payload["canonical_identity"] = sha256(
        {k: v for k, v in payload.items() if k != "canonical_identity"}
    )
    _reject(payload, validator_env)


def test_self_consistent_activity_zero_tamper_rejected_by_independent_anchor(validator_env):
    payload = _payload()
    tampered_activity = [
        {"index": 0, "name": "shared.weight", "state": "GRAD"},
        {"index": 1, "name": "heads.arg1_pointer.weight", "state": "GRAD"},
        {"index": 2, "name": "heads.arg2_pointer.weight", "state": "GRAD"},
    ]
    before_items = [
        ("shared.weight", torch.ones(1, dtype=torch.float32)),
        ("heads.arg1_pointer.weight", torch.zeros(1, dtype=torch.float32)),
        ("heads.arg2_pointer.weight", torch.zeros(1, dtype=torch.float32)),
    ]
    after_items = [
        ("shared.weight", torch.tensor([0.5], dtype=torch.float32)),
        ("heads.arg1_pointer.weight", torch.zeros(1, dtype=torch.float32)),
        ("heads.arg2_pointer.weight", torch.zeros(1, dtype=torch.float32)),
    ]
    before_hash = producer._named_gradient_sha256(before_items)
    after_hash = producer._named_gradient_sha256(after_items)
    for result in payload["arm_seed_results"]:
        if result["seed"] != 17:
            continue
        update = result["updates"][0]
        update["gradient_activity"] = copy.deepcopy(tampered_activity)
        update["gradient_activity_sha256"] = sha256(update["gradient_activity"])
        update["gradient_before_sha256"] = before_hash
        update["gradient_after_sha256"] = after_hash if result["arm"] == "clip_1_0" else before_hash
    _resign(payload)
    with pytest.raises(
        ValueError,
        match=r"^A2_CLIP_VALIDATOR_GRADIENT_ACTIVITY_SEMANTICS:clip_1_0:17:0$",
    ):
        validator.validate_claims_from_evidence(payload, implementation_commit=IMPLEMENTATION)


def test_self_copied_reference_and_candidate_rejected(validator_env):
    payload = _payload()
    result = next(
        item
        for item in payload["arm_seed_results"]
        if item["arm"] == "clip_1_0" and item["seed"] == 17
    )
    result["final_trained_canonical_sha256"] = FAKE_HASH_C
    result["final_optimizer_canonical_sha256"] = FAKE_HASH_C
    result["checkpoints"][-1]["trained_canonical_sha256"] = FAKE_HASH_C
    result["checkpoints"][-1]["optimizer_canonical_sha256"] = FAKE_HASH_C
    projection = validator._control_projection(result)
    record = payload["control_equivalence"]["by_seed"]["17"]
    record["reference_projection"] = copy.deepcopy(projection)
    record["reference_projection_sha256"] = sha256(projection)
    record["candidate_projection_sha256"] = sha256(projection)
    record["prefix_9_update_projection_sha256"] = sha256(
        validator._control_prefix_projection(projection)
    )
    record["reference_checkpoints"] = copy.deepcopy(projection["checkpoints"])
    record["candidate_checkpoints"] = copy.deepcopy(projection["checkpoints"])
    for prefix in ("reference", "candidate"):
        record[f"{prefix}_final_trained_canonical_sha256"] = FAKE_HASH_C
        record[f"{prefix}_final_optimizer_canonical_sha256"] = FAKE_HASH_C
    payload["canonical_identity"] = sha256(
        {k: v for k, v in payload.items() if k != "canonical_identity"}
    )
    _reject(payload, validator_env)


def test_gradient_hash_is_read_only_and_deterministic():
    p1 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([1.0])
    items = [("a", p1.grad), ("b", p2.grad)]
    before = [g.clone() for _, g in items]
    first = producer._named_gradient_sha256(items)
    second = producer._named_gradient_sha256(items)
    assert first == second
    assert all(torch.equal(old, new) for old, (_, new) in zip(before, items, strict=True))


def test_reduced_real_model_intervention_semantics(monkeypatch):
    _dataset, rows = producer._train_rows()
    assert [row["task_id"] for row in rows] == list(producer.CANONICAL_ORDER)
    assert set(producer.CANONICAL_ORDER).isdisjoint({"bw-00000004", "bw-00000005"})

    calls: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def observed(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observed)
    results = {
        arm: producer._train_arm(
            rows,
            seed=17,
            arm=arm,
            max_epoch=1,
            checkpoint_epochs=(),
        )
        for arm in producer.ARMS
    }

    assert producer.GRADIENT_HASH_VERSION == validator.GRADIENT_HASH_VERSION
    assert producer.GRADIENT_HASH_VERSION == "a2-named-gradients-exact/1.1"
    first = [results[arm]["updates"][0] for arm in producer.ARMS]
    assert len({update["gradient_before_sha256"] for update in first}) == 1
    assert len({update["pre_intervention_global_l2_norm"] for update in first}) == 1
    assert calls.count(1.0) == 3
    assert calls.count(5.0) == 3
    assert len(calls) == 6
    assert any(update["intervention_applied"] for update in results["clip_1_0"]["updates"])
    assert all(
        update["gradient_mutated"]
        == (update["gradient_before_sha256"] != update["gradient_after_sha256"])
        for result in results.values()
        for update in result["updates"]
    )
    assert all(
        not update["gradient_mutated"] and not update["intervention_applied"]
        for update in results["no_clip"]["updates"]
    )
    assert all(
        update["clip_primitive_return_norm"] is None
        for update in results["no_clip"]["updates"]
    )
    for result in results.values():
        manifest = result["gradient_parameter_manifest"]
        manifest_hash = result["gradient_parameter_manifest_sha256"]
        for update in result["updates"]:
            assert update["gradient_parameter_manifest_sha256"] == manifest_hash
            assert update["gradient_activity_sha256"] == sha256(update["gradient_activity"])
            assert update["gradient_activity"] == validator._expected_gradient_activity(
                manifest,
                arg1_target_count=update["arg1_target_count"],
                arg2_target_count=update["arg2_target_count"],
            )
    task1 = results["no_clip"]["updates"][0]
    by_name = {entry["name"]: entry["state"] for entry in task1["gradient_activity"]}
    assert by_name["heads.arg1_pointer.weight"] == "NO_GRAD"
    assert by_name["heads.arg2_pointer.weight"] == "NO_GRAD"


def test_bridge_task_literal_is_typed_and_has_three_stage_plan():
    from scripts.run_reviewer_execution_bridge import (
        REQUEST_PATTERN,
        TASKS,
        BridgeRequest,
        task_plan,
    )

    assert "a2-gradient-clipping-v1" in TASKS
    body = "/reviewer-bridge/v1 task=a2-gradient-clipping-v1 request=clip-test-0001 sha=" + "a" * 40
    assert REQUEST_PATTERN.fullmatch(body)
    assert not REQUEST_PATTERN.fullmatch(body + " arm=no_clip")
    request = BridgeRequest(
        "a2-gradient-clipping-v1",
        "clip-test-0001",
        "a" * 40,
        1,
        "egorkogor",
        36,
        "egorkogor/Planning-Model",
    )
    plan = task_plan(request, __import__("pathlib").Path("/tmp/evidence"))
    assert [name for name, _ in plan] == [
        "fixed-target-preflight",
        "producer",
        "independent-validator",
    ]
    assert "scripts.run_a2_gradient_clipping" in plan[1][1]
    assert "scripts.run_a2_gradient_clipping" in plan[2][1]
    assert "--validate-only" in plan[2][1]


def test_scientific_contract_has_no_heldout_or_verdict_escape_hatches():
    assert producer.CANONICAL_ORDER == validator.TASKS
    assert set(producer.CANONICAL_ORDER).isdisjoint({"bw-00000004", "bw-00000005"})
    assert producer.ARMS == {"clip_1_0": 1.0, "clip_5_0": 5.0, "no_clip": None}
    assert producer.MAX_EPOCH == 100
    assert producer.EXPECTED_UPDATES == 300
    assert producer.CLIPPING_CONTRACT == validator.CLIPPING_CONTRACT


def test_bridge_workflow_adds_only_typed_clipping_literal():
    from pathlib import Path

    workflow = Path(".github/workflows/reviewer-execution-bridge.yml").read_text(
        encoding="utf-8"
    )
    assert workflow.count("a2-gradient-clipping-v1") == 2
    assert (
        "task=(status-v1|preflight-v1|a2-sufficient-budget-task-order-v1|"
        "a2-gradient-clipping-v1) "
    ) in workflow
    assert (
        "status-v1|preflight-v1|a2-sufficient-budget-task-order-v1|"
        "a2-gradient-clipping-v1) ;;"
    ) in workflow
    assert "workflow_dispatch:" not in workflow


def test_source_inventory_closes_new_claim_bearing_paths():
    assert producer.SOURCE_FILES == validator.SOURCE_FILES
    required = {
        ".github/workflows/reviewer-execution-bridge.yml",
        "docs/evaluations/A2_GRADIENT_CLIPPING_CAUSAL_SPEC_RU.md",
        "planner_toy/a2_gradient_clipping.py",
        "planner_toy/a2_gradient_clipping_reference.py",
        "planner_toy/a2_gradient_clipping_validator.py",
        "scripts/run_a2_gradient_clipping.py",
        "scripts/run_reviewer_execution_bridge.py",
    }
    assert required.issubset(set(producer.SOURCE_FILES))
