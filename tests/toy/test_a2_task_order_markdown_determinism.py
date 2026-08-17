from __future__ import annotations

import copy

import planner_toy.a2_sufficient_budget_task_order as experiment
import planner_toy.a2_sufficient_budget_task_order_validator as validator
from planner_toy.canonical import sha256


def _payload() -> dict:
    summaries = {}
    for arm, order in experiment.ARMS.items():
        summaries[arm] = {
            "task_order": list(order),
            "first_position0_operator_rescue_update_by_seed": {
                "17": 33,
                "29": 33,
                "43": 15,
            },
            "first_full_free_running_rescue_update_by_seed": {
                "17": 51,
                "29": 45,
                "43": 51,
            },
        }
    payload = {
        "schema_version": experiment.VERSION,
        "status": experiment.STATUS,
        "implementation_commit": "a" * 40,
        "source_sha256": "sha256:test-source",
        "source_files": [],
        "variant": experiment.VARIANT,
        "seeds": list(experiment.SEEDS),
        "arms": {name: list(order) for name, order in experiment.ARMS.items()},
        "checkpoint_epochs": list(experiment.CHECKPOINT_EPOCHS),
        "max_epoch": experiment.MAX_EPOCH,
        "heldout_accessed": False,
        "go_latent": "NOT EVALUATED",
        "dataset": {"evaluated_task_ids": list(experiment.EXPECTED_TRAIN_TASK_IDS)},
        "cross_seed_arm_summaries": summaries,
    }
    payload["canonical_identity"] = sha256(payload)
    return payload


def _reversed_mapping(mapping: dict) -> dict:
    return dict(reversed(list(mapping.items())))


def test_render_markdown_is_invariant_to_mapping_insertion_order() -> None:
    payload = _payload()
    expected = experiment.render_markdown(payload)

    reordered = copy.deepcopy(payload)
    reordered["arms"] = _reversed_mapping(reordered["arms"])
    reordered["cross_seed_arm_summaries"] = _reversed_mapping(
        reordered["cross_seed_arm_summaries"]
    )
    for summary in reordered["cross_seed_arm_summaries"].values():
        summary["first_position0_operator_rescue_update_by_seed"] = _reversed_mapping(
            summary["first_position0_operator_rescue_update_by_seed"]
        )
        summary["first_full_free_running_rescue_update_by_seed"] = _reversed_mapping(
            summary["first_full_free_running_rescue_update_by_seed"]
        )

    assert experiment.render_markdown(reordered) == expected


def test_run_persist_validate_round_trip_survives_canonical_json_sorting(
    tmp_path, monkeypatch
) -> None:
    payload = _payload()
    source = {
        "source_sha256": payload["source_sha256"],
        "source_files": payload["source_files"],
    }

    monkeypatch.setattr(
        experiment,
        "_produce_payload",
        lambda *, implementation_commit: copy.deepcopy(payload),
    )
    monkeypatch.setattr(
        experiment,
        "source_identity_at_commit",
        lambda implementation_commit: copy.deepcopy(source),
    )
    monkeypatch.setattr(
        validator,
        "validate_claims_from_evidence",
        lambda payload, *, implementation_commit: {
            "independent_claim_validation": "PASS"
        },
    )

    experiment.run(tmp_path, implementation_commit=payload["implementation_commit"])
    result = experiment.validate_experiment(
        tmp_path, implementation_commit=payload["implementation_commit"]
    )

    assert result["valid"] is True
    assert result["independent_claim_validation"] == "PASS"
