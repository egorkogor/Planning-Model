from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from planner_toy.learnability import OUTPUT_JSON as CORE_OUTPUT_JSON
from planner_toy.learnability_v0_2 import (
    CORE_DIRECTORY,
    OUTPUT_JSON,
    _artifact_identity,
    run,
    validate_payload,
)

REPOSITORY = Path(__file__).parents[2]
IMPLEMENTATION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
).strip()


@pytest.fixture(scope="module")
def sealed_v0_2_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("learnability-v0-2-tamper") / "run"
    run(
        root,
        implementation_commit=IMPLEMENTATION,
        seeds=(17,),
        task_ids=("bw-00000003",),
    )
    return root


def _mutate_claim(payload: dict, field: str) -> None:
    if field == "per_update_training_observation":
        payload[field][0]["gradient_norm_post_clip"] += 0.5
    elif field == "training_trajectory_summary":
        payload[field]["17"]["nonzero_gradient_update_count"] += 1
    elif field == "final_teacher_forced":
        payload[field]["overall"]["predicted_end_count"] += 1
    elif field == "gold_history_projected":
        payload[field][0]["exact_plan_match"] = not payload[field][0]["exact_plan_match"]
    elif field == "history_mode_summary":
        payload[field]["teacher_forced_operator_accuracy"] = 0.123
    elif field == "free_running":
        payload[field][0]["mean_end_probability"] = 0.123
    elif field == "free_running_aggregate":
        payload[field]["overall"]["zero_action_plan_count"] += 1
    elif field == "checkpoint_deltas":
        payload[field][0]["action_head_weight_delta_l2"] += 1.0
    elif field == "interpretation":
        payload[field]["cross_seed_localization_consistent"] = not payload[field][
            "cross_seed_localization_consistent"
        ]
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(field)


@pytest.mark.parametrize(
    "field",
    (
        "per_update_training_observation",
        "training_trajectory_summary",
        "final_teacher_forced",
        "gold_history_projected",
        "history_mode_summary",
        "free_running",
        "free_running_aggregate",
        "checkpoint_deltas",
        "interpretation",
    ),
)
def test_claim_bearing_section_cannot_be_tampered_and_rehashed(
    sealed_v0_2_root, field
) -> None:
    payload = json.loads((sealed_v0_2_root / OUTPUT_JSON).read_text())
    core_payload = json.loads(
        (sealed_v0_2_root / CORE_DIRECTORY / CORE_OUTPUT_JSON).read_text()
    )
    tampered = copy.deepcopy(payload)
    _mutate_claim(tampered, field)
    tampered["canonical_identity"] = _artifact_identity(tampered)

    with pytest.raises(
        ValueError,
        match=rf"LEARNABILITY_V0_2_CLAIM_EVIDENCE_MISMATCH:{field}$",
    ):
        validate_payload(tampered, core_payload=core_payload)
