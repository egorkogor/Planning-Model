from __future__ import annotations

import copy
import hashlib
import json

import pytest

from planner_toy.e2e import parse_work_plan, validate_lineage


def test_no_end_records_zero_execution_failure(tmp_path, replay_dirs) -> None:
    root = tmp_path / "failed"
    from planner_toy.e2e import run

    run(root, failure_mode="NO_END", reuse_from=replay_dirs[0])
    result = json.loads((root / "run-result.json").read_text())
    assert result["failure_code"] == "PLAN_NO_END"
    assert result["executed_action_count"] == 0
    assert json.loads((root / "attempt-log.json").read_text()) == []
    assert json.loads((root / "episode-plan-manifest.json").read_text())["plan_status"] == "FAILED"
    assert json.loads((root / "episode-log.json").read_text())["plan_generation_status"] == "FAILED"
    assert json.loads((root / "evaluation-result.json").read_text())["success"] is False


def test_two_clean_replays_are_recursively_byte_identical(replay_dirs) -> None:
    one, two, result_one, result_two = replay_dirs
    assert result_one == result_two
    relative = sorted(path.relative_to(one) for path in one.rglob("*") if path.is_file())
    assert relative
    assert all((one / name).read_bytes() == (two / name).read_bytes() for name in relative)


def test_multistep_plan_executes_and_changes_state(e2e_artifacts) -> None:
    plan = e2e_artifacts["plan"]
    attempts = e2e_artifacts["attempts"]
    assert len(plan["steps"]) == 5  # four actions plus END
    assert len(attempts) == 4
    assert all(item["state_before_hash"] != item["state_after_hash"] for item in attempts)
    assert e2e_artifacts["evaluation"]["success"] is True
    assert e2e_artifacts["manifest"]["planner_call_count"] == 1


def test_invalid_plan_is_not_repaired() -> None:
    with pytest.raises(ValueError):
        parse_work_plan([["PICK_UP", "@UNKNOWN"], ["END"]], ["@B0"])


def test_checkpoint_config_and_work_plan_paths_bind_real_files(replay_dirs, e2e_artifacts) -> None:
    root = replay_dirs[0]

    def digest(path):
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    request = e2e_artifacts["request"]
    assert request["planner_checkpoint_sha256"] == digest(root / "model/trained.pt")
    assert request["planner_config_sha256"] == digest(root / "planner-config.json")
    assert (root / e2e_artifacts["manifest"]["work_plan_path"]).is_file()


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("plan", "steps", []),
        ("manifest", "planner_seed", 42),
        ("episode", "goal_success", False),
        ("evaluation", "success", False),
    ],
)
def test_real_artifact_mutations_fail_closed(e2e_artifacts, target, field, value) -> None:
    mutated = copy.deepcopy(e2e_artifacts)
    mutated[target][field] = value
    with pytest.raises(ValueError):
        validate_lineage(
            mutated["task"],
            mutated["request"],
            mutated["plan"],
            mutated["manifest"],
            mutated["attempts"],
            mutated["episode"],
            mutated["evaluation"],
        )


def test_rehashed_attempt_action_mutation_is_semantically_rejected(e2e_artifacts) -> None:
    mutated = copy.deepcopy(e2e_artifacts)
    mutated["attempts"][0]["candidate_typed_action"]["action"] = "STACK"
    with pytest.raises(ValueError, match="frozen action"):
        validate_lineage(
            mutated["task"],
            mutated["request"],
            mutated["plan"],
            mutated["manifest"],
            mutated["attempts"],
            mutated["episode"],
            mutated["evaluation"],
        )
