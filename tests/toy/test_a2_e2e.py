from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from planner_toy.canonical import artifact_hash
from planner_toy.e2e import parse_work_plan, toy_hash, validate_lineage


def validate(values):
    validate_lineage(
        root=values["root"],
        task=values["task"],
        request=values["request"],
        config=values["config"],
        checkpoint=values["checkpoint"],
        work_plan=values["work_plan"],
        manifest=values["manifest"],
        attempts=values["attempts"],
        episode=values["episode"],
        evaluation=values["evaluation"],
    )


def rehash(values):
    plan = values["work_plan"]
    content = {
        k: v for k, v in plan.items() if k not in {"plan_content_hash", "plan_artifact_hash"}
    }
    plan["plan_content_hash"] = toy_hash("plan_content", content)
    artifact = {k: v for k, v in plan.items() if k != "plan_artifact_hash"}
    plan["plan_artifact_hash"] = toy_hash("plan_artifact", artifact)
    values["manifest"]["work_plan_content_hash"] = plan["plan_content_hash"]
    values["manifest"]["work_plan_artifact_hash"] = plan["plan_artifact_hash"]
    values["manifest"]["manifest_hash"] = artifact_hash(values["manifest"], "manifest_hash")
    for attempt in values["attempts"]:
        attempt["episode_plan_manifest_hash"] = values["manifest"]["manifest_hash"]
        attempt["plan_content_hash"] = plan["plan_content_hash"]
        attempt["plan_artifact_hash"] = plan["plan_artifact_hash"]
    values["episode"]["episode_plan_manifest_hash"] = values["manifest"]["manifest_hash"]
    values["evaluation"]["episode_plan_manifest_hash"] = values["manifest"]["manifest_hash"]
    values["evaluation"]["attempt_log_hash"] = toy_hash("attempt_jsonl", values["attempts"])
    values["evaluation"]["evaluation_result_hash"] = artifact_hash(
        values["evaluation"], "evaluation_result_hash"
    )


def test_toy_schemas_validate_emitted_artifacts(e2e_artifacts) -> None:
    root = Path("planner_toy/schemas")
    mapping = {
        "request": "toy_planner_request",
        "config": "toy_development_config",
        "checkpoint": "toy_checkpoint_manifest",
        "optimizer": "toy_optimizer_evidence",
        "work_plan": "toy_work_plan",
        "manifest": "toy_episode_plan_manifest",
        "episode": "toy_episode_log",
        "evaluation": "toy_evaluation_result",
    }
    for key, schema in mapping.items():
        Draft202012Validator(json.loads((root / f"{schema}.schema.json").read_text())).validate(
            e2e_artifacts[key]
        )
    schema = json.loads((root / "toy_attempt_log.schema.json").read_text())
    for attempt in e2e_artifacts["attempts"]:
        Draft202012Validator(schema).validate(attempt)


def test_two_clean_replays_are_recursively_byte_identical(replay_dirs) -> None:
    one, two, result_one, result_two = replay_dirs
    assert result_one == result_two
    names = sorted(p.relative_to(one) for p in one.rglob("*") if p.is_file())
    assert all((one / name).read_bytes() == (two / name).read_bytes() for name in names)


def test_multistep_and_bindings(e2e_artifacts) -> None:
    assert len(e2e_artifacts["work_plan"]["steps"]) == 5
    assert len(e2e_artifacts["attempts"]) == 4
    assert all(a["state_before_hash"] != a["state_after_hash"] for a in e2e_artifacts["attempts"])
    validate(e2e_artifacts)


def test_failure_is_empty_jsonl_and_semantically_valid(failure_replay_dirs) -> None:
    root, second, result, second_result = failure_replay_dirs
    assert result["success"] is False
    assert result == second_result
    names = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
    assert all((root / name).read_bytes() == (second / name).read_bytes() for name in names)
    assert (root / "attempt-log.jsonl").read_bytes() == b""
    assert not (root / "results/development/plans/work-plan.json").exists()

    def load(path):
        return json.loads((root / path).read_bytes())

    from planner_toy.dataset import generate

    task = next(
        row
        for row in generate()["train"]
        if row["task_id"] == load("planner-request.json")["task_id"]
    )
    values = {
        "root": root,
        "task": task,
        "request": load("planner-request.json"),
        "config": load("development-config.json"),
        "checkpoint": load("checkpoint-manifest.json"),
        "work_plan": None,
        "manifest": load("episode-plan-manifest.json"),
        "attempts": [],
        "episode": load("episode-log.json"),
        "evaluation": load("evaluation-result.json"),
    }
    validate(values)
    schema_root = Path("planner_toy/schemas")
    for key, schema_name in (
        ("manifest", "toy_episode_plan_manifest"),
        ("episode", "toy_episode_log"),
        ("evaluation", "toy_evaluation_result"),
    ):
        schema = json.loads((schema_root / f"{schema_name}.schema.json").read_text())
        Draft202012Validator(schema).validate(values[key])
    changed = copy.deepcopy(values)
    changed["episode"]["planner_calls"] = 0
    with pytest.raises(ValueError):
        validate(changed)


def test_inapplicable_plan_records_frozen_executor_failure(tmp_path, replay_dirs) -> None:
    from planner_toy.e2e import run

    root = tmp_path / "executor-failure"
    result = run(root, failure_mode="INAPPLICABLE", reuse_from=replay_dirs[0])
    assert result["success"] is False
    assert result["failure_code"] == "EXECUTOR_PRECONDITION_FAILED"
    manifest = json.loads((root / "episode-plan-manifest.json").read_bytes())
    assert manifest["plan_status"] == "READY"
    attempts = [json.loads(line) for line in (root / "attempt-log.jsonl").read_text().splitlines()]
    assert [attempt["status"] for attempt in attempts] == ["APPLIED", "FAILED"]
    assert attempts[1]["state_before_hash"] == attempts[1]["state_after_hash"]
    episode = json.loads((root / "episode-log.json").read_bytes())
    assert episode["attempts_total"] == 2 and episode["executed_length"] == 1


def test_reuse_rejects_rewrapped_foreign_provenance(tmp_path, replay_dirs) -> None:
    import shutil

    from planner_toy.e2e import run

    source = tmp_path / "foreign"
    shutil.copytree(replay_dirs[0], source)
    config_path = source / "development-config.json"
    config = json.loads(config_path.read_bytes())
    config["training_task_id"] = "bw-99999999"
    config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        run(tmp_path / "reused", failure_mode="NO_END", reuse_from=source)


def test_invalid_plan_is_not_repaired() -> None:
    with pytest.raises(ValueError):
        parse_work_plan([["PICK_UP", "@UNKNOWN"], ["END"]], ["@B0"])


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("manifest", "checkpoint_file_hash", "sha256:" + "1" * 64),
        ("episode", "planner_calls", 2),
        ("episode", "attempts_total", 3),
        ("episode", "plan_positions_consumed", 3),
        ("evaluation", "success", False),
        ("evaluation", "executed_action_count", 3),
        ("evaluation", "goal_hash", "sha256:" + "2" * 64),
        ("evaluation", "replanning_count", 1),
    ],
)
def test_coherent_downstream_mutations_fail(e2e_artifacts, target, field, value) -> None:
    changed = copy.deepcopy(e2e_artifacts)
    changed[target][field] = value
    if target == "manifest":
        changed[target]["manifest_hash"] = artifact_hash(changed[target], "manifest_hash")
    if target == "evaluation":
        changed[target]["evaluation_result_hash"] = artifact_hash(
            changed[target], "evaluation_result_hash"
        )
    with pytest.raises(ValueError):
        validate(changed)


@pytest.mark.parametrize(
    "field", ["task_id", "config_hash", "checkpoint_file_hash", "plan_content_hash"]
)
def test_attempt_binding_mutations_fail(e2e_artifacts, field) -> None:
    changed = copy.deepcopy(e2e_artifacts)
    changed["attempts"][0][field] = "mutated"
    changed["evaluation"]["attempt_log_hash"] = toy_hash("attempt_jsonl", changed["attempts"])
    changed["evaluation"]["evaluation_result_hash"] = artifact_hash(
        changed["evaluation"], "evaluation_result_hash"
    )
    with pytest.raises(ValueError):
        validate(changed)


@pytest.mark.parametrize("operation", ["reverse", "delete"])
def test_attempt_sequence_mutations_fail(e2e_artifacts, operation) -> None:
    changed = copy.deepcopy(e2e_artifacts)
    changed["attempts"] = (
        list(reversed(changed["attempts"])) if operation == "reverse" else changed["attempts"][:-1]
    )
    changed["evaluation"]["attempt_log_hash"] = toy_hash("attempt_jsonl", changed["attempts"])
    changed["evaluation"]["evaluation_result_hash"] = artifact_hash(
        changed["evaluation"], "evaluation_result_hash"
    )
    with pytest.raises(ValueError):
        validate(changed)


def test_changed_plan_action_with_coherent_hashes_fails(e2e_artifacts) -> None:
    changed = copy.deepcopy(e2e_artifacts)
    changed["work_plan"]["steps"][0]["action"] = "STACK"
    changed["attempts"][0]["candidate_action"]["action"] = "STACK"
    rehash(changed)
    with pytest.raises(ValueError):
        validate(changed)
