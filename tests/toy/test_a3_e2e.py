from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import pytest

from planner_toy.canonical import artifact_hash, canonical_bytes
from planner_toy.dataset import generate
from planner_toy.e2e import file_hash, run, validate_run_directory, verify_replay


def task_for(root: Path) -> dict:
    request = json.loads((root / "planner-request.json").read_text())
    return next(row for row in generate()["train"] if row["task_id"] == request["task_id"])


@pytest.fixture(scope="module")
def independent_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("a3-a4-independent")
    runs = {}
    for variant in ("A3", "A4"):
        one, two = root / f"{variant}-one", root / f"{variant}-two"
        first, second = run(one, variant=variant), run(two, variant=variant)
        runs[variant] = (one, two, first, second)
    return runs


@pytest.mark.parametrize("variant", ["A3", "A4"])
def test_independent_training_replays_are_recursively_identical(independent_runs, variant) -> None:
    one, two, first, second = independent_runs[variant]
    assert first["replay_hash"] == second["replay_hash"]
    assert first == second
    one_names = sorted(path.relative_to(one) for path in one.rglob("*") if path.is_file())
    two_names = sorted(path.relative_to(two) for path in two.rglob("*") if path.is_file())
    assert one_names == two_names
    assert all((one / name).read_bytes() == (two / name).read_bytes() for name in one_names)


@pytest.mark.parametrize("variant", ["A3", "A4"])
def test_variant_e2e_frozen_execution_and_trace(independent_runs, variant) -> None:
    root, _, result, _ = independent_runs[variant]
    assert result["success"] is True and result["planner_call_count"] == 1
    trace = json.loads((root / "semantic-trace.json").read_text())
    plan = json.loads((root / "results/development/plans/work-plan.json").read_text())
    assert len(trace["steps"]) == len(plan["steps"]) > 2
    assert trace["feedback_applied"] is (variant == "A3")


def test_reuse_from_is_separate_from_independent_replay(tmp_path, independent_runs) -> None:
    source = independent_runs["A3"][0]
    reused = tmp_path / "reused"
    assert run(reused, variant="A3", reuse_from=source)["success"] is True
    verify_replay(reused, task_for(reused))


@pytest.mark.parametrize("variant", ["A3", "A4"])
def test_no_end_failure_validates_without_plan_or_trace(
    tmp_path, independent_runs, variant
) -> None:
    root = tmp_path / variant
    result = run(
        root, variant=variant, failure_mode="NO_END", reuse_from=independent_runs[variant][0]
    )
    assert result["failure_code"] == "PLAN_NO_END"
    assert not (root / "semantic-trace.json").exists()
    assert not (root / "semantic-latents.f32").exists()
    assert not (root / "results/development/plans/work-plan.json").exists()
    verify_replay(root, task_for(root))


@pytest.mark.parametrize("code", ["LATENT_NONFINITE", "LATENT_DIMENSION", "LATENT_ZERO_NORM"])
def test_typed_latent_generation_failures_are_preserved(tmp_path, independent_runs, code) -> None:
    root = tmp_path / code
    result = run(root, variant="A3", failure_mode=code, reuse_from=independent_runs["A3"][0])
    assert result["failure_code"] == code
    manifest = json.loads((root / "episode-plan-manifest.json").read_text())
    episode = json.loads((root / "episode-log.json").read_text())
    evaluation = json.loads((root / "evaluation-result.json").read_text())
    assert (
        manifest["failure_code"] == episode["terminal_error"] == evaluation["failure_code"] == code
    )
    verify_replay(root, task_for(root))


def test_fully_coherent_latent_mutation_rejected_by_checkpoint_replay(
    tmp_path, independent_runs
) -> None:
    changed = tmp_path / "changed"
    shutil.copytree(independent_runs["A3"][0], changed)
    latent = changed / "semantic-latents.f32"
    payload = bytearray(latent.read_bytes())
    values = list(struct.unpack("<384f", payload[:1536]))
    values[0] = -values[0]
    norm = math.sqrt(sum(value * value for value in values))
    values = [value / norm for value in values]
    payload[:1536] = struct.pack("<384f", *values)
    latent.write_bytes(payload)
    trace_path = changed / "semantic-trace.json"
    trace = json.loads(trace_path.read_text())
    previous = None
    for index, row in enumerate(trace["steps"]):
        chunk = bytes(payload[index * 1536 : (index + 1) * 1536])
        row["z_sha256"] = "sha256:" + hashlib.sha256(chunk).hexdigest()
        row["latent_norm"] = math.sqrt(sum(v * v for v in struct.unpack("<384f", chunk)))
        row["previous_z_sha256"] = previous
        previous = row["z_sha256"]
    trace["latent_file_sha256"] = file_hash(latent)
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    with pytest.raises(ValueError, match="CHECKPOINT_LATENT_REPLAY_MISMATCH"):
        validate_run_directory(changed, task_for(changed))


@pytest.mark.parametrize(
    "field,value", [("variant", "A2"), ("success", False), ("failure_code", "PLAN_NO_END")]
)
def test_run_result_coherent_mutations_fail_semantics(
    tmp_path, independent_runs, field, value
) -> None:
    changed = tmp_path / field
    shutil.copytree(independent_runs["A3"][0], changed)
    path = changed / "run-result.json"
    result = json.loads(path.read_text())
    result[field] = value
    result["run_result_hash"] = artifact_hash(result, "run_result_hash")
    path.write_bytes(canonical_bytes(result) + b"\n")
    with pytest.raises(ValueError, match="run result semantics mismatch"):
        verify_replay(changed, task_for(changed))
