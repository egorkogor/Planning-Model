from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from validation.hashing import hash_json
from validation.statistics_validator import validate_analysis_input, validate_sample_size_report
from validation.code_fingerprint import analysis_code_digest


def _pairs(ids: list[str], value: int = 1) -> list[dict]:
    return [{"pair_id": task_id, "left": 1 if value >= 0 else 0, "right": 0 if value >= 0 else 1, "difference": float(value)} for task_id in ids]


def test_execution_artifacts_are_registered() -> None:
    registry = yaml.safe_load(Path("docs/operator/report_registry_v1.yaml").read_text())
    rules = {row["path_pattern"]: row for row in registry["rules"]}
    assert rules["results/**/plans/*.json"]["schema"] == "docs/schemas/work_plan.schema.json"
    assert rules["results/**/episodes/*.json"]["schema"] == "docs/schemas/episode_log.schema.json"
    assert rules["results/**/attempts/*.jsonl"]["schema_per_line"] == "docs/schemas/attempt_log.schema.json"
    assert rules["sealed/**/selected-task-manifest.json"]["schema"] == "docs/schemas/selected_task_manifest.schema.json"


def test_analysis_input_requires_one_exact_task_set_for_every_comparison() -> None:
    ids = ["bw-00000001", "bw-00000002"]
    obj = {
        "expected_task_ids": ids,
        "expected_task_count": 2,
        "expected_task_set_sha256": hash_json({"task_ids": sorted(ids)}),
        "comparisons": {
            "E1-E0": {"analysis_type": "TASK_PAIRED", "complete_pair_count": 2, "pairs": _pairs(ids)},
            "E1-E2": {"analysis_type": "TASK_PAIRED", "complete_pair_count": 2, "pairs": _pairs(ids)},
        },
    }
    assert not validate_analysis_input(obj)
    bad = deepcopy(obj)
    bad["comparisons"]["E1-E2"]["pairs"][1]["pair_id"] = "bw-99999999"
    assert any("task set differs" in error for error in validate_analysis_input(bad))


def test_sample_size_component_cannot_use_an_unrelated_comparison() -> None:
    ids = [f"bw-{i:08d}" for i in range(1, 21)]
    mapping = yaml.safe_load(Path("docs/statistics/statistics_contract_v1.yaml").read_text())["sample_size"]["component_comparison_by_stage"]["STAGE1B"]
    requirements = {}
    for name, comparison in mapping.items():
        requirements[name] = {"comparison": comparison, "analysis_type": "TASK_PAIRED", "complete_pair_count": 20, "pairs": _pairs(ids, 0)}
    sample_input = {"stage": "STAGE1B", "requirements": requirements}
    obj = {
        "stage": "STAGE1B",
        "method": "estimator_matched_paired_binary_simulation_v5",
        "analysis_code_sha256": analysis_code_digest(),
        "inputs": {"pilot_sd_by_requirement": {name: 0.1 for name in requirements}},
    }
    # Stop before expensive recomputation: the semantic comparison error must be detected first.
    bad = deepcopy(sample_input)
    bad["requirements"]["current_vs_shuffled_power"]["comparison"] = "UNRELATED_LOW_VARIANCE_COMPARISON"
    errors = validate_sample_size_report(obj, bad)
    assert any("current_vs_shuffled_power: comparison" in error for error in errors)


def test_replay_metric_has_one_canonical_name() -> None:
    replay = yaml.safe_load(Path("docs/controls/p_replay_contract_v1.yaml").read_text())
    stats = yaml.safe_load(Path("docs/statistics/statistics_contract_v1.yaml").read_text())
    assert replay["contexts"]["PLANNER_CONFIRMATORY_A3"]["metric_name"] == "P_REPLAY_GOAL_SUCCESS"
    comparisons = {
        gate["comparison"]
        for gate in stats["stage_gates"]["PLANNER"]["stage1b_eligibility_gates"]
    }
    assert "P_REPLAY_GOAL_SUCCESS" in comparisons


def _planner_lineage_fixture(tmp_path: Path) -> dict:
    from validation.fixtures import H1
    from validation.test_v114_full_plan_lineage import (
        _planner_confirmatory_bundle_pair,
        _selection_fields,
        _write_bundle,
    )
    source, replay = _planner_confirmatory_bundle_pair()
    rows = [
        _write_bundle(tmp_path, "PLANNER_A3_RAW", source),
        _write_bundle(tmp_path, "P_FULL_PLAN_REPLAY_RAW", replay),
    ]
    index = {
        "schema_version": "work-planner-lineage/1.0",
        "run_id": "run-1",
        "stage": "PLANNER",
        "compute_profile": None,
        "compute_profile_sha256": None,
        "records": rows,
        "index_hash": H1,
        **_selection_fields(tmp_path, "PLANNER", ["bw-00000001"]),
    }
    index["index_hash"] = hash_json({k: v for k, v in index.items() if k != "index_hash"})
    return index


def test_lineage_cannot_drop_a_selected_task_or_replay_another_run(tmp_path: Path) -> None:
    from validation.full_plan_lineage_validator import validate_lineage_index

    index = _planner_lineage_fixture(tmp_path)
    assert not validate_lineage_index(tmp_path, index, expected_stage="PLANNER")

    # Expand the signed selection to two tasks while keeping outcomes for one task only.
    selected_path = tmp_path / index["selected_task_manifest"]
    selected = json.loads(selected_path.read_text())
    selected["task_ids"].append("bw-00000002")
    selected["task_count"] = 2
    selected["manifest_hash"] = hash_json({k: v for k, v in selected.items() if k != "manifest_hash"})
    selected_path.write_text(json.dumps(selected))
    selected_sha = "sha256:" + __import__("hashlib").sha256(selected_path.read_bytes()).hexdigest()
    index["selected_task_manifest_sha256"] = selected_sha
    index["expected_task_count"] = 2
    sealer_path = tmp_path / index["sealer_manifest"]
    sealer = json.loads(sealer_path.read_text())
    sealer["task_count"] = 2
    sealer["selected_task_manifest_sha256"] = selected_sha
    sealer_path.write_text(json.dumps(sealer))
    index["sealer_manifest_sha256"] = "sha256:" + __import__("hashlib").sha256(sealer_path.read_bytes()).hexdigest()
    index["index_hash"] = hash_json({k: v for k, v in index.items() if k != "index_hash"})
    assert any("records.task_set" in error for error in validate_lineage_index(tmp_path, index, expected_stage="PLANNER"))

    replay = _planner_lineage_fixture(tmp_path / "replay")
    replay["run_id"] = "run-new"
    replay["index_hash"] = hash_json({k: v for k, v in replay.items() if k != "index_hash"})
    errors = validate_lineage_index(tmp_path / "replay", replay, expected_stage="PLANNER")
    assert any("run_id/stage differs" in error for error in errors)


def test_analysis_input_is_bound_to_the_sealer_selected_manifest() -> None:
    ids = ["bw-00000001", "bw-00000002"]
    selected = {
        "schema_version": "work-planner-selected-tasks/1.0",
        "run_id": "run-1",
        "stage": "STAGE1B",
        "task_ids": ids,
        "task_count": 2,
        "created_at": "2026-07-24T10:00:00Z",
        "manifest_hash": "",
    }
    selected["manifest_hash"] = hash_json({k: v for k, v in selected.items() if k != "manifest_hash"})
    obj = {
        "run_id": "run-1",
        "stage": "STAGE1B",
        "expected_task_ids": ids,
        "expected_task_count": 2,
        "expected_task_set_sha256": hash_json({"task_ids": sorted(ids)}),
        "comparisons": {
            "E1-E0": {"analysis_type": "TASK_PAIRED", "complete_pair_count": 2, "pairs": _pairs(ids)},
        },
    }
    assert not validate_analysis_input(obj, selected)
    bad = deepcopy(selected)
    bad["task_ids"] = ["bw-00000001", "bw-99999999"]
    bad["manifest_hash"] = hash_json({k: v for k, v in bad.items() if k != "manifest_hash"})
    assert any("differ from selected task manifest" in e for e in validate_analysis_input(obj, bad))


def test_evaluator_task_count_is_recomputed_from_lineage(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import validation.confirmatory_lineage_validator as clv

    monkeypatch.setattr(clv, "ROOT", tmp_path)
    monkeypatch.setattr(clv, "schema_errors", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(clv, "validate_dispatch", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(clv, "validate_lineage_index", lambda *_args, **_kwargs: [])

    dispatch_path = tmp_path / "dispatch/evaluator-planner.json"
    pointer_path = tmp_path / "freezes/planner-confirmatory.approved.json"
    candidate_path = tmp_path / "freezes/planner-confirmatory.candidate.json"
    result_dir = tmp_path / "results/planner-confirmatory"
    for path in (dispatch_path, pointer_path, candidate_path, result_dir / "lineage-index.json", tmp_path / "reports/resource-plan.json"):
        path.parent.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "dispatch_id": "DSP-0001",
        "dispatch_hash": "sha256:" + "d" * 64,
        "approved_freeze_pointer_hash": "sha256:" + "p" * 64,
        "freeze_hash": "sha256:" + "f" * 64,
        "created_at": "2026-07-24T11:00:00Z",
    }
    pointer = {"candidate_path": "freezes/planner-confirmatory.candidate.json"}
    candidate = {"git_commit": "a" * 40, "sealed_dataset_commitment": {"task_count": 2}}
    lineage = {
        "run_id": "run-1",
        "stage": "PLANNER",
        "records": [{"task_id": "bw-00000001", "arm": "PLANNER_A3_RAW"}],
    }
    dispatch_path.write_text(json.dumps(dispatch))
    pointer_path.write_text(json.dumps(pointer))
    candidate_path.write_text(json.dumps(candidate))
    (result_dir / "lineage-index.json").write_text(json.dumps(lineage))
    env = "eval-env"
    (tmp_path / "reports/resource-plan.json").write_text(json.dumps({"evaluator": {"environment_identity": env}}))

    report = {
        "run_id": "run-1",
        "input_hashes": {
            "dispatch/evaluator-planner.json": clv.digest(dispatch_path),
            "freezes/planner-confirmatory.approved.json": clv.digest(pointer_path),
        },
    }
    obj = {
        "run_id": "run-1",
        "stage": "PLANNER",
        "dispatch_id": dispatch["dispatch_id"],
        "dispatch_hash": dispatch["dispatch_hash"],
        "approved_freeze_pointer_hash": dispatch["approved_freeze_pointer_hash"],
        "freeze_hash": dispatch["freeze_hash"],
        "git_commit": candidate["git_commit"],
        "task_count": 2,
        "completed_at": "2026-07-24T12:00:00Z",
        "evaluator_environment_sha256": "sha256:" + hashlib.sha256(env.encode()).hexdigest(),
    }
    obj["raw_artifacts"] = clv.result_artifact_map(result_dir, root=tmp_path)
    errors = clv.validate_evaluator_manifest(obj, report)
    assert any("task_count differs from exact lineage task set" in error for error in errors)
