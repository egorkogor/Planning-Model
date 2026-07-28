from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from validation.full_plan_lineage_validator import PLANNER_ARMS, PLANNER_SEEDS, validate_lineage_index
from validation.hashing import hash_json
from validation.statistics_validator import validate_analysis_input
from validation.test_v115_launch_minimum import _planner_lineage_fixture


def _pair(pair_id: str, left: int = 1, right: int = 0) -> dict:
    return {
        "pair_id": pair_id,
        "left": left,
        "right": right,
        "difference": float(left - right),
    }


def test_planner_lineage_requires_exact_task_seed_arm_matrix(tmp_path: Path) -> None:
    index = _planner_lineage_fixture(tmp_path)
    assert not validate_lineage_index(tmp_path, index, expected_stage="PLANNER")
    assert len(index["records"]) == len(PLANNER_SEEDS) * len(PLANNER_ARMS)

    duplicate = deepcopy(index)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    duplicate["index_hash"] = hash_json({k: v for k, v in duplicate.items() if k != "index_hash"})
    errors = validate_lineage_index(tmp_path, duplicate, expected_stage="PLANNER")
    assert any("duplicate Planner result" in error for error in errors)

    missing_seed = deepcopy(index)
    missing_seed["records"] = [row for row in missing_seed["records"] if row["planner_seed"] != 101]
    missing_seed["index_hash"] = hash_json({k: v for k, v in missing_seed.items() if k != "index_hash"})
    errors = validate_lineage_index(tmp_path, missing_seed, expected_stage="PLANNER")
    assert any("seed[101].arms" in error and "missing=" in error for error in errors)


def test_planner_analysis_requires_exact_five_locked_seed_groups() -> None:
    task_ids = ["bw-00000001", "bw-00000002"]
    groups = {
        str(seed): [_pair(task_id) for task_id in task_ids]
        for seed in PLANNER_SEEDS
    }
    obj = {
        "stage": "PLANNER",
        "expected_task_ids": task_ids,
        "expected_task_count": len(task_ids),
        "expected_task_set_sha256": hash_json({"task_ids": sorted(task_ids)}),
        "comparisons": {
            "A3-A1_HORIZON": {
                "analysis_type": "PLANNER_HIERARCHICAL",
                "complete_pair_count": len(task_ids),
                "seed_groups": groups,
            }
        },
    }
    assert not validate_analysis_input(obj)

    bad = deepcopy(obj)
    del bad["comparisons"]["A3-A1_HORIZON"]["seed_groups"]["505"]
    assert any("exactly the five locked final seed groups" in error for error in validate_analysis_input(bad))


def test_stage1a_comparisons_require_identical_snapshot_sets_per_task() -> None:
    task_ids = ["bw-00000001", "bw-00000002"]
    snapshots = {
        task_id: [_pair(f"{task_id}/snap-01"), _pair(f"{task_id}/snap-02")]
        for task_id in task_ids
    }
    obj = {
        "stage": "STAGE1A",
        "expected_task_ids": task_ids,
        "expected_task_count": len(task_ids),
        "expected_task_set_sha256": hash_json({"task_ids": sorted(task_ids)}),
        "comparisons": {
            "I1-I0": {
                "analysis_type": "STAGE1A_CLUSTERED",
                "complete_pair_count": len(task_ids),
                "task_clusters": deepcopy(snapshots),
            },
            "I1-I2": {
                "analysis_type": "STAGE1A_CLUSTERED",
                "complete_pair_count": len(task_ids),
                "task_clusters": deepcopy(snapshots),
            },
        },
    }
    assert not validate_analysis_input(obj)

    bad = deepcopy(obj)
    bad["comparisons"]["I1-I2"]["task_clusters"][task_ids[0]][1]["pair_id"] = f"{task_ids[0]}/snap-X"
    errors = validate_analysis_input(bad)
    assert any("snapshot pair_id sets must exactly match" in error for error in errors)
