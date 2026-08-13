from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.fixed_target_contract as ft
from planner_toy.dataset import generate
from scripts.fixed_target_quality_sharded import (
    QUALIFICATION_RUNTIME_FIELDS,
    SEEDS,
    VARIANTS,
    assemble,
    checkout_commit,
    collect_runtime11_observation,
    initialize_attempt,
    run_unit,
)
from tests.toy.test_fixed_cpu_target import valid_contract

EXPECTED_DATASET_HASH = (
    "sha256:60e4ce06d6cfc90dc467fb4e82b2eb71cf2d92d37471eee3aeda64f864c541df"
)
EXPECTED_RAW_TRAIN_IDS = [
    "bw-00000001",
    "bw-00000003",
    "bw-00000002",
]
EXPECTED_CANONICAL_TRAIN_IDS = [
    "bw-00000001",
    "bw-00000002",
    "bw-00000003",
]
EXPECTED_EVAL_IDS = ["bw-00000004", "bw-00000005"]


@pytest.fixture(scope="module", autouse=True)
def runtime11_environment():
    previous = {
        name: os.environ.get(name)
        for name in (
            "ATEN_CPU_CAPABILITY",
            "MKL_CBWR",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    os.environ.update(
        {
            "ATEN_CPU_CAPABILITY": "default",
            "MKL_CBWR": "COMPATIBLE",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_raw_dataset_lineage_is_not_canonical_execution_order() -> None:
    dataset = generate(17)
    raw_train_ids = [row["task_id"] for row in dataset["train"]]
    canonical_train_ids = [
        row["task_id"]
        for row in sorted(dataset["train"], key=lambda row: row["task_id"])
    ]
    raw_eval_ids = [row["task_id"] for row in dataset["validation"]]
    canonical_eval_ids = [
        row["task_id"]
        for row in sorted(dataset["validation"], key=lambda row: row["task_id"])
    ]

    assert raw_train_ids == EXPECTED_RAW_TRAIN_IDS
    assert canonical_train_ids == EXPECTED_CANONICAL_TRAIN_IDS
    assert raw_train_ids != canonical_train_ids
    assert raw_eval_ids == EXPECTED_EVAL_IDS
    assert canonical_eval_ids == EXPECTED_EVAL_IDS
    assert dataset["dataset_hash"] == EXPECTED_DATASET_HASH
    assert dataset["dataset_hash"] == ft.HISTORICAL_QUALITY_DATASET_HASH

    expected = ft._expected_canonical_dataset_manifest()
    assert expected == {
        "dataset_hash": EXPECTED_DATASET_HASH,
        "train_task_ids": EXPECTED_CANONICAL_TRAIN_IDS,
        "eval_task_ids": EXPECTED_EVAL_IDS,
    }
    assert tuple(expected["train_task_ids"]) == ft.HISTORICAL_ORDERED_TRAIN_TASK_IDS
    assert tuple(expected["eval_task_ids"]) == ft.HISTORICAL_ORDERED_EVAL_TASK_IDS

    config = {
        "dataset_manifest_hash": expected["dataset_hash"],
        "train_task_ids": copy.deepcopy(expected["train_task_ids"]),
        "eval_task_ids": copy.deepcopy(expected["eval_task_ids"]),
    }
    assert ft._validate_canonical_dataset_binding(expected, config) == expected

    raw_manifest = copy.deepcopy(expected)
    raw_manifest["train_task_ids"] = raw_train_ids
    raw_config = copy.deepcopy(config)
    raw_config["train_task_ids"] = raw_train_ids
    with pytest.raises(ValueError, match="FIXED_TARGET_DATASET_MANIFEST_MISMATCH"):
        ft._validate_canonical_dataset_binding(raw_manifest, raw_config)


@pytest.fixture(scope="module")
def assembled_runtime11_evaluation(tmp_path_factory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("dataset-order-runtime11")
    contract = valid_contract()
    observation = collect_runtime11_observation(contract, "qualification-only")
    for field in QUALIFICATION_RUNTIME_FIELDS:
        contract[field] = observation[field]
    observation = collect_runtime11_observation(contract, "qualification-only")
    initialize_attempt(
        root,
        contract,
        checkout_commit(),
        "qualification-only",
        "dataset-order-integrity-regression",
        observation,
    )
    for variant in VARIANTS:
        for seed in SEEDS:
            run_unit(root, variant, seed, observation)

    # This regression targets packaged-evaluation compatibility. Avoid the
    # producer's redundant pre-copy recursive replay here; the authoritative
    # validator below is deliberately unpatched and revalidates all nine runs.
    with patch(
        "scripts.fixed_target_quality_sharded.validate_unit_artifacts",
        lambda *_args, **_kwargs: None,
    ):
        result = assemble(root)
    return root, result


def test_assembled_runtime11_tree_passes_authoritative_integrity_validation(
    assembled_runtime11_evaluation: tuple[Path, dict],
) -> None:
    root, result = assembled_runtime11_evaluation
    evaluation_root = root / "evaluation"

    replay = ft._validate_evaluation_integrity(evaluation_root)
    persisted_replay = (evaluation_root / "replay-hash.txt").read_text().strip()

    assert replay == persisted_replay
    assert replay == result["replay_hash"]

    dataset_manifest = json.loads((evaluation_root / "dataset-manifest.json").read_text())
    evaluation_config = json.loads((evaluation_root / "evaluation-config.json").read_text())
    assert ft._validate_canonical_dataset_binding(dataset_manifest, evaluation_config) == (
        dataset_manifest
    )
