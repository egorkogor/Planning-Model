from __future__ import annotations

import json
import os
import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.fixed_target_acceptance_v1_1 as v11
import scripts.fixed_target_contract as ft
import scripts.run_fixed_target_acceptance as cli
from tests.toy.test_fixed_cpu_target import valid_contract

WORKFLOW_RUN_ID = 31698229623
JOB_ID = 94440844586
TEST_RUNNER_IMAGE = "planning-model-formal-provenance-summary-regression"
_RUNTIME_ENV = {
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_CBWR": "COMPATIBLE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@pytest.fixture(scope="module")
def runtime11_environment():
    previous = {name: os.environ.get(name) for name in _RUNTIME_ENV}
    os.environ.update(_RUNTIME_ENV)
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _live_formal_contract(sharded) -> dict:
    contract = valid_contract()
    observation = sharded.collect_runtime11_observation(contract, "qualification-only")
    direct_fields = (
        "os",
        "os_version",
        "architecture",
        "cpu_vendor",
        "cpu_family",
        "cpu_model",
        "cpu_stepping",
        "cpu_model_name",
        "python_implementation",
        "python_version",
        "python_build",
        "python_compiler",
        "pip_version",
        "torch_version",
        "torch_build_configuration_sha256",
        "mkl_available",
        "openmp_available",
        "mkldnn_available",
        "ATEN_CPU_CAPABILITY",
        "actual_atten_cpu_capability",
        "MKL_CBWR",
        "torch_num_threads",
        "torch_num_interop_threads",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "mkldnn_enabled",
        "deterministic_algorithms",
        "deterministic_warn_only",
    )
    for field in direct_fields:
        contract[field] = observation[field]
    contract["kernel_policy"] = {"mode": "exact", "value": observation["kernel_version"]}
    contract["microcode_policy"] = {"mode": "exact", "value": observation["microcode"]}
    contract["required_cpu_flags"] = observation["cpu_flags"]
    contract["logical_cpu_count_policy"] = {
        "mode": "exact",
        "value": observation["logical_cpu_count"],
    }
    contract["runner_image"] = TEST_RUNNER_IMAGE
    ft.validate_target_contract(contract)
    return contract


@pytest.fixture(scope="module")
def packaged_formal_attempt(tmp_path_factory, runtime11_environment):
    import scripts.fixed_target_quality_sharded as sharded

    contract = _live_formal_contract(sharded)
    execution_commit = sharded.checkout_commit()
    original_runner_identity = sharded._runner_identity

    def runner_identity(execution_context: str) -> tuple[str, str]:
        if execution_context == v11.FORMAL_EXECUTION_CONTEXT:
            return contract["runner_type"], contract["runner_image"]
        return original_runner_identity(execution_context)

    patcher = pytest.MonkeyPatch()
    patcher.setattr(sharded, "_runner_identity", runner_identity)
    patcher.setattr(
        sharded,
        "require_trusted_implementation_commit",
        lambda commit, **_kwargs: commit,
    )
    patcher.setattr(
        ft,
        "require_trusted_implementation_commit",
        lambda commit, **_kwargs: commit,
    )
    try:
        observation = sharded.collect_runtime11_observation(
            contract, v11.FORMAL_EXECUTION_CONTEXT
        )
        root = tmp_path_factory.mktemp("formal-provenance-summary-runtime11")
        sharded.initialize_attempt(
            root,
            contract,
            execution_commit,
            v11.FORMAL_EXECUTION_CONTEXT,
            "formal-provenance-summary-regression",
            observation,
        )
        for variant in sharded.VARIANTS:
            for seed in sharded.SEEDS:
                sharded.run_unit(root, variant, seed, observation)

        # This regression targets the sealed attempt consumer boundary. Avoid the
        # producer's redundant pre-copy deep replay; derive_formal_attempt_summary
        # below runs the authoritative, unpatched persisted evaluation validation.
        with patch.object(
            sharded,
            "validate_unit_artifacts",
            lambda *_args, **_kwargs: None,
        ):
            assembly = sharded.assemble(root)

        packaged = tmp_path_factory.mktemp("formal-provenance-summary-packaged") / "attempt-1"
        assert (
            cli.command_package_formal_attempt(
                Namespace(
                    attempt_root=root,
                    destination=packaged,
                    attempt_index=1,
                    workflow_run_id=WORKFLOW_RUN_ID,
                    job_id=JOB_ID,
                    workflow_sha=execution_commit,
                )
            )
            == 0
        )
        yield {
            "packaged": packaged,
            "execution_commit": execution_commit,
            "assembly": assembly,
        }
    finally:
        patcher.undo()


def test_sealed_attempt_summary_preserves_validated_orchestration_identity(
    packaged_formal_attempt,
) -> None:
    packaged = packaged_formal_attempt["packaged"]
    summary = v11.derive_formal_attempt_summary(packaged, 1)
    persisted_replay = (packaged / "evaluation/replay-hash.txt").read_text().strip()

    assert summary["result"] == "PASS"
    assert summary["workflow_run_id"] == WORKFLOW_RUN_ID
    assert summary["job_id"] == JOB_ID
    assert summary["workflow_sha"] == packaged_formal_attempt["execution_commit"]
    assert summary["execution_implementation_commit"] == packaged_formal_attempt[
        "execution_commit"
    ]
    assert len(summary["units"]) == 9
    assert summary["claim_identities"]["replay_hash"] == persisted_replay
    assert packaged_formal_attempt["assembly"]["replay_hash"] == persisted_replay


def test_unresealed_orchestration_identity_tamper_is_rejected(
    packaged_formal_attempt,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "attempt-1"
    shutil.copytree(packaged_formal_attempt["packaged"], tampered)
    provenance_path = tampered / "formal-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["workflow_run_id"] += 1
    provenance["job_id"] += 1
    provenance["workflow_sha"] = (
        "f" * 40 if provenance["workflow_sha"] != "f" * 40 else "e" * 40
    )
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    cli._reseal_attempt_manifest(tampered, 1)

    preflight = json.loads((tampered / "preflight.json").read_text())
    execution = json.loads((tampered / "execution-evidence.json").read_text())
    evaluation_config = json.loads(
        (tampered / "evaluation/evaluation-config.json").read_text()
    )
    with pytest.raises(ValueError, match="FIXED_TARGET_FORMAL_PROVENANCE_HASH_MISMATCH"):
        v11._validate_formal_provenance(
            tampered,
            1,
            preflight=preflight,
            execution=execution,
            evaluation_config=evaluation_config,
        )


def test_resealed_workflow_sha_tamper_still_fails_execution_binding(
    packaged_formal_attempt,
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "attempt-1"
    shutil.copytree(packaged_formal_attempt["packaged"], tampered)
    provenance_path = tampered / "formal-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["workflow_sha"] = (
        "f" * 40 if provenance["workflow_sha"] != "f" * 40 else "e" * 40
    )
    provenance["formal_provenance_sha256"] = v11.formal_provenance_sha256(provenance)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    cli._reseal_attempt_manifest(tampered, 1)

    preflight = json.loads((tampered / "preflight.json").read_text())
    execution = json.loads((tampered / "execution-evidence.json").read_text())
    evaluation_config = json.loads(
        (tampered / "evaluation/evaluation-config.json").read_text()
    )
    with pytest.raises(ValueError, match="FIXED_TARGET_WORKFLOW_IMPLEMENTATION_MISMATCH"):
        v11._validate_formal_provenance(
            tampered,
            1,
            preflight=preflight,
            execution=execution,
            evaluation_config=evaluation_config,
        )
