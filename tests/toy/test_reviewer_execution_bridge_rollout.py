from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKFLOW = Path('.github/workflows/reviewer-execution-bridge.yml')


def test_workflow_status_bootstraps_versioned_principal_before_preflight_or_science() -> None:
    text = WORKFLOW.read_text(encoding='utf-8')
    assert 'task: ${{ steps.parse.outputs.task }}' in text
    assert 'PRINCIPAL_CONTRACT_VERSION: reviewer-bridge-execution-principal/1.0' in text
    assert 'PRINCIPAL_MARKER: /etc/planning-model-reviewer-bridge-exec-v1.json' in text
    assert 'test "$task" = status-v1' in text
    assert '/usr/sbin/useradd --system --user-group --home-dir /nonexistent' in text
    assert 'REVIEWER_BRIDGE_EXECUTION_PRINCIPAL_STATUS_BOOTSTRAP_REQUIRED' in text
    assert 'principal-marker-candidate.json' in text
    assert 'STATUS_BOOTSTRAP_EVIDENCE_NOT_ACCEPTABLE' in text
    assert 'Finalize versioned execution-principal bootstrap' in text


def test_workflow_materializes_fixed_target_dispatch_before_requested_python() -> None:
    text = WORKFLOW.read_text(encoding='utf-8')
    assert 'target_contract="$repo/configs/fixed-cpu-target-1.0.json"' in text
    assert 'REVIEWER_BRIDGE_FIXED_TARGET_CONTRACT_VERSION_MISMATCH' in text
    assert text.count('ATEN_CPU_CAPABILITY="$aten_cpu_capability"') == 2
    assert text.count('MKL_CBWR="$mkl_cbwr"') == 2
    launch = text.index('"$python_executable" "$driver" execute')
    assert text.rfind('ATEN_CPU_CAPABILITY="$aten_cpu_capability"', 0, launch) >= 0
    assert text.rfind('MKL_CBWR="$mkl_cbwr"', 0, launch) >= 0
    isolated = text[text.rfind('/usr/bin/env -i', 0, launch):launch]
    assert 'GITHUB_TOKEN' not in isolated
    assert 'GITHUB_ENV' not in isolated
    assert 'GITHUB_PATH' not in isolated


def test_real_env_i_child_preserves_target_dispatch_and_drops_control(tmp_path: Path) -> None:
    env_bin = shutil.which('env')
    assert env_bin is not None
    probe = tmp_path / 'process_start_probe.py'
    probe.write_text(
        "import json,os\n"
        "keys=('ATEN_CPU_CAPABILITY','MKL_CBWR','OMP_NUM_THREADS','MKL_NUM_THREADS',"
        "'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')\n"
        "forbidden=[k for k in os.environ if k.startswith('GITHUB_') or "
        "k.startswith('GIT_CONFIG_') or k.startswith('BRIDGE_')]\n"
        "print(json.dumps({'target':{k:os.environ.get(k) for k in keys},"
        "'forbidden':forbidden},sort_keys=True))\n",
        encoding='utf-8',
    )
    parent = dict(os.environ)
    parent.update(
        {
            'GITHUB_TOKEN': 'SECRET',
            'GITHUB_ENV': '/tmp/github-env',
            'GITHUB_PATH': '/tmp/github-path',
            'BRIDGE_DRIVER': '/tmp/driver',
            'GIT_CONFIG_COUNT': '1',
        }
    )
    completed = subprocess.run(
        [
            env_bin,
            '-i',
            f'HOME={tmp_path}',
            f"PATH={os.environ.get('PATH', '')}",
            'ATEN_CPU_CAPABILITY=avx2',
            'MKL_CBWR=COMPATIBLE',
            'OMP_NUM_THREADS=1',
            'MKL_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1',
            'NUMEXPR_NUM_THREADS=1',
            sys.executable,
            str(probe),
        ],
        env=parent,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed['target'] == {
        'ATEN_CPU_CAPABILITY': 'avx2',
        'MKL_CBWR': 'COMPATIBLE',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
    }
    assert observed['forbidden'] == []


def test_workflow_seals_execution_principal_and_python_runtime_provenance() -> None:
    text = WORKFLOW.read_text(encoding='utf-8')
    for field in (
        'execution_principal',
        'execution_uid',
        'execution_gid',
        'execution_home',
        'python_executable',
        'python_implementation',
        'python_version',
        'process_start_env',
    ):
        assert f'"{field}"' in text
    for error in (
        'REVIEWER_BRIDGE_EXECUTION_UID_MISMATCH',
        'REVIEWER_BRIDGE_EXECUTION_GID_MISMATCH',
        'REVIEWER_BRIDGE_EXECUTION_HOME_MISMATCH',
        'REVIEWER_BRIDGE_EXECUTION_PYTHON_MISMATCH',
        'REVIEWER_BRIDGE_EXECUTION_PYTHON_VERSION_MISMATCH',
        'REVIEWER_BRIDGE_EXECUTION_PROCESS_ENV_MISMATCH',
    ):
        assert error in text
    provenance_patch = (
        'execution.update(context); '
        'bridge._json_dump(source/"github-execution.json",execution)'
    )
    assert provenance_patch in text
    assert 'old_result["manifest_sha256"]' in text
    assert 'old_result["archive_sha256"]' in text
