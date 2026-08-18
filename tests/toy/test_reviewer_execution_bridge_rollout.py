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


def test_workflow_hands_off_unique_traversable_execution_root_to_seal() -> None:
    text = WORKFLOW.read_text(encoding='utf-8')
    assert 'mktemp -d /tmp/planning-model-reviewer-bridge-execution.XXXXXXXX' in text
    assert '${RUNNER_TEMP%/}/planning-model-reviewer-bridge-execution' not in text
    assert "printf 'execution_root=%s\\n' \"$execution_root\" >> \"$GITHUB_OUTPUT\"" in text
    assert 'EXECUTION_ROOT: ${{ steps.execute.outputs.execution_root }}' in text
    assert 'REVIEWER_BRIDGE_EXECUTION_ROOT_INVALID' in text
    assert 'REVIEWER_BRIDGE_EXECUTION_ROOT_OWNERSHIP_MISMATCH' in text
    assert 'REVIEWER_BRIDGE_EXECUTION_WORKSPACE_OWNERSHIP_MISMATCH' in text
    seal = text[text.index('- id: seal'):text.index('- id: publish')]
    assert 'rm -rf -- "$execution_root"' in seal
    assert 'chmod 711 "$execution_root"' not in seal


def test_real_os_principal_cannot_traverse_runner_temp_but_can_use_dedicated_root(
    tmp_path: Path,
) -> None:
    sudo = shutil.which('sudo')
    useradd = shutil.which('useradd')
    userdel = shutil.which('userdel')
    assert sudo is not None and useradd is not None and userdel is not None
    subprocess.run([sudo, '-n', 'true'], check=True)

    user = f'pmbridge{os.getpid()}'
    subprocess.run(
        [
            sudo,
            '-n',
            useradd,
            '--system',
            '--user-group',
            '--home-dir',
            '/nonexistent',
            '--no-create-home',
            '--shell',
            '/usr/sbin/nologin',
            user,
        ],
        check=True,
    )
    try:
        uid = int(subprocess.check_output(['id', '-u', user], text=True).strip())
        gid = int(subprocess.check_output(['id', '-g', user], text=True).strip())
        python_executable = os.path.realpath(sys.executable)
        subprocess.run([sudo, '-n', '-u', user, '--', 'test', '-x', python_executable], check=True)

        blocked_parent = tmp_path / 'runner-temp-parent'
        blocked_workspace = blocked_parent / 'execution'
        blocked_parent.mkdir(mode=0o700)
        blocked_workspace.mkdir(mode=0o700)
        blocked_driver = blocked_workspace / 'driver.py'
        blocked_driver.write_text("print('blocked-path-ran')\n", encoding='utf-8')
        subprocess.run(
            [sudo, '-n', 'chown', f'{uid}:{gid}', str(blocked_workspace), str(blocked_driver)],
            check=True,
        )
        blocked = subprocess.run(
            [sudo, '-n', '-u', user, '--', python_executable, str(blocked_driver)],
            capture_output=True,
            text=True,
        )
        assert blocked.returncode != 0
        assert 'blocked-path-ran' not in blocked.stdout

        dedicated_root = Path(
            subprocess.check_output(
                ['mktemp', '-d', '/tmp/planning-model-reviewer-bridge-execution.XXXXXXXX'],
                text=True,
            ).strip()
        )
        try:
            dedicated_root.chmod(0o711)
            workspace = dedicated_root / 'run'
            workspace.mkdir(mode=0o700)
            driver = workspace / 'driver.py'
            driver.write_text("print('dedicated-path-ran')\n", encoding='utf-8')
            subprocess.run(
                [sudo, '-n', 'chown', f'{uid}:{gid}', str(workspace), str(driver)],
                check=True,
            )
            allowed = subprocess.run(
                [sudo, '-n', '-u', user, '--', python_executable, str(driver)],
                check=True,
                capture_output=True,
                text=True,
            )
            assert allowed.stdout.strip() == 'dedicated-path-ran'

            control = tmp_path / 'control'
            control.mkdir(mode=0o700)
            secret = control / 'secret'
            secret.write_text('control-only\n', encoding='utf-8')
            unreadable = subprocess.run(
                [sudo, '-n', '-u', user, '--', 'cat', str(secret)],
                capture_output=True,
                text=True,
            )
            assert unreadable.returncode != 0
        finally:
            shutil.rmtree(dedicated_root, ignore_errors=True)
    finally:
        subprocess.run([sudo, '-n', 'pkill', '-KILL', '-u', user], check=False)
        subprocess.run([sudo, '-n', userdel, user], check=True)


def test_workflow_binds_launcher_to_real_sys_executable_not_path_wrapper(tmp_path: Path) -> None:
    text = WORKFLOW.read_text(encoding='utf-8')
    launcher = (
        "python_executable=\"$(python -c 'import os,sys; "
        "print(os.path.realpath(sys.executable))')\""
    )
    assert launcher in text
    assert 'python_executable="$(command -v python)"' not in text

    wrapper = tmp_path / 'python'
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable!s} \"$@\"\n",
        encoding='utf-8',
    )
    wrapper.chmod(0o755)
    assert os.path.realpath(wrapper) != os.path.realpath(sys.executable)
    observed = subprocess.check_output(
        [str(wrapper), '-c', 'import os,sys; print(os.path.realpath(sys.executable))'],
        text=True,
    ).strip()
    assert observed == os.path.realpath(sys.executable)


def test_cleanup_removes_failed_bootstrap_marker_candidate() -> None:
    text = WORKFLOW.read_text(encoding='utf-8')
    cleanup = text[text.index('Cleanup sealed control state after successful publication'):]
    assert '"$root/principal-marker-candidate.json"' in cleanup
