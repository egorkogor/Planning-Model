from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

import pytest

import scripts.run_reviewer_execution_bridge as bridge

WORKFLOW = Path(".github/workflows/reviewer-execution-bridge.yml")
SHA = "a" * 40
WORKFLOW_SHA = "b" * 40


def _event(*, task: str = "status-v1", request: str = "req-runtime-0001") -> dict:
    return {
        "repository": {"full_name": "egorkogor/Planning-Model"},
        "issue": {"number": 36},
        "sender": {"login": "egorkogor"},
        "comment": {
            "id": 5334849824,
            "body": f"/reviewer-bridge/v1 task={task} request={request} sha={SHA}",
            "user": {"login": "egorkogor"},
        },
    }


def _fallback_diagnostic_fragment() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index('              diagnostics=dest/"untrusted-execution-diagnostics"')
    end = text.index(
        '              bridge._json_dump(dest/"task-failure.json",failure)', start
    )
    return textwrap.dedent(text[start:end])


def _run_fallback_diagnostic_fragment(
    *,
    source: Path,
    dest: Path,
    stdout_path: Path,
    stderr_path: Path,
    context_path: Path,
    secondary_error: str,
    bridge_rc: int = 0,
) -> dict:
    namespace = {
        "json": json,
        "Path": Path,
        "bridge": bridge,
        "source": source,
        "dest": dest,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "context_path": str(context_path),
        "error": secondary_error,
        "bridge_rc": bridge_rc,
    }
    exec(_fallback_diagnostic_fragment(), namespace)
    return namespace


def _diagnostic_control_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    stdout = tmp_path / "execution-driver.stdout"
    stderr = tmp_path / "execution-driver.stderr"
    context = tmp_path / "execution-context.json"
    stdout.write_bytes(b'{"terminal_status":"FAILED"}\n')
    stderr.write_bytes(b"driver stderr\n")
    context.write_bytes(b'{"execution_principal":"planning-model-bridge-exec"}\n')
    return stdout, stderr, context


def test_workflow_preserves_venv_launcher_and_verifies_runtime() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    launcher_line = (
        'python_executable="$(python -c '
        "'import os,sys; print(os.path.abspath(sys.executable))')\""
    )
    assert launcher_line in text
    assert "print(os.path.realpath(sys.executable))" not in text
    assert '"$python_executable" "$driver" execute' in text
    assert "import torch" in text
    assert "import scripts.run_fixed_target_acceptance" in text
    assert 'torch.__version__ != contract.get("torch_version")' in text
    assert 'context.get("torch_version") != contract.get("torch_version")' in text
    assert 'context.get("runtime_imports_verified") is not True' in text
    assert '"python_resolved_executable":os.path.realpath(invoked)' in text
    assert 'context.get("python_resolved_executable") != os.path.realpath(invoked_python)' in text
    assert "Path(str(context.get(\"python_executable\"))).resolve()" not in text


@pytest.mark.skipif(os.name != "posix", reason="canonical bridge runner is Linux")
def test_venv_symlink_is_invoked_via_venv_path_not_real_target(tmp_path: Path) -> None:
    env_dir = tmp_path / "runtime-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
    launcher = env_dir / "bin" / "python"
    completed = subprocess.run(
        [
            str(launcher),
            "-c",
            "import os,sys; print(os.path.abspath(sys.executable)); "
            "print(os.path.realpath(sys.executable))",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    invoked, real_target = completed.stdout.splitlines()
    assert invoked == str(launcher)
    assert real_target == os.path.realpath(launcher)
    assert invoked != real_target


def test_driver_children_keep_the_invoked_python_executable(tmp_path: Path) -> None:
    preflight_request = bridge.parse_event(
        _event(task="preflight-v1", request="req-runtime-0002")
    )
    preflight = bridge.task_plan(preflight_request, tmp_path)
    assert preflight[0][1][0] == sys.executable

    science_request = bridge.parse_event(
        _event(task="a2-sufficient-budget-task-order-v1", request="req-runtime-0003")
    )
    science = bridge.task_plan(science_request, tmp_path)
    assert science
    assert all(argv[0] == sys.executable for _, argv in science)


def test_early_infrastructure_failure_is_snapshotted_but_boundary_stays_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bridge"
    work = root / "100-1"
    work.mkdir(parents=True)
    event = tmp_path / "event.json"
    event.write_text(json.dumps(_event()), encoding="utf-8")

    def fail_before_source_identity(*args, **kwargs):
        raise RuntimeError("synthetic first production failure")

    monkeypatch.setattr(bridge, "validate_trusted_checkout", fail_before_source_identity)
    result = bridge.execute_request(
        event_path=event,
        repo_root=tmp_path,
        bridge_root=root,
        workspace=work,
        workflow_sha=WORKFLOW_SHA,
        run_id=100,
        run_attempt=1,
        job_name="reviewer-bridge-canonical",
        runner_name="canonical",
    )
    source = work / "evidence"
    originals = {
        name: (source / name).read_bytes()
        for name in (
            "task-failure.json",
            "result.json",
            "status.json",
            "bridge-source.json",
        )
    }
    assert result["failed_task"] == "bridge-infrastructure"
    assert json.loads(originals["task-failure.json"])["stderr"] == (
        "synthetic first production failure"
    )
    assert json.loads(originals["bridge-source.json"])["available"] is False
    trusted = source / "trusted-commit-validator.jsonl"
    trusted.write_bytes(b'{"trusted":false,"synthetic":true}\n')
    originals[trusted.name] = trusted.read_bytes()

    dest = tmp_path / "sealed"
    dest.mkdir()
    stdout_path, stderr_path, context_path = _diagnostic_control_files(tmp_path)
    secondary = "ValueError: REVIEWER_BRIDGE_EVIDENCE_SOURCE_IDENTITY_MISMATCH"
    namespace = _run_fallback_diagnostic_fragment(
        source=source,
        dest=dest,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        context_path=context_path,
        secondary_error=secondary,
    )

    diagnostics = dest / "untrusted-execution-diagnostics"
    for name, original in originals.items():
        assert (diagnostics / name).read_bytes() == original
    assert (diagnostics / "execution-driver.stdout").read_bytes() == stdout_path.read_bytes()
    assert (diagnostics / "execution-driver.stderr").read_bytes() == stderr_path.read_bytes()
    assert (diagnostics / "execution-context.json").read_bytes() == context_path.read_bytes()

    hashes = json.loads((diagnostics / "sha256.json").read_text(encoding="utf-8"))
    assert hashes["non_authoritative"] is True
    for name in originals:
        assert hashes["files"][name] == bridge._sha256_file(diagnostics / name)

    failure = namespace["failure"]
    assert failure["failed_task"] == "execution-boundary"
    assert failure["stderr"] == secondary
    assert failure["untrusted_diagnostics"]["non_authoritative"] is True
    observed = failure["observed_execution_failure"]
    assert observed["failed_task"] == "bridge-infrastructure"
    assert observed["terminal_status"] == "FAILED"


@pytest.mark.parametrize("failed_task", ["producer", "independent-validator"])
def test_task_failure_snapshot_survives_secondary_seal_failure_without_becoming_authoritative(
    tmp_path: Path, failed_task: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    failing = tmp_path / "fail.py"
    failing.write_text(
        "import sys\nprint('first stdout')\nprint('first stderr', file=sys.stderr)\nsys.exit(17)\n",
        encoding="utf-8",
    )
    original_failure = bridge._execute_plan(
        tmp_path,
        source,
        [(failed_task, [sys.executable, str(failing)])],
    )
    assert original_failure is not None
    original = (source / "task-failure.json").read_bytes()

    dest = tmp_path / "sealed"
    dest.mkdir()
    stdout_path, stderr_path, context_path = _diagnostic_control_files(tmp_path)
    secondary = "ValueError: REVIEWER_BRIDGE_EVIDENCE_SOURCE_IDENTITY_MISMATCH"
    namespace = _run_fallback_diagnostic_fragment(
        source=source,
        dest=dest,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        context_path=context_path,
        secondary_error=secondary,
        bridge_rc=0,
    )

    diagnostics = dest / "untrusted-execution-diagnostics"
    assert (diagnostics / "task-failure.json").read_bytes() == original
    preserved = json.loads((diagnostics / "task-failure.json").read_text(encoding="utf-8"))
    assert preserved["failed_task"] == failed_task
    assert preserved["return_code"] == 17
    assert preserved["stdout"] == "first stdout\n"
    assert preserved["stderr"] == "first stderr\n"

    authoritative = namespace["failure"]
    assert authoritative["failed_task"] == "execution-boundary"
    assert authoritative["stderr"] == secondary
    assert authoritative["observed_execution_failure"]["failed_task"] == failed_task
    assert authoritative["observed_execution_failure"]["return_code"] == 17
    hashes = json.loads((diagnostics / "sha256.json").read_text(encoding="utf-8"))
    assert hashes["files"]["task-failure.json"] == bridge._sha256_bytes(original)


def test_workflow_fallback_keeps_expected_identity_authoritative_and_hashes_diagnostics() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'diagnostics=dest/"untrusted-execution-diagnostics"' in text
    for name in (
        "task-failure.json",
        "result.json",
        "status.json",
        "bridge-source.json",
        "trusted-commit-validator.jsonl",
        "execution-driver.stdout",
        "execution-driver.stderr",
    ):
        assert name in text
    assert '"non_authoritative":True' in text
    assert 'bridge._json_dump(diagnostics/"sha256.json"' in text
    assert 'bridge._json_dump(dest/"bridge-source.json",expected)' in text
    assert '"failed_task":"execution-boundary"' in text
    assert '"observed_execution_failure"' in text
    assert '"failed_task":failed_task' not in text
