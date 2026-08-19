from __future__ import annotations

import json
import os
import stat
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


def _snapshot_python_fragment() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    marker = '          python - "$workspace" "$control_root" "$diagnostic_snapshot" <<\'PY\'\n'
    start = text.index(marker) + len(marker)
    end = text.index("\n          PY\n", start)
    return textwrap.dedent(text[start:end])


def _fallback_diagnostic_fragment() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index('              diagnostics=dest/"untrusted-execution-diagnostics"')
    end = text.index(
        '              bridge._json_dump(dest/"task-failure.json",failure)', start
    )
    return textwrap.dedent(text[start:end])


def _snapshot_cleanup_fragment() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index('          if test -e "$diagnostic_snapshot"; then')
    end = text.index('          if test -n "$execution_root"; then', start)
    return textwrap.dedent(text[start:end])


def _run_snapshot_fragment(
    *, workspace: Path, control_root: Path, diagnostic_snapshot: Path, tmp_path: Path
) -> None:
    script = tmp_path / "snapshot-fragment.py"
    script.write_text(_snapshot_python_fragment(), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(workspace),
            str(control_root),
            str(diagnostic_snapshot),
        ],
        check=True,
    )


def _run_fallback_diagnostic_fragment(
    *,
    diagnostic_snapshot: Path,
    dest: Path,
    secondary_error: str,
    bridge_rc: int = 0,
) -> dict:
    namespace = {
        "json": json,
        "os": os,
        "Path": Path,
        "bridge": bridge,
        "diagnostic_snapshot": str(diagnostic_snapshot),
        "dest": dest,
        "error": secondary_error,
        "bridge_rc": bridge_rc,
    }
    exec(_fallback_diagnostic_fragment(), namespace)
    return namespace


def _run_snapshot_cleanup_fragment(
    diagnostic_snapshot: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "snapshot-cleanup.sh"
    script.write_text(
        "set -euo pipefail\n"
        "diagnostic_snapshot=$1\n"
        "control_uid=$(id -u)\n"
        "control_gid=$(id -g)\n"
        + _snapshot_cleanup_fragment()
        + "\nprintf 'publish-eligible\\n'\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(script), str(diagnostic_snapshot)],
        check=True,
        capture_output=True,
        text=True,
    )


def _diagnostic_control_files(control_root: Path) -> tuple[Path, Path, Path]:
    control_root.mkdir(parents=True, exist_ok=True)
    stdout = control_root / "execution-driver.stdout"
    stderr = control_root / "execution-driver.stderr"
    context = control_root / "execution-context.json"
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
    assert text.count('/usr/bin/env -C "$repo" -i') == 2


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


def test_snapshot_precedes_any_reseal_mutation_and_fallback_reads_snapshot_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    seal = text.index("      - id: seal")
    kill = text.index("REVIEWER_BRIDGE_DETACHED_EXECUTION_PROCESS_SURVIVED", seal)
    reclaim = text.index('sudo -n chown -R "$control_uid:$control_gid" "$workspace"', kill)
    snapshot = text.index('diagnostic_snapshot="$control_root/diagnostic-snapshot-', reclaim)
    snapshot_run = text.index(
        'python - "$workspace" "$control_root" "$diagnostic_snapshot"', snapshot
    )
    first_source_mutation = text.index(
        'bridge._json_dump(source/"github-execution.json",execution)', snapshot_run
    )
    destructive_rebuild = text.index(
        'for name in ("manifest.json","evidence.tar.gz","archive.sha256","result.json")',
        first_source_mutation,
    )
    assert kill < reclaim < snapshot < snapshot_run < first_source_mutation < destructive_rebuild

    fallback = _fallback_diagnostic_fragment()
    assert "candidate=snapshot/name" in fallback
    assert "candidate=source/name" not in fallback
    assert "stdout_path" not in fallback
    assert "stderr_path" not in fallback
    assert "context_path" not in fallback


def test_destructive_reseal_window_preserves_original_diagnostics_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bridge"
    workspace = root / "100-1"
    workspace.mkdir(parents=True)
    event = tmp_path / "event.json"
    event.write_text(json.dumps(_event()), encoding="utf-8")

    def fail_before_source_identity(*args, **kwargs):
        raise RuntimeError("synthetic first production failure")

    monkeypatch.setattr(bridge, "validate_trusted_checkout", fail_before_source_identity)
    result = bridge.execute_request(
        event_path=event,
        repo_root=tmp_path,
        bridge_root=root,
        workspace=workspace,
        workflow_sha=WORKFLOW_SHA,
        run_id=100,
        run_attempt=1,
        job_name="reviewer-bridge-canonical",
        runner_name="canonical",
    )
    source = workspace / "evidence"
    assert result["failed_task"] == "bridge-infrastructure"
    trusted = source / "trusted-commit-validator.jsonl"
    trusted.write_bytes(b'{"trusted":false,"synthetic":true}\n')

    control_root = tmp_path / "control"
    stdout_path, stderr_path, context_path = _diagnostic_control_files(control_root)
    originals = {
        name: (source / name).read_bytes()
        for name in (
            "task-failure.json",
            "result.json",
            "status.json",
            "bridge-source.json",
            "trusted-commit-validator.jsonl",
        )
    }
    originals.update(
        {
            "execution-driver.stdout": stdout_path.read_bytes(),
            "execution-driver.stderr": stderr_path.read_bytes(),
            "execution-context.json": context_path.read_bytes(),
        }
    )

    diagnostic_snapshot = control_root / "diagnostic-snapshot-100-1"
    _run_snapshot_fragment(
        workspace=workspace,
        control_root=control_root,
        diagnostic_snapshot=diagnostic_snapshot,
        tmp_path=tmp_path,
    )
    if os.name == "posix":
        assert stat.S_IMODE(diagnostic_snapshot.stat().st_mode) == 0o500
        for child in diagnostic_snapshot.iterdir():
            assert stat.S_IMODE(child.stat().st_mode) == 0o400

    # Reproduce the destructive window: source metadata is removed/mutated before
    # a secondary seal failure occurs and before result.json can be restored.
    (source / "result.json").unlink()
    (source / "status.json").write_bytes(b"mutated-status\n")
    (source / "task-failure.json").write_bytes(b"mutated-failure\n")
    (source / "bridge-source.json").write_bytes(b"mutated-source\n")
    trusted.write_bytes(b"mutated-validator\n")
    stdout_path.write_bytes(b"mutated-stdout\n")
    stderr_path.write_bytes(b"mutated-stderr\n")
    context_path.write_bytes(b"mutated-context\n")

    dest = tmp_path / "sealed"
    dest.mkdir()
    secondary = "ValueError: REVIEWER_BRIDGE_EVIDENCE_SYMLINK_FORBIDDEN"
    namespace = _run_fallback_diagnostic_fragment(
        diagnostic_snapshot=diagnostic_snapshot,
        dest=dest,
        secondary_error=secondary,
    )

    diagnostics = dest / "untrusted-execution-diagnostics"
    for name, original in originals.items():
        assert (diagnostics / name).read_bytes() == original

    hashes = json.loads((diagnostics / "sha256.json").read_text(encoding="utf-8"))
    assert hashes["non_authoritative"] is True
    for name, original in originals.items():
        assert hashes["files"][name] == bridge._sha256_bytes(original)

    preserved_failure = json.loads(originals["task-failure.json"])
    assert preserved_failure["stderr"] == "synthetic first production failure"
    assert json.loads(originals["bridge-source.json"])["available"] is False

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
    workspace = tmp_path / "workspace"
    source = workspace / "evidence"
    source.mkdir(parents=True)
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

    control_root = tmp_path / "control"
    _diagnostic_control_files(control_root)
    diagnostic_snapshot = control_root / "diagnostic-snapshot-200-1"
    _run_snapshot_fragment(
        workspace=workspace,
        control_root=control_root,
        diagnostic_snapshot=diagnostic_snapshot,
        tmp_path=tmp_path,
    )

    # The execution-owned source can now be destroyed without affecting the first failure.
    (source / "task-failure.json").unlink()
    dest = tmp_path / "sealed"
    dest.mkdir()
    secondary = "ValueError: REVIEWER_BRIDGE_EVIDENCE_SOURCE_IDENTITY_MISMATCH"
    namespace = _run_fallback_diagnostic_fragment(
        diagnostic_snapshot=diagnostic_snapshot,
        dest=dest,
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


def test_read_only_snapshot_cleanup_preserves_publish_eligibility_after_early_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bridge"
    workspace = root / "300-1"
    workspace.mkdir(parents=True)
    event = tmp_path / "event.json"
    event.write_text(json.dumps(_event(request="req-runtime-0300")), encoding="utf-8")

    def fail_before_requested_driver(*args, **kwargs):
        raise RuntimeError("synthetic pre-driver runtime boundary failure")

    monkeypatch.setattr(bridge, "validate_trusted_checkout", fail_before_requested_driver)
    original_result = bridge.execute_request(
        event_path=event,
        repo_root=tmp_path,
        bridge_root=root,
        workspace=workspace,
        workflow_sha=WORKFLOW_SHA,
        run_id=300,
        run_attempt=1,
        job_name="reviewer-bridge-canonical",
        runner_name="canonical",
    )
    assert original_result["failed_task"] == "bridge-infrastructure"

    control_root = tmp_path / "control"
    _diagnostic_control_files(control_root)
    diagnostic_snapshot = control_root / "diagnostic-snapshot-300-1"
    _run_snapshot_fragment(
        workspace=workspace,
        control_root=control_root,
        diagnostic_snapshot=diagnostic_snapshot,
        tmp_path=tmp_path,
    )
    assert stat.S_IMODE(diagnostic_snapshot.stat().st_mode) == 0o500

    sealed = tmp_path / "sealed"
    dest = sealed / "evidence"
    dest.mkdir(parents=True)
    secondary = "PermissionError: synthetic runtime probe boundary"
    namespace = _run_fallback_diagnostic_fragment(
        diagnostic_snapshot=diagnostic_snapshot,
        dest=dest,
        secondary_error=secondary,
        bridge_rc=125,
    )
    request = bridge.parse_event(_event(request="req-runtime-0300"))
    expected_source = {
        "workflow_sha": WORKFLOW_SHA,
        "source_files": [],
        "source_identity": bridge._sha256_bytes(b"synthetic-control-source"),
    }
    runner_identity = {"available": False, "runner_name": "canonical"}
    failure = namespace["failure"]
    bridge._json_dump(dest / "request.json", bridge.asdict(request))
    bridge._json_dump(
        dest / "github-execution.json",
        {
            "event_name": "issue_comment",
            "run_id": 300,
            "run_attempt": 1,
            "job_name": "reviewer-bridge-canonical",
            "workflow_sha": WORKFLOW_SHA,
            "comment_id": request.comment_id,
            "control_issue": request.issue_number,
        },
    )
    bridge._json_dump(dest / "runner-identity.json", runner_identity)
    bridge._json_dump(dest / "bridge-source.json", expected_source)
    bridge._json_dump(dest / "task-failure.json", failure)
    bridge._json_dump(
        dest / "status.json",
        bridge._terminal_status_payload(
            request=request,
            terminal_status="FAILED",
            failure=failure,
        ),
    )
    manifest = bridge.build_evidence_manifest(
        dest,
        request=request,
        runner_identity=runner_identity,
        source_identity=expected_source,
    )
    bridge._json_dump(dest / "manifest.json", manifest)
    archive = bridge._write_deterministic_archive(dest)
    result = {
        "valid": False,
        "terminal_status": "FAILED",
        "task": request.task,
        "request_id": request.request_id,
        "implementation_sha": request.implementation_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_sha256": archive,
        "transport_ref": bridge.transport_ref(request.request_id),
        "failed_task": "execution-boundary",
        "return_code": 125,
    }
    bridge._json_dump(dest / "result.json", result)
    assert bridge.validate_publishable_evidence(
        dest,
        request=request,
        run_id=300,
        workflow_sha=WORKFLOW_SHA,
    )["failed_task"] == "execution-boundary"

    completed = _run_snapshot_cleanup_fragment(diagnostic_snapshot, tmp_path)
    assert completed.stdout.strip() == "publish-eligible"
    assert not diagnostic_snapshot.exists()
    assert bridge.validate_publishable_evidence(
        dest,
        request=request,
        run_id=300,
        workflow_sha=WORKFLOW_SHA,
    )["failed_task"] == "execution-boundary"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "if: always() && steps.seal.outcome == 'success'" in workflow


def test_snapshot_cleanup_failure_is_nonfatal_after_seal() -> None:
    fragment = _snapshot_cleanup_fragment()
    assert 'if ! chmod 700 "$diagnostic_snapshot"; then' in fragment
    assert 'elif ! rm -rf -- "$diagnostic_snapshot"; then' in fragment
    assert "::warning::REVIEWER_BRIDGE_DIAGNOSTIC_SNAPSHOT_CLEANUP" in fragment
    assert "exit 1" not in fragment


def test_workflow_fallback_keeps_expected_identity_authoritative_and_hashes_diagnostics() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'diagnostic_snapshot="$control_root/diagnostic-snapshot-' in text
    assert 'diagnostics=dest/"untrusted-execution-diagnostics"' in text
    for name in (
        "task-failure.json",
        "result.json",
        "status.json",
        "bridge-source.json",
        "trusted-commit-validator.jsonl",
        "execution-driver.stdout",
        "execution-driver.stderr",
        "execution-context.json",
    ):
        assert name in text
    assert '"non_authoritative":True' in text
    assert 'bridge._json_dump(diagnostics/"sha256.json"' in text
    assert 'bridge._json_dump(dest/"bridge-source.json",expected)' in text
    assert '"failed_task":"execution-boundary"' in text
    assert '"observed_execution_failure"' in text
    assert '"failed_task":failed_task' not in text
