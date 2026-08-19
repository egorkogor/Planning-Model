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


def _event() -> dict:
    return {
        "repository": {"full_name": "egorkogor/Planning-Model"},
        "issue": {"number": 36},
        "sender": {"login": "egorkogor"},
        "comment": {
            "id": 5334849824,
            "body": f"/reviewer-bridge/v1 task=status-v1 request=req-runtime-0001 sha={SHA}",
            "user": {"login": "egorkogor"},
        },
    }


def _fallback_primary_fragment() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    load_start = text.index("              primary_failure_bytes=None")
    load_end = text.index("              shutil.rmtree(dest,ignore_errors=True)", load_start)
    output_start = text.index(
        '              out=Path(stdout_path).read_text(encoding="utf-8",errors="replace")',
        load_end,
    )
    output_end = text.index(
        '              bridge._json_dump(dest/"task-failure.json",failure)',
        output_start,
    )
    return textwrap.dedent(text[load_start:load_end] + text[output_start:output_end])


def _run_fallback_primary_fragment(
    *,
    source: Path,
    dest: Path,
    stdout_path: Path,
    stderr_path: Path,
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
        "error": secondary_error,
        "bridge_rc": bridge_rc,
    }
    exec(_fallback_primary_fragment(), namespace)
    return namespace


def test_workflow_preserves_venv_launcher_and_verifies_runtime() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "os.path.realpath(sys.executable)" not in text
    assert "os.path.abspath(sys.executable)" in text
    assert '"$python_executable" "$driver" execute' in text
    assert "import torch" in text
    assert "import scripts.run_fixed_target_acceptance" in text
    assert 'torch.__version__ != contract.get("torch_version")' in text
    assert 'context.get("torch_version") != contract.get("torch_version")' in text
    assert 'context.get("runtime_imports_verified") is not True' in text
    assert "Path(str(context.get(\"python_executable\"))).resolve()" not in text
    assert (
        'os.path.abspath(str(context.get("python_executable"))) != '
        "os.path.abspath(expected_python)"
    ) in text


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


def test_driver_children_keep_the_preserved_python_executable(tmp_path: Path) -> None:
    request = bridge.parse_event(_event())
    preflight = bridge.task_plan(
        bridge.parse_event(
            {
                **_event(),
                "comment": {
                    **_event()["comment"],
                    "body": f"/reviewer-bridge/v1 task=preflight-v1 request=req-runtime-0002 sha={SHA}",
                },
            }
        ),
        tmp_path,
    )
    assert preflight[0][1][0] == sys.executable
    assert request.task == "status-v1"
    science = bridge.task_plan(
        bridge.parse_event(
            {
                **_event(),
                "comment": {
                    **_event()["comment"],
                    "body": (
                        "/reviewer-bridge/v1 task=a2-sufficient-budget-task-order-v1 "
                        f"request=req-runtime-0003 sha={SHA}"
                    ),
                },
            }
        ),
        tmp_path,
    )
    assert science
    assert all(argv[0] == sys.executable for _, argv in science)


def test_early_bridge_infrastructure_failure_survives_secondary_seal_failure(
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
    original = (source / "task-failure.json").read_bytes()
    original_failure = json.loads(original)
    assert result["failed_task"] == "bridge-infrastructure"
    assert original_failure["stderr"] == "synthetic first production failure"
    assert json.loads((source / "bridge-source.json").read_text())["available"] is False

    dest = tmp_path / "sealed"
    dest.mkdir()
    stdout_path = tmp_path / "stdout"
    stderr_path = tmp_path / "stderr"
    stdout_path.write_text('{"terminal_status":"FAILED"}\n', encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    secondary = "ValueError: REVIEWER_BRIDGE_EVIDENCE_SOURCE_IDENTITY_MISMATCH"
    namespace = _run_fallback_primary_fragment(
        source=source,
        dest=dest,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        secondary_error=secondary,
    )

    assert (dest / "primary-task-failure.json").read_bytes() == original
    failure = namespace["failure"]
    assert namespace["failed_task"] == "bridge-infrastructure"
    assert failure["failed_task"] == "bridge-infrastructure"
    assert failure["stderr"] == "synthetic first production failure"
    assert failure["secondary_seal_error"] == secondary
    assert failure["secondary_failed_task"] == "execution-boundary"
    assert failure["primary_failure_sha256"] == bridge._sha256_bytes(original)


@pytest.mark.parametrize("failed_task", ["producer", "independent-validator"])
def test_task_failure_survives_secondary_seal_failure(
    tmp_path: Path, failed_task: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    failing = tmp_path / "fail.py"
    failing.write_text(
        "import sys\nprint('first stdout')\nprint('first stderr', file=sys.stderr)\nsys.exit(17)\n",
        encoding="utf-8",
    )
    failure = bridge._execute_plan(
        tmp_path,
        source,
        [(failed_task, [sys.executable, str(failing)])],
    )
    assert failure is not None
    original = (source / "task-failure.json").read_bytes()

    dest = tmp_path / "sealed"
    dest.mkdir()
    stdout_path = tmp_path / "stdout"
    stderr_path = tmp_path / "stderr"
    stdout_path.write_text("driver summary\n", encoding="utf-8")
    stderr_path.write_text("driver stderr\n", encoding="utf-8")
    secondary = "ValueError: REVIEWER_BRIDGE_EVIDENCE_SOURCE_IDENTITY_MISMATCH"
    namespace = _run_fallback_primary_fragment(
        source=source,
        dest=dest,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        secondary_error=secondary,
    )

    assert (dest / "primary-task-failure.json").read_bytes() == original
    preserved = namespace["failure"]
    assert namespace["failed_task"] == failed_task
    assert namespace["return_code"] == 17
    assert preserved["failed_task"] == failed_task
    assert preserved["return_code"] == 17
    assert preserved["stdout"] == "first stdout\n"
    assert preserved["stderr"] == "first stderr\n"
    assert preserved["secondary_seal_error"] == secondary
    assert preserved["secondary_failed_task"] == "execution-boundary"


def test_workflow_fallback_exposes_primary_failure_and_secondary_diagnostic() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'primary_failure_path=source/"task-failure.json"' in text
    assert '(dest/"primary-task-failure.json").write_bytes(primary_failure_bytes)' in text
    assert 'failure["primary_failure_sha256"]' in text
    assert 'failure["secondary_seal_error"]' in text
    assert 'failure["secondary_failed_task"]="execution-boundary"' in text
    assert '"failed_task":failed_task' in text
