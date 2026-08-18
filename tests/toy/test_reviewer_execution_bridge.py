from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

import scripts.run_reviewer_execution_bridge as bridge

SHA = "a" * 40
WORKFLOW_SHA = "b" * 40
WORKFLOW = Path(".github/workflows/reviewer-execution-bridge.yml")


def _event(*, actor: str = "egorkogor", issue: int = 36, body: str | None = None) -> dict:
    body = body or f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA}"
    return {
        "repository": {"full_name": "egorkogor/Planning-Model"},
        "issue": {"number": issue},
        "sender": {"login": actor},
        "comment": {"id": 5310000000, "body": body, "user": {"login": actor}},
    }


def _write_event(path: Path, event: dict | None = None) -> None:
    path.write_text(json.dumps(event or _event()), encoding="utf-8")


def _failed_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str = "producer"):
    root = tmp_path / "bridge"
    work = root / "100-1"
    work.mkdir(parents=True)
    event = tmp_path / "event.json"
    _write_event(event)
    image = tmp_path / "image"
    image.write_text("image-123\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "RUNNER_IMAGE_ID_PATH", image)
    monkeypatch.setenv("FIXED_TARGET_RUNNER_IMAGE", "image-123")
    monkeypatch.setattr(bridge, "validate_trusted_checkout", lambda *a, **k: '{"trusted":true}')
    monkeypatch.setattr(
        bridge,
        "bridge_source_identity",
        lambda *a, **k: {"workflow_sha": WORKFLOW_SHA, "source_identity": "sha256:source"},
    )
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\n"
        "print('synthetic stdout')\n"
        "print('synthetic stderr',file=sys.stderr)\n"
        "sys.exit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge, "task_plan", lambda *a, **k: [(task, [sys.executable, str(script)])]
    )
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
    return bridge.parse_event(_event()), work / "evidence", result


def test_exact_owner_control_issue_request_parses() -> None:
    request = bridge.parse_event(_event())
    assert (request.task, request.request_id, request.implementation_sha) == (
        "status-v1", "req-0001", SHA
    )


@pytest.mark.parametrize(
    "event,error",
    [
        (_event(actor="mallory"), "ACTOR_FORBIDDEN"),
        (_event(issue=35), "CONTROL_ISSUE_MISMATCH"),
        (
            _event(body=f"/reviewer-bridge/v1 task=unknown request=req-0001 sha={SHA}"),
            "COMMAND_INVALID",
        ),
        (
            _event(body=f"/reviewer-bridge/v1 task=status-v1 request=../../tmp sha={SHA}"),
            "COMMAND_INVALID",
        ),
        (
            _event(
                body=f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA}\nwhoami"
            ),
            "COMMAND_INVALID",
        ),
    ],
)
def test_request_guards_fail_closed(event: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        bridge.parse_event(event)


def test_allowlist_plans_are_fixed(tmp_path: Path) -> None:
    status = bridge.parse_event(_event())
    assert bridge.task_plan(status, tmp_path) == []
    preflight = bridge.parse_event(
        _event(body=f"/reviewer-bridge/v1 task=preflight-v1 request=req-0002 sha={SHA}")
    )
    assert [name for name, _ in bridge.task_plan(preflight, tmp_path)] == ["fixed-target-preflight"]
    science = bridge.parse_event(
        _event(
            body=(
                "/reviewer-bridge/v1 task=a2-sufficient-budget-task-order-v1 "
                f"request=req-0003 sha={SHA}"
            )
        )
    )
    plan = bridge.task_plan(science, tmp_path)
    assert [name for name, _ in plan] == [
        "fixed-target-preflight",
        "producer",
        "independent-validator",
    ]
    flat = " ".join(arg for _, argv in plan for arg in argv)
    assert "scripts.run_a2_sufficient_budget_task_order" in flat
    assert "clipping" not in flat.lower() and "python -c" not in flat and "ssh" not in flat.lower()


def test_reservation_rejects_duplicate_without_repository_python(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "bridge"
    work = root / "100-1"
    work.mkdir(parents=True)
    event = tmp_path / "event.json"
    _write_event(event)
    monkeypatch.setattr(bridge, "_validate_workflow_sha", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "validate_requested_commit_on_protected_main", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "_ref_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        bridge,
        "validate_trusted_checkout",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("requested python executed")),
    )
    with pytest.raises(ValueError, match="REQUEST_ALREADY_CONSUMED"):
        bridge.reserve_request(
            event_path=event,
            repo_root=tmp_path,
            bridge_root=root,
            workspace=work,
            workflow_sha=WORKFLOW_SHA,
            run_id=100,
            run_attempt=1,
            token="unused",
        )


def test_workflow_uses_real_principal_boundary_and_no_cross_job_spool() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "uses:" not in text
    assert "workflow_dispatch" not in text
    assert "runs-on: ubuntu-latest" in text
    assert text.count("planning-model-canonical-cpu-v1") == 1
    assert "BRIDGE_EXECUTION_USER: planning-model-bridge-exec" in text
    assert 'sudo -n -u "$user" -- /usr/bin/env -i' in text
    assert "EXECUTION_PRINCIPAL_SHARES_CONTROL_GROUP" in text
    assert "EXECUTION_PRINCIPAL_CAN_ESCALATE" in text
    assert "DETACHED_EXECUTION_PROCESS_SURVIVED" in text
    assert "/var/tmp/planning-model-reviewer-bridge-spool" not in text
    assert "reviewer-bridge-publisher:" not in text
    assert "reviewer-bridge-execution:" not in text


def test_workflow_write_token_appears_only_off_host_reservation_and_after_kill() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("GITHUB_TOKEN: ${{ github.token }}") == 2
    canonical = text.index("  reviewer-bridge-canonical:")
    kill = text.index("REVIEWER_BRIDGE_DETACHED_EXECUTION_PROCESS_SURVIVED", canonical)
    publish = text.index("GITHUB_TOKEN: ${{ github.token }}", canonical)
    assert kill < publish
    execute = text[text.index("- id: execute", canonical):text.index("- id: seal", canonical)]
    assert "GITHUB_TOKEN: ${{ github.token }}" not in execute
    assert "GITHUB_EVENT_PATH" not in execute[execute.index("/usr/bin/env -i"):]


def test_workflow_rebuilds_control_code_after_execution_uid_is_dead() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    seal = text.index("- id: seal")
    killed = text.index("DETACHED_EXECUTION_PROCESS_SURVIVED", seal)
    fresh = text.index("bootstrap-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}", seal)
    source_check = text.index("EVIDENCE_SOURCE_IDENTITY_MISMATCH", seal)
    publish = text.index("- id: publish", seal)
    assert killed < fresh < source_check < publish
    assert "bridge.bridge_source_identity(repo,workflow_sha)" in text
    assert "validate_publishable_evidence" in text


def test_workflow_synthesizes_trusted_failure_if_execution_evidence_is_invalid() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert '"failed_task":"execution-boundary"' in text
    assert '"terminal_status":"FAILED"' in text
    assert "build_evidence_manifest" in text
    assert "_write_deterministic_archive" in text
    assert '"execution_principal":"planning-model-bridge-exec"' in text
    assert "execution-driver.stdout" in text and "execution-driver.stderr" in text


def test_workflow_materializes_fixed_target_runner_image_and_binds_workflow_sha() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "/etc/planning-model-runner-image-id" in text
    assert 'FIXED_TARGET_RUNNER_IMAGE="$runner_image"' in text
    assert "REVIEWER_BRIDGE_FIXED_TARGET_RUNNER_IMAGE_DRIFT" in text
    obj = '"${WORKFLOW_SHA}:scripts/run_reviewer_execution_bridge.py"'
    assert text.count(obj) >= 3


def test_runner_identity_requires_exact_materialized_image(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "image"
    image.write_text("image-123\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "RUNNER_IMAGE_ID_PATH", image)
    monkeypatch.delenv("FIXED_TARGET_RUNNER_IMAGE", raising=False)
    with pytest.raises(ValueError, match="FIXED_TARGET_RUNNER_IMAGE_MISSING"):
        bridge._runner_identity("canonical")
    monkeypatch.setenv("FIXED_TARGET_RUNNER_IMAGE", "wrong")
    with pytest.raises(ValueError, match="FIXED_TARGET_RUNNER_IMAGE_DRIFT"):
        bridge._runner_identity("canonical")
    monkeypatch.setenv("FIXED_TARGET_RUNNER_IMAGE", "image-123")
    assert bridge._runner_identity("canonical")["fixed_target_runner_image"] == "image-123"


@pytest.mark.parametrize(
    ("task", "status"),
    [("producer", "FAILED"), ("independent-validator", "VALIDATOR_FAILED")],
)
def test_failed_execution_is_terminal_and_content_addressed(
    tmp_path: Path, monkeypatch, task: str, status: str
) -> None:
    request, evidence, result = _failed_evidence(tmp_path, monkeypatch, task)
    failure = json.loads((evidence / "task-failure.json").read_text())
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert result["valid"] is False and result["terminal_status"] == status
    assert failure["failed_task"] == task and failure["return_code"] == 7
    assert failure["stdout"] == "synthetic stdout\n" and failure["stderr"] == "synthetic stderr\n"
    assert "task-failure.json" in manifest["files"]
    assert manifest["manifest_sha256"].startswith("sha256:")
    assert bridge.validate_publishable_evidence(
        evidence, request=request, run_id=100, workflow_sha=WORKFLOW_SHA
    ) == result


def test_repository_subprocess_environment_scrubs_transport_and_control(monkeypatch) -> None:
    sentinels = {
        "GITHUB_TOKEN": "token",
        "GH_TOKEN": "gh",
        "GITHUB_PAT": "pat",
        "GITHUB_ENV": "/tmp/env",
        "GITHUB_PATH": "/tmp/path",
        "BRIDGE_DRIVER": "/tmp/driver",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SAFE_ENV", "kept")
    env = bridge._repository_subprocess_env()
    assert env["SAFE_ENV"] == "kept"
    assert not bridge.REPOSITORY_CREDENTIAL_ENV_KEYS.intersection(env)
    assert not bridge.REPOSITORY_CONTROL_ENV_KEYS.intersection(env)
    assert not any(key.startswith("GIT_CONFIG_") for key in env)


def test_scoped_push_auth_has_no_direct_token_env(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "old")
    env = bridge._git_authenticated_env("intended")
    assert "GITHUB_TOKEN" not in env and "GH_TOKEN" not in env
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")
    assert "intended" not in env.values()


def test_real_repository_child_cannot_observe_scrubbed_control_state(
    tmp_path: Path, monkeypatch
) -> None:
    for key, value in {
        "GITHUB_TOKEN": "TOKEN",
        "GITHUB_ENV": "/tmp/env",
        "GITHUB_PATH": "/tmp/path",
        "BRIDGE_DRIVER": "/tmp/driver",
        "GIT_CONFIG_COUNT": "1",
    }.items():
        monkeypatch.setenv(key, value)
    probe = tmp_path / "probe.py"
    forbidden = sorted(bridge.REPOSITORY_CREDENTIAL_ENV_KEYS | bridge.REPOSITORY_CONTROL_ENV_KEYS)
    probe.write_text(
        "import os,sys\n"
        f"bad={forbidden!r}\n"
        "leak=[k for k in os.environ if k in bad or k.startswith('GIT_CONFIG_')]\n"
        "raise SystemExit(91 if leak else 0)\n",
        encoding="utf-8",
    )
    assert (
        bridge._execute_plan(
            tmp_path, tmp_path, [("producer", [sys.executable, str(probe)])]
        )
        is None
    )


def test_publishable_evidence_rejects_symlink(tmp_path: Path, monkeypatch) -> None:
    request, evidence, _ = _failed_evidence(tmp_path, monkeypatch)
    outside = tmp_path / "secret"
    outside.write_text("secret")
    link = evidence / "link"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    with pytest.raises(ValueError, match="EVIDENCE_SYMLINK_FORBIDDEN"):
        bridge.validate_publishable_evidence(
            evidence, request=request, run_id=100, workflow_sha=WORKFLOW_SHA
        )


def test_cleanup_and_transport_namespace_are_contained(tmp_path: Path) -> None:
    root = tmp_path / "bridge"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="WORKSPACE_OUTSIDE_ROOT"):
        bridge.cleanup_workspace(root, outside)
    work = root / "123-1"
    work.mkdir()
    bridge.cleanup_workspace(root, work)
    assert not work.exists() and outside.exists()
    assert bridge.transport_ref("req-0001") == "refs/heads/evidence/reviewer-bridge/req-0001"
    assert "not source" in bridge.TRANSPORT_WARNING
    assert "must never be merged" in bridge.TRANSPORT_WARNING


def _sudo_prefix() -> list[str] | None:
    sudo = shutil.which("sudo")
    if sudo is None:
        return None
    completed = subprocess.run([sudo, "-n", "true"], check=False)
    return [sudo, "-n"] if completed.returncode == 0 else None


def test_detached_attacker_cannot_cross_real_os_principal_boundary() -> None:
    sudo = _sudo_prefix()
    required = ["useradd", "userdel", "pkill", "pgrep"]
    if sudo is None or any(shutil.which(command) is None for command in required):
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("CI must support the real OS-principal escalation regression")
        pytest.skip("passwordless sudo and disposable-user tools required")

    root = Path(tempfile.mkdtemp(prefix="reviewer-bridge-boundary-"))
    root.chmod(0o755)
    user = f"bridgetest{os.getpid()}"
    control = root / "control"
    execution = root / "execution"
    control.mkdir(mode=0o700)
    execution.mkdir(mode=0o711)
    attacker = execution / "attacker.py"
    report = execution / "report.jsonl"
    driver = control / "publisher-driver.py"
    token = control / "write-token"
    sealed = control / "sealed-evidence"
    leak = execution / "leak"
    pid_file = execution / "attacker.pid"
    attacker.write_text(
        "import json,os,time\nfrom pathlib import Path\nimport sys\n"
        "driver,token,sealed,report,leak,pid=map(Path,sys.argv[1:])\n"
        "pid.write_text(str(os.getpid()))\n"
        "for _ in range(300):\n"
        " row={'driver':False,'token':False,'sealed':False}\n"
        " try: driver.write_text('ATTACK'); row['driver']=True\n except OSError: pass\n"
        " try: leak.write_text(token.read_text()); row['token']=True\n except OSError: pass\n"
        " try: sealed.write_text('ATTACK'); row['sealed']=True\n except OSError: pass\n"
        " with report.open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\\n')\n"
        " time.sleep(.01)\n",
        encoding="utf-8",
    )
    created = False
    process: subprocess.Popen[str] | None = None
    try:
        subprocess.run(
            [
                *sudo, "useradd", "--system", "--no-create-home",
                "--shell", "/usr/sbin/nologin", user,
            ],
            check=True,
        )
        created = True
        subprocess.run([*sudo, "chown", "-R", f"{user}:{user}", str(execution)], check=True)
        process = subprocess.Popen(
            [
                *sudo, "-u", user, "--", sys.executable, str(attacker),
                str(driver), str(token), str(sealed), str(report), str(leak), str(pid_file),
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        deadline = time.time() + 5
        while time.time() < deadline and not pid_file.exists():
            time.sleep(.02)
        assert pid_file.exists()
        driver.write_text("TRUSTED_DRIVER")
        token.write_text("WRITE_TOKEN_CONTROL_ONLY")
        sealed.write_text("FROZEN")
        for path in (driver, token, sealed):
            path.chmod(0o600)
        time.sleep(.25)
        assert driver.read_text() == "TRUSTED_DRIVER"
        assert sealed.read_text() == "FROZEN"
        assert not leak.exists()
        rows = [json.loads(line) for line in report.read_text().splitlines()]
        assert rows and not any(any(row.values()) for row in rows)

        uid = subprocess.run(
            ["id", "-u", user], check=True, capture_output=True, text=True
        ).stdout.strip()
        subprocess.run([*sudo, "pkill", "-KILL", "-u", uid], check=False)
        deadline = time.time() + 5
        while time.time() < deadline:
            check = subprocess.run(
                ["pgrep", "-u", uid], check=False, stdout=subprocess.DEVNULL
            )
            if check.returncode != 0:
                break
            time.sleep(.02)
        final_check = subprocess.run(
            ["pgrep", "-u", uid], check=False, stdout=subprocess.DEVNULL
        )
        assert final_check.returncode != 0
        push_env = bridge._git_authenticated_env("WRITE_TOKEN_CONTROL_ONLY")
        assert "GITHUB_TOKEN" not in push_env
        assert "WRITE_TOKEN_CONTROL_ONLY" not in push_env.values()
    finally:
        if process is not None:
            process.wait(timeout=5)
        if created:
            subprocess.run([*sudo, "pkill", "-KILL", "-u", user], check=False)
            subprocess.run([*sudo, "userdel", user], check=False)
        shutil.rmtree(root, ignore_errors=True)


def test_parser_preserves_explicit_rerun_attempt() -> None:
    args = bridge.parser().parse_args(
        [
            "reserve", "--event", "event.json", "--repo-root", ".",
            "--bridge-root", ".", "--workspace", "1-1",
            "--workflow-sha", WORKFLOW_SHA, "--run-id", "1",
            "--run-attempt", "2",
        ]
    )
    assert args.run_attempt == 2
