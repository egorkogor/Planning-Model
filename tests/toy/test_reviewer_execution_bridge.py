from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.run_reviewer_execution_bridge as bridge

SHA = "a" * 40
WORKFLOW = Path(".github/workflows/reviewer-execution-bridge.yml")


def _event(
    *,
    actor: str = "egorkogor",
    issue: int = 36,
    body: str | None = None,
) -> dict:
    if body is None:
        body = f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA}"
    return {
        "repository": {"full_name": "egorkogor/Planning-Model"},
        "issue": {"number": issue},
        "sender": {"login": actor},
        "comment": {"id": 5310000000, "body": body, "user": {"login": actor}},
    }


def _write_event(path: Path, event: dict) -> None:
    import json

    path.write_text(json.dumps(event), encoding="utf-8")


def test_exact_owner_control_issue_status_request_parses() -> None:
    request = bridge.parse_event(_event())
    assert request.task == "status-v1"
    assert request.request_id == "req-0001"
    assert request.implementation_sha == SHA


def test_non_owner_comment_is_rejected() -> None:
    with pytest.raises(ValueError, match="REVIEWER_BRIDGE_ACTOR_FORBIDDEN"):
        bridge.parse_event(_event(actor="mallory"))


def test_wrong_issue_is_rejected() -> None:
    with pytest.raises(ValueError, match="REVIEWER_BRIDGE_CONTROL_ISSUE_MISMATCH"):
        bridge.parse_event(_event(issue=35))


@pytest.mark.parametrize(
    "body",
    [
        f"/reviewer-bridge/v1 task=unknown request=req-0001 sha={SHA}",
        f"/reviewer-bridge/v1 task=status-v1 request=x sha={SHA}",
        f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={'A' * 40}",
        f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA} extra=x",
    ],
)
def test_malformed_or_unknown_commands_fail_closed(body: str) -> None:
    with pytest.raises(ValueError, match="REVIEWER_BRIDGE_COMMAND_INVALID"):
        bridge.parse_event(_event(body=body))


@pytest.mark.parametrize(
    "payload",
    [
        f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA};id",
        f"/reviewer-bridge/v1 task=status-v1 request=../../tmp sha={SHA}",
        f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA}\nwhoami",
        f"/reviewer-bridge/v1 task=status-v1 request=req-0001 sha={SHA} path=/tmp/x",
    ],
)
def test_shell_and_path_injection_is_rejected(payload: str) -> None:
    with pytest.raises(ValueError, match="REVIEWER_BRIDGE_COMMAND_INVALID"):
        bridge.parse_event(_event(body=payload))


def test_status_and_preflight_have_fixed_allowlisted_plans(tmp_path: Path) -> None:
    status = bridge.parse_event(_event())
    assert bridge.task_plan(status, tmp_path) == []
    preflight_body = (
        f"/reviewer-bridge/v1 task=preflight-v1 request=req-0002 sha={SHA}"
    )
    preflight = bridge.parse_event(_event(body=preflight_body))
    plan = bridge.task_plan(preflight, tmp_path)
    assert [name for name, _ in plan] == ["fixed-target-preflight"]
    argv = plan[0][1]
    assert "scripts.run_fixed_target_acceptance" in argv
    assert "preflight" in argv
    assert bridge.TARGET_CONTRACT in argv
    assert "-c" not in argv


def test_scientific_registry_is_one_named_existing_producer_without_clipping(
    tmp_path: Path,
) -> None:
    request = bridge.parse_event(
        _event(
            body=(
                "/reviewer-bridge/v1 task=a2-sufficient-budget-task-order-v1 "
                f"request=req-0003 sha={SHA}"
            )
        )
    )
    plan = bridge.task_plan(request, tmp_path)
    assert [name for name, _ in plan] == [
        "fixed-target-preflight",
        "producer",
        "independent-validator",
    ]
    flattened = " ".join(arg for _, argv in plan for arg in argv)
    assert "scripts.run_a2_sufficient_budget_task_order" in flattened
    assert "clipping" not in flattened.lower()
    assert "bash" not in flattened
    assert "ssh" not in flattened
    assert "python -c" not in flattened


def test_exact_trusted_status_checkout_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = bridge.parse_event(_event())

    def fake_git(repo_root: Path, *args: str, capture_output: bool = False) -> str:
        del repo_root, capture_output
        if args == ("rev-parse", "HEAD"):
            return SHA
        if args[:2] == ("status", "--porcelain=v1"):
            return ""
        raise AssertionError(args)

    completed = subprocess.CompletedProcess(
        args=["trusted-validator"],
        returncode=0,
        stdout='{"trusted": true}',
        stderr="",
    )
    monkeypatch.setattr(bridge, "_git", fake_git)
    monkeypatch.setattr(
        bridge,
        "_validate_workflow_sha",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(bridge, "_run", lambda *args, **kwargs: completed)
    result = bridge.validate_trusted_checkout(
        tmp_path,
        request,
        workflow_sha="b" * 40,
    )
    assert result == '{"trusted": true}'


def test_untrusted_commit_validator_failure_is_propagated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = bridge.parse_event(_event())

    def fake_git(repo_root: Path, *args: str, capture_output: bool = False) -> str:
        del repo_root, capture_output
        if args == ("rev-parse", "HEAD"):
            return SHA
        if args[:2] == ("status", "--porcelain=v1"):
            return ""
        return ""

    monkeypatch.setattr(bridge, "_git", fake_git)
    monkeypatch.setattr(
        bridge,
        "_validate_workflow_sha",
        lambda *args, **kwargs: None,
    )

    def reject(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["trusted-validator"])

    monkeypatch.setattr(bridge, "_run", reject)
    with pytest.raises(subprocess.CalledProcessError):
        bridge.validate_trusted_checkout(
            tmp_path,
            request,
            workflow_sha="b" * 40,
        )


def test_duplicate_request_id_is_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    workspace = bridge_root / "100-1"
    workspace.mkdir(parents=True)
    event_path = tmp_path / "event.json"
    _write_event(event_path, _event())
    monkeypatch.setattr(
        bridge,
        "validate_trusted_checkout",
        lambda *args, **kwargs: "ok",
    )
    monkeypatch.setattr(bridge, "_ref_exists", lambda *args, **kwargs: True)
    with pytest.raises(ValueError, match="REVIEWER_BRIDGE_REQUEST_ALREADY_CONSUMED"):
        bridge.reserve_request(
            event_path=event_path,
            repo_root=tmp_path,
            bridge_root=bridge_root,
            workspace=workspace,
            workflow_sha="b" * 40,
            run_id=100,
            run_attempt=1,
            token="not-used",
        )


def test_workflow_has_strict_guards_and_no_marketplace_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "issue_comment:" in text
    assert "github.event.issue.number == 36" in text
    assert "github.actor == 'egorkogor'" in text
    prefix_guard = "startsWith(github.event.comment.body, '/reviewer-bridge/v1 ')"
    assert prefix_guard in text
    assert "uses:" not in text
    assert "actions/checkout" not in text
    assert "actions/upload-artifact" not in text
    assert "workflow_dispatch" not in text
    assert "ssh " not in text.lower()


def test_workflow_bridge_driver_is_bound_to_workflow_sha() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    driver_object = '"${WORKFLOW_SHA}:scripts/run_reviewer_execution_bridge.py"'
    assert driver_object in text
    assert "BRIDGE_DRIVER=%s" in text
    assert 'python "$BRIDGE_DRIVER" reserve' in text
    assert 'python "$BRIDGE_DRIVER" execute' in text
    assert 'python "$BRIDGE_DRIVER" publish' in text
    assert "python -m scripts.run_reviewer_execution_bridge" not in text


def test_evidence_manifest_binds_request_sha_runner_and_bridge_source(
    tmp_path: Path,
) -> None:
    request = bridge.parse_event(_event())
    (tmp_path / "producer-evidence").mkdir()
    (tmp_path / "producer-evidence" / "result.json").write_text(
        '{"ok":true}\n',
        encoding="utf-8",
    )
    runner = {"runner_name": "canonical", "runner_image_id": "image-123"}
    source = {"workflow_sha": "b" * 40, "source_identity": "sha256:source"}
    manifest = bridge.build_evidence_manifest(
        tmp_path,
        request=request,
        runner_identity=runner,
        source_identity=source,
    )
    assert manifest["request_id"] == request.request_id
    assert manifest["implementation_sha"] == SHA
    assert manifest["runner_identity"] == runner
    assert manifest["bridge_source"] == source
    result_digest = manifest["files"]["producer-evidence/result.json"]
    assert result_digest.startswith("sha256:")
    assert manifest["manifest_sha256"].startswith("sha256:")


def test_evidence_archive_is_content_addressed(tmp_path: Path) -> None:
    (tmp_path / "request.json").write_text(
        '{"request":"req-0001"}\n',
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        '{"manifest":"sha256:test"}\n',
        encoding="utf-8",
    )
    digest = bridge._write_deterministic_archive(tmp_path)
    archive_digest = (tmp_path / "archive.sha256").read_text(encoding="utf-8")
    assert digest.startswith("sha256:")
    assert archive_digest == digest + "\n"
    assert bridge._sha256_file(tmp_path / "evidence.tar.gz") == digest


def test_cleanup_cannot_delete_outside_bridge_owned_workspace(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="REVIEWER_BRIDGE_WORKSPACE_OUTSIDE_ROOT"):
        bridge.cleanup_workspace(bridge_root, outside)
    assert outside.is_dir()


def test_cleanup_deletes_only_exact_bridge_owned_run_workspace(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    workspace = bridge_root / "123-1"
    workspace.mkdir(parents=True)
    (workspace / "evidence.txt").write_text("x", encoding="utf-8")
    bridge.cleanup_workspace(bridge_root, workspace)
    assert not workspace.exists()
    assert bridge_root.is_dir()


def test_transport_ref_is_explicit_non_source_orphan_namespace() -> None:
    ref = bridge.transport_ref("req-0001")
    assert ref == "refs/heads/evidence/reviewer-bridge/req-0001"
    assert "EVIDENCE ONLY" in bridge.TRANSPORT_WARNING
    assert "not source" in bridge.TRANSPORT_WARNING
    assert "must never be merged" in bridge.TRANSPORT_WARNING


def test_workflow_materializes_required_fixed_target_runner_image() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "/etc/planning-model-runner-image-id" in text
    assert "REVIEWER_BRIDGE_RUNNER_IMAGE_ID_MISSING_OR_EMPTY" in text
    assert "FIXED_TARGET_RUNNER_IMAGE=%s" in text
    assert "REVIEWER_BRIDGE_FIXED_TARGET_RUNNER_IMAGE_DRIFT" in text


def test_runner_identity_requires_materialized_exact_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "runner-image-id"
    image_path.write_text("image-123\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "RUNNER_IMAGE_ID_PATH", image_path)
    monkeypatch.delenv("FIXED_TARGET_RUNNER_IMAGE", raising=False)
    with pytest.raises(ValueError, match="FIXED_TARGET_RUNNER_IMAGE_MISSING"):
        bridge._runner_identity("canonical")
    monkeypatch.setenv("FIXED_TARGET_RUNNER_IMAGE", "image-drift")
    with pytest.raises(ValueError, match="FIXED_TARGET_RUNNER_IMAGE_DRIFT"):
        bridge._runner_identity("canonical")
    monkeypatch.setenv("FIXED_TARGET_RUNNER_IMAGE", "image-123")
    identity = bridge._runner_identity("canonical")
    assert identity["fixed_target_runner_image"] == "image-123"


@pytest.mark.parametrize(
    ("failed_task", "terminal_status"),
    [("producer", "FAILED"), ("independent-validator", "VALIDATOR_FAILED")],
)
def test_failed_execution_persists_terminal_content_addressed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_task: str,
    terminal_status: str,
) -> None:
    import json
    import sys

    bridge_root = tmp_path / "bridge"
    workspace = bridge_root / "100-1"
    workspace.mkdir(parents=True)
    event_path = tmp_path / "event.json"
    _write_event(event_path, _event())
    image_path = tmp_path / "runner-image-id"
    image_path.write_text("image-123\n", encoding="utf-8")
    monkeypatch.setattr(bridge, "RUNNER_IMAGE_ID_PATH", image_path)
    monkeypatch.setenv("FIXED_TARGET_RUNNER_IMAGE", "image-123")
    monkeypatch.setattr(
        bridge,
        "validate_trusted_checkout",
        lambda *args, **kwargs: '{"trusted":true}',
    )
    monkeypatch.setattr(
        bridge,
        "bridge_source_identity",
        lambda *args, **kwargs: {
            "workflow_sha": "b" * 40,
            "source_identity": "sha256:source",
        },
    )
    fail_script = tmp_path / "synthetic_fail.py"
    fail_script.write_text(
        "import sys\nprint('synthetic stdout')\n"
        "print('synthetic stderr', file=sys.stderr)\nsys.exit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge,
        "task_plan",
        lambda *args, **kwargs: [(failed_task, [sys.executable, str(fail_script)])],
    )

    result = bridge.execute_request(
        event_path=event_path,
        repo_root=tmp_path,
        bridge_root=bridge_root,
        workspace=workspace,
        workflow_sha="b" * 40,
        run_id=100,
        run_attempt=1,
        job_name="reviewer-execution-bridge",
        runner_name="canonical",
    )

    evidence = workspace / "evidence"
    failure = json.loads((evidence / "task-failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((evidence / "status.json").read_text(encoding="utf-8"))
    assert result["valid"] is False
    assert result["terminal_status"] == terminal_status
    assert result["return_code"] == 7
    assert failure["failed_task"] == failed_task
    assert failure["return_code"] == 7
    assert failure["stdout"] == "synthetic stdout\n"
    assert failure["stderr"] == "synthetic stderr\n"
    assert status["valid"] is False
    assert status["terminal_status"] == terminal_status
    assert "task-failure.json" in manifest["files"]
    assert f"{failed_task}.log" in manifest["files"]
    assert manifest["manifest_sha256"].startswith("sha256:")
    assert result["archive_sha256"].startswith("sha256:")
    assert (evidence / "evidence.tar.gz").is_file()
    assert (evidence / "archive.sha256").read_text(encoding="utf-8").strip() == result[
        "archive_sha256"
    ]


def test_workflow_publishes_before_terminal_failure_and_cleanup() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    publish = text.index("Publish terminal evidence to orphan non-source transport ref")
    propagate = text.index("Propagate terminal task failure after evidence publication")
    cleanup = text.index("Cleanup only bridge-owned workspace")
    assert publish < propagate < cleanup
    assert "if: always() && steps.reserve.outcome == 'success'" in text


def test_trusted_validator_scrubs_parent_write_credentials_and_push_auth_is_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = bridge.parse_event(_event())
    sentinels = {
        "GITHUB_TOKEN": "sentinel-github-token",
        "GH_TOKEN": "sentinel-gh-token",
        "GITHUB_PAT": "sentinel-pat",
        "GIT_ASKPASS": "/tmp/sentinel-askpass",
        "GIT_CONFIG_COUNT": "7",
        "GIT_CONFIG_KEY_0": "http.https://sentinel.invalid/.extraheader",
        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: sentinel-old-header",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SAFE_ENV", "kept")

    def fake_git(repo_root: Path, *args: str, capture_output: bool = False) -> str:
        del repo_root, capture_output
        if args == ("rev-parse", "HEAD"):
            return SHA
        if args[:2] == ("status", "--porcelain=v1"):
            return ""
        raise AssertionError(args)

    captured: dict[str, str] = {}

    def fake_run(*args, **kwargs):
        del args
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=["trusted-validator"],
            returncode=0,
            stdout='{"trusted": true}',
            stderr="",
        )

    monkeypatch.setattr(bridge, "_git", fake_git)
    monkeypatch.setattr(
        bridge,
        "_validate_workflow_sha",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(bridge, "_run", fake_run)

    result = bridge.validate_trusted_checkout(
        tmp_path,
        request,
        workflow_sha="b" * 40,
    )
    assert result == '{"trusted": true}'
    assert captured["SAFE_ENV"] == "kept"
    assert not bridge.REPOSITORY_CREDENTIAL_ENV_KEYS.intersection(captured)
    assert not any(key.startswith("GIT_CONFIG_") for key in captured)

    push_env = bridge._git_authenticated_env("intended-push-token")
    assert push_env["SAFE_ENV"] == "kept"
    assert not bridge.REPOSITORY_CREDENTIAL_ENV_KEYS.intersection(push_env)
    assert push_env["GIT_CONFIG_COUNT"] == "1"
    assert push_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert push_env["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")
    for sentinel in sentinels.values():
        assert sentinel not in push_env.values()


def test_allowlisted_repository_tasks_also_receive_scrubbed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "sentinel-github-token")
    monkeypatch.setenv("GH_TOKEN", "sentinel-gh-token")
    monkeypatch.setenv("GITHUB_PAT", "sentinel-pat")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.https://github.com/.extraheader")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "AUTHORIZATION: sentinel")
    monkeypatch.setenv("SAFE_ENV", "kept")
    captured: dict[str, str] = {}

    def fake_subprocess_run(*args, **kwargs):
        del args
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=["producer"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    failure = bridge._execute_plan(
        tmp_path,
        tmp_path,
        [("producer", ["python", "producer.py"])],
    )
    assert failure is None
    assert captured["SAFE_ENV"] == "kept"
    assert not bridge.REPOSITORY_CREDENTIAL_ENV_KEYS.intersection(captured)
    assert not any(key.startswith("GIT_CONFIG_") for key in captured)


def test_real_repository_child_process_cannot_observe_transport_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    sentinels = {
        "GITHUB_TOKEN": "SENTINEL_GITHUB_TOKEN_DO_NOT_LEAK",
        "GH_TOKEN": "SENTINEL_GH_TOKEN_DO_NOT_LEAK",
        "GITHUB_PAT": "SENTINEL_PAT_DO_NOT_LEAK",
        "GIT_ASKPASS": "/tmp/SENTINEL_GIT_ASKPASS_DO_NOT_LEAK",
        "SSH_ASKPASS": "/tmp/SENTINEL_SSH_ASKPASS_DO_NOT_LEAK",
        "GIT_SSH": "/tmp/SENTINEL_GIT_SSH_DO_NOT_LEAK",
        "GIT_SSH_COMMAND": "SENTINEL_GIT_SSH_COMMAND_DO_NOT_LEAK",
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "http.https://sentinel.invalid/.extraheader",
        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: SENTINEL_EXTRAHEADER_DO_NOT_LEAK",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SAFE_ENV", "kept")

    probe = tmp_path / "credential_env_probe.py"
    probe.write_text(
        "import os\n"
        "import sys\n"
        "forbidden = {\n"
        "    'GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_PAT', 'GIT_ASKPASS',\n"
        "    'SSH_ASKPASS', 'GIT_SSH', 'GIT_SSH_COMMAND',\n"
        "}\n"
        "leaked = sorted(\n"
        "    key for key in os.environ\n"
        "    if key in forbidden or key.startswith('GIT_CONFIG_')\n"
        ")\n"
        "if leaked:\n"
        "    print('credential env leaked: ' + ','.join(leaked), file=sys.stderr)\n"
        "    raise SystemExit(91)\n"
        "if os.environ.get('SAFE_ENV') != 'kept':\n"
        "    print('safe runtime env missing', file=sys.stderr)\n"
        "    raise SystemExit(92)\n"
        "print('repository-child-env-sanitized')\n",
        encoding="utf-8",
    )

    failure = bridge._execute_plan(
        tmp_path,
        tmp_path,
        [("producer", [sys.executable, str(probe)])],
    )

    assert failure is None
    assert (tmp_path / "producer.log").read_text(encoding="utf-8") == (
        "repository-child-env-sanitized\n"
    )
