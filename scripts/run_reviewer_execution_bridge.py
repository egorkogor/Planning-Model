"""Guarded issue-comment execution bridge for the canonical CPU runner."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BRIDGE_VERSION = "reviewer-execution-bridge/1.0"
COMMAND_PREFIX = "/reviewer-bridge/v1"
REPOSITORY = "egorkogor/Planning-Model"
OWNER = "egorkogor"
CONTROL_ISSUE = 36
TARGET_CONTRACT = "configs/fixed-cpu-target-1.0.json"
RUNNER_IMAGE_ID_PATH = Path("/etc/planning-model-runner-image-id")
WORKFLOW_PATH = ".github/workflows/reviewer-execution-bridge.yml"
BRIDGE_SOURCE_PATH = "scripts/run_reviewer_execution_bridge.py"
TRANSPORT_REF_PREFIX = "refs/heads/evidence/reviewer-bridge/"
TRANSPORT_WARNING = (
    "EVIDENCE ONLY. This orphan-history ref is not source and must never be merged "
    "as scientific implementation.\n"
)
TASKS = (
    "status-v1",
    "preflight-v1",
    "a2-sufficient-budget-task-order-v1",
)
REQUEST_PATTERN = re.compile(
    r"^/reviewer-bridge/v1 "
    r"task=(?P<task>status-v1|preflight-v1|a2-sufficient-budget-task-order-v1) "
    r"request=(?P<request>[a-z0-9][a-z0-9-]{6,62}[a-z0-9]) "
    r"sha=(?P<sha>[0-9a-f]{40})$"
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
WORKSPACE_NAME_PATTERN = re.compile(r"^[0-9]+-[0-9]+$")
REPOSITORY_CREDENTIAL_ENV_KEYS = frozenset(
    {
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_PAT",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
    }
)


@dataclass(frozen=True)
class BridgeRequest:
    task: str
    request_id: str
    implementation_sha: str
    comment_id: int
    actor: str
    issue_number: int
    repository: str


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repository_subprocess_env(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    for key in tuple(env):
        if key in REPOSITORY_CREDENTIAL_ENV_KEYS or key.startswith("GIT_CONFIG_"):
            env.pop(key, None)
    return env


def _run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    effective_env = _repository_subprocess_env() if env is None else env
    return subprocess.run(
        argv,
        cwd=cwd,
        env=effective_env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def parse_event(event: dict[str, Any]) -> BridgeRequest:
    repository = event.get("repository", {}).get("full_name")
    issue_number = event.get("issue", {}).get("number")
    actor = event.get("sender", {}).get("login")
    comment = event.get("comment", {})
    comment_actor = comment.get("user", {}).get("login")
    body = comment.get("body")
    comment_id = comment.get("id")

    if repository != REPOSITORY:
        raise ValueError("REVIEWER_BRIDGE_REPOSITORY_FORBIDDEN")
    if issue_number != CONTROL_ISSUE:
        raise ValueError("REVIEWER_BRIDGE_CONTROL_ISSUE_MISMATCH")
    if actor != OWNER or comment_actor != OWNER:
        raise ValueError("REVIEWER_BRIDGE_ACTOR_FORBIDDEN")
    if not isinstance(body, str):
        raise ValueError("REVIEWER_BRIDGE_COMMENT_BODY_INVALID")
    match = REQUEST_PATTERN.fullmatch(body)
    if match is None:
        raise ValueError("REVIEWER_BRIDGE_COMMAND_INVALID")
    if not isinstance(comment_id, int) or comment_id < 1:
        raise ValueError("REVIEWER_BRIDGE_COMMENT_ID_INVALID")

    return BridgeRequest(
        task=match.group("task"),
        request_id=match.group("request"),
        implementation_sha=match.group("sha"),
        comment_id=comment_id,
        actor=actor,
        issue_number=issue_number,
        repository=repository,
    )


def parse_event_file(path: Path) -> BridgeRequest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("REVIEWER_BRIDGE_EVENT_INVALID")
    return parse_event(value)


def transport_ref(request_id: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{6,62}[a-z0-9]", request_id) is None:
        raise ValueError("REVIEWER_BRIDGE_REQUEST_ID_INVALID")
    return TRANSPORT_REF_PREFIX + request_id


def assert_owned_workspace(bridge_root: Path, workspace: Path) -> tuple[Path, Path]:
    root = bridge_root.resolve(strict=False)
    work = workspace.resolve(strict=False)
    if work.parent != root:
        raise ValueError("REVIEWER_BRIDGE_WORKSPACE_OUTSIDE_ROOT")
    if WORKSPACE_NAME_PATTERN.fullmatch(work.name) is None:
        raise ValueError("REVIEWER_BRIDGE_WORKSPACE_NAME_INVALID")
    return root, work


def _git(repo_root: Path, *args: str, capture_output: bool = False) -> str:
    result = _run(
        ["git", *args],
        cwd=repo_root,
        capture_output=capture_output,
    )
    return result.stdout.strip() if capture_output else ""


def _git_authenticated_env(token: str) -> dict[str, str]:
    if not token:
        raise ValueError("REVIEWER_BRIDGE_GITHUB_TOKEN_MISSING")
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = _repository_subprocess_env()
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {encoded}"
    return env


def _validate_workflow_sha(repo_root: Path, workflow_sha: str) -> None:
    if SHA_PATTERN.fullmatch(workflow_sha) is None:
        raise ValueError("REVIEWER_BRIDGE_WORKFLOW_SHA_INVALID")
    _git(repo_root, "cat-file", "-e", f"{workflow_sha}^{{commit}}")
    _git(repo_root, "merge-base", "--is-ancestor", workflow_sha, "origin/main")
    for path in (WORKFLOW_PATH, BRIDGE_SOURCE_PATH):
        _git(repo_root, "cat-file", "-e", f"{workflow_sha}:{path}")


def validate_trusted_checkout(
    repo_root: Path,
    request: BridgeRequest,
    *,
    workflow_sha: str,
) -> str:
    head = _git(repo_root, "rev-parse", "HEAD", capture_output=True)
    if head != request.implementation_sha:
        raise ValueError("REVIEWER_BRIDGE_IMPLEMENTATION_CHECKOUT_MISMATCH")
    dirty = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        capture_output=True,
    )
    if dirty:
        raise ValueError("REVIEWER_BRIDGE_IMPLEMENTATION_DIRTY_TREE")
    _validate_workflow_sha(repo_root, workflow_sha)
    result = _run(
        [
            sys.executable,
            "-m",
            "scripts.run_fixed_target_acceptance",
            "validate-trusted-commit",
            "--implementation-commit",
            request.implementation_sha,
            "--protected-ref",
            "origin/main",
        ],
        cwd=repo_root,
        env=_repository_subprocess_env(),
        capture_output=True,
    )
    return result.stdout.strip()


def _runner_identity(runner_name: str) -> dict[str, Any]:
    if not RUNNER_IMAGE_ID_PATH.is_file():
        raise ValueError("REVIEWER_BRIDGE_RUNNER_IMAGE_ID_MISSING")
    image_id = RUNNER_IMAGE_ID_PATH.read_text(encoding="utf-8").strip()
    if not image_id:
        raise ValueError("REVIEWER_BRIDGE_RUNNER_IMAGE_ID_EMPTY")
    materialized = os.environ.get("FIXED_TARGET_RUNNER_IMAGE", "").strip()
    if not materialized:
        raise ValueError("REVIEWER_BRIDGE_FIXED_TARGET_RUNNER_IMAGE_MISSING")
    if materialized != image_id:
        raise ValueError("REVIEWER_BRIDGE_FIXED_TARGET_RUNNER_IMAGE_DRIFT")
    if not runner_name:
        raise ValueError("REVIEWER_BRIDGE_RUNNER_NAME_EMPTY")
    return {
        "runner_name": runner_name,
        "runner_image_id": image_id,
        "runner_image_id_path": str(RUNNER_IMAGE_ID_PATH),
        "fixed_target_runner_image": materialized,
    }


def _git_show_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        env=_repository_subprocess_env(),
        check=True,
        capture_output=True,
    )
    return result.stdout


def bridge_source_identity(repo_root: Path, workflow_sha: str) -> dict[str, Any]:
    files = []
    for path in (WORKFLOW_PATH, BRIDGE_SOURCE_PATH):
        data = _git_show_bytes(repo_root, workflow_sha, path)
        files.append({"path": path, "sha256": _sha256_bytes(data)})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "workflow_sha": workflow_sha,
        "source_files": files,
        "source_identity": _sha256_bytes(canonical),
    }


def _preflight_argv(request: BridgeRequest, evidence_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.run_fixed_target_acceptance",
        "preflight",
        "--target-contract",
        TARGET_CONTRACT,
        "--implementation-commit",
        request.implementation_sha,
        "--protected-ref",
        "origin/main",
        "--output",
        str(evidence_root / "fixed-target-preflight.json"),
    ]


def task_plan(request: BridgeRequest, evidence_root: Path) -> list[tuple[str, list[str]]]:
    if request.task == "status-v1":
        return []
    if request.task == "preflight-v1":
        return [("fixed-target-preflight", _preflight_argv(request, evidence_root))]
    if request.task == "a2-sufficient-budget-task-order-v1":
        experiment = evidence_root / "producer-evidence"
        return [
            ("fixed-target-preflight", _preflight_argv(request, evidence_root)),
            (
                "producer",
                [
                    sys.executable,
                    "-m",
                    "scripts.run_a2_sufficient_budget_task_order",
                    "--output-dir",
                    str(experiment),
                    "--implementation-commit",
                    request.implementation_sha,
                ],
            ),
            (
                "independent-validator",
                [
                    sys.executable,
                    "-m",
                    "scripts.run_a2_sufficient_budget_task_order",
                    "--output-dir",
                    str(experiment),
                    "--implementation-commit",
                    request.implementation_sha,
                    "--validate-only",
                ],
            ),
        ]
    raise ValueError("REVIEWER_BRIDGE_TASK_NOT_ALLOWLISTED")


def _write_command_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    path.write_text(output, encoding="utf-8")


def _execute_plan(
    repo_root: Path,
    evidence_root: Path,
    plan: list[tuple[str, list[str]]],
) -> dict[str, Any] | None:
    for name, argv in plan:
        completed = subprocess.run(
            argv,
            cwd=repo_root,
            env=_repository_subprocess_env(),
            check=False,
            text=True,
            capture_output=True,
        )
        _write_command_log(evidence_root / f"{name}.log", completed)
        if completed.returncode != 0:
            terminal_status = (
                "VALIDATOR_FAILED" if name == "independent-validator" else "FAILED"
            )
            failure = {
                "terminal_status": terminal_status,
                "failed_task": name,
                "return_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            _json_dump(evidence_root / "task-failure.json", failure)
            return failure
    return None


def _evidence_files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("REVIEWER_BRIDGE_EVIDENCE_SYMLINK_FORBIDDEN")
        if path.is_file() and path.name not in {"evidence.tar.gz", "archive.sha256"}:
            result.append(path)
    return result


def build_evidence_manifest(
    evidence_root: Path,
    *,
    request: BridgeRequest,
    runner_identity: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    files = {
        path.relative_to(evidence_root).as_posix(): _sha256_file(path)
        for path in _evidence_files(evidence_root)
        if path.name != "manifest.json"
    }
    manifest = {
        "version": BRIDGE_VERSION,
        "request_id": request.request_id,
        "task": request.task,
        "implementation_sha": request.implementation_sha,
        "runner_identity": runner_identity,
        "bridge_source": source_identity,
        "files": files,
    }
    unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = _sha256_bytes(unsigned)
    return manifest


def _write_deterministic_archive(evidence_root: Path) -> str:
    archive_path = evidence_root / "evidence.tar.gz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for path in _evidence_files(evidence_root):
                relative = path.relative_to(evidence_root).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    archive_path.write_bytes(buffer.getvalue())
    digest = _sha256_file(archive_path)
    (evidence_root / "archive.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def _terminal_status_payload(
    *,
    request: BridgeRequest,
    terminal_status: str,
    failure: dict[str, Any] | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": BRIDGE_VERSION,
        "task": request.task,
        "implementation_sha": request.implementation_sha,
        "terminal_status": terminal_status,
        "valid": terminal_status == "SUCCEEDED",
        "go_latent": "NOT EVALUATED",
        "science_policy": "NO_RERUN_PERMISSION_GRANTED",
    }
    if failure is not None:
        value["failure"] = failure
    return value


def execute_request(
    *,
    event_path: Path,
    repo_root: Path,
    bridge_root: Path,
    workspace: Path,
    workflow_sha: str,
    run_id: int,
    run_attempt: int,
    job_name: str,
    runner_name: str,
) -> dict[str, Any]:
    _, work = assert_owned_workspace(bridge_root, workspace)
    if run_attempt != 1:
        raise ValueError("REVIEWER_BRIDGE_WORKFLOW_RERUN_FORBIDDEN")
    request = parse_event_file(event_path)
    evidence_root = work / "evidence"
    if evidence_root.exists():
        raise ValueError("REVIEWER_BRIDGE_EVIDENCE_REUSE_FORBIDDEN")
    evidence_root.mkdir(parents=True)

    _json_dump(evidence_root / "request.json", asdict(request))
    _json_dump(
        evidence_root / "github-execution.json",
        {
            "event_name": "issue_comment",
            "run_id": run_id,
            "run_attempt": run_attempt,
            "job_name": job_name,
            "workflow_sha": workflow_sha,
            "comment_id": request.comment_id,
            "control_issue": request.issue_number,
        },
    )

    runner_identity: dict[str, Any] = {
        "available": False,
        "runner_name": runner_name,
        "runner_image_id_path": str(RUNNER_IMAGE_ID_PATH),
    }
    source_identity: dict[str, Any] = {
        "available": False,
        "workflow_sha": workflow_sha,
    }
    failure: dict[str, Any] | None = None
    trust_result = ""

    try:
        trust_result = validate_trusted_checkout(
            repo_root,
            request,
            workflow_sha=workflow_sha,
        )
        runner_identity = _runner_identity(runner_name)
        source_identity = bridge_source_identity(repo_root, workflow_sha)
        _json_dump(evidence_root / "runner-identity.json", runner_identity)
        _json_dump(evidence_root / "bridge-source.json", source_identity)
        (evidence_root / "trusted-commit-validator.jsonl").write_text(
            trust_result + "\n", encoding="utf-8"
        )
        _json_dump(
            evidence_root / "status.json",
            _terminal_status_payload(
                request=request,
                terminal_status="RUNNING",
                failure=None,
            ),
        )

        plan = task_plan(request, evidence_root)
        failure = _execute_plan(repo_root, evidence_root, plan)
        if failure is None:
            dirty = _git(
                repo_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                capture_output=True,
            )
            if dirty:
                failure = {
                    "terminal_status": "FAILED",
                    "failed_task": "post-task-clean-tree",
                    "return_code": None,
                    "stdout": "",
                    "stderr": "REVIEWER_BRIDGE_IMPLEMENTATION_DIRTY_AFTER_TASK",
                }
                _json_dump(evidence_root / "task-failure.json", failure)
    except Exception as exc:
        failure = {
            "terminal_status": "FAILED",
            "failed_task": "bridge-infrastructure",
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
            "exception_type": type(exc).__name__,
        }
        _json_dump(evidence_root / "task-failure.json", failure)
        if trust_result:
            (evidence_root / "trusted-commit-validator.jsonl").write_text(
                trust_result + "\n", encoding="utf-8"
            )
        _json_dump(evidence_root / "runner-identity.json", runner_identity)
        _json_dump(evidence_root / "bridge-source.json", source_identity)

    terminal_status = "SUCCEEDED" if failure is None else failure["terminal_status"]
    _json_dump(
        evidence_root / "status.json",
        _terminal_status_payload(
            request=request,
            terminal_status=terminal_status,
            failure=failure,
        ),
    )

    manifest = build_evidence_manifest(
        evidence_root,
        request=request,
        runner_identity=runner_identity,
        source_identity=source_identity,
    )
    _json_dump(evidence_root / "manifest.json", manifest)
    archive_sha256 = _write_deterministic_archive(evidence_root)
    result = {
        "valid": failure is None,
        "terminal_status": terminal_status,
        "task": request.task,
        "request_id": request.request_id,
        "implementation_sha": request.implementation_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_sha256": archive_sha256,
        "transport_ref": transport_ref(request.request_id),
    }
    if failure is not None:
        result["failed_task"] = failure["failed_task"]
        result["return_code"] = failure["return_code"]
    _json_dump(evidence_root / "result.json", result)
    return result


def _remote_url(repo_root: Path) -> str:
    value = _git(repo_root, "remote", "get-url", "origin", capture_output=True)
    expected = f"https://github.com/{REPOSITORY}.git"
    if value not in {expected, f"https://github.com/{REPOSITORY}"}:
        raise ValueError("REVIEWER_BRIDGE_ORIGIN_URL_INVALID")
    return expected


def _ref_exists(repo_root: Path, ref: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", ref],
        cwd=repo_root,
        env=_repository_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 2:
        return False
    raise RuntimeError("REVIEWER_BRIDGE_REMOTE_REF_CHECK_FAILED")


def reserve_request(
    *,
    event_path: Path,
    repo_root: Path,
    bridge_root: Path,
    workspace: Path,
    workflow_sha: str,
    run_id: int,
    run_attempt: int,
    token: str,
) -> str:
    _, work = assert_owned_workspace(bridge_root, workspace)
    if run_attempt != 1:
        raise ValueError("REVIEWER_BRIDGE_WORKFLOW_RERUN_FORBIDDEN")
    request = parse_event_file(event_path)
    validate_trusted_checkout(repo_root, request, workflow_sha=workflow_sha)
    ref = transport_ref(request.request_id)
    if _ref_exists(repo_root, ref):
        raise ValueError("REVIEWER_BRIDGE_REQUEST_ALREADY_CONSUMED")

    transport = work / "transport-reservation"
    if transport.exists():
        raise ValueError("REVIEWER_BRIDGE_TRANSPORT_REUSE_FORBIDDEN")
    transport.mkdir()
    _run(["git", "init"], cwd=transport)
    _run(["git", "config", "user.name", "planning-model-reviewer-bridge"], cwd=transport)
    _run(
        ["git", "config", "user.email", "reviewer-bridge@users.noreply.github.com"],
        cwd=transport,
    )
    _run(["git", "remote", "add", "origin", _remote_url(repo_root)], cwd=transport)
    (transport / "EVIDENCE_ONLY_DO_NOT_MERGE.md").write_text(
        TRANSPORT_WARNING, encoding="utf-8"
    )
    _json_dump(
        transport / "reservation.json",
        {
            "version": BRIDGE_VERSION,
            "status": "RESERVED",
            "request": asdict(request),
            "run_id": run_id,
            "run_attempt": run_attempt,
            "workflow_sha": workflow_sha,
            "transport_ref": ref,
            "source_branch": False,
            "merge_forbidden": True,
        },
    )
    _run(["git", "add", "--all"], cwd=transport)
    _run(
        ["git", "commit", "-m", "Reviewer bridge request reservation [skip ci]"],
        cwd=transport,
    )
    env = _git_authenticated_env(token)
    completed = subprocess.run(
        ["git", "push", "origin", f"HEAD:{ref}"],
        cwd=transport,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("REVIEWER_BRIDGE_REQUEST_RESERVATION_FAILED")
    return ref


def publish_evidence(
    *,
    event_path: Path,
    repo_root: Path,
    bridge_root: Path,
    workspace: Path,
    run_id: int,
    token: str,
) -> str:
    _, work = assert_owned_workspace(bridge_root, workspace)
    request = parse_event_file(event_path)
    evidence = work / "evidence"
    if not evidence.is_dir():
        raise ValueError("REVIEWER_BRIDGE_EVIDENCE_MISSING")
    ref = transport_ref(request.request_id)

    transport = work / "transport-publish"
    if transport.exists():
        raise ValueError("REVIEWER_BRIDGE_TRANSPORT_REUSE_FORBIDDEN")
    transport.mkdir()
    _run(["git", "init"], cwd=transport)
    _run(["git", "config", "user.name", "planning-model-reviewer-bridge"], cwd=transport)
    _run(
        ["git", "config", "user.email", "reviewer-bridge@users.noreply.github.com"],
        cwd=transport,
    )
    _run(["git", "remote", "add", "origin", _remote_url(repo_root)], cwd=transport)
    _run(["git", "fetch", "--no-tags", "origin", ref], cwd=transport)
    _run(["git", "checkout", "-b", "transport", "FETCH_HEAD"], cwd=transport)
    reservation = json.loads((transport / "reservation.json").read_text(encoding="utf-8"))
    if reservation.get("request", {}).get("request_id") != request.request_id:
        raise ValueError("REVIEWER_BRIDGE_RESERVATION_REQUEST_MISMATCH")
    if reservation.get("run_id") != run_id:
        raise ValueError("REVIEWER_BRIDGE_RESERVATION_RUN_MISMATCH")
    if reservation.get("merge_forbidden") is not True:
        raise ValueError("REVIEWER_BRIDGE_TRANSPORT_NOT_MARKED_NON_SOURCE")

    destination = transport / "evidence"
    shutil.copytree(evidence, destination)
    _json_dump(
        transport / "transport.json",
        {
            "version": BRIDGE_VERSION,
            "transport_ref": ref,
            "request_id": request.request_id,
            "run_id": run_id,
            "source_branch": False,
            "orphan_history": True,
            "merge_forbidden": True,
        },
    )
    _run(["git", "add", "--all"], cwd=transport)
    _run(["git", "commit", "-m", "Reviewer bridge evidence [skip ci]"], cwd=transport)
    env = _git_authenticated_env(token)
    _run(["git", "push", "origin", f"HEAD:{ref}"], cwd=transport, env=env)
    return ref


def cleanup_workspace(bridge_root: Path, workspace: Path) -> None:
    _, work = assert_owned_workspace(bridge_root, workspace)
    if work.exists():
        shutil.rmtree(work)


def _common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse-request")
    parse.add_argument("--event", type=Path, required=True)

    reserve = sub.add_parser("reserve")
    _common_runtime_args(reserve)
    reserve.add_argument("--workflow-sha", required=True)
    reserve.add_argument("--run-id", type=int, required=True)
    reserve.add_argument("--run-attempt", type=int, required=True)

    execute = sub.add_parser("execute")
    _common_runtime_args(execute)
    execute.add_argument("--workflow-sha", required=True)
    execute.add_argument("--run-id", type=int, required=True)
    execute.add_argument("--run-attempt", type=int, required=True)
    execute.add_argument("--job-name", required=True)
    execute.add_argument("--runner-name", required=True)

    publish = sub.add_parser("publish")
    _common_runtime_args(publish)
    publish.add_argument("--run-id", type=int, required=True)

    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--bridge-root", type=Path, required=True)
    cleanup.add_argument("--workspace", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "parse-request":
        print(json.dumps(asdict(parse_event_file(args.event)), sort_keys=True))
        return 0
    if args.command == "reserve":
        ref = reserve_request(
            event_path=args.event,
            repo_root=args.repo_root,
            bridge_root=args.bridge_root,
            workspace=args.workspace,
            workflow_sha=args.workflow_sha,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            token=os.environ.get("GITHUB_TOKEN", ""),
        )
        print(json.dumps({"reserved": True, "transport_ref": ref}, sort_keys=True))
        return 0
    if args.command == "execute":
        value = execute_request(
            event_path=args.event,
            repo_root=args.repo_root,
            bridge_root=args.bridge_root,
            workspace=args.workspace,
            workflow_sha=args.workflow_sha,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            job_name=args.job_name,
            runner_name=args.runner_name,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "publish":
        ref = publish_evidence(
            event_path=args.event,
            repo_root=args.repo_root,
            bridge_root=args.bridge_root,
            workspace=args.workspace,
            run_id=args.run_id,
            token=os.environ.get("GITHUB_TOKEN", ""),
        )
        print(json.dumps({"published": True, "transport_ref": ref}, sort_keys=True))
        return 0
    if args.command == "cleanup":
        cleanup_workspace(args.bridge_root, args.workspace)
        print(json.dumps({"cleaned": True}, sort_keys=True))
        return 0
    raise ValueError("REVIEWER_BRIDGE_COMMAND_NOT_IMPLEMENTED")


if __name__ == "__main__":
    raise SystemExit(main())
