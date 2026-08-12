"""CLI for fixed-target preflight, formal packaging, and authoritative validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import scripts.fixed_target_contract as ft
from scripts.fixed_target_acceptance_v1_1 import (
    _prepare_contract_environment,
    _validate_formal_target_contract,
    build_acceptance_record,
    build_formal_provenance,
    derive_formal_attempt_summary,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _attempt_file_hashes(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("FIXED_TARGET_ATTEMPT_SYMLINK_FORBIDDEN")
        if path.is_file() and path.name != "attempt_manifest.json":
            relative = path.relative_to(root).as_posix()
            files[relative] = ft.sha256_bytes(path.read_bytes())
    return files


def _reseal_attempt_manifest(root: Path, attempt_index: int) -> dict[str, Any]:
    manifest = {
        "attempt_manifest_version": ft.ATTEMPT_MANIFEST_VERSION,
        "attempt_index": attempt_index,
        "files": _attempt_file_hashes(root),
        "attempt_manifest_sha256": "",
    }
    manifest["attempt_manifest_sha256"] = ft.attempt_manifest_sha256(manifest)
    _write(root / "attempt_manifest.json", manifest)
    ft.validate_attempt_manifest(root, manifest, attempt_index)
    return manifest


def command_blocked(args: argparse.Namespace) -> int:
    _write(args.output, ft.blocked_acceptance_record(args.required_action))
    return 0


def command_trusted_commit(args: argparse.Namespace) -> int:
    commit = ft.require_trusted_implementation_commit(
        args.implementation_commit,
        protected_ref=args.protected_ref,
    )
    print(json.dumps({"trusted": True, "implementation_commit": commit}, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    ft.require_trusted_implementation_commit(
        args.implementation_commit,
        protected_ref=args.protected_ref,
    )
    contract = _read(args.target_contract)
    ft.validate_target_contract(contract)
    if getattr(args, "formal", False):
        _validate_formal_target_contract(contract)
    runtime = ft.build_runtime_contract(contract)
    ft.validate_runtime_contract(runtime, contract)
    observation = ft.collect_target_observation(contract)
    inventory = (
        ft.sharded_source_inventory_at_commit(args.implementation_commit)
        if getattr(args, "formal", False)
        else ft.source_inventory_at_commit(args.implementation_commit)
    )
    payload = {
        "implementation_commit": args.implementation_commit,
        "target_contract": contract,
        "target_contract_sha256": ft.target_contract_sha256(contract),
        "runtime_contract": runtime,
        "runtime_contract_sha256": ft.runtime_contract_sha256(runtime),
        "target_observation": observation,
        "source_inventory": inventory,
    }
    _write(args.output, payload)
    return 0


def command_package_formal_attempt(args: argparse.Namespace) -> int:
    contract = _read(args.attempt_root / "target-contract.json")
    _validate_formal_target_contract(contract)
    _prepare_contract_environment(contract)
    from scripts.fixed_target_quality_sharded import package_foundation_attempt  # noqa: PLC0415

    package_foundation_attempt(args.attempt_root, args.destination, args.attempt_index)
    provenance = build_formal_provenance(
        args.attempt_root,
        args.attempt_index,
        workflow_run_id=args.workflow_run_id,
        job_id=args.job_id,
        workflow_sha=args.workflow_sha,
    )
    _write(args.destination / "formal-provenance.json", provenance)
    manifest = _reseal_attempt_manifest(args.destination, args.attempt_index)
    print(
        json.dumps(
            {
                "packaged": True,
                "attempt_index": args.attempt_index,
                "attempt_identity_sha256": provenance["attempt_identity_sha256"],
                "formal_provenance_sha256": provenance["formal_provenance_sha256"],
                "attempt_manifest_sha256": manifest["attempt_manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_validate_formal_attempt(args: argparse.Namespace) -> int:
    summary = derive_formal_attempt_summary(args.attempt_root, args.attempt_index)
    print(
        json.dumps(
            {
                "valid": True,
                "attempt_index": args.attempt_index,
                "attempt_identity_sha256": summary["attempt_identity_sha256"],
                "canonical_result_identity": summary["canonical_result_identity"],
                "replay_hash": summary["claim_identities"]["replay_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_build_formal_bundle(args: argparse.Namespace) -> int:
    if (args.bundle_root / "acceptance.json").exists():
        raise ValueError("FIXED_TARGET_ACCEPTANCE_FILE_ALREADY_EXISTS")
    if {path.name for path in args.bundle_root.iterdir()} != {
        "attempt-1",
        "attempt-2",
        "attempt-3",
    }:
        raise ValueError("FIXED_TARGET_BUNDLE_ATTEMPT_COVERAGE_MISMATCH")
    summaries = [
        derive_formal_attempt_summary(args.bundle_root / f"attempt-{index}", index)
        for index in range(1, 4)
    ]
    if any(
        summary["execution_implementation_commit"] != args.implementation_commit
        for summary in summaries
    ):
        raise ValueError("FIXED_TARGET_ACCEPTANCE_IMPLEMENTATION_MISMATCH")
    for summary in summaries:
        if (
            summary["workflow_run_id"] != args.workflow_run_id
            or summary["job_id"] != args.job_id
            or summary["workflow_sha"] != args.workflow_sha
        ):
            raise ValueError("FIXED_TARGET_FORMAL_WORKFLOW_PROVENANCE_MISMATCH")
    acceptance = build_acceptance_record(summaries)
    _write(args.bundle_root / "acceptance.json", acceptance)
    print(
        json.dumps(
            {
                "built": True,
                "acceptance_version": acceptance["acceptance_version"],
                "attempt_count": 3,
                "acceptance_identity": acceptance["acceptance_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_validate_record(args: argparse.Namespace) -> int:
    value = _read(args.acceptance)
    ft.validate_acceptance_record(value)
    print(
        json.dumps(
            {"valid": True, "accepted": value["accepted"], "status": value["status"]},
            sort_keys=True,
        )
    )
    return 0


def command_validate_bundle(args: argparse.Namespace) -> int:
    result = ft.validate_acceptance_bundle(args.bundle_root)
    print(json.dumps(result, sort_keys=True))
    return 0


def command_final_gate(args: argparse.Namespace) -> int:
    # Thin wrapper: the authoritative validator is the only source of the verdict.
    result = ft.validate_acceptance_bundle(args.bundle_root)
    if result.get("accepted") is not True:
        raise ValueError("FIXED_TARGET_FINAL_GATE_NOT_ACCEPTED")
    print(json.dumps({"formal_pass": True, "authoritative_result": result}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)

    blocked = sub.add_parser("blocked-record")
    blocked.add_argument("--required-action", required=True)
    blocked.add_argument("--output", type=Path, required=True)
    blocked.set_defaults(func=command_blocked)

    trusted = sub.add_parser("validate-trusted-commit")
    trusted.add_argument("--implementation-commit", required=True)
    trusted.add_argument("--protected-ref", default="origin/main")
    trusted.set_defaults(func=command_trusted_commit)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--target-contract", type=Path, required=True)
    preflight.add_argument("--implementation-commit", required=True)
    preflight.add_argument("--protected-ref", default="origin/main")
    preflight.add_argument("--formal", action="store_true")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(func=command_preflight)

    package = sub.add_parser("package-formal-attempt")
    package.add_argument("--attempt-root", type=Path, required=True)
    package.add_argument("--destination", type=Path, required=True)
    package.add_argument("--attempt-index", type=int, choices=(1, 2, 3), required=True)
    package.add_argument("--workflow-run-id", type=int, required=True)
    package.add_argument("--job-id", type=int, required=True)
    package.add_argument("--workflow-sha", required=True)
    package.set_defaults(func=command_package_formal_attempt)

    validate_attempt = sub.add_parser("validate-formal-attempt")
    validate_attempt.add_argument("--attempt-root", type=Path, required=True)
    validate_attempt.add_argument("--attempt-index", type=int, choices=(1, 2, 3), required=True)
    validate_attempt.set_defaults(func=command_validate_formal_attempt)

    build_bundle = sub.add_parser("build-formal-bundle")
    build_bundle.add_argument("--bundle-root", type=Path, required=True)
    build_bundle.add_argument("--implementation-commit", required=True)
    build_bundle.add_argument("--workflow-run-id", type=int, required=True)
    build_bundle.add_argument("--job-id", type=int, required=True)
    build_bundle.add_argument("--workflow-sha", required=True)
    build_bundle.set_defaults(func=command_build_formal_bundle)

    validate_record = sub.add_parser("validate-record")
    validate_record.add_argument("--acceptance", type=Path, required=True)
    validate_record.set_defaults(func=command_validate_record)

    validate_bundle = sub.add_parser("validate-bundle")
    validate_bundle.add_argument("--bundle-root", type=Path, required=True)
    validate_bundle.set_defaults(func=command_validate_bundle)

    final_gate = sub.add_parser("final-gate")
    final_gate.add_argument("--bundle-root", type=Path, required=True)
    final_gate.set_defaults(func=command_final_gate)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
