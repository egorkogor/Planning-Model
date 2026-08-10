"""CLI for fixed-target foundation preflight and evidence validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.fixed_target_contract import (
    blocked_acceptance_record,
    build_runtime_contract,
    collect_target_observation,
    require_trusted_implementation_commit,
    runtime_contract_sha256,
    source_inventory_at_commit,
    target_contract_sha256,
    validate_acceptance_bundle,
    validate_acceptance_record,
    validate_runtime_contract,
    validate_target_contract,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def command_blocked(args: argparse.Namespace) -> int:
    _write(args.output, blocked_acceptance_record(args.required_action))
    return 0


def command_trusted_commit(args: argparse.Namespace) -> int:
    commit = require_trusted_implementation_commit(
        args.implementation_commit,
        protected_ref=args.protected_ref,
    )
    print(json.dumps({"trusted": True, "implementation_commit": commit}, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    require_trusted_implementation_commit(
        args.implementation_commit,
        protected_ref=args.protected_ref,
    )
    contract = _read(args.target_contract)
    validate_target_contract(contract)
    runtime = build_runtime_contract(contract)
    validate_runtime_contract(runtime, contract)
    observation = collect_target_observation(contract)
    payload = {
        "implementation_commit": args.implementation_commit,
        "target_contract": contract,
        "target_contract_sha256": target_contract_sha256(contract),
        "runtime_contract": runtime,
        "runtime_contract_sha256": runtime_contract_sha256(runtime),
        "target_observation": observation,
        "source_inventory": source_inventory_at_commit(args.implementation_commit),
    }
    _write(args.output, payload)
    return 0


def command_validate_record(args: argparse.Namespace) -> int:
    value = _read(args.acceptance)
    validate_acceptance_record(value)
    print(
        json.dumps(
            {"valid": True, "accepted": value["accepted"], "status": value["status"]},
            sort_keys=True,
        )
    )
    return 0


def command_validate_bundle(args: argparse.Namespace) -> int:
    result = validate_acceptance_bundle(args.bundle_root)
    print(json.dumps(result, sort_keys=True))
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
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(func=command_preflight)

    validate_record = sub.add_parser("validate-record")
    validate_record.add_argument("--acceptance", type=Path, required=True)
    validate_record.set_defaults(func=command_validate_record)

    validate_bundle = sub.add_parser("validate-bundle")
    validate_bundle.add_argument("--bundle-root", type=Path, required=True)
    validate_bundle.set_defaults(func=command_validate_bundle)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
