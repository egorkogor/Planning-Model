"""CLI for fixed-target preflight, blocked evidence, and acceptance validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.fixed_target_contract import (
    blocked_acceptance_record,
    build_runtime_contract,
    collect_target_observation,
    runtime_contract_sha256,
    source_inventory_at_commit,
    target_contract_sha256,
    validate_acceptance,
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


def command_preflight(args: argparse.Namespace) -> int:
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


def command_validate(args: argparse.Namespace) -> int:
    value = _read(args.acceptance)
    validate_acceptance(value)
    print(
        json.dumps(
            {"valid": True, "accepted": value["accepted"], "status": value["status"]},
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)

    blocked = sub.add_parser("blocked-record")
    blocked.add_argument("--required-action", required=True)
    blocked.add_argument("--output", type=Path, required=True)
    blocked.set_defaults(func=command_blocked)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--target-contract", type=Path, required=True)
    preflight.add_argument("--implementation-commit", required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(func=command_preflight)

    validate = sub.add_parser("validate-acceptance")
    validate.add_argument("--acceptance", type=Path, required=True)
    validate.set_defaults(func=command_validate)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
