"""Exact update-level probe for canonical A2 CPU investigations.

Process-start controls are validated before PyTorch is imported. The retained
profiles reproduce historical defaults or explicit investigation alternatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.canonical_training_probe_contract import (
    COMPARISON_VERSION,
    EXECUTION_CONTRACT_VERSION,
    PARITY_VERSION,
    PROBE_VERSION,
    _PROFILE_SPECS,
    _canonical_bytes,
    _parse_bool,
    compute_evidence_identity,
    compute_probe_identity,
    validate_execution_contract,
    validate_probe_artifact,
    validate_probe_identity,
)
from scripts.canonical_training_probe_core import compare_probes, run_probe
from scripts.canonical_training_probe_parity import run_quality_training_parity

__all__ = [
    "COMPARISON_VERSION",
    "EXECUTION_CONTRACT_VERSION",
    "PARITY_VERSION",
    "PROBE_VERSION",
    "compare_probes",
    "compute_evidence_identity",
    "compute_probe_identity",
    "run_probe",
    "run_quality_training_parity",
    "validate_execution_contract",
    "validate_probe_artifact",
    "validate_probe_identity",
]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--profile", choices=tuple(_PROFILE_SPECS), required=True)
    run_parser.add_argument("--optimizer-foreach", type=_parse_bool)
    run_parser.add_argument("--optimizer-fused", type=_parse_bool)
    run_parser.add_argument("--seed", type=int, default=17)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--expect", choices=("equal", "different", "incomparable")
    )
    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--seed", type=int, default=17)
    parity_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_probe(
            profile=args.profile,
            seed=args.seed,
            optimizer_foreach=args.optimizer_foreach,
            optimizer_fused=args.optimizer_fused,
        )
        output = args.output
    elif args.command == "parity":
        result = run_quality_training_parity(seed=args.seed)
        output = args.output
        if result["equal"] is not True:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(_canonical_bytes(result) + b"\n")
            raise SystemExit("quality training parity failed")
    else:
        result = compare_probes(_read_json(args.left), _read_json(args.right))
        output = args.output
        if args.expect is not None:
            observed = (
                "incomparable"
                if result["comparable"] is False
                else "equal"
                if result["equal"] is True
                else "different"
            )
            if observed != args.expect:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(_canonical_bytes(result) + b"\n")
                raise SystemExit(
                    "probe comparison expectation failed:"
                    f"expected={args.expect}:actual={observed}"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(result) + b"\n")


if __name__ == "__main__":
    main()
