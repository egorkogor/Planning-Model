"""CLI for runtime/1.1 sharded fixed-target quality execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-attempt")
    init.add_argument("--attempt-root", type=Path, required=True)
    init.add_argument("--target-contract", type=Path, required=True)
    init.add_argument("--nonce")
    init.add_argument("--execution-implementation-commit", required=True)
    init.add_argument(
        "--execution-context",
        choices=("formal-fixed-target", "qualification-only"),
        required=True,
    )
    unit = sub.add_parser("run-unit")
    unit.add_argument("--attempt-root", type=Path, required=True)
    unit.add_argument("--variant", required=True)
    unit.add_argument("--seed", type=int, required=True)
    verify = sub.add_parser("verify-unit")
    verify.add_argument("--attempt-root", type=Path, required=True)
    verify.add_argument("--variant", required=True)
    verify.add_argument("--seed", type=int, required=True)
    finish = sub.add_parser("assemble")
    finish.add_argument("--attempt-root", type=Path, required=True)
    finish.add_argument("--qualification-receipts", action="store_true")
    package = sub.add_parser("package-foundation-attempt")
    package.add_argument("--attempt-root", type=Path, required=True)
    package.add_argument("--destination", type=Path, required=True)
    package.add_argument("--attempt-index", type=int, required=True)
    args = parser.parse_args()
    contract_path = (
        args.target_contract
        if args.command == "init-attempt"
        else args.attempt_root / "target-contract.json"
    )
    contract = json.loads(contract_path.read_text())
    for name in (
        "ATEN_CPU_CAPABILITY",
        "MKL_CBWR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(contract[name])
    from scripts.fixed_target_contract import prepare_process_environment  # noqa: PLC0415

    prepare_process_environment(contract)
    # Importing this module imports torch; process controls must already be set.
    from scripts.fixed_target_quality_sharded import (  # noqa: PLC0415
        assemble,
        collect_runtime11_observation,
        initialize_attempt,
        package_foundation_attempt,
        run_unit,
        verify_qualification_unit,
    )

    if args.command == "init-attempt":
        execution_context = args.execution_context
    else:
        execution_context = json.loads((args.attempt_root / "attempt-id.json").read_text())[
            "execution_context"
        ]
    observation = collect_runtime11_observation(contract, execution_context)

    if args.command == "init-attempt":
        result = initialize_attempt(
            args.attempt_root,
            contract,
            args.execution_implementation_commit,
            execution_context,
            args.nonce,
            observation,
        )
    elif args.command == "run-unit":
        result = run_unit(args.attempt_root, args.variant, args.seed, observation)
    elif args.command == "verify-unit":
        result = verify_qualification_unit(args.attempt_root, args.variant, args.seed)
    elif args.command == "assemble":
        result = assemble(
            args.attempt_root, qualification_receipts=args.qualification_receipts
        )
    else:
        result = package_foundation_attempt(args.attempt_root, args.destination, args.attempt_index)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
