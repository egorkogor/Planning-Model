"""Run the non-confirmatory deterministic toy A2 vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from planner_toy.e2e import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/toy-a2-run"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
