"""Compatibility entrypoint for the canonical trust-topology validator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from validation.trust_topology_validator import main

    return main


main = _load_main()

if __name__ == "__main__":
    raise SystemExit(main())
