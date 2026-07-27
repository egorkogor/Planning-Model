from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analysis_code_digest() -> str:
    """Hash every normative analysis module and the locked statistics validator."""
    files = sorted(
        list((ROOT / "analysis").glob("*.py")) + [ROOT / "validation/statistics_validator.py"],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    rows: list[bytes] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        rows.append(f"{rel}\0{len(raw)}\0{raw.hex()}\n".encode("utf-8"))
    return "sha256:" + hashlib.sha256(b"".join(rows)).hexdigest()
