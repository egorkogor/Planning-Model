from __future__ import annotations

import hashlib
import struct

DOMAIN = b"planner-init-v1\x00"

def canonical_tensor_seed(run_seed: int, canonical_parameter_name: str) -> int:
    if run_seed < 0 or run_seed >= 2**64:
        raise ValueError("run_seed must fit uint64")
    if not canonical_parameter_name:
        raise ValueError("canonical_parameter_name must be non-empty")
    payload = DOMAIN + struct.pack(">Q", run_seed) + b"\x00" + canonical_parameter_name.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
