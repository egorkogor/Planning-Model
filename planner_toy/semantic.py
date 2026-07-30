"""Non-normative deterministic semantic targets for toy BlocksWorld only."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

DIMENSION = 384
TARGET_SEED = 17
TARGET_SOURCE = "toy-blocksworld-step-signature-v1 (non-normative)"
TARGET_CONFIG = {
    "schema": "toy-semantic-target-config/1.0",
    "dimension": DIMENSION,
    "seed": TARGET_SEED,
    "fields": ["action", "arg1", "arg2"],
    "normalization": "l2",
}
TARGET_CONFIG_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(TARGET_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def target_for_step(step: list[str]) -> torch.Tensor:
    """Encode only the current typed step; no future state or complete plan is included."""
    signature = json.dumps(step, ensure_ascii=False, separators=(",", ":")).encode()
    seed = hashlib.sha256(b"toy-semantic-v1\0" + signature).digest()
    raw = np.empty(DIMENSION, dtype="<f4")
    for index in range(DIMENSION):
        block = hashlib.sha256(seed + index.to_bytes(4, "big")).digest()
        raw[index] = (int.from_bytes(block[:4], "big") / 2**31) - 1.0
    value = torch.from_numpy(raw.copy())
    return value / torch.linalg.vector_norm(value)


def targets(row: dict) -> torch.Tensor:
    plan = row["oracle_work_plan"]
    values = [target_for_step(step) for step in plan]
    values.extend(torch.zeros(DIMENSION) for _ in range(17 - len(values)))
    return torch.stack(values).unsqueeze(0)
