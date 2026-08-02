from __future__ import annotations

import math
import struct

import pytest

from planner_toy.e2e import PlanParseFailure, parse_work_plan
from planner_toy.numeric_identity import canonical_float32_sha256


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -1.0, 1e-20, 1e20])
def test_low_mantissa_variation_preserves_canonical_hash(value: float) -> None:
    original = bits(value)
    if original & 0x7F800000 == 0x7F800000:
        pytest.fail("fixture must be finite")
    changed = (original & 0xFFFFFF00) | ((original + 1) & 0xFF)
    assert struct.pack("<I", changed) != struct.pack("<I", original)
    assert canonical_float32_sha256(struct.pack("<I", changed)) == (
        canonical_float32_sha256(struct.pack("<I", original))
    )


@pytest.mark.parametrize("direction", [-1, 1])
def test_change_above_quantization_cell_changes_hash(direction: int) -> None:
    original = bits(1.0)
    changed = original + direction * 0x100
    assert canonical_float32_sha256(struct.pack("<I", changed)) != (
        canonical_float32_sha256(struct.pack("<I", original))
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_are_never_canonicalized(value: float) -> None:
    with pytest.raises(ValueError, match="CANONICAL_NUMERIC_NONFINITE"):
        canonical_float32_sha256(struct.pack("<f", value))


@pytest.mark.parametrize(
    "raw,code",
    [
        ([["UNKNOWN", "@B0"]], "PLAN_PARSE_ERROR"),
        ([["PICK_UP", "@UNKNOWN"]], "PLAN_UNKNOWN_REF"),
        ([["PICK_UP"]], "PLAN_PARSE_ERROR"),
        ([["PICK_UP", "@B0"]], "PLAN_NO_END"),
    ],
)
def test_parser_failure_precedence(raw, code) -> None:
    with pytest.raises(PlanParseFailure, match=code):
        parse_work_plan(raw, ["@B0"])
