from __future__ import annotations

import hashlib
import math
import struct

import pytest
import torch

from planner_toy.e2e import PlanParseFailure, parse_work_plan
from planner_toy.numeric_identity import (
    canonical_float32_sha256,
    canonical_state_dict_sha256,
    canonical_tensor_sha256,
    canonical_torch_object_sha256,
    encode_torch_object,
    exact_torch_object_sha256,
)


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


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": {"b": 1}}, {"a": {}, "b": 1}),
        ({1: "value"}, {"1": "value"}),
        ([], ()), ({"a": []}, {"a": ()}),
        ([1, 23], [12, 3]),
        ({"a": {}, "b": {}}, {"a": {"b": {}}}),
        (True, 1), (1, 1.0), (1.0, "1"), ("1", b"1"),
        ({}, []), ([], ()), ({"a": {"b": []}}, {"a": {"b": ()}}),
        (-1, 2**200),
    ],
)
def test_typed_object_encoding_has_no_structural_collisions(left, right) -> None:
    assert exact_torch_object_sha256(left) != exact_torch_object_sha256(right)
    assert canonical_torch_object_sha256(left) != canonical_torch_object_sha256(right)


def test_dictionary_encoding_is_insertion_order_independent() -> None:
    left = {"a": 1, 2: "b"}
    right = {2: "b", "a": 1}
    assert exact_torch_object_sha256(left) == exact_torch_object_sha256(right)
    assert canonical_torch_object_sha256(left) == canonical_torch_object_sha256(right)


def test_streaming_hash_is_hash_of_documented_binary_encoding() -> None:
    value = {"state": {0: [torch.tensor([1.0]), None]}, "groups": (True, b"x")}
    assert exact_torch_object_sha256(value) == (
        "sha256:" + hashlib.sha256(encode_torch_object(value)).hexdigest()
    )
    assert canonical_torch_object_sha256(value) == (
        "sha256:" + hashlib.sha256(encode_torch_object(value, canonical=True)).hexdigest()
    )


def test_tensor_shape_and_dtype_are_identity() -> None:
    flat = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    shaped = flat.reshape(2, 2)
    wider = flat.to(torch.int64)
    assert exact_torch_object_sha256(flat) != exact_torch_object_sha256(shaped)
    assert exact_torch_object_sha256(flat) != exact_torch_object_sha256(wider)
    assert canonical_torch_object_sha256(flat) != canonical_torch_object_sha256(shaped)
    assert canonical_torch_object_sha256(flat) != canonical_torch_object_sha256(wider)


def test_unsupported_object_and_floating_dtype_fail_closed() -> None:
    class Unsupported:
        pass

    with pytest.raises(ValueError, match="QUALITY_HASH_OBJECT_TYPE_UNSUPPORTED"):
        exact_torch_object_sha256(Unsupported())
    with pytest.raises(ValueError, match="QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED"):
        canonical_torch_object_sha256(torch.ones(1, dtype=torch.float64))


def test_optimizer_like_structural_mutation_changes_both_hashes() -> None:
    state = {
        "state": {0: {"step": torch.tensor(9.0), "exp_avg": torch.ones(2),
                      "exp_avg_sq": torch.ones(2)}},
        "param_groups": [{"params": [0], "lr": 0.001}],
    }
    changed = {
        "state": {0: {"step": torch.tensor(9.0), "exp_avg": torch.ones(2)}},
        "exp_avg_sq": torch.ones(2),
        "param_groups": [{"params": [0], "lr": 0.001}],
    }
    assert exact_torch_object_sha256(state) != exact_torch_object_sha256(changed)
    assert canonical_torch_object_sha256(state) != canonical_torch_object_sha256(changed)


@pytest.mark.parametrize("sign", [1, -1])
def test_float32_cell_boundaries_across_identity_layers(sign: int) -> None:
    base_bits = bits(float(sign)) & 0xFFFFFF00
    inside_bits = base_bits | 0xFF
    next_bits = base_bits + (0x100 if sign > 0 else -0x100)
    base = torch.frombuffer(bytearray(struct.pack("<I", base_bits)), dtype=torch.float32).clone()
    inside = torch.frombuffer(
        bytearray(struct.pack("<I", inside_bits)), dtype=torch.float32
    ).clone()
    adjacent = torch.frombuffer(
        bytearray(struct.pack("<I", next_bits)), dtype=torch.float32
    ).clone()
    assert exact_torch_object_sha256(base) != exact_torch_object_sha256(inside)
    assert canonical_tensor_sha256(base) == canonical_tensor_sha256(inside)
    assert canonical_tensor_sha256(base) != canonical_tensor_sha256(adjacent)
    assert canonical_state_dict_sha256({"x": base}) == canonical_state_dict_sha256({"x": inside})
    assert canonical_state_dict_sha256({"x": base}) != canonical_state_dict_sha256({"x": adjacent})
    assert canonical_torch_object_sha256({"state": {0: {"exp_avg": base}}}) == (
        canonical_torch_object_sha256({"state": {0: {"exp_avg": inside}}})
    )


@pytest.mark.parametrize("raw_bits", [0, 0x80000000, 1, 0xFF, 0x7F7FFF00])
def test_float32_zero_subnormal_and_large_cells(raw_bits: int) -> None:
    payload = struct.pack("<I", raw_bits)
    assert canonical_float32_sha256(payload)
