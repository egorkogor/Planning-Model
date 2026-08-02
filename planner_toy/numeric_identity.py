"""Versioned cross-host numeric identity for quality v0.1.

Raw file hashes remain the integrity contract.  This module only defines the
coarser, byte-exact representation used when comparing independent CPU hosts.
"""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
import torch

CANONICAL_NUMERIC_POLICY = {
    "version": "toy-quality-numeric-identity/1.0",
    "encoding": "ieee754-float32-little-endian",
    "quantization": "clear-8-least-significant-mantissa-bits",
    "cleared_mantissa_bits": 8,
    # One quantization cell is at most 2^-15 of a normal value.  This is a
    # representation bound, not a claim that values inside it are equal.
    "relative_cell_bound": 2.0**-15,
    "absolute_zero_cell_bound": 2.0**-141,
}


def canonicalize_float32_bytes(payload: bytes) -> bytes:
    if len(payload) % 4:
        raise ValueError("CANONICAL_NUMERIC_SIZE_INVALID")
    bits = np.frombuffer(payload, dtype="<u4").copy()
    exponent = bits & np.uint32(0x7F800000)
    mantissa = bits & np.uint32(0x007FFFFF)
    if np.any(exponent == np.uint32(0x7F800000)):
        raise ValueError("CANONICAL_NUMERIC_NONFINITE")
    zero_cell = (exponent == 0) & (mantissa < 256)
    bits &= np.uint32(0xFFFFFF00)
    bits[zero_cell] = 0
    return bits.astype("<u4", copy=False).tobytes()


def canonical_float32_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_float32_bytes(payload)).hexdigest()


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    return canonical_float32_sha256(value.numpy().tobytes())


def canonical_state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        source = state[name].detach().cpu().contiguous()
        digest.update(
            name.encode() + b"\0" + str(source.dtype).encode() + b"\0"
            + str(tuple(source.shape)).encode() + b"\0"
        )
        if source.is_floating_point():
            digest.update(canonicalize_float32_bytes(source.to(torch.float32).numpy().tobytes()))
        else:
            digest.update(source.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def canonical_torch_object_sha256(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if torch.is_tensor(item):
            source = item.detach().cpu().contiguous()
            digest.update(
                b"tensor\0" + str(source.dtype).encode() + b"\0"
                + str(tuple(source.shape)).encode() + b"\0"
            )
            if source.is_floating_point():
                digest.update(
                    canonicalize_float32_bytes(source.to(torch.float32).numpy().tobytes())
                )
            else:
                digest.update(source.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(b"key\0" + str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, list | tuple):
            digest.update(f"sequence:{len(item)}\0".encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode() + b"\0")

    visit(value)
    return "sha256:" + digest.hexdigest()


def canonical_norm(payload: bytes) -> float:
    canonical = canonicalize_float32_bytes(payload)
    values = [value for (value,) in struct.iter_unpack("<f", canonical)]
    return float(math.sqrt(sum(value * value for value in values)))
