"""Versioned, unambiguous numeric and object identities for quality v0.1."""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
import torch

TORCH_OBJECT_ENCODING_VERSION = "toy-quality-torch-object-encoding/2.0"
CANONICAL_NUMERIC_POLICY = {
    "version": "toy-quality-numeric-identity/1.0",
    "encoding": "ieee754-float32-little-endian",
    "quantization": "clear-8-least-significant-mantissa-bits",
    "cleared_mantissa_bits": 8,
    "relative_cell_bound": 2.0**-15,
    "absolute_zero_cell_bound": 2.0**-141,
    "torch_object_encoding_version": TORCH_OBJECT_ENCODING_VERSION,
}
CANONICAL_NUMERIC_POLICY_SCOPE = (
    "Version 1.0 policy selected for the pinned quality-v0.1 runtime and validated "
    "by independent GitHub-hosted workers. Changes to runtime, dtype, training "
    "implementation, or hardware class require explicit policy revalidation or a new "
    "policy version."
)

_TAGS = {
    type(None): b"N", bool: b"B", int: b"I", float: b"F", str: b"S",
    bytes: b"Y", dict: b"D", list: b"L", tuple: b"U",
}
_TENSOR_DTYPES = {
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float32,
}


def _u64(value: int) -> bytes:
    if not 0 <= value < 2**64:
        raise ValueError("QUALITY_HASH_OBJECT_SIZE_INVALID")
    return struct.pack(">Q", value)


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + _u64(len(payload)) + payload


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


def encode_torch_object(value: object, *, canonical: bool = False) -> bytes:
    """Encode supported Python/Torch objects with typed, length-delimited frames."""
    def encode(item: object) -> bytes:
        item_type = type(item)
        if item is None:
            return _frame(_TAGS[item_type], b"")
        if item_type is bool:
            return _frame(_TAGS[bool], b"\x01" if item else b"\x00")
        if item_type is int:
            sign = b"\x01" if item < 0 else b"\x00"
            magnitude = abs(item).to_bytes(max(1, (abs(item).bit_length() + 7) // 8), "big")
            return _frame(_TAGS[int], sign + _u64(len(magnitude)) + magnitude)
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("CANONICAL_NUMERIC_NONFINITE")
            return _frame(_TAGS[float], struct.pack(">d", item))
        if item_type is str:
            return _frame(_TAGS[str], item.encode("utf-8"))
        if item_type is bytes:
            return _frame(_TAGS[bytes], item)
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            if tensor.dtype not in _TENSOR_DTYPES:
                raise ValueError("QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED")
            dtype = str(tensor.dtype).encode("ascii")
            data = tensor.numpy().tobytes()
            if canonical and tensor.is_floating_point():
                if tensor.dtype != torch.float32:
                    raise ValueError("QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED")
                data = canonicalize_float32_bytes(data)
            payload = b"".join([
                _u64(len(dtype)), dtype, _u64(tensor.ndim),
                *(_u64(dimension) for dimension in tensor.shape),
                _u64(len(data)), data,
            ])
            return _frame(b"T", payload)
        if isinstance(item, dict):
            pairs = [(encode(key), encode(child)) for key, child in item.items()]
            pairs.sort(key=lambda pair: pair[0])
            parts = [_u64(len(pairs))]
            for key, child in pairs:
                parts.extend((_u64(len(key)), key, _u64(len(child)), child))
            return _frame(_TAGS[dict], b"".join(parts))
        if item_type in {list, tuple}:
            children = [encode(child) for child in item]
            payload = _u64(len(children)) + b"".join(
                _u64(len(child)) + child for child in children
            )
            return _frame(_TAGS[item_type], payload)
        raise ValueError("QUALITY_HASH_OBJECT_TYPE_UNSUPPORTED")

    header = TORCH_OBJECT_ENCODING_VERSION.encode("ascii")
    return _u64(len(header)) + header + encode(value)


def _hash_torch_object(value: object, *, canonical: bool) -> str:
    """Hash the same encoding without materializing large nested tensor payloads."""
    def tensor_parts(item: torch.Tensor) -> tuple[bytes, bytes, list[bytes]]:
        tensor = item.detach().cpu().contiguous()
        if tensor.dtype not in _TENSOR_DTYPES:
            raise ValueError("QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED")
        dtype = str(tensor.dtype).encode("ascii")
        data = tensor.numpy().tobytes()
        if canonical and tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise ValueError("QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED")
            data = canonicalize_float32_bytes(data)
        metadata = [_u64(len(dtype)), dtype, _u64(tensor.ndim)]
        metadata.extend(_u64(dimension) for dimension in tensor.shape)
        metadata.append(_u64(len(data)))
        return dtype, data, metadata

    def encoded_length(item: object) -> int:
        item_type = type(item)
        if item is None:
            payload_length = 0
        elif item_type is bool:
            payload_length = 1
        elif item_type is int:
            magnitude_length = max(1, (abs(item).bit_length() + 7) // 8)
            payload_length = 1 + 8 + magnitude_length
        elif item_type is float:
            if not math.isfinite(item):
                raise ValueError("CANONICAL_NUMERIC_NONFINITE")
            payload_length = 8
        elif item_type is str:
            payload_length = len(item.encode("utf-8"))
        elif item_type is bytes:
            payload_length = len(item)
        elif torch.is_tensor(item):
            tensor = item.detach().cpu()
            if tensor.dtype not in _TENSOR_DTYPES:
                raise ValueError("QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED")
            if canonical and tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError("QUALITY_HASH_TENSOR_DTYPE_UNSUPPORTED")
            dtype_length = len(str(tensor.dtype).encode("ascii"))
            payload_length = 8 + dtype_length + 8 + 8 * tensor.ndim + 8
            payload_length += tensor.numel() * tensor.element_size()
        elif isinstance(item, dict):
            payload_length = 8 + sum(
                8 + len(encode_torch_object(key, canonical=canonical))
                - (8 + len(TORCH_OBJECT_ENCODING_VERSION))
                + 8 + encoded_length(child)
                for key, child in item.items()
            )
        elif item_type in {list, tuple}:
            payload_length = 8 + sum(8 + encoded_length(child) for child in item)
        else:
            raise ValueError("QUALITY_HASH_OBJECT_TYPE_UNSUPPORTED")
        return 9 + payload_length

    def update(item: object) -> None:
        item_type = type(item)
        if item is None:
            tag, parts = _TAGS[item_type], []
        elif item_type is bool:
            tag, parts = _TAGS[bool], [b"\x01" if item else b"\x00"]
        elif item_type is int:
            magnitude = abs(item).to_bytes(max(1, (abs(item).bit_length() + 7) // 8), "big")
            tag, parts = _TAGS[int], [
                b"\x01" if item < 0 else b"\x00", _u64(len(magnitude)), magnitude
            ]
        elif item_type is float:
            if not math.isfinite(item):
                raise ValueError("CANONICAL_NUMERIC_NONFINITE")
            tag, parts = _TAGS[float], [struct.pack(">d", item)]
        elif item_type is str:
            tag, parts = _TAGS[str], [item.encode("utf-8")]
        elif item_type is bytes:
            tag, parts = _TAGS[bytes], [item]
        elif torch.is_tensor(item):
            _, data, metadata = tensor_parts(item)
            tag, parts = b"T", [*metadata, data]
        elif isinstance(item, dict):
            keys = [(encode_torch_object(key, canonical=canonical), key) for key in item]
            keys.sort(key=lambda pair: pair[0])
            payload_length = encoded_length(item) - 9
            digest.update(_TAGS[dict] + _u64(payload_length) + _u64(len(keys)))
            header_length = 8 + len(TORCH_OBJECT_ENCODING_VERSION)
            for encoded_key, key in keys:
                key_frame = encoded_key[header_length:]
                digest.update(_u64(len(key_frame)) + key_frame)
                digest.update(_u64(encoded_length(item[key])))
                update(item[key])
            return
        elif item_type in {list, tuple}:
            payload_length = encoded_length(item) - 9
            digest.update(_TAGS[item_type] + _u64(payload_length) + _u64(len(item)))
            for child in item:
                digest.update(_u64(encoded_length(child)))
                update(child)
            return
        else:
            raise ValueError("QUALITY_HASH_OBJECT_TYPE_UNSUPPORTED")
        payload_length = sum(map(len, parts))
        digest.update(tag + _u64(payload_length))
        for part in parts:
            digest.update(part)

    digest = hashlib.sha256()
    header = TORCH_OBJECT_ENCODING_VERSION.encode("ascii")
    digest.update(_u64(len(header)) + header)
    update(value)
    return "sha256:" + digest.hexdigest()


def exact_torch_object_sha256(value: object) -> str:
    return _hash_torch_object(value, canonical=False)


def canonical_torch_object_sha256(value: object) -> str:
    return _hash_torch_object(value, canonical=True)


def canonical_float32_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_float32_bytes(payload)).hexdigest()


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    return canonical_torch_object_sha256(tensor)


def canonical_state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    return canonical_torch_object_sha256(state)


def canonical_norm(payload: bytes) -> float:
    canonical = canonicalize_float32_bytes(payload)
    values = [value for (value,) in struct.iter_unpack("<f", canonical)]
    return float(math.sqrt(sum(value * value for value in values)))
