from __future__ import annotations

import struct
from collections.abc import Iterable


def encode_u32(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 0xFFFFFFFF:
        raise ValueError("value must be an unsigned 32-bit integer")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        output.append(byte)
        if not value:
            return bytes(output)


def _encode_signed(value: int, bits: int) -> bytes:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"value must be a signed {bits}-bit integer")

    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        sign_set = bool(byte & 0x40)
        done = (value == 0 and not sign_set) or (value == -1 and sign_set)
        output.append(byte if done else byte | 0x80)
        if done:
            return bytes(output)


def encode_s32(value: int) -> bytes:
    return _encode_signed(value, 32)


def encode_s64(value: int) -> bytes:
    return _encode_signed(value, 64)


def encode_f32(value: float) -> bytes:
    return struct.pack("<f", value)


def encode_f64(value: float) -> bytes:
    return struct.pack("<d", value)


def encode_name(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return encode_u32(len(encoded)) + encoded


def encode_vector(values: Iterable[bytes]) -> bytes:
    values = tuple(values)
    return encode_u32(len(values)) + b"".join(values)


def encode_section(section_id: int, payload: bytes) -> bytes:
    if not 0 <= section_id <= 12:
        raise ValueError("invalid core WebAssembly section id")
    return bytes((section_id,)) + encode_u32(len(payload)) + payload
