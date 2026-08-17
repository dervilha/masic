import math
import struct
import unittest

from masic.wasm.encoding import (
    encode_f32,
    encode_f64,
    encode_name,
    encode_s32,
    encode_s64,
    encode_section,
    encode_u32,
    encode_vector,
)


class UnsignedLeb128Tests(unittest.TestCase):
    def test_known_values(self):
        cases = {
            0: "00",
            1: "01",
            127: "7f",
            128: "8001",
            624485: "e58e26",
            0xFFFFFFFF: "ffffffff0f",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(encode_u32(value).hex(), expected)

    def test_rejects_invalid_values(self):
        for value in (-1, 0x100000000, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_u32(value)


class SignedLeb128Tests(unittest.TestCase):
    def test_s32_known_values(self):
        cases = {0: "00", -1: "7f", 63: "3f", 64: "c000", -64: "40", -65: "bf7f", -624485: "9bf159"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(encode_s32(value).hex(), expected)

    def test_s64_boundaries(self):
        self.assertEqual(encode_s64(-(1 << 63)), bytes.fromhex("8080808080808080807f"))
        self.assertEqual(encode_s64((1 << 63) - 1), bytes.fromhex("ffffffffffffffffff00"))

    def test_rejects_out_of_range_values(self):
        for encoder, value in ((encode_s32, 1 << 31), (encode_s32, -(1 << 31) - 1), (encode_s64, 1 << 63)):
            with self.subTest(encoder=encoder.__name__, value=value), self.assertRaises(ValueError):
                encoder(value)


class BinaryEncodingTests(unittest.TestCase):
    def test_float_encoding_is_little_endian_ieee754(self):
        self.assertEqual(encode_f32(1.5), struct.pack("<f", 1.5))
        self.assertEqual(encode_f64(math.pi), struct.pack("<d", math.pi))

    def test_name_vector_and_section(self):
        self.assertEqual(encode_name("λ"), b"\x02\xce\xbb")
        self.assertEqual(encode_vector((b"a", b"bc")), b"\x02abc")
        self.assertEqual(encode_section(1, b"abc"), b"\x01\x03abc")

