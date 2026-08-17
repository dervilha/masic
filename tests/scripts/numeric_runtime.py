#!/usr/bin/env python3
"""Execute arithmetic for each MASIC scalar family in an optional Wasmtime engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from runtime import load_wasmtime


def build_numeric_cases() -> WebAssembly:
    with WebAssembly("numeric-runtime") as wasm:
        @wasm.func(export=True)
        def signed8_wrap(left: i8, right: i8) -> i8:
            return left + right

        @wasm.func(export=True)
        def unsigned8_wrap(left: u8, right: u8) -> u8:
            return left + right

        @wasm.func(export=True)
        def signed16_divide(left: i16, right: i16) -> i16:
            return left // right

        @wasm.func(export=True)
        def unsigned16_remainder(left: u16, right: u16) -> u16:
            return left % right

        @wasm.func(export=True)
        def signed32_shift(left: i32, right: i32) -> i32:
            return left >> right

        @wasm.func(export=True)
        def unsigned32_shift(left: u32, right: u32) -> u32:
            return left >> right

        @wasm.func(export=True)
        def signed64_multiply(left: i64, right: i64) -> i64:
            return left * right

        @wasm.func(export=True)
        def unsigned64_divide(left: u64, right: u64) -> u64:
            return left // right

        @wasm.func(export=True)
        def float32_add(left: f32, right: f32) -> f32:
            return left + right

        @wasm.func(export=True)
        def float64_divide(left: f64, right: f64) -> f64:
            return left / right

    return wasm.compile()


CASES = (
    ("signed8_wrap", (127, 1), -128),
    ("unsigned8_wrap", (255, 1), 0),
    ("signed16_divide", (-5, 2), -2),
    ("unsigned16_remainder", (10, 3), 1),
    ("signed32_shift", (-8, 1), -4),
    ("unsigned32_shift", (0x80000000, 1), 0x40000000),
    ("signed64_multiply", (3_000_000_000, 3), 9_000_000_000),
    ("unsigned64_divide", (1 << 63, 2), 1 << 62),
    ("float32_add", (1.5, 2.25), 3.75),
    ("float64_divide", (7.5, 2.5), 3.0),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_numeric_cases()
    print(f"compiled {len(wasm.functions)} scalar functions into {len(bytes(wasm))} bytes")
    runtime, reason = load_wasmtime(bytes(wasm))
    if runtime is None:
        print(f"runtime skipped: {reason}")
        return 2 if args.require_engine else 0

    for name, arguments, expected in CASES:
        actual = runtime.invoke(name, *arguments)
        if actual != expected:
            raise AssertionError(f"{name}{arguments}: expected {expected!r}, got {actual!r}")
        print(f"ok {name}{arguments} == {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
