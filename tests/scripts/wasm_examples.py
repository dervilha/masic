#!/usr/bin/env python3
"""Compile public-API examples and, when available, execute them in Wasmtime.

Run from the repository root with:
    python tests/scripts/wasm_examples.py
    python tests/scripts/wasm_examples.py --require-engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, block, branch, i32, loop, malloc, mfree
from runtime import load_wasmtime


def build_examples() -> WebAssembly:
    """Use only the public MASIC syntax exercised by this script."""
    with WebAssembly("examples") as wasm:
        @wasm.func(export=True)
        def add(left: i32, right: i32) -> i32:
            return left + right

        @wasm.func()
        def add_default(value: i32, adjustment: i32 = 2) -> i32:
            return value + adjustment

        @wasm.func(export=True)
        def increment(value: i32) -> i32:
            return add_default(value)

        @wasm.func(export=True)
        def absolute(value: i32) -> i32:
            result = value
            with branch(value < 0) as negative:
                result = -value
            with negative.els():
                result = value
            return result

        @wasm.func()
        def square(value: i32) -> i32:
            return value * value

        @wasm.func(export=True)
        def sum_of_squares(left: i32, right: i32) -> i32:
            return square(left) + square(right)

        @wasm.func(export=True)
        def sum_odd_below(limit: i32) -> i32:
            value = i32(0)
            total = i32(0)
            with loop() as repeat:
                with branch(value >= limit):
                    repeat.brk()
                value += 1
                with branch(value % 2 == 0):
                    repeat.cont()
                total += value
            return total

        @wasm.func(export=True)
        def factorial(value: i32) -> i32:
            result = i32(1)
            with loop() as repeat:
                with branch(value <= 1):
                    repeat.brk()
                result *= value
                value -= 1
            return result

        @wasm.func(export=True)
        def block_exit(value: i32) -> i32:
            result = i32(0)
            with block() as stop:
                with branch(value == 0):
                    stop.brk()
                result = 1
            return result

        @wasm.func(export=True)
        def memory_round_trip() -> i32:
            values = malloc(2, dtype=i32)
            values[0] = 42
            before = values[0]
            values[1] = 99
            mfree(values)
            return before

    return wasm.compile()


CASES = (
    ("add", (19, 23), 42),
    ("increment", (40,), 42),
    ("absolute", (-17,), 17),
    ("absolute", (0,), 0),
    ("sum_of_squares", (3, 4), 25),
    ("sum_odd_below", (10,), 25),
    ("factorial", (6,), 720),
    ("block_exit", (0,), 0),
    ("block_exit", (7,), 1),
    ("memory_round_trip", (), 42),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_examples()
    print(f"compiled {len(wasm.functions)} functions into {len(bytes(wasm))} bytes")
    runtime, reason = load_wasmtime(bytes(wasm))
    if runtime is None:
        print(f"runtime skipped: {reason}")
        return 2 if args.require_engine else 0

    for name, arguments, expected in CASES:
        actual = runtime.invoke(name, *arguments)
        if actual != expected:
            raise AssertionError(f"{name}{arguments}: expected {expected}, got {actual}")
        print(f"ok {name}{arguments} == {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
