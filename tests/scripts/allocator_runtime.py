#!/usr/bin/env python3
"""Exercise allocator ordering, reuse, splitting, coalescing, and memory growth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, i32, malloc, mfree, u32
from runtime import load_wasmtime


def build_allocator_cases() -> WebAssembly:
    with WebAssembly("allocator-runtime") as wasm:
        @wasm.func(export=True)
        def ordered_load() -> i32:
            values = malloc(1, dtype=i32)
            values[0] = 42
            before = values[0]
            values[0] = 99
            return before

        @wasm.func(export=True)
        def reuse() -> i32:
            first = malloc(4, dtype=i32)
            guard = malloc(1, dtype=i32)
            mfree(first)
            replacement = malloc(4, dtype=i32)
            return replacement.address - first.address

        @wasm.func(export=True)
        def split() -> i32:
            large = malloc(10, dtype=i32)
            guard = malloc(1, dtype=i32)
            mfree(large)
            first = malloc(1, dtype=i32)
            second = malloc(1, dtype=i32)
            return second.address - first.address

        @wasm.func(export=True)
        def coalesce() -> i32:
            first = malloc(1, dtype=i32)
            second = malloc(1, dtype=i32)
            guard = malloc(1, dtype=i32)
            mfree(second)
            mfree(first)
            merged = malloc(5, dtype=i32)
            return merged.address - first.address

        @wasm.func(export=True)
        def grow() -> i32:
            values = malloc(20000, dtype=i32)
            values[19999] = 73
            return values[19999]

        @wasm.func(export=True)
        def dynamic(count: u32) -> i32:
            values = malloc(count, dtype=i32)
            values[0] = 91
            return values[0]

        @wasm.func(export=True)
        def null_allocation() -> u32:
            values = malloc(0, dtype=i32)
            mfree(values)
            return values.address

    return wasm.compile()


CASES = (
    ("ordered_load", (), 42),
    ("reuse", (), 0),
    ("split", (), 16),
    ("coalesce", (), 0),
    ("dynamic", (3,), 91),
    ("null_allocation", (), 0),
    ("grow", (), 73),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_allocator_cases()
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

    result, store, instance = runtime.invoke_details("grow")
    if result != 73:
        raise AssertionError(f"grow() rerun: expected 73, got {result}")
    memory = instance.exports(store)["memory"]
    if memory.data_len(store) <= 65536:
        raise AssertionError("grow() did not expand linear memory")
    print(f"ok grow expanded memory to {memory.data_len(store)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
