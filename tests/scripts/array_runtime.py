#!/usr/bin/env python3
"""Execute stdlib.Array reserve, push, relocation, and typed reads in Wasmtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, i32
from masic.wasm import stdlib
from runtime import load_wasmtime


def build_array_cases() -> WebAssembly:
    with WebAssembly("array-runtime") as module:
        std = module.include(stdlib)

        @module.func(export=True)
        def reserve_preserves(value: i32) -> i32:
            items = std.Array[i32](capacity=1)
            items.push(10)
            items.reserve(4)
            items.push(value)
            result = items[0] + items[1] + items.size() + items.capacity()
            items.free()
            return result

        @module.func(export=True)
        def push_grows_from_zero() -> i32:
            items = std.Array[i32]()
            items.push(3)
            items.push(4)
            result = items[0] * 10 + items[1]
            items.free()
            return result

    return module.compile()


CASES = (("reserve_preserves", (7,), 23), ("push_grows_from_zero", (), 34))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_array_cases()
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
