#!/usr/bin/env python3
"""Run persistent globals, table dispatch, and start initialization in Wasmtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, i32
from runtime import load_wasmtime


def build_sections_module() -> WebAssembly:
    with WebAssembly("sections-runtime") as wasm:
        counter = wasm.create_global(i32, 0, mutable=True, export="counter")

        @wasm.func()
        def add_one(value: i32) -> i32:
            return value + 1

        @wasm.func()
        def sub_one(value: i32) -> i32:
            return value - 1

        operations = wasm.create_table([add_one, sub_one], export="operations")

        @wasm.func(export=True)
        def dispatch(index: i32, value: i32) -> i32:
            return operations.call(index, value)

        @wasm.func(export=True)
        def bump() -> i32:
            counter.value += 1
            return counter.value

        @wasm.start
        def initialize() -> None:
            counter.value = 41
            return

    return wasm.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_sections_module()
    print(f"compiled sections module into {len(bytes(wasm))} bytes")
    runtime, reason = load_wasmtime(bytes(wasm))
    if runtime is None:
        print(f"runtime skipped: {reason}")
        return 2 if args.require_engine else 0

    if runtime.invoke("dispatch", 0, 10) != 11:
        raise AssertionError("table entry 0 did not add one")
    if runtime.invoke("dispatch", 1, 10) != 9:
        raise AssertionError("table entry 1 did not subtract one")
    if runtime.invoke("bump") != 42:
        raise AssertionError("start section did not initialize the global")
    print("ok start global and indirect table dispatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
