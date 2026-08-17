#!/usr/bin/env python3
"""Execute the included stdlib.String UTF-8 layout and methods in Wasmtime."""

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


def build_string_cases() -> WebAssembly:
    with WebAssembly("stdlib-runtime") as module:
        std = module.include(stdlib)

        @module.func(export=True)
        def utf8_metadata() -> i32:
            text = std.String("Olá")
            result = text.length() + text.byte_at(2)
            text.free()
            return result

        @module.func(export=True)
        def empty() -> i32:
            text = std.String("")
            result = text.is_empty()
            text.free()
            return result

        @module.func(export=True)
        def bracket_access() -> i32:
            text = std.String("MASIC")
            result = text[1]
            text.free()
            return result

    return module.compile()


CASES = (("utf8_metadata", (), 199), ("empty", (), 1), ("bracket_access", (), 65))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_string_cases()
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
