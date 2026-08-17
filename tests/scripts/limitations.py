#!/usr/bin/env python3
"""Make current MASIC language limits and memory-safety hazards explicit.

The compiler rejections below are required behaviour.  The optional runtime
checks demonstrate that pointer indexing is raw Wasm linear-memory access, not
ownership-checked Python indexing.  Do not use this allocator with untrusted
pointer arithmetic or double frees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import CompileError, WebAssembly, branch, i32, malloc
from runtime import load_wasmtime


def native_if(value: i32) -> i32:
    if value == 0:
        return 1
    return value


def native_while(value: i32) -> i32:
    while value > 0:
        value -= 1
    return value


def late_local(value: i32) -> i32:
    with branch(value == 0):
        only_on_one_path = i32(1)
    return value


REJECTED = (
    ("native if", native_if, "explicit branch() and loop()"),
    ("native while", native_while, "explicit branch() and loop()"),
    ("branch-only local", late_local, "initialized before entering a control scope"),
)


def build_hazard_module() -> WebAssembly:
    with WebAssembly("limitations") as wasm:
        @wasm.func(export=True)
        def cross_allocation_read() -> i32:
            first = malloc(1, dtype=i32)
            second = malloc(1, dtype=i32)
            first[0] = 11
            second[0] = 99
            return first[4]

        @wasm.func(export=True)
        def out_of_bounds_read() -> i32:
            values = malloc(1, dtype=i32)
            return values[16384]

    return wasm.compile()


def check_compiler_rejections() -> None:
    for label, function, expected_text in REJECTED:
        try:
            WebAssembly("rejections").func()(function)
        except CompileError as error:
            if expected_text not in str(error):
                raise AssertionError(f"{label}: unexpected error: {error}") from error
            print(f"expected compiler rejection: {label}: {error}")
        else:
            raise AssertionError(f"{label}: compiler unexpectedly accepted unsupported source")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping runtime hazards")
    args = parser.parse_args(argv)

    check_compiler_rejections()
    wasm = build_hazard_module()
    print("compiled raw-pointer hazard probes")
    runtime, reason = load_wasmtime(bytes(wasm))
    if runtime is None:
        print(f"runtime hazards skipped: {reason}")
        return 2 if args.require_engine else 0

    leaked = runtime.invoke("cross_allocation_read")
    if leaked != 99:
        raise AssertionError(f"cross-allocation read changed: expected 99, got {leaked}")
    print("demonstrated unchecked allocation boundary: first[4] read neighboring allocation value 99")

    try:
        runtime.invoke("out_of_bounds_read")
    except Exception as error:  # the exact trap class belongs to the optional engine
        print(f"demonstrated linear-memory bounds trap: {type(error).__name__}: {error}")
    else:
        raise AssertionError("out-of-bounds Wasm load unexpectedly completed")

    print("not executed: double-free behaviour is undefined for the current allocator and may corrupt its free list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
