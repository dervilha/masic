#!/usr/bin/env python3
"""Run imported host calls and shared linear memory through optional Wasmtime."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, i32, malloc, pointer, u8, u32
from masic.wasm import stdlib


def build_interop_module() -> WebAssembly:
    with WebAssembly("interop-runtime") as wasm:
        std = wasm.include(stdlib)
        @wasm.import_func("host")
        def record(value: i32) -> None:
            ...

        @wasm.import_func("host", name="combine")
        def host_combine(value: i32, count: u32) -> i32:
            ...

        @wasm.func(export=True)
        def consume(address: u32, count: u32) -> i32:
            values = pointer[i32].from_address(address)
            record(values[0])
            return host_combine(values[0] + values[1], count)

        @wasm.func(export=True)
        def create_buffer(size: u32) -> u32:
            return malloc(size, dtype=u8).address

        @wasm.func(export=True)
        def static_then_allocate() -> i32:
            marker = std.String("S")
            storage = malloc(1, dtype=u8)
            storage[0] = 1
            return marker[0] + storage[0]

    return wasm.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping execution")
    args = parser.parse_args(argv)

    wasm = build_interop_module()
    try:
        import wasmtime
    except ImportError:
        print("runtime skipped: Wasmtime is not installed (install it separately to execute Wasm).")
        return 2 if args.require_engine else 0

    engine = wasmtime.Engine()
    module = wasmtime.Module(engine, bytes(wasm))
    store = wasmtime.Store(engine)
    recorded: list[int] = []
    record = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32()], []), lambda value: recorded.append(value))
    combine = wasmtime.Func(
        store,
        wasmtime.FuncType([wasmtime.ValType.i32(), wasmtime.ValType.i32()], [wasmtime.ValType.i32()]),
        lambda value, count: value + count,
    )
    instance = wasmtime.Instance(store, module, [record, combine])
    exports = instance.exports(store)
    memory = exports["memory"]
    address = exports["create_buffer"](store, 8)
    memory.write(store, struct.pack("<ii", 40, 2), address)
    result = exports["consume"](store, address, 2)
    static_result = exports["static_then_allocate"](store)
    if (recorded, result, static_result) != ([40], 44, 84):
        raise AssertionError(f"expected host record [40] and result 44, got {recorded!r} and {result!r}")
    print(f"shared memory address {address}: host record={recorded[0]}, Wasm result={result}, static data={static_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
