#!/usr/bin/env python3
"""Compare a native batch with the same calls made inside one Wasm loop.

The timed Wasm path makes one cached Python-to-Wasm call per sample.  The
exported batch function then calls the compiled worker ``--calls`` times from a
Wasm loop, so store creation, instantiation, export lookup, and a Python loop
are excluded from the execution timing. Compilation and instance setup are
reported separately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from masic import WebAssembly, branch, i32, loop
from runtime import load_wasmtime


def native_sum_odd_below(limit: int) -> int:
    total = 0
    for value in range(limit):
        if value % 2:
            total += value
    return total


def build_benchmark_module() -> WebAssembly:
    with WebAssembly("benchmark") as wasm:
        @wasm.func(export=True)
        def sum_odd_below(limit: i32) -> i32:
            value = i32(0)
            total = i32(0)
            with loop() as repeat:
                with branch(value >= limit):
                    repeat.brk()
                with branch(value % 2 == 0):
                    value += 1
                    repeat.cont()
                total += value
                value += 1
            return total

        @wasm.func(export=True)
        def sum_odd_below_batch(limit: i32, calls: i32) -> i32:
            iteration = i32(0)
            total = i32(0)
            with loop() as repeat:
                with branch(iteration >= calls):
                    repeat.brk()
                total += sum_odd_below(limit)
                iteration += 1
            return total

    return wasm.compile()


def native_batch(limit: int, calls: int) -> int:
    total = 0
    for _ in range(calls):
        total += native_sum_odd_below(limit)
    return total


def time_samples(call, samples: int) -> tuple[object, float]:
    started = perf_counter()
    result = None
    for _ in range(samples):
        result = call()
    return result, perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1_000, help="exclusive upper bound for the sum")
    parser.add_argument("--calls", "--iterations", dest="calls", type=int, default=1_000, help="worker calls per batch")
    parser.add_argument("--samples", type=int, default=25, help="timed batches after warm-up")
    parser.add_argument("--require-engine", action="store_true", help="fail instead of skipping Wasm timing")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.calls < 1 or args.samples < 1:
        parser.error("--limit must be non-negative; --calls and --samples must be positive")

    compile_started = perf_counter()
    wasm = build_benchmark_module()
    compile_seconds = perf_counter() - compile_started
    expected = native_batch(args.limit, args.calls)
    if not -(1 << 31) <= expected < (1 << 31):
        parser.error("the batch result must fit in signed i32 for an equivalent comparison")
    native_result, native_seconds = time_samples(
        lambda: native_batch(args.limit, args.calls), args.samples
    )
    if native_result != expected:
        raise AssertionError("native baseline produced an inconsistent result")

    print(f"MASIC source-to-Wasm build: {len(bytes(wasm))} bytes in {compile_seconds:.6f}s")
    print(
        f"native Python: {args.samples} batches × {args.calls} worker calls "
        f"in {native_seconds:.6f}s ({expected})"
    )

    engine_started = perf_counter()
    runtime, reason = load_wasmtime(bytes(wasm))
    engine_compile_seconds = perf_counter() - engine_started
    if runtime is None:
        print(f"Wasm timing skipped: {reason}")
        return 2 if args.require_engine else 0

    instance_started = perf_counter()
    batched = runtime.cached_export("sum_odd_below_batch")
    instance_seconds = perf_counter() - instance_started
    warm_result = batched.invoke(args.limit, args.calls)
    if warm_result != expected:
        raise AssertionError(f"Wasm warm-up: expected {expected}, got {warm_result}")
    wasm_result, wasm_seconds = time_samples(
        lambda: batched.invoke(args.limit, args.calls), args.samples
    )
    if wasm_result != expected:
        raise AssertionError(f"Wasm result: expected {expected}, got {wasm_result}")
    ratio = wasm_seconds / native_seconds if native_seconds else float("inf")
    print(f"Wasmtime module compilation: {engine_compile_seconds:.6f}s")
    print(f"Wasmtime instance/export setup: {instance_seconds:.6f}s")
    print(
        f"Wasm batched execution: {args.samples} cached calls × {args.calls} Wasm worker calls "
        f"in {wasm_seconds:.6f}s ({wasm_result})"
    )
    print(f"Wasm/native execution ratio: {ratio:.2f}x; setup and compilation excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
