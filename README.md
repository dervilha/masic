# MASIC

MASIC is a Python-first toolkit for machine-controlling applications. Its first
implemented subsystem is a small compiler that turns readable, typed Python
function definitions into compact WebAssembly binaries.

It is deliberately pre-alpha: the supported language is small, direct, and
tested, rather than pretending to compile arbitrary Python.

## Start here

The complete copy-and-adapt reference is the
[WebAssembly Syntax Guide](WebAssembly%20Syntax%20Guide.md). It documents every
currently supported construct, its generated WebAssembly role, and its
important limits.

```python
from masic import WebAssembly, branch, i32, loop

with WebAssembly("sum") as wasm:
    @wasm.func(export=True)
    def sum_except_five(limit: i32) -> i32:
        current = i32(0)
        total = i32(0)

        with loop() as repeat:     # explicit Wasm loop
            with branch(current >= limit):
                repeat.brk()

            current += 1
            with branch(current == 5):
                repeat.cont()
            total += current

        return total               # ordinary Python return syntax

wasm.save("sum.wasm")
```

`@wasm.func` reads and compiles the function source; it does not run its body
as Python. Python arithmetic operators construct typed Wasm expressions, local
assignment becomes Wasm local assignment, and `branch()`, `loop()`, and
`block()` are explicit structured-control scopes.

## Current capabilities

- Typed numeric values: signed/unsigned 8-, 16-, 32-, and 64-bit integers,
  plus `f32` and `f64`; normal arithmetic, bitwise operators, casts, and
  comparisons.
- Typed function declarations and calls, default numeric arguments, exported
  functions, `-> None` functions, and imported host functions.
- One automatic, exported wasm32 linear memory when memory is needed; typed
  pointers, a small allocator, and raw host/Wasm shared-memory interop.
- Explicit Wasm control flow with natural Python `return` syntax.
- Includeable `stdlib` namespace with literal UTF-8 `String` and resizable
  numeric `Array[T]` values.
- Opt-in persistent numeric globals, an initialized function table for
  indirect calls, a single start function, and optional standard Wasm debug
  name metadata.

Module sections are demand-driven: a function-only module does not pay for
memory, data, global, table, start, or debug sections it does not use.

## Interoperability in one example

Imports are declared before local functions. MASIC owns the linear memory;
the host reads and writes it through its WebAssembly engine and passes raw
`u32` addresses to exported functions.

```python
from masic import WebAssembly, i32, pointer, u32

with WebAssembly("interop") as wasm:
    @wasm.import_func("host", name="record")
    def log(value: i32) -> None:
        ...                         # supplied by the embedding host

    @wasm.func(export=True)
    def add_pair(address: u32) -> i32:
        values = pointer[i32].from_address(address)
        log(values[0])
        return values[0] + values[1]
```

For more examples—including allocation, String, Array, globals, indirect
calls, and start initialization—use the
[syntax guide](WebAssembly%20Syntax%20Guide.md).

## Boundaries and safety

This compiler is intentionally not a Python runtime. Native Python `if`,
`while`, `for`, exceptions, Python containers, arbitrary class instances, and
imports inside compiled functions are unsupported.

Pointers are raw Wasm addresses. MASIC validates types and static negative
indexes, but does not track allocation size, ownership, or lifetime. A dynamic
overread within linear memory can observe neighboring data; an out-of-range
linear-memory access traps in the Wasm engine. Double-free is unsafe. Imported
host functions are an ABI and trust boundary, too.

The current module model intentionally has one automatic memory and one
initialized table. It does not import host memory, globals, or tables; support
table mutation; or provide bulk-memory/passive-segment features.

## Testing

The required suite uses only the Python standard library:

```console
./test-local.sh
```

The public scripts under `tests/scripts/` compile realistic user programs and,
when separately installed, execute them with Wasmtime. They cover algorithms,
numeric behavior, allocator lifecycle, stdlib values, host interop, module
sections, benchmarks, and documented failure cases. See
[tests/README.md](tests/README.md) for commands. Wasmtime is optional and is
not required by MASIC or its normal test suite.

## Project status

MASIC is version `0.1.0`, pre-alpha. The current WebAssembly API is useful for
small numeric kernels and explicit low-level interop experiments. Expect the
surface to evolve; rely on the syntax guide and tests as the authoritative
description of supported behavior.
