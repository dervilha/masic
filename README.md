# MASIC

MASIC builds WebAssembly modules from a small, typed Python subset. Expressions
use ordinary Python operators; function-local assignment is compiled into Wasm
locals, while structured control flow remains explicit.

```python
from masic import WebAssembly, branch, i32, loop


with WebAssembly() as wasm:
    @wasm.func(export=True)
    def sum_until(limit: i32) -> i32:
        value = i32(0)
        total = i32(0)

        with loop() as repeat:
            with branch(value >= limit):
                repeat.brk()

            value += 1

            with branch(value == 5):
                repeat.cont()

            total += value

        return total


wasm.compile().save("module.wasm")
```

`branch(condition)` accepts an integer Wasm expression. Its `as` name exposes
an optional `els()` scope; `loop()` exposes `brk()` and `cont()`; and `block()`
creates a breakable non-loop scope. Native Python `if`, `while`, `for`, and
`break`/`continue` are not part of the compiled subset.

The current implementation supports typed numeric values, functions, memory
allocation, pointers, source-compiled locals, and structured control flow.
Structs and module inclusion are planned but not yet available.

## Validation and current boundaries

The required test suite uses only `unittest` from the Python standard library:

```console
./test-local.sh
```

The repository also has executable public-API probes under `tests/scripts/`:
examples and classic algorithms, allocator lifecycle checks, a native-Python
comparison benchmark, and a limitations/security probe. They compile on every
machine; they execute generated Wasm only when the separately installed
`wasmtime` package is available. See [tests/README.md](tests/README.md) for
commands.

MASIC deliberately supports a small source subset. Decorated functions must be
source-backed named functions and must finish with `return`; native Python
control flow, containers, exceptions, imports inside compiled functions, and
dynamic local creation inside a control scope are not supported. Integer
arithmetic follows Wasm width and trapping rules rather than Python's unbounded
integer semantics.

Pointers are raw linear-memory addresses. MASIC checks dtype, static negative
indices, and the Wasm address range, but it does **not** track allocation
ownership or check an access against the allocated length. An in-bounds-linear-
memory overread can observe a neighboring allocation; an out-of-linear-memory
access traps in the engine. Double frees are unsafe. Do not expose these memory
operations to untrusted inputs without a higher-level bounds/ownership layer.
