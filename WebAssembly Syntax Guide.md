# MASIC WebAssembly Syntax Guide

MASIC compiles a small, typed subset of Python into a WebAssembly module. The
decorated function body is **read as source; it is not executed by Python**.

```python
from masic import WebAssembly, i32

with WebAssembly("math") as wasm:
    @wasm.func(export=True)       # exported as "add"
    def add(left: i32, right: i32) -> i32:
        return left + right       # ordinary Python return syntax

wasm.save("math.wasm")           # or: binary = bytes(wasm)
```

The `with` block assembles automatically. `wasm.compile()` is available when
building outside a context manager.

## Imports and numeric types

```python
from masic import (
    WebAssembly,
    i8, u8, i16, u16, i32, u32, i64, u64, f32, f64,
)

with WebAssembly("numbers") as wasm:
    @wasm.func(export=True)
    def calculate(a: i32, b: i32) -> i32:
        # Python arithmetic becomes Wasm arithmetic.
        value = a * 2 + b
        value += 1                # assignment updates the Wasm local
        return value
```

| Python annotation | Wasm stack type | Stored width |
| --- | --- | --- |
| `i8`, `u8`, `i16`, `u16`, `i32`, `u32` | `i32` | 1, 2, or 4 bytes |
| `i64`, `u64` | `i64` | 8 bytes |
| `f32` | `f32` | 4 bytes |
| `f64` | `f64` | 8 bytes |

Use MASIC types in every parameter and result annotation. Use a type call to
convert an expression or select a literal type:

```python
@wasm.func(export=True)
def narrow(value: i64) -> u8:
    return u8(value)              # explicit Wasm conversion
```

Supported expression operators are `+`, `-`, `*`, `/`, `//`, `%`, `&`, `|`,
`^`, `<<`, `>>`, unary `+`, `-`, `~`, and `==`, `!=`, `<`, `<=`, `>`, `>=`.
Comparisons produce a Wasm `i32` boolean. Integer behavior is fixed-width
Wasm behavior, not Python's unbounded-integer behavior.

## Functions, calls, and returns

```python
with WebAssembly("calls") as wasm:
    @wasm.func()
    def twice(value: i32) -> i32:
        return value * 2

    @wasm.func(export=True)
    def use_default(value: i32, scale: i32 = 3) -> i32:
        return twice(value) * scale  # local Wasm-to-Wasm call

    @wasm.func(export=True)
    def notify(value: i32) -> None:
        # A void function has a bare, ordinary Python return.
        return
```

Functions must be named, source-backed `def`s. They must end in `return`:
`return expression` for a numeric result, or bare `return` for `-> None`.
Void calls are statements only:

```python
@wasm.func()
def log_then_double(value: i32) -> i32:
    notify(value)                 # valid: statement
    return twice(value)
```

## Explicit control flow

Python `if`, `while`, `for`, `break`, and `continue` are not compiled. Use
explicit scopes, which make the generated control flow obvious.

```python
from masic import branch, loop

@wasm.func(export=True)
def sum_except_five(limit: i32) -> i32:
    current = i32(0)              # locals start before a control scope
    total = i32(0)

    with loop() as repeat:        # repeats forever until repeat.brk()
        with branch(current >= limit):
            repeat.brk()          # leave the loop

        current += 1

        with branch(current == 5):
            repeat.cont()         # next iteration

        total += current

    return total
```

`branch()` can have an explicit else binding. `block()` is a non-loop,
breakable scope.

```python
from masic import block, branch

@wasm.func(export=True)
def choose(value: i32) -> i32:
    result = i32(0)

    with branch(value > 0) as test:
        result = 1
    with test.els():              # must immediately follow its branch
        result = -1

    with block() as early_exit:
        with branch(value == 99):
            early_exit.brk()      # exit only this block
        result += 10

    return result
```

`branch` supports neither `.brk()` nor `.cont()`; `block` supports `.brk()`;
`loop` supports both. Do not declare a new local inside one of these scopes.

## Linear memory and allocation

Memory is automatic and is exported as `memory` only when a feature needs it.
`malloc()` adds MASIC's allocator; `mfree()` releases an allocation.

```python
from masic import malloc, mfree, pointer, sizeof, i32, u8, u32

@wasm.func(export=True)
def make_pair(left: i32, right: i32) -> u32:
    values = malloc(2, dtype=i32) # two i32 elements; returns pointer[i32]
    values[0] = left               # typed Wasm store
    values[1] = right
    return values.address          # wasm32 address, therefore u32

@wasm.func(export=True)
def read_pair(address: u32) -> i32:
    values = pointer[i32].from_address(address)
    return values[0] + values[1]   # typed Wasm loads

@wasm.func()
def discard_one() -> None:
    bytes_ = malloc(64, dtype=u8)
    mfree(bytes_)
    return

# `sizeof(i64) == 8`; `sizeof(pointer[i32]) == 4`.
```

`pointer[T].from_address()` is the shared-memory bridge: a host can write to
the exported memory at an address and call a MASIC export with that address.
Pointers have no length, ownership, or lifetime checks. Dynamic indexes must
be bounds-checked by the caller; out-of-memory-range access traps in Wasm.

## Standard library: String and Array

Include the namespace once per destination module. Repeated inclusion returns
the same namespace, including during circular library inclusion.

```python
from masic import WebAssembly, i32
from masic.wasm import stdlib

with WebAssembly("collections") as wasm:
    std = wasm.include(stdlib)     # std belongs to this `wasm` module

    @wasm.func(export=True)
    def text_and_array(value: i32) -> i32:
        text = std.String("Olá")  # Python literal -> static UTF-8 data
        first_byte = text[0]       # or: text.byte_at(0)
        length = text.length()     # UTF-8 BYTE count, not code points
        empty = text.is_empty()    # i32 boolean
        text.free()                # harmless no-op for static literals

        values = std.Array[i32](capacity=1)
        values.push(first_byte)
        values.reserve(4)          # grows and preserves initialized values
        values.push(value)
        result = values[0] + values[1] + values.size() + values.capacity()
        values.free()              # frees the allocator storage
        return result + length + empty
```

`String` accepts only Python string literals. Its layout is `u32 byte_length`
then UTF-8 bytes. There are no String parameters/returns, concatenation,
equality, searching, or Unicode code-point indexing.

`Array[T]` accepts a concrete numeric MASIC type. Its layout is `u32 size`,
`u32 capacity`, then packed elements. It supports `size()`, `capacity()`,
`reserve()`, `push()`, `array[index]`, and `free()`—not pop, insert, remove,
slices, or Array parameters/returns.

## Host functions and shared memory

Declare imports before every local `@wasm.func`. An imported declaration has
only `...` or `pass` for its body; the host provides the implementation.

```python
from masic import WebAssembly, i32, pointer, u32

with WebAssembly("interop") as wasm:
    @wasm.import_func("host", name="log_i32")  # import module + field
    def log(value: i32) -> None:
        ...

    @wasm.import_func("host")                    # field defaults to `combine`
    def combine(value: i32, count: u32) -> i32:
        pass

    @wasm.func(export=True)
    def consume(address: u32, count: u32) -> i32:
        values = pointer[i32].from_address(address)
        log(values[0])                            # calls the host
        return combine(values[0] + values[1], count)
```

Imports use numeric Wasm types only. There is no imported memory API: MASIC
owns its one automatic exported memory. The embedding host accesses it through
the engine and exchanges `u32` addresses with exported functions.

## Persistent state, indirect calls, and start

```python
from masic import WebAssembly, i32

with WebAssembly("state", debug=True) as wasm:   # includes Wasm name metadata
    counter = wasm.create_global(i32, 0, mutable=True, export="counter")

    @wasm.func()
    def add_one(value: i32) -> i32:
        return value + 1

    @wasm.func()
    def sub_one(value: i32) -> i32:
        return value - 1

    # Entries must already be declared and have exactly the same signature.
    operations = wasm.create_table([add_one, sub_one], export="operations")

    @wasm.func(export=True)
    def dispatch(index: i32, value: i32) -> i32:
        return operations.call(index, value)       # Wasm call_indirect

    @wasm.func(export=True)
    def bump() -> i32:
        counter.value += 1
        return counter.value

    @wasm.start
    def initialize() -> None:                      # exactly () -> None
        counter.value = 41
        return
```

Globals are numeric and read/written through `.value`; only mutable globals
can be assigned. The current implementation supports one initialized,
immutable-after-instantiation table. An out-of-range table index traps. Start
functions are not exported unless separately called through another export.

## What is intentionally unsupported

- Native Python control flow, Python containers, exceptions, imports inside a
  compiled function, lambdas, async functions, and dynamic local creation in a
  control scope.
- Host-provided/imported memory, memory/table/global imports, multiple
  memories or tables, table mutation, passive segments, bulk-memory operations,
  and custom sections other than opt-in debug names.
- Safe memory ownership or automatic bounds checks. Treat raw pointers and
  imported host functions as an explicit trust boundary.

Run the deterministic suite with `./test-local.sh`. Optional executable
Wasmtime probes live in [`tests/README.md`](tests/README.md).
