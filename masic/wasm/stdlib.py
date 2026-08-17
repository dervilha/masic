"""The includeable MASIC standard-library namespace and its first String type."""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .errors import CompileError, ExpressionError
from .functions import LocalRef, active_function
from .memory import WASM32_POINTER_SIZE, malloc, mfree, pointer
from .module import WebAssembly
from .types import expr, i32, is_numeric_type, u8, u32


class _Stdlib(WebAssembly):
    """A template module whose exports are bound by ``module.include(stdlib)``."""

    def _create_namespace(self, module: WebAssembly) -> _StdlibNamespace:
        return _StdlibNamespace(module)

    def _include_into(self, module: WebAssembly, namespace: _StdlibNamespace) -> None:
        """Reserved for stdlib dependencies that will be bound after this namespace."""


class _StdlibNamespace:
    """Standard-library values bound to one destination WebAssembly module."""

    __slots__ = ("Array", "String", "_module")

    def __init__(self, module: WebAssembly) -> None:
        self._module = module
        self.Array = _ArrayFactory(module)
        self.String = _StringFactory(module)

    def __repr__(self) -> str:
        return f"<stdlib namespace for {self._module.name!r}>"


class _StringFactory:
    """Construct literal-backed String compiler values in one destination module."""

    __slots__ = ("_module",)

    def __init__(self, module: WebAssembly) -> None:
        self._module = module

    def __call__(self, literal: object) -> String:
        if not isinstance(literal, str):
            raise ExpressionError("String() expects a Python string literal")
        encoded = literal.encode("utf-8")
        total_size = String.HEADER_SIZE + len(encoded)
        if total_size > 0xFFFFFFFF:
            raise ExpressionError("String literal exceeds the wasm32 address range")
        active_function(module=self._module)
        address = self._module._intern_static(struct.pack("<I", len(encoded)) + encoded, alignment=4)
        storage = pointer[u8].from_address(address)
        return String(self._module, storage, len(encoded))

    def __repr__(self) -> str:
        return f"<stdlib.String for {self._module.name!r}>"


class _ArrayFactory:
    """Bind ``Array[T]`` factories to one destination WebAssembly module."""

    __slots__ = ("_module",)

    def __init__(self, module: WebAssembly) -> None:
        self._module = module

    def __getitem__(self, dtype: object) -> _ArrayType:
        if not is_numeric_type(dtype):
            raise CompileError("Array[T] requires a concrete Wasm numeric element type")
        return _ArrayType(self._module, dtype)

    def __repr__(self) -> str:
        return f"<stdlib.Array for {self._module.name!r}>"


class _ArrayType:
    """Construct one typed Array struct in an active destination function."""

    __slots__ = ("_dtype", "_module")

    def __init__(self, module: WebAssembly, dtype: type[expr]) -> None:
        self._module = module
        self._dtype = dtype

    def __call__(self, capacity: object = 0) -> Array:
        builder = active_function(module=self._module)
        capacity_value = _array_count(capacity, module=self._module, label="Array capacity")
        if isinstance(capacity, int) and not isinstance(capacity, bool):
            storage = malloc(Array.HEADER_SIZE + capacity * self._dtype.byte_size, dtype=u8)
        else:
            storage = malloc(capacity_value * self._dtype.byte_size + Array.HEADER_SIZE, dtype=u8)
        address_local = builder.materialized_local(storage.address)
        if address_local is None:
            raise CompileError("Array allocation address was not materialized")
        header = pointer(self._module, u32, storage.address)
        header[0] = 0
        header[1] = capacity_value
        return Array(self._module, self._dtype, address_local)

    def __repr__(self) -> str:
        return f"<stdlib.Array[{self._dtype.__name__}] for {self._module.name!r}>"


@dataclass(frozen=True, slots=True)
class String:
    """A read-only UTF-8 String struct stored in the module data section.

    The layout is a u32 byte length followed immediately by UTF-8 bytes.  This
    first implementation accepts only compile-time Python string literals.
    """

    _module: WebAssembly
    _storage: pointer
    _byte_length: int

    HEADER_SIZE = WASM32_POINTER_SIZE

    def _active(self) -> None:
        active_function(module=self._module)

    def _header(self) -> pointer:
        return pointer(self._module, u32, self._storage.address)

    def length(self) -> u32:
        """Return the UTF-8 byte length loaded from this String's header."""
        self._active()
        return self._header()[0]

    def byte_at(self, index: object) -> u8:
        """Return one byte, checking literal indexes against the known length."""
        self._active()
        if isinstance(index, int) and not isinstance(index, bool):
            if not 0 <= index < self._byte_length:
                raise ExpressionError(f"String byte index {index} is outside [0, {self._byte_length})")
            return self._storage[self.HEADER_SIZE + index]
        if not isinstance(index, expr) or index.category != "int":
            raise ExpressionError("String index must be an integer or Wasm integer expression")
        # Dynamic indexes use raw linear memory, so callers must establish bounds.
        return self._storage[index + self.HEADER_SIZE]

    def __getitem__(self, index: object) -> u8:
        return self.byte_at(index)

    def is_empty(self) -> i32:
        """Return a Wasm i32 boolean derived from the stored byte length."""
        return self.length() == 0

    def free(self) -> None:
        """No-op: String literals are static module data, not allocator storage."""
        self._active()


@dataclass(frozen=True, slots=True)
class Array:
    """A typed, resizable Array struct allocated in active module linear memory.

    Layout: a u32 size, a u32 capacity, then tightly packed elements of one
    concrete Wasm numeric type. The address is held in a private Wasm local so
    ``reserve()`` can relocate storage without changing user-facing syntax.
    """

    _module: WebAssembly
    _dtype: type[expr]
    _address_local: LocalRef

    HEADER_SIZE = WASM32_POINTER_SIZE * 2

    def _active(self):
        return active_function(module=self._module)

    def _address(self) -> u32:
        return u32.local(self._address_local.index, module=self._module)

    def _storage(self) -> pointer:
        return pointer(self._module, u8, self._address())

    def _header(self) -> pointer:
        return pointer(self._module, u32, self._address())

    def _elements(self) -> pointer:
        return pointer(self._module, self._dtype, self._address() + self.HEADER_SIZE)

    def size(self) -> u32:
        """Return the current element count loaded from the Array header."""
        self._active()
        return self._header()[0]

    def capacity(self) -> u32:
        """Return the current element capacity loaded from the Array header."""
        self._active()
        return self._header()[1]

    def reserve(self, requested: object) -> None:
        """Grow to at least ``requested`` elements, preserving existing values."""
        builder = self._active()
        requested_value = builder.materialize(
            _array_count(requested, module=self._module, label="Array reserve capacity")
        )
        current_capacity = self.capacity()
        current_size = self.size()
        condition = requested_value > current_capacity
        builder.emit(*condition.instructions, builder.instruction("if"))

        new_storage = malloc(
            requested_value * self._dtype.byte_size + self.HEADER_SIZE,
            dtype=u8,
        )
        new_header = pointer(self._module, u32, new_storage.address)
        new_header[0] = current_size
        new_header[1] = requested_value
        self._copy_elements(new_storage, current_size)
        mfree(self._storage())
        builder.emit(*new_storage.address.instructions, builder.set(self._address_local))
        builder.emit(builder.instruction("end"))

    def push(self, value: object) -> None:
        """Append one typed value, growing the backing allocation if it is full."""
        self._active()
        index = self.size()
        self.reserve(index + 1)
        self._elements()[index] = value
        self._header()[0] = index + 1

    def __getitem__(self, index: object) -> expr:
        """Load a typed element; dynamic indexes require caller-provided bounds checks."""
        self._active()
        return self._elements()[index]

    def free(self) -> None:
        """Release the current backing allocation through MASIC's allocator."""
        self._active()
        mfree(self._storage())

    def _copy_elements(self, new_storage: pointer, size: u32) -> None:
        """Emit a Wasm loop that copies this Array's initialized element range."""
        builder = self._active()
        index_local = builder.local(u32)
        builder.emit(*u32.constant(0).instructions, builder.set(index_local))
        builder.emit(builder.instruction("block"), builder.instruction("loop"))
        index = u32.local(index_local.index, module=self._module)
        builder.emit(*(index >= size).instructions, builder.instruction("br_if", 1))
        source = self._elements()
        destination = pointer(self._module, self._dtype, new_storage.address + self.HEADER_SIZE)
        destination[index] = source[index]
        builder.emit(*(index + 1).instructions, builder.set(index_local), builder.instruction("br", 0))
        builder.emit(builder.instruction("end"), builder.instruction("end"))


def _array_count(value: object, *, module: WebAssembly, label: str) -> u32:
    """Convert a static or Wasm integer count into a module-local unsigned value."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ExpressionError(f"{label} cannot be negative")
        return u32.constant(value)
    if not isinstance(value, expr) or value.category != "int":
        raise ExpressionError(f"{label} must be an integer or Wasm integer expression")
    if value._module is not None and value._module is not module:
        raise ExpressionError(f"{label} belongs to another WebAssembly module")
    return value.cast(u32)


stdlib = _Stdlib("masic.stdlib")

__all__ = ["Array", "String", "stdlib"]
