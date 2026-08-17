from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .errors import CompileError, ExpressionError
from .functions import FunctionBuilder, FunctionDeclaration, FunctionSignature, LocalRef, active_function
from .instructions import Instruction, WasmInstructions
from .types import coerce_value, expr, is_numeric_type, merge_modules, u32

if TYPE_CHECKING:
    from .module import WebAssembly

WASM32_POINTER_SIZE = 4


@dataclass(frozen=True, slots=True)
class AllocatorLayout:
    page_size: int = 65536
    heap_start: int = 4
    header_size: int = 4
    footer_size: int = 4
    alignment: int = 8
    allocated_flag: int = 1

    @property
    def overhead(self) -> int:
        return self.header_size + self.footer_size

    @property
    def minimum_block_size(self) -> int:
        return self.alignment * 2

    @property
    def size_mask(self) -> int:
        return -self.alignment

    @property
    def maximum_request(self) -> int:
        unaligned = 0xFFFFFFFF - self.heap_start - self.overhead - (self.alignment - 1)
        return unaligned & self.size_mask

    @property
    def page_shift(self) -> int:
        return self.page_size.bit_length() - 1


ALLOCATOR_LAYOUT = AllocatorLayout()


@dataclass(frozen=True, slots=True)
class PointerType:
    dtype: type[expr]
    byte_size: int = WASM32_POINTER_SIZE

    def __repr__(self) -> str:
        return f"pointer[{self.dtype.__name__}]"


@lru_cache(maxsize=None)
def _pointer_type(dtype: type[expr]) -> PointerType:
    _validate_dtype(dtype)
    return PointerType(dtype)


def _validate_dtype(dtype: object) -> None:
    if not is_numeric_type(dtype):
        raise CompileError("dtype must be a concrete Wasm numeric type")


def sizeof(dtype: object) -> int:
    if isinstance(dtype, pointer):
        return dtype._type.byte_size
    if isinstance(dtype, PointerType):
        return dtype.byte_size
    if isinstance(dtype, expr):
        dtype = type(dtype)
    if is_numeric_type(dtype):
        assert dtype.spec is not None
        return dtype.spec.byte_size
    raise TypeError("sizeof() expects a concrete Wasm type, expression, or pointer")


class pointer:
    """A typed Python compiler object backed by one materialized Wasm address local."""

    __slots__ = ("_module", "_type", "_address")

    def __init__(self, module: WebAssembly, dtype: type[expr], address: u32) -> None:
        self._module = module
        self._type = _pointer_type(dtype)
        self._address = address

    def __class_getitem__(cls, dtype: type[expr]) -> PointerType:
        return _pointer_type(dtype)

    @property
    def dtype(self) -> type[expr]:
        return self._type.dtype

    @property
    def address(self) -> u32:
        return self._address

    def _effective_address(self, index: object) -> tuple[tuple[Instruction, ...], int]:
        if isinstance(index, int) and not isinstance(index, bool):
            if index < 0:
                raise ExpressionError("pointer indices cannot be negative")
            offset = index * sizeof(self.dtype)
            if offset > 0xFFFFFFFF:
                raise ExpressionError("pointer index is outside the wasm32 address range")
            return self.address.instructions, offset
        if not isinstance(index, expr):
            raise ExpressionError("pointer index must be an integer or Wasm integer expression")
        if index.category != "int":
            raise ExpressionError("pointer index must be an integer expression")
        scaled = index.cast(u32) * sizeof(self.dtype)
        module = merge_modules(self.address, scaled)
        if module is not self._module:
            raise ExpressionError("pointer index belongs to another module")
        return self.address.instructions + scaled.instructions + (WasmInstructions.create("i32.add"),), 0

    def __getitem__(self, index: object) -> expr:
        builder = active_function(module=self._module)
        address, offset = self._effective_address(index)
        assert self.dtype.spec is not None
        load = self.dtype(
            address + (
                WasmInstructions.create(
                    self.dtype.spec.load,
                    alignment=self.dtype.spec.alignment,
                    offset=offset,
                ),
            ),
            module=self._module,
            trace=(f"load {self!r}[{index!r}]",),
        )
        return builder.materialize(load)

    def __setitem__(self, index: object, value: object) -> None:
        builder = active_function(module=self._module)
        address, offset = self._effective_address(index)
        try:
            stored = coerce_value(value, self.dtype)
        except ExpressionError:
            raise ExpressionError(f"cannot store {type(value).__name__} through {self._type!r}")
        if stored._module is not None and stored._module is not self._module:
            raise ExpressionError("stored expression belongs to another module")
        assert self.dtype.spec is not None
        builder.emit(
            *address,
            *stored.instructions,
            WasmInstructions.create(
                self.dtype.spec.store,
                alignment=self.dtype.spec.alignment,
                offset=offset,
            ),
        )

    def __repr__(self) -> str:
        return f"<{self._type!r} at {self.address}>"


def malloc(count: int | expr, *, dtype: type[expr]) -> pointer:
    _validate_dtype(dtype)
    builder = active_function()
    module = builder.module
    malloc_index, _ = ensure_allocator(module)
    element_size = sizeof(dtype)

    if isinstance(count, int) and not isinstance(count, bool):
        if count < 0:
            raise ExpressionError("malloc count cannot be negative")
        byte_count = count * element_size
        if byte_count > 0xFFFFFFFF:
            raise ExpressionError("malloc size exceeds the wasm32 address range")
        size_value = u32.constant(byte_count)
    elif isinstance(count, expr):
        if count.category != "int":
            raise ExpressionError("malloc count must be an integer expression")
        count_value = count.cast(u32)
        if count_value._module is not None and count_value._module is not module:
            raise ExpressionError("malloc count belongs to another module")
        # The count participates in both multiplication and overflow checking.
        # Materialize it so a call expression or future effectful expression runs once.
        count_value = builder.materialize(count_value)
        # Select zero when multiplication would overflow; malloc(0) returns null.
        maximum = 0xFFFFFFFF // element_size
        multiplied = count_value * element_size
        condition = count_value <= maximum
        instructions = (
            multiplied.instructions
            + u32.constant(0).instructions
            + condition.instructions
            + (WasmInstructions.create("select"),)
        )
        size_value = u32(instructions, module=module, trace=("checked allocation size",))
    else:
        raise ExpressionError("malloc count must be an integer or Wasm integer expression")

    call = u32(
        size_value.instructions + (WasmInstructions.create("call", malloc_index),),
        module=module,
        trace=("malloc",),
    )
    return pointer(module, dtype, builder.materialize(call))


def mfree(address: pointer) -> None:
    if not isinstance(address, pointer):
        raise TypeError("mfree() expects a pointer returned by malloc()")
    builder = active_function(module=address._module)
    _, free_index = ensure_allocator(builder.module)
    builder.emit(
        *address.address.instructions,
        WasmInstructions.create("call", free_index),
    )


class _AllocatorCompiler:
    def __init__(self, parameter_name: str) -> None:
        self.builder = FunctionBuilder(None, (u32,))
        self.parameter = self.builder.parameter(0, parameter_name)

    def local(self, name: str) -> LocalRef:
        return self.builder.local(u32, name)

    def get(self, local: LocalRef) -> Instruction:
        return self.builder.get(local)

    def set(self, local: LocalRef) -> Instruction:
        return self.builder.set(local)

    def op(self, name: str) -> Instruction:
        return self.builder.instruction(name)

    def const(self, value: int) -> Instruction:
        return self.builder.instruction("i32.const", value)

    def load(self, offset: int = 0) -> Instruction:
        return self.builder.instruction("i32.load", alignment=4, offset=offset)

    def store(self, offset: int = 0) -> Instruction:
        return self.builder.instruction("i32.store", alignment=4, offset=offset)

    def global_get(self) -> Instruction:
        return self.builder.instruction("global.get", 0)

    def global_set(self) -> Instruction:
        return self.builder.instruction("global.set", 0)

    def branch(self, depth: int, *, conditional: bool = False) -> Instruction:
        return self.builder.instruction("br_if" if conditional else "br", depth)


def allocator_declarations(start_index: int) -> tuple[FunctionDeclaration, FunctionDeclaration]:
    malloc_signature = FunctionSignature(("size",), (u32,), u32)
    free_signature = FunctionSignature(("address",), (u32,), None)
    return (
        FunctionDeclaration(start_index, "$malloc", malloc_signature, _malloc_body(), False),
        FunctionDeclaration(start_index + 1, "$mfree", free_signature, _free_body(), False),
    )


def ensure_allocator(module: WebAssembly) -> tuple[int, int]:
    from .module import MemoryRequirement

    return module._install_internal_dependency(
        "allocator",
        allocator_declarations,
        memory=MemoryRequirement(
            initial_pages=1,
            mutable_i32_globals=(ALLOCATOR_LAYOUT.heap_start,),
            export=True,
        ),
    )


def _malloc_body():
    c, layout = _AllocatorCompiler("size"), ALLOCATOR_LAYOUT
    required, cursor, tag = c.local("required"), c.local("cursor"), c.local("tag")
    block_size, remainder = c.local("block_size"), c.local("remainder")
    new_end, current_pages, required_pages = c.local("new_end"), c.local("current_pages"), c.local("required_pages")
    g, s, op, k = c.get, c.set, c.op, c.const
    block, loop, iff, end, ret = (op(name) for name in ("block", "loop", "if", "end", "return"))
    ins: list[Instruction] = []
    ins += [g(c.parameter), op("i32.eqz"), iff, k(0), ret, end]
    ins += [g(c.parameter), k(layout.maximum_request - (1 << 32)), op("i32.gt_u"), iff, k(0), ret, end]
    ins += [g(c.parameter), k(layout.alignment - 1), op("i32.add"), k(layout.size_mask), op("i32.and"),
            k(layout.overhead), op("i32.add"), s(required)]
    ins += [k(layout.heap_start), s(cursor), block, loop]
    ins += [g(cursor), c.global_get(), op("i32.ge_u"), c.branch(1, conditional=True)]
    ins += [g(cursor), c.load(), s(tag), g(tag), k(layout.size_mask), op("i32.and"), s(block_size)]
    ins += [g(tag), k(layout.allocated_flag), op("i32.and"), op("i32.eqz"), iff]
    ins += [g(block_size), g(required), op("i32.ge_u"), iff]
    ins += [g(block_size), g(required), op("i32.sub"), s(remainder)]
    ins += [g(remainder), k(layout.minimum_block_size), op("i32.ge_u"), iff]
    ins += [g(cursor), g(required), k(layout.allocated_flag), op("i32.or"), c.store()]
    ins += [g(cursor), g(required), op("i32.add"), k(layout.footer_size), op("i32.sub"),
            g(required), k(layout.allocated_flag), op("i32.or"), c.store()]
    ins += [g(cursor), g(required), op("i32.add"), g(remainder), c.store()]
    ins += [g(cursor), g(block_size), op("i32.add"), k(layout.footer_size), op("i32.sub"), g(remainder), c.store(), end]
    ins += [g(remainder), k(layout.minimum_block_size), op("i32.ge_u"), iff,
            g(cursor), k(layout.header_size), op("i32.add"), ret, end]
    ins += [g(cursor), g(block_size), k(layout.allocated_flag), op("i32.or"), c.store()]
    ins += [g(cursor), g(block_size), op("i32.add"), k(layout.footer_size), op("i32.sub"),
            g(block_size), k(layout.allocated_flag), op("i32.or"), c.store()]
    ins += [g(cursor), k(layout.header_size), op("i32.add"), ret, end, end]
    ins += [g(cursor), g(block_size), op("i32.add"), s(cursor), c.branch(0), end, end]
    ins += [c.global_get(), s(cursor), c.global_get(), g(required), op("i32.add"), s(new_end)]
    ins += [g(new_end), g(cursor), op("i32.lt_u"), iff, k(0), ret, end]
    ins += [op("memory.size"), s(current_pages)]
    ins += [g(new_end), k(1), op("i32.sub"), k(layout.page_shift), op("i32.shr_u"),
            k(1), op("i32.add"), s(required_pages)]
    ins += [g(required_pages), g(current_pages), op("i32.gt_u"), iff]
    ins += [g(required_pages), g(current_pages), op("i32.sub"), op("memory.grow"), k(-1), op("i32.eq"),
            iff, k(0), ret, end, end]
    ins += [g(cursor), g(required), k(layout.allocated_flag), op("i32.or"), c.store()]
    ins += [g(new_end), k(layout.footer_size), op("i32.sub"), g(required), k(layout.allocated_flag), op("i32.or"), c.store()]
    ins += [g(new_end), c.global_set(), g(cursor), k(layout.header_size), op("i32.add")]
    c.builder.emit(*ins)
    return c.builder.finish()


def _free_body():
    c, layout = _AllocatorCompiler("address"), ALLOCATOR_LAYOUT
    block_ref, size, next_ref, previous_size = (c.local(name) for name in ("block", "size", "next", "previous_size"))
    g, s, op, k = c.get, c.set, c.op, c.const
    iff, end, ret = (op(name) for name in ("if", "end", "return"))
    ins: list[Instruction] = []
    ins += [g(c.parameter), op("i32.eqz"), iff, ret, end]
    ins += [g(c.parameter), k(layout.header_size), op("i32.sub"), s(block_ref)]
    ins += [g(block_ref), c.load(), s(size)]
    ins += [g(size), k(layout.allocated_flag), op("i32.and"), op("i32.eqz"), iff, ret, end]
    ins += [g(size), k(layout.size_mask), op("i32.and"), s(size)]
    ins += [g(block_ref), g(size), op("i32.add"), s(next_ref)]
    ins += [g(next_ref), c.global_get(), op("i32.lt_u"), iff]
    ins += [g(next_ref), c.load(), k(layout.allocated_flag), op("i32.and"), op("i32.eqz"), iff]
    ins += [g(size), g(next_ref), c.load(), k(layout.size_mask), op("i32.and"), op("i32.add"), s(size), end, end]
    ins += [g(block_ref), k(layout.heap_start), op("i32.gt_u"), iff]
    ins += [g(block_ref), k(layout.footer_size), op("i32.sub"), c.load(), s(previous_size)]
    ins += [g(previous_size), k(layout.allocated_flag), op("i32.and"), op("i32.eqz"), iff]
    ins += [g(previous_size), k(layout.size_mask), op("i32.and"), s(previous_size)]
    ins += [g(block_ref), g(previous_size), op("i32.sub"), s(block_ref)]
    ins += [g(size), g(previous_size), op("i32.add"), s(size), end, end]
    ins += [g(block_ref), g(size), op("i32.add"), c.global_get(), op("i32.eq"), iff]
    ins += [g(block_ref), c.global_set(), ret, end]
    ins += [g(block_ref), g(size), c.store()]
    ins += [g(block_ref), g(size), op("i32.add"), k(layout.footer_size), op("i32.sub"), g(size), c.store()]
    c.builder.emit(*ins)
    return c.builder.finish()
