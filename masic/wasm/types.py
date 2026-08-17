from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from .errors import ExpressionError
from .instructions import Instruction, WasmInstructions

if TYPE_CHECKING:
    from .module import WebAssembly


@dataclass(frozen=True, slots=True)
class NumericTypeSpec:
    wasm_type: int
    stack_bits: int
    bits: int
    category: str
    signed: bool | None
    byte_size: int
    load: str
    store: str
    alignment: int


class expr:
    """An immutable instruction sequence with a statically known numeric type."""

    spec: ClassVar[NumericTypeSpec | None] = None
    # Compatibility attributes are derived from ``spec`` by __init_subclass__.
    wasm_type: ClassVar[int]
    stack_bits: ClassVar[int]
    bits: ClassVar[int]
    signed: ClassVar[bool | None]
    category: ClassVar[str]
    byte_size: ClassVar[int]

    __slots__ = ("_instructions", "_module", "_trace")
    __hash__ = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.spec is None:
            return
        cls.wasm_type = cls.spec.wasm_type
        cls.stack_bits = cls.spec.stack_bits
        cls.bits = cls.spec.bits
        cls.signed = cls.spec.signed
        cls.category = cls.spec.category
        cls.byte_size = cls.spec.byte_size

    def __init__(
        self,
        instructions: tuple[Instruction, ...] = (),
        *,
        module: WebAssembly | None = None,
        trace: tuple[str, ...] = (),
    ) -> None:
        self._instructions = tuple(instructions)
        self._module = module
        self._trace = tuple(trace)

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        return self._instructions

    @property
    def debug_trace(self) -> tuple[str, ...]:
        return self._trace

    def encode(self) -> bytes:
        return b"".join(instruction.encode() for instruction in self._instructions)

    def __str__(self) -> str:
        body = "; ".join(map(str, self._instructions))
        return f"<{type(self).__name__} expr: {body}>"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def local(cls, index: int, *, module: WebAssembly) -> expr:
        value = cls(
            (WasmInstructions.create("local.get", index),),
            module=module,
            trace=(f"local[{index}]",),
        )
        return value._normalized()

    @classmethod
    def constant(cls, value: int | float) -> expr:
        cls._validate_literal(value)
        wat_type = f"{'f' if cls.category == 'float' else 'i'}{cls.stack_bits}"
        encoded_value = value
        if cls.category == "int" and not cls.signed and value > (1 << (cls.stack_bits - 1)) - 1:
            encoded_value = value - (1 << cls.stack_bits)
        instruction = WasmInstructions.create(
            f"{wat_type}.const",
            encoded_value,
            display_name=f"{cls.__name__}.const",
        )
        return cls((instruction,), trace=(repr(encoded_value),))

    @classmethod
    def _validate_literal(cls, value: Any) -> None:
        if cls.category == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ExpressionError(f"{value!r} is not a numeric literal")
            return
        if not isinstance(value, int) or isinstance(value, bool):
            raise ExpressionError(f"{value!r} is not an integer literal")
        minimum = -(1 << (cls.bits - 1)) if cls.signed else 0
        maximum = (1 << (cls.bits - (1 if cls.signed else 0))) - 1
        if not minimum <= value <= maximum:
            raise ExpressionError(f"{value} is outside {cls.__name__}'s range [{minimum}, {maximum}]")

    def cast(self, target: type[expr]) -> expr:
        if not is_numeric_type(target):
            raise ExpressionError("cast target must be a concrete Wasm numeric type")
        if type(self) is target:
            return self

        instructions = list(self._instructions)
        source = type(self)
        name: str | None = None

        if source.category == target.category == "int":
            if source.stack_bits == 64 and target.stack_bits == 32:
                name = "i32.wrap_i64"
            elif source.stack_bits == 32 and target.stack_bits == 64:
                name = "i64.extend_i32_s" if source.signed else "i64.extend_i32_u"
        elif source.category == target.category == "float":
            if source.stack_bits == 64 and target.stack_bits == 32:
                name = "f32.demote_f64"
            elif source.stack_bits == 32 and target.stack_bits == 64:
                name = "f64.promote_f32"
        elif source.category == "int" and target.category == "float":
            name = _INT_TO_FLOAT[(source.stack_bits, target.stack_bits, bool(source.signed))]
        elif source.category == "float" and target.category == "int":
            name = _FLOAT_TO_INT[(source.stack_bits, target.stack_bits, bool(target.signed))]

        if name is not None:
            instructions.append(WasmInstructions.create(name))
        result = target(
            tuple(instructions),
            module=self._module,
            trace=self._trace + (f"cast {target.__name__}",),
        )
        return result._normalized()

    def _normalized(self) -> expr:
        if self.category != "int" or self.bits >= self.stack_bits:
            return self
        instructions = list(self._instructions)
        if self.signed:
            instructions.append(WasmInstructions.create(f"i32.extend{self.bits}_s"))
        else:
            mask = (1 << self.bits) - 1
            instructions.extend(type(self).constant(mask)._instructions)
            instructions.append(WasmInstructions.create("i32.and"))
        return type(self)(
            tuple(instructions),
            module=self._module,
            trace=self._trace + (f"normalize {type(self).__name__}",),
        )

    def _coerce(self, other: object) -> expr:
        return coerce_value(other, type(self))

    def _binary(self, other: object, operation: str) -> expr:
        other_expr = self._coerce(other)
        module = merge_modules(self, other_expr)
        name = self._binary_instruction(operation)
        instructions = self._instructions + other_expr._instructions + (WasmInstructions.create(name),)
        result = type(self)(instructions, module=module, trace=self._trace + other_expr._trace + (operation,))
        return result._normalized()

    def _reflected(self, other: object, operation: str) -> expr:
        left = coerce_value(other, type(self))
        return left._binary(self, operation)

    def _comparison(self, other: object, operation: str) -> expr:
        other_expr = self._coerce(other)
        module = merge_modules(self, other_expr)
        name = self._comparison_instruction(operation)
        instructions = self._instructions + other_expr._instructions + (WasmInstructions.create(name),)
        return i32(instructions, module=module, trace=self._trace + other_expr._trace + (operation,))

    def _binary_instruction(self, operation: str) -> str:
        if self.category == "float":
            if operation not in {"add", "sub", "mul", "div"}:
                raise ExpressionError(f"{operation} is not defined for {type(self).__name__}")
            return f"{type(self).__name__}.{operation}"

        if operation not in {"add", "sub", "mul", "div", "rem", "and", "or", "xor", "shl", "shr"}:
            raise ExpressionError(f"{operation} is not defined for {type(self).__name__}")
        suffix = operation
        if operation in {"div", "rem", "shr"}:
            suffix += "_s" if self.signed else "_u"
        return f"i{self.stack_bits}.{suffix}"

    def _comparison_instruction(self, operation: str) -> str:
        if self.category == "float":
            return f"{type(self).__name__}.{operation}"
        suffix = operation if operation in {"eq", "ne"} else f"{operation}_{'s' if self.signed else 'u'}"
        return f"i{self.stack_bits}.{suffix}"

    def __bool__(self) -> bool:
        raise ExpressionError("Wasm expressions cannot be evaluated as Python booleans")

    def __add__(self, other: object) -> expr: return self._binary(other, "add")
    def __radd__(self, other: object) -> expr: return self._reflected(other, "add")
    def __sub__(self, other: object) -> expr: return self._binary(other, "sub")
    def __rsub__(self, other: object) -> expr: return self._reflected(other, "sub")
    def __mul__(self, other: object) -> expr: return self._binary(other, "mul")
    def __rmul__(self, other: object) -> expr: return self._reflected(other, "mul")
    def __truediv__(self, other: object) -> expr: return self._binary(other, "div")
    def __rtruediv__(self, other: object) -> expr: return self._reflected(other, "div")
    def __floordiv__(self, other: object) -> expr: return self._binary(other, "div")
    def __rfloordiv__(self, other: object) -> expr: return self._reflected(other, "div")
    def __mod__(self, other: object) -> expr: return self._binary(other, "rem")
    def __rmod__(self, other: object) -> expr: return self._reflected(other, "rem")
    def __and__(self, other: object) -> expr: return self._binary(other, "and")
    def __rand__(self, other: object) -> expr: return self._reflected(other, "and")
    def __or__(self, other: object) -> expr: return self._binary(other, "or")
    def __ror__(self, other: object) -> expr: return self._reflected(other, "or")
    def __xor__(self, other: object) -> expr: return self._binary(other, "xor")
    def __rxor__(self, other: object) -> expr: return self._reflected(other, "xor")
    def __lshift__(self, other: object) -> expr: return self._binary(other, "shl")
    def __rlshift__(self, other: object) -> expr: return self._reflected(other, "shl")
    def __rshift__(self, other: object) -> expr: return self._binary(other, "shr")
    def __rrshift__(self, other: object) -> expr: return self._reflected(other, "shr")
    def __eq__(self, other: object) -> expr: return self._comparison(other, "eq")
    def __ne__(self, other: object) -> expr: return self._comparison(other, "ne")
    def __lt__(self, other: object) -> expr: return self._comparison(other, "lt")
    def __le__(self, other: object) -> expr: return self._comparison(other, "le")
    def __gt__(self, other: object) -> expr: return self._comparison(other, "gt")
    def __ge__(self, other: object) -> expr: return self._comparison(other, "ge")

    def __pos__(self) -> expr:
        return self

    def __neg__(self) -> expr:
        if self.category == "float":
            name = f"{type(self).__name__}.neg"
            return type(self)(
                self._instructions + (WasmInstructions.create(name),),
                module=self._module,
                trace=self._trace + ("neg",),
            )
        return type(self).constant(0)._binary(self, "sub")

    def __invert__(self) -> expr:
        if self.category != "int":
            raise ExpressionError(f"invert is not defined for {type(self).__name__}")
        return self._binary(-1 if self.signed else (1 << self.bits) - 1, "xor")


class _integer(expr):
    pass


class _float(expr):
    pass


class i8(_integer):
    spec = NumericTypeSpec(0x7F, 32, 8, "int", True, 1, "i32.load8_s", "i32.store8", 1)


class u8(_integer):
    spec = NumericTypeSpec(0x7F, 32, 8, "int", False, 1, "i32.load8_u", "i32.store8", 1)


class i16(_integer):
    spec = NumericTypeSpec(0x7F, 32, 16, "int", True, 2, "i32.load16_s", "i32.store16", 2)


class u16(_integer):
    spec = NumericTypeSpec(0x7F, 32, 16, "int", False, 2, "i32.load16_u", "i32.store16", 2)


class i32(_integer):
    spec = NumericTypeSpec(0x7F, 32, 32, "int", True, 4, "i32.load", "i32.store", 4)


class u32(_integer):
    spec = NumericTypeSpec(0x7F, 32, 32, "int", False, 4, "i32.load", "i32.store", 4)


class i64(_integer):
    spec = NumericTypeSpec(0x7E, 64, 64, "int", True, 8, "i64.load", "i64.store", 8)


class u64(_integer):
    spec = NumericTypeSpec(0x7E, 64, 64, "int", False, 8, "i64.load", "i64.store", 8)


class f32(_float):
    spec = NumericTypeSpec(0x7D, 32, 32, "float", None, 4, "f32.load", "f32.store", 4)


class f64(_float):
    spec = NumericTypeSpec(0x7C, 64, 64, "float", None, 8, "f64.load", "f64.store", 8)


def is_numeric_type(value: object) -> bool:
    return (
        isinstance(value, type)
        and issubclass(value, expr)
        and value is not expr
        and value.spec is not None
    )


def coerce_value(value: object, target: type[expr]) -> expr:
    if not is_numeric_type(target):
        raise ExpressionError("target must be a concrete Wasm numeric type")
    if isinstance(value, expr):
        return value.cast(target)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return target.constant(value)
    raise ExpressionError(f"cannot use {type(value).__name__} with {target.__name__}")


def merge_modules(left: expr, right: expr) -> WebAssembly | None:
    if left._module is not None and right._module is not None and left._module is not right._module:
        raise ExpressionError("expressions from different WebAssembly modules cannot be combined")
    return left._module or right._module


_INT_TO_FLOAT = {
    (32, 32, True): "f32.convert_i32_s",
    (32, 32, False): "f32.convert_i32_u",
    (64, 32, True): "f32.convert_i64_s",
    (64, 32, False): "f32.convert_i64_u",
    (32, 64, True): "f64.convert_i32_s",
    (32, 64, False): "f64.convert_i32_u",
    (64, 64, True): "f64.convert_i64_s",
    (64, 64, False): "f64.convert_i64_u",
}

_FLOAT_TO_INT = {
    (32, 32, True): "i32.trunc_f32_s",
    (32, 32, False): "i32.trunc_f32_u",
    (64, 32, True): "i32.trunc_f64_s",
    (64, 32, False): "i32.trunc_f64_u",
    (32, 64, True): "i64.trunc_f32_s",
    (32, 64, False): "i64.trunc_f32_u",
    (64, 64, True): "i64.trunc_f64_s",
    (64, 64, False): "i64.trunc_f64_u",
}
