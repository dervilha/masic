from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from .encoding import encode_f32, encode_f64, encode_s32, encode_s64, encode_u32

ImmediateKind = Literal["none", "u32", "s32", "s64", "f32", "f64", "block", "memarg", "memidx"]


@dataclass(frozen=True, slots=True)
class InstructionSpec:
    opcode: int
    immediate: ImmediateKind = "none"


class WasmInstructions:
    """Single source of truth for supported WAT instructions and binary encodings."""

    _SPECS: ClassVar[dict[str, InstructionSpec]] = {
        # Parametric and control instructions
        "unreachable": InstructionSpec(0x00),
        "nop": InstructionSpec(0x01),
        "block": InstructionSpec(0x02, "block"),
        "loop": InstructionSpec(0x03, "block"),
        "if": InstructionSpec(0x04, "block"),
        "else": InstructionSpec(0x05),
        "end": InstructionSpec(0x0B),
        "br": InstructionSpec(0x0C, "u32"),
        "br_if": InstructionSpec(0x0D, "u32"),
        "return": InstructionSpec(0x0F),
        "call": InstructionSpec(0x10, "u32"),
        "drop": InstructionSpec(0x1A),
        "select": InstructionSpec(0x1B),
        # Variables
        "local.get": InstructionSpec(0x20, "u32"),
        "local.set": InstructionSpec(0x21, "u32"),
        "local.tee": InstructionSpec(0x22, "u32"),
        "global.get": InstructionSpec(0x23, "u32"),
        "global.set": InstructionSpec(0x24, "u32"),
        # Memory
        "i32.load": InstructionSpec(0x28, "memarg"),
        "i64.load": InstructionSpec(0x29, "memarg"),
        "f32.load": InstructionSpec(0x2A, "memarg"),
        "f64.load": InstructionSpec(0x2B, "memarg"),
        "i32.load8_s": InstructionSpec(0x2C, "memarg"),
        "i32.load8_u": InstructionSpec(0x2D, "memarg"),
        "i32.load16_s": InstructionSpec(0x2E, "memarg"),
        "i32.load16_u": InstructionSpec(0x2F, "memarg"),
        "i64.load8_s": InstructionSpec(0x30, "memarg"),
        "i64.load8_u": InstructionSpec(0x31, "memarg"),
        "i64.load16_s": InstructionSpec(0x32, "memarg"),
        "i64.load16_u": InstructionSpec(0x33, "memarg"),
        "i64.load32_s": InstructionSpec(0x34, "memarg"),
        "i64.load32_u": InstructionSpec(0x35, "memarg"),
        "i32.store": InstructionSpec(0x36, "memarg"),
        "i64.store": InstructionSpec(0x37, "memarg"),
        "f32.store": InstructionSpec(0x38, "memarg"),
        "f64.store": InstructionSpec(0x39, "memarg"),
        "i32.store8": InstructionSpec(0x3A, "memarg"),
        "i32.store16": InstructionSpec(0x3B, "memarg"),
        "i64.store8": InstructionSpec(0x3C, "memarg"),
        "i64.store16": InstructionSpec(0x3D, "memarg"),
        "i64.store32": InstructionSpec(0x3E, "memarg"),
        "memory.size": InstructionSpec(0x3F, "memidx"),
        "memory.grow": InstructionSpec(0x40, "memidx"),
        # Constants
        "i32.const": InstructionSpec(0x41, "s32"),
        "i64.const": InstructionSpec(0x42, "s64"),
        "f32.const": InstructionSpec(0x43, "f32"),
        "f64.const": InstructionSpec(0x44, "f64"),
        # Numeric comparisons, arithmetic, conversions, and sign extension
        **{
            name: InstructionSpec(opcode)
            for opcode, name in enumerate(
                (
                    "i32.eqz", "i32.eq", "i32.ne", "i32.lt_s", "i32.lt_u", "i32.gt_s",
                    "i32.gt_u", "i32.le_s", "i32.le_u", "i32.ge_s", "i32.ge_u", "i64.eqz",
                    "i64.eq", "i64.ne", "i64.lt_s", "i64.lt_u", "i64.gt_s", "i64.gt_u",
                    "i64.le_s", "i64.le_u", "i64.ge_s", "i64.ge_u", "f32.eq", "f32.ne",
                    "f32.lt", "f32.gt", "f32.le", "f32.ge", "f64.eq", "f64.ne", "f64.lt",
                    "f64.gt", "f64.le", "f64.ge", "i32.clz", "i32.ctz", "i32.popcnt",
                    "i32.add", "i32.sub", "i32.mul", "i32.div_s", "i32.div_u", "i32.rem_s",
                    "i32.rem_u", "i32.and", "i32.or", "i32.xor", "i32.shl", "i32.shr_s",
                    "i32.shr_u", "i32.rotl", "i32.rotr", "i64.clz", "i64.ctz", "i64.popcnt",
                    "i64.add", "i64.sub", "i64.mul", "i64.div_s", "i64.div_u", "i64.rem_s",
                    "i64.rem_u", "i64.and", "i64.or", "i64.xor", "i64.shl", "i64.shr_s",
                    "i64.shr_u", "i64.rotl", "i64.rotr", "f32.abs", "f32.neg", "f32.ceil",
                    "f32.floor", "f32.trunc", "f32.nearest", "f32.sqrt", "f32.add", "f32.sub",
                    "f32.mul", "f32.div", "f32.min", "f32.max", "f32.copysign", "f64.abs",
                    "f64.neg", "f64.ceil", "f64.floor", "f64.trunc", "f64.nearest", "f64.sqrt",
                    "f64.add", "f64.sub", "f64.mul", "f64.div", "f64.min", "f64.max",
                    "f64.copysign", "i32.wrap_i64", "i32.trunc_f32_s", "i32.trunc_f32_u",
                    "i32.trunc_f64_s", "i32.trunc_f64_u", "i64.extend_i32_s", "i64.extend_i32_u",
                    "i64.trunc_f32_s", "i64.trunc_f32_u", "i64.trunc_f64_s", "i64.trunc_f64_u",
                    "f32.convert_i32_s", "f32.convert_i32_u", "f32.convert_i64_s",
                    "f32.convert_i64_u", "f32.demote_f64", "f64.convert_i32_s", "f64.convert_i32_u",
                    "f64.convert_i64_s", "f64.convert_i64_u", "f64.promote_f32", "i32.reinterpret_f32",
                    "i64.reinterpret_f64", "f32.reinterpret_i32", "f64.reinterpret_i64",
                    "i32.extend8_s", "i32.extend16_s", "i64.extend8_s", "i64.extend16_s",
                    "i64.extend32_s",
                ),
                start=0x45,
            )
        },
    }

    @classmethod
    def spec(cls, wat: str) -> InstructionSpec:
        try:
            return cls._SPECS[wat]
        except KeyError as error:
            raise ValueError(f"unsupported WebAssembly instruction {wat!r}") from error

    @classmethod
    def create(
        cls,
        wat: str,
        argument: int | float | None = None,
        *,
        alignment: int | None = None,
        offset: int = 0,
        result_type: int | None = None,
        display_name: str | None = None,
    ) -> Instruction:
        spec = cls.spec(wat)
        immediate = cls._encode_immediate(spec.immediate, argument, alignment, offset, result_type)
        debug_argument: object | None = argument
        if spec.immediate == "memarg":
            debug_argument = (alignment, offset)
        elif spec.immediate == "block" and result_type is None:
            debug_argument = None
        return Instruction(spec.opcode, display_name or wat, immediate, debug_argument)

    @staticmethod
    def _encode_immediate(
        kind: ImmediateKind,
        argument: int | float | None,
        alignment: int | None,
        offset: int,
        result_type: int | None,
    ) -> bytes:
        if kind == "none":
            if argument is not None or alignment is not None or offset:
                raise ValueError("instruction does not accept an immediate")
            return b""
        if kind == "block":
            if argument is not None or alignment is not None or offset:
                raise ValueError("block instruction accepts only result_type")
            return b"\x40" if result_type is None else bytes((result_type,))
        if kind == "memarg":
            if argument is not None or alignment is None:
                raise ValueError("memory instruction requires alignment and optional offset")
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError("alignment must be a positive power of two")
            return encode_u32(alignment.bit_length() - 1) + encode_u32(offset)
        if kind == "memidx":
            index = 0 if argument is None else argument
            return encode_u32(index)
        if argument is None:
            raise ValueError(f"instruction requires a {kind} immediate")
        encoders = {
            "u32": encode_u32,
            "s32": encode_s32,
            "s64": encode_s64,
            "f32": encode_f32,
            "f64": encode_f64,
        }
        return encoders[kind](argument)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Instruction:
    opcode: int
    name: str
    immediate: bytes = b""
    argument: object | None = None

    def encode(self) -> bytes:
        return bytes((self.opcode,)) + self.immediate

    def __str__(self) -> str:
        return self.name if self.argument is None else f"{self.name} {self.argument}"
