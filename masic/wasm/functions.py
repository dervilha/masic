from __future__ import annotations

import inspect
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar, get_type_hints

from .errors import CompileError
from .instructions import Instruction, WasmInstructions
from .types import coerce_value, expr, is_numeric_type

if TYPE_CHECKING:
    from .module import WebAssembly

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class FunctionBody:
    instructions: tuple[Instruction, ...]
    local_types: tuple[type[expr], ...] = ()

    def encode(self) -> bytes:
        return b"".join(instruction.encode() for instruction in self.instructions)


@dataclass(frozen=True, slots=True)
class LocalRef:
    index: int
    value_type: type[expr]
    name: str | None = None


_ACTIVE_FUNCTION: ContextVar[FunctionBuilder | None] = ContextVar("masic_active_function", default=None)


class FunctionBuilder:
    """The sole builder for user and internal function bodies."""

    def __init__(self, module: WebAssembly | None, parameter_types: tuple[type[expr], ...]) -> None:
        self.module = module
        self.parameter_types = parameter_types
        self.local_types: list[type[expr]] = []
        self.statements: list[Instruction] = []
        self._token: Token[FunctionBuilder | None] | None = None

    def __enter__(self) -> FunctionBuilder:
        if self.module is None:
            return self
        if _ACTIVE_FUNCTION.get() is not None:
            raise CompileError("nested Wasm function declarations are not supported")
        self._token = _ACTIVE_FUNCTION.set(self)
        return self

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _ACTIVE_FUNCTION.reset(self._token)
            self._token = None

    def parameter(self, index: int, name: str | None = None) -> LocalRef:
        try:
            value_type = self.parameter_types[index]
        except IndexError as error:
            raise ValueError(f"parameter index {index} is out of range") from error
        return LocalRef(index, value_type, name)

    def local(self, value_type: type[expr], name: str | None = None) -> LocalRef:
        if not is_numeric_type(value_type):
            raise CompileError("local type must be a concrete Wasm numeric type")
        reference = LocalRef(len(self.parameter_types) + len(self.local_types), value_type, name)
        self.local_types.append(value_type)
        return reference

    def emit(self, *instructions: Instruction) -> None:
        self.statements.extend(instructions)

    def instruction(
        self,
        wat: str,
        argument: int | float | None = None,
        *,
        alignment: int | None = None,
        offset: int = 0,
        result_type: int | None = None,
    ) -> Instruction:
        return WasmInstructions.create(
            wat,
            argument,
            alignment=alignment,
            offset=offset,
            result_type=result_type,
        )

    def get(self, local: LocalRef) -> Instruction:
        return WasmInstructions.create("local.get", local.index)

    def set(self, local: LocalRef) -> Instruction:
        return WasmInstructions.create("local.set", local.index)

    def materialize(self, value: expr, name: str | None = None) -> expr:
        if self.module is None:
            raise CompileError("expressions can only be materialized in a module function")
        if value._module is not None and value._module is not self.module:
            raise CompileError("cannot materialize an expression from another module")
        local = self.local(type(value), name)
        self.emit(*value.instructions, self.set(local))
        return type(value).local(local.index, module=self.module)

    def finish(self, result: expr | None = None) -> FunctionBody:
        instructions = tuple(self.statements)
        if result is not None:
            instructions += result.instructions
        return FunctionBody(instructions, tuple(self.local_types))


def active_function(*, module: WebAssembly | None = None) -> FunctionBuilder:
    builder = _ACTIVE_FUNCTION.get()
    if builder is None:
        raise CompileError("this operation can only be used while declaring a Wasm function")
    if module is not None and builder.module is not module:
        raise CompileError("operation belongs to another WebAssembly module")
    return builder


@dataclass(frozen=True, slots=True)
class FunctionSignature:
    parameter_names: tuple[str, ...]
    parameter_types: tuple[type[expr], ...]
    return_type: type[expr] | None
    defaults: tuple[tuple[str, Any], ...] = ()

    @property
    def defaults_by_name(self) -> dict[str, Any]:
        return dict(self.defaults)

    @property
    def wasm_key(self) -> tuple[tuple[int, ...], int | None]:
        result = None if self.return_type is None else self.return_type.wasm_type
        return tuple(item.wasm_type for item in self.parameter_types), result


@dataclass(frozen=True, slots=True)
class FunctionDeclaration:
    index: int
    name: str
    signature: FunctionSignature
    body: FunctionBody
    export: bool


class FunctionCompiler:
    """Compiles one decorated Python callable into a registered function."""

    def __init__(self, module: WebAssembly) -> None:
        self.module = module

    def compile(self, function: F, *, export: bool) -> func:
        if self.module._has_function(function.__name__):
            raise CompileError(f"function {function.__name__!r} is already registered")

        python_signature = inspect.signature(function)
        try:
            annotations = get_type_hints(function)
        except Exception as error:
            raise CompileError(f"could not resolve annotations for {function.__name__}(): {error}") from error

        parameter_names: list[str] = []
        parameter_types: list[type[expr]] = []
        defaults: list[tuple[str, Any]] = []
        arguments: list[expr] = []

        for index, (parameter_name, parameter) in enumerate(python_signature.parameters.items()):
            if parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
                raise CompileError(f"{function.__name__}(): variadic and keyword-only parameters are not supported")
            annotation = annotations.get(parameter_name)
            self._validate_type(annotation, f"parameter {parameter_name!r} of {function.__name__}()")
            parameter_names.append(parameter_name)
            parameter_types.append(annotation)
            arguments.append(annotation.local(index, module=self.module))
            if parameter.default is not inspect.Parameter.empty:
                annotation.constant(parameter.default)
                defaults.append((parameter_name, parameter.default))

        return_type = annotations.get("return")
        self._validate_type(return_type, f"return value of {function.__name__}()")
        signature = FunctionSignature(tuple(parameter_names), tuple(parameter_types), return_type, tuple(defaults))

        try:
            with FunctionBuilder(self.module, tuple(parameter_types)) as builder:
                returned = function(*arguments)
        except CompileError:
            raise
        except Exception as error:
            raise CompileError(f"failed while building {function.__name__}(): {error}") from error

        if not isinstance(returned, expr) and (
            not isinstance(returned, (int, float)) or isinstance(returned, bool)
        ):
            raise CompileError(
                f"{function.__name__}() must return a Wasm expression or numeric literal, "
                f"got {type(returned).__name__}"
            )
        result = coerce_value(returned, return_type)
        if result._module is not None and result._module is not self.module:
            raise CompileError(f"{function.__name__}() returned an expression from another module")

        declaration = FunctionDeclaration(
            index=self.module._next_function_index(),
            name=function.__name__,
            signature=signature,
            body=builder.finish(result),
            export=export,
        )
        self.module._register_function(declaration)
        return func(self.module, declaration, function)

    @staticmethod
    def _validate_type(annotation: object, location: str) -> None:
        if not is_numeric_type(annotation):
            raise CompileError(f"{location} must have a concrete Wasm numeric type annotation")


class func:
    """A registered Wasm function that creates a call expression when invoked."""

    __slots__ = ("_module", "_declaration", "__name__", "__qualname__")

    def __init__(self, module: WebAssembly, declaration: FunctionDeclaration, original: Any) -> None:
        self._module = module
        self._declaration = declaration
        self.__name__ = declaration.name
        self.__qualname__ = getattr(original, "__qualname__", declaration.name)

    @property
    def index(self) -> int:
        return self._declaration.index

    @property
    def signature(self) -> FunctionSignature:
        return self._declaration.signature

    def __call__(self, *args: object, **kwargs: object) -> expr:
        if self.signature.return_type is None:
            raise CompileError(f"void function {self.__name__}() cannot be used as an expression")
        parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=self.signature.defaults_by_name.get(name, inspect.Parameter.empty),
            )
            for name in self.signature.parameter_names
        ]
        call_signature = inspect.Signature(parameters)
        try:
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as error:
            raise CompileError(f"invalid call to {self.__name__}(): {error}") from error

        instructions: tuple[Instruction, ...] = ()
        trace: tuple[str, ...] = ()
        for name, expected_type in zip(self.signature.parameter_names, self.signature.parameter_types):
            value = bound.arguments[name]
            argument = coerce_value(value, expected_type)
            if argument._module is not None and argument._module is not self._module:
                raise CompileError(f"argument {name!r} to {self.__name__}() belongs to another module")
            instructions += argument.instructions
            trace += argument.debug_trace

        instruction = WasmInstructions.create("call", self.index)
        return self.signature.return_type(
            instructions + (instruction,),
            module=self._module,
            trace=trace + (f"call {self.__name__}",),
        )

    def __repr__(self) -> str:
        return f"<func {self.__name__} index={self.index}>"
