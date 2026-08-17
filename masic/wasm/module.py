from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, TypeVar

from .encoding import encode_name, encode_section, encode_u32, encode_vector
from .errors import CompileError, ModuleStateError
from .functions import FunctionCompiler, FunctionDeclaration, FunctionSignature, func
from .instructions import WasmInstructions

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class MemoryRequirement:
    initial_pages: int
    mutable_i32_globals: tuple[int, ...] = ()
    export: bool = False


class WebAssembly:
    """A context-managed builder for a WebAssembly binary module."""

    MAGIC_AND_VERSION = b"\x00asm\x01\x00\x00\x00"

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("module name must be a non-empty string")
        self.name = name
        self._functions: list[FunctionDeclaration] = []
        self._function_names: set[str] = set()
        self._bytecode: bytes | None = None
        self._closed = False
        self._inside_context = False
        self._requirements: dict[str, object] = {}
        self._internal_dependencies: dict[str, tuple[int, ...]] = {}

    def __enter__(self) -> WebAssembly:
        if self._closed or self._inside_context:
            raise ModuleStateError("WebAssembly builders cannot be re-entered")
        self._inside_context = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._inside_context = False
        if exc_type is None:
            self._assemble()
        return False

    def func(self, export: bool = False) -> Callable[[F], func]:
        if not isinstance(export, bool):
            raise TypeError("export must be a bool")

        def decorator(function: F) -> func:
            if self._closed:
                raise ModuleStateError("functions cannot be added after assembly")
            return FunctionCompiler(self).compile(function, export=export)

        return decorator

    def _has_function(self, name: str) -> bool:
        return name in self._function_names

    def _next_function_index(self) -> int:
        return len(self._functions)

    def _register_function(self, declaration: FunctionDeclaration) -> None:
        if declaration.index != self._next_function_index():
            raise CompileError("function index does not match module declaration order")
        if self._has_function(declaration.name):
            raise CompileError(f"function {declaration.name!r} is already registered")
        self._functions.append(declaration)
        self._function_names.add(declaration.name)

    @property
    def functions(self) -> tuple[FunctionDeclaration, ...]:
        return tuple(self._functions)

    @property
    def bytecode(self) -> bytes:
        if self._bytecode is None:
            raise ModuleStateError("module has not been assembled yet")
        return self._bytecode

    @property
    def data(self) -> bytes:
        return self.bytecode

    def _install_internal_dependency(
        self,
        name: str,
        factory: Callable[[int], tuple[FunctionDeclaration, ...]],
        *,
        memory: MemoryRequirement | None = None,
    ) -> tuple[int, ...]:
        if name in self._internal_dependencies:
            return self._internal_dependencies[name]
        if self._closed:
            raise ModuleStateError("internal dependencies cannot be added after assembly")
        declarations = factory(self._next_function_index())
        for declaration in declarations:
            self._register_function(declaration)
        if memory is not None:
            existing = self._requirements.get("memory")
            if existing is not None and existing != memory:
                raise CompileError("conflicting linear-memory requirements")
            self._requirements["memory"] = memory
        indices = tuple(declaration.index for declaration in declarations)
        self._internal_dependencies[name] = indices
        return indices

    def __bytes__(self) -> bytes:
        return self.bytecode

    def _assemble(self) -> bytes:
        if self._bytecode is not None:
            return self._bytecode

        self._bytecode = _ModuleEncoder(self).encode()
        self._closed = True
        return self._bytecode

    def save(self, filepath: str | Path) -> Path:
        if self._bytecode is None:
            self._assemble()
        path = Path(filepath)
        path.write_bytes(self.bytecode)
        return path.resolve()

    def __repr__(self) -> str:
        state = "compiled" if self._bytecode is not None else "building"
        return f"<WebAssembly {self.name!r} {state}, functions={len(self._functions)}>"


class _ModuleEncoder:
    """Owns binary section construction for a completed module model."""

    def __init__(self, module: WebAssembly) -> None:
        self.module = module

    def encode(self) -> bytes:
        functions = self.module._functions
        signature_indices: dict[tuple[tuple[int, ...], int | None], int] = {}
        signatures: list[FunctionSignature] = []
        function_type_indices: list[int] = []
        for declaration in functions:
            key = declaration.signature.wasm_key
            if key not in signature_indices:
                signature_indices[key] = len(signatures)
                signatures.append(declaration.signature)
            function_type_indices.append(signature_indices[key])

        sections: list[bytes] = []
        if signatures:
            type_entries = []
            for signature in signatures:
                parameters = encode_vector(bytes((item.wasm_type,)) for item in signature.parameter_types)
                results = encode_vector(()) if signature.return_type is None else encode_vector(
                    (bytes((signature.return_type.wasm_type,)),)
                )
                type_entries.append(b"\x60" + parameters + results)
            sections.append(encode_section(1, encode_vector(type_entries)))
            sections.append(encode_section(3, encode_vector(encode_u32(index) for index in function_type_indices)))

        memory = self.module._requirements.get("memory")
        if memory is not None:
            assert isinstance(memory, MemoryRequirement)
            sections.append(encode_section(5, encode_vector((b"\x00" + encode_u32(memory.initial_pages),))))
            globals_ = []
            for initial_value in memory.mutable_i32_globals:
                initializer = (
                    WasmInstructions.create("i32.const", initial_value).encode()
                    + WasmInstructions.create("end").encode()
                )
                globals_.append(b"\x7f\x01" + initializer)
            if globals_:
                sections.append(encode_section(6, encode_vector(globals_)))

        exports = [declaration for declaration in functions if declaration.export]
        if exports or (memory is not None and memory.export):
            entries = [
                encode_name(declaration.name) + b"\x00" + encode_u32(declaration.index)
                for declaration in exports
            ]
            if memory is not None and memory.export:
                entries.append(encode_name("memory") + b"\x02" + encode_u32(0))
            sections.append(encode_section(7, encode_vector(entries)))

        if functions:
            bodies = []
            for declaration in functions:
                local_groups: list[bytes] = []
                for value_type in declaration.body.local_types:
                    encoded = encode_u32(1) + bytes((value_type.wasm_type,))
                    local_groups.append(encoded)
                terminator = WasmInstructions.create("return").encode() + WasmInstructions.create("end").encode()
                body = encode_vector(local_groups) + declaration.body.encode() + terminator
                bodies.append(encode_u32(len(body)) + body)
            sections.append(encode_section(10, encode_vector(bodies)))

        return WebAssembly.MAGIC_AND_VERSION + b"".join(sections)
