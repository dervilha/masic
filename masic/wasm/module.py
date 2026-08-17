from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, TypeVar

from .encoding import encode_name, encode_section, encode_u32, encode_vector
from .errors import CompileError, ExpressionError, ModuleStateError
from .functions import (
    CallEffect,
    FunctionCompiler,
    FunctionDeclaration,
    FunctionSignature,
    ImportDeclaration,
    active_function,
    func,
)
from .instructions import WasmInstructions
from .types import coerce_value, expr, is_numeric_type, u32

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class MemoryRequirement:
    initial_pages: int
    export: bool = False


@dataclass(frozen=True, slots=True)
class GlobalDeclaration:
    index: int
    value_type: type[expr]
    initial: int | float
    mutable: bool
    export_name: str | None = None


@dataclass(frozen=True, slots=True)
class TableDeclaration:
    index: int
    signature: FunctionSignature
    function_indices: tuple[int, ...]
    export_name: str | None = None


class global_value:
    """A Wasm-owned scalar global, readable and writable as ``.value`` in source."""

    __slots__ = ("_module", "_declaration")

    def __init__(self, module: WebAssembly, declaration: GlobalDeclaration) -> None:
        self._module = module
        self._declaration = declaration

    @property
    def value(self) -> expr:
        active_function(module=self._module)
        return self._declaration.value_type(
            (WasmInstructions.create("global.get", self._declaration.index),),
            module=self._module,
            trace=(f"global[{self._declaration.index}]",),
        )

    @value.setter
    def value(self, value: object) -> None:
        if not self._declaration.mutable:
            raise ExpressionError("cannot assign to an immutable Wasm global")
        builder = active_function(module=self._module)
        stored = coerce_value(value, self._declaration.value_type)
        if stored._module is not None and stored._module is not self._module:
            raise ExpressionError("global value belongs to another WebAssembly module")
        builder.emit(*stored.instructions, WasmInstructions.create("global.set", self._declaration.index))


class table:
    """An initialized homogeneous function table for indirect calls."""

    __slots__ = ("_module", "_declaration")

    def __init__(self, module: WebAssembly, declaration: TableDeclaration) -> None:
        self._module = module
        self._declaration = declaration

    def call(self, index: object, *args: object, **kwargs: object) -> expr | CallEffect:
        if kwargs:
            raise CompileError("indirect table calls accept positional arguments only")
        if len(args) != len(self._declaration.signature.parameter_types):
            raise CompileError(
                f"indirect table call expects {len(self._declaration.signature.parameter_types)} arguments, got {len(args)}"
            )
        if isinstance(index, int) and not isinstance(index, bool):
            index_value = u32.constant(index)
        elif isinstance(index, expr) and index.category == "int":
            index_value = index.cast(u32)
        else:
            raise ExpressionError("table index must be an integer or Wasm integer expression")
        if index_value._module is not None and index_value._module is not self._module:
            raise ExpressionError("table index belongs to another WebAssembly module")
        instructions = ()
        trace = ()
        for value, value_type in zip(args, self._declaration.signature.parameter_types):
            argument = coerce_value(value, value_type)
            if argument._module is not None and argument._module is not self._module:
                raise ExpressionError("table argument belongs to another WebAssembly module")
            instructions += argument.instructions
            trace += argument.debug_trace
        instructions += index_value.instructions
        instructions += (
            WasmInstructions.create(
                "call_indirect",
                (self._module._signature_index(self._declaration.signature), self._declaration.index),
            ),
        )
        if self._declaration.signature.return_type is None:
            return CallEffect(instructions, trace + ("call_indirect",))
        return self._declaration.signature.return_type(
            instructions,
            module=self._module,
            trace=trace + ("call_indirect",),
        )


class WebAssembly:
    """A context-managed builder for a WebAssembly binary module."""

    MAGIC_AND_VERSION = b"\x00asm\x01\x00\x00\x00"

    def __init__(self, name: str = "module", *, debug: bool = False) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("module name must be a non-empty string")
        if not isinstance(debug, bool):
            raise TypeError("debug must be a bool")
        self.name = name
        self.debug = debug
        self._imports: list[ImportDeclaration] = []
        self._functions: list[FunctionDeclaration] = []
        self._globals: list[GlobalDeclaration] = []
        self._tables: list[TableDeclaration] = []
        self._signatures: list[FunctionSignature] = []
        self._signature_indices: dict[tuple[tuple[int, ...], int | None], int] = {}
        self._start_index: int | None = None
        self._allocator_global_index: int | None = None
        self._function_names: set[str] = set()
        self._bytecode: bytes | None = None
        self._closed = False
        self._inside_context = False
        self._requirements: dict[str, object] = {}
        self._internal_dependencies: dict[str, tuple[int, ...]] = {}
        self._included_libraries: dict[WebAssembly, object] = {}
        self._static_data: dict[tuple[bytes, int], int] = {}
        self._static_segments: list[tuple[int, bytes]] = []
        self._static_end = 0

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

    def import_func(self, module: str, *, name: str | None = None) -> Callable[[F], func]:
        """Declare a host-provided Wasm function before local functions.

        ``module`` and ``name`` are the external import module and field names;
        omit ``name`` to use the Python function name.
        """
        if not isinstance(module, str) or not module:
            raise TypeError("import module must be a non-empty string")
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError("import name must be a non-empty string when provided")

        def decorator(function: F) -> func:
            if self._closed:
                raise ModuleStateError("imports cannot be added after assembly")
            if self._functions:
                raise CompileError("imports must be declared before local Wasm functions")
            return FunctionCompiler(self).compile_import(function, module_name=module, field_name=name)

        return decorator

    def start(self, function: F) -> func:
        """Declare the single no-argument, void function run at instantiation."""
        if self._start_index is not None:
            raise CompileError("a module can declare only one start function")
        compiler = FunctionCompiler(self)
        signature = compiler._signature(function)
        if signature.parameter_types or signature.return_type is not None:
            raise CompileError("@wasm.start must declare () -> None")
        compiled = compiler.compile(function, export=False)
        self._start_index = compiled.index
        return compiled

    def create_global(
        self,
        value_type: type[expr],
        initial: int | float,
        *,
        mutable: bool = False,
        export: str | None = None,
    ) -> global_value:
        """Create a persistent Wasm scalar global for use through ``.value``."""
        if self._closed:
            raise ModuleStateError("globals cannot be added after assembly")
        if not is_numeric_type(value_type):
            raise TypeError("global type must be a concrete Wasm numeric type")
        if not isinstance(mutable, bool):
            raise TypeError("mutable must be a bool")
        self._validate_export_name(export)
        value_type.constant(initial)
        declaration = GlobalDeclaration(len(self._globals), value_type, initial, mutable, export)
        self._globals.append(declaration)
        return global_value(self, declaration)

    def create_table(self, functions: list[func] | tuple[func, ...], *, export: str | None = None) -> table:
        """Create one initialized homogeneous table for indirect calls."""
        if self._closed:
            raise ModuleStateError("tables cannot be added after assembly")
        if self._tables:
            raise CompileError("this MASIC version supports one function table")
        if not isinstance(functions, (list, tuple)) or not functions:
            raise TypeError("table functions must be a non-empty list or tuple")
        if not all(isinstance(item, func) for item in functions):
            raise TypeError("table entries must be MASIC functions")
        if any(item._module is not self for item in functions):
            raise CompileError("table entries belong to another WebAssembly module")
        signature = functions[0].signature
        if any(item.signature.wasm_key != signature.wasm_key for item in functions[1:]):
            raise CompileError("table entries must share one Wasm signature")
        self._validate_export_name(export)
        declaration = TableDeclaration(0, signature, tuple(item.index for item in functions), export)
        self._tables.append(declaration)
        self._signature_index(signature)
        return table(self, declaration)

    def include(self, library: WebAssembly) -> object:
        """Bind an includeable library namespace to this open module once."""
        if not isinstance(library, WebAssembly):
            raise TypeError("library must be a WebAssembly namespace")
        if self._closed:
            raise ModuleStateError("libraries cannot be included after assembly")
        if library is self:
            raise CompileError("a module cannot include itself")
        try:
            return self._included_libraries[library]
        except KeyError:
            pass
        create_namespace = getattr(library, "_create_namespace", None)
        include_into = getattr(library, "_include_into", None)
        if not callable(create_namespace) or not callable(include_into):
            raise CompileError(f"{library.name!r} is not an includeable library namespace")
        namespace = create_namespace(self)
        self._included_libraries[library] = namespace
        try:
            include_into(self, namespace)
        except BaseException:
            del self._included_libraries[library]
            raise
        return namespace

    def _has_function(self, name: str) -> bool:
        return name in self._function_names

    def _next_function_index(self) -> int:
        return len(self._imports) + len(self._functions)

    def _next_import_index(self) -> int:
        return len(self._imports)

    def _register_import(self, declaration: ImportDeclaration) -> None:
        if declaration.index != self._next_import_index():
            raise CompileError("import index does not match module declaration order")
        if self._has_function(declaration.name):
            raise CompileError(f"function {declaration.name!r} is already registered")
        self._imports.append(declaration)
        self._function_names.add(declaration.name)
        self._signature_index(declaration.signature)

    def _register_function(self, declaration: FunctionDeclaration) -> None:
        if declaration.index != self._next_function_index():
            raise CompileError("function index does not match module declaration order")
        if self._has_function(declaration.name):
            raise CompileError(f"function {declaration.name!r} is already registered")
        if declaration.export:
            self._validate_export_name(declaration.name)
        self._functions.append(declaration)
        self._function_names.add(declaration.name)
        self._signature_index(declaration.signature)

    def _signature_index(self, signature: FunctionSignature) -> int:
        key = signature.wasm_key
        try:
            return self._signature_indices[key]
        except KeyError:
            index = len(self._signatures)
            self._signature_indices[key] = index
            self._signatures.append(signature)
            return index

    def _validate_export_name(self, name: str | None) -> None:
        if name is None:
            return
        if not isinstance(name, str) or not name:
            raise TypeError("export must be a non-empty string when provided")
        existing = [item.export_name for item in (*self._globals, *self._tables)]
        existing += [item.name for item in self._functions if item.export]
        memory = self._requirements.get("memory")
        if isinstance(memory, MemoryRequirement) and memory.export:
            existing.append("memory")
        if name in existing:
            raise CompileError(f"export {name!r} is already registered")

    def _ensure_allocator_global(self) -> int:
        if self._allocator_global_index is None:
            self._allocator_global_index = len(self._globals)
            self._globals.append(GlobalDeclaration(self._allocator_global_index, u32, 4, True))
        return self._allocator_global_index

    @property
    def functions(self) -> tuple[FunctionDeclaration, ...]:
        return tuple(self._functions)

    @property
    def imports(self) -> tuple[ImportDeclaration, ...]:
        return tuple(self._imports)

    @property
    def globals(self) -> tuple[GlobalDeclaration, ...]:
        return tuple(self._globals)

    @property
    def tables(self) -> tuple[TableDeclaration, ...]:
        return tuple(self._tables)

    def _intern_static(self, data: bytes, *, alignment: int = 1) -> int:
        """Store immutable bytes in linear memory once and return their address."""
        if self._closed:
            raise ModuleStateError("static data cannot be added after assembly")
        if not isinstance(data, bytes):
            raise TypeError("static data must be bytes")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("static data alignment must be a positive power of two")
        key = (data, alignment)
        try:
            return self._static_data[key]
        except KeyError:
            address = (self._static_end + alignment - 1) & -alignment
            if address + len(data) > 0xFFFFFFFF:
                raise CompileError("static data exceeds the wasm32 address range")
            self._static_data[key] = address
            self._static_segments.append((address, data))
            self._static_end = address + len(data)
            self._ensure_memory(export=True)
            return address

    def _ensure_memory(self, *, export: bool) -> None:
        existing = self._requirements.get("memory")
        if existing is None:
            if export:
                self._validate_export_name("memory")
            self._requirements["memory"] = MemoryRequirement(1, export=export)
            return
        assert isinstance(existing, MemoryRequirement)
        if export and not existing.export:
            self._validate_export_name("memory")
        self._requirements["memory"] = replace(existing, export=existing.export or export)

    @property
    def bytecode(self) -> bytes:
        if self._bytecode is None:
            raise ModuleStateError("module has not been assembled yet")
        return self._bytecode

    @property
    def data(self) -> bytes:
        return self.bytecode

    @property
    def bytes(self) -> bytes:
        """The compiled WebAssembly binary."""
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
            if existing is not None:
                assert isinstance(existing, MemoryRequirement)
                memory = replace(
                    memory,
                    initial_pages=max(existing.initial_pages, memory.initial_pages),
                    export=existing.export or memory.export,
                )
            self._requirements["memory"] = memory
        indices = tuple(declaration.index for declaration in declarations)
        self._internal_dependencies[name] = indices
        return indices

    def __bytes__(self) -> bytes:
        return self.bytecode

    def _assemble(self) -> bytes:
        if self._bytecode is not None:
            return self._bytecode

        self._finalize_memory_layout()
        self._bytecode = _ModuleEncoder(self).encode()
        self._closed = True
        return self._bytecode

    def _finalize_memory_layout(self) -> None:
        """Place static data before the allocator heap once all source is known."""
        memory = self._requirements.get("memory")
        if memory is None:
            return
        assert isinstance(memory, MemoryRequirement)
        heap_start = max(4, (self._static_end + 7) & -8)
        pages_for_data = max(1, (heap_start + 65535) // 65536)
        if "allocator" in self._internal_dependencies:
            from .memory import ALLOCATOR_LAYOUT, allocator_declarations

            indices = self._internal_dependencies["allocator"]
            start = indices[0]
            assert self._allocator_global_index is not None
            declarations = allocator_declarations(
                start,
                replace(ALLOCATOR_LAYOUT, heap_start=heap_start),
                global_index=self._allocator_global_index,
            )
            position = start - len(self._imports)
            self._functions[position : position + len(declarations)] = declarations
            global_ = self._globals[self._allocator_global_index]
            self._globals[self._allocator_global_index] = replace(global_, initial=heap_start)
        self._requirements["memory"] = replace(memory, initial_pages=max(memory.initial_pages, pages_for_data))

    def compile(self) -> WebAssembly:
        """Assemble this module, returning it as the compiled artifact."""
        self._assemble()
        return self

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
        imports = self.module._imports
        functions = self.module._functions
        sections: list[bytes] = []
        if self.module.debug and (imports or functions):
            sections.append(encode_section(0, self._name_section()))
        if self.module._signatures:
            type_entries = []
            for signature in self.module._signatures:
                parameters = encode_vector(bytes((item.wasm_type,)) for item in signature.parameter_types)
                results = encode_vector(()) if signature.return_type is None else encode_vector(
                    (bytes((signature.return_type.wasm_type,)),)
                )
                type_entries.append(b"\x60" + parameters + results)
            sections.append(encode_section(1, encode_vector(type_entries)))
            if imports:
                import_entries = []
                for declaration in imports:
                    import_entries.append(
                        encode_name(declaration.module_name)
                        + encode_name(declaration.field_name)
                        + b"\x00"
                        + encode_u32(self.module._signature_index(declaration.signature))
                    )
                sections.append(encode_section(2, encode_vector(import_entries)))
            if functions:
                sections.append(
                    encode_section(3, encode_vector(
                        encode_u32(self.module._signature_index(item.signature)) for item in functions
                    ))
                )

        tables = self.module._tables
        if tables:
            sections.append(encode_section(4, encode_vector(
                b"\x70\x00" + encode_u32(len(table.function_indices)) for table in tables
            )))

        memory = self.module._requirements.get("memory")
        if memory is not None:
            assert isinstance(memory, MemoryRequirement)
            sections.append(encode_section(5, encode_vector((b"\x00" + encode_u32(memory.initial_pages),))))
        if self.module._globals:
            globals_ = []
            for declaration in self.module._globals:
                initializer = (
                    declaration.value_type.constant(declaration.initial).encode()
                    + WasmInstructions.create("end").encode()
                )
                globals_.append(bytes((declaration.value_type.wasm_type, int(declaration.mutable))) + initializer)
            sections.append(encode_section(6, encode_vector(globals_)))

        exports = [declaration for declaration in functions if declaration.export]
        exported_globals = [item for item in self.module._globals if item.export_name is not None]
        exported_tables = [item for item in tables if item.export_name is not None]
        if exports or exported_globals or exported_tables or (memory is not None and memory.export):
            entries = [
                encode_name(declaration.name) + b"\x00" + encode_u32(declaration.index)
                for declaration in exports
            ]
            entries += [
                encode_name(declaration.export_name) + b"\x03" + encode_u32(declaration.index)
                for declaration in exported_globals
            ]
            entries += [
                encode_name(declaration.export_name) + b"\x01" + encode_u32(declaration.index)
                for declaration in exported_tables
            ]
            if memory is not None and memory.export:
                entries.append(encode_name("memory") + b"\x02" + encode_u32(0))
            sections.append(encode_section(7, encode_vector(entries)))

        if self.module._start_index is not None:
            sections.append(encode_section(8, encode_u32(self.module._start_index)))

        if tables:
            segments = []
            for declaration in tables:
                initializer = WasmInstructions.create("i32.const", 0).encode() + WasmInstructions.create("end").encode()
                segments.append(b"\x00" + initializer + encode_vector(
                    encode_u32(index) for index in declaration.function_indices
                ))
            sections.append(encode_section(9, encode_vector(segments)))

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

        if self.module._static_segments:
            segments = []
            for offset, data in self.module._static_segments:
                initializer = WasmInstructions.create("i32.const", offset).encode() + WasmInstructions.create("end").encode()
                segments.append(b"\x00" + initializer + encode_u32(len(data)) + data)
            sections.append(encode_section(11, encode_vector(segments)))

        return WebAssembly.MAGIC_AND_VERSION + b"".join(sections)

    def _name_section(self) -> bytes:
        declarations = [*self.module._imports, *self.module._functions]
        function_payload = encode_vector(
            encode_u32(declaration.index) + encode_name(declaration.name) for declaration in declarations
        )
        payload = encode_name("name") + b"\x01" + encode_u32(len(function_payload)) + function_payload
        local_entries = []
        for declaration in self.module._functions:
            names = [
                encode_u32(index) + encode_name(name)
                for index, name in enumerate((*declaration.signature.parameter_names, *declaration.body.local_names))
                if name is not None
            ]
            if names:
                local_entries.append(encode_u32(declaration.index) + encode_vector(names))
        if local_entries:
            local_payload = encode_vector(local_entries)
            payload += b"\x02" + encode_u32(len(local_payload)) + local_payload
        return payload
