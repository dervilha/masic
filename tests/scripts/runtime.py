"""Optional Wasmtime adapter used only by the executable integration scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WasmtimeRuntime:
    """Instantiate a fresh module instance for each public function call."""

    engine: object
    module: object
    wasmtime: object

    def invoke(self, name: str, *arguments: int | float) -> object:
        result, _, _ = self.invoke_details(name, *arguments)
        return result

    def invoke_details(self, name: str, *arguments: int | float) -> tuple[object, object, object]:
        store = self.wasmtime.Store(self.engine)
        instance = self.wasmtime.Instance(store, self.module, [])
        function = instance.exports(store)[name]
        return function(store, *arguments), store, instance

    def cached_export(self, name: str) -> WasmtimeExport:
        """Instantiate once and retain an exported function for repeated calls."""
        store = self.wasmtime.Store(self.engine)
        instance = self.wasmtime.Instance(store, self.module, [])
        function = instance.exports(store)[name]
        return WasmtimeExport(store, instance, function)


@dataclass(slots=True)
class WasmtimeExport:
    """A single Wasmtime instance and an export bound to its persistent store."""

    store: object
    instance: object
    function: object

    def invoke(self, *arguments: int | float) -> object:
        return self.function(self.store, *arguments)


def load_wasmtime(wasm: bytes) -> tuple[WasmtimeRuntime | None, str | None]:
    """Return an optional runtime without making it a project dependency."""
    try:
        import wasmtime
    except ImportError:
        return None, "Wasmtime is not installed (install it separately to execute Wasm)."

    try:
        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, wasm)
    except Exception as error:  # pragma: no cover - depends on external engine versions
        return None, f"Wasmtime could not load the generated module: {error}"
    return WasmtimeRuntime(engine, module, wasmtime), None
