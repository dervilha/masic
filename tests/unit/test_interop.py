from __future__ import annotations

import unittest

from masic import CompileError, WebAssembly, i32, pointer, u32


class ImportTests(unittest.TestCase):
    def test_imports_precede_local_functions_and_encode_external_names(self):
        with WebAssembly("interop") as wasm:
            @wasm.import_func("host", name="multiply")
            def product(left: i32, right: i32) -> i32:
                ...

            @wasm.import_func("host")
            def record(value: i32) -> None:
                pass

            @wasm.func(export=True)
            def use_imports(value: i32) -> i32:
                record(value)
                return product(value, 3)

        self.assertEqual([(item.module_name, item.field_name, item.index) for item in wasm.imports], [
            ("host", "multiply", 0),
            ("host", "record", 1),
        ])
        self.assertEqual(wasm.functions[0].index, 2)
        self.assertIn(b"host", bytes(wasm))
        self.assertIn(b"multiply", bytes(wasm))
        self.assertEqual([item.name for item in wasm.functions[0].body.instructions], [
            "local.get", "call", "local.get", "i32.const", "call",
        ])

    def test_void_local_function_uses_natural_bare_return(self):
        with WebAssembly("void") as wasm:
            @wasm.import_func("host")
            def record(value: i32) -> None:
                ...

            @wasm.func()
            def forward(value: i32) -> None:
                record(value)
                return

        self.assertIsNone(wasm.functions[0].signature.return_type)
        self.assertEqual([item.name for item in wasm.functions[0].body.instructions], ["local.get", "call"])

    def test_void_call_is_rejected_when_used_as_a_value(self):
        wasm = WebAssembly("bad-void")

        @wasm.import_func("host")
        def record(value: i32) -> None:
            ...

        with self.assertRaisesRegex(CompileError, "void calls may only"):
            @wasm.func()
            def invalid(value: i32) -> i32:
                ignored = record(value)
                return 0

    def test_imports_cannot_follow_local_definitions(self):
        wasm = WebAssembly("ordering")

        @wasm.func()
        def local(value: i32) -> i32:
            return value

        with self.assertRaisesRegex(CompileError, "before local"):
            @wasm.import_func("host")
            def too_late(value: i32) -> i32:
                ...


class SharedMemoryTests(unittest.TestCase):
    def test_raw_host_address_builds_typed_loads_without_allocator(self):
        with WebAssembly("memory") as wasm:
            @wasm.func(export=True)
            def first_and_second(address: u32) -> i32:
                values = pointer[i32].from_address(address)
                return values[0] + values[1]

        names = [item.name for item in wasm.functions[0].body.instructions]
        self.assertEqual(names, ["local.get", "i32.load", "local.set", "local.get", "i32.load", "local.set", "local.get", "local.get", "i32.add"])
        self.assertIn(b"memory", bytes(wasm))
        self.assertNotIn("$malloc", [function.name for function in wasm.functions])

    def test_raw_address_rejects_non_integer_values(self):
        wasm = WebAssembly("bad-address")
        with self.assertRaisesRegex(CompileError, "pointer address must be an integer"):
            @wasm.func()
            def invalid() -> i32:
                values = pointer[i32].from_address(1.5)
                return values[0]

