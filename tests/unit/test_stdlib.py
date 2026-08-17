import unittest

from masic import CompileError, ExpressionError, ModuleStateError, WebAssembly, i32
from masic.wasm import stdlib


class StdlibNamespaceTests(unittest.TestCase):
    def test_stdlib_is_exported_by_the_wasm_namespace(self):
        import masic.wasm

        self.assertIs(masic.wasm.stdlib, stdlib)

    def test_include_is_idempotent_and_binds_string_to_the_destination_module(self):
        with WebAssembly("strings") as wasm:
            strings = wasm.include(stdlib)
            self.assertIs(strings, wasm.include(stdlib))

            @wasm.func(export=True)
            def metadata() -> i32:
                text = strings.String("Olá")
                result = text.length() + text[2]
                text.free()
                return result

        self.assertNotIn("$malloc", [function.name for function in wasm.functions])
        self.assertNotIn("$mfree", [function.name for function in wasm.functions])
        self.assertIn(b"Ol\xc3\xa1", bytes(wasm))
        self.assertIn(b"memory", bytes(wasm))

    def test_recursive_include_receives_the_namespace_created_for_the_outer_include(self):
        class CircularLibrary(WebAssembly):
            def _create_namespace(self, module):
                return {"module": module}

            def _include_into(self, module, namespace):
                namespace["recursive"] = module.include(self)

        library = CircularLibrary("circular")
        target = WebAssembly("target")
        namespace = target.include(library)
        self.assertIs(namespace, namespace["recursive"])
        self.assertIs(namespace, target.include(library))

    def test_empty_literal_uses_a_header_and_exposes_an_i32_boolean(self):
        with WebAssembly("empty") as wasm:
            strings = wasm.include(stdlib)

            @wasm.func()
            def empty() -> i32:
                text = strings.String("")
                result = text.is_empty()
                text.free()
                return result

        instructions = wasm.functions[-1].body.instructions
        self.assertNotIn("i32.store", [item.name for item in instructions])
        self.assertIn("i32.eq", [item.name for item in instructions])

    def test_literal_index_is_checked_during_compilation(self):
        wasm = WebAssembly("bounds")
        strings = wasm.include(stdlib)
        with self.assertRaisesRegex(CompileError, "String byte index 2"):
            @wasm.func()
            def invalid() -> i32:
                text = strings.String("ok")
                return text[2]

    def test_non_includeable_modules_and_closed_destinations_are_rejected(self):
        with self.assertRaisesRegex(CompileError, "not an includeable"):
            WebAssembly("target").include(WebAssembly("not-a-library"))

        target = WebAssembly("closed")
        target.compile()
        with self.assertRaisesRegex(ModuleStateError, "included after assembly"):
            target.include(stdlib)

    def test_string_factory_requires_a_string_value(self):
        with WebAssembly("types") as wasm:
            strings = wasm.include(stdlib)
            with self.assertRaises(ExpressionError):
                strings.String(1)


class ArrayTests(unittest.TestCase):
    def test_typed_array_reserves_pushes_and_preserves_values(self):
        with WebAssembly("arrays") as wasm:
            std = wasm.include(stdlib)

            @wasm.func(export=True)
            def values(value: i32) -> i32:
                items = std.Array[i32](capacity=1)
                items.push(10)
                items.reserve(4)
                items.push(value)
                result = items[0] + items[1] + items.size() + items.capacity()
                items.free()
                return result

        body = wasm.functions[-1].body
        names = [item.name for item in body.instructions]
        self.assertIn("loop", names)
        self.assertIn("i32.load", names)
        self.assertGreaterEqual(names.count("i32.store"), 4)

    def test_push_grows_from_zero_capacity_and_array_type_is_validated(self):
        with WebAssembly("grow") as wasm:
            std = wasm.include(stdlib)

            @wasm.func()
            def one() -> i32:
                items = std.Array[i32]()
                items.push(7)
                result = items[0]
                items.free()
                return result

        self.assertIn("loop", [item.name for item in wasm.functions[-1].body.instructions])
        with self.assertRaisesRegex(CompileError, "concrete Wasm numeric"):
            std.Array[int]

    def test_negative_static_capacity_and_reserve_are_rejected(self):
        wasm = WebAssembly("invalid-array")
        std = wasm.include(stdlib)
        with self.assertRaisesRegex(CompileError, "Array capacity cannot be negative"):
            @wasm.func()
            def invalid_capacity() -> i32:
                std.Array[i32](-1)
                return 0

        with self.assertRaisesRegex(CompileError, "Array reserve capacity cannot be negative"):
            @wasm.func()
            def invalid_reserve() -> i32:
                items = std.Array[i32]()
                items.reserve(-1)
                return 0
