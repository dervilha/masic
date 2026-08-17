from __future__ import annotations

import unittest
from pathlib import Path

from masic import CompileError, ExpressionError, ModuleStateError, WebAssembly, f64, func, i8, i32, i64, u8, u32


class DecoratorTests(unittest.TestCase):
    def test_decorator_replaces_python_function_with_func(self):
        with WebAssembly("calls") as wasm:
            @wasm.func()
            def add(a: i32, b: i32 = 2) -> i32:
                return a + b

            self.assertIsInstance(add, func)
            self.assertEqual(add.index, 0)
            call = add(40)
            self.assertIs(type(call), i32)
            self.assertEqual([item.name for item in call.instructions], ["i32.const", "i32.const", "call"])

    def test_calls_can_be_composed_and_use_stable_indices(self):
        with WebAssembly("calls") as wasm:
            @wasm.func()
            def add(a: i32, b: i32) -> i32:
                return a + b

            @wasm.func(export=True)
            def add_three(value: i32) -> i32:
                return add(value, 3)

        self.assertEqual([item.name for item in wasm.functions[1].body.instructions], ["local.get", "i32.const", "call"])
        self.assertEqual(wasm.functions[1].body.instructions[-1].argument, 0)

    def test_call_supports_keywords_and_implicit_argument_casts(self):
        with WebAssembly("calls") as wasm:
            @wasm.func()
            def widen(value: i64) -> i64:
                return value

            value = widen(value=i32.local(0, module=wasm))
            self.assertEqual([item.name for item in value.instructions], ["local.get", "i64.extend_i32_s", "call"])

    def test_invalid_call_shape_is_a_compile_error(self):
        with WebAssembly("calls") as wasm:
            @wasm.func()
            def identity(value: i32) -> i32:
                return value

            for invoke in (lambda: identity(), lambda: identity(1, 2), lambda: identity(missing=1)):
                with self.subTest(invoke=invoke), self.assertRaises(CompileError):
                    invoke()

    def test_return_value_is_cast_to_annotation(self):
        with WebAssembly("returns") as wasm:
            @wasm.func()
            def convert(value: f64) -> i32:
                return value

        self.assertEqual([item.name for item in wasm.functions[0].body.instructions], ["local.get", "i32.trunc_f64_s"])

    def test_numeric_literal_return_is_supported(self):
        with WebAssembly("literal") as wasm:
            @wasm.func()
            def answer() -> i32:
                return 42

        self.assertEqual([item.name for item in wasm.functions[0].body.instructions], ["i32.const"])

    def test_postponed_annotations_are_resolved(self):
        with WebAssembly("annotations") as wasm:
            @wasm.func()
            def identity(value: "u32") -> "u32":
                return value

        self.assertIs(wasm.functions[0].signature.return_type, u32)


class DeclarationErrorTests(unittest.TestCase):
    def test_export_flag_must_be_boolean(self):
        with self.assertRaises(TypeError):
            WebAssembly("bad").func(export=1)

    def test_missing_or_non_wasm_annotations_are_rejected(self):
        invalid_functions = []

        def missing(value) -> i32:
            return value

        def bad_parameter(value: int) -> i32:
            return value

        def bad_return(value: i32) -> int:
            return value

        invalid_functions.extend((missing, bad_parameter, bad_return))
        for function in invalid_functions:
            with self.subTest(function=function.__name__), self.assertRaises(CompileError):
                WebAssembly("bad").func()(function)

    def test_invalid_return_object_is_rejected(self):
        wasm = WebAssembly("bad")
        with self.assertRaises(CompileError):
            @wasm.func()
            def invalid(value: i32) -> i32:
                return "not an expression"

    def test_variadic_and_keyword_only_parameters_are_rejected(self):
        def variadic(*values: i32) -> i32:
            return values[0]

        def keyword_only(*, value: i32) -> i32:
            return value

        for function in (variadic, keyword_only):
            with self.subTest(function=function.__name__), self.assertRaises(CompileError):
                WebAssembly("bad").func()(function)

    def test_default_must_fit_annotated_type(self):
        def invalid_default(value: i8 = 128) -> i8:
            return value

        with self.assertRaises(ExpressionError):
            WebAssembly("bad").func()(invalid_default)

    def test_duplicate_names_are_rejected(self):
        wasm = WebAssembly("duplicates")

        @wasm.func()
        def same(value: i32) -> i32:
            return value

        with self.assertRaises(CompileError):
            @wasm.func()
            def same(value: i32) -> i32:
                return value

    def test_cross_module_call_is_rejected(self):
        first = WebAssembly("first")

        @first.func()
        def identity(value: i32) -> i32:
            return value

        second = WebAssembly("second")

        def from_first(value: i32) -> i32:
            return identity(value)

        with self.assertRaises(CompileError):
            second.func()(from_first)


class AssemblyTests(unittest.TestCase):
    EXPECTED_ADD = bytes.fromhex(
        "0061736d01000000"
        "01070160027f7f017f"
        "03020100"
        "070701036164640000"
        "0a0a010800200020016a0f0b"
    )

    def test_exported_add_matches_golden_binary(self):
        with WebAssembly("MyModule") as wasm:
            @wasm.func(export=True)
            def add(a: i32, b: i32) -> i32:
                return a + b

        self.assertEqual(bytes(wasm), self.EXPECTED_ADD)

    def test_empty_module_is_valid_header_only_binary(self):
        with WebAssembly("empty") as wasm:
            pass
        self.assertEqual(bytes(wasm), WebAssembly.MAGIC_AND_VERSION)

    def test_wasm_signatures_are_deduplicated(self):
        with WebAssembly("dedupe") as wasm:
            @wasm.func()
            def signed(value: i32) -> i32:
                return value

            @wasm.func()
            def narrow(value: i8) -> i8:
                return value

            @wasm.func()
            def unsigned(value: u8) -> u8:
                return value

        self.assertEqual(bytes(wasm).count(b"\x60"), 1)

    def test_different_stack_types_have_different_signatures(self):
        with WebAssembly("types") as wasm:
            @wasm.func()
            def thirty_two(value: i32) -> i32:
                return value

            @wasm.func()
            def sixty_four(value: i64) -> i64:
                return value

        self.assertIn(b"\x60\x01\x7f\x01\x7f", bytes(wasm))
        self.assertIn(b"\x60\x01\x7e\x01\x7e", bytes(wasm))

    def test_assemble_is_idempotent_and_closes_builder(self):
        wasm = WebAssembly("state")
        first = wasm._assemble()
        self.assertIs(first, wasm._assemble())
        with self.assertRaises(ModuleStateError):
            @wasm.func()
            def too_late() -> i32:
                return 0

    def test_compiled_context_cannot_be_reentered(self):
        wasm = WebAssembly("state")
        with wasm:
            pass
        with self.assertRaises(ModuleStateError):
            with wasm:
                pass

    def test_bytecode_before_assembly_is_an_error(self):
        with self.assertRaises(ModuleStateError):
            bytes(WebAssembly("open"))

    def test_context_exception_is_not_suppressed_or_assembled(self):
        wasm = WebAssembly("failure")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with wasm:
                raise RuntimeError("boom")
        with self.assertRaises(ModuleStateError):
            bytes(wasm)

    def test_save_assembles_and_writes_exact_bytes(self):
        wasm = WebAssembly("saved")

        @wasm.func(export=True)
        def identity(value: i32) -> i32:
            return value

        destination = Path(__file__).parent / ".test-module.wasm"
        try:
            resolved = wasm.save(destination)
            self.assertEqual(resolved, destination.resolve())
            self.assertEqual(destination.read_bytes(), bytes(wasm))
        finally:
            destination.unlink(missing_ok=True)
