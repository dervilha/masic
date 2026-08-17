from __future__ import annotations

import unittest

from masic import CompileError, WebAssembly, i32, malloc, pointer, u32


def section_ids(binary: bytes) -> list[int]:
    """Return module section IDs without interpreting their payloads."""
    ids: list[int] = []
    cursor = 8
    while cursor < len(binary):
        ids.append(binary[cursor])
        cursor += 1
        length = 0
        shift = 0
        while True:
            byte = binary[cursor]
            cursor += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        cursor += length
    return ids


class GlobalTests(unittest.TestCase):
    def test_persistent_global_uses_clear_value_syntax(self):
        with WebAssembly("global") as wasm:
            counter = wasm.create_global(i32, 40, mutable=True, export="counter")

            @wasm.func(export=True)
            def increment() -> i32:
                counter.value += 2
                return counter.value

        self.assertEqual(section_ids(bytes(wasm)), [1, 3, 6, 7, 10])
        self.assertEqual(
            [instruction.name for instruction in wasm.functions[0].body.instructions],
            ["global.get", "i32.const", "i32.add", "global.set", "global.get"],
        )

    def test_immutable_global_rejects_assignment(self):
        with self.assertRaisesRegex(CompileError, "immutable Wasm global"):
            with WebAssembly("global") as wasm:
                answer = wasm.create_global(i32, 42)

                @wasm.func()
                def change() -> None:
                    answer.value = 0
                    return

    def test_unsigned_global_initializer_uses_its_expression_encoding(self):
        with WebAssembly("global") as wasm:
            wasm.create_global(u32, 0xFFFFFFFF)

        self.assertIn(b"\x7f\x00\x41\x7f\x0b", bytes(wasm))

    def test_automatic_memory_reserves_its_export_name(self):
        with self.assertRaisesRegex(CompileError, "export 'memory' is already registered"):
            with WebAssembly("memory") as wasm:
                wasm.create_global(i32, 0, export="memory")

                @wasm.func()
                def access(address: u32) -> i32:
                    return pointer[i32].from_address(address)[0]

    def test_allocator_cursor_does_not_assume_global_zero(self):
        with WebAssembly("allocator") as wasm:
            wasm.create_global(i32, 0, mutable=True)

            @wasm.func()
            def allocate() -> i32:
                return malloc(1, dtype=i32).address

        self.assertEqual(len(wasm.globals), 2)
        malloc_body = wasm.functions[1].body.instructions
        self.assertIn(1, [item.argument for item in malloc_body if item.name == "global.get"])


class TableAndStartTests(unittest.TestCase):
    def test_table_dispatch_emits_only_the_sections_it_uses(self):
        with WebAssembly("dispatch") as wasm:
            state = wasm.create_global(i32, 0, mutable=True, export="state")

            @wasm.func()
            def add_one(value: i32) -> i32:
                return value + 1

            @wasm.func()
            def sub_one(value: i32) -> i32:
                return value - 1

            operations = wasm.create_table([add_one, sub_one], export="operations")

            @wasm.func(export=True)
            def dispatch(index: i32, value: i32) -> i32:
                return operations.call(index, value)

            @wasm.start
            def initialize() -> None:
                state.value = 41
                return

        self.assertEqual(section_ids(bytes(wasm)), [1, 3, 4, 6, 7, 8, 9, 10])
        self.assertEqual(wasm.functions[2].body.instructions[-1].name, "call_indirect")
        self.assertNotIn(11, section_ids(bytes(wasm)))
        self.assertNotIn(12, section_ids(bytes(wasm)))

    def test_tables_require_declared_functions_with_one_signature(self):
        with WebAssembly("tables") as wasm:
            @wasm.func()
            def unary(value: i32) -> i32:
                return value

            @wasm.func()
            def nullary() -> i32:
                return 0

            with self.assertRaisesRegex(CompileError, "share one Wasm signature"):
                wasm.create_table([unary, nullary])

    def test_start_must_be_void_and_take_no_parameters(self):
        with self.assertRaisesRegex(CompileError, "@wasm.start"):
            with WebAssembly("start") as wasm:
                @wasm.start
                def invalid(value: i32) -> None:
                    return


class DebugNameTests(unittest.TestCase):
    def test_name_section_is_opt_in(self):
        with WebAssembly("release") as release:
            @release.func()
            def identity(value: i32) -> i32:
                return value

        with WebAssembly("debug", debug=True) as debug:
            @debug.func()
            def identity(value: i32) -> i32:
                return value

        self.assertNotIn(0, section_ids(bytes(release)))
        self.assertEqual(section_ids(bytes(debug))[0], 0)
        self.assertIn(b"identity", bytes(debug))
