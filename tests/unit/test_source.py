import unittest

from masic import CompileError, WebAssembly, block, branch, i32, loop


class SourceSyntaxTests(unittest.TestCase):
    def test_literal_assignment_and_augmented_assignment_become_locals(self):
        with WebAssembly() as wasm:
            @wasm.func(export=True)
            def arithmetic(value: i32) -> i32:
                total = i32(10)
                total += value * 2
                return total

        body = wasm.functions[0].body
        self.assertEqual(body.local_types, (i32,))
        self.assertEqual(
            [instruction.name for instruction in body.instructions],
            ["i32.const", "local.set", "local.get", "local.get", "i32.const", "i32.mul", "i32.add", "local.set", "local.get"],
        )

    def test_branch_with_else_updates_an_existing_local(self):
        with WebAssembly() as wasm:
            @wasm.func(export=True)
            def absolute(value: i32) -> i32:
                result = value
                with branch(value < 0) as negative:
                    result = -value
                with negative.els():
                    result = value
                return result

        self.assertEqual(
            [instruction.name for instruction in wasm.functions[0].body.instructions],
            [
                "local.get", "local.set", "local.get", "i32.const", "i32.lt_s", "if",
                "i32.const", "local.get", "i32.sub", "local.set", "else", "local.get",
                "local.set", "end", "local.get",
            ],
        )

    def test_existing_local_accepts_a_plain_numeric_literal(self):
        with WebAssembly() as wasm:
            @wasm.func()
            def reset(value: i32) -> i32:
                result = value
                result = 1
                return result

        self.assertEqual(
            [instruction.name for instruction in wasm.functions[0].body.instructions],
            ["local.get", "local.set", "i32.const", "local.set", "local.get"],
        )

    def test_loop_break_and_continue_use_the_correct_nested_depths(self):
        with WebAssembly() as wasm:
            @wasm.func(export=True)
            def count(limit: i32) -> i32:
                value = i32(0)
                with loop() as repeat:
                    with branch(value >= limit):
                        repeat.brk()
                    value += 1
                    with branch(value == 5):
                        repeat.cont()
                return value

        instructions = wasm.functions[0].body.instructions
        branches = [instruction.argument for instruction in instructions if instruction.name == "br"]
        self.assertEqual(branches, [2, 1, 0])
        self.assertEqual([instruction.name for instruction in instructions][-4:], ["br", "end", "end", "local.get"])

    def test_named_block_supports_break(self):
        with WebAssembly() as wasm:
            @wasm.func()
            def stop_early(value: i32) -> i32:
                with block() as stop:
                    with branch(value == 0):
                        stop.brk()
                return value

        branches = [instruction.argument for instruction in wasm.functions[0].body.instructions if instruction.name == "br"]
        self.assertEqual(branches, [1])

    def test_native_control_flow_is_rejected(self):
        wasm = WebAssembly()
        with self.assertRaisesRegex(CompileError, "explicit branch\\(\\) and loop\\(\\)"):
            @wasm.func()
            def invalid(value: i32) -> i32:
                if value == 0:
                    return 1
                return value

    def test_literal_constructor_and_compile_aliases_are_public(self):
        self.assertEqual(i32(42).encode(), i32.constant(42).encode())
        wasm = WebAssembly()

        @wasm.func(export=True)
        def answer() -> i32:
            return i32(42)

        self.assertIs(wasm.compile(), wasm)
        self.assertEqual(wasm.bytes, bytes(wasm))
