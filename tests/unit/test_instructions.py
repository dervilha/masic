import unittest

from masic import WasmInstructions


class InstructionRegistryTests(unittest.TestCase):
    def test_instruction_is_built_from_wat_name(self):
        self.assertEqual(WasmInstructions.create("i32.add").encode(), b"\x6a")
        self.assertEqual(WasmInstructions.create("local.get", 128).encode(), b"\x20\x80\x01")
        self.assertEqual(WasmInstructions.create("i32.const", -1).encode(), b"\x41\x7f")

    def test_memory_arguments_are_encoded_by_the_registry(self):
        instruction = WasmInstructions.create("i32.load", alignment=4, offset=12)
        self.assertEqual(instruction.encode(), b"\x28\x02\x0c")
        self.assertEqual(instruction.argument, (4, 12))

    def test_blocks_and_memory_indices_have_centralized_defaults(self):
        self.assertEqual(WasmInstructions.create("block").encode(), b"\x02\x40")
        self.assertEqual(WasmInstructions.create("memory.grow").encode(), b"\x40\x00")

    def test_indirect_calls_encode_type_and_table_indices(self):
        self.assertEqual(WasmInstructions.create("call_indirect", (3, 0)).encode(), b"\x11\x03\x00")

    def test_invalid_names_and_immediates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            WasmInstructions.create("i32.not_real")
        with self.assertRaisesRegex(ValueError, "requires a u32"):
            WasmInstructions.create("local.get")
        with self.assertRaisesRegex(ValueError, "positive power of two"):
            WasmInstructions.create("i32.store", alignment=3)

    def test_specs_are_the_single_opcode_source(self):
        self.assertEqual(WasmInstructions.spec("f64.div").opcode, 0xA3)
        self.assertEqual(WasmInstructions.spec("i32.extend16_s").opcode, 0xC1)
