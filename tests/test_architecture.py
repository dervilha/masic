import unittest
from pathlib import Path

import masic
import masic.wasm
from masic import WebAssembly, i8, i32, malloc, sizeof
from masic.wasm.memory import ALLOCATOR_LAYOUT


PACKAGE = Path(masic.wasm.__file__).parent


class ArchitectureTests(unittest.TestCase):
    def test_removed_single_role_modules_stay_consolidated(self):
        self.assertFalse((PACKAGE / "expressions.py").exists())
        self.assertFalse((PACKAGE / "builder.py").exists())

    def test_numeric_spec_owns_memory_and_value_metadata(self):
        self.assertEqual(i8.spec.load, "i32.load8_s")
        self.assertEqual(i8.spec.store, "i32.store8")
        self.assertEqual(i8.spec.byte_size, sizeof(i8))
        self.assertEqual(i32.spec.wasm_type, i32.wasm_type)

    def test_top_level_exports_have_one_authority(self):
        self.assertIs(masic.__all__, masic.wasm.__all__)

    def test_allocator_dependency_is_installed_once(self):
        with WebAssembly("dependencies") as wasm:
            @wasm.func(export=True)
            def allocate_twice() -> i32:
                malloc(1, dtype=i32)
                malloc(1, dtype=i32)
                return 0

        self.assertEqual([function.name for function in wasm.functions].count("$malloc"), 1)
        self.assertEqual([function.name for function in wasm.functions].count("$mfree"), 1)

    def test_allocator_constants_are_derived_from_one_layout(self):
        self.assertEqual(ALLOCATOR_LAYOUT.overhead, 8)
        self.assertEqual(ALLOCATOR_LAYOUT.minimum_block_size, 16)
        self.assertEqual(ALLOCATOR_LAYOUT.size_mask, -8)
        self.assertEqual(ALLOCATOR_LAYOUT.maximum_request, 0xFFFFFFE8)

    def test_only_instruction_registry_constructs_instruction_values(self):
        offenders = []
        for path in PACKAGE.glob("*.py"):
            if path.name == "instructions.py":
                continue
            if "Instruction(" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        self.assertEqual(offenders, [])
