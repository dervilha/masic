from __future__ import annotations

import unittest

from masic import (
    CompileError,
    ExpressionError,
    WebAssembly,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    malloc,
    mfree,
    pointer,
    sizeof,
    u8,
    u16,
    u32,
    u64,
)


class SizeofTests(unittest.TestCase):
    def test_numeric_and_pointer_sizes_are_python_integers(self):
        cases = {
            i8: 1,
            u8: 1,
            i16: 2,
            u16: 2,
            i32: 4,
            u32: 4,
            i64: 8,
            u64: 8,
            f32: 4,
            f64: 8,
            pointer[i32]: 4,
        }
        for dtype, expected in cases.items():
            with self.subTest(dtype=dtype):
                self.assertIs(type(sizeof(dtype)), int)
                self.assertEqual(sizeof(dtype), expected)

    def test_invalid_sizeof_operand_is_rejected(self):
        for value in (object(), int, "i32"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                sizeof(value)


class PointerCompilationTests(unittest.TestCase):
    def test_memory_operations_require_an_active_function(self):
        with self.assertRaises(CompileError):
            malloc(1, dtype=i32)

    def test_malloc_is_materialized_once_and_pointer_operations_are_ordered(self):
        with WebAssembly("ordered") as wasm:
            @wasm.func(export=True)
            def main() -> i32:
                values: pointer[i32] = malloc(5, dtype=i32)
                values[0] = 42
                before = values[0]
                values[0] = 99
                mfree(values)
                return before

        main = wasm.functions[-1]
        names = [instruction.name for instruction in main.body.instructions]
        self.assertEqual(names.count("call"), 2)  # one malloc and one mfree
        self.assertEqual(names.count("i32.load"), 1)
        self.assertEqual(names.count("i32.store"), 2)
        self.assertLess(names.index("i32.load"), names.index("i32.store", names.index("i32.store") + 1))
        self.assertEqual(main.body.local_types, (u32, i32))

    def test_every_numeric_dtype_selects_a_typed_load_and_store(self):
        expected = {
            i8: ("i32.load8_s", "i32.store8"),
            u8: ("i32.load8_u", "i32.store8"),
            i16: ("i32.load16_s", "i32.store16"),
            u16: ("i32.load16_u", "i32.store16"),
            i32: ("i32.load", "i32.store"),
            u32: ("i32.load", "i32.store"),
            i64: ("i64.load", "i64.store"),
            u64: ("i64.load", "i64.store"),
            f32: ("f32.load", "f32.store"),
            f64: ("f64.load", "f64.store"),
        }
        for dtype, (load_name, store_name) in expected.items():
            with self.subTest(dtype=dtype.__name__):
                wasm = WebAssembly(dtype.__name__)

                def access():
                    values = malloc(1, dtype=dtype)
                    values[0] = 1
                    return values[0]

                access.__annotations__ = {"return": dtype}
                wasm.func()(access)
                names = [item.name for item in wasm.functions[-1].body.instructions]
                self.assertIn(load_name, names)
                self.assertIn(store_name, names)

    def test_constant_index_uses_memarg_offset_and_dynamic_index_scales(self):
        with WebAssembly("indices") as wasm:
            @wasm.func(export=True)
            def access(index: u32) -> i32:
                values = malloc(8, dtype=i32)
                values[3] = 7
                values[index] = 9
                return values[3]

        stores = [item for item in wasm.functions[-1].body.instructions if item.name == "i32.store"]
        self.assertEqual(stores[0].argument, (4, 12))
        self.assertEqual(stores[1].argument, (4, 0))
        self.assertIn("i32.mul", [item.name for item in wasm.functions[-1].body.instructions])

    def test_invalid_allocation_and_indices_are_rejected(self):
        wasm = WebAssembly("invalid")

        with self.assertRaises(ExpressionError):
            @wasm.func()
            def negative_count() -> i32:
                malloc(-1, dtype=i32)
                return 0

        with self.assertRaises(ExpressionError):
            @wasm.func()
            def negative_index() -> i32:
                values = malloc(1, dtype=i32)
                return values[-1]


try:
    import wasmtime
except ImportError:  # pragma: no cover - the unit suite remains usable without an engine
    wasmtime = None


@unittest.skipIf(wasmtime is None, "wasmtime is not installed")
class AllocatorRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with WebAssembly("allocator") as wasm:
            @wasm.func(export=True)
            def ordered_load() -> i32:
                values = malloc(1, dtype=i32)
                values[0] = 42
                before = values[0]
                values[0] = 99
                return before

            @wasm.func(export=True)
            def reuse() -> i32:
                first = malloc(4, dtype=i32)
                guard = malloc(1, dtype=i32)
                mfree(first)
                replacement = malloc(4, dtype=i32)
                return replacement.address - first.address

            @wasm.func(export=True)
            def split() -> i32:
                large = malloc(10, dtype=i32)
                guard = malloc(1, dtype=i32)
                mfree(large)
                first = malloc(1, dtype=i32)
                second = malloc(1, dtype=i32)
                return second.address - first.address

            @wasm.func(export=True)
            def coalesce() -> i32:
                first = malloc(1, dtype=i32)
                second = malloc(1, dtype=i32)
                guard = malloc(1, dtype=i32)
                mfree(second)
                mfree(first)
                merged = malloc(5, dtype=i32)
                return merged.address - first.address

            @wasm.func(export=True)
            def grow() -> i32:
                values = malloc(20000, dtype=i32)
                values[19999] = 73
                return values[19999]

            @wasm.func(export=True)
            def dynamic(count: u32) -> i32:
                values = malloc(count, dtype=i32)
                values[0] = 91
                return values[0]

            @wasm.func(export=True)
            def null_allocation() -> u32:
                values = malloc(0, dtype=i32)
                mfree(values)
                return values.address

        cls.engine = wasmtime.Engine()
        cls.compiled = wasmtime.Module(cls.engine, bytes(wasm))

    def invoke(self, name, *arguments):
        store = wasmtime.Store(self.engine)
        instance = wasmtime.Instance(store, self.compiled, [])
        return instance.exports(store)[name](store, *arguments), store, instance

    def test_load_is_materialized_at_python_evaluation_time(self):
        self.assertEqual(self.invoke("ordered_load")[0], 42)

    def test_freed_block_is_reused(self):
        self.assertEqual(self.invoke("reuse")[0], 0)

    def test_large_free_block_is_split(self):
        self.assertEqual(self.invoke("split")[0], 16)

    def test_adjacent_free_blocks_are_coalesced(self):
        self.assertEqual(self.invoke("coalesce")[0], 0)

    def test_dynamic_count_is_scaled_by_dtype(self):
        self.assertEqual(self.invoke("dynamic", 3)[0], 91)

    def test_zero_count_returns_null_and_free_null_is_safe(self):
        self.assertEqual(self.invoke("null_allocation")[0], 0)

    def test_allocator_grows_linear_memory(self):
        result, store, instance = self.invoke("grow")
        self.assertEqual(result, 73)
        memory = instance.exports(store)["memory"]
        self.assertGreater(memory.data_len(store), 65536)
