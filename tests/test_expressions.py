import unittest

from masic import ExpressionError, WebAssembly, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64


def names(value):
    return [instruction.name for instruction in value.instructions]


class ConstantTests(unittest.TestCase):
    def test_all_numeric_types_create_constants(self):
        cases = (
            (i8, -128, "i8.const"),
            (u8, 255, "u8.const"),
            (i16, -32768, "i16.const"),
            (u16, 65535, "u16.const"),
            (i32, -(1 << 31), "i32.const"),
            (u32, (1 << 32) - 1, "u32.const"),
            (i64, -(1 << 63), "i64.const"),
            (u64, (1 << 64) - 1, "u64.const"),
            (f32, 1.25, "f32.const"),
            (f64, 1.25, "f64.const"),
        )
        for value_type, literal, instruction_name in cases:
            with self.subTest(value_type=value_type.__name__):
                value = value_type.constant(literal)
                self.assertIs(type(value), value_type)
                self.assertEqual(names(value), [instruction_name])
                self.assertTrue(value.encode())

    def test_integer_range_is_enforced(self):
        for value_type, literal in ((i8, 128), (i8, -129), (u8, -1), (u16, 65536), (u64, 1 << 64)):
            with self.subTest(value_type=value_type.__name__, literal=literal), self.assertRaises(ExpressionError):
                value_type.constant(literal)

    def test_bool_is_not_a_numeric_literal(self):
        with self.assertRaises(ExpressionError):
            i32.constant(True)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.module = WebAssembly("operators")
        self.left = i32.local(0, module=self.module)
        self.right = i32.local(1, module=self.module)

    def test_signed_integer_binary_operators(self):
        cases = {
            "add": (self.left + self.right, "i32.add"),
            "sub": (self.left - self.right, "i32.sub"),
            "mul": (self.left * self.right, "i32.mul"),
            "truediv": (self.left / self.right, "i32.div_s"),
            "floordiv": (self.left // self.right, "i32.div_s"),
            "mod": (self.left % self.right, "i32.rem_s"),
            "and": (self.left & self.right, "i32.and"),
            "or": (self.left | self.right, "i32.or"),
            "xor": (self.left ^ self.right, "i32.xor"),
            "shl": (self.left << self.right, "i32.shl"),
            "shr": (self.left >> self.right, "i32.shr_s"),
        }
        for label, (value, expected) in cases.items():
            with self.subTest(operation=label):
                self.assertEqual(names(value)[-1], expected)

    def test_unsigned_operators_select_unsigned_opcodes(self):
        left = u64.local(0, module=self.module)
        right = u64.local(1, module=self.module)
        self.assertEqual(names(left / right)[-1], "i64.div_u")
        self.assertEqual(names(left % right)[-1], "i64.rem_u")
        self.assertEqual(names(left >> right)[-1], "i64.shr_u")

    def test_stack_width_and_signedness_opcode_matrix(self):
        cases = (
            (i32, "i32.add", "i32.div_s", "i32.lt_s"),
            (u32, "i32.add", "i32.div_u", "i32.lt_u"),
            (i64, "i64.add", "i64.div_s", "i64.lt_s"),
            (u64, "i64.add", "i64.div_u", "i64.lt_u"),
            (f32, "f32.add", "f32.div", "f32.lt"),
            (f64, "f64.add", "f64.div", "f64.lt"),
        )
        for value_type, add, divide, less_than in cases:
            with self.subTest(value_type=value_type.__name__):
                left = value_type.local(0, module=self.module)
                right = value_type.local(1, module=self.module)
                self.assertEqual(names(left + right)[-1], add)
                self.assertEqual(names(left / right)[-1], divide)
                self.assertEqual(names(left < right)[-1], less_than)

    def test_float_operators(self):
        left = f64.local(0, module=self.module)
        right = f64.local(1, module=self.module)
        for value, expected in ((left + right, "f64.add"), (left - right, "f64.sub"), (left * right, "f64.mul"), (left / right, "f64.div")):
            with self.subTest(expected=expected):
                self.assertEqual(names(value)[-1], expected)
        with self.assertRaises(ExpressionError):
            left & right

    def test_comparisons_return_i32(self):
        cases = (
            (self.left == self.right, "i32.eq"),
            (self.left != self.right, "i32.ne"),
            (self.left < self.right, "i32.lt_s"),
            (self.left > self.right, "i32.gt_s"),
            (self.left <= self.right, "i32.le_s"),
            (self.left >= self.right, "i32.ge_s"),
        )
        for value, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(type(value), i32)
                self.assertEqual(names(value)[-1], expected)

    def test_unsigned_and_float_comparisons(self):
        unsigned = u32.local(0, module=self.module) < u32.local(1, module=self.module)
        floating = f32.local(0, module=self.module) >= f32.local(1, module=self.module)
        self.assertEqual(names(unsigned)[-1], "i32.lt_u")
        self.assertEqual(names(floating)[-1], "f32.ge")

    def test_reflected_literal_operations_preserve_order(self):
        value = 10 - self.left
        self.assertEqual(names(value), ["i32.const", "local.get", "i32.sub"])

    def test_unary_operators(self):
        self.assertIs(+self.left, self.left)
        self.assertEqual(names(-self.left), ["i32.const", "local.get", "i32.sub"])
        self.assertEqual(names(~self.left)[-2:], ["i32.const", "i32.xor"])
        floating = f32.local(0, module=self.module)
        self.assertEqual(names(-floating)[-1], "f32.neg")
        with self.assertRaises(ExpressionError):
            ~floating

    def test_expression_cannot_be_a_python_condition(self):
        with self.assertRaises(ExpressionError):
            bool(self.left)

    def test_repr_and_debug_trace_expose_compilation_steps(self):
        value = self.left + 2
        self.assertIn("i32.add", repr(value))
        self.assertEqual(value.debug_trace[-1], "add")


class CastingTests(unittest.TestCase):
    def setUp(self):
        self.module = WebAssembly("casts")

    def test_left_operand_controls_implicit_cast(self):
        left = i32.local(0, module=self.module)
        right = f64.local(1, module=self.module)
        result = left + right
        self.assertIs(type(result), i32)
        self.assertEqual(names(result), ["local.get", "local.get", "i32.trunc_f64_s", "i32.add"])

    def test_integer_width_conversions(self):
        self.assertEqual(names(i32.local(0, module=self.module).cast(i64))[-1], "i64.extend_i32_s")
        self.assertEqual(names(u32.local(0, module=self.module).cast(u64))[-1], "i64.extend_i32_u")
        self.assertEqual(names(i64.local(0, module=self.module).cast(i32))[-1], "i32.wrap_i64")

    def test_integer_float_conversions(self):
        self.assertEqual(names(i64.local(0, module=self.module).cast(f32))[-1], "f32.convert_i64_s")
        self.assertEqual(names(f64.local(0, module=self.module).cast(u32))[-1], "i32.trunc_f64_u")
        self.assertEqual(names(f32.local(0, module=self.module).cast(f64))[-1], "f64.promote_f32")
        self.assertEqual(names(f64.local(0, module=self.module).cast(f32))[-1], "f32.demote_f64")

    def test_narrow_signed_results_are_normalized(self):
        result = i8.local(0, module=self.module) + i8.local(1, module=self.module)
        self.assertEqual(names(result)[-2:], ["i32.add", "i32.extend8_s"])

    def test_narrow_unsigned_results_are_masked(self):
        result = u16.local(0, module=self.module) + u16.local(1, module=self.module)
        self.assertEqual(names(result)[-3:], ["i32.add", "u16.const", "i32.and"])

    def test_cross_module_expressions_are_rejected(self):
        one = i32.local(0, module=WebAssembly("one"))
        two = i32.local(0, module=WebAssembly("two"))
        with self.assertRaises(ExpressionError):
            one + two
