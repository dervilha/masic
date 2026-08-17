"""Lower the supported Python source subset into FunctionBuilder instructions."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import control
from .errors import CompileError, ExpressionError
from .instructions import WasmInstructions
from .types import coerce_value, expr, is_numeric_type

if TYPE_CHECKING:
    from .functions import FunctionBuilder, FunctionSignature, LocalRef
    from .module import WebAssembly


@dataclass(frozen=True, slots=True)
class _ExpressionBinding:
    local: LocalRef


@dataclass(frozen=True, slots=True)
class _ControlBinding:
    kind: str
    break_target: int | None = None
    continue_target: int | None = None


class SourceFunctionCompiler:
    """Compile one inspectable Python function without executing its body."""

    def __init__(
        self,
        function: Any,
        builder: FunctionBuilder,
        signature: FunctionSignature,
        module: WebAssembly,
    ) -> None:
        self.function = function
        self.builder = builder
        self.signature = signature
        self.module = module
        self.bindings: dict[str, _ExpressionBinding | _ControlBinding | object] = {}
        self.controls: list[_ControlBinding] = []
        self._control_depth = 0
        self._statement_depth = 0
        closure = inspect.getclosurevars(function)
        self.external_values = {**closure.globals, **closure.nonlocals, **closure.builtins}
        for index, name in enumerate(signature.parameter_names):
            self.bindings[name] = _ExpressionBinding(builder.parameter(index, name))

    def compile(self) -> expr | None:
        definition = self._definition()
        if not definition.body or not isinstance(definition.body[-1], ast.Return):
            raise self._error(definition, "compiled functions must end with a Python return statement")
        self._compile_statements(definition.body[:-1])
        final_return = definition.body[-1]
        assert isinstance(final_return, ast.Return)
        if self.signature.return_type is None:
            if final_return.value is not None:
                raise self._error(final_return, "void functions must use a bare Python return statement")
            return None
        if final_return.value is None:
            raise self._error(final_return, "compiled functions must return a value")
        return self._coerce_return(self._expression(final_return.value), final_return)

    def _definition(self) -> ast.FunctionDef:
        try:
            source = inspect.getsource(self.function)
        except (OSError, TypeError) as error:
            raise CompileError(
                f"could not read source for {self.function.__name__}(); "
                "@wasm.func requires a source-backed named function"
            ) from error
        tree = ast.parse(textwrap.dedent(source))
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == self.function.__name__:
                if isinstance(statement, ast.AsyncFunctionDef):
                    raise self._error(statement, "async functions are not supported")
                return statement
        raise CompileError(f"could not locate source for {self.function.__name__}()")

    def _compile_statements(self, statements: list[ast.stmt]) -> None:
        index = 0
        while index < len(statements):
            statement = statements[index]
            if isinstance(statement, ast.With) and self._is_branch(statement):
                else_statement = None
                binding_name = self._optional_binding_name(statement)
                if index + 1 < len(statements) and binding_name is not None:
                    candidate = statements[index + 1]
                    if isinstance(candidate, ast.With) and self._is_else(candidate, binding_name):
                        else_statement = candidate
                        index += 1
                self._compile_branch(statement, else_statement)
            else:
                self._compile_statement(statement)
            index += 1

    def _compile_statement(self, statement: ast.stmt) -> None:
        if self._compile_control_command(statement):
            return
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                raise self._error(statement, "multiple assignment targets are not supported")
            self._assign(statement.targets[0], self._expression(statement.value), statement)
            return
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                raise self._error(statement, "annotations require an initial value")
            self._assign(statement.target, self._expression(statement.value), statement)
            return
        if isinstance(statement, ast.AugAssign):
            self._augmented_assign(statement)
            return
        if isinstance(statement, ast.Expr):
            value = self._expression(statement.value)
            if isinstance(value, expr):
                self.builder.emit(*value.instructions, WasmInstructions.create("drop"))
            else:
                from .functions import CallEffect

                if isinstance(value, CallEffect):
                    self.builder.emit(*value.instructions)
            return
        if isinstance(statement, ast.Return):
            if statement.value is None:
                if self.signature.return_type is not None:
                    raise self._error(statement, "compiled functions must return a value")
                self.builder.emit(WasmInstructions.create("return"))
                return
            if self.signature.return_type is None:
                raise self._error(statement, "void functions must use a bare Python return statement")
            result = self._coerce_return(self._expression(statement.value), statement)
            self.builder.emit(*result.instructions, WasmInstructions.create("return"))
            return
        if isinstance(statement, ast.With):
            if self._is_loop(statement):
                self._compile_loop(statement)
                return
            if self._is_block(statement):
                self._compile_block(statement)
                return
            if self._is_branch(statement):
                self._compile_branch(statement, None)
                return
        if isinstance(statement, ast.Pass):
            return
        if isinstance(statement, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            raise self._error(statement, "use explicit branch() and loop() scopes for control flow")
        raise self._error(statement, f"unsupported statement {type(statement).__name__}")

    def _assign(self, target: ast.expr, value: object, node: ast.AST) -> None:
        from .functions import CallEffect

        if isinstance(value, CallEffect):
            raise self._error(node, "void calls may only be used as standalone statements")
        if isinstance(target, ast.Name):
            existing = self.bindings.get(target.id)
            if isinstance(existing, _ControlBinding):
                raise self._error(node, f"{target.id!r} is a control-scope name")
            if isinstance(existing, _ExpressionBinding):
                try:
                    stored = coerce_value(value, existing.local.value_type)
                except ExpressionError as error:
                    raise self._error(node, str(error)) from error
                self.builder.emit(*stored.instructions, self.builder.set(existing.local))
                return
            if isinstance(value, expr):
                if existing is not None:
                    raise self._error(node, f"{target.id!r} cannot be reassigned")
                if self._statement_depth:
                    raise self._error(node, "variables must be initialized before entering a control scope")
                local = self.builder.materialized_local(value)
                if local is None:
                    local = self.builder.local(type(value), target.id)
                    self.builder.emit(*value.instructions, self.builder.set(local))
                self.bindings[target.id] = _ExpressionBinding(local)
                return
            if existing is not None:
                raise self._error(node, f"{target.id!r} cannot be reassigned")
            if self._statement_depth:
                raise self._error(node, "values must be initialized before entering a control scope")
            self.bindings[target.id] = value
            return
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            self._store(target, value, node)
            return
        raise self._error(node, "assignment target is not supported")

    def _augmented_assign(self, statement: ast.AugAssign) -> None:
        if isinstance(statement.target, ast.Name):
            binding = self.bindings.get(statement.target.id)
            if not isinstance(binding, _ExpressionBinding):
                raise self._error(statement, f"{statement.target.id!r} is not a numeric local")
            left = self._read_binding(binding)
            assign = lambda value: self.builder.emit(*value.instructions, self.builder.set(binding.local))
        elif isinstance(statement.target, (ast.Attribute, ast.Subscript)):
            left = self._expression(statement.target)
            assign = lambda value: self._store(statement.target, value, statement)
        else:
            raise self._error(statement, "unsupported augmented assignment target")
        right = self._expression(statement.value)
        operation = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.BitAnd: lambda: left & right,
            ast.BitOr: lambda: left | right,
            ast.BitXor: lambda: left ^ right,
            ast.LShift: lambda: left << right,
            ast.RShift: lambda: left >> right,
        }.get(type(statement.op))
        if operation is None:
            raise self._error(statement, f"unsupported augmented assignment {type(statement.op).__name__}")
        value = operation()
        if not isinstance(value, expr):
            raise self._error(statement, "augmented assignment requires a numeric value")
        assign(value)

    def _store(self, target: ast.Attribute | ast.Subscript, value: object, node: ast.AST) -> None:
        try:
            if isinstance(target, ast.Subscript):
                container = self._expression(target.value)
                index = self._expression(target.slice)
                container[index] = value
                return
            receiver = self._expression(target.value)
            setattr(receiver, target.attr, value)
        except (AttributeError, ExpressionError, TypeError) as error:
            raise self._error(node, f"invalid assignment target: {error}") from error

    def _compile_branch(self, statement: ast.With, else_statement: ast.With | None) -> None:
        call = self._control_call(statement, control.branch)
        if len(call.args) != 1 or call.keywords:
            raise self._error(statement, "branch() takes exactly one condition")
        condition = self._expression(call.args[0])
        if not isinstance(condition, expr) or condition.category != "int":
            raise self._error(call.args[0], "branch condition must be an integer Wasm expression")
        self.builder.emit(*condition.instructions, WasmInstructions.create("if"))
        binding = _ControlBinding("branch")
        self.controls.append(binding)
        self._control_depth += 1
        self._compile_control_body(statement, binding)
        if else_statement is not None:
            self.builder.emit(WasmInstructions.create("else"))
            self._compile_control_body(else_statement, binding)
        self._control_depth -= 1
        self.controls.pop()
        self.builder.emit(WasmInstructions.create("end"))

    def _compile_block(self, statement: ast.With) -> None:
        self._control_call(statement, control.block)
        self.builder.emit(WasmInstructions.create("block"))
        binding = _ControlBinding("block", break_target=len(self.controls))
        self.controls.append(binding)
        self._control_depth += 1
        self._compile_control_body(statement, binding)
        self._control_depth -= 1
        self.controls.pop()
        self.builder.emit(WasmInstructions.create("end"))

    def _compile_loop(self, statement: ast.With) -> None:
        self._control_call(statement, control.loop)
        self.builder.emit(WasmInstructions.create("block"), WasmInstructions.create("loop"))
        binding = _ControlBinding(
            "loop",
            break_target=len(self.controls),
            continue_target=len(self.controls) + 1,
        )
        self.controls.extend((binding, binding))
        self._control_depth += 2
        self._compile_control_body(statement, binding)
        self.builder.emit(WasmInstructions.create("br", 0))
        self._control_depth -= 2
        self.controls.pop()
        self.controls.pop()
        self.builder.emit(WasmInstructions.create("end"), WasmInstructions.create("end"))

    def _compile_control_body(self, statement: ast.With, binding: _ControlBinding) -> None:
        name = self._optional_binding_name(statement)
        previous = self.bindings.get(name) if name is not None else None
        if name is not None:
            self.bindings[name] = binding
        self._statement_depth += 1
        self._compile_statements(statement.body)
        self._statement_depth -= 1
        if name is not None:
            if previous is None:
                del self.bindings[name]
            else:
                self.bindings[name] = previous

    def _compile_control_command(self, statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or call.args or call.keywords:
            return False
        if not isinstance(call.func.value, ast.Name) or call.func.attr not in {"brk", "cont"}:
            return False
        binding = self.bindings.get(call.func.value.id)
        if not isinstance(binding, _ControlBinding):
            return False
        target = binding.break_target if call.func.attr == "brk" else binding.continue_target
        if target is None:
            raise self._error(call, f"{binding.kind} scopes do not support .{call.func.attr}()")
        self.builder.emit(WasmInstructions.create("br", self._control_depth - 1 - target))
        return True

    def _expression(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise self._error(node, "only numeric and string literals are supported")
            return node.value
        if isinstance(node, ast.Name):
            binding = self.bindings.get(node.id)
            if isinstance(binding, _ExpressionBinding):
                return self._read_binding(binding)
            if isinstance(binding, _ControlBinding):
                return binding
            if binding is not None:
                return binding
            try:
                return self.external_values[node.id]
            except KeyError as error:
                raise self._error(node, f"unknown name {node.id!r}") from error
        if isinstance(node, ast.BinOp):
            left, right = self._expression(node.left), self._expression(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.FloorDiv: lambda: left // right,
                ast.Mod: lambda: left % right,
                ast.BitAnd: lambda: left & right,
                ast.BitOr: lambda: left | right,
                ast.BitXor: lambda: left ^ right,
                ast.LShift: lambda: left << right,
                ast.RShift: lambda: left >> right,
            }
            try:
                return operations[type(node.op)]()
            except KeyError as error:
                raise self._error(node, f"unsupported operator {type(node.op).__name__}") from error
        if isinstance(node, ast.UnaryOp):
            value = self._expression(node.operand)
            operations = {ast.UAdd: lambda: +value, ast.USub: lambda: -value, ast.Invert: lambda: ~value}
            try:
                return operations[type(node.op)]()
            except KeyError as error:
                raise self._error(node, f"unsupported unary operator {type(node.op).__name__}") from error
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise self._error(node, "chained comparisons are not supported")
            left, right = self._expression(node.left), self._expression(node.comparators[0])
            operations = {
                ast.Eq: lambda: left == right,
                ast.NotEq: lambda: left != right,
                ast.Lt: lambda: left < right,
                ast.LtE: lambda: left <= right,
                ast.Gt: lambda: left > right,
                ast.GtE: lambda: left >= right,
            }
            try:
                return operations[type(node.ops[0])]()
            except KeyError as error:
                raise self._error(node, f"unsupported comparison {type(node.ops[0]).__name__}") from error
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Attribute):
            receiver = self._expression(node.value)
            try:
                return getattr(receiver, node.attr)
            except AttributeError as error:
                raise self._error(node, f"unknown attribute {node.attr!r}") from error
        if isinstance(node, ast.Subscript):
            container = self._expression(node.value)
            index = self._expression(node.slice)
            try:
                return container[index]
            except ExpressionError:
                raise
            except TypeError as error:
                raise self._error(node, str(error)) from error
        raise self._error(node, f"unsupported expression {type(node).__name__}")

    def _call(self, node: ast.Call) -> object:
        if any(keyword.arg is None for keyword in node.keywords):
            raise self._error(node, "starred keyword arguments are not supported")
        callee = self._expression(node.func)
        args = [self._expression(argument) for argument in node.args]
        keywords = {keyword.arg: self._expression(keyword.value) for keyword in node.keywords}
        if is_numeric_type(callee):
            if keywords or len(args) != 1:
                raise self._error(node, f"{callee.__name__}() takes exactly one numeric literal or expression")
            return coerce_value(args[0], callee)
        if not callable(callee):
            raise self._error(node, "attempted to call a non-callable value")
        try:
            return callee(*args, **keywords)
        except ExpressionError:
            raise
        except (CompileError, TypeError, ValueError) as error:
            raise self._error(node, str(error)) from error

    def _read_binding(self, binding: _ExpressionBinding) -> expr:
        return binding.local.value_type.local(binding.local.index, module=self.module)

    def _coerce_return(self, value: object, node: ast.AST) -> expr:
        if not isinstance(value, expr) and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise self._error(node, f"return value must be a Wasm expression or numeric literal, got {type(value).__name__}")
        result = coerce_value(value, self.signature.return_type)
        if result._module is not None and result._module is not self.module:
            raise self._error(node, "return value belongs to another WebAssembly module")
        return result

    def _is_branch(self, statement: ast.With) -> bool:
        return self._is_control_with(statement, control.branch)

    def _is_block(self, statement: ast.With) -> bool:
        return self._is_control_with(statement, control.block)

    def _is_loop(self, statement: ast.With) -> bool:
        return self._is_control_with(statement, control.loop)

    def _is_control_with(self, statement: ast.With, marker: object) -> bool:
        if len(statement.items) != 1 or not isinstance(statement.items[0].context_expr, ast.Call):
            return False
        try:
            return self._expression(statement.items[0].context_expr.func) is marker
        except CompileError:
            return False

    def _control_call(self, statement: ast.With, marker: object) -> ast.Call:
        if len(statement.items) != 1 or not isinstance(statement.items[0].context_expr, ast.Call):
            raise self._error(statement, "control scopes require a single call context")
        call = statement.items[0].context_expr
        if self._expression(call.func) is not marker:
            raise self._error(call, "unsupported context manager")
        return call

    def _optional_binding_name(self, statement: ast.With) -> str | None:
        target = statement.items[0].optional_vars if statement.items else None
        if target is None:
            return None
        if not isinstance(target, ast.Name):
            raise self._error(target, "control scopes require a simple name after 'as'")
        return target.id

    def _is_else(self, statement: ast.With, name: str) -> bool:
        if len(statement.items) != 1 or statement.items[0].optional_vars is not None:
            return False
        context = statement.items[0].context_expr
        return (
            isinstance(context, ast.Call)
            and not context.args
            and not context.keywords
            and isinstance(context.func, ast.Attribute)
            and context.func.attr == "els"
            and isinstance(context.func.value, ast.Name)
            and context.func.value.id == name
        )

    def _error(self, node: ast.AST, message: str) -> CompileError:
        line = getattr(node, "lineno", None)
        location = f" at line {line}" if line is not None else ""
        return CompileError(f"{self.function.__name__}(){location}: {message}")
