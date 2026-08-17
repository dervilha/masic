"""Public markers for structured control flow in compiled function source."""

from __future__ import annotations

from typing import NoReturn

from .errors import CompileError


def _outside_compiled_function(name: str) -> NoReturn:
    raise CompileError(f"{name}() can only be used in source compiled by @wasm.func")


def branch(condition: object) -> NoReturn:
    """Begin an explicit WebAssembly conditional scope.

    This marker is interpreted by the function source compiler and is never
    executed as ordinary Python code.
    """
    _outside_compiled_function("branch")


def block() -> NoReturn:
    """Begin an explicit named WebAssembly block scope."""
    _outside_compiled_function("block")


def loop() -> NoReturn:
    """Begin an explicit WebAssembly loop scope."""
    _outside_compiled_function("loop")
