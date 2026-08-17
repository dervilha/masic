
class MasicError(Exception):
    """Base class for public masic errors."""


class CompileError(MasicError):
    """Raised when a Python declaration cannot be compiled to WebAssembly."""


class ExpressionError(CompileError):
    """Raised for an invalid operation involving typed expressions."""


class ModuleStateError(MasicError):
    """Raised when a module operation is invalid in its current lifecycle state."""
