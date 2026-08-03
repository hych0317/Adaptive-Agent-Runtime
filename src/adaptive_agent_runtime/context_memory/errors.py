"""Errors owned by Context-Memory Runtime."""


class ContextMemoryError(Exception):
    """Base exception for Context-Memory Runtime."""


class ContextTransitionError(ContextMemoryError):
    """Raised when a Context Unit lifecycle transition is invalid."""


class ContextNotFoundError(ContextMemoryError):
    """Raised when a requested Context Unit is unavailable."""


class ContextRecoveryError(ContextMemoryError):
    """Raised when an archived Context Unit cannot be restored."""


class ContextBudgetExceededError(ContextMemoryError):
    """Raised when mandatory Context cannot fit in an assembly budget."""


class ContextSnapshotConflictError(ContextMemoryError):
    """Raised when a Context write is based on a stale snapshot."""


class MemoryConsolidationError(ContextMemoryError):
    """Raised when a Memory Candidate cannot be consolidated safely."""


class MemoryNotFoundError(ContextMemoryError):
    """Raised when a target Memory Unit is unavailable."""


class MemorySnapshotConflictError(ContextMemoryError):
    """Raised when a Memory write is based on a stale snapshot."""
