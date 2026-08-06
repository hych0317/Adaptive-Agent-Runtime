"""Errors raised by runtime mechanics rather than task execution."""


class RuntimeCoreError(Exception):
    """Base exception for the Runtime Core."""


class RuntimeInvariantError(RuntimeCoreError):
    """Raised when a Core state transition would violate an invariant."""


class RuntimeInfrastructureError(RuntimeCoreError):
    """Raised when state or trace infrastructure cannot preserve a run."""


class RuntimeResumeError(RuntimeCoreError):
    """Raised when a persisted run cannot be resumed safely."""


class RuntimeResumeBlockedError(RuntimeResumeError):
    """Raised when recovery requires an external decision before replay."""


class RunBudgetError(RuntimeCoreError):
    """Raised when aggregate Run usage cannot be admitted or accounted."""


class RunBudgetExhaustedError(RunBudgetError):
    """Raised before work that cannot fit inside the remaining Run budget."""

    def __init__(self, resource: str, message: str) -> None:
        super().__init__(message)
        self.resource = resource


class RunUsageAccountingError(RunBudgetError):
    """Raised when a configured aggregate budget lacks authoritative usage."""
