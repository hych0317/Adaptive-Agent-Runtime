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
