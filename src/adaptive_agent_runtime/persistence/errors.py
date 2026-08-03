"""Storage-specific failures exposed without leaking sqlite exceptions."""


class PersistenceError(Exception):
    """Base error for durable runtime storage."""


class PersistenceConflictError(PersistenceError):
    """Raised when a write is stale or reuses an identity inconsistently."""


class PersistenceSchemaError(PersistenceError):
    """Raised when the database schema is not compatible with this runtime."""
