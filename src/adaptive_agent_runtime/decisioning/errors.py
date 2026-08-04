"""Failures exposed by the bounded decision lifecycle."""


class DecisionError(Exception):
    """Base error for Decision Infrastructure."""


class DecisionInvariantError(DecisionError):
    """Raised when lifecycle artifacts do not describe the same decision."""


class DecisionBudgetExceededError(DecisionError):
    """Raised when Runtime budget forbids another lifecycle operation."""


class DecisionCheckpointConflictError(DecisionError):
    """Raised when a checkpoint write is stale or reuses a revision."""


class DecisionResumeError(DecisionError):
    """Raised when a pending decision cannot safely resume."""


class DecisionInfrastructureError(DecisionError):
    """Raised when a lifecycle collaborator fails outside decision semantics."""
