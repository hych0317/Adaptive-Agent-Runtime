"""Errors owned by Agent Orchestration."""


class OrchestrationError(Exception):
    """Base exception for orchestration failures."""


class GraphTransitionError(OrchestrationError):
    """Raised when a task graph transition is not valid."""


class OrchestrationStateError(OrchestrationError):
    """Raised when Planner state and Runtime feedback do not agree."""

