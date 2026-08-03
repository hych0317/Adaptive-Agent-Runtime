"""Errors raised by replay and governed Runtime evolution."""


class EvolutionError(Exception):
    """Base error for Runtime evolution."""


class EvolutionConflictError(EvolutionError):
    """A configuration or application snapshot is stale."""


class UnsupportedOptimizationError(EvolutionError):
    """The proposal cannot be translated into a concrete configuration."""


class ReplayValidationError(EvolutionError):
    """The candidate did not pass isolated replay validation."""

