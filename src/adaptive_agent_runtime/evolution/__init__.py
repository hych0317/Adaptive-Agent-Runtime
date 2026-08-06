"""Legacy Replay models retained without public mutation capabilities.

Phase 4-A intentionally does not export deployment, activation, or rollback
services.  Historical tests may import the explicitly internal implementation
module and opt into its disabled-by-default mutation flag.
"""
from adaptive_agent_runtime.evolution.contracts import (
    OptimizationChangePlanner,
    ReplayCaseStore,
    ReplayValidationPolicy,
    ReplayWorkloadExecutor,
    RuntimeConfigurationStore,
)
from adaptive_agent_runtime.evolution.errors import (
    EvolutionBoundaryError,
    EvolutionConflictError,
    EvolutionError,
    ReplayValidationError,
    UnsupportedOptimizationError,
)
from adaptive_agent_runtime.evolution.models import (
    OptimizationApplication,
    OptimizationApplicationStatus,
    OptimizationDeployment,
    ReplayCase,
    ReplayExecutionResult,
    ReplayObservation,
    ReplayValidation,
    RuntimeConfigurationSnapshot,
    stable_evolution_id,
)
from adaptive_agent_runtime.evolution.replay import (
    DeterministicReplayValidator,
    RuntimeReplayRunner,
)

__all__ = [
    "DeterministicReplayValidator",
    "EvolutionBoundaryError",
    "EvolutionConflictError",
    "EvolutionError",
    "OptimizationApplication",
    "OptimizationApplicationStatus",
    "OptimizationChangePlanner",
    "OptimizationDeployment",
    "ReplayCase",
    "ReplayCaseStore",
    "ReplayExecutionResult",
    "ReplayObservation",
    "ReplayValidation",
    "ReplayValidationError",
    "ReplayValidationPolicy",
    "ReplayWorkloadExecutor",
    "RuntimeConfigurationSnapshot",
    "RuntimeConfigurationStore",
    "RuntimeReplayRunner",
    "UnsupportedOptimizationError",
    "stable_evolution_id",
]
