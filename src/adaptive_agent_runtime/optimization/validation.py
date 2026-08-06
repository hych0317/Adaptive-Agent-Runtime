"""Deterministic Runtime validation for bounded Phase 4-A targets."""

from __future__ import annotations

from adaptive_agent_runtime.optimization.models import (
    OptimizationTarget,
    OptimizationTargetKey,
)


def validate_optimization_value(
    target: OptimizationTarget,
    proposed_value: object,
) -> None:
    """Reject arbitrary patches and values outside the target schema."""

    constraints = target.constraints
    if constraints.value_type == "integer":
        if isinstance(proposed_value, bool) or not isinstance(proposed_value, int):
            raise ValueError("Optimization target requires an integer value")
        if constraints.minimum is not None and proposed_value < constraints.minimum:
            raise ValueError("Optimization value is below the Runtime minimum")
        if constraints.maximum is not None and proposed_value > constraints.maximum:
            raise ValueError("Optimization value exceeds the Runtime maximum")
    elif constraints.value_type == "boolean":
        if not isinstance(proposed_value, bool):
            raise ValueError("Optimization target requires a boolean value")
    elif constraints.value_type == "strategy_ref":
        if not isinstance(proposed_value, str) or not proposed_value:
            raise ValueError("Optimization target requires a strategy reference")
    else:
        raise ValueError("Optimization target schema is not supported")
    if constraints.allowed_values and proposed_value not in constraints.allowed_values:
        raise ValueError("Optimization value is outside the Runtime allowlist")
    if proposed_value == target.current_value:
        raise ValueError("Optimization Proposal must differ from the current baseline")

    phase_4a_targets = {
        OptimizationTargetKey.PLANNER_MAX_NODES,
        OptimizationTargetKey.PLANNER_MAX_DEPTH,
    }
    if target.target_key not in phase_4a_targets:
        raise ValueError("Optimization target is not enabled in Phase 4-A")

