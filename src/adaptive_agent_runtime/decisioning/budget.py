"""Deterministic accounting for Runtime-controlled decision budgets."""

from __future__ import annotations

from adaptive_agent_runtime.decisioning.models import (
    DecisionBudget,
    DecisionBudgetUsage,
)


def consume_decision_cycle(usage: DecisionBudgetUsage) -> DecisionBudgetUsage:
    return usage.model_copy(
        update={"decision_cycles": usage.decision_cycles + 1}
    )


def consume_agent_call(
    usage: DecisionBudgetUsage,
    *,
    elapsed_seconds: float,
    cost_units: float,
) -> DecisionBudgetUsage:
    if elapsed_seconds < 0.0 or cost_units < 0.0:
        raise ValueError("decision usage increments cannot be negative")
    return usage.model_copy(
        update={
            "agent_calls": usage.agent_calls + 1,
            "elapsed_seconds": usage.elapsed_seconds + elapsed_seconds,
            "consumed_cost_units": usage.consumed_cost_units + cost_units,
        }
    )


def consume_revision(usage: DecisionBudgetUsage) -> DecisionBudgetUsage:
    return usage.model_copy(
        update={"revision_count": usage.revision_count + 1}
    )


def budget_violations(
    budget: DecisionBudget,
    usage: DecisionBudgetUsage,
) -> tuple[str, ...]:
    violations: list[str] = []
    if usage.decision_cycles > budget.max_decision_cycles:
        violations.append("max_decision_cycles")
    if usage.agent_calls > budget.max_agent_calls:
        violations.append("max_agent_calls")
    if usage.revision_count > budget.max_revision_count:
        violations.append("max_revision_count")
    if (
        budget.max_elapsed_seconds is not None
        and usage.elapsed_seconds > budget.max_elapsed_seconds
    ):
        violations.append("max_elapsed_seconds")
    if (
        budget.max_cost_units is not None
        and usage.consumed_cost_units > budget.max_cost_units
    ):
        violations.append("max_cost_units")
    return tuple(violations)


def can_start_cycle(
    budget: DecisionBudget,
    usage: DecisionBudgetUsage,
) -> bool:
    candidate = consume_decision_cycle(usage)
    return not budget_violations(budget, candidate)


def can_call_agent(
    budget: DecisionBudget,
    usage: DecisionBudgetUsage,
) -> bool:
    return usage.agent_calls < budget.max_agent_calls
