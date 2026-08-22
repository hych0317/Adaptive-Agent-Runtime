"""Common deterministic orchestration across normal and recovery scenarios."""

from __future__ import annotations

from collections.abc import Sequence

from applications.governance_scenario_suite.baseline import (
    FullAARScenarioExecutor,
    PlainAgentExecutor,
)
from applications.governance_scenario_suite.context_memory_executors import (
    NoAuthoritativeConstraintExecutor,
    NoMemoryScopeFilterExecutor,
)
from applications.governance_scenario_suite.contracts import (
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.evidence import (
    ScenarioExecution,
    ScenarioRunResult,
)
from applications.governance_scenario_suite.executors import (
    NoExactEffectBindingExecutor,
)
from applications.governance_scenario_suite.recovery import (
    CrossProcessRecoveryRunner,
)
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import (
    ScenarioExecutor,
    ScenarioExecutorRegistry,
)


class ProfileExecutorAdapter:
    def __init__(
        self,
        profile: ScenarioProfile,
        delegate: ScenarioExecutor,
    ) -> None:
        self.profile = profile
        self._delegate = delegate

    def execute(self, context: object) -> ScenarioExecution:
        return self._delegate.execute(context)


def run_deterministic_suite(
    scenarios: Sequence[ScenarioSpec],
    *,
    profile: ScenarioProfile,
    scenario_ids: tuple[str, ...] | None = None,
) -> tuple[ScenarioRunResult, ...]:
    selected = _select_scenarios(scenarios, scenario_ids)
    executor = ProfileExecutorAdapter(profile, _executor_for(profile))
    normal_runner = GovernanceScenarioRunner(
        executors=ScenarioExecutorRegistry({profile: executor})
    )
    recovery_runner = CrossProcessRecoveryRunner()
    results: list[ScenarioRunResult] = []
    for scenario in selected:
        profiled = scenario.model_copy(update={"profile": profile})
        if profiled.family_id.startswith("R"):
            results.append(recovery_runner.run(profiled))
        else:
            results.append(normal_runner.run(profiled))
    return tuple(results)


def _select_scenarios(
    scenarios: Sequence[ScenarioSpec],
    scenario_ids: tuple[str, ...] | None,
) -> tuple[ScenarioSpec, ...]:
    if scenario_ids is None:
        return tuple(scenarios)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be unique")
    by_id = {item.id: item for item in scenarios}
    missing = tuple(item for item in scenario_ids if item not in by_id)
    if missing:
        raise ValueError(f"unknown scenario IDs: {', '.join(missing)}")
    return tuple(by_id[item] for item in scenario_ids)


def _executor_for(profile: ScenarioProfile) -> ScenarioExecutor:
    if profile is ScenarioProfile.PLAIN_AGENT:
        return PlainAgentExecutor()
    if profile is ScenarioProfile.NO_EXACT_EFFECT_BINDING:
        return NoExactEffectBindingExecutor()
    if profile is ScenarioProfile.NO_MEMORY_SCOPE_FILTER:
        return NoMemoryScopeFilterExecutor()
    if profile is ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK:
        return NoAuthoritativeConstraintExecutor()
    return FullAARScenarioExecutor()
