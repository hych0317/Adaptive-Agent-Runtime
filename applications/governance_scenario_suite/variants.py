"""Explicit test-only Profile registry."""

from __future__ import annotations

from typing import Mapping, Protocol

from pydantic import JsonValue

from applications.governance_scenario_suite.contracts import (
    EffectSpec,
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.evidence import ScenarioExecution


class ScenarioExecutionContext(Protocol):
    scenario: ScenarioSpec


class ScenarioExecutor(Protocol):
    profile: ScenarioProfile

    def execute(self, context: object) -> ScenarioExecution: ...


class ProposalModel(Protocol):
    @property
    def calls(self) -> int: ...

    @property
    def contexts(self) -> tuple[str, ...]: ...

    @property
    def metadata(self) -> Mapping[str, JsonValue]: ...

    def propose(self, projected_context: str) -> EffectSpec: ...


class ScenarioExecutorRegistry:
    """Reject implicit fallbacks so every unsafe Profile is deliberate."""

    def __init__(self, executors: Mapping[ScenarioProfile, ScenarioExecutor]) -> None:
        self._executors = dict(executors)
        if len(self._executors) != len(executors):
            raise ValueError("Scenario Profile executors must be unique")
        for profile, executor in self._executors.items():
            if executor.profile is not profile:
                raise ValueError("Scenario executor is registered under another Profile")

    def resolve(self, profile: ScenarioProfile) -> ScenarioExecutor:
        try:
            return self._executors[profile]
        except KeyError as exc:
            raise ValueError(
                f"Scenario Profile '{profile.value}' has no explicit executor"
            ) from exc


class ScriptedModelStub:
    """Deterministic proposal source that records the exact projected context."""

    def __init__(self, proposals: tuple[EffectSpec, ...]) -> None:
        if not proposals:
            raise ValueError("Model Stub requires at least one proposal")
        self._proposals = proposals
        self._next = 0
        self._contexts: list[str] = []

    @property
    def calls(self) -> int:
        return self._next

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(self._contexts)

    @property
    def metadata(self) -> Mapping[str, JsonValue]:
        return {"model_kind": "scripted_stub"}

    def propose(self, projected_context: str) -> EffectSpec:
        if self._next >= len(self._proposals):
            raise RuntimeError("Model Stub proposal script is exhausted")
        self._contexts.append(projected_context)
        proposal = self._proposals[self._next]
        self._next += 1
        return proposal
