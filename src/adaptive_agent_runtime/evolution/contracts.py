"""Narrow contracts for replay, configuration, and Optimization Apply."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime import RuntimeModule
from adaptive_agent_runtime.evaluation import OptimizationProposal
from adaptive_agent_runtime.evolution.models import (
    OptimizationApplication,
    OptimizationDeployment,
    ReplayCase,
    ReplayExecutionResult,
    ReplayObservation,
    ReplayValidation,
    RuntimeConfigurationSnapshot,
)


@runtime_checkable
class ReplayWorkloadExecutor(RuntimeModule, Protocol):
    async def execute(
        self,
        case: ReplayCase,
        configuration: RuntimeConfigurationSnapshot,
    ) -> ReplayExecutionResult: ...


@runtime_checkable
class ReplayValidationPolicy(RuntimeModule, Protocol):
    def validate(
        self,
        cases: Sequence[ReplayCase],
        results: Sequence[ReplayObservation],
        candidate: RuntimeConfigurationSnapshot,
    ) -> ReplayValidation: ...


@runtime_checkable
class OptimizationChangePlanner(RuntimeModule, Protocol):
    def plan(
        self,
        proposal: OptimizationProposal,
        baseline: RuntimeConfigurationSnapshot,
    ) -> RuntimeConfigurationSnapshot: ...


@runtime_checkable
class RuntimeConfigurationStore(RuntimeModule, Protocol):
    async def initialize(self, snapshot: RuntimeConfigurationSnapshot) -> None: ...

    async def load_active(
        self,
        component: str,
    ) -> RuntimeConfigurationSnapshot | None: ...

    async def activate(
        self,
        deployment: OptimizationDeployment,
    ) -> OptimizationApplication: ...

    async def rollback(
        self,
        application_id: UUID,
    ) -> OptimizationApplication: ...

    async def load_application(
        self,
        application_id: UUID,
    ) -> OptimizationApplication | None: ...


@runtime_checkable
class ReplayCaseStore(RuntimeModule, Protocol):
    async def save(self, case: ReplayCase) -> None: ...

    async def load(self, case_id: UUID) -> ReplayCase | None: ...

    async def list_all(self) -> tuple[ReplayCase, ...]: ...
