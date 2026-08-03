"""Narrow contracts for inference registration, routing, trace, and execution."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.contracts import InferenceBackend
from adaptive_agent_runtime.llm.gateway.models import (
    InferenceGatewayPolicy,
    InferenceGatewayTraceEntry,
    InferenceGatewayTraceEvent,
)
from adaptive_agent_runtime.llm.models import (
    InferenceRequest,
    InferenceTargetProfile,
    NormalizedModelResponse,
)


@runtime_checkable
class InferenceBackendRegistry(RuntimeModule, Protocol):
    def register(self, backend: InferenceBackend) -> None: ...

    def backend_for(self, target_id: str) -> InferenceBackend: ...

    def list_profiles(self) -> tuple[InferenceTargetProfile, ...]: ...


@runtime_checkable
class InferenceRouter(RuntimeModule, Protocol):
    def candidates(
        self,
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
        profiles: Sequence[InferenceTargetProfile],
    ) -> tuple[InferenceTargetProfile, ...]: ...


@runtime_checkable
class InferenceGatewayTraceSink(RuntimeModule, Protocol):
    async def record(
        self,
        event: InferenceGatewayTraceEvent,
    ) -> InferenceGatewayTraceEntry: ...


@runtime_checkable
class InferenceGateway(RuntimeModule, Protocol):
    async def execute(
        self,
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
    ) -> NormalizedModelResponse: ...
