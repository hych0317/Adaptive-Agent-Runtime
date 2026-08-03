"""Narrow interfaces for Capability, Provider, selection, and execution."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.tool_ecosystem.governance import ToolExecutionPolicy
from adaptive_agent_runtime.tool_ecosystem.models import (
    Capability,
    CapabilityMatch,
    CapabilityRequirement,
    ProviderAvailability,
    ToolAttempt,
    ToolInvocation,
    ToolObservation,
    ToolProviderMetadata,
    ToolProviderResult,
    ToolSelection,
    ToolSelectionContext,
    ToolTraceEntry,
    ToolTraceEvent,
)


@runtime_checkable
class CapabilityCatalog(RuntimeModule, Protocol):
    def register(self, capability: Capability) -> None: ...

    def get(self, capability_id: str) -> Capability | None: ...

    def list_all(self) -> tuple[Capability, ...]: ...


@runtime_checkable
class CapabilityMatcher(RuntimeModule, Protocol):
    def match(
        self,
        requirement: CapabilityRequirement,
        capability: Capability,
    ) -> CapabilityMatch | None: ...


@runtime_checkable
class ToolProvider(RuntimeModule, Protocol):
    @property
    def provider_id(self) -> str: ...

    async def invoke(
        self,
        invocation: ToolInvocation,
    ) -> ToolProviderResult: ...


@runtime_checkable
class ToolRegistry(RuntimeModule, Protocol):
    def register(
        self,
        metadata: ToolProviderMetadata,
        provider: ToolProvider,
    ) -> None: ...

    def metadata_for(self, provider_id: str) -> ToolProviderMetadata: ...

    def provider_for(self, provider_id: str) -> ToolProvider: ...

    def metadata_for_capability(
        self,
        capability_id: str,
    ) -> tuple[ToolProviderMetadata, ...]: ...

    def set_availability(
        self,
        provider_id: str,
        availability: ProviderAvailability,
    ) -> ToolProviderMetadata: ...

    def list_metadata(self) -> tuple[ToolProviderMetadata, ...]: ...


@runtime_checkable
class CapabilityCandidateResolver(RuntimeModule, Protocol):
    def candidates(
        self,
        requirement: CapabilityRequirement,
    ) -> tuple[ToolProviderMetadata, ...]: ...


@runtime_checkable
class ToolSelector(RuntimeModule, Protocol):
    def select(
        self,
        requirement: CapabilityRequirement,
        candidates: Sequence[ToolProviderMetadata],
        context: ToolSelectionContext,
    ) -> ToolSelection: ...


@runtime_checkable
class ToolTraceSink(RuntimeModule, Protocol):
    async def record(self, event: ToolTraceEvent) -> ToolTraceEntry: ...


@runtime_checkable
class ToolExecutionGovernor(RuntimeModule, Protocol):
    def should_retry(
        self,
        attempt: ToolAttempt,
        policy: ToolExecutionPolicy,
    ) -> bool: ...

    def retry_delay(self, policy: ToolExecutionPolicy) -> float: ...


@runtime_checkable
class ToolExecutor(RuntimeModule, Protocol):
    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy,
    ) -> ToolObservation: ...
