"""Narrow contracts implemented by LLM inference backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.models import (
    BackendProbeResult,
    InferenceRequest,
    InferenceTargetProfile,
    NormalizedModelResponse,
)


@runtime_checkable
class InferenceBackend(RuntimeModule, Protocol):
    """Execute provider-neutral inference without Runtime apply authority."""

    @property
    def target_id(self) -> str: ...

    @property
    def profile(self) -> InferenceTargetProfile: ...

    async def probe(self) -> BackendProbeResult:
        """Return current executable, protocol, and authentication availability."""

        ...

    async def invoke(
        self,
        request: InferenceRequest,
    ) -> NormalizedModelResponse:
        """Run one bounded inference request and return a normalized response."""

        ...
