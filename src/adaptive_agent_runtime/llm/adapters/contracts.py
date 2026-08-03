"""Contracts at the Runtime Context-to-LLM information boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.adapters.models import (
    ContextProjectionRequest,
    LLMContextPackage,
)
from adaptive_agent_runtime.llm.models import (
    InferenceRequest,
    InferenceTargetProfile,
    NormalizedModelResponse,
)


@runtime_checkable
class LLMContextAdapter(RuntimeModule, Protocol):
    def project(self, request: ContextProjectionRequest) -> LLMContextPackage:
        """Project governed Runtime context without mutating either side."""

        ...


@runtime_checkable
class InferenceResponseValidator(RuntimeModule, Protocol):
    def validate(
        self,
        request: InferenceRequest,
        target: InferenceTargetProfile,
        response: NormalizedModelResponse,
    ) -> NormalizedModelResponse:
        """Validate normalized output against its request and target profile."""

        ...
