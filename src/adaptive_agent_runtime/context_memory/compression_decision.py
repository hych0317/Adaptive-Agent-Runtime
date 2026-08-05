"""Runtime-owned contracts for governed semantic Context compression."""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator

from adaptive_agent_runtime.context_memory.context_models import (
    ContextCompressionResult,
)
from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonValue,
)


CONTEXT_COMPRESSION_DECISION_TYPE = "context.semantic_compression"
CONTEXT_COMPRESSION_APPLY_OPERATION = "context.compress"

_CONTEXT_COMPRESSION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/context-memory/compression-decision",
)


def stable_context_compression_id(*parts: object) -> UUID:
    """Create a stable identity for one source revision's compression decision."""

    return uuid5(
        _CONTEXT_COMPRESSION_NAMESPACE,
        "|".join(str(item) for item in parts),
    )


class ContextCompressionExecutionPolicy(ContextMemoryModel):
    """Runtime-enforced limits for one semantic compression decision."""

    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_revision_count: int = Field(default=0, ge=0, le=0)
    target_token_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)

    def target_tokens(self, original_estimated_tokens: int) -> int:
        if original_estimated_tokens <= 1:
            raise ValueError("Context Unit is too small for token-reducing compression")
        estimated = int(original_estimated_tokens * self.target_token_ratio)
        return max(1, min(original_estimated_tokens - 1, estimated))


class ContextCompressionDecisionPayload(ContextMemoryModel):
    """Runtime snapshot kept outside the isolated Compression Agent context."""

    context_id: UUID
    source_revision: int = Field(ge=0)
    source_snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_estimated_tokens: int = Field(ge=2)
    target_max_tokens: int = Field(ge=1)
    agent_input: ImmutableJsonValue
    execution_policy: ContextCompressionExecutionPolicy = Field(
        default_factory=ContextCompressionExecutionPolicy
    )

    @model_validator(mode="after")
    def validate_token_budget(self) -> ContextCompressionDecisionPayload:
        if self.target_max_tokens >= self.original_estimated_tokens:
            raise ValueError("compression target must reduce source token usage")
        expected = self.execution_policy.target_tokens(
            self.original_estimated_tokens
        )
        if self.target_max_tokens != expected:
            raise ValueError("compression target does not match Runtime policy")
        return self


class ContextCompressionEffect(ContextMemoryModel):
    """Runtime-normalized compression output; the Agent cannot apply it."""

    context_id: UUID
    source_revision: int = Field(ge=0)
    source_snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: ImmutableJsonValue
    core_conclusions: tuple[str, ...] = Field(min_length=1)
    original_estimated_tokens: int = Field(ge=2)
    target_max_tokens: int = Field(ge=1)
    estimated_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_reduction(self) -> ContextCompressionEffect:
        if self.target_max_tokens >= self.original_estimated_tokens:
            raise ValueError("compression target must reduce source token usage")
        if self.estimated_tokens > self.target_max_tokens:
            raise ValueError("compressed Context exceeds the Runtime token target")
        if self.estimated_tokens >= self.original_estimated_tokens:
            raise ValueError("compressed Context must reduce estimated token usage")
        return self

    def to_result(self) -> ContextCompressionResult:
        return ContextCompressionResult(
            content=self.content,
            core_conclusions=self.core_conclusions,
            estimated_tokens=self.estimated_tokens,
        )
