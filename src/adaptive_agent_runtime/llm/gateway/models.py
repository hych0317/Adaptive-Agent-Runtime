"""Routing, execution-budget, and trace models for managed inference."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.llm.errors import InferenceFailureCode
from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    LLMModel,
    utc_now,
)
from adaptive_agent_runtime.llm.models import (
    BackendKind,
    InferenceCorrelation,
    InferenceUsage,
)


class InferenceRoutingPolicy(LLMModel):
    allowed_target_ids: tuple[str, ...] = ()
    preferred_target_ids: tuple[str, ...] = ()
    required_target_tags: tuple[str, ...] = ()
    preferred_target_tags: tuple[str, ...] = ()
    allowed_backend_kinds: tuple[BackendKind, ...] = tuple(BackendKind)

    @model_validator(mode="after")
    def validate_policy(self) -> InferenceRoutingPolicy:
        groups: tuple[tuple[object, ...], ...] = (
            self.allowed_target_ids,
            self.preferred_target_ids,
            self.required_target_tags,
            self.preferred_target_tags,
            self.allowed_backend_kinds,
        )
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("inference routing policy values must be unique")
        if not self.allowed_backend_kinds:
            raise ValueError("routing policy requires allowed backend kinds")
        if self.allowed_target_ids and not set(
            self.preferred_target_ids
        ).issubset(self.allowed_target_ids):
            raise ValueError("preferred targets must be allowed targets")
        return self


class InferenceExecutionBudget(LLMModel):
    """Bound attempts, elapsed time, and the response returned to Runtime."""

    max_attempts: int = Field(default=1, ge=1)
    max_elapsed_seconds: float | None = Field(default=None, gt=0.0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_response_cost: float | None = Field(default=None, ge=0.0)
    currency: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_budget(self) -> InferenceExecutionBudget:
        if (self.max_response_cost is None) is not (self.currency is None):
            raise ValueError(
                "response cost budget and currency must appear together"
            )
        return self


class InferenceRetryPolicy(LLMModel):
    """Bound same-target retries and cross-target failure recovery."""

    retry_same_target: bool = True
    fallback_failure_codes: tuple[InferenceFailureCode, ...] = tuple(
        InferenceFailureCode
    )

    @model_validator(mode="after")
    def validate_policy(self) -> InferenceRetryPolicy:
        if len(set(self.fallback_failure_codes)) != len(
            self.fallback_failure_codes
        ):
            raise ValueError("fallback failure codes must be unique")
        return self


class InferenceGatewayPolicy(LLMModel):
    routing: InferenceRoutingPolicy = Field(
        default_factory=InferenceRoutingPolicy
    )
    budget: InferenceExecutionBudget = Field(
        default_factory=InferenceExecutionBudget
    )
    retry: InferenceRetryPolicy = Field(default_factory=InferenceRetryPolicy)


class InferenceGatewayTraceEventKind(StrEnum):
    ROUTING_COMPLETED = "inference.routing_completed"
    ATTEMPT_STARTED = "inference.attempt_started"
    ATTEMPT_FAILED = "inference.attempt_failed"
    ATTEMPT_SUCCEEDED = "inference.attempt_succeeded"
    BUDGET_REJECTED = "inference.budget_rejected"


class InferenceGatewayTraceEvent(LLMModel):
    event_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    correlation: InferenceCorrelation = Field(
        default_factory=InferenceCorrelation
    )
    trace_attributes: ImmutableJsonObject = Field(default_factory=dict)
    kind: InferenceGatewayTraceEventKind
    candidate_target_ids: tuple[str, ...] = ()
    target_id: str | None = Field(default=None, min_length=1)
    attempt_number: int | None = Field(default=None, ge=1)
    failure_code: InferenceFailureCode | None = None
    message: str | None = Field(default=None, min_length=1)
    usage: InferenceUsage | None = None
    occurred_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_event(self) -> InferenceGatewayTraceEvent:
        if len(set(self.candidate_target_ids)) != len(
            self.candidate_target_ids
        ):
            raise ValueError("trace candidate targets must be unique")
        if self.kind is InferenceGatewayTraceEventKind.ROUTING_COMPLETED:
            if self.target_id is not None or self.attempt_number is not None:
                raise ValueError("routing trace cannot identify an attempt")
        elif self.target_id is None or self.attempt_number is None:
            raise ValueError("attempt trace requires target and attempt number")
        if self.kind is InferenceGatewayTraceEventKind.ATTEMPT_FAILED:
            if self.failure_code is None:
                raise ValueError("failed attempt trace requires a failure code")
        elif self.failure_code is not None:
            raise ValueError("only failed attempt trace can carry a failure code")
        return self


class InferenceGatewayTraceEntry(LLMModel):
    sequence: int = Field(ge=1)
    event: InferenceGatewayTraceEvent
