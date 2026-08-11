"""Immutable Capability, Provider, execution, and trace models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Any, Mapping, Self, TypeAlias
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    model_validator,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


ImmutableJsonObject: TypeAlias = Annotated[
    Mapping[str, JsonValue],
    BeforeValidator(_thaw_json),
    AfterValidator(_freeze_json),
    PlainSerializer(_thaw_json),
]
ImmutableJsonValue: TypeAlias = Annotated[
    JsonValue,
    BeforeValidator(_thaw_json),
    AfterValidator(_freeze_json),
    PlainSerializer(_thaw_json),
]


class ToolModel(BaseModel):
    """Deeply immutable base model for Tool Ecosystem snapshots."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        values = self.model_dump(mode="python")
        if update:
            values.update(update)
        return self.__class__.model_validate(values)


class ProviderAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ToolAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ToolExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    POLICY_REJECTED = "policy_rejected"


class ToolProviderOutcome(StrEnum):
    """Provider-declared result before retry and execution aggregation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class RetryStatus(StrEnum):
    NOT_RETRIED = "not_retried"
    SUCCEEDED_AFTER_RETRY = "succeeded_after_retry"
    FAILED_AFTER_RETRY = "failed_after_retry"
    EXHAUSTED = "exhausted"
    ABORTED = "aborted"


class ToolTraceEventKind(StrEnum):
    EXECUTION_STARTED = "tool.execution_started"
    PROVIDER_UNAVAILABLE = "tool.provider_unavailable"
    ATTEMPT_STARTED = "tool.attempt_started"
    ATTEMPT_SUCCEEDED = "tool.attempt_succeeded"
    ATTEMPT_FAILED = "tool.attempt_failed"
    ATTEMPT_TIMED_OUT = "tool.attempt_timed_out"
    RETRY_SCHEDULED = "tool.retry_scheduled"
    EXECUTION_FINISHED = "tool.execution_finished"


class Capability(ToolModel):
    capability_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_tags(self) -> Capability:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("capability tags must be unique")
        return self


class CapabilityRequirement(ToolModel):
    requirement_id: UUID = Field(default_factory=uuid4)
    capability_id: str = Field(min_length=1)
    required_capability_tags: tuple[str, ...] = ()
    required_provider_tags: tuple[str, ...] = ()
    preferred_provider_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_tags(self) -> CapabilityRequirement:
        groups = (
            self.required_capability_tags,
            self.required_provider_tags,
            self.preferred_provider_tags,
        )
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("capability requirement tags must be unique")
        return self


class CapabilityMatch(ToolModel):
    requirement_id: UUID
    capability: Capability
    score: float = Field(ge=0.0, le=1.0)


class ToolProviderMetadata(ToolModel):
    provider_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: ImmutableJsonObject = Field(default_factory=dict)
    availability: ProviderAvailability = ProviderAvailability.AVAILABLE
    tags: tuple[str, ...] = ()
    selection_priority: int = 0

    @model_validator(mode="after")
    def validate_tags(self) -> ToolProviderMetadata:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("provider tags must be unique")
        return self


class ToolSelectionContext(ToolModel):
    tags: tuple[str, ...] = ()
    facts: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tags(self) -> ToolSelectionContext:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("selection context tags must be unique")
        return self


class CapabilityRequest(ToolModel):
    requirement: CapabilityRequirement
    arguments: ImmutableJsonObject = Field(default_factory=dict)
    selection_context: ToolSelectionContext = Field(
        default_factory=ToolSelectionContext
    )


class ToolSelection(ToolModel):
    requirement_id: UUID
    capability_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ToolCorrelation(ToolModel):
    """Optional Runtime identifiers copied into every Tool trace fact."""

    run_id: UUID | None = None
    task_id: UUID | None = None
    node_id: UUID | None = None
    action_id: UUID | None = None


class ToolInvocation(ToolModel):
    invocation_id: UUID = Field(default_factory=uuid4)
    requirement_id: UUID
    capability_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    arguments: ImmutableJsonObject = Field(default_factory=dict)
    correlation: ToolCorrelation = Field(default_factory=ToolCorrelation)
    requested_at: AwareDatetime = Field(default_factory=utc_now)


class ToolProviderResult(ToolModel):
    outcome: ToolProviderOutcome
    output: ImmutableJsonValue = None
    error: str | None = None
    retryable: bool = True

    @model_validator(mode="after")
    def validate_result(self) -> ToolProviderResult:
        if self.outcome is ToolProviderOutcome.SUCCEEDED:
            if self.error is not None:
                raise ValueError("successful provider result cannot contain an error")
            if self.retryable:
                raise ValueError("successful provider result cannot be retryable")
        elif not self.error:
            raise ValueError("failed provider result requires an error")
        return self

    @property
    def succeeded(self) -> bool:
        return self.outcome is ToolProviderOutcome.SUCCEEDED

    @property
    def timed_out(self) -> bool:
        return self.outcome is ToolProviderOutcome.TIMED_OUT

    @classmethod
    def ok(cls, *, output: JsonValue = None) -> ToolProviderResult:
        return cls(
            outcome=ToolProviderOutcome.SUCCEEDED,
            output=output,
            retryable=False,
        )

    @classmethod
    def failed(
        cls,
        *,
        error: str,
        retryable: bool = True,
    ) -> ToolProviderResult:
        return cls(
            outcome=ToolProviderOutcome.FAILED,
            error=error,
            retryable=retryable,
        )

    @classmethod
    def timed_out_result(
        cls,
        *,
        error: str,
        retryable: bool = True,
    ) -> ToolProviderResult:
        return cls(
            outcome=ToolProviderOutcome.TIMED_OUT,
            error=error,
            retryable=retryable,
        )


class ToolAttempt(ToolModel):
    attempt_number: int = Field(ge=1)
    status: ToolAttemptStatus
    started_at: AwareDatetime
    completed_at: AwareDatetime
    output: ImmutableJsonValue = None
    error: str | None = None
    retryable: bool = False

    @model_validator(mode="after")
    def validate_attempt(self) -> ToolAttempt:
        if self.completed_at < self.started_at:
            raise ValueError("tool attempt completion cannot precede its start")
        if self.status is ToolAttemptStatus.SUCCEEDED:
            if self.error is not None or self.retryable:
                raise ValueError("successful tool attempt has invalid failure fields")
        elif not self.error:
            raise ValueError("failed or timed-out tool attempt requires an error")
        return self


class ToolObservation(ToolModel):
    invocation_id: UUID
    requirement_id: UUID
    capability_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    status: ToolExecutionStatus
    retry_status: RetryStatus
    output: ImmutableJsonValue = None
    error: str | None = None
    attempts: tuple[ToolAttempt, ...] = ()
    correlation: ToolCorrelation = Field(default_factory=ToolCorrelation)
    started_at: AwareDatetime
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_observation(self) -> ToolObservation:
        if self.completed_at < self.started_at:
            raise ValueError("tool completion cannot precede its start")
        numbers = tuple(item.attempt_number for item in self.attempts)
        if numbers != tuple(range(1, len(self.attempts) + 1)):
            raise ValueError("tool attempt numbers must be consecutive")
        if self.status in {
            ToolExecutionStatus.PROVIDER_UNAVAILABLE,
            ToolExecutionStatus.POLICY_REJECTED,
        }:
            if not self.error:
                raise ValueError("pre-execution rejection requires an error")
            if (
                self.status is ToolExecutionStatus.POLICY_REJECTED
                and self.attempts
            ):
                raise ValueError("policy rejection cannot contain attempts")
            if any(
                attempt.status is ToolAttemptStatus.SUCCEEDED
                for attempt in self.attempts
            ):
                raise ValueError("execution cannot be rejected after success")
            expected_retry_status = (
                RetryStatus.ABORTED
                if (
                    self.status is ToolExecutionStatus.PROVIDER_UNAVAILABLE
                    and self.attempts
                )
                else RetryStatus.NOT_RETRIED
            )
        elif not self.attempts:
            raise ValueError("tool execution result requires at least one attempt")
        elif self.status is ToolExecutionStatus.SUCCEEDED:
            if self.error is not None:
                raise ValueError("successful tool observation cannot contain an error")
            if self.attempts[-1].status is not ToolAttemptStatus.SUCCEEDED:
                raise ValueError("successful execution requires a successful last attempt")
            if any(
                attempt.status is ToolAttemptStatus.SUCCEEDED
                for attempt in self.attempts[:-1]
            ):
                raise ValueError("execution cannot continue after a successful attempt")
            expected_retry_status = (
                RetryStatus.SUCCEEDED_AFTER_RETRY
                if len(self.attempts) > 1
                else RetryStatus.NOT_RETRIED
            )
        else:
            if not self.error:
                raise ValueError("failed tool observation requires an error")
            if self.output is not None:
                raise ValueError("failed tool observation cannot contain output")
            if any(
                attempt.status is ToolAttemptStatus.SUCCEEDED
                for attempt in self.attempts
            ):
                raise ValueError("failed execution cannot contain a successful attempt")
            expected = (
                ToolAttemptStatus.TIMED_OUT
                if self.status is ToolExecutionStatus.TIMED_OUT
                else ToolAttemptStatus.FAILED
            )
            if self.attempts[-1].status is not expected:
                raise ValueError("tool status must match its last attempt")
            expected_retry_status = RetryStatus.NOT_RETRIED
            if len(self.attempts) > 1:
                expected_retry_status = (
                    RetryStatus.EXHAUSTED
                    if self.attempts[-1].retryable
                    else RetryStatus.FAILED_AFTER_RETRY
                )
        if self.status is not ToolExecutionStatus.SUCCEEDED and self.output is not None:
            raise ValueError("unsuccessful tool observation cannot contain output")
        if self.retry_status is not expected_retry_status:
            raise ValueError("retry status is inconsistent with execution attempts")
        return self

    @property
    def succeeded(self) -> bool:
        return self.status is ToolExecutionStatus.SUCCEEDED


class ToolTraceEvent(ToolModel):
    event_id: UUID = Field(default_factory=uuid4)
    invocation_id: UUID
    requirement_id: UUID
    capability_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    kind: ToolTraceEventKind
    attempt_number: int | None = Field(default=None, ge=1)
    correlation: ToolCorrelation = Field(default_factory=ToolCorrelation)
    occurred_at: AwareDatetime = Field(default_factory=utc_now)
    payload: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_attempt_number(self) -> ToolTraceEvent:
        attempt_kinds = {
            ToolTraceEventKind.ATTEMPT_STARTED,
            ToolTraceEventKind.ATTEMPT_SUCCEEDED,
            ToolTraceEventKind.ATTEMPT_FAILED,
            ToolTraceEventKind.ATTEMPT_TIMED_OUT,
            ToolTraceEventKind.RETRY_SCHEDULED,
        }
        if self.kind in attempt_kinds and self.attempt_number is None:
            raise ValueError("attempt and retry trace events require attempt_number")
        if self.kind not in attempt_kinds and self.attempt_number is not None:
            raise ValueError("execution-level trace events cannot have attempt_number")
        if self.kind is ToolTraceEventKind.RETRY_SCHEDULED:
            next_attempt = self.payload.get("next_attempt")
            if (
                isinstance(next_attempt, bool)
                or next_attempt != (self.attempt_number or 0) + 1
            ):
                raise ValueError(
                    "retry trace next_attempt must follow attempt_number"
                )
        return self


class ToolTraceEntry(ToolModel):
    entry_id: UUID = Field(default_factory=uuid4)
    sequence: int = Field(ge=1)
    event: ToolTraceEvent
    recorded_at: AwareDatetime = Field(default_factory=utc_now)
