"""Immutable data contracts shared by Runtime Core modules."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Any, Mapping, TypeAlias
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
    """Return a timezone-aware timestamp for runtime records."""

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
    if isinstance(value, tuple):
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


class FrozenModel(BaseModel):
    """Pydantic model whose fields cannot be reassigned after creation."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class RunStatus(StrEnum):
    """Lifecycle status of a runtime execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanDecisionType(StrEnum):
    """The only decisions understood by the Phase 1 event loop."""

    EXECUTE = "execute"
    COMPLETE = "complete"
    FAIL = "fail"


class CoreEventKind(StrEnum):
    """Namespaced event names emitted directly by Runtime Core."""

    RUNTIME_STARTED = "runtime.started"
    RUNTIME_RESUMED = "runtime.resumed"
    PLAN_CREATED = "plan.created"
    ACTION_STARTED = "action.started"
    OBSERVATION_RECEIVED = "observation.received"
    STATE_UPDATED = "state.updated"
    RUNTIME_COMPLETED = "runtime.completed"
    RUNTIME_FAILED = "runtime.failed"


class AgentTask(FrozenModel):
    """A domain-neutral task accepted by the Runtime Core."""

    task_id: UUID = Field(default_factory=uuid4)
    description: str = Field(min_length=1)
    input: ImmutableJsonObject = Field(default_factory=dict)


class ActionRequest(FrozenModel):
    """A generic execution intent with no Tool or Agent semantics."""

    action_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    arguments: ImmutableJsonObject = Field(default_factory=dict)


class Observation(FrozenModel):
    """Feedback produced by an ActionExecutor."""

    action_id: UUID
    succeeded: bool
    output: ImmutableJsonValue = None
    error: str | None = None
    metadata: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result(self) -> Observation:
        if self.succeeded and self.error is not None:
            raise ValueError("a successful observation cannot contain an error")
        if not self.succeeded and not self.error:
            raise ValueError("a failed observation must contain an error")
        return self

    @classmethod
    def ok(
        cls,
        action_id: UUID,
        *,
        output: JsonValue = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Observation:
        return cls(
            action_id=action_id,
            succeeded=True,
            output=output,
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        action_id: UUID,
        *,
        error: str,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Observation:
        return cls(
            action_id=action_id,
            succeeded=False,
            error=error,
            metadata=metadata or {},
        )


class PlanDecision(FrozenModel):
    """A Planner decision to execute, complete, or terminate as failed."""

    decision: PlanDecisionType
    action: ActionRequest | None = None
    output: ImmutableJsonValue = None
    error: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> PlanDecision:
        if self.decision is PlanDecisionType.EXECUTE and self.action is None:
            raise ValueError("an execute decision requires an action")
        if self.decision is not PlanDecisionType.EXECUTE and self.action is not None:
            raise ValueError("only an execute decision can contain an action")
        if self.decision is PlanDecisionType.FAIL and not self.error:
            raise ValueError("a fail decision requires an error")
        if self.decision is not PlanDecisionType.FAIL and self.error is not None:
            raise ValueError("only a fail decision can contain an error")
        if self.decision is not PlanDecisionType.COMPLETE and self.output is not None:
            raise ValueError("only a complete decision can contain final output")
        return self

    @classmethod
    def execute(
        cls,
        action: ActionRequest,
        *,
        reason: str | None = None,
    ) -> PlanDecision:
        return cls(
            decision=PlanDecisionType.EXECUTE,
            action=action,
            reason=reason,
        )

    @classmethod
    def complete(
        cls,
        *,
        output: JsonValue = None,
        reason: str | None = None,
    ) -> PlanDecision:
        return cls(
            decision=PlanDecisionType.COMPLETE,
            output=output,
            reason=reason,
        )

    @classmethod
    def fail(
        cls,
        *,
        error: str,
        reason: str | None = None,
    ) -> PlanDecision:
        return cls(
            decision=PlanDecisionType.FAIL,
            error=error,
            reason=reason,
        )


class AgentState(FrozenModel):
    """An immutable snapshot of state owned by Runtime Core."""

    run_id: UUID
    task: AgentTask
    status: RunStatus = RunStatus.PENDING
    revision: int = Field(default=0, ge=0)
    step_count: int = Field(default=0, ge=0)
    last_plan: PlanDecision | None = None
    last_observation: Observation | None = None
    output: ImmutableJsonValue = None
    error: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AgentState:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.status is RunStatus.FAILED and not self.error:
            raise ValueError("a failed state must contain an error")
        if self.status is not RunStatus.FAILED and self.error is not None:
            raise ValueError("only a failed state can contain an error")
        if self.status is not RunStatus.COMPLETED and self.output is not None:
            raise ValueError("only a completed state can contain final output")
        return self


class RuntimeEvent(FrozenModel):
    """A runtime fact before it is persisted as an ordered trace entry."""

    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    occurred_at: AwareDatetime = Field(default_factory=utc_now)
    payload: ImmutableJsonObject = Field(default_factory=dict)


class TraceEntry(FrozenModel):
    """An append-only, ordered record of one RuntimeEvent."""

    entry_id: UUID = Field(default_factory=uuid4)
    sequence: int = Field(ge=1)
    event: RuntimeEvent
    recorded_at: AwareDatetime = Field(default_factory=utc_now)

    @property
    def run_id(self) -> UUID:
        return self.event.run_id


class RunResult(FrozenModel):
    """The terminal state returned from one runtime execution."""

    final_state: AgentState

    @property
    def succeeded(self) -> bool:
        return self.final_state.status is RunStatus.COMPLETED
