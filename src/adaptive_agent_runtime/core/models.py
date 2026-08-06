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
    TERMINATED = "terminated"


class RunPhase(StrEnum):
    """Runtime-owned phase used by phase-aware time accounting."""

    IDLE = "idle"
    PLANNING = "planning"
    ACTION_RUNNING = "action_running"
    WAITING_EXTERNAL = "waiting_external"
    RECOVERY = "recovery"


class RunTerminationReason(StrEnum):
    MAX_ACTION_STEPS = "max_action_steps"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    WALL_CLOCK_DEADLINE = "wall_clock_deadline"
    ACTIVE_EXECUTION_BUDGET = "active_execution_budget"
    TOOL_TIMEOUT = "tool_timeout"
    EXTERNAL_JOB_DEADLINE = "external_job_deadline"
    EXTERNAL_JOB_HEARTBEAT_LOST = "external_job_heartbeat_lost"
    REPEATED_INVOCATION = "repeated_invocation"
    NO_PROGRESS_STEPS = "no_progress_steps"
    NO_PROGRESS_TIME = "no_progress_time"
    CRITICAL_TOOL_FAILURE = "critical_tool_failure"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    USAGE_ACCOUNTING_UNAVAILABLE = "usage_accounting_unavailable"


class TerminationCheckPhase(StrEnum):
    BEFORE_PLANNING = "before_planning"
    AFTER_PLANNING = "after_planning"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    RESUME = "resume"


class ProgressKind(StrEnum):
    TASK_PROGRESS = "task_progress"
    RECOVERY_PROGRESS = "recovery_progress"
    NO_PROGRESS = "no_progress"


class FailureCriticality(StrEnum):
    OPTIONAL = "optional"
    REQUIRED = "required"
    CRITICAL = "critical"


class FailureRecoveryStatus(StrEnum):
    AVAILABLE = "available"
    EXHAUSTED = "exhausted"
    UNAVAILABLE = "unavailable"


class OutcomeCertainty(StrEnum):
    CERTAIN = "certain"
    IN_DOUBT = "in_doubt"


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
    STOP_TRIGGERED = "runtime.stop_triggered"
    RUNTIME_TERMINATED = "runtime.terminated"


class RunStopPolicy(FrozenModel):
    """Immutable, run-scoped stop limits owned by Runtime Core."""

    version: str = Field(default="1", min_length=1)
    max_action_steps: int = Field(default=16, ge=1)
    max_wall_clock_seconds: float | None = Field(default=None, gt=0.0)
    max_active_execution_seconds: float | None = Field(default=None, gt=0.0)
    default_tool_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_tool_timeout_seconds: float = Field(default=300.0, gt=0.0)
    external_job_deadline_seconds: float | None = Field(default=None, gt=0.0)
    cleanup_grace_seconds: float = Field(default=10.0, ge=0.0)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_monetary_cost: float | None = Field(default=None, ge=0.0)
    currency: str | None = Field(default=None, min_length=1)
    repeated_invocation_limit: int | None = Field(default=3, ge=2)
    max_no_progress_steps: int | None = Field(default=5, ge=1)
    max_no_progress_seconds: float | None = Field(default=300.0, gt=0.0)
    stop_on_critical_nonrecoverable_failure: bool = True

    @model_validator(mode="after")
    def validate_policy(self) -> RunStopPolicy:
        if self.default_tool_timeout_seconds > self.max_tool_timeout_seconds:
            raise ValueError("default Tool timeout cannot exceed its maximum")
        if (self.max_monetary_cost is None) is not (self.currency is None):
            raise ValueError("monetary cost budget and currency must appear together")
        return self


class RunUsage(FrozenModel):
    total_tokens: int = Field(default=0, ge=0)
    monetary_cost: float = Field(default=0.0, ge=0.0)
    currency: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_usage(self) -> RunUsage:
        if self.monetary_cost > 0.0 and self.currency is None:
            raise ValueError("positive monetary usage requires a currency")
        return self


class FailureDisposition(FrozenModel):
    failure_code: str = Field(min_length=1)
    criticality: FailureCriticality = FailureCriticality.REQUIRED
    retryable: bool = False
    recovery_status: FailureRecoveryStatus = FailureRecoveryStatus.UNAVAILABLE
    outcome_certainty: OutcomeCertainty = OutcomeCertainty.CERTAIN


class ObservationControl(FrozenModel):
    """Trusted execution facts consumed by deterministic stop checks."""

    progress_kind: ProgressKind | None = None
    progress_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    failure: FailureDisposition | None = None
    external_job_id: str | None = Field(default=None, min_length=1)
    external_job_heartbeat: bool = False
    external_job_completed: bool = False

    @model_validator(mode="after")
    def validate_external_job(self) -> ObservationControl:
        if (
            self.external_job_heartbeat or self.external_job_completed
        ) and self.external_job_id is None:
            raise ValueError("external Job signals require a Job id")
        if self.external_job_heartbeat and self.external_job_completed:
            raise ValueError("external Job heartbeat and completion are exclusive")
        return self


class RunTerminationEvidence(FrozenModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    data: ImmutableJsonObject = Field(default_factory=dict)


class RunTermination(FrozenModel):
    primary_reason: RunTerminationReason
    matched_reasons: tuple[RunTerminationReason, ...] = Field(min_length=1)
    phase: TerminationCheckPhase
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage: RunUsage = Field(default_factory=RunUsage)
    evidence: tuple[RunTerminationEvidence, ...] = Field(min_length=1)
    triggered_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_termination(self) -> RunTermination:
        if self.primary_reason not in self.matched_reasons:
            raise ValueError("primary termination reason must be matched")
        if len(set(self.matched_reasons)) != len(self.matched_reasons):
            raise ValueError("termination reasons must be unique")
        return self


class RunControlState(FrozenModel):
    """Persisted counters required for deterministic resume behavior."""

    policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    deadline_at: AwareDatetime | None = None
    active_execution_seconds: float = Field(default=0.0, ge=0.0)
    no_progress_active_seconds: float = Field(default=0.0, ge=0.0)
    no_progress_steps: int = Field(default=0, ge=0)
    last_progress_at: AwareDatetime | None = None
    last_progress_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    last_invocation_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    consecutive_identical_invocations: int = Field(default=0, ge=0)
    last_tool_invocation_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    consecutive_identical_tool_invocations: int = Field(default=0, ge=0)
    external_job_id: str | None = Field(default=None, min_length=1)
    external_job_started_at: AwareDatetime | None = None
    external_job_last_heartbeat_at: AwareDatetime | None = None
    phase: RunPhase = RunPhase.IDLE
    usage: RunUsage = Field(default_factory=RunUsage)


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
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    repeat_detection_exempt: bool = False


class Observation(FrozenModel):
    """Feedback produced by an ActionExecutor."""

    action_id: UUID
    succeeded: bool
    output: ImmutableJsonValue = None
    error: str | None = None
    metadata: ImmutableJsonObject = Field(default_factory=dict)
    control: ObservationControl = Field(default_factory=ObservationControl)

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
        control: ObservationControl | None = None,
    ) -> Observation:
        return cls(
            action_id=action_id,
            succeeded=True,
            output=output,
            metadata=metadata or {},
            control=control or ObservationControl(),
        )

    @classmethod
    def failed(
        cls,
        action_id: UUID,
        *,
        error: str,
        metadata: Mapping[str, JsonValue] | None = None,
        control: ObservationControl | None = None,
    ) -> Observation:
        return cls(
            action_id=action_id,
            succeeded=False,
            error=error,
            metadata=metadata or {},
            control=control or ObservationControl(),
        )


class PlanDecision(FrozenModel):
    """A Planner decision to execute, complete, or terminate as failed."""

    decision: PlanDecisionType
    action: ActionRequest | None = None
    output: ImmutableJsonValue = None
    error: str | None = None
    reason: str | None = None
    failure: FailureDisposition | None = None

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
        if self.decision is not PlanDecisionType.FAIL and self.failure is not None:
            raise ValueError("only a fail decision can contain failure disposition")
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
        failure: FailureDisposition | None = None,
    ) -> PlanDecision:
        return cls(
            decision=PlanDecisionType.FAIL,
            error=error,
            reason=reason,
            failure=failure,
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
    control: RunControlState = Field(default_factory=RunControlState)
    termination: RunTermination | None = None
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
        if self.status is RunStatus.TERMINATED and self.termination is None:
            raise ValueError("a terminated state requires a termination record")
        if (
            self.status not in {RunStatus.FAILED, RunStatus.TERMINATED}
            and self.termination is not None
        ):
            raise ValueError("only failed or terminated states accept termination")
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
