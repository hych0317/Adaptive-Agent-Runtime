"""Deterministic stop-policy evaluation for Runtime Core."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from typing import Any

from adaptive_agent_runtime.core.models import (
    ActionRequest,
    AgentState,
    FailureCriticality,
    FailureRecoveryStatus,
    Observation,
    ProgressKind,
    RunControlState,
    RunPhase,
    RunStatus,
    RunStopPolicy,
    RunTermination,
    RunTerminationEvidence,
    RunTerminationReason,
    RunUsage,
    TerminationCheckPhase,
    utc_now,
)


_VOLATILE_ARGUMENT_KEYS = frozenset(
    {
        "action_id",
        "call_key",
        "request_id",
        "trace_id",
        "invocation_id",
        "event_id",
        "timestamp",
        "created_at",
        "updated_at",
    }
)

_REASON_PRIORITY = (
    RunTerminationReason.CRITICAL_TOOL_FAILURE,
    RunTerminationReason.RECOVERY_EXHAUSTED,
    RunTerminationReason.WALL_CLOCK_DEADLINE,
    RunTerminationReason.COST_BUDGET_EXHAUSTED,
    RunTerminationReason.TOKEN_BUDGET_EXHAUSTED,
    RunTerminationReason.ACTIVE_EXECUTION_BUDGET,
    RunTerminationReason.MAX_ACTION_STEPS,
    RunTerminationReason.TOOL_TIMEOUT,
    RunTerminationReason.REPEATED_INVOCATION,
    RunTerminationReason.NO_PROGRESS_STEPS,
    RunTerminationReason.NO_PROGRESS_TIME,
    RunTerminationReason.EXTERNAL_JOB_DEADLINE,
    RunTerminationReason.EXTERNAL_JOB_HEARTBEAT_LOST,
    RunTerminationReason.USAGE_ACCOUNTING_UNAVAILABLE,
)


@dataclass(frozen=True)
class StopAssessment:
    termination: RunTermination
    terminal_status: RunStatus


def stop_policy_fingerprint(policy: RunStopPolicy) -> str:
    return _fingerprint(policy.model_dump(mode="json"))


def invocation_fingerprint(action: ActionRequest) -> str:
    return _fingerprint(
        {
            "name": action.name,
            "arguments": _remove_volatile(action.arguments),
            "timeout_seconds": action.timeout_seconds,
        }
    )


class RunTerminationController:
    """Evaluate one frozen policy without consulting an Agent."""

    def __init__(
        self,
        policy: RunStopPolicy,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self.policy_fingerprint = stop_policy_fingerprint(policy)

    def initial_control(self, created_at: datetime) -> RunControlState:
        deadline = None
        if self.policy.max_wall_clock_seconds is not None:
            deadline = created_at + timedelta(
                seconds=self.policy.max_wall_clock_seconds
            )
        return RunControlState(
            policy_fingerprint=self.policy_fingerprint,
            deadline_at=deadline,
            last_progress_at=created_at,
            usage=RunUsage(currency=self.policy.currency),
        )

    def reconcile_control(self, state: AgentState) -> RunControlState:
        control = state.control
        if control.policy_fingerprint is None:
            deadline = None
            if self.policy.max_wall_clock_seconds is not None:
                deadline = state.created_at + timedelta(
                    seconds=self.policy.max_wall_clock_seconds
                )
            return control.model_copy(
                update={
                    "policy_fingerprint": self.policy_fingerprint,
                    "deadline_at": deadline,
                    "last_progress_at": control.last_progress_at or state.created_at,
                    "usage": control.usage.model_copy(
                        update={"currency": control.usage.currency or self.policy.currency}
                    ),
                }
            )
        if control.policy_fingerprint != self.policy_fingerprint:
            raise ValueError("persisted Run stop policy does not match this Runtime")
        return control

    def with_usage(
        self,
        control: RunControlState,
        usage: RunUsage,
    ) -> RunControlState:
        return control.model_copy(update={"usage": usage})

    def explicit_stop(
        self,
        reason: RunTerminationReason,
        *,
        phase: TerminationCheckPhase,
        control: RunControlState,
        message: str,
    ) -> StopAssessment:
        assessment = self._assessment(
            (reason,),
            phase=phase,
            control=control,
            evidence=(
                RunTerminationEvidence(
                    code=reason.value,
                    message=message,
                    data={},
                ),
            ),
        )
        assert assessment is not None
        return assessment

    def after_planning(
        self,
        control: RunControlState,
        *,
        elapsed_seconds: float,
        usage: RunUsage,
    ) -> RunControlState:
        return control.model_copy(
            update={
                "active_execution_seconds": (
                    control.active_execution_seconds + max(0.0, elapsed_seconds)
                ),
                "pending_planning_seconds": (
                    control.pending_planning_seconds
                    + max(0.0, elapsed_seconds)
                ),
                "phase": (
                    RunPhase.WAITING_EXTERNAL
                    if control.external_job_id is not None
                    else RunPhase.IDLE
                ),
                "usage": usage,
            }
        )

    def resolve_planning_without_action(
        self,
        control: RunControlState,
        *,
        made_progress: bool,
    ) -> RunControlState:
        """Discard deferred planning time when no Action can classify it.

        A failed or voluntarily final planning turn is not evidence that task
        execution stagnated. Only an executed Action may classify its preceding
        planning time as progress or no progress.
        """

        update: dict[str, object] = {"pending_planning_seconds": 0.0}
        if made_progress:
            update.update(
                {
                    "no_progress_steps": 0,
                    "no_progress_active_seconds": 0.0,
                    "last_progress_at": self._clock(),
                }
            )
        return control.model_copy(update=update)

    def before_planning(
        self,
        control: RunControlState,
        *,
        phase: TerminationCheckPhase,
        include_stagnation: bool = True,
    ) -> StopAssessment | None:
        return self._assessment(
            self._limit_reasons(
                control,
                include_stagnation=include_stagnation,
            ),
            phase=phase,
            control=control,
        )

    def before_action(
        self,
        state: AgentState,
        action: ActionRequest,
        control: RunControlState,
    ) -> tuple[RunControlState, StopAssessment | None]:
        reasons = list(self._limit_reasons(control))
        evidence: list[RunTerminationEvidence] = []
        if state.step_count >= self.policy.max_action_steps:
            reasons.append(RunTerminationReason.MAX_ACTION_STEPS)
            evidence.append(
                RunTerminationEvidence(
                    code="action_steps.exhausted",
                    message=(
                        "maximum action steps reached "
                        f"({self.policy.max_action_steps})"
                    ),
                    data={
                        "step_count": state.step_count,
                        "limit": self.policy.max_action_steps,
                    },
                )
            )

        invocation = invocation_fingerprint(action)
        repeated = (
            control.consecutive_identical_invocations + 1
            if control.last_invocation_fingerprint == invocation
            else 1
        )
        control = control.model_copy(
            update={
                "last_invocation_fingerprint": invocation,
                "consecutive_identical_invocations": repeated,
            }
        )
        if (
            not action.repeat_detection_exempt
            and self.policy.repeated_invocation_limit is not None
            and repeated >= self.policy.repeated_invocation_limit
        ):
            reasons.append(RunTerminationReason.REPEATED_INVOCATION)
            evidence.append(
                RunTerminationEvidence(
                    code="invocation.repeated",
                    message="consecutive identical invocation limit reached",
                    data={
                        "fingerprint": invocation,
                        "count": repeated,
                        "limit": self.policy.repeated_invocation_limit,
                    },
                )
            )

        if (
            action.timeout_seconds is not None
            and action.timeout_seconds > self.policy.max_tool_timeout_seconds
        ):
            reasons.append(RunTerminationReason.TOOL_TIMEOUT)
            evidence.append(
                RunTerminationEvidence(
                    code="tool.timeout_above_maximum",
                    message="declared Tool timeout exceeds the Runtime maximum",
                    data={
                        "declared_seconds": action.timeout_seconds,
                        "maximum_seconds": self.policy.max_tool_timeout_seconds,
                    },
                )
            )
        if control.deadline_at is not None and action.timeout_seconds is not None:
            remaining = (control.deadline_at - self._clock()).total_seconds()
            required = action.timeout_seconds + self.policy.cleanup_grace_seconds
            if remaining < required:
                reasons.append(RunTerminationReason.WALL_CLOCK_DEADLINE)
                evidence.append(
                    RunTerminationEvidence(
                        code="tool.insufficient_deadline",
                        message=(
                            "remaining Run wall-clock time cannot admit the full "
                            "Tool timeout and cleanup grace"
                        ),
                        data={
                            "remaining_seconds": max(0.0, remaining),
                            "required_seconds": required,
                        },
                    )
                )
        assessment = self._assessment(
            tuple(reasons),
            phase=TerminationCheckPhase.BEFORE_ACTION,
            control=control,
            evidence=tuple(evidence),
        )
        return control, assessment

    def after_action(
        self,
        control: RunControlState,
        action: ActionRequest,
        observation: Observation,
        *,
        elapsed_seconds: float,
        usage: RunUsage,
    ) -> tuple[RunControlState, StopAssessment | None]:
        del action
        control = control.model_copy(
            update={
                "active_execution_seconds": (
                    control.active_execution_seconds + max(0.0, elapsed_seconds)
                ),
                "phase": RunPhase.IDLE,
                "usage": usage,
            }
        )
        progress = observation.control.progress_kind
        if progress is None:
            progress = (
                ProgressKind.TASK_PROGRESS
                if observation.succeeded
                else ProgressKind.NO_PROGRESS
            )
        progress_fingerprint = observation.control.progress_fingerprint
        if progress_fingerprint is None and progress is ProgressKind.TASK_PROGRESS:
            progress_fingerprint = _fingerprint(
                {
                    "succeeded": observation.succeeded,
                    "output": observation.output,
                }
            )
        if progress in {
            ProgressKind.TASK_PROGRESS,
            ProgressKind.RECOVERY_PROGRESS,
        }:
            control = control.model_copy(
                update={
                    "no_progress_steps": 0,
                    "no_progress_active_seconds": 0.0,
                    "pending_planning_seconds": 0.0,
                    "last_progress_at": self._clock(),
                    "last_progress_fingerprint": progress_fingerprint,
                }
            )
        elif progress is ProgressKind.NO_PROGRESS:
            control = control.model_copy(
                update={
                    "no_progress_steps": control.no_progress_steps + 1,
                    "no_progress_active_seconds": (
                        control.no_progress_active_seconds
                        + control.pending_planning_seconds
                    ),
                    "pending_planning_seconds": 0.0,
                }
            )
        elif observation.control.external_job_heartbeat:
            control = control.model_copy(
                update={
                    "no_progress_active_seconds": 0.0,
                    "pending_planning_seconds": 0.0,
                }
            )

        job_id = observation.control.external_job_id
        if observation.control.external_job_completed:
            if control.external_job_id in {None, job_id}:
                control = control.model_copy(
                    update={
                        "external_job_id": None,
                        "external_job_started_at": None,
                        "external_job_last_heartbeat_at": None,
                        "phase": RunPhase.IDLE,
                    }
                )
        elif observation.control.external_job_heartbeat:
            now = self._clock()
            is_new = control.external_job_id != job_id
            control = control.model_copy(
                update={
                    "external_job_id": job_id,
                    "external_job_started_at": (
                        now if is_new else control.external_job_started_at or now
                    ),
                    "external_job_last_heartbeat_at": now,
                    "phase": RunPhase.WAITING_EXTERNAL,
                    "no_progress_steps": 0,
                    "no_progress_active_seconds": 0.0,
                    "pending_planning_seconds": 0.0,
                }
            )

        reasons = list(self._limit_reasons(control))
        now = self._clock()
        if (
            self.policy.external_job_deadline_seconds is not None
            and control.external_job_started_at is not None
            and (now - control.external_job_started_at).total_seconds()
            >= self.policy.external_job_deadline_seconds
        ):
            reasons.append(RunTerminationReason.EXTERNAL_JOB_DEADLINE)
        if (
            self.policy.max_no_progress_seconds is not None
            and control.external_job_last_heartbeat_at is not None
            and not observation.control.external_job_heartbeat
            and (now - control.external_job_last_heartbeat_at).total_seconds()
            >= self.policy.max_no_progress_seconds
        ):
            reasons.append(RunTerminationReason.EXTERNAL_JOB_HEARTBEAT_LOST)
        failure = observation.control.failure
        if (
            failure is not None
            and self.policy.stop_on_critical_nonrecoverable_failure
            and failure.criticality is FailureCriticality.CRITICAL
            and not failure.retryable
            and failure.recovery_status
            in {FailureRecoveryStatus.EXHAUSTED, FailureRecoveryStatus.UNAVAILABLE}
        ):
            reasons.append(RunTerminationReason.CRITICAL_TOOL_FAILURE)
            if failure.recovery_status is FailureRecoveryStatus.EXHAUSTED:
                reasons.append(RunTerminationReason.RECOVERY_EXHAUSTED)
        assessment = self._assessment(
            tuple(reasons),
            phase=TerminationCheckPhase.AFTER_ACTION,
            control=control,
            evidence=(
                RunTerminationEvidence(
                    code=(failure.failure_code if failure is not None else "limits"),
                    message=(
                        observation.error
                        or "Runtime stop policy matched after Action execution"
                    ),
                    data={"action_succeeded": observation.succeeded},
                ),
            ),
        )
        return control, assessment

    def _absolute_limit_reasons(
        self,
        control: RunControlState,
    ) -> tuple[RunTerminationReason, ...]:
        reasons: list[RunTerminationReason] = []
        if control.deadline_at is not None and self._clock() >= control.deadline_at:
            reasons.append(RunTerminationReason.WALL_CLOCK_DEADLINE)
        if (
            self.policy.max_active_execution_seconds is not None
            and control.active_execution_seconds
            >= self.policy.max_active_execution_seconds
        ):
            reasons.append(RunTerminationReason.ACTIVE_EXECUTION_BUDGET)
        if (
            self.policy.max_total_tokens is not None
            and control.usage.total_tokens >= self.policy.max_total_tokens
        ):
            reasons.append(RunTerminationReason.TOKEN_BUDGET_EXHAUSTED)
        if (
            self.policy.max_monetary_cost is not None
            and control.usage.monetary_cost >= self.policy.max_monetary_cost
        ):
            reasons.append(RunTerminationReason.COST_BUDGET_EXHAUSTED)
        return tuple(reasons)

    def _stagnation_limit_reasons(
        self,
        control: RunControlState,
    ) -> tuple[RunTerminationReason, ...]:
        reasons: list[RunTerminationReason] = []
        if (
            self.policy.max_no_progress_steps is not None
            and control.no_progress_steps >= self.policy.max_no_progress_steps
        ):
            reasons.append(RunTerminationReason.NO_PROGRESS_STEPS)
        if (
            self.policy.max_no_progress_seconds is not None
            and control.no_progress_active_seconds
            >= self.policy.max_no_progress_seconds
        ):
            reasons.append(RunTerminationReason.NO_PROGRESS_TIME)
        return tuple(reasons)

    def _limit_reasons(
        self,
        control: RunControlState,
        *,
        include_stagnation: bool = True,
    ) -> tuple[RunTerminationReason, ...]:
        reasons = list(self._absolute_limit_reasons(control))
        if include_stagnation:
            reasons.extend(self._stagnation_limit_reasons(control))
        return tuple(reasons)

    def _assessment(
        self,
        reasons: Sequence[RunTerminationReason],
        *,
        phase: TerminationCheckPhase,
        control: RunControlState,
        evidence: tuple[RunTerminationEvidence, ...] = (),
    ) -> StopAssessment | None:
        unique = tuple(dict.fromkeys(reasons))
        if not unique:
            return None
        primary = next(reason for reason in _REASON_PRIORITY if reason in unique)
        if not evidence:
            evidence = (
                RunTerminationEvidence(
                    code=primary.value,
                    message=f"Runtime stop condition matched: {primary.value}",
                    data={},
                ),
            )
        terminal_status = (
            RunStatus.FAILED
            if primary
            in {
                RunTerminationReason.CRITICAL_TOOL_FAILURE,
                RunTerminationReason.RECOVERY_EXHAUSTED,
            }
            else RunStatus.TERMINATED
        )
        return StopAssessment(
            termination=RunTermination(
                primary_reason=primary,
                matched_reasons=unique,
                phase=phase,
                policy_fingerprint=self.policy_fingerprint,
                usage=control.usage,
                evidence=evidence,
                triggered_at=self._clock(),
            ),
            terminal_status=terminal_status,
        )


def _remove_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _remove_volatile(item)
            for key, item in value.items()
            if str(key).lower() not in _VOLATILE_ARGUMENT_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_remove_volatile(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
