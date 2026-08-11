"""Core Action executor whose only effect path is Tool Invocation Decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from adaptive_agent_runtime.core import (
    ActionRequest,
    AgentState,
    FailureDisposition,
    FailureRecoveryStatus,
    Observation,
    ObservationControl,
    ProgressKind,
)

from adaptive_agent_runtime.tool_ecosystem import (
    ToolExecutionStatus,
    ToolInvocationProposalDraft,
)

from applications.terminal_bench.contracts import TerminalTrialJournal
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_ACTION,
    TERMINAL_COMMAND_CAPABILITY,
    TERMINAL_COMPLETION_REJECTION_ACTION,
    TERMINAL_PROPOSAL_REJECTION_ACTION,
    TerminalCommandIntent,
    TerminalExecResult,
    TerminalExecutionState,
    utc_now,
)
from applications.terminal_bench.progress import (
    TerminalProgressAssessment,
    assess_terminal_progress,
    terminal_failure_code,
    terminal_outcome_certainty,
)
from applications.terminal_bench.tool_decision import (
    TerminalToolDecisionResult,
    TerminalToolInvocationDecisionHandler,
)


class TerminalActionExecutor:
    module_id = "terminal_bench.executor.decision_bound"

    def __init__(
        self,
        *,
        decisions: TerminalToolInvocationDecisionHandler,
        journal: TerminalTrialJournal,
    ) -> None:
        self._decisions = decisions
        self._journal = journal

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        if action.name == TERMINAL_COMPLETION_REJECTION_ACTION:
            return Observation.ok(
                action.action_id,
                output={
                    "completion_rejected": True,
                    "reason": action.arguments.get("reason"),
                },
                control=ObservationControl(progress_kind=ProgressKind.NO_PROGRESS),
            )
        if action.name == TERMINAL_PROPOSAL_REJECTION_ACTION:
            return Observation.ok(
                action.action_id,
                output={
                    "proposal_rejected": True,
                    "rejection": action.arguments.get("rejection"),
                },
                control=ObservationControl(
                    progress_kind=ProgressKind.RECOVERY_PROGRESS
                ),
            )
        if action.name != TERMINAL_COMMAND_ACTION:
            return Observation.failed(
                action.action_id,
                error=f"unsupported terminal action '{action.name}'",
            )
        try:
            intent = TerminalCommandIntent.model_validate(action.arguments)
        except Exception as exc:
            return Observation.failed(
                action.action_id,
                error=f"invalid terminal action: {exc}",
            )
        proposal = ToolInvocationProposalDraft(
            call_key=intent.call_key,
            capability_id=TERMINAL_COMMAND_CAPABILITY,
            arguments=intent.tool_arguments(),
        )
        decision = await self._decisions.handle(
            proposal=proposal,
            action=action,
            state=state,
            producer_id="terminal_bench.turn_planner",
        )
        return self._observation(action, intent, decision)

    def _observation(
        self,
        action: ActionRequest,
        intent: TerminalCommandIntent,
        decision: TerminalToolDecisionResult,
    ) -> Observation:
        outcome = decision.outcome
        if outcome is None:
            result = self._not_executed_result(
                decision.reason or decision.status,
                in_doubt=decision.execution_in_doubt,
            )
            self._journal.record_execution(decision.invocation.invocation_id, result)
            metadata = self._metadata(
                decision,
                result,
                effect_fingerprint=decision.effect_fingerprint or "0" * 64,
            )
            return Observation.failed(
                action.action_id,
                error=f"terminal decision did not apply: {decision.reason or decision.status}",
                metadata=cast(Mapping[str, JsonValue], metadata),
                control=ObservationControl(
                    progress_kind=ProgressKind.NO_PROGRESS,
                    failure=FailureDisposition(
                        failure_code="terminal.decision_not_applied",
                        retryable=False,
                        recovery_status=FailureRecoveryStatus.UNAVAILABLE,
                        outcome_certainty=(
                            OutcomeCertainty.IN_DOUBT
                            if decision.execution_in_doubt
                            else OutcomeCertainty.CERTAIN
                        ),
                    ),
                ),
            )
        effect = outcome.effect
        tool_observation = outcome.observation
        execution_result = self._journal.execution_for(
            effect.invocation.invocation_id
        )
        if execution_result is None and tool_observation.succeeded:
            execution_result = TerminalExecResult.model_validate(
                tool_observation.output
            )
            self._journal.record_execution(
                effect.invocation.invocation_id,
                execution_result,
            )
        if execution_result is None:
            execution_result = self._unknown_result(tool_observation)
            self._journal.record_execution(
                effect.invocation.invocation_id,
                execution_result,
            )
        progress = assess_terminal_progress(
            intent,
            execution_result,
            self._journal.recent_records(512),
        )
        metadata = self._metadata(
            decision,
            execution_result,
            effect_fingerprint=decision.effect_fingerprint or "0" * 64,
            progress=progress,
        )
        if execution_result.settled:
            return Observation.ok(
                action.action_id,
                output=execution_result.model_dump(mode="json"),
                metadata=cast(Mapping[str, JsonValue], metadata),
                control=ObservationControl(
                    progress_kind=progress.kind,
                    progress_fingerprint=progress.fingerprint,
                ),
            )
        return Observation.failed(
            action.action_id,
            error=(
                "terminal execution "
                + (
                    "TIMED_OUT"
                    if execution_result.timed_out
                    else execution_result.execution_state.value
                )
                + ": "
                f"{execution_result.stderr or tool_observation.error or 'unknown state'}"
            ),
            metadata=cast(Mapping[str, JsonValue], metadata),
            control=ObservationControl(
                progress_kind=progress.kind,
                progress_fingerprint=progress.fingerprint,
                failure=FailureDisposition(
                    failure_code=terminal_failure_code(execution_result),
                    retryable=False,
                    recovery_status=FailureRecoveryStatus.UNAVAILABLE,
                    outcome_certainty=terminal_outcome_certainty(
                        execution_result
                    ),
                ),
            ),
        )

    @staticmethod
    def _metadata(
        decision: TerminalToolDecisionResult,
        result: TerminalExecResult,
        *,
        effect_fingerprint: str,
        progress: TerminalProgressAssessment | None = None,
    ) -> dict[str, JsonValue]:
        metadata: dict[str, JsonValue] = {
            "terminal_result": result.model_dump(mode="json"),
            "terminal_decision": {
                "request_id": str(decision.request_id),
                "invocation_id": str(decision.invocation.invocation_id),
                "effect_fingerprint": effect_fingerprint,
                "status": decision.status,
            },
        }
        if progress is not None:
            metadata["terminal_progress"] = progress.metadata()
        return metadata

    @staticmethod
    def _not_executed_result(
        reason: str,
        *,
        in_doubt: bool,
    ) -> TerminalExecResult:
        now = utc_now()
        return TerminalExecResult(
            stderr=reason,
            started_at=now,
            completed_at=now,
            duration_ms=0,
            execution_state=(
                TerminalExecutionState.IN_DOUBT
                if in_doubt
                else TerminalExecutionState.FAILED_TO_START
            ),
            transport_failed=in_doubt,
        )

    @staticmethod
    def _unknown_result(tool_observation) -> TerminalExecResult:  # type: ignore[no-untyped-def]
        timed_out = tool_observation.status is ToolExecutionStatus.TIMED_OUT
        return TerminalExecResult(
            stderr=tool_observation.error or "terminal process state is unknown",
            started_at=tool_observation.started_at,
            completed_at=tool_observation.completed_at,
            duration_ms=max(
                0,
                int(
                    (
                        tool_observation.completed_at - tool_observation.started_at
                    ).total_seconds()
                    * 1000
                ),
            ),
            execution_state=TerminalExecutionState.IN_DOUBT,
            timed_out=timed_out,
            transport_failed=True,
        )
