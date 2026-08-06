"""Core Action executor whose only effect path is Tool Invocation Decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from adaptive_agent_runtime.core import ActionRequest, AgentState, Observation

from adaptive_agent_runtime.tool_ecosystem import (
    ToolExecutionStatus,
    ToolInvocationProposalDraft,
)

from applications.terminal_bench.contracts import TerminalTrialJournal
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_ACTION,
    TERMINAL_COMMAND_CAPABILITY,
    TerminalCommandIntent,
    TerminalExecResult,
    TerminalExecutionState,
    utc_now,
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
        return self._observation(action, decision)

    def _observation(
        self,
        action: ActionRequest,
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
        metadata = self._metadata(
            decision,
            execution_result,
            effect_fingerprint=decision.effect_fingerprint or "0" * 64,
        )
        if execution_result.execution_state is TerminalExecutionState.COMPLETED:
            return Observation.ok(
                action.action_id,
                output=execution_result.model_dump(mode="json"),
                metadata=cast(Mapping[str, JsonValue], metadata),
            )
        return Observation.failed(
            action.action_id,
            error=(
                f"terminal execution {execution_result.execution_state.value}: "
                f"{execution_result.stderr or tool_observation.error or 'unknown state'}"
            ),
            metadata=cast(Mapping[str, JsonValue], metadata),
        )

    @staticmethod
    def _metadata(
        decision: TerminalToolDecisionResult,
        result: TerminalExecResult,
        *,
        effect_fingerprint: str,
    ) -> dict[str, JsonValue]:
        return {
            "terminal_result": result.model_dump(mode="json"),
            "terminal_decision": {
                "request_id": str(decision.request_id),
                "invocation_id": str(decision.invocation.invocation_id),
                "effect_fingerprint": effect_fingerprint,
                "status": decision.status,
            },
        }

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
