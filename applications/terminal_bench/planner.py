"""Sequential terminal planning, bounded context, and trial-local journaling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import JsonValue

from adaptive_agent_runtime.core import (
    ActionRequest,
    AgentState,
    Observation,
    PlanDecision,
)
from adaptive_agent_runtime.llm import (
    InferenceCorrelation,
    InferenceGateway,
    InferenceGatewayPolicy,
    InferenceRequest,
    InferenceRequirements,
    ModelResponseKind,
    StructuredOutputLevel,
)

from applications.terminal_bench.contracts import (
    TerminalTrialJournal,
    TerminalTurnProposalCapability,
)
from applications.terminal_bench.models import (
    AAR_TERMINAL_SEQUENTIAL_PROFILE,
    TERMINAL_COMMAND_ACTION,
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalPendingCommand,
    TerminalProcessReference,
    TerminalSessionSnapshot,
    TerminalTrialSummary,
    TerminalTurnDecision,
    TerminalTurnDraft,
    TerminalTurnProposal,
    TerminalTurnRequest,
    terminal_fingerprint,
    utc_now,
)


_TERMINAL_ACTION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/terminal-sequential/action",
)

_EXECUTION_SEMANTICS = (
    "Every exec call is an independent, non-interactive shell; shell-local state is not persistent.",
    "Do not rely on a previous cd, export, alias, shell variable, or interactive session.",
    "Use cwd for the working directory and env for the complete explicit environment map.",
    "A non-zero return code is an observed completed command, not a transport failure.",
    "IN_DOUBT means the command may have started; never repeat it. Use a new inspection command.",
    "Start background services in detached form and supply PID file, log path, and status command.",
    "No tmux, interactive terminal, verifier API, oracle API, sidecar execution, or host access is available.",
)


class JsonlTerminalTrialJournal:
    """Append-only transcript plus a replayed trial-local working snapshot."""

    module_id = "terminal_bench.journal.jsonl"

    def __init__(self, trial_id: str, path: str | Path | None = None) -> None:
        if not trial_id:
            raise ValueError("trial_id is required")
        self._trial_id = trial_id
        self._path = Path(path) if path is not None else None
        self._session = TerminalSessionSnapshot(trial_id=trial_id)
        self._pending: TerminalPendingCommand | None = None
        self._executions: dict[UUID, TerminalExecResult] = {}
        self._records: list[TerminalCommandRecord] = []
        self._agent_summary: str | None = None
        self._trace_consistent = True
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.is_file():
                self._replay()

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def trace_consistent(self) -> bool:
        return self._trace_consistent and self._pending is None

    def snapshot(self) -> TerminalSessionSnapshot:
        return self._session

    def recent_records(self, limit: int) -> tuple[TerminalCommandRecord, ...]:
        if limit < 1:
            return ()
        return tuple(self._records[-limit:])

    def records(self) -> tuple[TerminalCommandRecord, ...]:
        return tuple(self._records)

    def pending(self) -> TerminalPendingCommand | None:
        return self._pending

    def save_pending(self, pending: TerminalPendingCommand) -> None:
        if self._pending is not None:
            if self._pending == pending:
                return
            raise RuntimeError("another terminal command is already pending")
        if any(item.intent.call_key == pending.intent.call_key for item in self._records):
            raise ValueError("terminal call_key was already committed")
        self._pending = pending
        self._append("pending.saved", pending.model_dump(mode="json"))

    def record_execution(
        self,
        invocation_id: UUID,
        result: TerminalExecResult,
    ) -> None:
        existing = self._executions.get(invocation_id)
        if existing is not None:
            if existing != result:
                self._trace_consistent = False
                raise RuntimeError(
                    "invocation identity produced conflicting terminal results"
                )
            return
        self._executions[invocation_id] = result
        self._append(
            "execution.recorded",
            {
                "invocation_id": str(invocation_id),
                "result": result.model_dump(mode="json"),
            },
        )

    def execution_for(self, invocation_id: UUID) -> TerminalExecResult | None:
        return self._executions.get(invocation_id)

    def commit_observation(self, observation: Observation) -> None:
        if any(item.action_id == observation.action_id for item in self._records):
            if self._pending is not None and self._pending.action.action_id == observation.action_id:
                self._pending = None
            return
        pending = self._pending
        if pending is None or pending.action.action_id != observation.action_id:
            self._trace_consistent = False
            raise RuntimeError("terminal observation has no matching pending command")
        metadata = dict(observation.metadata)
        decision = metadata.get("terminal_decision")
        if not isinstance(decision, Mapping):
            self._trace_consistent = False
            raise RuntimeError("terminal observation lacks decision correlation")
        raw_result: object = (
            observation.output
            if observation.succeeded
            else metadata.get("terminal_result")
        )
        result = TerminalExecResult.model_validate(raw_result)
        invocation_id = UUID(str(decision["invocation_id"]))
        recorded = self._executions.get(invocation_id)
        if recorded is None or recorded != result:
            self._trace_consistent = False
            raise RuntimeError("observation does not match Provider execution journal")
        record = TerminalCommandRecord(
            action_id=observation.action_id,
            invocation_id=invocation_id,
            decision_request_id=UUID(str(decision["request_id"])),
            effect_fingerprint=str(decision["effect_fingerprint"]),
            intent=pending.intent,
            result=result,
            governance_status=str(decision["status"]),
        )
        self._records.append(record)
        self._pending = None
        self._session = self._advance_session(self._session, record)
        self._append(
            "command.committed",
            {
                "record": record.model_dump(mode="json"),
                "session": self._session.model_dump(mode="json"),
            },
        )

    def record_usage(self, proposal: TerminalTurnProposal) -> None:
        usage = proposal.usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        total_tokens = usage.total_tokens
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        cost = (
            usage.monetary_cost or 0.0
            if usage.currency in {None, "USD"}
            else 0.0
        )
        self._session = self._session.model_copy(
            update={
                "input_tokens": self._session.input_tokens + input_tokens,
                "output_tokens": self._session.output_tokens + output_tokens,
                "total_tokens": self._session.total_tokens + total_tokens,
                "cost_usd": self._session.cost_usd + cost,
            }
        )
        self._append(
            "inference.usage",
            {
                "usage": usage.model_dump(mode="json"),
                "session": self._session.model_dump(mode="json"),
            },
        )

    def mark_complete(self, summary: str) -> None:
        self._agent_summary = summary
        self._append("agent.completed", {"summary": summary})

    def write_summary(self, summary: TerminalTrialSummary) -> None:
        if self._path is None:
            return
        summary_path = self._path.with_name("aar-summary.json")
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary.replace(summary_path)

    def _advance_session(
        self,
        session: TerminalSessionSnapshot,
        record: TerminalCommandRecord,
    ) -> TerminalSessionSnapshot:
        result = record.result
        references = list(session.process_references)
        if (
            result.execution_state is TerminalExecutionState.COMPLETED
            and result.return_code == 0
            and record.intent.process_reference is not None
        ):
            references = [
                item
                for item in references
                if item.reference_id
                != record.intent.process_reference.reference_id
            ]
            references.append(record.intent.process_reference)
        update: dict[str, object] = {
            "committed_commands": session.committed_commands + 1,
            "timed_out_commands": session.timed_out_commands + int(result.timed_out),
            "in_doubt_commands": session.in_doubt_commands
            + int(result.execution_state is TerminalExecutionState.IN_DOUBT),
            "process_references": tuple(references),
        }
        if record.governance_status != "applied":
            update["denied_commands"] = session.denied_commands + 1
        if result.execution_state is TerminalExecutionState.COMPLETED:
            update["current_cwd"] = record.intent.cwd
            update["environment"] = dict(record.intent.env)
        return session.model_copy(update=update)

    def _append(self, kind: str, payload: object) -> None:
        if self._path is None:
            return
        event = {
            "kind": kind,
            "occurred_at": utc_now().isoformat(),
            "payload": payload,
        }
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            stream.write("\n")

    def _replay(self) -> None:
        assert self._path is not None
        for line_number, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                kind = str(event["kind"])
                payload = event["payload"]
                self._replay_event(kind, payload)
            except Exception as exc:
                raise RuntimeError(
                    f"invalid terminal transcript at line {line_number}"
                ) from exc

    def _replay_event(self, kind: str, payload: Any) -> None:
        if kind == "pending.saved":
            self._pending = TerminalPendingCommand.model_validate(payload)
        elif kind == "execution.recorded":
            self._executions[UUID(str(payload["invocation_id"]))] = (
                TerminalExecResult.model_validate(payload["result"])
            )
        elif kind == "command.committed":
            record = TerminalCommandRecord.model_validate(payload["record"])
            if not any(item.action_id == record.action_id for item in self._records):
                self._records.append(record)
            self._session = TerminalSessionSnapshot.model_validate(payload["session"])
            self._pending = None
        elif kind == "inference.usage":
            self._session = TerminalSessionSnapshot.model_validate(payload["session"])
        elif kind == "agent.completed":
            self._agent_summary = str(payload["summary"])


class GatewayTerminalTurnProposalCapability:
    """Ask the managed inference gateway for one authority-free JSON draft."""

    module_id = "terminal_bench.planner.gateway_turn_proposal"
    capability_id = "terminal_turn_proposal"

    def __init__(
        self,
        *,
        gateway: InferenceGateway,
        gateway_policy: InferenceGatewayPolicy,
        target_id: str,
        max_output_tokens: int | None = 2048,
    ) -> None:
        self._gateway = gateway
        self._gateway_policy = gateway_policy
        self._target_id = target_id
        self._max_output_tokens = max_output_tokens

    async def propose(self, request: TerminalTurnRequest) -> TerminalTurnProposal:
        inference = InferenceRequest(
            cognitive_capability_id=self.capability_id,
            required_target_id=self._target_id,
            input={
                "contract_version": "1",
                "instruction": (
                    "Solve the terminal task one bounded command at a time. "
                    "Return exactly one execute or complete draft. The Runtime, "
                    "not you, owns execution authority. Respect every execution "
                    "semantic supplied in the payload."
                ),
                "payload": request.model_dump(mode="json"),
            },
            response_schema=cast(
                dict[str, JsonValue],
                TerminalTurnDraft.model_json_schema(mode="validation"),
            ),
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA,
                max_output_tokens=self._max_output_tokens,
            ),
            correlation=InferenceCorrelation(),
            trace_attributes={
                "application": "terminal_bench",
                "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                "trial_id": request.session.trial_id,
            },
        )
        gateway_policy = self._gateway_policy
        if request.remaining_tokens is not None:
            gateway_policy = gateway_policy.model_copy(
                update={
                    "budget": gateway_policy.budget.model_copy(
                        update={"max_total_tokens": request.remaining_tokens}
                    )
                }
            )
        if request.remaining_cost_usd is not None:
            gateway_policy = gateway_policy.model_copy(
                update={
                    "budget": gateway_policy.budget.model_copy(
                        update={
                            "max_response_cost": request.remaining_cost_usd,
                            "currency": "USD",
                        }
                    )
                }
            )
        response = await self._gateway.execute(inference, gateway_policy)
        if response.kind is not ModelResponseKind.OUTPUT:
            raise ValueError("terminal planner does not accept model Tool intents")
        return TerminalTurnProposal(
            draft=TerminalTurnDraft.model_validate(response.output),
            usage=response.usage,
            model_id=response.model_id,
        )


class TerminalSequentialPlanner:
    """Map one model draft to one deterministic Core Action."""

    module_id = "terminal_bench.planner.sequential"

    def __init__(
        self,
        *,
        capability: TerminalTurnProposalCapability,
        journal: TerminalTrialJournal,
        policy: TerminalExecutionPolicy | None = None,
    ) -> None:
        self._capability = capability
        self._journal = journal
        self._policy = policy or TerminalExecutionPolicy()

    async def plan(self, state: AgentState) -> PlanDecision:
        self._commit_core_observation(state)
        pending = self._journal.pending()
        if pending is not None:
            return PlanDecision.execute(
                pending.action,
                reason="Resume the original persisted terminal proposal.",
            )
        session = self._journal.snapshot()
        budget_error = self._budget_error(session)
        if budget_error is not None:
            return PlanDecision.fail(error=budget_error)
        request = self._turn_request(state, session)
        proposal = await self._capability.propose(request)
        self._journal.record_usage(proposal)
        session = self._journal.snapshot()
        budget_error = self._budget_error(session, allow_command_limit=False)
        if budget_error is not None:
            return PlanDecision.fail(error=budget_error)
        draft = proposal.draft
        if draft.decision is TerminalTurnDecision.COMPLETE:
            assert draft.summary is not None
            self._journal.mark_complete(draft.summary)
            return PlanDecision.complete(
                output={
                    "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                    "agent_complete": True,
                    "summary": draft.summary,
                },
                reason=draft.rationale,
            )
        try:
            intent = self._resolve_intent(draft, session)
            self._validate_intent(intent)
            self._reject_duplicate_or_uncertain_replay(intent)
        except ValueError as exc:
            return PlanDecision.fail(error=f"invalid terminal proposal: {exc}")
        action_id = uuid5(
            _TERMINAL_ACTION_NAMESPACE,
            "|".join(
                (
                    str(state.run_id),
                    str(state.revision),
                    intent.call_key,
                    intent.execution_fingerprint,
                )
            ),
        )
        action = ActionRequest(
            action_id=action_id,
            name=TERMINAL_COMMAND_ACTION,
            arguments=intent.model_dump(mode="json"),
        )
        self._journal.save_pending(
            TerminalPendingCommand(
                action=action,
                intent=intent,
                state_revision=state.revision,
                proposal=draft,
            )
        )
        return PlanDecision.execute(action, reason=draft.rationale)

    def _commit_core_observation(self, state: AgentState) -> None:
        pending = self._journal.pending()
        observation = state.last_observation
        if (
            pending is not None
            and observation is not None
            and observation.action_id == pending.action.action_id
            and state.revision > pending.state_revision
        ):
            self._journal.commit_observation(observation)

    def _turn_request(
        self,
        state: AgentState,
        session: TerminalSessionSnapshot,
    ) -> TerminalTurnRequest:
        records = self._journal.recent_records(self._policy.max_context_records)
        history = tuple(
            item.history_item(output_limit=self._policy.max_context_output_characters)
            for item in records
        )
        remaining_tokens = (
            None
            if self._policy.max_total_tokens is None
            else max(0, self._policy.max_total_tokens - session.total_tokens)
        )
        remaining_cost = (
            None
            if self._policy.max_cost_usd is None
            else max(0.0, self._policy.max_cost_usd - session.cost_usd)
        )
        return TerminalTurnRequest(
            instruction=state.task.description,
            session=session,
            recent_history=history,
            remaining_commands=max(
                0,
                self._policy.max_commands - session.committed_commands,
            ),
            remaining_tokens=remaining_tokens,
            remaining_cost_usd=remaining_cost,
            execution_semantics=_EXECUTION_SEMANTICS,
        )

    def _resolve_intent(
        self,
        draft: TerminalTurnDraft,
        session: TerminalSessionSnapshot,
    ) -> TerminalCommandIntent:
        assert draft.call_key is not None and draft.command is not None
        cwd = draft.cwd if draft.cwd is not None else session.current_cwd
        environment = (
            dict(draft.env)
            if draft.env is not None
            else dict(session.environment)
        )
        timeout_sec = draft.timeout_sec or self._policy.default_timeout_sec
        return TerminalCommandIntent(
            trial_id=session.trial_id,
            call_key=draft.call_key,
            command=draft.command,
            cwd=cwd,
            env=environment,
            timeout_sec=timeout_sec,
            process_reference=draft.process_reference,
        )

    def _validate_intent(self, intent: TerminalCommandIntent) -> None:
        if len(intent.command) > self._policy.max_command_characters:
            raise ValueError("command exceeds Runtime character budget")
        if intent.timeout_sec > self._policy.max_timeout_sec:
            raise ValueError("timeout exceeds Runtime maximum")
        if len(intent.env) > self._policy.max_environment_variables:
            raise ValueError("environment map exceeds Runtime entry budget")
        if any(not key or "=" in key or "\x00" in key for key in intent.env):
            raise ValueError("environment variable name is invalid")
        if any(
            len(value) > self._policy.max_environment_value_characters
            for value in intent.env.values()
        ):
            raise ValueError("environment value exceeds Runtime character budget")

    def _reject_duplicate_or_uncertain_replay(
        self,
        intent: TerminalCommandIntent,
    ) -> None:
        records = self._journal.recent_records(self._policy.max_commands)
        if any(item.intent.call_key == intent.call_key for item in records):
            raise ValueError("call_key must be unique within the trial")
        if any(
            item.result.execution_state is TerminalExecutionState.IN_DOUBT
            and item.intent.execution_fingerprint == intent.execution_fingerprint
            for item in records
        ):
            raise ValueError(
                "an IN_DOUBT command cannot be replayed; inspect its state with a new command"
            )

    def _budget_error(
        self,
        session: TerminalSessionSnapshot,
        *,
        allow_command_limit: bool = True,
    ) -> str | None:
        if allow_command_limit and session.committed_commands >= self._policy.max_commands:
            return f"terminal command budget exhausted ({self._policy.max_commands})"
        if (
            self._policy.max_total_tokens is not None
            and session.total_tokens >= self._policy.max_total_tokens
        ):
            return "terminal model token budget exhausted"
        if (
            self._policy.max_cost_usd is not None
            and session.cost_usd >= self._policy.max_cost_usd
        ):
            return "terminal model cost budget exhausted"
        return None


def process_reference(
    *,
    reference_id: str,
    pid_file: str,
    log_path: str,
    status_check_command: str,
    stop_command: str | None = None,
) -> TerminalProcessReference:
    """Small public constructor used by deterministic tests and examples."""

    return TerminalProcessReference(
        reference_id=reference_id,
        pid_file=pid_file,
        log_path=log_path,
        status_check_command=status_check_command,
        stop_command=stop_command,
    )
