"""Sequential terminal planning, bounded context, and trial-local journaling."""

from __future__ import annotations

import json
import posixpath
import re
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
    RunBudgetExhaustedError,
    RunTermination,
)
from adaptive_agent_runtime.llm import (
    InferenceCorrelation,
    InferenceGateway,
    InferenceGatewayPolicy,
    InferenceRequest,
    InferenceRequirements,
    InferenceExecutionBudgetError,
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
    TERMINAL_COMPLETION_REJECTION_ACTION,
    TERMINAL_PROPOSAL_REJECTION_ACTION,
    TerminalCommandRole,
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionLimits,
    TerminalExecutionState,
    TerminalPendingCommand,
    TerminalProposalRejection,
    TerminalProcessReference,
    TerminalRequirement,
    TerminalRejectedDraft,
    TerminalSessionSnapshot,
    TerminalTrialSummary,
    TerminalTurnDecision,
    TerminalTurnDraft,
    TerminalTurnProposal,
    TerminalTurnRequest,
    TerminalToolCapabilities,
    TerminalVerifiedCheckpoint,
    TerminalVerificationStatePolicy,
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
    "cwd must be null or an absolute POSIX path; never pass '.' or another relative path.",
    "A non-zero return code is an observed completed command, not a transport failure.",
    "IN_DOUBT means the command may have started; never repeat it. Use a new inspection command.",
    "Start background services in detached form and supply PID file, log path, and status command.",
    "Label read-only discovery and capability probes as inspect; inspect success is not task completion.",
    "Host-side Codex inference workspace paths never exist inside the task container.",
    "Label commands that create or change task artifacts as work.",
    "Before complete, run an independent verify command whose exit status encodes the task checks; printing or inspecting output alone is not verification.",
    "The last committed command must be verify, complete with return code 0, and have no timeout or transport failure.",
    "apply_patch is not installed in task containers; use one of payload.tool_capabilities.portable_file_edit_methods.",
    "A verify command must declare independent evidence and be read-only with respect to task state.",
    "After a successful verify command the Runtime locks the verified state; complete immediately without another command.",
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

    def abandon_pending(
        self,
        action_id: UUID,
        *,
        reason: str,
        phase: str,
    ) -> None:
        pending = self._pending
        if pending is None:
            return
        if pending.action.action_id != action_id:
            self._trace_consistent = False
            raise RuntimeError("abandoned Action does not match pending command")
        self._append(
            "pending.abandoned",
            {
                "action_id": str(action_id),
                "reason": reason,
                "phase": phase,
            },
        )
        self._pending = None

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

    def completion_gate_error(self) -> str | None:
        if not self._records:
            return "no committed verification command exists"
        record = self._records[-1]
        if record.intent.command_role is not TerminalCommandRole.VERIFY:
            return "the last committed command is not marked verify"
        if not any(
            item.intent.command_role is TerminalCommandRole.WORK
            for item in self._records[:-1]
        ):
            return "the verification command has no preceding work command"
        result = record.result
        if result.execution_state is not TerminalExecutionState.COMPLETED:
            return "the verification command did not complete with a known result"
        if result.return_code != 0:
            return "the verification command returned a non-zero status"
        if result.timed_out or result.transport_failed:
            return "the verification command timed out or had a transport failure"
        if record.intent.verification is None:
            return "the verification command has no independent evidence contract"
        checkpoint = self._session.verified_checkpoint
        if checkpoint is None or checkpoint.action_id != record.action_id:
            return "the latest successful verification has no locked checkpoint"
        return None

    def record_completion_rejection(self, reason: str) -> None:
        if not reason:
            raise ValueError("completion rejection reason is required")
        self._session = self._session.model_copy(
            update={
                "completion_rejections": self._session.completion_rejections + 1,
                "completion_blocker": reason,
            }
        )
        self._append(
            "agent.completion_rejected",
            {
                "reason": reason,
                "session": self._session.model_dump(mode="json"),
            },
        )

    def record_proposal_rejection(
        self,
        rejection: TerminalProposalRejection,
    ) -> None:
        self._session = self._session.model_copy(
            update={
                "proposal_rejections": self._session.proposal_rejections + 1,
                "proposal_blocker": rejection.message,
                "last_proposal_rejection": rejection,
            }
        )
        self._append(
            "agent.proposal_rejected",
            {
                "rejection": rejection.model_dump(mode="json"),
                "session": self._session.model_dump(mode="json"),
            },
        )

    def mark_complete(self, summary: str) -> None:
        gate_error = self.completion_gate_error()
        if gate_error is not None:
            raise RuntimeError(f"terminal completion rejected: {gate_error}")
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
        verified_checkpoint = session.verified_checkpoint
        if (
            result.execution_state is TerminalExecutionState.COMPLETED
            and result.return_code == 0
            and record.intent.command_role is TerminalCommandRole.VERIFY
            and record.intent.verification is not None
        ):
            verified_checkpoint = TerminalVerifiedCheckpoint(
                action_id=record.action_id,
                call_key=record.intent.call_key,
                command_fingerprint=record.intent.execution_fingerprint,
                verification=record.intent.verification,
                committed_at=record.committed_at,
            )
        failed_verification_attempts = session.failed_verification_attempts
        verification_corrections = session.verification_corrections
        if (
            record.intent.command_role is TerminalCommandRole.VERIFY
            and result.execution_state is TerminalExecutionState.COMPLETED
            and result.return_code != 0
        ):
            failed_verification_attempts += 1
        elif (
            record.intent.command_role is TerminalCommandRole.WORK
            and failed_verification_attempts > verification_corrections
        ):
            verification_corrections += 1
        consecutive_inspections = (
            session.consecutive_inspections + 1
            if record.intent.command_role is TerminalCommandRole.INSPECT
            else 0
        )
        inspection_commands = session.inspection_commands + int(
            record.intent.command_role is TerminalCommandRole.INSPECT
        )
        update: dict[str, object] = {
            "committed_commands": session.committed_commands + 1,
            "timed_out_commands": session.timed_out_commands + int(result.timed_out),
            "in_doubt_commands": session.in_doubt_commands
            + int(result.execution_state is TerminalExecutionState.IN_DOUBT),
            "process_references": tuple(references),
            "completion_blocker": None,
            "proposal_blocker": None,
            "last_proposal_rejection": None,
            "consecutive_inspections": consecutive_inspections,
            "inspection_commands": inspection_commands,
            "verified_checkpoint": verified_checkpoint,
            "failed_verification_attempts": failed_verification_attempts,
            "verification_corrections": verification_corrections,
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
            pending_payload = dict(payload)
            proposal_payload = pending_payload.get("proposal")
            if isinstance(proposal_payload, Mapping):
                proposal_payload = dict(proposal_payload)
                proposal_payload.setdefault(
                    "command_role",
                    TerminalCommandRole.WORK.value,
                )
                pending_payload["proposal"] = proposal_payload
            self._pending = TerminalPendingCommand.model_validate(pending_payload)
        elif kind == "pending.abandoned":
            pending = self._pending
            if (
                pending is None
                or pending.action.action_id != UUID(str(payload["action_id"]))
            ):
                self._trace_consistent = False
                raise RuntimeError(
                    "abandoned transcript Action has no matching pending command"
                )
            self._pending = None
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
        elif kind == "agent.completion_rejected":
            self._session = TerminalSessionSnapshot.model_validate(payload["session"])
        elif kind == "agent.proposal_rejected":
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
        required_structured_output: StructuredOutputLevel = StructuredOutputLevel.JSON_SCHEMA,
        strict_json_schema: bool = False,
        delivery_timeout_seconds: float = 120.0,
        minimum_timeout_seconds: float = 120.0,
        minimum_delivery_timeout_seconds: float = 60.0,
    ) -> None:
        self._gateway = gateway
        self._gateway_policy = gateway_policy
        self._target_id = target_id
        self._max_output_tokens = max_output_tokens
        self._required_structured_output = required_structured_output
        self._strict_json_schema = strict_json_schema
        if delivery_timeout_seconds <= 0:
            raise ValueError("delivery inference timeout must be positive")
        self._delivery_timeout_seconds = delivery_timeout_seconds
        if minimum_timeout_seconds <= 0:
            raise ValueError("minimum inference timeout must be positive")
        self._minimum_timeout_seconds = minimum_timeout_seconds
        if minimum_delivery_timeout_seconds <= 0:
            raise ValueError(
                "minimum delivery inference timeout must be positive"
            )
        self._minimum_delivery_timeout_seconds = (
            minimum_delivery_timeout_seconds
        )
        if delivery_timeout_seconds < minimum_delivery_timeout_seconds:
            raise ValueError(
                "delivery inference timeout cannot be below the minimum "
                "delivery timeout"
            )

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
                    "semantic supplied in the payload. Inspect representative "
                    "authoritative inputs before editing; do not infer record "
                    "semantics solely from filenames. Derive every output field "
                    "from task artifacts instead of fabricating expected data. "
                    "If payload.artifact_first_mode is true, create or modify the "
                    "smallest viable required artifact on this turn whenever the "
                    "task already identifies its output path or format. If a "
                    "capability or authoritative-input probe is necessary, combine "
                    "that bounded probe with artifact-producing work when safe. "
                    "After one inspection-only command the Runtime may reject "
                    "another inspection until work is attempted. "
                    "If an installer reports missing or conflicting dependencies, "
                    "address the complete reported set before verification. "
                    "For source-build or package-install tasks, inspect the "
                    "project's packaging metadata, then run its canonical "
                    "end-to-end install or build command early to reproduce "
                    "the actual failure. Do not make speculative compatibility "
                    "patches before a command demonstrates the need. "
                    "Treat the most recent failed command, especially a failed "
                    "verification, as authoritative: address its exact reported "
                    "errors before unrelated inspection. Use targeted portable "
                    "POSIX tools or feature-detect optional tools; do not assume "
                    "a particular search utility exists. Never mask a diagnostic "
                    "failure with '|| true' when its output determines the next "
                    "action. "
                    "Use command_role=inspect for read-only discovery and capability "
                    "probes. Paths under /tmp/aar-codex-inference-* belong only to "
                    "the host-side inference sandbox and must never appear in a "
                    "task-container command. If payload.recovery_mode is true, stop "
                    "general inspection and issue the single most useful artifact-"
                    "producing or targeted recovery command. If payload.delivery_mode "
                    "is true, do not perform another general inspection. If required "
                    "artifacts are ready, issue the independent verify command now; "
                    "otherwise issue the single most consequential artifact-producing "
                    "or repair command. Keep that command focused enough to preserve "
                    "time for one final verification. Treat "
                    "payload.remaining_wall_clock_seconds as a hard budget. "
                    "If payload.repair_mode is true, issue one work command that "
                    "addresses the complete exact failure set from the latest "
                    "verification; do not inspect, verify, or complete first. If "
                    "payload.verification_due is true, issue the independent verify "
                    "command now and do not make another task-state change. "
                    "Preserve the Runtime-reserved correction and verification slots. "
                    "Install declared build and runtime dependencies in a coherent "
                    "batch when possible instead of discovering them one at a "
                    "time. If packaging conditionally enables native extensions "
                    "only when build dependencies are importable, prepare the "
                    "complete toolchain before the first install and use an "
                    "installation mode that exposes it, such as disabling build "
                    "isolation when justified. Never accept a pure-Python install "
                    "as success when the task requires native extensions. "
                    "A successful compile is not a successful install: "
                    "install the resulting package into the required interpreter "
                    "and verify imports plus the project's official tests or an "
                    "equivalent independent smoke test. "
                    "When several required components can fail independently, "
                    "make verification continue through all checks, report every "
                    "failure, and exit nonzero at the end. When an error identifies "
                    "an exact source file, symbol, or line, combine targeted "
                    "inspection and repair in one bounded command when safe. Batch "
                    "all visible compatibility fixes before rebuilding. "
                    "Keep diagnostic and verification output concise and context-"
                    "efficient: write verbose logs to task-local files and print a "
                    "bounded root-cause summary for every failed check plus the log "
                    "paths, while preserving a nonzero exit status. Do not repeat "
                    "an unchanged failed verification command; change the script or "
                    "task state based on its failure. When a failure identifies one "
                    "member of a deprecated or removed API family, perform a bounded "
                    "search for related members and batch the justified fixes. "
                    "Reserve a command for independent verification after changing "
                    "task artifacts. Verification must use separately derived "
                    "evidence or official tests, not a copy of the production "
                    "algorithm. "
                    "Do not complete unless the last committed command is a "
                    "successful verify command. For execute, include a call_key "
                    "that is absent from payload.used_call_keys; retries need a "
                    "new attempt suffix. Set cwd to null or an absolute POSIX path, "
                    "never '.' or another relative path, and set summary to null. "
                    "timeout_sec must be null or fall within payload.execution_limits; "
                    "never request a value above max_timeout_sec. The only execution "
                    "tool is shell_exec. apply_patch is explicitly unavailable; use "
                    "a portable file edit method listed in payload.tool_capabilities. "
                    "For inspect or work, set verification to null. For verify, provide a "
                    "verification object naming official tests or an independently "
                    "derived check, its evidence sources, affected artifact paths, "
                    "every stable requirement_id from payload.requirements in "
                    "requirement_coverage, and validation methods. Coverage must "
                    "contain IDs only; descriptions authored by you are rejected. "
                    "Use state_policy=read_only. An independent check must cover "
                    "artifact existence, format, semantic correctness, and the "
                    "end-to-end consumer workflow. After a failed verification, "
                    "repair the evidenced defect before verifying again; a check "
                    "alone is insufficient. Official tests must identify the actual "
                    "official test command or exact test path, for example "
                    "'/tests/test_outputs.py' or '/app/test_outputs.py'. "
                    "Verification must not be repeated unchanged after failure. "
                    "Verification must not persistently mutate task state. Disposable "
                    "Git mutations are allowed only under a directory created by "
                    "mktemp -d, with an EXIT cleanup trap, and with every mutating "
                    "git command using git -C beneath that temporary root. After a "
                    "successful verify, complete immediately; do not make another "
                    "change or run another command. "
                    "For complete, include summary and set call_key, command, "
                    "command_role, cwd, env, timeout_sec, and process_reference to "
                    "null, and also set verification to null. Never return rationale; the Runtime supplies its local "
                    "audit reason."
                ),
                "payload": request.model_dump(mode="json"),
            },
            response_schema=_terminal_turn_response_schema(
                strict=self._strict_json_schema,
                max_timeout_sec=request.execution_limits.max_timeout_sec,
            ),
            requirements=InferenceRequirements(
                required_structured_output=self._required_structured_output,
                max_output_tokens=self._max_output_tokens,
            ),
            correlation=InferenceCorrelation(
                run_id=request.run_id,
                task_id=request.task_id,
            ),
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
        if request.remaining_wall_clock_seconds is not None:
            remaining = max(0.0, request.remaining_wall_clock_seconds)
            configured = gateway_policy.budget.max_elapsed_seconds
            action_reserve = min(
                remaining,
                float(request.execution_limits.default_timeout_sec)
                + request.execution_limits.cleanup_grace_seconds,
            )
            future_turn_reserve = 0.0
            if not request.delivery_mode:
                future_action_reserve = (
                    float(request.execution_limits.default_timeout_sec)
                    + request.execution_limits.cleanup_grace_seconds
                )
                future_turn_reserve = (
                    self._minimum_delivery_timeout_seconds
                    + future_action_reserve
                )
            bounded = max(
                0.0,
                remaining - action_reserve - future_turn_reserve,
            )
            if configured is not None:
                bounded = min(bounded, configured)
            if (
                request.delivery_mode
                or request.repair_mode
                or request.verification_due
            ):
                bounded = min(bounded, self._delivery_timeout_seconds)
            gateway_policy = gateway_policy.model_copy(
                update={
                    "budget": gateway_policy.budget.model_copy(
                        update={"max_elapsed_seconds": bounded}
                    )
                }
            )
        elif (
            request.delivery_mode
            or request.repair_mode
            or request.verification_due
        ):
            configured = gateway_policy.budget.max_elapsed_seconds
            bounded = self._delivery_timeout_seconds
            if configured is not None:
                bounded = min(bounded, configured)
            gateway_policy = gateway_policy.model_copy(
                update={
                    "budget": gateway_policy.budget.model_copy(
                        update={"max_elapsed_seconds": bounded}
                    )
                }
            )
        effective_timeout = gateway_policy.budget.max_elapsed_seconds
        minimum_timeout = (
            self._minimum_delivery_timeout_seconds
            if (
                request.delivery_mode
                or request.repair_mode
                or request.verification_due
            )
            else self._minimum_timeout_seconds
        )
        if (
            effective_timeout is not None
            and effective_timeout < minimum_timeout
        ):
            raise InferenceExecutionBudgetError(
                "insufficient wall-clock capacity for another inference: "
                f"available {effective_timeout:.1f} seconds; "
                f"minimum {minimum_timeout:.1f} seconds"
            )
        response = await self._gateway.execute(inference, gateway_policy)
        if response.kind is not ModelResponseKind.OUTPUT:
            raise ValueError("terminal planner does not accept model Tool intents")
        return TerminalTurnProposal(
            draft=TerminalTurnDraft.model_validate(response.output),
            usage=response.usage,
            model_id=response.model_id,
        )


def _terminal_turn_response_schema(
    *, strict: bool = False, max_timeout_sec: int | None = None
) -> dict[str, JsonValue]:
    schema = cast(
        dict[str, JsonValue],
        TerminalTurnDraft.model_json_schema(mode="validation"),
    )
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("rationale", None)
        timeout_schema = properties.get("timeout_sec")
        if max_timeout_sec is not None and isinstance(timeout_schema, dict):
            variants = timeout_schema.get("anyOf")
            if isinstance(variants, list):
                for variant in variants:
                    if (
                        isinstance(variant, dict)
                        and variant.get("type") == "integer"
                    ):
                        variant["maximum"] = max_timeout_sec
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [item for item in required if item != "rationale"]
    schema["allOf"] = [
        {
            "if": {
                "properties": {"decision": {"const": "execute"}},
                "required": ["decision"],
            },
            "then": {
                "required": ["call_key", "command", "command_role"],
                "properties": {"summary": {"type": "null"}},
            },
        },
        {
            "if": {
                "properties": {"decision": {"const": "complete"}},
                "required": ["decision"],
            },
            "then": {
                "required": ["summary"],
                "properties": {
                    "call_key": {"type": "null"},
                    "command": {"type": "null"},
                    "cwd": {"type": "null"},
                    "env": {"type": "null"},
                    "timeout_sec": {"type": "null"},
                    "process_reference": {"type": "null"},
                    "verification": {"type": "null"},
                },
            },
        },
        {
            "if": {
                "properties": {
                    "decision": {"const": "execute"},
                    "command_role": {"const": "verify"},
                },
                "required": ["decision", "command_role"],
            },
            "then": {
                "required": ["verification"],
                "properties": {
                    "verification": {
                        "$ref": "#/$defs/TerminalVerificationContract"
                    }
                },
            },
        },
        {
            "if": {
                "properties": {
                    "decision": {"const": "execute"},
                    "command_role": {"enum": ["inspect", "work"]},
                },
                "required": ["decision", "command_role"],
            },
            "then": {"properties": {"verification": {"type": "null"}}},
        },
    ]
    if strict:
        schema.pop("allOf", None)
        _normalize_strict_terminal_schema(schema)
    return schema


def _normalize_strict_terminal_schema(node: object) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_strict_terminal_schema(item)
        return
    if not isinstance(node, dict):
        return
    if "$ref" in node:
        node.pop("default", None)
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
        node["additionalProperties"] = False
        env_schema = properties.get("env")
        if isinstance(env_schema, dict):
            variants = env_schema.get("anyOf")
            if isinstance(variants, list):
                for variant in variants:
                    if (
                        isinstance(variant, dict)
                        and variant.get("type") == "object"
                    ):
                        variant["properties"] = {}
                        variant["required"] = []
                        variant["additionalProperties"] = False
    for value in tuple(node.values()):
        _normalize_strict_terminal_schema(value)


class _TerminalProposalValidationError(ValueError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        field: str,
        rejected_value: object,
        expected: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.rejected_value = (
            None
            if rejected_value is None
            else str(rejected_value)[:512]
        )
        self.expected = expected


_APPLY_PATCH_TOKEN = re.compile(r"(?<![A-Za-z0-9_.-])apply_patch(?![A-Za-z0-9_.-])")
_HOST_WORKSPACE_PATH = re.compile(
    r"(?:/tmp/aar-codex-(?:inference|agent)-[^\s;&|]*|[A-Za-z]:\\\\[^\s;&|]*)"
)
_MUTATING_VERIFICATION_PATTERNS = (
    re.compile(
        r"\bgit\s+(?:-C\s+\S+\s+)*(?:update-ref|commit|reset|clean|merge|rebase|cherry-pick|stash|restore)\b"
    ),
    re.compile(r"\bgit\s+(?:-C\s+\S+\s+)*checkout\s+[^\n;&|]*?-[bB](?:\s|$)"),
    re.compile(r"\bgit\s+(?:-C\s+\S+\s+)*switch\s+[^\n;&|]*?-[cC](?:\s|$)"),
    re.compile(
        r"\bgit\s+(?:-C\s+\S+\s+)*branch\s+(?:(?:-[fmdDMCc])\s+)?[^\s-][^\s;&|]*"
    ),
    re.compile(
        r"\bgit\s+(?:-C\s+\S+\s+)*tag\s+(?:(?:-[afd])\s+)?[^\s-][^\s;&|]*"
    ),
    re.compile(r"\bgit\s+(?:-C\s+\S+\s+)*worktree\s+(?:add|move|remove|prune)\b"),
)
_TEMP_ROOT_ASSIGNMENT = re.compile(
    r"(?m)^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r"[\"']?\$\(\s*mktemp\s+(?:-d|--directory)\b[^)]*\)[\"']?"
)


def _sanitized_rejected_draft(draft: TerminalTurnDraft) -> TerminalRejectedDraft:
    command = draft.command
    return TerminalRejectedDraft(
        decision=draft.decision,
        call_key=draft.call_key,
        command_preview=(
            f"<redacted command; characters={len(command)}>"
            if command is not None
            else None
        ),
        command_fingerprint=(
            terminal_fingerprint(command) if command is not None else None
        ),
        command_role=draft.command_role,
        cwd=draft.cwd,
        environment_keys=tuple(sorted((draft.env or {}).keys())),
        timeout_sec=draft.timeout_sec,
    )


def _disposable_temp_variables(command: str) -> tuple[str, ...]:
    variables: list[str] = []
    lines = command.splitlines()
    for match in _TEMP_ROOT_ASSIGNMENT.finditer(command):
        name = match.group("name")
        references = (f"${name}", "${" + name + "}")
        has_cleanup = any(
            "trap" in line
            and "EXIT" in line
            and any(reference in line for reference in references)
            for line in lines
        )
        if has_cleanup:
            variables.append(name)
    return tuple(variables)


def _mutation_uses_disposable_git_root(
    mutation: str,
    command: str,
) -> bool:
    for name in _disposable_temp_variables(command):
        roots = (
            f'-C "${name}',
            f"-C '${name}",
            f"-C ${name}",
            '-C "${' + name + '}',
            "-C '${" + name + '}',
            '-C ${' + name + '}',
        )
        if any(root in mutation for root in roots):
            return True
    return False


def _known_verification_mutation(command: str) -> str | None:
    for pattern in _MUTATING_VERIFICATION_PATTERNS:
        match = pattern.search(command)
        if match is not None:
            mutation = match.group(0)
            if _mutation_uses_disposable_git_root(mutation, command):
                continue
            return mutation[:160]
    return None


def _identifies_official_test_source(source: str) -> bool:
    lowered = source.lower()
    if any(marker in lowered for marker in ("/tests", "pytest", "official")):
        return True
    return bool(
        re.search(
            r"(?:^|[/\s])(?:test_[^/\s]+\.py|[^/\s]+_test\.py|"
            r"verify\.(?:sh|py)|eval\.py)"
            r"(?:$|[\s:])",
            lowered,
        )
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
        await self.reconcile_observation(state)
        pending = self._journal.pending()
        if pending is not None:
            return PlanDecision.execute(
                pending.action,
                reason="Resume the original persisted terminal proposal.",
            )
        session = self._journal.snapshot()
        if session.verified_checkpoint is not None:
            gate_error = self._journal.completion_gate_error()
            if gate_error is None:
                summary = (
                    "Independent verification succeeded; the Runtime finalized "
                    "the locked verified checkpoint."
                )
                self._journal.mark_complete(summary)
                return PlanDecision.complete(
                    output={
                        "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                        "agent_complete": True,
                        "summary": summary,
                    },
                    reason=(
                        "Finalize the verified checkpoint without another model "
                        "inference."
                    ),
                )
        budget_error = self._budget_error(session, allow_command_limit=False)
        if budget_error is not None:
            return PlanDecision.fail(error=budget_error)
        request = self._turn_request(state, session)
        try:
            proposal = await self._capability.propose(request)
        except InferenceExecutionBudgetError as exc:
            if request.delivery_mode:
                raise RunBudgetExhaustedError(
                    "active_execution",
                    "terminal delivery inference budget exhausted: " + exc.reason,
                ) from exc
            retry_request = self._turn_request(
                state,
                session,
                force_delivery=True,
            )
            try:
                proposal = await self._capability.propose(retry_request)
            except InferenceExecutionBudgetError as retry_exc:
                raise RunBudgetExhaustedError(
                    "active_execution",
                    "terminal inference recovery budget exhausted: "
                    + retry_exc.reason,
                ) from retry_exc
            request = retry_request
        self._journal.record_usage(proposal)
        session = self._journal.snapshot()
        budget_error = self._budget_error(
            session,
            allow_command_limit=False,
            allow_exact_limit=True,
        )
        if budget_error is not None:
            return PlanDecision.fail(error=budget_error)
        draft = proposal.draft
        if draft.decision is TerminalTurnDecision.COMPLETE:
            assert draft.summary is not None
            gate_error = self._journal.completion_gate_error()
            if gate_error is None:
                self._journal.mark_complete(draft.summary)
                return PlanDecision.complete(
                    output={
                        "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                        "agent_complete": True,
                        "summary": draft.summary,
                    },
                    reason=draft.rationale,
                )
            if (
                session.completion_rejections
                >= self._policy.max_completion_rejections
            ):
                return PlanDecision.fail(
                    error=f"terminal completion rejected: {gate_error}",
                    reason=(
                        "Agent exhausted bounded completion correction attempts."
                    ),
                )
            self._journal.record_completion_rejection(gate_error)
            session = self._journal.snapshot()
            rejection_action = ActionRequest(
                action_id=uuid5(
                    _TERMINAL_ACTION_NAMESPACE,
                    "|".join(
                        (
                            str(state.run_id),
                            str(state.revision),
                            "completion-rejected",
                            str(session.completion_rejections),
                        )
                    ),
                ),
                name=TERMINAL_COMPLETION_REJECTION_ACTION,
                arguments={
                    "reason": gate_error,
                    "completion_rejections": session.completion_rejections,
                },
                repeat_detection_exempt=True,
            )
            return PlanDecision.execute(
                rejection_action,
                reason="Persist completion blocker and return it to the Planner.",
            )
        if session.committed_commands >= self._policy.max_commands:
            return PlanDecision.fail(
                error=(
                    "terminal command budget exhausted "
                    f"({self._policy.max_commands})"
                )
            )
        try:
            intent = self._resolve_intent(draft, session)
            self._validate_intent(
                intent,
                session,
                recovery_mode=request.recovery_mode,
                repair_mode=request.repair_mode,
                verification_due=request.verification_due,
                required_requirements=_task_requirements(
                    state.task.description
                ),
            )
            self._reject_duplicate_or_uncertain_replay(intent)
        except ValueError as exc:
            validation = (
                exc
                if isinstance(exc, _TerminalProposalValidationError)
                else _TerminalProposalValidationError(
                    code="terminal.proposal.invalid",
                    message=str(exc) or exc.__class__.__name__,
                    field="proposal",
                    rejected_value=None,
                    expected="a proposal satisfying the Runtime execution contract",
                )
            )
            return self._proposal_rejection_decision(
                state=state,
                session=session,
                draft=draft,
                validation=validation,
            )
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
            timeout_seconds=float(intent.timeout_sec),
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

    async def reconcile_abandoned_action(
        self,
        state: AgentState,
        action: ActionRequest,
        termination: RunTermination,
    ) -> None:
        del state
        self._journal.abandon_pending(
            action.action_id,
            reason=termination.primary_reason.value,
            phase=termination.phase.value,
        )

    async def reconcile_observation(self, state: AgentState) -> None:
        """Commit Core's persisted observation without starting another turn."""

        self._commit_core_observation(state)

    def _proposal_rejection_decision(
        self,
        *,
        state: AgentState,
        session: TerminalSessionSnapshot,
        draft: TerminalTurnDraft,
        validation: _TerminalProposalValidationError,
    ) -> PlanDecision:
        if session.proposal_rejections >= self._policy.max_proposal_rejections:
            return PlanDecision.fail(
                error=f"invalid terminal proposal: {validation.message}",
                reason="Agent exhausted bounded proposal correction attempts.",
            )
        rejection = TerminalProposalRejection(
            code=validation.code,
            message=validation.message,
            field=validation.field,
            rejected_value=validation.rejected_value,
            expected=validation.expected,
            draft=_sanitized_rejected_draft(draft),
        )
        self._journal.record_proposal_rejection(rejection)
        updated_session = self._journal.snapshot()
        action = ActionRequest(
            action_id=uuid5(
                _TERMINAL_ACTION_NAMESPACE,
                "|".join(
                    (
                        str(state.run_id),
                        str(state.revision),
                        "proposal-rejected",
                        str(updated_session.proposal_rejections),
                        rejection.code,
                    )
                ),
            ),
            name=TERMINAL_PROPOSAL_REJECTION_ACTION,
            arguments={
                "rejection": rejection.model_dump(mode="json"),
                "proposal_rejections": updated_session.proposal_rejections,
            },
            repeat_detection_exempt=True,
        )
        return PlanDecision.execute(
            action,
            reason="Return a correctable proposal rejection to the Planner.",
        )

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
        *,
        force_delivery: bool = False,
    ) -> TerminalTurnRequest:
        requirements = _task_requirements(state.task.description)
        elapsed_seconds = max(
            0.0,
            (utc_now() - session.started_at).total_seconds(),
        )
        remaining_wall_clock_seconds = (
            None
            if self._policy.max_wall_clock_seconds is None
            else max(0.0, self._policy.max_wall_clock_seconds - elapsed_seconds)
        )
        delivery_mode = force_delivery or bool(
            self._policy.max_wall_clock_seconds is not None
            and elapsed_seconds
            >= self._policy.max_wall_clock_seconds
            * self._policy.delivery_mode_fraction
        )
        all_records = self._journal.recent_records(self._policy.max_commands)
        context_record_limit = (
            self._policy.max_delivery_context_records
            if delivery_mode
            else self._policy.max_context_records
        )
        context_output_limit = (
            self._policy.max_delivery_context_output_characters
            if delivery_mode
            else self._policy.max_context_output_characters
        )
        records = all_records[-context_record_limit:]
        used_call_keys = tuple(item.intent.call_key for item in all_records)
        history = tuple(
            item.history_item(output_limit=context_output_limit)
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
        repair_mode = (
            session.failed_verification_attempts
            > session.verification_corrections
        )
        last_record = all_records[-1] if all_records else None
        verification_due = bool(
            session.failed_verification_attempts > 0
            and session.failed_verification_attempts
            == session.verification_corrections
            and last_record is not None
            and last_record.intent.command_role is TerminalCommandRole.WORK
            and last_record.result.command_completed
            and last_record.result.return_code == 0
        )
        artifact_first_mode = not any(
            item.intent.command_role
            in (TerminalCommandRole.WORK, TerminalCommandRole.VERIFY)
            for item in all_records
        )
        artifact_inspection_limit = (
            self._policy.max_artifact_first_inspections
        )
        artifact_recovery = bool(
            artifact_first_mode
            and artifact_inspection_limit is not None
            and session.inspection_commands >= artifact_inspection_limit
        )
        recovery_mode = bool(
            self._recovery_mode(state)
            or delivery_mode
            or repair_mode
            or verification_due
            or artifact_recovery
        )
        return TerminalTurnRequest(
            run_id=state.run_id,
            task_id=state.task.task_id,
            instruction=state.task.description,
            requirements=requirements,
            session=session,
            recent_history=history,
            used_call_keys=used_call_keys,
            execution_limits=TerminalExecutionLimits(
                default_timeout_sec=self._policy.default_timeout_sec,
                max_timeout_sec=self._policy.max_timeout_sec,
                cleanup_grace_seconds=self._policy.cleanup_grace_seconds,
                max_command_characters=self._policy.max_command_characters,
                max_environment_variables=(
                    self._policy.max_environment_variables
                ),
                max_environment_value_characters=(
                    self._policy.max_environment_value_characters
                ),
            ),
            tool_capabilities=TerminalToolCapabilities(),
            remaining_commands=max(
                0,
                self._policy.max_commands - session.committed_commands,
            ),
            remaining_tokens=remaining_tokens,
            remaining_cost_usd=remaining_cost,
            elapsed_seconds=elapsed_seconds,
            remaining_wall_clock_seconds=remaining_wall_clock_seconds,
            delivery_mode=delivery_mode,
            recovery_mode=recovery_mode,
            artifact_first_mode=artifact_first_mode,
            repair_mode=repair_mode,
            verification_due=verification_due,
            execution_semantics=_EXECUTION_SEMANTICS,
        )

    def _resolve_intent(
        self,
        draft: TerminalTurnDraft,
        session: TerminalSessionSnapshot,
    ) -> TerminalCommandIntent:
        assert (
            draft.call_key is not None
            and draft.command is not None
            and draft.command_role is not None
        )
        cwd = _resolve_terminal_cwd(draft.cwd, session.current_cwd)
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
            command_role=draft.command_role,
            process_reference=draft.process_reference,
            verification=draft.verification,
        )

    def _validate_intent(
        self,
        intent: TerminalCommandIntent,
        session: TerminalSessionSnapshot,
        *,
        recovery_mode: bool = False,
        repair_mode: bool = False,
        verification_due: bool = False,
        required_requirements: tuple[TerminalRequirement, ...],
    ) -> None:
        if session.verified_checkpoint is not None:
            raise _TerminalProposalValidationError(
                code="terminal.verified_state.locked",
                message=(
                    "task state is locked after successful verification; "
                    "complete without another command"
                ),
                field="decision",
                rejected_value="execute",
                expected="complete",
            )
        if len(intent.command) > self._policy.max_command_characters:
            raise _TerminalProposalValidationError(
                code="terminal.command.too_long",
                message="command exceeds Runtime character budget",
                field="command",
                rejected_value=len(intent.command),
                expected=f"at most {self._policy.max_command_characters} characters",
            )
        if _APPLY_PATCH_TOKEN.search(intent.command) is not None:
            raise _TerminalProposalValidationError(
                code="terminal.tool.unavailable",
                message=(
                    "apply_patch is unavailable in task containers; use a "
                    "portable file edit method from tool_capabilities"
                ),
                field="command",
                rejected_value="apply_patch",
                expected="shell_exec using an available portable edit method",
            )
        host_path = _HOST_WORKSPACE_PATH.search(
            " ".join(item for item in (intent.cwd, intent.command) if item)
        )
        if host_path is not None:
            raise _TerminalProposalValidationError(
                code="terminal.path.host_workspace_unavailable",
                message="host-side inference workspace paths do not exist in the task container",
                field="command",
                rejected_value=host_path.group(0),
                expected="a task-container path such as /app or the current session cwd",
            )
        if repair_mode and intent.command_role is not TerminalCommandRole.WORK:
            raise _TerminalProposalValidationError(
                code="terminal.verification.repair_required",
                message=(
                    "the latest failed verification requires a targeted work "
                    "correction before another verification"
                ),
                field="command_role",
                rejected_value=intent.command_role.value,
                expected="work",
            )
        if (
            verification_due
            and intent.command_role is not TerminalCommandRole.VERIFY
        ):
            raise _TerminalProposalValidationError(
                code="terminal.verification.due",
                message=(
                    "a successful verification correction must be followed by "
                    "independent verification before more task-state changes"
                ),
                field="command_role",
                rejected_value=intent.command_role.value,
                expected="verify",
            )
        if recovery_mode and intent.command_role is TerminalCommandRole.INSPECT:
            raise _TerminalProposalValidationError(
                code="terminal.recovery.inspect_disallowed",
                message="recovery mode requires an artifact-producing or targeted repair command",
                field="command_role",
                rejected_value=intent.command_role.value,
                expected="work or verify",
            )
        if intent.timeout_sec > self._policy.max_timeout_sec:
            raise _TerminalProposalValidationError(
                code="terminal.timeout.above_maximum",
                message="timeout exceeds Runtime maximum",
                field="timeout_sec",
                rejected_value=intent.timeout_sec,
                expected=f"an integer from 1 through {self._policy.max_timeout_sec}",
            )
        if len(intent.env) > self._policy.max_environment_variables:
            raise _TerminalProposalValidationError(
                code="terminal.environment.too_many_entries",
                message="environment map exceeds Runtime entry budget",
                field="env",
                rejected_value=len(intent.env),
                expected=f"at most {self._policy.max_environment_variables} entries",
            )
        if any(not key or "=" in key or "\x00" in key for key in intent.env):
            raise _TerminalProposalValidationError(
                code="terminal.environment.invalid_name",
                message="environment variable name is invalid",
                field="env",
                rejected_value="<keys only>",
                expected="non-empty names without '=' or NUL",
            )
        if any(
            len(value) > self._policy.max_environment_value_characters
            for value in intent.env.values()
        ):
            raise _TerminalProposalValidationError(
                code="terminal.environment.value_too_long",
                message="environment value exceeds Runtime character budget",
                field="env",
                rejected_value="<redacted>",
                expected=(
                    "each value at most "
                    f"{self._policy.max_environment_value_characters} characters"
                ),
            )
        if intent.command_role is TerminalCommandRole.VERIFY:
            if intent.verification is None:
                raise _TerminalProposalValidationError(
                    code="terminal.verification.contract_missing",
                    message="verify command requires an independent evidence contract",
                    field="verification",
                    rejected_value=None,
                    expected=(
                        "official_tests or independent_check evidence with "
                        "sources, artifact paths, and state_policy=read_only"
                    ),
                )
            if (
                intent.verification.state_policy
                is not TerminalVerificationStatePolicy.READ_ONLY
            ):
                raise _TerminalProposalValidationError(
                    code="terminal.verification.state_policy_unsupported",
                    message="only read-only verification is currently supported",
                    field="verification.state_policy",
                    rejected_value=intent.verification.state_policy.value,
                    expected="read_only",
                )
            required_ids = {
                item.requirement_id for item in required_requirements
            }
            covered_ids = set(intent.verification.requirement_coverage)
            unknown = sorted(covered_ids.difference(required_ids))
            missing = sorted(required_ids.difference(covered_ids))
            if unknown or missing:
                details: list[str] = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if unknown:
                    details.append("unknown " + ", ".join(unknown))
                raise _TerminalProposalValidationError(
                    code="terminal.verification.requirements_incomplete",
                    message=(
                        "verification requirement coverage is incomplete: "
                        + "; ".join(details)
                    ),
                    field="verification.requirement_coverage",
                    rejected_value=", ".join(intent.verification.requirement_coverage),
                    expected="every requirement_id from payload.requirements exactly once",
                )
            mutation = _known_verification_mutation(intent.command)
            if mutation is not None:
                raise _TerminalProposalValidationError(
                    code="terminal.verification.persistent_mutation",
                    message=(
                        "verification command contains a known persistent-state "
                        "mutation"
                    ),
                    field="command",
                    rejected_value=mutation,
                    expected=(
                        "a read-only check using disposable names or an "
                        "independent official test"
                    ),
                )
            if any(
                not posixpath.isabs(path)
                for path in intent.verification.artifact_paths
            ):
                raise _TerminalProposalValidationError(
                    code="terminal.verification.artifact_path_relative",
                    message="verification artifact paths must be absolute POSIX paths",
                    field="verification.artifact_paths",
                    rejected_value="<paths>",
                    expected="absolute task-container paths",
                )
            if (
                intent.verification.evidence_kind.value == "official_tests"
                and not any(
                    _identifies_official_test_source(source)
                    for source in intent.verification.evidence_sources
                )
            ):
                raise _TerminalProposalValidationError(
                    code="terminal.verification.official_source_unidentified",
                    message="official verification must identify the actual official test command or path",
                    field="verification.evidence_sources",
                    rejected_value="<sources>",
                    expected="an official test path or command",
                )
        elif intent.verification is not None:
            raise _TerminalProposalValidationError(
                code="terminal.verification.unexpected",
                message="work command must set verification to null",
                field="verification",
                rejected_value="non-null",
                expected="null",
            )

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

    def _recovery_mode(self, state: AgentState) -> bool:
        inspection_maximum = self._policy.max_consecutive_inspections
        if (
            inspection_maximum is not None
            and self._journal.snapshot().consecutive_inspections
            >= inspection_maximum
        ):
            return True
        total_inspection_maximum = self._policy.max_total_inspections
        if (
            total_inspection_maximum is not None
            and self._journal.snapshot().inspection_commands
            >= total_inspection_maximum
        ):
            return True
        maximum = self._policy.max_no_progress_steps
        if maximum is None:
            return False
        return state.control.no_progress_steps >= max(1, maximum - 1)

    def _budget_error(
        self,
        session: TerminalSessionSnapshot,
        *,
        allow_command_limit: bool = True,
        allow_exact_limit: bool = False,
    ) -> str | None:
        if allow_command_limit and session.committed_commands >= self._policy.max_commands:
            return f"terminal command budget exhausted ({self._policy.max_commands})"
        if (
            self._policy.max_total_tokens is not None
            and (
                session.total_tokens > self._policy.max_total_tokens
                if allow_exact_limit
                else session.total_tokens >= self._policy.max_total_tokens
            )
        ):
            return "terminal model token budget exhausted"
        if (
            self._policy.max_cost_usd is not None
            and (
                session.cost_usd > self._policy.max_cost_usd
                if allow_exact_limit
                else session.cost_usd >= self._policy.max_cost_usd
            )
        ):
            return "terminal model cost budget exhausted"
        return None


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z0-9`/])")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+|[0-9]+[.)]\s+)")


def _task_requirements(instruction: str) -> tuple[TerminalRequirement, ...]:
    """Derive stable requirement IDs without delegating coverage to the model."""

    units: list[str] = []
    paragraph_lines: list[str] = []

    def flush() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(paragraph_lines).strip()
        paragraph_lines.clear()
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            normalized = " ".join(sentence.split())
            if normalized:
                units.append(normalized[:2000])

    for raw_line in instruction.replace("\r\n", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            flush()
            continue
        if _LIST_ITEM.match(raw_line):
            flush()
            paragraph_lines.append(_LIST_ITEM.sub("", raw_line, count=1))
            flush()
            continue
        if stripped.startswith("#"):
            flush()
            continue
        paragraph_lines.append(stripped)
    flush()
    if not units:
        units.append("Complete the task exactly as instructed.")
    return tuple(
        TerminalRequirement(
            requirement_id=f"req-{index:03d}",
            description=unit,
        )
        for index, unit in enumerate(units, start=1)
    )


def _resolve_terminal_cwd(
    proposed_cwd: str | None,
    current_cwd: str | None,
) -> str | None:
    if proposed_cwd in (None, ".", "./"):
        candidate = current_cwd
    else:
        candidate = proposed_cwd
    if candidate in (None, ".", "./"):
        return None
    if candidate.startswith("/"):
        return posixpath.normpath(candidate)
    if current_cwd is not None and current_cwd.startswith("/"):
        resolved = posixpath.normpath(posixpath.join(current_cwd, candidate))
        if resolved.startswith("/"):
            return resolved
    raise ValueError(
        "cwd must be an absolute POSIX path when no absolute current cwd "
        "is available"
    )


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
