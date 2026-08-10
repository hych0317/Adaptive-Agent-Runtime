"""Strict non-AAR sequential baseline for Terminal-Bench.

The baseline deliberately reuses the AAR model proposal protocol while
executing accepted drafts directly against the trial environment.  It exists
to isolate the effect of Runtime architecture from model, prompt, schema,
budget, and Harbor environment choices.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

from adaptive_agent_runtime.llm import HttpxJSONTransport, InferenceUsage

from applications.terminal_bench.composition import (
    _CapturingJSONTransport,
    TerminalModelConfig,
    build_terminal_model_capability,
)
from applications.terminal_bench.contracts import (
    TerminalEnvironment,
    TerminalTurnProposalCapability,
)
from applications.terminal_bench.models import (
    TerminalCommandRole,
    TerminalExecResult,
    TerminalExecutionLimits,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalHistoryItem,
    TerminalProcessReference,
    TerminalSessionSnapshot,
    TerminalToolCapabilities,
    TerminalTrialSummary,
    TerminalTurnDecision,
    TerminalTurnDraft,
    TerminalTurnRequest,
    utc_now,
)
from applications.terminal_bench.planner import (
    _EXECUTION_SEMANTICS,
    _task_requirements,
)


PLAIN_SEQUENTIAL_PROFILE = "Plain Sequential Agent (strict architecture ablation)"

_PLAIN_TRIAL_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/terminal-sequential/plain-ablation",
)


@dataclass(frozen=True)
class PlainRunArtifacts:
    summary: TerminalTrialSummary


class PlainSequentialApplication:
    """Run model drafts sequentially without AAR execution architecture."""

    def __init__(
        self,
        *,
        trial_id: str,
        logs_dir: str | Path,
        environment: TerminalEnvironment,
        capability: TerminalTurnProposalCapability,
        policy: TerminalExecutionPolicy | None = None,
    ) -> None:
        if not trial_id:
            raise ValueError("trial_id is required")
        self.trial_id = trial_id
        self._logs_dir = Path(logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._environment = environment
        self._capability = capability
        self._policy = policy or TerminalExecutionPolicy()
        self._session = TerminalSessionSnapshot(trial_id=trial_id)
        self._history: list[TerminalHistoryItem] = []
        self._transcript_path = self._logs_dir / "plain-transcript.jsonl"
        self._ran = False

    async def run(self, instruction: str) -> PlainRunArtifacts:
        if self._ran:
            raise RuntimeError("PlainSequentialApplication is single-use")
        self._ran = True
        started_at = utc_now()
        started_clock = monotonic()
        run_id = uuid5(_PLAIN_TRIAL_NAMESPACE, f"run|{self.trial_id}")
        task_id = uuid5(_PLAIN_TRIAL_NAMESPACE, f"task|{self.trial_id}")
        status = "running"
        agent_complete = False
        final_output: dict[str, object] = {
            "profile": PLAIN_SEQUENTIAL_PROFILE,
            "agent_complete": False,
        }
        self._append(
            "plain.run.started",
            {
                "trial_id": self.trial_id,
                "run_id": str(run_id),
                "task_id": str(task_id),
            },
        )

        while True:
            budget_error = self._budget_error(started_clock)
            if budget_error is not None:
                status = budget_error
                final_output["error"] = budget_error
                break
            request = self._turn_request(
                instruction=instruction,
                run_id=run_id,
                task_id=task_id,
                started_clock=started_clock,
            )
            proposal = await self._capability.propose(request)
            self._record_usage(proposal.usage)
            self._append(
                "plain.inference.completed",
                {
                    "draft": proposal.draft.model_dump(mode="json"),
                    "usage": proposal.usage.model_dump(mode="json"),
                    "model_id": proposal.model_id,
                },
            )
            post_inference_error = self._budget_error(
                started_clock,
                allow_exact_limit=True,
            )
            if post_inference_error is not None:
                status = post_inference_error
                final_output["error"] = post_inference_error
                break

            draft = proposal.draft
            if draft.decision is TerminalTurnDecision.COMPLETE:
                agent_complete = True
                status = "completed"
                final_output = {
                    "profile": PLAIN_SEQUENTIAL_PROFILE,
                    "agent_complete": True,
                    "summary": cast(str, draft.summary),
                }
                self._append(
                    "plain.agent.completed",
                    {"summary": draft.summary},
                )
                break
            if self._session.committed_commands >= self._policy.max_commands:
                status = "command_budget_exhausted"
                final_output["error"] = status
                break

            resolved, protocol_error = self._resolve_execution(draft)
            if protocol_error is not None:
                status = "invalid_proposal"
                final_output["error"] = protocol_error
                self._append(
                    "plain.proposal.invalid",
                    {"error": protocol_error},
                )
                break
            assert resolved is not None
            cwd, environment, timeout_sec = resolved
            result = await self._environment.exec(
                cast(str, draft.command),
                cwd=cwd,
                env=environment,
                timeout_sec=timeout_sec,
            )
            self._commit_execution(
                draft=draft,
                result=result,
                cwd=cwd,
                environment=environment,
                timeout_sec=timeout_sec,
                run_id=run_id,
            )

        completed_at = utc_now()
        summary = TerminalTrialSummary(
            profile=PLAIN_SEQUENTIAL_PROFILE,
            trial_id=self.trial_id,
            run_id=run_id,
            task_id=task_id,
            agent_complete=agent_complete,
            runtime_status=status,
            final_output=final_output,
            command_count=self._session.committed_commands,
            denial_count=0,
            timeout_count=self._session.timed_out_commands,
            in_doubt_count=self._session.in_doubt_commands,
            input_tokens=self._session.input_tokens,
            output_tokens=self._session.output_tokens,
            total_tokens=self._session.total_tokens,
            cost_usd=self._session.cost_usd,
            latency_ms=max(0, int((monotonic() - started_clock) * 1000)),
            trace_consistent=True,
            started_at=started_at,
            completed_at=completed_at,
        )
        self._write_summary(summary)
        return PlainRunArtifacts(summary=summary)

    def close(self) -> None:
        """Match the AAR application boundary; this baseline owns no store."""

    def _turn_request(
        self,
        *,
        instruction: str,
        run_id: UUID,
        task_id: UUID,
        started_clock: float,
    ) -> TerminalTurnRequest:
        elapsed_seconds = max(0.0, monotonic() - started_clock)
        remaining_wall_clock_seconds = (
            None
            if self._policy.max_wall_clock_seconds is None
            else max(
                0.0,
                self._policy.max_wall_clock_seconds - elapsed_seconds,
            )
        )
        delivery_mode = bool(
            self._policy.max_wall_clock_seconds is not None
            and elapsed_seconds
            >= self._policy.max_wall_clock_seconds
            * self._policy.delivery_mode_fraction
        )
        history_limit = (
            self._policy.max_delivery_context_records
            if delivery_mode
            else self._policy.max_context_records
        )
        output_limit = (
            self._policy.max_delivery_context_output_characters
            if delivery_mode
            else self._policy.max_context_output_characters
        )
        history = tuple(
            self._bounded_history_item(item, output_limit)
            for item in self._history[-history_limit:]
        )
        artifact_first_mode = not any(
            item.command_role in (TerminalCommandRole.WORK, TerminalCommandRole.VERIFY)
            for item in self._history
        )
        return TerminalTurnRequest(
            run_id=run_id,
            task_id=task_id,
            instruction=instruction,
            # Keep the exact AAR request profile and prompt contract on purpose.
            # The Harbor metadata and local summary identify this as Plain.
            requirements=_task_requirements(instruction),
            session=self._session,
            recent_history=history,
            used_call_keys=tuple(item.call_key for item in self._history),
            execution_limits=TerminalExecutionLimits(
                default_timeout_sec=self._policy.default_timeout_sec,
                max_timeout_sec=self._policy.max_timeout_sec,
                cleanup_grace_seconds=self._policy.cleanup_grace_seconds,
                max_command_characters=self._policy.max_command_characters,
                max_environment_variables=self._policy.max_environment_variables,
                max_environment_value_characters=(
                    self._policy.max_environment_value_characters
                ),
            ),
            tool_capabilities=TerminalToolCapabilities(),
            remaining_commands=max(
                0,
                self._policy.max_commands - self._session.committed_commands,
            ),
            remaining_tokens=(
                None
                if self._policy.max_total_tokens is None
                else max(
                    0,
                    self._policy.max_total_tokens - self._session.total_tokens,
                )
            ),
            remaining_cost_usd=(
                None
                if self._policy.max_cost_usd is None
                else max(0.0, self._policy.max_cost_usd - self._session.cost_usd)
            ),
            elapsed_seconds=elapsed_seconds,
            remaining_wall_clock_seconds=remaining_wall_clock_seconds,
            delivery_mode=delivery_mode,
            recovery_mode=False,
            artifact_first_mode=artifact_first_mode,
            repair_mode=False,
            verification_due=False,
            execution_semantics=_EXECUTION_SEMANTICS,
        )

    def _resolve_execution(
        self,
        draft: TerminalTurnDraft,
    ) -> tuple[tuple[str | None, dict[str, str], int] | None, str | None]:
        assert draft.command is not None
        if len(draft.command) > self._policy.max_command_characters:
            return None, "command exceeds the shared character budget"
        try:
            cwd = _resolve_cwd(draft.cwd, self._session.current_cwd)
        except ValueError as exc:
            return None, str(exc)
        environment = (
            dict(draft.env)
            if draft.env is not None
            else dict(self._session.environment)
        )
        if len(environment) > self._policy.max_environment_variables:
            return None, "environment exceeds the shared entry budget"
        if any(not key or "=" in key or "\x00" in key for key in environment):
            return None, "environment contains an invalid variable name"
        if any(
            len(value) > self._policy.max_environment_value_characters
            for value in environment.values()
        ):
            return None, "environment value exceeds the shared character budget"
        timeout_sec = draft.timeout_sec or self._policy.default_timeout_sec
        if timeout_sec > self._policy.max_timeout_sec:
            return None, "timeout exceeds the shared maximum"
        return (cwd, environment, timeout_sec), None

    def _commit_execution(
        self,
        *,
        draft: TerminalTurnDraft,
        result: TerminalExecResult,
        cwd: str | None,
        environment: dict[str, str],
        timeout_sec: int,
        run_id: UUID,
    ) -> None:
        assert draft.call_key is not None
        assert draft.command is not None
        assert draft.command_role is not None
        action_id = uuid5(
            _PLAIN_TRIAL_NAMESPACE,
            f"{run_id}|command|{len(self._history) + 1}",
        )
        item = TerminalHistoryItem(
            action_id=action_id,
            call_key=draft.call_key,
            command=draft.command,
            command_role=draft.command_role,
            cwd=cwd,
            environment_keys=tuple(sorted(environment)),
            timeout_sec=timeout_sec,
            verification=draft.verification,
            execution_state=result.execution_state,
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            transport_failed=result.transport_failed,
        )
        self._history.append(item)
        references = list(self._session.process_references)
        if (
            result.execution_state is TerminalExecutionState.COMPLETED
            and result.return_code == 0
            and draft.process_reference is not None
        ):
            references = [
                reference
                for reference in references
                if reference.reference_id != draft.process_reference.reference_id
            ]
            references.append(cast(TerminalProcessReference, draft.process_reference))
        update: dict[str, object] = {
            "committed_commands": self._session.committed_commands + 1,
            "timed_out_commands": self._session.timed_out_commands
            + int(result.timed_out),
            "in_doubt_commands": self._session.in_doubt_commands
            + int(result.execution_state is TerminalExecutionState.IN_DOUBT),
            "process_references": tuple(references),
            "consecutive_inspections": (
                self._session.consecutive_inspections + 1
                if draft.command_role is TerminalCommandRole.INSPECT
                else 0
            ),
            "inspection_commands": self._session.inspection_commands
            + int(draft.command_role is TerminalCommandRole.INSPECT),
        }
        if result.execution_state is TerminalExecutionState.COMPLETED:
            update["current_cwd"] = cwd
            update["environment"] = dict(environment)
        self._session = self._session.model_copy(update=update)
        self._append(
            "plain.command.executed",
            {
                "action_id": str(action_id),
                "call_key": draft.call_key,
                "command": draft.command,
                "command_role": draft.command_role.value,
                "cwd": cwd,
                "environment_keys": sorted(environment),
                "timeout_sec": timeout_sec,
                "result": result.model_dump(mode="json"),
            },
        )

    def _record_usage(self, usage: InferenceUsage) -> None:
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

    def _budget_error(
        self,
        started_clock: float,
        *,
        allow_exact_limit: bool = False,
    ) -> str | None:
        total_tokens = self._session.total_tokens
        if self._policy.max_total_tokens is not None and (
            total_tokens > self._policy.max_total_tokens
            if allow_exact_limit
            else total_tokens >= self._policy.max_total_tokens
        ):
            return "token_budget_exhausted"
        cost = self._session.cost_usd
        if self._policy.max_cost_usd is not None and (
            cost > self._policy.max_cost_usd
            if allow_exact_limit
            else cost >= self._policy.max_cost_usd
        ):
            return "cost_budget_exhausted"
        if (
            self._policy.max_wall_clock_seconds is not None
            and monotonic() - started_clock >= self._policy.max_wall_clock_seconds
        ):
            return "wall_clock_budget_exhausted"
        return None

    @staticmethod
    def _bounded_history_item(
        item: TerminalHistoryItem,
        limit: int,
    ) -> TerminalHistoryItem:
        def bounded(value: str) -> str:
            if len(value) <= limit:
                return value
            half = max(1, limit // 2)
            return value[:half] + "\n...[context truncated]...\n" + value[-half:]

        return item.model_copy(
            update={"stdout": bounded(item.stdout), "stderr": bounded(item.stderr)}
        )

    def _append(self, kind: str, payload: object) -> None:
        event = {
            "kind": kind,
            "occurred_at": utc_now().isoformat(),
            "payload": payload,
        }
        with self._transcript_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            stream.write("\n")

    def _write_summary(self, summary: TerminalTrialSummary) -> None:
        path = self._logs_dir / "plain-summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)


def build_plain_sequential_application(
    *,
    trial_id: str,
    logs_dir: str | Path,
    environment: TerminalEnvironment,
    proposal_capability: TerminalTurnProposalCapability | None = None,
    model_config: TerminalModelConfig | None = None,
    policy: TerminalExecutionPolicy | None = None,
) -> PlainSequentialApplication:
    """Compose a direct-execution baseline with the shared model protocol."""

    root = Path(logs_dir)
    root.mkdir(parents=True, exist_ok=True)
    capability = proposal_capability
    if capability is None:
        if model_config is None:
            raise ValueError("model_config or proposal_capability is required")
        capability, _ = build_terminal_model_capability(
            model_config,
            transport=_CapturingJSONTransport(
                HttpxJSONTransport(),
                root / "plain-model-responses.jsonl",
            ),
        )
    return PlainSequentialApplication(
        trial_id=trial_id,
        logs_dir=root,
        environment=environment,
        capability=capability,
        policy=policy,
    )


def _resolve_cwd(proposed_cwd: str | None, current_cwd: str | None) -> str | None:
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
    raise ValueError("cwd must be null or an absolute POSIX path")


__all__ = [
    "PLAIN_SEQUENTIAL_PROFILE",
    "PlainRunArtifacts",
    "PlainSequentialApplication",
    "build_plain_sequential_application",
]
