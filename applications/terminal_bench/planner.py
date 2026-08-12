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
from applications.terminal_bench.deadline import (
    TerminalDeadlineBudgetProfile,
    TerminalDeadlineSequence,
    TerminalDeadlineSlots,
    TerminalInferenceTiming,
    allocate_deadline_slots,
    allocate_profiled_terminal_deadline_sequence,
    allocate_terminal_deadline_sequence,
)
from applications.terminal_bench.models import (
    AAR_TERMINAL_SEQUENTIAL_PROFILE,
    TERMINAL_COMMAND_ACTION,
    TERMINAL_COMPLETION_REJECTION_ACTION,
    TERMINAL_PROPOSAL_REJECTION_ACTION,
    TerminalCommandRole,
    TerminalCompletionDisposition,
    TerminalEvidenceAssurance,
    TerminalEvidenceProvenance,
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionLimits,
    TerminalExecutionState,
    TerminalPendingCommand,
    TerminalTimeoutCapReason,
    TerminalProposalRejection,
    TerminalProcessReference,
    TerminalReconciliationReceipt,
    TerminalRequirement,
    TerminalRequirementKind,
    TerminalRequirementLedgerEntry,
    TerminalRequirementState,
    TerminalRejectedDraft,
    TerminalSessionSnapshot,
    TerminalReconciliationState,
    TerminalLedgerProjectionEntry,
    TerminalTaskContract,
    TerminalTaskLedger,
    TerminalTrialSummary,
    TerminalTurnDecision,
    TerminalTurnDraft,
    TerminalTurnProposal,
    TerminalTurnRequest,
    TerminalToolCapabilities,
    TerminalVerifiedCheckpoint,
    TerminalVerificationReceipt,
    TerminalPerformanceProtocol,
    TerminalVerificationStatePolicy,
    terminal_fingerprint,
    utc_now,
)
from applications.terminal_bench.progress import terminal_failure_signatures


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
    "IN_DOUBT means the command may have started; never repeat it or mutate related state until a read-only reconciliation command proves the process stopped and artifacts are known.",
    "Start background services in detached form and supply PID file, log path, and status command.",
    "Label read-only discovery and capability probes as inspect; inspect success is not task completion.",
    "Host-side Codex inference workspace paths never exist inside the task container.",
    "Label commands that create or change task artifacts as work.",
    "Before complete, run an independent verify command whose exit status encodes the task checks; printing or inspecting output alone is not verification.",
    "The last committed command must be verify, complete with return code 0, and have no timeout or transport failure.",
    "apply_patch is not installed in task containers; use one of payload.tool_capabilities.portable_file_edit_methods.",
    "A verify command must declare independent evidence and be read-only with respect to task state.",
    "Performance verification must use a fresh process with cold, unique inputs and must not warm or cache the measured inputs before timing.",
    "After a successful verify command the Runtime locks the verified state; complete immediately without another command.",
    "No tmux, interactive terminal, verifier API, oracle API, sidecar execution, or host access is available.",
)


def _task_contract_ledger_error(
    contract: TerminalTaskContract,
    ledger: TerminalTaskLedger | None,
) -> str | None:
    if ledger is None:
        return "task contract has no requirement ledger"
    if ledger.contract_version != contract.contract_version:
        return "task contract and ledger versions do not match"
    if ledger.contract_fingerprint != contract.contract_fingerprint:
        return "task contract and ledger fingerprints do not match"
    contract_ids = tuple(
        item.requirement_id for item in contract.requirements
    )
    ledger_ids = tuple(item.requirement_id for item in ledger.entries)
    if ledger_ids != contract_ids:
        return "task contract and ledger requirement IDs do not match"
    return None


def _current_generation_has_successful_work(
    session: TerminalSessionSnapshot,
) -> bool:
    return bool(
        session.successful_work_generation is not None
        and session.successful_work_generation == session.task_generation
    )


def _generation_state_is_known(session: TerminalSessionSnapshot) -> bool:
    if (
        session.known_state_generation is not None
        and session.known_state_generation == session.task_generation
    ):
        return True
    if _current_generation_has_successful_work(session):
        return True
    receipt = session.latest_reconciliation_receipt
    return bool(
        session.reconciliation_state
        is TerminalReconciliationState.STABLE_UNVERIFIED
        and receipt is not None
        and receipt.task_generation == session.task_generation
    )


def _verification_assurance(
    verification: object,
    result: TerminalExecResult | None = None,
    *,
    contract_complete: bool = True,
) -> TerminalEvidenceAssurance:
    """Trust only provenance and artifact evidence observed by Runtime."""

    from applications.terminal_bench.models import TerminalVerificationContract

    if not isinstance(verification, TerminalVerificationContract):
        return TerminalEvidenceAssurance.NONE
    runtime_evidence = result.runtime_verification if result is not None else None
    if (
        not contract_complete
        or runtime_evidence is None
        or not runtime_evidence.provenance_verified
        or runtime_evidence.requested_provenance
        is not verification.evidence_provenance
        or len(runtime_evidence.artifact_fingerprints)
        != len(verification.artifact_paths)
    ):
        return TerminalEvidenceAssurance.SELF_CHECKED
    provenance = verification.evidence_provenance
    if (
        provenance is TerminalEvidenceProvenance.TASK_PROVIDED
        and verification.evidence_kind.value == "official_tests"
    ):
        return TerminalEvidenceAssurance.TRUSTED
    if (
        provenance is TerminalEvidenceProvenance.EXTERNAL_STANDARD
        and verification.evidence_kind.value == "independent_check"
    ):
        return TerminalEvidenceAssurance.TRUSTED
    return TerminalEvidenceAssurance.SELF_CHECKED


def _runtime_evidence_provenance(
    verification: object,
    result: TerminalExecResult,
) -> TerminalEvidenceProvenance:
    from applications.terminal_bench.models import TerminalVerificationContract

    if not isinstance(verification, TerminalVerificationContract):
        return TerminalEvidenceProvenance.RUNTIME_OBSERVED
    runtime_evidence = result.runtime_verification
    if (
        runtime_evidence is not None
        and runtime_evidence.provenance_verified
        and runtime_evidence.requested_provenance
        is verification.evidence_provenance
    ):
        return runtime_evidence.requested_provenance
    if verification.evidence_provenance is TerminalEvidenceProvenance.AGENT_GENERATED:
        return TerminalEvidenceProvenance.AGENT_GENERATED
    return TerminalEvidenceProvenance.RUNTIME_OBSERVED


def _runtime_artifact_fingerprints(
    result: TerminalExecResult,
) -> tuple[str, ...]:
    evidence = result.runtime_verification
    return evidence.artifact_fingerprints if evidence is not None else ()


def _current_generation_is_verifiable(
    session: TerminalSessionSnapshot,
) -> bool:
    return _generation_state_is_known(session)


def _ledger_projection(
    session: TerminalSessionSnapshot,
) -> tuple[TerminalLedgerProjectionEntry, ...]:
    ledger = session.task_ledger
    if ledger is None:
        return ()
    latest_action = (
        session.latest_verification_receipt.action_id
        if session.latest_verification_receipt is not None
        else None
    )
    return tuple(
        TerminalLedgerProjectionEntry(
            requirement_id=entry.requirement_id,
            state=entry.state,
            assurance=entry.assurance,
            evidence_provenance=entry.evidence_provenance,
            result_fingerprint=entry.latest_result_fingerprint,
            changed_in_generation=entry.evidence_generation,
        )
        for entry in ledger.entries
        if entry.state is not TerminalRequirementState.SATISFIED
        or entry.assurance is not TerminalEvidenceAssurance.TRUSTED
        or entry.latest_evidence_action_id == latest_action
    )


def _legacy_generation_state(
    records: tuple[TerminalCommandRecord, ...],
) -> tuple[int, int | None, int | None]:
    """Derive generation state only while migrating a pre-contract trace."""

    generation = 0
    successful_generation: int | None = None
    known_generation: int | None = 0
    for record in records:
        if record.intent.command_role is not TerminalCommandRole.WORK:
            continue
        if record.result.execution_state is TerminalExecutionState.FAILED_TO_START:
            continue
        generation += 1
        successful_generation = generation if record.result.succeeded else None
        known_generation = generation if record.result.settled else None
    return generation, successful_generation, known_generation


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
        self._task_contract: TerminalTaskContract | None = None
        self._trace_consistent = True
        self._trace_error: str | None = None
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

    def trace_error(self) -> str | None:
        return self._trace_error

    def mark_trace_inconsistent(self, reason: str) -> None:
        if not reason:
            raise ValueError("trace inconsistency reason is required")
        self._trace_consistent = False
        if self._trace_error is None:
            self._trace_error = reason
            self._append("trace.inconsistent", {"reason": reason})

    def snapshot(self) -> TerminalSessionSnapshot:
        return self._session

    def task_contract(self) -> TerminalTaskContract | None:
        return self._task_contract

    def bind_task_contract(
        self,
        requirements: tuple[TerminalRequirement, ...],
        *,
        coverage_complete: bool = True,
        unmapped_fragments: tuple[str, ...] = (),
    ) -> None:
        """Persist one immutable, Runtime-derived requirement contract."""

        contract_fingerprint = terminal_fingerprint(
            {
                "contract_version": "1",
                "requirements": [
                    item.model_dump(mode="json") for item in requirements
                ],
            }
        )
        contract = TerminalTaskContract(
            contract_fingerprint=contract_fingerprint,
            requirements=requirements,
        )
        if self._task_contract is not None:
            ledger_error = _task_contract_ledger_error(
                self._task_contract,
                self._session.task_ledger,
            )
            if (
                self._task_contract != contract
                or ledger_error is not None
                or self._session.contract_coverage_complete
                is not coverage_complete
                or self._session.contract_unmapped_fragments
                != unmapped_fragments
            ):
                self._trace_consistent = False
                raise RuntimeError(
                    "terminal task contract changed within one trial"
                )
            return
        if self._session.task_ledger is not None:
            self._trace_consistent = False
            raise RuntimeError("terminal task ledger has no persisted contract")
        (
            generation,
            successful_work_generation,
            known_state_generation,
        ) = _legacy_generation_state(tuple(self._records))
        ledger = TerminalTaskLedger(
            contract_version=contract.contract_version,
            contract_fingerprint=contract_fingerprint,
            generation=generation,
            entries=tuple(
                TerminalRequirementLedgerEntry(
                    requirement_id=item.requirement_id,
                )
                for item in requirements
            ),
        )
        verification_receipt: TerminalVerificationReceipt | None = None
        verified_checkpoint = self._session.verified_checkpoint
        migration_error: str | None = None
        if (
            verified_checkpoint is not None
            and verified_checkpoint.receipt is None
        ):
            record = self._records[-1] if self._records else None
            result = record.result if record is not None else None
            verification = (
                record.intent.verification if record is not None else None
            )
            required_ids = {item.requirement_id for item in ledger.entries}
            migratable = bool(
                record is not None
                and result is not None
                and record.action_id == verified_checkpoint.action_id
                and record.intent.command_role is TerminalCommandRole.VERIFY
                and verification is not None
                and record.intent.execution_fingerprint
                == verified_checkpoint.command_fingerprint
                and terminal_fingerprint(verification)
                == terminal_fingerprint(verified_checkpoint.verification)
                and set(verification.requirement_coverage) == required_ids
                and result.succeeded
                and successful_work_generation == generation
                and _verification_assurance(
                    verification,
                    result,
                    contract_complete=coverage_complete,
                )
                is TerminalEvidenceAssurance.TRUSTED
            )
            if migratable:
                assert record is not None
                assert result is not None
                assert verification is not None
                result_fingerprint = terminal_fingerprint(result)
                verification_receipt = TerminalVerificationReceipt(
                    action_id=record.action_id,
                    task_generation=generation,
                    task_contract_fingerprint=contract_fingerprint,
                    verification_contract_fingerprint=(
                        terminal_fingerprint(verification)
                    ),
                    command_fingerprint=record.intent.execution_fingerprint,
                    result_fingerprint=result_fingerprint,
                    covered_requirement_ids=(
                        verification.requirement_coverage
                    ),
                    evidence_provenance=_runtime_evidence_provenance(
                        verification, result
                    ),
                    assurance=TerminalEvidenceAssurance.TRUSTED,
                    artifact_fingerprints=_runtime_artifact_fingerprints(result),
                    execution_state=result.execution_state,
                    return_code=result.return_code,
                    timed_out=result.timed_out,
                    transport_failed=result.transport_failed,
                    passed=True,
                    observed_at=record.committed_at,
                )
                ledger = ledger.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(
                                update={
                                    "state": TerminalRequirementState.SATISFIED,
                                    "assurance": TerminalEvidenceAssurance.TRUSTED,
                                    "evidence_provenance": (
                                        _runtime_evidence_provenance(verification, result)
                                    ),
                                    "artifact_fingerprints": (
                                        _runtime_artifact_fingerprints(result)
                                    ),
                                    "latest_evidence_action_id": record.action_id,
                                    "latest_result_fingerprint": result_fingerprint,
                                    "evidence_generation": generation,
                                }
                            )
                            for entry in ledger.entries
                        )
                    }
                )
                verified_checkpoint = verified_checkpoint.model_copy(
                    update={
                        "task_generation": generation,
                        "receipt": verification_receipt,
                    }
                )
            else:
                migration_error = (
                    "legacy verified checkpoint cannot be migrated to the "
                    "current task contract"
                )
        self._task_contract = contract
        self._session = self._session.model_copy(
            update={
                "task_generation": generation,
                "known_state_generation": known_state_generation,
                "successful_work_generation": successful_work_generation,
                "task_ledger": ledger,
                "latest_verification_receipt": verification_receipt,
                "verified_checkpoint": verified_checkpoint,
                "contract_coverage_complete": coverage_complete,
                "contract_unmapped_fragments": unmapped_fragments,
            }
        )
        self._append(
            "task.contract.bound",
            {
                "contract": contract.model_dump(mode="json"),
                "session": self._session.model_dump(mode="json"),
            },
        )
        if migration_error is not None:
            self.mark_trace_inconsistent(migration_error)

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
        if not self._session.contract_coverage_complete:
            return (
                "the Runtime task contract has unmapped critical requirements: "
                + "; ".join(self._session.contract_unmapped_fragments[:3])
            )
        if not self._records:
            return "no committed verification command exists"
        record = self._records[-1]
        if record.intent.command_role is not TerminalCommandRole.VERIFY:
            return "the last committed command is not marked verify"
        if not _current_generation_is_verifiable(self._session):
            return (
                "the verification command has no valid successful work in "
                "the current task generation"
            )
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
        if checkpoint.task_generation != self._session.task_generation:
            return "the locked checkpoint belongs to a stale task generation"
        ledger = self._session.task_ledger
        if ledger is not None:
            contract = self._task_contract
            if contract is None:
                return "the task requirement ledger has no persisted contract"
            ledger_error = _task_contract_ledger_error(contract, ledger)
            if ledger_error is not None:
                return ledger_error
            if ledger.generation != self._session.task_generation:
                return "the task requirement ledger belongs to a stale generation"
            receipt = checkpoint.receipt
            if receipt is None:
                return "the locked checkpoint has no Runtime verification receipt"
            if self._session.latest_verification_receipt != receipt:
                return "the checkpoint does not match the latest verification receipt"
            if receipt.task_contract_fingerprint != ledger.contract_fingerprint:
                return "the verification receipt belongs to a different task contract"
            if receipt.task_generation != self._session.task_generation:
                return "the verification receipt belongs to a stale task generation"
            if receipt.assurance is not TerminalEvidenceAssurance.TRUSTED:
                return "the verification receipt is not trusted independent evidence"
            if receipt.verification_contract_fingerprint != terminal_fingerprint(
                record.intent.verification
            ):
                return "the verification receipt contract fingerprint does not match"
            if receipt.result_fingerprint != terminal_fingerprint(result):
                return "the verification receipt result fingerprint does not match"
            required_ids = {item.requirement_id for item in ledger.entries}
            if set(receipt.covered_requirement_ids) != required_ids:
                return "the verification receipt does not cover the task contract"
            if any(
                item.state is not TerminalRequirementState.SATISFIED
                or item.assurance is not TerminalEvidenceAssurance.TRUSTED
                or item.evidence_provenance is not receipt.evidence_provenance
                or item.artifact_fingerprints != receipt.artifact_fingerprints
                or item.latest_evidence_action_id != receipt.action_id
                or item.latest_result_fingerprint != receipt.result_fingerprint
                or item.evidence_generation != self._session.task_generation
                for item in ledger.entries
            ):
                return "the task requirement ledger is not verified by the receipt"
        return None

    def submission_gate_error(self) -> str | None:
        """Return None only for a successful but non-trusted final verification."""

        if not self._records:
            return "no committed verification command exists"
        record = self._records[-1]
        if record.intent.command_role is not TerminalCommandRole.VERIFY:
            return "the last committed command is not marked verify"
        if not _current_generation_is_verifiable(self._session):
            return "submission has no stable current-generation task state"
        if not record.result.succeeded or record.intent.verification is None:
            return "the latest verification did not pass"
        receipt = self._session.latest_verification_receipt
        ledger = self._session.task_ledger
        if receipt is None or ledger is None:
            return "submission has no verification receipt and requirement ledger"
        if receipt.action_id != record.action_id or not receipt.passed:
            return "submission receipt does not match the latest verification"
        if receipt.assurance is TerminalEvidenceAssurance.TRUSTED:
            return "trusted verification must use the success lock"
        required_ids = {item.requirement_id for item in ledger.entries}
        if set(receipt.covered_requirement_ids) != required_ids:
            return "submission receipt does not cover the task contract"
        if any(
            item.state is not TerminalRequirementState.SATISFIED
            or item.assurance is not TerminalEvidenceAssurance.SELF_CHECKED
            or item.evidence_provenance is not receipt.evidence_provenance
            or item.artifact_fingerprints != receipt.artifact_fingerprints
            or item.latest_evidence_action_id != receipt.action_id
            or item.evidence_generation != self._session.task_generation
            for item in ledger.entries
        ):
            return "submission ledger is not self-checked by the latest receipt"
        return None

    def mark_submitted_unverified(self, summary: str) -> None:
        gate_error = self.submission_gate_error()
        if gate_error is not None:
            raise RuntimeError(f"terminal submission rejected: {gate_error}")
        self._agent_summary = summary
        self._session = self._session.model_copy(
            update={
                "completion_disposition": (
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED
                )
            }
        )
        self._append(
            "agent.submitted_unverified",
            {
                "summary": summary,
                "session": self._session.model_dump(mode="json"),
            },
        )

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

    def record_reconciliation_proposal_rejection(
        self,
        rejection: TerminalProposalRejection,
    ) -> None:
        self._session = self._session.model_copy(
            update={
                "reconciliation_proposal_rejections": (
                    self._session.reconciliation_proposal_rejections + 1
                ),
                "reconciliation_blocker": rejection.message,
                "last_reconciliation_proposal_rejection": rejection,
            }
        )
        self._append(
            "agent.reconciliation_proposal_rejected",
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
        self._session = self._session.model_copy(
            update={
                "completion_disposition": (
                    TerminalCompletionDisposition.SUCCESS_LOCKED
                )
            }
        )
        self._append(
            "agent.completed",
            {"summary": summary, "session": self._session.model_dump(mode="json")},
        )

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
            result.succeeded
            and record.intent.process_reference is not None
        ):
            references = [
                item
                for item in references
                if item.reference_id
                != record.intent.process_reference.reference_id
            ]
            references.append(record.intent.process_reference)
        task_ledger = session.task_ledger
        task_generation = session.task_generation
        known_state_generation = session.known_state_generation
        successful_work_generation = session.successful_work_generation
        pending_repair_receipt_id = session.pending_repair_receipt_id
        repair_applied_action_id = session.repair_applied_action_id
        pending_before_work = pending_repair_receipt_id
        work_changed_state = bool(
            record.intent.command_role is TerminalCommandRole.WORK
            and result.execution_state is not TerminalExecutionState.FAILED_TO_START
        )
        if work_changed_state:
            task_generation += 1
            known_state_generation = (
                task_generation if result.settled else None
            )
            successful_work_generation = (
                task_generation if result.succeeded else None
            )
            if task_ledger is not None:
                task_ledger = task_ledger.model_copy(
                    update={
                        "generation": task_generation,
                        "entries": tuple(
                            entry.model_copy(
                                update={
                                    "state": TerminalRequirementState.UNKNOWN,
                                    "assurance": TerminalEvidenceAssurance.NONE,
                                    "evidence_provenance": TerminalEvidenceProvenance.LEGACY_UNSPECIFIED,
                                    "artifact_fingerprints": (),
                                    "latest_evidence_action_id": None,
                                    "latest_result_fingerprint": None,
                                    "evidence_generation": None,
                                }
                            )
                            for entry in task_ledger.entries
                        ),
                    }
                )
            if result.succeeded and pending_before_work is not None:
                repair_applied_action_id = record.action_id
            elif pending_before_work is not None:
                repair_applied_action_id = None
        verification_receipt = session.latest_verification_receipt
        if (
            task_ledger is not None
            and record.intent.command_role is TerminalCommandRole.VERIFY
            and record.intent.verification is not None
        ):
            covered_ids = tuple(
                record.intent.verification.requirement_coverage
            )
            passed = result.succeeded
            conclusively_blocked = bool(
                result.settled
                and result.return_code not in (None, 0)
            )
            result_fingerprint = terminal_fingerprint(result)
            verification_receipt = TerminalVerificationReceipt(
                action_id=record.action_id,
                task_generation=task_generation,
                task_contract_fingerprint=(
                    task_ledger.contract_fingerprint
                ),
                verification_contract_fingerprint=terminal_fingerprint(
                    record.intent.verification
                ),
                command_fingerprint=record.intent.execution_fingerprint,
                result_fingerprint=result_fingerprint,
                covered_requirement_ids=covered_ids,
                execution_state=result.execution_state,
                return_code=result.return_code,
                timed_out=result.timed_out,
                transport_failed=result.transport_failed,
                passed=passed,
                evidence_provenance=_runtime_evidence_provenance(
                    record.intent.verification,
                    result,
                ),
                assurance=_verification_assurance(
                    record.intent.verification,
                    result,
                    contract_complete=session.contract_coverage_complete,
                ),
                artifact_fingerprints=_runtime_artifact_fingerprints(result),
                observed_at=record.committed_at,
            )
            covered = set(covered_ids)
            assurance = verification_receipt.assurance
            provenance = verification_receipt.evidence_provenance
            artifact_fingerprints = verification_receipt.artifact_fingerprints
            task_ledger = task_ledger.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(
                            update={
                                "state": (
                                    TerminalRequirementState.SATISFIED
                                    if passed
                                    else TerminalRequirementState.UNKNOWN
                                ),
                                "assurance": (
                                    assurance if passed else TerminalEvidenceAssurance.NONE
                                ),
                                "evidence_provenance": (
                                    provenance
                                    if passed
                                    else TerminalEvidenceProvenance.LEGACY_UNSPECIFIED
                                ),
                                "artifact_fingerprints": (
                                    artifact_fingerprints if passed else ()
                                ),
                                "latest_evidence_action_id": (
                                    record.action_id if passed else None
                                ),
                                "latest_result_fingerprint": (
                                    result_fingerprint if passed else None
                                ),
                                "evidence_generation": (
                                    task_generation if passed else None
                                ),
                            }
                        )
                        if entry.requirement_id in covered
                        else entry
                        for entry in task_ledger.entries
                    )
                }
            )
            if conclusively_blocked:
                pending_repair_receipt_id = record.action_id
                repair_applied_action_id = None
            elif passed:
                pending_repair_receipt_id = None
                repair_applied_action_id = None
        verified_checkpoint = session.verified_checkpoint
        if (
            result.succeeded
            and record.intent.command_role is TerminalCommandRole.VERIFY
            and record.intent.verification is not None
            and verification_receipt is not None
            and verification_receipt.assurance is TerminalEvidenceAssurance.TRUSTED
        ):
            verified_checkpoint = TerminalVerifiedCheckpoint(
                action_id=record.action_id,
                task_generation=task_generation,
                call_key=record.intent.call_key,
                command_fingerprint=record.intent.execution_fingerprint,
                verification=record.intent.verification,
                receipt=(
                    verification_receipt
                    if verification_receipt is not None
                    and verification_receipt.action_id == record.action_id
                    else None
                ),
                committed_at=record.committed_at,
            )
        failed_verification_attempts = session.failed_verification_attempts
        verification_corrections = session.verification_corrections
        if (
            record.intent.command_role is TerminalCommandRole.VERIFY
            and result.settled
            and result.return_code != 0
        ):
            failed_verification_attempts += 1
        elif (
            record.intent.command_role is TerminalCommandRole.WORK
            and result.succeeded
            and pending_before_work is not None
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
        reconciliation_state = session.reconciliation_state
        reconciliation_receipt = session.latest_reconciliation_receipt
        if work_changed_state:
            reconciliation_state = TerminalReconciliationState.NOT_REQUIRED
            reconciliation_receipt = None
        reconciliation_required = session.in_doubt_reconciliation_required
        if result.execution_state is TerminalExecutionState.IN_DOUBT:
            reconciliation_rejections = (
                session.reconciliation_proposal_rejections
                if reconciliation_required
                else 0
            )
            reconciliation_required = True
            reconciliation_state = TerminalReconciliationState.REQUIRED
            reconciliation_receipt = None
            known_state_generation = None
        elif (
            reconciliation_required
            and record.intent.command_role is TerminalCommandRole.INSPECT
            and result.succeeded
        ):
            reconciliation_required = False
            reconciliation_state = TerminalReconciliationState.STABLE_UNVERIFIED
            known_state_generation = task_generation
            reconciliation_receipt = TerminalReconciliationReceipt(
                action_id=record.action_id,
                task_generation=task_generation,
                result_fingerprint=terminal_fingerprint(result),
                observed_at=record.committed_at,
            )
            reconciliation_rejections = session.reconciliation_proposal_rejections
        else:
            reconciliation_rejections = session.reconciliation_proposal_rejections
        failure_signatures = session.latest_failure_signatures
        if not result.succeeded:
            failure_signatures = terminal_failure_signatures(result)
        elif record.intent.command_role is TerminalCommandRole.VERIFY:
            failure_signatures = ()
        update: dict[str, object] = {
            "committed_commands": session.committed_commands + 1,
            "timed_out_commands": session.timed_out_commands + int(result.timed_out),
            "in_doubt_commands": session.in_doubt_commands
            + int(result.execution_state is TerminalExecutionState.IN_DOUBT),
            "in_doubt_reconciliation_required": reconciliation_required,
            "reconciliation_state": reconciliation_state,
            "latest_reconciliation_receipt": reconciliation_receipt,
            "reconciliation_blocker": None,
            "last_reconciliation_proposal_rejection": None,
            "reconciliation_proposal_rejections": (
                reconciliation_rejections
            ),
            "process_references": tuple(references),
            "completion_blocker": None,
            "proposal_blocker": None,
            "last_proposal_rejection": None,
            "consecutive_inspections": consecutive_inspections,
            "inspection_commands": inspection_commands,
            "task_generation": task_generation,
            "known_state_generation": known_state_generation,
            "successful_work_generation": successful_work_generation,
            "pending_repair_receipt_id": pending_repair_receipt_id,
            "repair_applied_action_id": repair_applied_action_id,
            "verified_checkpoint": verified_checkpoint,
            "failed_verification_attempts": failed_verification_attempts,
            "verification_corrections": verification_corrections,
            "latest_failure_signatures": failure_signatures,
            "task_ledger": task_ledger,
            "latest_verification_receipt": verification_receipt,
        }
        if record.governance_status != "applied":
            update["denied_commands"] = session.denied_commands + 1
        if result.settled:
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
        if (
            self._session.known_state_generation is None
            and not self._session.in_doubt_reconciliation_required
        ):
            latest_work = next(
                (
                    record
                    for record in reversed(self._records)
                    if record.intent.command_role is TerminalCommandRole.WORK
                ),
                None,
            )
            if latest_work is not None and latest_work.result.settled:
                self._session = self._session.model_copy(
                    update={
                        "known_state_generation": self._session.task_generation
                    }
                )

    def _replay_event(self, kind: str, payload: Any) -> None:
        if kind == "task.contract.bound":
            contract = TerminalTaskContract.model_validate(
                payload["contract"]
            )
            session = TerminalSessionSnapshot.model_validate(
                payload["session"]
            )
            ledger_error = _task_contract_ledger_error(
                contract,
                session.task_ledger,
            )
            if ledger_error is not None:
                self._trace_consistent = False
                raise RuntimeError(ledger_error)
            if (
                self._task_contract is not None
                and self._task_contract != contract
            ):
                self._trace_consistent = False
                raise RuntimeError("terminal task contract changed in transcript")
            self._task_contract = contract
            self._session = session
        elif kind == "pending.saved":
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
        elif kind == "agent.reconciliation_proposal_rejected":
            self._session = TerminalSessionSnapshot.model_validate(payload["session"])
        elif kind == "agent.completed":
            self._agent_summary = str(payload["summary"])
            if "session" in payload:
                self._session = TerminalSessionSnapshot.model_validate(
                    payload["session"]
                )
        elif kind == "agent.submitted_unverified":
            self._agent_summary = str(payload["summary"])
            self._session = TerminalSessionSnapshot.model_validate(
                payload["session"]
            )
        elif kind == "trace.inconsistent":
            self._trace_consistent = False
            self._trace_error = str(payload["reason"])


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
        compact_max_output_tokens: int | None = 8192,
        emergency_max_output_tokens: int | None = 4096,
        required_structured_output: StructuredOutputLevel = StructuredOutputLevel.JSON_SCHEMA,
        strict_json_schema: bool = False,
        delivery_timeout_seconds: float = 120.0,
        emergency_timeout_seconds: float = 90.0,
        minimum_timeout_seconds: float = 120.0,
        minimum_delivery_timeout_seconds: float = 60.0,
        deadline_budget_profile: TerminalDeadlineBudgetProfile | None = None,
    ) -> None:
        self._gateway = gateway
        self._gateway_policy = gateway_policy
        self._target_id = target_id
        self._max_output_tokens = max_output_tokens
        self._compact_max_output_tokens = compact_max_output_tokens
        self._emergency_max_output_tokens = emergency_max_output_tokens
        self._required_structured_output = required_structured_output
        self._strict_json_schema = strict_json_schema
        self.deadline_budget_profile = deadline_budget_profile
        if deadline_budget_profile is not None:
            delivery_timeout_seconds = (
                deadline_budget_profile.compact_inference.maximum_seconds
            )
            emergency_timeout_seconds = (
                deadline_budget_profile.emergency_inference.maximum_seconds
            )
            minimum_timeout_seconds = (
                deadline_budget_profile.normal_inference.minimum_seconds
            )
            minimum_delivery_timeout_seconds = (
                deadline_budget_profile.compact_inference.minimum_seconds
            )
        if delivery_timeout_seconds <= 0:
            raise ValueError("delivery inference timeout must be positive")
        self._delivery_timeout_seconds = delivery_timeout_seconds
        if emergency_timeout_seconds <= 0:
            raise ValueError("emergency inference timeout must be positive")
        self._emergency_timeout_seconds = emergency_timeout_seconds
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
        self._minimum_emergency_timeout_seconds = (
            deadline_budget_profile.emergency_inference.minimum_seconds
            if deadline_budget_profile is not None
            else minimum_delivery_timeout_seconds
        )
        if delivery_timeout_seconds < minimum_delivery_timeout_seconds:
            raise ValueError(
                "delivery inference timeout cannot be below the minimum "
                "delivery timeout"
            )
        if emergency_timeout_seconds < minimum_delivery_timeout_seconds:
            raise ValueError(
                "emergency inference timeout cannot be below the minimum "
                "delivery timeout"
            )
        configured_normal_timeout = gateway_policy.budget.max_elapsed_seconds
        self.deadline_timing = TerminalInferenceTiming(
            normal_preferred_seconds=(
                deadline_budget_profile.normal_inference.preferred_seconds
                if deadline_budget_profile is not None
                else (
                    configured_normal_timeout
                    if configured_normal_timeout is not None
                    else max(
                        delivery_timeout_seconds,
                        minimum_timeout_seconds,
                    )
                )
            ),
            compact_preferred_seconds=(
                deadline_budget_profile.compact_inference.preferred_seconds
                if deadline_budget_profile is not None
                else delivery_timeout_seconds
            ),
            emergency_preferred_seconds=(
                deadline_budget_profile.emergency_inference.preferred_seconds
                if deadline_budget_profile is not None
                else emergency_timeout_seconds
            ),
            normal_minimum_seconds=minimum_timeout_seconds,
            compact_minimum_seconds=minimum_delivery_timeout_seconds,
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
                    "If the latest failure says command not found or identifies a "
                    "missing runtime, do not rewrite the same solution for guessed "
                    "interpreter names. Feature-detect the package manager and install "
                    "the complete required tool batch, or choose a genuinely available "
                    "portable implementation. Treat payload.session.latest_failure_"
                    "signatures as the complete minimum repair set and resolve every "
                    "entry together. "
                    "For source-build or package-install tasks, inspect the "
                    "project's packaging metadata, then run its canonical "
                    "end-to-end install or build command early to reproduce "
                    "the actual failure. Do not make speculative compatibility "
                    "patches before a command demonstrates the need. "
                    "Treat the most recent failed command, especially a failed "
                    "verification, as authoritative: address its exact reported "
                    "errors before unrelated inspection. When batch output names "
                    "one exact failed artifact, inspect only that artifact's bounded "
                    "raw or extracted evidence next, repair the named failure, and "
                    "only then rerun the batch. Do not rerun an unchanged whole "
                    "batch after it identifies a specific failing member. Use "
                    "targeted portable "
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
                    "payload.behavior_hints as conditional tactics derived only from "
                    "the task contract; apply relevant hints without inventing a "
                    "task-specific answer. "
                    "payload.remaining_wall_clock_seconds as a hard budget. "
                    "If payload.finalization_mode is true, stop broad exploration. If "
                    "payload.verification_due is true, verify now. Otherwise perform at "
                    "most one focused repair bounded by final_repair_timeout_sec, then "
                    "verify on the next turn. payload.execution_limits.deadline_sequence "
                    "is authoritative: reconcile_then_work_verify and "
                    "reconcile_then_verify permit only inspect now, work_then_verify "
                    "permits only work now, and direct_verify permits only verify now. "
                    "If payload.reconciliation_mode is true, return only one read-only "
                    "inspect command that exits zero solely when the prior IN_DOUBT "
                    "process is stopped and affected artifacts are in a known state; "
                    "exit nonzero if either fact remains uncertain. Do not mutate, "
                    "install, repair, verify, or complete during reconciliation. "
                    "If payload.emergency_mode is true, this is the single compact "
                    "recovery turn after a delivery inference timeout: return only "
                    "complete, the shortest sufficient independent verification, "
                    "or one bounded recovery command that fits the advertised "
                    "dynamic timeout. Do not perform broad analysis. "
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
                    "When the task names a target dependency version or a failure "
                    "shows one removed API, perform one bounded source scan for the "
                    "related compatibility family and repair all justified matches "
                    "before rebuilding. "
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
                    "Keep each mutating command to one coherent phase. Do not combine "
                    "dependency installation, Git history construction, conflict "
                    "resolution, source generation, and verification in one large "
                    "heredoc; persist successful phases separately and syntax-check "
                    "complex shell before execution. If the evaluator calls a Python "
                    "entry point using only built-in lists, mappings, and scalar "
                    "values, prefer a standard-library implementation unless the "
                    "task explicitly requires a third-party dependency or guarantees "
                    "it in the verifier environment. For long build, search, or "
                    "optimization loops, feature-detect an internal watchdog and set "
                    "it at least cleanup_grace_seconds plus timeout_admission_margin_"
                    "seconds shorter than the outer timeout, with an EXIT cleanup "
                    "trap. If no watchdog exists, partition the work into resumable "
                    "bounded phases; never rely only on the outer timeout for a "
                    "mutating loop. "
                    "Reserve a command for independent verification after changing "
                    "task artifacts. Verification must use separately derived "
                    "evidence or official tests, not a copy of the production "
                    "algorithm. Gate verification only on explicit task requirements "
                    "and the stable requirement descriptions in the payload. Do not "
                    "invent security, TLS, hardening, or repository-wide cleanliness "
                    "criteria; do not scan or modify unrelated artifacts solely "
                    "because names resemble credentials. Literal placeholders or "
                    "test credentials supplied by the task are allowed unless an "
                    "explicit requirement says to transform them. Verify explicit "
                    "preservation and no-modification constraints as well as positive "
                    "behavior, and for scoped repository edits confirm unrelated "
                    "tracked files remain unchanged. Set independence_method to the "
                    "actual independent "
                    "oracle family and process_isolation to a fresh process or "
                    "ephemeral fixture. For extraction, decoding, and forensic tasks, "
                    "validate the claimed answer through a second evidence modality, "
                    "not the same metadata label or parsing assumption used to create "
                    "it. For performance requirements set performance_protocol to "
                    "cold_unique_inputs, generate unseen inputs, measure first-use "
                    "behavior in a fresh process, and require a meaningful margin. "
                    "Never warm, cache, or time the same measured input beforehand. "
                    "Do not complete unless the last committed command is a "
                    "successful verify command. For execute, include a call_key "
                    "that is absent from payload.used_call_keys; retries need a "
                    "new attempt suffix. Set cwd to null or an absolute POSIX path, "
                    "never '.' or another relative path, and set summary to null. "
                    "timeout_sec must be null or fall within payload.execution_limits; "
                    "never request a value above max_timeout_sec. The only execution "
                    "tool is shell_exec. apply_patch is explicitly unavailable; use "
                    "a portable file edit method listed in payload.tool_capabilities. "
                    "For verify, also keep timeout_sec at or below "
                    "payload.execution_limits.max_verification_timeout_sec. "
                    "For inspect or work, set verification to null. For verify, provide a "
                    "verification object naming official tests or an independently "
                    "derived check, its evidence sources, affected artifact paths, "
                    "evidence_provenance, optional path=sha256 artifact fingerprints, "
                    "and every stable requirement ID. Use task_provided only for tests "
                    "or evidence that existed independently of your solution; use "
                    "external_standard only for a genuinely external oracle. Runtime "
                    "observations and agent-generated tests are self-checks and cannot "
                    "claim SUCCESS_LOCKED. "
                    "every stable requirement_id from payload.requirements in "
                    "requirement_coverage, and validation methods. Coverage must "
                    "contain IDs only; descriptions authored by you are rejected. "
                    "Requirement kind is a conservative Runtime classification "
                    "from explicit task headings; unclassified requirements remain "
                    "mandatory. payload.ledger_projection is shadow evidence, not "
                    "authority to skip work or verification; it contains only unmet "
                    "or newly changed requirement evidence while the full audit trail "
                    "remains in the Journal. "
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
                "payload": _terminal_model_payload(request),
            },
            response_schema=_terminal_turn_response_schema(
                strict=self._strict_json_schema,
            ),
            requirements=InferenceRequirements(
                required_structured_output=self._required_structured_output,
                max_output_tokens=self._response_max_output_tokens(request),
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
        precomputed_inference_cap = (
            request.execution_limits.max_inference_timeout_sec
        )
        if precomputed_inference_cap is not None:
            configured = gateway_policy.budget.max_elapsed_seconds
            bounded = precomputed_inference_cap
            if configured is not None:
                bounded = min(bounded, configured)
            if request.emergency_mode:
                bounded = min(bounded, self._emergency_timeout_seconds)
            elif (
                request.delivery_mode
                or request.finalization_mode
                or request.reconciliation_mode
                or request.repair_mode
                or request.verification_due
            ):
                bounded = min(bounded, self._delivery_timeout_seconds)
            if bounded <= 0.0:
                raise InferenceExecutionBudgetError(
                    "insufficient wall-clock capacity for another inference: "
                    "available 0.0 seconds; no positive inference window remains"
                )
            gateway_policy = gateway_policy.model_copy(
                update={
                    "budget": gateway_policy.budget.model_copy(
                        update={"max_elapsed_seconds": bounded}
                    )
                }
            )
        elif request.remaining_wall_clock_seconds is not None:
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
                or request.emergency_mode
                or request.finalization_mode
                or request.reconciliation_mode
                or request.repair_mode
                or request.verification_due
            ):
                timeout_cap = (
                    self._emergency_timeout_seconds
                    if request.emergency_mode
                    else self._delivery_timeout_seconds
                )
                bounded = min(bounded, timeout_cap)
            if bounded <= 0.0:
                raise InferenceExecutionBudgetError(
                    "insufficient wall-clock capacity for another inference: "
                    "available 0.0 seconds; no positive inference window "
                    "remains"
                )
            gateway_policy = gateway_policy.model_copy(
                update={
                    "budget": gateway_policy.budget.model_copy(
                        update={"max_elapsed_seconds": bounded}
                    )
                }
            )
        elif (
            request.delivery_mode
            or request.emergency_mode
            or request.finalization_mode
            or request.reconciliation_mode
            or request.repair_mode
            or request.verification_due
        ):
            configured = gateway_policy.budget.max_elapsed_seconds
            bounded = (
                self._emergency_timeout_seconds
                if request.emergency_mode
                else self._delivery_timeout_seconds
            )
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
            self._minimum_emergency_timeout_seconds
            if request.emergency_mode
            else (
                self._minimum_delivery_timeout_seconds
                if (
                    request.delivery_mode
                    or request.finalization_mode
                    or request.reconciliation_mode
                    or request.repair_mode
                    or request.verification_due
                )
                else self._minimum_timeout_seconds
            )
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

    def _response_max_output_tokens(
        self,
        request: TerminalTurnRequest,
    ) -> int | None:
        """Keep endgame responses small without widening provider support."""

        configured = self._max_output_tokens
        if configured is None:
            return None
        if request.emergency_mode:
            cap = self._emergency_max_output_tokens
        elif (
            request.delivery_mode
            or request.finalization_mode
            or request.reconciliation_mode
            or request.repair_mode
            or request.verification_due
        ):
            cap = self._compact_max_output_tokens
        else:
            cap = None
        return configured if cap is None else min(configured, cap)


def _terminal_turn_response_schema(
    *, strict: bool = False, max_timeout_sec: int | None = None
) -> dict[str, JsonValue]:
    del max_timeout_sec
    schema = cast(
        dict[str, JsonValue],
        TerminalTurnDraft.model_json_schema(mode="validation"),
    )
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("rationale", None)
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
_MUTATING_RECONCILIATION_PATTERNS = (
    re.compile(
        r"(?im)(?:^|[;&|]\s*)(?:rm|mv|cp|install|apt(?:-get)?|apk|dnf|yum|"
        r"pip3?|sed\s+-i|perl\s+-[A-Za-z]*i|tee)\b"
    ),
    re.compile(
        r"(?m)(?:^|\s)[0-9]*>>?\s*"
        r"(?!&[0-9]+(?:\s|$|[;&|]))"
        r"(?!/dev/null(?:\s|$|[;&|]))\S+"
    ),
)
_TEMP_ROOT_ASSIGNMENT = re.compile(
    r"(?m)^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r"[\"']?\$\(\s*mktemp\s+(?:-d|--directory)\b[^)]*\)[\"']?"
)
def _requires_performance_protocol(
    requirements: tuple[TerminalRequirement, ...],
) -> bool:
    description = " ".join(item.description for item in requirements).lower()
    return any(
        marker in description
        for marker in (
            "performance",
            "faster",
            "speedup",
            "latency",
            "median time",
            "benchmark",
        )
    )


def _terminal_behavior_hints(
    requirements: tuple[TerminalRequirement, ...],
) -> tuple[str, ...]:
    """Return generic tactics only when the task contract makes them relevant."""

    text = " ".join(item.description for item in requirements).lower()
    hints: list[str] = []
    if any(marker in text for marker in ("parse", "parser", "syntax", "line")):
        hints.append(
            "For parser failures, isolate the first failing line and make the "
            "smallest local repair before rerunning the bounded parser check."
        )
    if any(
        marker in text
        for marker in (
            "exact output",
            "byte-for-byte",
            "exactly",
            "expected output",
            "format",
        )
    ):
        hints.append(
            "For exact-output requirements, compare expected and actual bytes with "
            "a bounded diff; existence and parseability alone are insufficient."
        )
    if any(
        marker in text
        for marker in ("allowed labels", "label set", "valid tags", "enum", "one of")
    ):
        hints.append(
            "Validate every produced label or tag against the complete allowed set "
            "and report unexpected and missing members separately."
        )
    if _requires_performance_protocol(requirements):
        hints.append(
            "Profile the representative slow path first, optimize the measured "
            "bottleneck, then benchmark cold unseen inputs with a safety margin."
        )
    return tuple(hints)


def _terminal_model_payload(request: TerminalTurnRequest) -> dict[str, object]:
    """Project durable state into a bounded model context without losing audit data."""

    payload = request.model_dump(mode="json")
    session = payload.get("session")
    if isinstance(session, dict):
        session.pop("task_ledger", None)
    return payload


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


def _known_reconciliation_mutation(command: str) -> str | None:
    git_mutation = _known_verification_mutation(command)
    if git_mutation is not None:
        return git_mutation
    for pattern in _MUTATING_RECONCILIATION_PATTERNS:
        match = pattern.search(command)
        if match is not None:
            return match.group(0).strip()[:160]
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
        profile = getattr(capability, "deadline_budget_profile", None)
        self._deadline_budget_profile = (
            profile
            if isinstance(profile, TerminalDeadlineBudgetProfile)
            else None
        )
        timing = getattr(capability, "deadline_timing", None)
        self._deadline_timing = (
            timing if isinstance(timing, TerminalInferenceTiming) else None
        )
        self._uses_deadline_slots = bool(
            self._deadline_budget_profile is not None
            or self._deadline_timing is not None
        )

    async def plan(self, state: AgentState) -> PlanDecision:
        await self.reconcile_observation(state)
        requirements = _task_requirements(state.task.description)
        unmapped_fragments = _contract_coverage_gaps(
            state.task.description,
            requirements,
        )
        self._journal.bind_task_contract(
            requirements,
            coverage_complete=not unmapped_fragments,
            unmapped_fragments=unmapped_fragments,
        )
        trace_error = self._journal.trace_error()
        if trace_error is not None:
            return PlanDecision.fail(
                error="terminal trace is inconsistent: " + trace_error
            )
        pending = self._journal.pending()
        if pending is not None:
            return PlanDecision.execute(
                pending.action,
                reason="Resume the original persisted terminal proposal.",
            )
        session = self._journal.snapshot()
        if (
            session.in_doubt_reconciliation_required
            and session.consecutive_inspections
            >= self._policy.max_in_doubt_reconciliation_attempts
        ):
            return PlanDecision.fail(
                error=(
                    "terminal IN_DOUBT state remains unresolved after bounded "
                    "read-only reconciliation attempts"
                )
            )
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
                        "completion_disposition": (
                            TerminalCompletionDisposition.SUCCESS_LOCKED.value
                        ),
                        "summary": summary,
                    },
                    reason=(
                        "Finalize the verified checkpoint without another model "
                        "inference."
                    ),
                )
            self._journal.mark_trace_inconsistent(
                "verified checkpoint failed completion validation: "
                + gate_error
            )
            return PlanDecision.fail(
                error="terminal trace is inconsistent: " + gate_error
            )
        if (
            session.latest_verification_receipt is not None
            and self._journal.submission_gate_error() is None
        ):
            summary = (
                "The final self-check passed without a trusted independent "
                "oracle; artifacts were submitted for Harbor verification."
            )
            self._journal.mark_submitted_unverified(summary)
            return PlanDecision.complete(
                output={
                    "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                    "agent_complete": False,
                    "completion_disposition": (
                        TerminalCompletionDisposition.SUBMITTED_UNVERIFIED.value
                    ),
                    "summary": summary,
                },
                reason="Submit self-checked artifacts without claiming success.",
            )
        budget_error = self._budget_error(session, allow_command_limit=False)
        if budget_error is not None:
            return PlanDecision.fail(error=budget_error)
        request = self._turn_request(state, session)
        try:
            proposal = await self._capability.propose(request)
        except InferenceExecutionBudgetError as exc:
            if request.emergency_mode:
                raise RunBudgetExhaustedError(
                    "active_execution",
                    "terminal emergency inference budget exhausted: "
                    + exc.reason,
                ) from exc
            retry_request = self._turn_request(
                state,
                session,
                force_delivery=True,
                force_emergency=True,
            )
            retry_inference_cap = (
                retry_request.execution_limits.max_inference_timeout_sec
            )
            deadline_timing = self._deadline_timing
            if (
                self._uses_deadline_slots
                and deadline_timing is not None
                and retry_inference_cap is not None
                and retry_inference_cap
                < deadline_timing.compact_minimum_seconds
            ):
                raise RunBudgetExhaustedError(
                    "active_execution",
                    "terminal emergency inference budget exhausted: minimum "
                    "current inference and action do not fit",
                ) from exc
            try:
                proposal = await self._capability.propose(retry_request)
            except InferenceExecutionBudgetError as retry_exc:
                raise RunBudgetExhaustedError(
                    "active_execution",
                    "terminal emergency inference budget exhausted: "
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
        if (
            request.reconciliation_mode
            and draft.decision is TerminalTurnDecision.COMPLETE
        ):
            validation = _TerminalProposalValidationError(
                code="terminal.in_doubt.reconciliation_required",
                message=(
                    "an unresolved IN_DOUBT command must be reconciled before "
                    "completion"
                ),
                field="decision",
                rejected_value="complete",
                expected="execute one read-only inspect reconciliation command",
            )
            return self._proposal_rejection_decision(
                state=state,
                session=session,
                draft=draft,
                validation=validation,
                reconciliation_mode=True,
            )
        if draft.decision is TerminalTurnDecision.COMPLETE:
            assert draft.summary is not None
            gate_error = self._journal.completion_gate_error()
            if gate_error is None:
                self._journal.mark_complete(draft.summary)
                return PlanDecision.complete(
                    output={
                        "profile": AAR_TERMINAL_SEQUENTIAL_PROFILE,
                        "agent_complete": True,
                        "completion_disposition": (
                            TerminalCompletionDisposition.SUCCESS_LOCKED.value
                        ),
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
            intent = self._resolve_intent(
                draft,
                session,
                execution_limits=request.execution_limits,
                finalization_mode=request.finalization_mode,
                reconciliation_mode=request.reconciliation_mode,
                deadline_sequence=(
                    request.execution_limits.deadline_sequence
                ),
            )
            self._validate_intent(
                intent,
                session,
                recovery_mode=request.recovery_mode,
                repair_mode=request.repair_mode,
                verification_due=request.verification_due,
                finalization_mode=request.finalization_mode,
                reconciliation_mode=request.reconciliation_mode,
                deadline_sequence=(
                    request.execution_limits.deadline_sequence
                ),
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
                reconciliation_mode=request.reconciliation_mode,
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
        reconciliation_mode: bool = False,
    ) -> PlanDecision:
        rejection_count = (
            session.reconciliation_proposal_rejections
            if reconciliation_mode
            else session.proposal_rejections
        )
        rejection_limit = (
            self._policy.max_reconciliation_proposal_rejections
            if reconciliation_mode
            else self._policy.max_proposal_rejections
        )
        if rejection_count >= rejection_limit:
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
        if reconciliation_mode:
            self._journal.record_reconciliation_proposal_rejection(rejection)
        else:
            self._journal.record_proposal_rejection(rejection)
        updated_session = self._journal.snapshot()
        updated_count = (
            updated_session.reconciliation_proposal_rejections
            if reconciliation_mode
            else updated_session.proposal_rejections
        )
        action = ActionRequest(
            action_id=uuid5(
                _TERMINAL_ACTION_NAMESPACE,
                "|".join(
                    (
                        str(state.run_id),
                        str(state.revision),
                        "proposal-rejected",
                        str(updated_count),
                        rejection.code,
                    )
                ),
            ),
            name=TERMINAL_PROPOSAL_REJECTION_ACTION,
            arguments={
                "rejection": rejection.model_dump(mode="json"),
                "proposal_rejections": updated_count,
                "reconciliation": reconciliation_mode,
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
        force_emergency: bool = False,
    ) -> TerminalTurnRequest:
        requirements = _task_requirements(state.task.description)
        elapsed_seconds = max(
            0.0,
            (utc_now() - session.started_at).total_seconds(),
        )
        remaining_wall_clock_seconds = self._remaining_wall_clock_seconds(
            session,
        )
        all_records = self._journal.recent_records(self._policy.max_commands)
        has_successful_work = _current_generation_has_successful_work(session)
        has_verifiable_state = _current_generation_is_verifiable(session)
        finalization_mode = bool(
            remaining_wall_clock_seconds is not None
            and remaining_wall_clock_seconds
            <= self._effective_finalization_threshold_seconds()
            and (
                self._uses_deadline_slots
                or has_successful_work
            )
        )
        reconciliation_mode = session.in_doubt_reconciliation_required
        dynamic_timeout_sec = self._dynamic_timeout_limit(
            remaining_wall_clock_seconds
        )
        dynamic_verification_timeout_sec = min(
            dynamic_timeout_sec,
            self._policy.max_verification_timeout_sec,
        )
        delivery_mode = force_delivery or bool(
            self._policy.max_wall_clock_seconds is not None
            and elapsed_seconds
            >= self._policy.max_wall_clock_seconds
            * self._policy.delivery_mode_fraction
        )
        compact_context = bool(
            delivery_mode or finalization_mode or reconciliation_mode
        )
        context_record_limit = (
            self._policy.max_delivery_context_records
            if compact_context
            else self._policy.max_context_records
        )
        context_output_limit = (
            self._policy.max_delivery_context_output_characters
            if compact_context
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
        repair_mode = bool(
            session.pending_repair_receipt_id is not None
            and session.repair_applied_action_id is None
        )
        verification_due = bool(
            has_verifiable_state
            and not reconciliation_mode
            and (
                session.reconciliation_state
                is TerminalReconciliationState.STABLE_UNVERIFIED
                or session.repair_applied_action_id is not None
                or (
                    finalization_mode
                    and session.pending_repair_receipt_id is None
                )
            )
        )
        deadline_sequence: TerminalDeadlineSequence | None = None
        if self._uses_deadline_slots:
            if reconciliation_mode:
                deadline_sequence = (
                    TerminalDeadlineSequence.RECONCILE_THEN_VERIFY
                    if has_successful_work
                    else TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY
                )
            elif verification_due:
                deadline_sequence = TerminalDeadlineSequence.DIRECT_VERIFY
            elif finalization_mode or repair_mode:
                deadline_sequence = TerminalDeadlineSequence.WORK_THEN_VERIFY
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
            or finalization_mode
            or reconciliation_mode
            or repair_mode
            or verification_due
            or artifact_recovery
        )
        deadline_slots: TerminalDeadlineSlots | None = None
        if self._uses_deadline_slots:
            deadline_slots = self._deadline_slots(
                remaining_wall_clock_seconds,
                delivery_mode=delivery_mode,
                emergency_mode=force_emergency,
                finalization_mode=finalization_mode,
                reconciliation_mode=reconciliation_mode,
                repair_mode=repair_mode,
                verification_due=verification_due,
                sequence=deadline_sequence,
            )
            if deadline_slots.action_limit_seconds is not None:
                dynamic_timeout_sec = max(
                    1,
                    deadline_slots.action_limit_seconds,
                )
                dynamic_verification_timeout_sec = min(
                    dynamic_timeout_sec,
                    self._policy.max_verification_timeout_sec,
                )
        profile = self._deadline_budget_profile
        work_role_maximum = (
            self._policy.final_repair_timeout_sec
            if deadline_sequence is TerminalDeadlineSequence.WORK_THEN_VERIFY
            else self._policy.max_timeout_sec
        )
        inspection_role_maximum = 30 if reconciliation_mode else 60
        if profile is not None:
            work_role_maximum = int(
                (
                    profile.final_work
                    if deadline_sequence
                    is TerminalDeadlineSequence.WORK_THEN_VERIFY
                    else profile.normal_work
                ).maximum_seconds
            )
            inspection_role_maximum = int(
                (
                    profile.reconciliation_inspection
                    if reconciliation_mode
                    else profile.normal_inspection
                ).maximum_seconds
            )
        return TerminalTurnRequest(
            run_id=state.run_id,
            task_id=state.task.task_id,
            instruction=state.task.description,
            requirements=requirements,
            session=session,
            ledger_projection=_ledger_projection(session),
            behavior_hints=_terminal_behavior_hints(requirements),
            recent_history=history,
            used_call_keys=used_call_keys,
            execution_limits=TerminalExecutionLimits(
                default_timeout_sec=min(
                    self._policy.default_timeout_sec,
                    dynamic_timeout_sec,
                ),
                max_timeout_sec=dynamic_timeout_sec,
                max_work_timeout_sec=min(
                    dynamic_timeout_sec,
                    work_role_maximum,
                ),
                max_verification_timeout_sec=(
                    dynamic_verification_timeout_sec
                ),
                max_inspection_timeout_sec=(
                    min(
                        dynamic_timeout_sec,
                        inspection_role_maximum,
                    )
                    if deadline_sequence
                    in {
                        TerminalDeadlineSequence.RECONCILE_THEN_WORK_VERIFY,
                        TerminalDeadlineSequence.RECONCILE_THEN_VERIFY,
                    }
                    or deadline_sequence is None
                    else None
                ),
                deadline_sequence=deadline_sequence,
                max_inference_timeout_sec=(
                    None
                    if deadline_slots is None
                    else (
                        deadline_slots.inference_limit_seconds
                        if deadline_slots.feasible
                        else 0.0
                    )
                ),
                final_repair_timeout_sec=min(
                    dynamic_timeout_sec,
                    self._policy.final_repair_timeout_sec,
                ),
                cleanup_grace_seconds=self._policy.cleanup_grace_seconds,
                timeout_admission_margin_seconds=(
                    self._policy.timeout_admission_margin_seconds
                ),
                followup_inference_reserve_seconds=(
                    0.0
                    if deadline_slots is None
                    else deadline_slots.followup_inference_seconds
                ),
                verification_reserve_seconds=(
                    0.0
                    if deadline_slots is None
                    else deadline_slots.verification_seconds
                ),
                deadline_cleanup_reserve_seconds=(
                    0.0
                    if deadline_slots is None
                    else deadline_slots.cleanup_seconds
                ),
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
            emergency_mode=force_emergency,
            finalization_mode=finalization_mode,
            reconciliation_mode=reconciliation_mode,
            recovery_mode=recovery_mode,
            artifact_first_mode=artifact_first_mode,
            repair_mode=repair_mode,
            verification_due=verification_due,
            execution_semantics=_EXECUTION_SEMANTICS,
        )

    def _remaining_wall_clock_seconds(
        self,
        session: TerminalSessionSnapshot,
    ) -> float | None:
        if self._policy.max_wall_clock_seconds is None:
            return None
        elapsed_seconds = max(
            0.0,
            (utc_now() - session.started_at).total_seconds(),
        )
        return max(
            0.0,
            self._policy.max_wall_clock_seconds - elapsed_seconds,
        )

    def _effective_finalization_threshold_seconds(self) -> float:
        profile = self._deadline_budget_profile
        if profile is not None:
            required = (
                profile.compact_inference.minimum_seconds
                + profile.model_cancellation_cleanup_seconds
                + profile.final_work.minimum_seconds
                + self._policy.provider_grace_sec
                + profile.convergence_inference.minimum_seconds
                + profile.model_cancellation_cleanup_seconds
                + profile.verification.minimum_seconds
                + self._policy.provider_grace_sec
                + self._policy.cleanup_grace_seconds
                + self._policy.timeout_admission_margin_seconds
            )
            return max(
                self._policy.finalization_mode_threshold_seconds,
                required,
            )
        timing = self._deadline_timing
        if timing is None:
            return self._policy.finalization_mode_threshold_seconds
        future_verification_seconds = min(
            self._policy.final_repair_timeout_sec,
            self._policy.max_verification_timeout_sec,
        )
        required = (
            timing.compact_minimum_seconds
            + self._policy.final_repair_timeout_sec
            + timing.compact_minimum_seconds
            + future_verification_seconds
            + self._policy.cleanup_grace_seconds
            + self._policy.timeout_admission_margin_seconds
        )
        return max(
            self._policy.finalization_mode_threshold_seconds,
            required,
        )

    def _deadline_slots(
        self,
        remaining_wall_clock_seconds: float | None,
        *,
        delivery_mode: bool = False,
        emergency_mode: bool = False,
        finalization_mode: bool = False,
        reconciliation_mode: bool = False,
        repair_mode: bool = False,
        verification_due: bool = False,
        include_current_inference: bool = True,
        sequence: TerminalDeadlineSequence | None = None,
        command_role: TerminalCommandRole | None = None,
    ) -> TerminalDeadlineSlots:
        profile = self._deadline_budget_profile
        if profile is not None:
            return allocate_profiled_terminal_deadline_sequence(
                sequence=sequence,
                remaining_seconds=remaining_wall_clock_seconds,
                profile=profile,
                cleanup_seconds=(
                    self._policy.cleanup_grace_seconds
                    + self._policy.timeout_admission_margin_seconds
                ),
                provider_grace_seconds=float(self._policy.provider_grace_sec),
                include_current_inference=include_current_inference,
                emergency_mode=emergency_mode,
                compact_mode=bool(
                    delivery_mode
                    or finalization_mode
                    or reconciliation_mode
                    or repair_mode
                    or verification_due
                ),
                command_role=(
                    None if command_role is None else command_role.value
                ),
            )
        compact = bool(
            delivery_mode
            or emergency_mode
            or finalization_mode
            or reconciliation_mode
            or repair_mode
            or verification_due
        )
        timing = self._deadline_timing
        if timing is None:
            raise RuntimeError("deadline timing is unavailable")
        minimum_inference = (
            timing.compact_minimum_seconds
            if compact
            else timing.normal_minimum_seconds
        )
        preferred_inference = (
            timing.emergency_preferred_seconds
            if emergency_mode
            else (
                timing.compact_preferred_seconds
                if compact
                else timing.normal_preferred_seconds
            )
        )
        if not include_current_inference:
            minimum_inference = 0.0
            preferred_inference = 0.0
        if sequence is not None:
            return allocate_terminal_deadline_sequence(
                sequence=sequence,
                remaining_seconds=remaining_wall_clock_seconds,
                minimum_inference_seconds=minimum_inference,
                preferred_inference_seconds=preferred_inference,
                followup_inference_seconds=timing.compact_minimum_seconds,
                work_timeout_seconds=self._policy.final_repair_timeout_sec,
                verification_timeout_seconds=(
                    self._policy.max_verification_timeout_sec
                    if sequence is TerminalDeadlineSequence.DIRECT_VERIFY
                    else min(
                        self._policy.final_repair_timeout_sec,
                        self._policy.max_verification_timeout_sec,
                    )
                ),
                reconciliation_timeout_seconds=(
                    self._policy.final_repair_timeout_sec
                ),
                cleanup_seconds=(
                    self._policy.cleanup_grace_seconds
                    + self._policy.timeout_admission_margin_seconds
                ),
                include_current_inference=include_current_inference,
            )
        if verification_due:
            preferred_action = self._policy.max_verification_timeout_sec
            maximum_action = self._policy.max_verification_timeout_sec
            followup_inference = 0.0
            verification = 0.0
        elif finalization_mode:
            preferred_action = self._policy.final_repair_timeout_sec
            maximum_action = self._policy.final_repair_timeout_sec
            followup_inference = (
                timing.compact_minimum_seconds
            )
            verification = float(self._policy.max_verification_timeout_sec)
        else:
            preferred_action = min(
                self._policy.default_timeout_sec,
                self._policy.max_timeout_sec,
            )
            maximum_action = self._policy.max_timeout_sec
            followup_inference = (
                timing.compact_minimum_seconds
            )
            verification = float(self._policy.max_verification_timeout_sec)
        return allocate_deadline_slots(
            remaining_seconds=remaining_wall_clock_seconds,
            minimum_inference_seconds=minimum_inference,
            preferred_inference_seconds=preferred_inference,
            preferred_action_seconds=preferred_action,
            maximum_action_seconds=maximum_action,
            followup_inference_seconds=followup_inference,
            verification_seconds=verification,
            cleanup_seconds=(
                self._policy.cleanup_grace_seconds
                + self._policy.timeout_admission_margin_seconds
            ),
        )

    def _dynamic_timeout_limit(
        self,
        remaining_wall_clock_seconds: float | None,
    ) -> int:
        if remaining_wall_clock_seconds is None:
            return self._policy.max_timeout_sec
        deadline_capacity = int(
            max(
                1.0,
                remaining_wall_clock_seconds
                - self._policy.cleanup_grace_seconds
                - self._policy.timeout_admission_margin_seconds,
            )
        )
        return min(self._policy.max_timeout_sec, deadline_capacity)

    def _deadline_timeout_limit(
        self,
        session: TerminalSessionSnapshot,
        *,
        command_role: TerminalCommandRole,
        finalization_mode: bool = False,
        deadline_sequence: TerminalDeadlineSequence | None = None,
    ) -> int | None:
        remaining = self._remaining_wall_clock_seconds(session)
        if remaining is None:
            return None
        if self._uses_deadline_slots:
            fresh_finalization = bool(
                finalization_mode
                or remaining
                <= self._effective_finalization_threshold_seconds()
            )
            effective_sequence = deadline_sequence
            if effective_sequence is None and fresh_finalization:
                effective_sequence = (
                    TerminalDeadlineSequence.DIRECT_VERIFY
                    if command_role is TerminalCommandRole.VERIFY
                    else TerminalDeadlineSequence.WORK_THEN_VERIFY
                )
            slots = self._deadline_slots(
                remaining,
                finalization_mode=fresh_finalization,
                verification_due=(
                    command_role is TerminalCommandRole.VERIFY
                ),
                sequence=effective_sequence,
                include_current_inference=False,
                command_role=command_role,
            )
            return (
                slots.action_limit_seconds or 0
                if slots.feasible
                else 0
            )
        reserve = (
            self._policy.cleanup_grace_seconds
            + self._policy.timeout_admission_margin_seconds
        )
        has_successful_work = _current_generation_has_successful_work(session)
        if (
            command_role is TerminalCommandRole.WORK
            and has_successful_work
            and remaining
            > self._policy.finalization_mode_threshold_seconds
        ):
            reserve += self._policy.finalization_reserve_seconds
        return max(0, int(remaining - reserve))

    def _resolve_intent(
        self,
        draft: TerminalTurnDraft,
        session: TerminalSessionSnapshot,
        *,
        execution_limits: TerminalExecutionLimits,
        finalization_mode: bool = False,
        reconciliation_mode: bool = False,
        deadline_sequence: TerminalDeadlineSequence | None = None,
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
        default_timeout_sec = self._policy.default_timeout_sec
        role_cap = {
            TerminalCommandRole.WORK: execution_limits.max_work_timeout_sec,
            TerminalCommandRole.VERIFY: (
                execution_limits.max_verification_timeout_sec
            ),
            TerminalCommandRole.INSPECT: (
                execution_limits.max_inspection_timeout_sec
                or execution_limits.max_timeout_sec
            ),
        }[draft.command_role]
        advertised_cap = min(execution_limits.max_timeout_sec, role_cap)
        default_timeout_sec = min(default_timeout_sec, advertised_cap)
        requested_timeout_sec = draft.timeout_sec
        candidate_timeout = (
            requested_timeout_sec
            if requested_timeout_sec is not None
            else default_timeout_sec
        )
        applied_cap = advertised_cap
        fresh_deadline_cap: int | None = None
        if draft.command_role in {
            TerminalCommandRole.INSPECT,
            TerminalCommandRole.VERIFY,
        }:
            fresh_deadline_cap = self._deadline_timeout_limit(
                session,
                command_role=draft.command_role,
                finalization_mode=finalization_mode,
                deadline_sequence=deadline_sequence,
            )
            if fresh_deadline_cap is not None:
                applied_cap = min(applied_cap, max(1, fresh_deadline_cap))
        timeout_sec = min(candidate_timeout, applied_cap)
        if timeout_sec < candidate_timeout:
            timeout_cap_reason = (
                TerminalTimeoutCapReason.FRESH_DEADLINE_CAP
                if fresh_deadline_cap is not None
                and fresh_deadline_cap < advertised_cap
                else TerminalTimeoutCapReason.ADVERTISED_CAP
            )
        elif requested_timeout_sec is None:
            timeout_cap_reason = TerminalTimeoutCapReason.RUNTIME_DEFAULT
        else:
            timeout_cap_reason = TerminalTimeoutCapReason.MODEL_REQUESTED
        return TerminalCommandIntent(
            trial_id=session.trial_id,
            call_key=draft.call_key,
            command=draft.command,
            cwd=cwd,
            env=environment,
            timeout_sec=timeout_sec,
            requested_timeout_sec=requested_timeout_sec,
            advertised_timeout_cap_sec=advertised_cap,
            timeout_cap_reason=timeout_cap_reason,
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
        finalization_mode: bool = False,
        reconciliation_mode: bool = False,
        deadline_sequence: TerminalDeadlineSequence | None = None,
        required_requirements: tuple[TerminalRequirement, ...],
    ) -> None:
        del (
            recovery_mode,
            repair_mode,
            verification_due,
            finalization_mode,
            deadline_sequence,
        )
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
        if reconciliation_mode:
            if intent.command_role is not TerminalCommandRole.INSPECT:
                raise _TerminalProposalValidationError(
                    code="terminal.in_doubt.reconciliation_required",
                    message=(
                        "an unresolved IN_DOUBT command requires a read-only "
                        "reconciliation before any state-changing action"
                    ),
                    field="command_role",
                    rejected_value=intent.command_role.value,
                    expected="inspect",
                )
            mutation = _known_reconciliation_mutation(intent.command)
            if mutation is not None:
                raise _TerminalProposalValidationError(
                    code="terminal.in_doubt.reconciliation_mutation",
                    message="IN_DOUBT reconciliation must be read-only",
                    field="command",
                    rejected_value=mutation,
                    expected=(
                        "a status and artifact inspection that exits zero only "
                        "when the prior process is stopped and state is known"
                    ),
                )
        if (
            intent.command_role is TerminalCommandRole.VERIFY
            and not _generation_state_is_known(session)
        ):
            raise _TerminalProposalValidationError(
                code="terminal.verification.unknown_generation_state",
                message=(
                    "read-only verification requires a known current-generation "
                    "task state"
                ),
                field="command_role",
                rejected_value=intent.command_role.value,
                expected="settled work or successful read-only reconciliation",
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
            if (
                _requires_performance_protocol(required_requirements)
                and intent.verification.evidence_kind.value
                != "official_tests"
                and intent.verification.performance_protocol
                is not TerminalPerformanceProtocol.COLD_UNIQUE_INPUTS
            ):
                raise _TerminalProposalValidationError(
                    code="terminal.verification.performance_protocol_required",
                    message=(
                        "performance verification must use cold, unique inputs "
                        "in an isolated process"
                    ),
                    field="verification.performance_protocol",
                    rejected_value=(
                        intent.verification.performance_protocol.value
                    ),
                    expected="cold_unique_inputs",
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
            if (
                session.pending_repair_receipt_id is not None
                and session.repair_applied_action_id is None
            ):
                previous = next(
                    (
                        record
                        for record in reversed(
                            self._journal.recent_records(
                                self._policy.max_commands
                            )
                        )
                        if record.action_id
                        == session.pending_repair_receipt_id
                    ),
                    None,
                )
                if (
                    previous is not None
                    and previous.intent.command_role
                    is TerminalCommandRole.VERIFY
                    and previous.intent.verification is not None
                    and previous.intent.command == intent.command
                    and previous.intent.cwd == intent.cwd
                    and previous.intent.env == intent.env
                    and terminal_fingerprint(previous.intent.verification)
                    == terminal_fingerprint(intent.verification)
                ):
                    raise _TerminalProposalValidationError(
                        code="terminal.verification.unchanged_retry",
                        message=(
                            "a failed verification may be retried without work "
                            "only when its command, inputs, or evidence changes"
                        ),
                        field="verification",
                        rejected_value="unchanged",
                        expected="changed read-only verification evidence",
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
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_REQUIREMENT_SECTION_KINDS = {
    "must achieve": TerminalRequirementKind.ACHIEVE,
    "must produce": TerminalRequirementKind.PRODUCE,
    "must preserve": TerminalRequirementKind.PRESERVE,
    "must not do": TerminalRequirementKind.PROHIBIT,
    "thresholds": TerminalRequirementKind.THRESHOLD,
}
_CONTRACT_PATH_SIGNAL = re.compile(
    r"(?:/(?:[^\s`'\";,]|:(?!\s))+|\b[\w.-]+\."
    r"(?:json|csv|tsv|xml|ya?ml|toml|ini|txt|log|html?|md|pdf|sqlite3?|db)\b)",
    re.IGNORECASE,
)
_CONTRACT_QUANTITY_SIGNAL = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?%?(?![\w.-])")
_CONTRACT_POLICY_SIGNAL = re.compile(
    r"\b(?:exact(?:ly)?|format|schema|must\s+not|do\s+not|never|preserve|"
    r"unchanged|at\s+least|at\s+most|no\s+more\s+than)\b|"
    r"禁止|不得|不可|保留|保持不变|至少|至多",
    re.IGNORECASE,
)


def _contract_coverage_gaps(
    instruction: str,
    requirements: tuple[TerminalRequirement, ...],
) -> tuple[str, ...]:
    """Independently detect critical source clauses lost by requirement parsing."""

    mapped = "\n".join(item.description for item in requirements).casefold()
    gaps: list[str] = []
    for raw_line in instruction.replace("\r\n", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        signals = [
            match.group(0).rstrip(".:")
            for pattern in (
                _CONTRACT_PATH_SIGNAL,
                _CONTRACT_QUANTITY_SIGNAL,
                _CONTRACT_POLICY_SIGNAL,
            )
            for match in pattern.finditer(stripped)
        ]
        if signals and any(signal.casefold() not in mapped for signal in signals):
            normalized = " ".join(stripped.split())[:512]
            if normalized not in gaps:
                gaps.append(normalized)
    return tuple(gaps)


def _task_requirements(instruction: str) -> tuple[TerminalRequirement, ...]:
    """Derive stable requirement IDs and conservatively classify explicit sections."""

    units: list[tuple[str, TerminalRequirementKind, str | None]] = []
    paragraph_lines: list[str] = []
    current_kind = TerminalRequirementKind.UNCLASSIFIED
    current_section: str | None = None

    def flush() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(paragraph_lines).strip()
        paragraph_lines.clear()
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            normalized = " ".join(sentence.split())
            if normalized:
                units.append(
                    (normalized[:2000], current_kind, current_section)
                )

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
        heading_match = _MARKDOWN_HEADING.match(raw_line)
        if heading_match is not None:
            flush()
            section = heading_match.group(1).strip().rstrip(":").strip()
            current_kind = _REQUIREMENT_SECTION_KINDS.get(
                section.casefold(),
                TerminalRequirementKind.UNCLASSIFIED,
            )
            current_section = (
                section
                if current_kind is not TerminalRequirementKind.UNCLASSIFIED
                else None
            )
            continue
        if stripped.startswith("#"):
            flush()
            current_kind = TerminalRequirementKind.UNCLASSIFIED
            current_section = None
            continue
        paragraph_lines.append(stripped)
    flush()
    if not units:
        units.append(
            (
                "Complete the task exactly as instructed.",
                TerminalRequirementKind.UNCLASSIFIED,
                None,
            )
        )
    return tuple(
        TerminalRequirement(
            requirement_id=f"req-{index:03d}",
            description=description,
            kind=kind,
            source_section=source_section,
        )
        for index, (description, kind, source_section) in enumerate(units, start=1)
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
