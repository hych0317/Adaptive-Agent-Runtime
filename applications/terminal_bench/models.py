"""Harbor-independent domain models for the sequential terminal profile."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from adaptive_agent_runtime.core import ActionRequest
from adaptive_agent_runtime.llm import InferenceUsage

from applications.terminal_bench.deadline import TerminalDeadlineSequence


AAR_TERMINAL_SEQUENTIAL_PROFILE = "AAR Terminal Sequential Profile"
TERMINAL_COMMAND_ACTION = "terminal.command"
TERMINAL_COMPLETION_REJECTION_ACTION = "terminal.completion_rejection"
TERMINAL_PROPOSAL_REJECTION_ACTION = "terminal.proposal_rejection"
TERMINAL_COMMAND_CAPABILITY = "terminal.command.execute"
TERMINAL_COMMAND_PROVIDER = "harbor.environment.exec"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def terminal_fingerprint(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TerminalModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class TerminalExecutionState(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED_TO_START = "FAILED_TO_START"
    IN_DOUBT = "IN_DOUBT"


class TerminalTurnDecision(StrEnum):
    EXECUTE = "execute"
    COMPLETE = "complete"


class TerminalCommandRole(StrEnum):
    """Planner-declared role used by the Runtime completion gate."""

    INSPECT = "inspect"
    WORK = "work"
    VERIFY = "verify"


class TerminalTimeoutCapReason(StrEnum):
    """Why the authoritative applied timeout differs from the draft."""

    MODEL_REQUESTED = "model_requested"
    RUNTIME_DEFAULT = "runtime_default"
    ADVERTISED_CAP = "advertised_cap"
    FRESH_DEADLINE_CAP = "fresh_deadline_cap"
    LEGACY_UNSPECIFIED = "legacy_unspecified"


class TerminalVerificationEvidence(StrEnum):
    OFFICIAL_TESTS = "official_tests"
    INDEPENDENT_CHECK = "independent_check"


class TerminalEvidenceProvenance(StrEnum):
    """Where verification evidence originated, independent of its result."""

    TASK_PROVIDED = "task_provided"
    EXTERNAL_STANDARD = "external_standard"
    RUNTIME_OBSERVED = "runtime_observed"
    AGENT_GENERATED = "agent_generated"
    LEGACY_UNSPECIFIED = "legacy_unspecified"


class TerminalEvidenceAssurance(StrEnum):
    """Runtime assurance attached to a requirement outcome."""

    NONE = "none"
    SELF_CHECKED = "self_checked"
    TRUSTED = "trusted"


class TerminalCompletionDisposition(StrEnum):
    IN_PROGRESS = "in_progress"
    SUCCESS_LOCKED = "success_locked"
    SUBMITTED_UNVERIFIED = "submitted_unverified"
    SUBMITTED_KNOWN_FAILED = "submitted_known_failed"


class TerminalReconciliationState(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    STABLE_UNVERIFIED = "stable_unverified"


class TerminalVerificationStatePolicy(StrEnum):
    READ_ONLY = "read_only"


class TerminalVerificationIndependence(StrEnum):
    """How verification avoids reusing the implementation's own oracle."""

    INDEPENDENT_ORACLE = "independent_oracle"
    CROSS_IMPLEMENTATION = "cross_implementation"
    PROPERTY_BASED = "property_based"
    ALTERNATE_EVIDENCE = "alternate_evidence"
    OFFICIAL_TESTS = "official_tests"


class TerminalVerificationIsolation(StrEnum):
    """Process-state isolation promised by a verification command."""

    FRESH_PROCESS = "fresh_process"
    EPHEMERAL_FIXTURE = "ephemeral_fixture"


class TerminalPerformanceProtocol(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    COLD_UNIQUE_INPUTS = "cold_unique_inputs"


class TerminalVerificationDimension(StrEnum):
    """Minimum independent dimensions needed before a task can complete."""

    ARTIFACT = "artifact"
    FORMAT = "format"
    SEMANTIC = "semantic"
    END_TO_END = "end_to_end"


class TerminalRequirementKind(StrEnum):
    """Conservative classification derived only from explicit task sections."""

    ACHIEVE = "achieve"
    PRODUCE = "produce"
    PRESERVE = "preserve"
    PROHIBIT = "prohibit"
    THRESHOLD = "threshold"
    UNCLASSIFIED = "unclassified"


class TerminalRequirementState(StrEnum):
    """Shadow evidence state; it is not an execution authorization."""

    UNKNOWN = "unknown"
    SATISFIED = "satisfied"
    INCONCLUSIVE = "inconclusive"
    # Kept only so older transcripts remain parseable; new receipts use SATISFIED.
    VERIFIED = "verified"
    BLOCKED = "blocked"


class TerminalRequirement(TerminalModel):
    """One stable, Runtime-derived verification requirement."""

    requirement_id: str = Field(pattern=r"^req-[0-9]{3,}$")
    description: str = Field(min_length=1, max_length=2000)
    kind: TerminalRequirementKind = TerminalRequirementKind.UNCLASSIFIED
    source_section: str | None = Field(default=None, min_length=1, max_length=256)


class TerminalRequirementLedgerEntry(TerminalModel):
    requirement_id: str = Field(pattern=r"^req-[0-9]{3,}$")
    state: TerminalRequirementState = TerminalRequirementState.UNKNOWN
    assurance: TerminalEvidenceAssurance = TerminalEvidenceAssurance.NONE
    evidence_provenance: TerminalEvidenceProvenance = (
        TerminalEvidenceProvenance.LEGACY_UNSPECIFIED
    )
    artifact_fingerprints: tuple[str, ...] = ()
    latest_evidence_action_id: UUID | None = None
    latest_result_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_generation: int | None = Field(default=None, ge=0)


class TerminalReconciliationReceipt(TerminalModel):
    """Runtime-observed proof that an uncertain process is no longer active."""

    action_id: UUID
    task_generation: int = Field(default=0, ge=0)
    result_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: AwareDatetime


class TerminalLedgerProjectionEntry(TerminalModel):
    """Bounded model-facing delta; full evidence remains in the Journal."""

    requirement_id: str = Field(pattern=r"^req-[0-9]{3,}$")
    state: TerminalRequirementState
    assurance: TerminalEvidenceAssurance
    evidence_provenance: TerminalEvidenceProvenance
    result_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    changed_in_generation: int | None = Field(default=None, ge=0)


class TerminalTaskContract(TerminalModel):
    """Immutable task contract persisted once at trial start."""

    contract_version: str = Field(default="1", min_length=1, max_length=32)
    contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirements: tuple[TerminalRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> TerminalTaskContract:
        requirement_ids = tuple(
            item.requirement_id for item in self.requirements
        )
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("task contract requirement IDs must be unique")
        expected_fingerprint = terminal_fingerprint(
            {
                "contract_version": self.contract_version,
                "requirements": [
                    item.model_dump(mode="json")
                    for item in self.requirements
                ],
            }
        )
        if self.contract_fingerprint != expected_fingerprint:
            raise ValueError("task contract fingerprint does not match requirements")
        return self


class TerminalTaskLedger(TerminalModel):
    """Compact, journal-persisted shadow view of requirement evidence."""

    contract_version: str = Field(default="1", min_length=1, max_length=32)
    contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(default=0, ge=0)
    entries: tuple[TerminalRequirementLedgerEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> TerminalTaskLedger:
        identifiers = tuple(item.requirement_id for item in self.entries)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("task ledger requirement IDs must be unique")
        return self


class TerminalVerificationContract(TerminalModel):
    """Auditable evidence contract for a model-proposed verification."""

    evidence_kind: TerminalVerificationEvidence
    evidence_sources: tuple[str, ...] = Field(min_length=1)
    artifact_paths: tuple[str, ...] = Field(min_length=1)
    requirement_coverage: tuple[str, ...] = Field(min_length=1)
    coverage_dimensions: tuple[TerminalVerificationDimension, ...] = ()
    evidence_provenance: TerminalEvidenceProvenance = (
        TerminalEvidenceProvenance.LEGACY_UNSPECIFIED
    )
    artifact_fingerprints: tuple[str, ...] = ()
    validation_methods: tuple[str, ...] = Field(min_length=1)
    state_policy: TerminalVerificationStatePolicy = (
        TerminalVerificationStatePolicy.READ_ONLY
    )
    independence_method: TerminalVerificationIndependence = (
        TerminalVerificationIndependence.INDEPENDENT_ORACLE
    )
    process_isolation: TerminalVerificationIsolation = (
        TerminalVerificationIsolation.FRESH_PROCESS
    )
    performance_protocol: TerminalPerformanceProtocol = (
        TerminalPerformanceProtocol.NOT_APPLICABLE
    )

    @model_validator(mode="after")
    def validate_coverage(self) -> TerminalVerificationContract:
        collections = (
            self.evidence_sources,
            self.artifact_paths,
            self.requirement_coverage,
            self.coverage_dimensions,
            self.validation_methods,
            self.artifact_fingerprints,
        )
        if any(len(set(items)) != len(items) for items in collections):
            raise ValueError("verification evidence entries must be unique")
        if self.evidence_kind is TerminalVerificationEvidence.INDEPENDENT_CHECK:
            required = set(TerminalVerificationDimension)
            missing = required.difference(self.coverage_dimensions)
            if missing:
                names = ", ".join(sorted(item.value for item in missing))
                raise ValueError(
                    "independent verification is missing coverage dimensions: "
                    + names
                )
        if any(
            re.fullmatch(r"[^=\s]{1,4096}=sha256:[0-9a-f]{64}", item) is None
            for item in self.artifact_fingerprints
        ):
            raise ValueError("artifact fingerprints must be path=sha256:<hex>")
        fingerprint_paths = {
            item.split("=sha256:", 1)[0] for item in self.artifact_fingerprints
        }
        if fingerprint_paths.difference(self.artifact_paths):
            raise ValueError(
                "artifact fingerprints must reference declared artifact paths"
            )
        return self


class TerminalVerificationReceipt(TerminalModel):
    """Runtime-observed receipt for one committed verification command."""

    action_id: UUID
    task_generation: int = Field(default=0, ge=0)
    task_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    covered_requirement_ids: tuple[str, ...] = Field(min_length=1)
    evidence_provenance: TerminalEvidenceProvenance = (
        TerminalEvidenceProvenance.LEGACY_UNSPECIFIED
    )
    assurance: TerminalEvidenceAssurance = TerminalEvidenceAssurance.NONE
    artifact_fingerprints: tuple[str, ...] = ()
    execution_state: TerminalExecutionState
    return_code: int | None = None
    timed_out: bool = False
    transport_failed: bool = False
    passed: bool
    observed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_observation(self) -> TerminalVerificationReceipt:
        if len(set(self.covered_requirement_ids)) != len(
            self.covered_requirement_ids
        ):
            raise ValueError("verification receipt coverage must be unique")
        expected_pass = bool(
            self.execution_state is TerminalExecutionState.COMPLETED
            and self.return_code == 0
            and not self.timed_out
            and not self.transport_failed
        )
        if self.passed != expected_pass:
            raise ValueError("verification receipt pass state is inconsistent")
        return self


class TerminalExecutionLimits(TerminalModel):
    default_timeout_sec: int = Field(default=120, ge=1)
    max_timeout_sec: int = Field(default=300, ge=1)
    max_work_timeout_sec: int = Field(default=60, ge=1)
    max_verification_timeout_sec: int = Field(default=120, ge=1)
    max_inspection_timeout_sec: int | None = Field(default=None, ge=1)
    deadline_sequence: TerminalDeadlineSequence | None = None
    max_inference_timeout_sec: float | None = Field(default=None, ge=0.0)
    final_repair_timeout_sec: int = Field(default=60, ge=1)
    cleanup_grace_seconds: float = Field(default=10.0, ge=0.0)
    timeout_admission_margin_seconds: float = Field(default=8.0, ge=0.0)
    followup_inference_reserve_seconds: float = Field(default=0.0, ge=0.0)
    verification_reserve_seconds: float = Field(default=0.0, ge=0.0)
    deadline_cleanup_reserve_seconds: float = Field(default=0.0, ge=0.0)
    max_command_characters: int = Field(default=20_000, ge=1)
    max_environment_variables: int = Field(default=64, ge=0)
    max_environment_value_characters: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def validate_timeouts(self) -> TerminalExecutionLimits:
        if self.default_timeout_sec > self.max_timeout_sec:
            raise ValueError("default timeout cannot exceed maximum timeout")
        if self.max_verification_timeout_sec > self.max_timeout_sec:
            raise ValueError(
                "verification timeout cannot exceed maximum timeout"
            )
        if self.max_work_timeout_sec > self.max_timeout_sec:
            raise ValueError("work timeout cannot exceed maximum timeout")
        if (
            self.max_inspection_timeout_sec is not None
            and self.max_inspection_timeout_sec > self.max_timeout_sec
        ):
            raise ValueError("inspection timeout cannot exceed maximum timeout")
        if self.final_repair_timeout_sec > self.max_timeout_sec:
            raise ValueError(
                "final repair timeout cannot exceed maximum timeout"
            )
        return self


class TerminalToolCapabilities(TerminalModel):
    shell_exec: bool = True
    apply_patch: bool = False
    portable_file_edit_methods: tuple[str, ...] = (
        "python3 script with explicit path and atomic replace",
        "sed/awk/perl after feature detection",
        "shell heredoc only when quoting and overwrite scope are explicit",
    )


class TerminalRejectedDraft(TerminalModel):
    decision: TerminalTurnDecision
    call_key: str | None = Field(default=None, max_length=256)
    command_preview: str | None = Field(default=None, max_length=512)
    command_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    command_role: TerminalCommandRole | None = None
    cwd: str | None = Field(default=None, max_length=4096)
    environment_keys: tuple[str, ...] = ()
    timeout_sec: int | None = Field(default=None, ge=1)


class TerminalProposalRejection(TerminalModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1024)
    field: str = Field(min_length=1, max_length=128)
    rejected_value: str | None = Field(default=None, max_length=512)
    expected: str = Field(min_length=1, max_length=1024)
    draft: TerminalRejectedDraft


class TerminalVerifiedCheckpoint(TerminalModel):
    action_id: UUID
    task_generation: int = Field(default=0, ge=0)
    call_key: str = Field(min_length=1, max_length=256)
    command_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification: TerminalVerificationContract
    receipt: TerminalVerificationReceipt | None = None
    committed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_receipt(self) -> TerminalVerifiedCheckpoint:
        if self.receipt is not None:
            if self.receipt.action_id != self.action_id:
                raise ValueError("checkpoint receipt action does not match")
            if self.receipt.task_generation != self.task_generation:
                raise ValueError("checkpoint receipt generation does not match")
            if self.receipt.command_fingerprint != self.command_fingerprint:
                raise ValueError("checkpoint receipt command does not match")
            if self.receipt.verification_contract_fingerprint != terminal_fingerprint(
                self.verification
            ):
                raise ValueError("checkpoint receipt verification does not match")
            if self.receipt.covered_requirement_ids != (
                self.verification.requirement_coverage
            ):
                raise ValueError("checkpoint receipt coverage does not match")
            if not self.receipt.passed:
                raise ValueError("checkpoint receipt must represent a pass")
        return self


class TerminalProcessReference(TerminalModel):
    """Explicit evidence needed to inspect a detached background service."""

    reference_id: str = Field(min_length=1, max_length=128)
    pid_file: str = Field(min_length=1, max_length=4096)
    log_path: str = Field(min_length=1, max_length=4096)
    status_check_command: str = Field(min_length=1, max_length=20_000)
    stop_command: str | None = Field(default=None, min_length=1, max_length=20_000)


class TerminalRuntimeVerificationEvidence(TerminalModel):
    """Runtime-owned observation of evidence sources and produced artifacts."""

    requested_provenance: TerminalEvidenceProvenance
    provenance_verified: bool = False
    evidence_source_fingerprints: tuple[str, ...] = ()
    artifact_fingerprints: tuple[str, ...] = ()
    observed_at: AwareDatetime
    failure_reason: str | None = Field(default=None, min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_runtime_evidence(self) -> TerminalRuntimeVerificationEvidence:
        fingerprint_pattern = re.compile(r"^[^=\s]{1,4096}=sha256:[0-9a-f]{64}$")
        if any(
            fingerprint_pattern.fullmatch(item) is None
            for item in (
                *self.evidence_source_fingerprints,
                *self.artifact_fingerprints,
            )
        ):
            raise ValueError("Runtime evidence fingerprints must be path=sha256:<hex>")
        if self.provenance_verified and self.requested_provenance not in {
            TerminalEvidenceProvenance.TASK_PROVIDED,
            TerminalEvidenceProvenance.EXTERNAL_STANDARD,
        }:
            raise ValueError("only trusted provenance classes can be Runtime-verified")
        if self.provenance_verified and not self.evidence_source_fingerprints:
            raise ValueError("verified provenance requires source fingerprints")
        return self


class TerminalExecResult(TerminalModel):
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: int = Field(ge=0)
    execution_state: TerminalExecutionState
    timed_out: bool = False
    transport_failed: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    runtime_verification: TerminalRuntimeVerificationEvidence | None = None

    @model_validator(mode="after")
    def validate_execution_result(self) -> TerminalExecResult:
        if self.completed_at < self.started_at:
            raise ValueError("terminal completion cannot precede start")
        if self.execution_state is TerminalExecutionState.COMPLETED:
            if self.return_code is None:
                raise ValueError("completed terminal execution requires return_code")
            if self.transport_failed:
                raise ValueError("completed execution cannot be a transport failure")
        elif self.return_code is not None:
            raise ValueError("uncertain or unstarted execution cannot have return_code")
        if (
            self.execution_state is TerminalExecutionState.FAILED_TO_START
            and self.timed_out
        ):
            raise ValueError("a command known not to start cannot time out")
        return self

    @property
    def command_completed(self) -> bool:
        return self.execution_state is TerminalExecutionState.COMPLETED

    @property
    def settled(self) -> bool:
        """Whether execution has a known final status without Runtime failure."""

        return bool(
            self.execution_state is TerminalExecutionState.COMPLETED
            and not self.timed_out
            and not self.transport_failed
        )

    @property
    def succeeded(self) -> bool:
        """Whether the command settled and returned a successful exit status."""

        return bool(self.settled and self.return_code == 0)


class TerminalExecutionPolicy(TerminalModel):
    max_commands: int = Field(default=64, ge=1, le=512)
    default_timeout_sec: int = Field(default=120, ge=1)
    max_timeout_sec: int = Field(default=300, ge=1)
    max_verification_timeout_sec: int = Field(default=120, ge=1)
    provider_grace_sec: int = Field(default=10, ge=1, le=120)
    max_command_characters: int = Field(default=20_000, ge=1)
    max_output_characters: int = Field(default=64_000, ge=1)
    max_context_output_characters: int = Field(default=3_000, ge=1)
    max_context_records: int = Field(default=4, ge=1)
    max_delivery_context_output_characters: int = Field(default=1_000, ge=1)
    max_delivery_context_records: int = Field(default=2, ge=1)
    max_environment_variables: int = Field(default=64, ge=0)
    max_environment_value_characters: int = Field(default=4096, ge=1)
    max_total_tokens: int | None = Field(default=600_000, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    max_wall_clock_seconds: float | None = Field(default=None, gt=0.0)
    max_active_execution_seconds: float | None = Field(default=None, gt=0.0)
    external_job_deadline_seconds: float | None = Field(default=None, gt=0.0)
    cleanup_grace_seconds: float = Field(default=10.0, ge=0.0)
    timeout_admission_margin_seconds: float = Field(default=8.0, ge=0.0)
    repeated_invocation_limit: int | None = Field(default=3, ge=2)
    max_no_progress_steps: int | None = Field(default=5, ge=1)
    max_no_progress_seconds: float | None = Field(default=480.0, gt=0.0)
    deadline_reserve_seconds: float = Field(default=60.0, ge=0.0)
    delivery_mode_fraction: float = Field(default=0.40, gt=0.0, lt=1.0)
    finalization_mode_threshold_seconds: float = Field(default=300.0, gt=0.0)
    finalization_reserve_seconds: float = Field(default=190.0, ge=0.0)
    final_repair_timeout_sec: int = Field(default=60, ge=1)
    max_artifact_first_inspections: int | None = Field(
        default=1,
        ge=1,
        le=4,
    )
    max_consecutive_inspections: int | None = Field(default=3, ge=1, le=16)
    max_total_inspections: int | None = Field(default=5, ge=1, le=64)
    max_completion_rejections: int = Field(default=2, ge=0, le=10)
    max_proposal_rejections: int = Field(default=2, ge=0, le=10)
    max_reconciliation_proposal_rejections: int = Field(
        default=2,
        ge=0,
        le=10,
    )
    max_in_doubt_reconciliation_attempts: int = Field(default=2, ge=1, le=8)

    @model_validator(mode="after")
    def validate_timeouts(self) -> TerminalExecutionPolicy:
        if self.default_timeout_sec > self.max_timeout_sec:
            raise ValueError("default timeout cannot exceed maximum timeout")
        if (
            self.finalization_mode_threshold_seconds
            < self.finalization_reserve_seconds
        ):
            raise ValueError(
                "finalization threshold cannot be below its reserved budget"
            )
        return self


class TerminalSessionSnapshot(TerminalModel):
    trial_id: str = Field(min_length=1)
    current_cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    environment: dict[str, str] = Field(default_factory=dict)
    process_references: tuple[TerminalProcessReference, ...] = ()
    committed_commands: int = Field(default=0, ge=0)
    denied_commands: int = Field(default=0, ge=0)
    timed_out_commands: int = Field(default=0, ge=0)
    in_doubt_commands: int = Field(default=0, ge=0)
    in_doubt_reconciliation_required: bool = False
    completion_rejections: int = Field(default=0, ge=0)
    completion_blocker: str | None = Field(default=None, min_length=1)
    proposal_rejections: int = Field(default=0, ge=0)
    proposal_blocker: str | None = Field(default=None, min_length=1)
    last_proposal_rejection: TerminalProposalRejection | None = None
    reconciliation_proposal_rejections: int = Field(default=0, ge=0)
    reconciliation_blocker: str | None = Field(default=None, min_length=1)
    last_reconciliation_proposal_rejection: TerminalProposalRejection | None = None
    reconciliation_state: TerminalReconciliationState = (
        TerminalReconciliationState.NOT_REQUIRED
    )
    latest_reconciliation_receipt: TerminalReconciliationReceipt | None = None
    consecutive_inspections: int = Field(default=0, ge=0)
    inspection_commands: int = Field(default=0, ge=0)
    failed_verification_attempts: int = Field(default=0, ge=0)
    verification_corrections: int = Field(default=0, ge=0)
    task_generation: int = Field(default=0, ge=0)
    known_state_generation: int | None = Field(default=0, ge=0)
    successful_work_generation: int | None = Field(default=None, ge=0)
    pending_repair_receipt_id: UUID | None = None
    repair_applied_action_id: UUID | None = None
    latest_failure_signatures: tuple[str, ...] = ()
    started_at: AwareDatetime = Field(default_factory=utc_now)
    task_ledger: TerminalTaskLedger | None = None
    latest_verification_receipt: TerminalVerificationReceipt | None = None
    verified_checkpoint: TerminalVerifiedCheckpoint | None = None
    contract_coverage_complete: bool = True
    contract_unmapped_fragments: tuple[str, ...] = ()
    completion_disposition: TerminalCompletionDisposition = (
        TerminalCompletionDisposition.IN_PROGRESS
    )
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def migrate_reconciliation_state(cls, value: object) -> object:
        if isinstance(value, dict):
            migrated = dict(value)
            if (
                migrated.get("in_doubt_reconciliation_required") is True
                and "reconciliation_state" not in migrated
            ):
                migrated["reconciliation_state"] = (
                    TerminalReconciliationState.REQUIRED.value
                )
            if (
                "known_state_generation" not in migrated
                and not migrated.get("in_doubt_reconciliation_required", False)
            ):
                task_generation = migrated.get("task_generation", 0)
                migrated["known_state_generation"] = task_generation
            return migrated
        return value

    @model_validator(mode="after")
    def validate_verification_state(self) -> TerminalSessionSnapshot:
        receipt = self.latest_verification_receipt
        ledger = self.task_ledger
        checkpoint = self.verified_checkpoint
        if ledger is not None and ledger.generation != self.task_generation:
            raise ValueError("task ledger generation does not match session")
        if (
            self.known_state_generation is not None
            and self.known_state_generation != self.task_generation
        ):
            raise ValueError("known state must belong to current generation")
        if (
            self.successful_work_generation is not None
            and self.successful_work_generation != self.task_generation
        ):
            raise ValueError("successful work must belong to current generation")
        if (
            self.repair_applied_action_id is not None
            and self.pending_repair_receipt_id is None
        ):
            raise ValueError("applied repair requires a pending repair receipt")
        if self.pending_repair_receipt_id is not None:
            if (
                receipt is not None
                and receipt.action_id == self.pending_repair_receipt_id
                and receipt.passed
            ):
                raise ValueError("a passed receipt cannot remain pending repair")
        if receipt is not None:
            if ledger is None:
                raise ValueError("verification receipt requires a task ledger")
            if receipt.task_contract_fingerprint != ledger.contract_fingerprint:
                raise ValueError("verification receipt task contract does not match")
        if checkpoint is not None and checkpoint.receipt is not None:
            if checkpoint.task_generation != self.task_generation:
                raise ValueError("checkpoint generation does not match session")
            if receipt != checkpoint.receipt:
                raise ValueError(
                    "checkpoint receipt must equal the latest verification receipt"
                )
        reconciliation_receipt = self.latest_reconciliation_receipt
        if self.reconciliation_state is TerminalReconciliationState.REQUIRED:
            if not self.in_doubt_reconciliation_required:
                raise ValueError("required reconciliation must remain active")
        elif self.in_doubt_reconciliation_required:
            raise ValueError("active reconciliation must use REQUIRED state")
        if self.reconciliation_state is TerminalReconciliationState.STABLE_UNVERIFIED:
            if reconciliation_receipt is None:
                raise ValueError("stable reconciliation requires a receipt")
            if reconciliation_receipt.task_generation != self.task_generation:
                raise ValueError("reconciliation receipt belongs to a stale generation")
        if (
            self.completion_disposition is TerminalCompletionDisposition.SUCCESS_LOCKED
            and checkpoint is None
        ):
            raise ValueError("success_locked requires a verified checkpoint")
        if (
            self.completion_disposition
            in {
                TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                TerminalCompletionDisposition.SUBMITTED_KNOWN_FAILED,
            }
        ):
            if self.in_doubt_reconciliation_required:
                raise ValueError(
                    "submitted state cannot retain unresolved IN_DOUBT state"
                )
            if self.known_state_generation != self.task_generation:
                raise ValueError(
                    "submitted state requires a known current generation"
                )
            if (
                receipt is not None
                and receipt.passed
                and receipt.assurance is TerminalEvidenceAssurance.TRUSTED
                and self.contract_coverage_complete
                and ledger is not None
                and set(receipt.covered_requirement_ids)
                == {item.requirement_id for item in ledger.entries}
                and all(
                    item.state is TerminalRequirementState.SATISFIED
                    and item.assurance is TerminalEvidenceAssurance.TRUSTED
                    and item.evidence_generation == self.task_generation
                    for item in ledger.entries
                )
            ):
                raise ValueError(
                    "trusted evidence must not use a submitted disposition"
                )
        if (
            self.completion_disposition
            is TerminalCompletionDisposition.SUBMITTED_KNOWN_FAILED
        ):
            if (
                receipt is None
                or self.pending_repair_receipt_id is None
                or receipt.action_id != self.pending_repair_receipt_id
                or receipt.passed
                or receipt.execution_state is not TerminalExecutionState.COMPLETED
                or receipt.return_code in (None, 0)
                or receipt.timed_out
                or receipt.transport_failed
                or not self.latest_failure_signatures
            ):
                raise ValueError(
                    "submitted_known_failed requires a settled unresolved "
                    "verification failure"
                )
        if self.contract_coverage_complete and self.contract_unmapped_fragments:
            raise ValueError(
                "complete task contract cannot retain unmapped fragments"
            )
        if not self.contract_coverage_complete and not self.contract_unmapped_fragments:
            raise ValueError(
                "incomplete task contract requires unmapped fragments"
            )
        return self


class TerminalHistoryItem(TerminalModel):
    action_id: UUID
    call_key: str = Field(min_length=1)
    command: str = Field(min_length=1)
    command_role: TerminalCommandRole = TerminalCommandRole.WORK
    cwd: str | None = None
    environment_keys: tuple[str, ...] = ()
    timeout_sec: int = Field(ge=1)
    requested_timeout_sec: int | None = Field(default=None, ge=1)
    advertised_timeout_cap_sec: int | None = Field(default=None, ge=1)
    timeout_cap_reason: TerminalTimeoutCapReason = (
        TerminalTimeoutCapReason.LEGACY_UNSPECIFIED
    )
    verification: TerminalVerificationContract | None = None
    execution_state: TerminalExecutionState
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    transport_failed: bool = False


class TerminalTurnRequest(TerminalModel):
    run_id: UUID
    task_id: UUID
    instruction: str = Field(min_length=1)
    profile: str = AAR_TERMINAL_SEQUENTIAL_PROFILE
    requirements: tuple[TerminalRequirement, ...] = Field(min_length=1)
    session: TerminalSessionSnapshot
    ledger_projection: tuple[TerminalLedgerProjectionEntry, ...] = ()
    behavior_hints: tuple[str, ...] = ()
    recent_history: tuple[TerminalHistoryItem, ...] = ()
    used_call_keys: tuple[str, ...] = ()
    execution_limits: TerminalExecutionLimits = Field(
        default_factory=TerminalExecutionLimits
    )
    tool_capabilities: TerminalToolCapabilities = Field(
        default_factory=TerminalToolCapabilities
    )
    remaining_commands: int = Field(ge=0)
    remaining_tokens: int | None = Field(default=None, ge=0)
    remaining_cost_usd: float | None = Field(default=None, ge=0.0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    remaining_wall_clock_seconds: float | None = Field(default=None, ge=0.0)
    delivery_mode: bool = False
    emergency_mode: bool = False
    finalization_mode: bool = False
    reconciliation_mode: bool = False
    recovery_mode: bool = False
    artifact_first_mode: bool = False
    repair_mode: bool = False
    verification_due: bool = False
    execution_semantics: tuple[str, ...] = Field(min_length=1)


class TerminalTurnDraft(TerminalModel):
    """Authority-free model draft for one command or voluntary completion."""

    decision: TerminalTurnDecision
    call_key: str | None = Field(default=None, min_length=1, max_length=256)
    command: str | None = Field(default=None, min_length=1)
    command_role: TerminalCommandRole | None = None
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    env: dict[str, str] | None = None
    timeout_sec: int | None = Field(default=None, ge=1)
    process_reference: TerminalProcessReference | None = None
    verification: TerminalVerificationContract | None = None
    summary: str | None = Field(default=None, min_length=1)
    rationale: str = Field(
        default="Model-proposed terminal turn.",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_turn(self) -> TerminalTurnDraft:
        execution_fields = (self.call_key, self.command)
        if self.decision is TerminalTurnDecision.EXECUTE:
            if any(item is None for item in execution_fields):
                raise ValueError("execute draft requires call_key and command")
            if self.command_role is None:
                raise ValueError("execute draft requires command_role")
            if self.summary is not None:
                raise ValueError("execute draft cannot contain completion summary")
        else:
            if any(item is not None for item in execution_fields):
                raise ValueError("complete draft cannot contain a command identity")
            if any(
                item is not None
                for item in (
                    self.cwd,
                    self.env,
                    self.timeout_sec,
                    self.process_reference,
                    self.command_role,
                    self.verification,
                )
            ):
                raise ValueError("complete draft cannot contain execution settings")
            if self.summary is None:
                raise ValueError("complete draft requires summary")
        return self


class TerminalTurnProposal(TerminalModel):
    draft: TerminalTurnDraft
    usage: InferenceUsage = Field(default_factory=InferenceUsage)
    model_id: str | None = Field(default=None, min_length=1)


class TerminalCommandIntent(TerminalModel):
    """Runtime-resolved, fully explicit command carried by one Core Action."""

    trial_id: str = Field(min_length=1)
    call_key: str = Field(min_length=1, max_length=256)
    command: str = Field(min_length=1)
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: int = Field(ge=1)
    requested_timeout_sec: int | None = Field(default=None, ge=1)
    advertised_timeout_cap_sec: int | None = Field(default=None, ge=1)
    timeout_cap_reason: TerminalTimeoutCapReason = (
        TerminalTimeoutCapReason.LEGACY_UNSPECIFIED
    )
    command_role: TerminalCommandRole = TerminalCommandRole.WORK
    process_reference: TerminalProcessReference | None = None
    verification: TerminalVerificationContract | None = None

    @model_validator(mode="after")
    def validate_timeout_audit(self) -> TerminalCommandIntent:
        if (
            self.advertised_timeout_cap_sec is not None
            and self.timeout_sec > self.advertised_timeout_cap_sec
        ):
            raise ValueError("applied timeout cannot exceed advertised cap")
        return self

    def tool_arguments(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "command": self.command,
            "command_role": self.command_role.value,
            "cwd": self.cwd,
            "env": dict(self.env),
            "timeout_sec": self.timeout_sec,
            "process_reference": (
                self.process_reference.model_dump(mode="json")
                if self.process_reference is not None
                else None
            ),
            "verification": (
                self.verification.model_dump(mode="json")
                if self.verification is not None
                else None
            ),
        }

    @property
    def execution_fingerprint(self) -> str:
        return terminal_fingerprint(self.tool_arguments())


class TerminalPendingCommand(TerminalModel):
    action: ActionRequest
    intent: TerminalCommandIntent
    state_revision: int = Field(ge=0)
    proposal: TerminalTurnDraft
    created_at: AwareDatetime = Field(default_factory=utc_now)


class TerminalCommandRecord(TerminalModel):
    action_id: UUID
    invocation_id: UUID
    decision_request_id: UUID
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent: TerminalCommandIntent
    result: TerminalExecResult
    governance_status: str = Field(min_length=1)
    committed_at: AwareDatetime = Field(default_factory=utc_now)

    def history_item(self, *, output_limit: int) -> TerminalHistoryItem:
        def bounded(value: str) -> str:
            if len(value) <= output_limit:
                return value
            half = max(1, output_limit // 2)
            return value[:half] + "\n...[context truncated]...\n" + value[-half:]

        return TerminalHistoryItem(
            action_id=self.action_id,
            call_key=self.intent.call_key,
            command=self.intent.command,
            command_role=self.intent.command_role,
            cwd=self.intent.cwd,
            environment_keys=tuple(sorted(self.intent.env)),
            timeout_sec=self.intent.timeout_sec,
            requested_timeout_sec=self.intent.requested_timeout_sec,
            advertised_timeout_cap_sec=(
                self.intent.advertised_timeout_cap_sec
            ),
            timeout_cap_reason=self.intent.timeout_cap_reason,
            verification=self.intent.verification,
            execution_state=self.result.execution_state,
            return_code=self.result.return_code,
            stdout=bounded(self.result.stdout),
            stderr=bounded(self.result.stderr),
            timed_out=self.result.timed_out,
            transport_failed=self.result.transport_failed,
        )


class TerminalTrialSummary(TerminalModel):
    profile: str = AAR_TERMINAL_SEQUENTIAL_PROFILE
    trial_id: str = Field(min_length=1)
    run_id: UUID
    task_id: UUID
    agent_complete: bool
    completion_disposition: TerminalCompletionDisposition = (
        TerminalCompletionDisposition.IN_PROGRESS
    )
    runtime_status: str = Field(min_length=1)
    final_output: Any = None
    command_count: int = Field(ge=0)
    denial_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    in_doubt_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    trace_consistent: bool
    started_at: AwareDatetime
    completed_at: AwareDatetime


class TerminalVerifierTimeoutAttribution(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    TASK_ATTRIBUTABLE = "task_attributable"
    INCONCLUSIVE = "inconclusive"


class TerminalBenchmarkOutcome(StrEnum):
    BENCHMARK_PASS = "benchmark_pass"
    BENCHMARK_FAIL = "benchmark_fail"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    INCONCLUSIVE = "inconclusive"


class TerminalBenchmarkAnalysis(TerminalModel):
    trial_id: str = Field(min_length=1)
    verifier_rewards: dict[str, float] = Field(default_factory=dict)
    verifier_reward: float | None = None
    verifier_timeout_attribution: TerminalVerifierTimeoutAttribution | None = None
    benchmark_pass: bool
    agent_complete: bool
    completion_matches_verifier: bool
    outcome: TerminalBenchmarkOutcome
    infrastructure_error: bool = False
    infrastructure_error_reason: str | None = Field(default=None, min_length=1)
    command_count: int = Field(ge=0)
    denial_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    in_doubt_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    trace_consistent: bool
