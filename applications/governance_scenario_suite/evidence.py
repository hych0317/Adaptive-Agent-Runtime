"""Internal evidence bundle and safe persisted scenario results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, StrictInt

from applications.ecommerce_support.models import (
    DomainSnapshot,
    ExternalLedgerRecord,
)
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ReasonCode,
    ScenarioContractModel,
    ScenarioProfile,
    ScenarioSpec,
    ScenarioVerdict,
)


class EvidenceSource(StrEnum):
    AUTHORITATIVE_STATE = "authoritative_state"
    EXTERNAL_LEDGER = "external_ledger"
    AUDIT = "audit"
    MODEL_CONTEXT = "model_context"
    MEMORY = "memory"


class MemoryEvidenceView(ScenarioContractModel):
    candidate_canaries: tuple[str, ...] = ()
    bundle_canaries: tuple[str, ...] = ()
    memory_keys: tuple[str, ...] = ()
    revision_count: StrictInt = Field(default=0, ge=0)

    @property
    def all_canaries(self) -> tuple[str, ...]:
        return (*self.candidate_canaries, *self.bundle_canaries)


class ScenarioExecution(ScenarioContractModel):
    decision: ScenarioVerdict
    reason_code: ReasonCode | None = None
    model_contexts: tuple[str, ...] = ()
    memory: MemoryEvidenceView = Field(default_factory=MemoryEvidenceView)
    available_sources: frozenset[EvidenceSource] = frozenset()
    model_call_count: StrictInt = Field(default=0, ge=0)
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceBundle:
    """Full in-memory evidence; sensitive raw values are never serialized directly."""

    scenario: ScenarioSpec
    execution: ScenarioExecution
    pre_state: DomainSnapshot
    post_state: DomainSnapshot
    external_ledger: tuple[ExternalLedgerRecord, ...]
    available_sources: frozenset[EvidenceSource]


class EvidenceSummary(ScenarioContractModel):
    pre_state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_ledger_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_context_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_effect_count: StrictInt = Field(ge=0)
    external_attempt_count: StrictInt = Field(ge=0)
    audit_event_types: tuple[str, ...] = ()
    audit_reason_codes: tuple[str, ...] = ()
    available_sources: frozenset[EvidenceSource]


class OracleFinding(ScenarioContractModel):
    code: str = Field(min_length=1)
    verdict: EvaluationVerdict
    message: str = Field(min_length=1)
    source: EvidenceSource | None = None
    details: Mapping[str, JsonValue] = Field(default_factory=dict)


class ScenarioRunResult(ScenarioContractModel):
    schema_version: int = Field(default=1, ge=1)
    run_id: UUID
    scenario_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    profile: ScenarioProfile
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_verdict: EvaluationVerdict
    actual_decision: ScenarioVerdict
    actual_reason_code: ReasonCode | None = None
    findings: tuple[OracleFinding, ...]
    evidence: EvidenceSummary
    metrics: Mapping[str, JsonValue] = Field(default_factory=dict)
    started_at: AwareDatetime
    completed_at: AwareDatetime


class ScenarioBatchResult(ScenarioContractModel):
    schema_version: int = Field(default=1, ge=1)
    suite_id: str = Field(min_length=1)
    results: tuple[ScenarioRunResult, ...]
    generated_at: AwareDatetime

    @property
    def release_gate_passed(self) -> bool:
        return bool(self.results) and all(
            item.evaluation_verdict is EvaluationVerdict.PASS
            for item in self.results
        )
