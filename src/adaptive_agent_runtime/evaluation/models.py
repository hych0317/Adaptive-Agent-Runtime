"""Immutable, execution-agnostic models for Agent Evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Any, Mapping, Self, TypeAlias
from uuid import UUID, uuid4, uuid5

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


_EVALUATION_NAMESPACE = UUID("8f6b04bb-84c0-4cb5-9d8f-79bd0b0f5cf3")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_evaluation_id(*parts: object) -> UUID:
    """Derive stable result identifiers without mutable registries."""

    return uuid5(_EVALUATION_NAMESPACE, "|".join(str(part) for part in parts))


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
    if isinstance(value, (list, tuple)):
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


class EvaluationModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        values = self.model_dump(mode="python")
        if update:
            values.update(update)
        return self.__class__.model_validate(values)


def canonical_evaluation_json(value: object) -> str:
    """Serialize Evaluation identity inputs with stable mapping order."""

    if isinstance(value, BaseModel):
        serializable = value.model_dump(mode="json")
    else:
        serializable = _thaw_json(value)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class EvaluationComponent(StrEnum):
    RUNTIME = "runtime"
    ORCHESTRATION = "orchestration"
    TOOL = "tool"
    CONTEXT = "context"
    MEMORY = "memory"
    CONTEXT_MEMORY = "context_memory"


class TraceCategory(StrEnum):
    TASK = "task"
    TASK_GRAPH = "task_graph"
    NODE_EXECUTION = "node_execution"
    TOOL_CALL = "tool_call"
    CONTEXT_CHANGE = "context_change"
    MEMORY_OPERATION = "memory_operation"
    FINAL_RESULT = "final_result"
    RUNTIME = "runtime"


class TraceCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


class TraceOrdering(StrEnum):
    PARTIAL = "partial"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationScope(StrEnum):
    OUTCOME = "outcome"
    TRAJECTORY = "trajectory"
    COMPONENT = "component"


class EvaluationVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"


class EvaluationCorrelation(EvaluationModel):
    run_id: UUID
    task_id: UUID | None = None
    graph_id: UUID | None = None
    node_id: UUID | None = None
    action_id: UUID | None = None
    invocation_id: UUID | None = None
    context_id: UUID | None = None
    memory_id: UUID | None = None
    candidate_id: UUID | None = None


class EvaluationFact(EvaluationModel):
    fact_id: UUID = Field(default_factory=uuid4)
    component: EvaluationComponent
    category: TraceCategory
    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    occurred_at: AwareDatetime
    correlation: EvaluationCorrelation
    source_scope: str = Field(min_length=1)
    source_sequence: int | None = Field(default=None, ge=0)
    source_record_id: str = Field(min_length=1)
    payload: ImmutableJsonObject = Field(default_factory=dict)


class TraceCoverage(EvaluationModel):
    run_id: UUID
    component: EvaluationComponent
    completeness: TraceCompleteness
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_diagnostics(self) -> TraceCoverage:
        if len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("coverage diagnostics must be unique")
        return self


class TraceBatch(EvaluationModel):
    facts: tuple[EvaluationFact, ...] = ()
    coverage: tuple[TraceCoverage, ...] = ()

    @model_validator(mode="after")
    def validate_batch(self) -> TraceBatch:
        fact_ids = tuple(fact.fact_id for fact in self.facts)
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("trace batch fact ids must be unique")
        coverage_keys = tuple(
            (item.run_id, item.component) for item in self.coverage
        )
        if len(set(coverage_keys)) != len(coverage_keys):
            raise ValueError("trace batch coverage entries must be unique")
        return self


class AgentExecutionTrace(EvaluationModel):
    trace_id: UUID
    run_id: UUID
    task_id: UUID
    facts: tuple[EvaluationFact, ...]
    coverage: tuple[TraceCoverage, ...]
    ordering: TraceOrdering = TraceOrdering.PARTIAL
    collected_at: AwareDatetime

    @model_validator(mode="after")
    def validate_trace(self) -> AgentExecutionTrace:
        fact_ids = tuple(fact.fact_id for fact in self.facts)
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("agent trace fact ids must be unique")
        for fact in self.facts:
            if fact.correlation.run_id != self.run_id:
                raise ValueError("agent trace contains a fact from another run")
            if fact.correlation.task_id not in {None, self.task_id}:
                raise ValueError("agent trace contains a fact from another task")
        coverage_components = tuple(item.component for item in self.coverage)
        if len(set(coverage_components)) != len(coverage_components):
            raise ValueError("agent trace component coverage must be unique")
        if any(item.run_id != self.run_id for item in self.coverage):
            raise ValueError("agent trace contains coverage for another run")
        if self.facts and self.collected_at < max(
            fact.occurred_at for fact in self.facts
        ):
            raise ValueError("trace collection cannot precede its facts")
        return self

    def facts_for(
        self,
        *components: EvaluationComponent,
    ) -> tuple[EvaluationFact, ...]:
        allowed = set(components)
        return tuple(fact for fact in self.facts if fact.component in allowed)

    def coverage_for(
        self,
        component: EvaluationComponent,
    ) -> TraceCoverage | None:
        return next(
            (item for item in self.coverage if item.component is component),
            None,
        )


class EvaluationStateSnapshot(EvaluationModel):
    run_id: UUID
    task_id: UUID
    task_description: str = Field(min_length=1)
    status: ExecutionStatus
    revision: int = Field(ge=0)
    step_count: int = Field(ge=0)
    output: ImmutableJsonValue = None
    error: str | None = None
    captured_at: AwareDatetime

    @model_validator(mode="after")
    def validate_state(self) -> EvaluationStateSnapshot:
        if self.status is ExecutionStatus.FAILED and not self.error:
            raise ValueError("failed state snapshot requires an error")
        if self.status is not ExecutionStatus.FAILED and self.error is not None:
            raise ValueError("only failed state snapshots can contain an error")
        if self.status is not ExecutionStatus.COMPLETED and self.output is not None:
            raise ValueError("only completed state snapshots can contain output")
        return self


class ExecutionResultSnapshot(EvaluationModel):
    run_id: UUID
    task_id: UUID
    succeeded: bool
    output: ImmutableJsonValue = None
    error: str | None = None
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_result(self) -> ExecutionResultSnapshot:
        if self.succeeded and self.error is not None:
            raise ValueError("successful execution result cannot contain an error")
        if not self.succeeded and not self.error:
            raise ValueError("failed execution result requires an error")
        if not self.succeeded and self.output is not None:
            raise ValueError("failed execution result cannot contain output")
        return self


class EvaluationSubject(EvaluationModel):
    trace: AgentExecutionTrace
    state: EvaluationStateSnapshot
    result: ExecutionResultSnapshot

    @model_validator(mode="after")
    def validate_identity(self) -> EvaluationSubject:
        identities = {
            (self.trace.run_id, self.trace.task_id),
            (self.state.run_id, self.state.task_id),
            (self.result.run_id, self.result.task_id),
        }
        if len(identities) != 1:
            raise ValueError("evaluation inputs must describe one run and task")
        if self.state.status not in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
        }:
            raise ValueError("an execution result requires a terminal state snapshot")
        state_succeeded = self.state.status is ExecutionStatus.COMPLETED
        if self.result.succeeded is not state_succeeded:
            raise ValueError("state and execution result disagree on success")
        if self.state.output != self.result.output:
            raise ValueError("state and execution result contain different output")
        if self.state.error != self.result.error:
            raise ValueError("state and execution result contain different error")
        return self


class EvaluationCriteria(EvaluationModel):
    require_completion: bool = True
    require_output: bool = True
    required_output_keys: tuple[str, ...] = ()
    expected_output_values: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_criteria(self) -> EvaluationCriteria:
        if len(set(self.required_output_keys)) != len(self.required_output_keys):
            raise ValueError("required output keys must be unique")
        return self


class OutputQualityAssessment(EvaluationModel):
    score: float = Field(ge=0.0, le=1.0)
    missing_keys: tuple[str, ...] = ()
    mismatched_keys: tuple[str, ...] = ()
    metrics: ImmutableJsonObject = Field(default_factory=dict)


class EvaluationFinding(EvaluationModel):
    finding_id: UUID
    code: str = Field(min_length=1)
    severity: FindingSeverity
    component: EvaluationComponent
    summary: str = Field(min_length=1)
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    evidence_fact_ids: tuple[UUID, ...] = ()
    details: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evidence(self) -> EvaluationFinding:
        if len(set(self.evidence_fact_ids)) != len(self.evidence_fact_ids):
            raise ValueError("finding evidence ids must be unique")
        return self


class EvaluationResult(EvaluationModel):
    evaluation_id: UUID
    trace_id: UUID
    run_id: UUID
    task_id: UUID
    evaluator_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    scope: EvaluationScope
    component: EvaluationComponent | None = None
    verdict: EvaluationVerdict
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    metrics: ImmutableJsonObject = Field(default_factory=dict)
    findings: tuple[EvaluationFinding, ...] = ()
    evaluated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_result(self) -> EvaluationResult:
        if self.scope is EvaluationScope.COMPONENT:
            if self.component is None:
                raise ValueError("component evaluation requires a component")
        elif self.component is not None:
            raise ValueError("only component evaluation can set component")
        if self.verdict is EvaluationVerdict.INCONCLUSIVE:
            if self.score is not None:
                raise ValueError("inconclusive evaluation cannot contain a score")
        elif self.score is None:
            raise ValueError("conclusive evaluation requires a score")
        finding_ids = tuple(item.finding_id for item in self.findings)
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("evaluation finding ids must be unique")
        if self.verdict is EvaluationVerdict.PASS and any(
            item.severity in {FindingSeverity.ERROR, FindingSeverity.CRITICAL}
            for item in self.findings
        ):
            raise ValueError("passing evaluation cannot contain failure findings")
        return self


class EvaluationReport(EvaluationModel):
    report_id: UUID
    trace_id: UUID
    run_id: UUID
    task_id: UUID
    outcome: EvaluationResult
    trajectory: EvaluationResult
    components: tuple[EvaluationResult, ...]
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        results = (self.outcome, self.trajectory, *self.components)
        if self.outcome.scope is not EvaluationScope.OUTCOME:
            raise ValueError("report outcome has the wrong scope")
        if self.trajectory.scope is not EvaluationScope.TRAJECTORY:
            raise ValueError("report trajectory has the wrong scope")
        if any(item.scope is not EvaluationScope.COMPONENT for item in self.components):
            raise ValueError("report component result has the wrong scope")
        identities = {
            (item.trace_id, item.run_id, item.task_id) for item in results
        }
        identities.add((self.trace_id, self.run_id, self.task_id))
        if len(identities) != 1:
            raise ValueError("evaluation report mixes traces or runs")
        components = tuple(item.component for item in self.components)
        if len(set(components)) != len(components):
            raise ValueError("evaluation report components must be unique")
        return self

    @property
    def results(self) -> tuple[EvaluationResult, ...]:
        return (self.outcome, self.trajectory, *self.components)


class FailureOccurrence(EvaluationModel):
    occurrence_id: UUID
    run_id: UUID
    trace_id: UUID
    evaluation_id: UUID
    finding_ids: tuple[UUID, ...] = Field(min_length=1)
    observed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_findings(self) -> FailureOccurrence:
        if len(set(self.finding_ids)) != len(self.finding_ids):
            raise ValueError("failure occurrence finding ids must be unique")
        return self


class RootCauseHypothesis(EvaluationModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    evidence_finding_codes: tuple[str, ...] = Field(min_length=1)


class FailurePattern(EvaluationModel):
    pattern_id: UUID
    pattern_key: str = Field(min_length=1)
    component: EvaluationComponent
    finding_code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    root_cause: RootCauseHypothesis
    occurrences: tuple[FailureOccurrence, ...] = Field(min_length=1)
    pattern_confidence: float = Field(ge=0.0, le=1.0)
    first_seen: AwareDatetime
    last_seen: AwareDatetime

    @model_validator(mode="after")
    def validate_pattern(self) -> FailurePattern:
        if self.last_seen < self.first_seen:
            raise ValueError("failure pattern last_seen precedes first_seen")
        occurrence_ids = tuple(item.occurrence_id for item in self.occurrences)
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("failure pattern occurrences must be unique")
        return self

    @property
    def affected_run_ids(self) -> tuple[UUID, ...]:
        return tuple(sorted({item.run_id for item in self.occurrences}, key=str))


class OptimizationCandidate(EvaluationModel):
    candidate_id: UUID
    pattern_id: UUID
    target_component: EvaluationComponent
    objective: str = Field(min_length=1)
    change_kind: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_benefit: str = Field(min_length=1)
    expected_benefit_score: float = Field(ge=0.0, le=1.0)
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    affected_run_ids: tuple[UUID, ...] = Field(min_length=1)
    evidence_finding_ids: tuple[UUID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> OptimizationCandidate:
        if len(set(self.affected_run_ids)) != len(self.affected_run_ids):
            raise ValueError("candidate affected run ids must be unique")
        if len(set(self.evidence_finding_ids)) != len(
            self.evidence_finding_ids
        ):
            raise ValueError("candidate evidence finding ids must be unique")
        return self


class FailureAnalysis(EvaluationModel):
    patterns: tuple[FailurePattern, ...]
    candidates: tuple[OptimizationCandidate, ...]
    analyzed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_analysis(self) -> FailureAnalysis:
        patterns = {item.pattern_id: item for item in self.patterns}
        if len(patterns) != len(self.patterns):
            raise ValueError("failure analysis pattern ids must be unique")
        candidate_ids = {item.candidate_id for item in self.candidates}
        if len(candidate_ids) != len(self.candidates):
            raise ValueError("failure analysis candidate ids must be unique")
        for candidate in self.candidates:
            pattern = patterns.get(candidate.pattern_id)
            if pattern is None:
                raise ValueError(
                    "optimization candidate references an unknown pattern"
                )
            if candidate.target_component is not pattern.component:
                raise ValueError(
                    "optimization candidate targets another pattern component"
                )
            if set(candidate.affected_run_ids) != set(pattern.affected_run_ids):
                raise ValueError(
                    "optimization candidate affected runs differ from its pattern"
                )
            pattern_evidence = {
                finding_id
                for occurrence in pattern.occurrences
                for finding_id in occurrence.finding_ids
            }
            if not set(candidate.evidence_finding_ids).issubset(pattern_evidence):
                raise ValueError(
                    "optimization candidate contains evidence outside its pattern"
                )
            if candidate.assessment_confidence > pattern.pattern_confidence:
                raise ValueError(
                    "optimization candidate confidence exceeds its pattern"
                )
        return self


class ConservativeOptimizationPolicy(EvaluationModel):
    min_affected_runs: int = Field(default=2, ge=2)
    min_pattern_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    min_expected_benefit: float = Field(default=0.5, ge=0.0, le=1.0)
    min_evidence_findings: int = Field(default=2, ge=1)


class OptimizationProposal(EvaluationModel):
    """Legacy/test-only proposal used by historical Evaluation/Evolution tests."""
    proposal_id: UUID
    status: ProposalStatus = ProposalStatus.PROPOSED
    source_pattern_ids: tuple[UUID, ...] = Field(min_length=1)
    source_candidate_ids: tuple[UUID, ...] = Field(min_length=1)
    target_component: EvaluationComponent
    change_kind: str = Field(min_length=1)
    change_spec: ImmutableJsonObject
    rationale: str = Field(min_length=1)
    expected_benefit: str = Field(min_length=1)
    proposal_confidence: float = Field(ge=0.0, le=1.0)
    validation_plan: tuple[str, ...] = Field(min_length=1)
    rollback_plan: tuple[str, ...] = Field(min_length=1)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_proposal(self) -> OptimizationProposal:
        if len(set(self.source_pattern_ids)) != len(self.source_pattern_ids):
            raise ValueError("proposal source pattern ids must be unique")
        if len(set(self.source_candidate_ids)) != len(
            self.source_candidate_ids
        ):
            raise ValueError("proposal source candidate ids must be unique")
        return self
