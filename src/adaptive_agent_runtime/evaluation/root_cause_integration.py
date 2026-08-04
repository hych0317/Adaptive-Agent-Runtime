"""Read-only adapters around semantic Root Cause decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from pydantic import Field

from adaptive_agent_runtime.decisioning import DecisionEvidenceReference
from adaptive_agent_runtime.evaluation.models import (
    AgentExecutionTrace,
    EvaluationCorrelation,
    EvaluationFact,
    EvaluationModel,
    EvaluationReport,
    FindingSeverity,
    TraceBatch,
    TraceCompleteness,
)
from adaptive_agent_runtime.evaluation.root_cause import (
    DeterministicFindingSnapshot,
    RootCauseAssessment,
    RootCauseConclusion,
    RootCauseDecisionPayload,
    RootCauseEvidenceBinding,
    RootCauseEvidenceStrength,
    RootCauseExecutionPolicy,
    RootCauseTrigger,
    root_cause_failure_signature,
)


class RootCauseDecisionInput(EvaluationModel):
    payload: RootCauseDecisionPayload
    evidence: tuple[DecisionEvidenceReference, ...] = Field(min_length=1)


class RootCauseInputAssembler:
    """Build bounded evidence snapshots without constructing a fake terminal run."""

    module_id = "evaluation.root_cause.input_assembler"

    def inline_failure(
        self,
        *,
        run_id: UUID,
        task_id: UUID,
        node_id: UUID,
        action_id: UUID,
        failure_summary: str,
        trace_batch: TraceBatch,
        policy: RootCauseExecutionPolicy,
    ) -> RootCauseDecisionInput:
        correlation = EvaluationCorrelation(
            run_id=run_id,
            task_id=task_id,
            node_id=node_id,
            action_id=action_id,
        )
        direct_id = f"observation:{action_id}"
        evidence: list[DecisionEvidenceReference] = [
            DecisionEvidenceReference(
                evidence_id=direct_id,
                kind="runtime.observation.failure",
                source="runtime_core",
                reliability=1.0,
                summary=failure_summary,
            )
        ]
        bindings: list[RootCauseEvidenceBinding] = [
            RootCauseEvidenceBinding(
                evidence_id=direct_id,
                kind="runtime.observation.failure",
                source_record_id=str(action_id),
                correlation=correlation,
                reliability=1.0,
                direct_failure=True,
            )
        ]
        relevant = tuple(
            fact
            for fact in trace_batch.facts
            if fact.correlation.run_id == run_id
            and fact.correlation.task_id in {None, task_id}
            and fact.correlation.node_id in {None, node_id}
            and fact.correlation.action_id in {None, action_id}
        )
        remaining = max(0, policy.max_evidence_items - 1)
        for fact in relevant[-remaining:] if remaining else ():
            evidence_id = f"trace:{fact.fact_id}"
            if evidence_id == direct_id:
                continue
            evidence.append(
                DecisionEvidenceReference(
                    evidence_id=evidence_id,
                    kind=fact.kind,
                    source="runtime_trace",
                    reliability=0.9,
                    summary=_fact_summary(fact),
                )
            )
            bindings.append(
                RootCauseEvidenceBinding(
                    evidence_id=evidence_id,
                    kind=fact.kind,
                    source_record_id=fact.source_record_id,
                    correlation=EvaluationCorrelation(
                        run_id=run_id,
                        task_id=task_id,
                        node_id=(fact.correlation.node_id or node_id),
                        action_id=(fact.correlation.action_id or action_id),
                    ),
                    reliability=0.9,
                )
            )
        completeness = _trace_batch_completeness(trace_batch, run_id)
        payload = RootCauseDecisionPayload(
            trigger=RootCauseTrigger.INLINE_FAILURE,
            failure_signature=root_cause_failure_signature(failure_summary),
            failure_summary=failure_summary,
            correlation=correlation,
            trace_completeness=completeness,
            evidence_bindings=tuple(bindings),
            execution_policy=policy,
        )
        return RootCauseDecisionInput(payload=payload, evidence=tuple(evidence))

    def post_run(
        self,
        report: EvaluationReport,
        trace: AgentExecutionTrace,
        *,
        policy: RootCauseExecutionPolicy,
    ) -> RootCauseDecisionInput | None:
        failures = tuple(
            (result, finding)
            for result in report.results
            for finding in result.findings
            if finding.severity in {FindingSeverity.ERROR, FindingSeverity.CRITICAL}
        )
        if not failures:
            return None
        fact_by_id = {fact.fact_id: fact for fact in trace.facts}
        evidence: list[DecisionEvidenceReference] = []
        bindings: list[RootCauseEvidenceBinding] = []
        evidence_id_by_fact: dict[UUID, str] = {}
        for fact_id in dict.fromkeys(
            fact_id
            for _, finding in failures
            for fact_id in finding.evidence_fact_ids
        ):
            if len(bindings) >= policy.max_evidence_items:
                break
            fact = fact_by_id.get(fact_id)
            if fact is None:
                continue
            evidence_id = f"trace:{fact.fact_id}"
            evidence_id_by_fact[fact.fact_id] = evidence_id
            evidence.append(
                DecisionEvidenceReference(
                    evidence_id=evidence_id,
                    kind=fact.kind,
                    source="evaluation_trace",
                    reliability=0.9,
                    summary=_fact_summary(fact),
                )
            )
            bindings.append(
                RootCauseEvidenceBinding(
                    evidence_id=evidence_id,
                    kind=fact.kind,
                    source_record_id=fact.source_record_id,
                    correlation=fact.correlation,
                    reliability=0.9,
                    direct_failure=("failed" in fact.kind or "failure" in fact.kind),
                )
            )
        deterministic = tuple(
            DeterministicFindingSnapshot(
                finding_id=finding.finding_id,
                evaluation_id=result.evaluation_id,
                component=finding.component,
                code=finding.code,
                severity=finding.severity,
                summary=finding.summary,
                evidence_ids=tuple(
                    evidence_id_by_fact[item]
                    for item in finding.evidence_fact_ids
                    if item in evidence_id_by_fact
                ),
            )
            for result, finding in failures
        )
        if not bindings:
            fallback_id = f"finding:{failures[0][1].finding_id}"
            finding = failures[0][1]
            evidence.append(
                DecisionEvidenceReference(
                    evidence_id=fallback_id,
                    kind="evaluation.deterministic_finding",
                    source="evaluation_kernel",
                    reliability=finding.assessment_confidence,
                    summary=finding.summary,
                )
            )
            bindings.append(
                RootCauseEvidenceBinding(
                    evidence_id=fallback_id,
                    kind="evaluation.deterministic_finding",
                    source_record_id=str(finding.finding_id),
                    correlation=EvaluationCorrelation(
                        run_id=report.run_id,
                        task_id=report.task_id,
                    ),
                    reliability=finding.assessment_confidence,
                )
            )
        failure_summary = "; ".join(
            dict.fromkeys(finding.summary for _, finding in failures)
        )
        payload = RootCauseDecisionPayload(
            trigger=RootCauseTrigger.POST_RUN_EVALUATION,
            failure_signature=root_cause_failure_signature(
                "|".join(finding.code for _, finding in failures),
                component="evaluation",
            ),
            failure_summary=failure_summary,
            correlation=EvaluationCorrelation(
                run_id=report.run_id,
                task_id=report.task_id,
            ),
            trace_id=report.trace_id,
            trace_completeness=_trace_completeness(trace),
            evidence_bindings=tuple(bindings),
            deterministic_findings=deterministic,
            execution_policy=policy,
        )
        return RootCauseDecisionInput(payload=payload, evidence=tuple(evidence))


class RecoveryEvidenceQuery(EvaluationModel):
    run_id: UUID
    task_id: UUID
    node_id: UUID
    action_id: UUID
    failure_signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecoveryDiagnosticEvidence(EvaluationModel):
    assessment_id: UUID
    evidence_id: str = Field(min_length=1)
    kind: str = "evaluation.root_cause.assessment"
    summary: str = Field(min_length=1)
    reliability: float = Field(ge=0.0, le=1.0)


class RootCauseAssessmentReader(Protocol):
    async def list_for_run(self, run_id: UUID) -> tuple[RootCauseAssessment, ...]: ...


class RootCauseRecoveryEvidenceProvider:
    """Admit only applied, evidence-bound assessments into Recovery context."""

    module_id = "evaluation.root_cause.recovery_evidence"

    def __init__(self, store: RootCauseAssessmentReader) -> None:
        self._store = store

    async def evidence_for(
        self,
        query: RecoveryEvidenceQuery,
    ) -> tuple[RecoveryDiagnosticEvidence, ...]:
        assessments = await self._store.list_for_run(query.run_id)
        eligible = tuple(
            item
            for item in assessments
            if item.conclusion is RootCauseConclusion.SUPPORTED
            and item.evidence_strength is not RootCauseEvidenceStrength.INSUFFICIENT
            and item.correlation.task_id == query.task_id
            and item.correlation.node_id == query.node_id
            and item.correlation.action_id == query.action_id
            and item.failure_signature == query.failure_signature
        )
        return tuple(
            RecoveryDiagnosticEvidence(
                assessment_id=item.assessment_id,
                evidence_id=f"root-cause:{item.assessment_id}",
                summary=(
                    f"Validated semantic diagnosis '{item.primary_code}': "
                    f"{item.primary_description}"
                ),
                reliability=(
                    0.85
                    if item.evidence_strength
                    is RootCauseEvidenceStrength.CORROBORATED
                    else 0.65
                ),
            )
            for item in sorted(
                eligible,
                key=lambda assessment: (
                    assessment.assessed_at,
                    str(assessment.assessment_id),
                ),
                reverse=True,
            )[:2]
        )


def _trace_batch_completeness(
    batch: TraceBatch,
    run_id: UUID,
) -> TraceCompleteness:
    values = tuple(item.completeness for item in batch.coverage if item.run_id == run_id)
    if not values:
        return TraceCompleteness.MISSING
    if TraceCompleteness.MISSING in values:
        return TraceCompleteness.MISSING
    if TraceCompleteness.PARTIAL in values:
        return TraceCompleteness.PARTIAL
    return TraceCompleteness.COMPLETE


def _trace_completeness(trace: AgentExecutionTrace) -> TraceCompleteness:
    values = tuple(item.completeness for item in trace.coverage)
    if not values or TraceCompleteness.MISSING in values:
        return TraceCompleteness.MISSING
    if TraceCompleteness.PARTIAL in values:
        return TraceCompleteness.PARTIAL
    return TraceCompleteness.COMPLETE


def _fact_summary(fact: EvaluationFact) -> str:
    error = _find_error(fact.payload)
    return f"{fact.kind}: {error}" if error else f"Observed Runtime fact '{fact.kind}'."


def _find_error(value: object) -> str | None:
    if isinstance(value, Mapping):
        direct = value.get("error")
        if isinstance(direct, str) and direct:
            return direct
        for item in value.values():
            found = _find_error(item)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_error(item)
            if found:
                return found
    return None
