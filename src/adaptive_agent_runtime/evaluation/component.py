"""Component-scoped evaluation using only normalized trace data."""

from __future__ import annotations

from typing import Iterable, Mapping
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.evaluation.models import (
    EvaluationComponent,
    EvaluationFact,
    EvaluationFinding,
    EvaluationResult,
    EvaluationScope,
    EvaluationSubject,
    EvaluationVerdict,
    FindingSeverity,
    TraceCompleteness,
    stable_evaluation_id,
)


_SEVERITY_PENALTY = {
    FindingSeverity.INFO: 0.0,
    FindingSeverity.WARNING: 0.1,
    FindingSeverity.ERROR: 0.3,
    FindingSeverity.CRITICAL: 0.5,
}


class _DeterministicComponentEvaluator:
    module_id: str
    evaluator_version = "1"
    component: EvaluationComponent
    observed_components: tuple[EvaluationComponent, ...]

    def evaluate(self, subject: EvaluationSubject) -> EvaluationResult:
        trace = subject.trace
        facts = trace.facts_for(*self.observed_components)
        completeness = tuple(
            coverage.completeness
            for component in self.observed_components
            if (coverage := trace.coverage_for(component)) is not None
        )
        if not facts:
            return EvaluationResult(
                evaluation_id=stable_evaluation_id(
                    self.module_id,
                    self.evaluator_version,
                    trace.trace_id,
                    "not-observed",
                ),
                trace_id=trace.trace_id,
                run_id=trace.run_id,
                task_id=trace.task_id,
                evaluator_id=self.module_id,
                evaluator_version=self.evaluator_version,
                scope=EvaluationScope.COMPONENT,
                component=self.component,
                verdict=EvaluationVerdict.INCONCLUSIVE,
                assessment_confidence=0.0,
                metrics={
                    "observed_fact_count": 0,
                    "coverage": [item.value for item in completeness],
                },
                evaluated_at=subject.result.completed_at,
            )

        confidence = self._coverage_confidence(completeness)
        findings = tuple(
            sorted(self._findings(trace.trace_id, facts, confidence), key=lambda x: x.code)
        )
        has_reported_coverage = (
            len(completeness) == len(self.observed_components)
            and all(
                item is not TraceCompleteness.MISSING for item in completeness
            )
        )
        has_failure = any(
            item.severity in {FindingSeverity.ERROR, FindingSeverity.CRITICAL}
            for item in findings
        )
        if not has_reported_coverage and not has_failure:
            return EvaluationResult(
                evaluation_id=stable_evaluation_id(
                    self.module_id,
                    self.evaluator_version,
                    trace.trace_id,
                    "coverage-missing",
                    *(item.finding_id for item in findings),
                ),
                trace_id=trace.trace_id,
                run_id=trace.run_id,
                task_id=trace.task_id,
                evaluator_id=self.module_id,
                evaluator_version=self.evaluator_version,
                scope=EvaluationScope.COMPONENT,
                component=self.component,
                verdict=EvaluationVerdict.INCONCLUSIVE,
                assessment_confidence=0.0,
                metrics={
                    "observed_fact_count": len(facts),
                    "finding_count": len(findings),
                    "coverage": [item.value for item in completeness],
                },
                findings=findings,
                evaluated_at=subject.result.completed_at,
            )
        penalty = min(
            1.0,
            sum(_SEVERITY_PENALTY[item.severity] for item in findings),
        )
        score = 1.0 - penalty
        verdict = (
            EvaluationVerdict.FAIL
            if has_failure
            else EvaluationVerdict.PASS
        )
        evaluation_id = stable_evaluation_id(
            self.module_id,
            self.evaluator_version,
            trace.trace_id,
            verdict.value,
            *(item.finding_id for item in findings),
        )
        return EvaluationResult(
            evaluation_id=evaluation_id,
            trace_id=trace.trace_id,
            run_id=trace.run_id,
            task_id=trace.task_id,
            evaluator_id=self.module_id,
            evaluator_version=self.evaluator_version,
            scope=EvaluationScope.COMPONENT,
            component=self.component,
            verdict=verdict,
            score=score,
            assessment_confidence=confidence,
            metrics={
                "observed_fact_count": len(facts),
                "finding_count": len(findings),
                "coverage": [item.value for item in completeness],
            },
            findings=findings,
            evaluated_at=subject.result.completed_at,
        )

    def _findings(
        self,
        trace_id: UUID,
        facts: tuple[EvaluationFact, ...],
        confidence: float,
    ) -> Iterable[EvaluationFinding]:
        raise NotImplementedError

    @staticmethod
    def _coverage_confidence(
        completeness: tuple[TraceCompleteness, ...],
    ) -> float:
        if not completeness:
            return 0.5
        scores = {
            TraceCompleteness.COMPLETE: 1.0,
            TraceCompleteness.PARTIAL: 0.7,
            TraceCompleteness.MISSING: 0.0,
        }
        return sum(scores[item] for item in completeness) / len(completeness)

    @staticmethod
    def _finding(
        trace_id: UUID,
        *,
        code: str,
        severity: FindingSeverity,
        component: EvaluationComponent,
        summary: str,
        evidence: tuple[EvaluationFact, ...],
        confidence: float,
    ) -> EvaluationFinding:
        evidence_ids = tuple(sorted((item.fact_id for item in evidence), key=str))
        return EvaluationFinding(
            finding_id=stable_evaluation_id(
                "finding",
                trace_id,
                code,
                *evidence_ids,
            ),
            code=code,
            severity=severity,
            component=component,
            summary=summary,
            assessment_confidence=confidence,
            evidence_fact_ids=evidence_ids,
        )


class OrchestrationComponentEvaluator(_DeterministicComponentEvaluator):
    module_id = "evaluation.component.orchestration"
    component = EvaluationComponent.ORCHESTRATION
    observed_components = (EvaluationComponent.ORCHESTRATION,)

    def _findings(
        self,
        trace_id: UUID,
        facts: tuple[EvaluationFact, ...],
        confidence: float,
    ) -> Iterable[EvaluationFinding]:
        failures = tuple(fact for fact in facts if self._is_node_failure(fact.payload))
        mutation_rejections = tuple(
            fact
            for fact in facts
            if fact.kind == "orchestration.mutation_rejected"
            or fact.payload.get("status") == "rejected"
        )
        if failures:
            yield self._finding(
                trace_id,
                code="component.orchestration.node_failure",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.ORCHESTRATION,
                summary="One or more Task Graph nodes failed.",
                evidence=failures,
                confidence=confidence,
            )
        if mutation_rejections:
            yield self._finding(
                trace_id,
                code="component.orchestration.mutation_rejected",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.ORCHESTRATION,
                summary="A dynamic graph mutation was rejected.",
                evidence=mutation_rejections,
                confidence=confidence,
            )

    @staticmethod
    def _is_node_failure(payload: Mapping[str, JsonValue]) -> bool:
        observation = payload.get("observation")
        if isinstance(observation, Mapping):
            if observation.get("succeeded") is False:
                return True
        graph = payload.get("graph")
        if isinstance(graph, Mapping):
            nodes = graph.get("nodes")
            if isinstance(nodes, (list, tuple)):
                return any(
                    isinstance(node, Mapping)
                    and node.get("status") in {"failed", "blocked"}
                    for node in nodes
                )
        return payload.get("status") in {"failed", "blocked"}


class ToolComponentEvaluator(_DeterministicComponentEvaluator):
    module_id = "evaluation.component.tool"
    component = EvaluationComponent.TOOL
    observed_components = (EvaluationComponent.TOOL,)

    def _findings(
        self,
        trace_id: UUID,
        facts: tuple[EvaluationFact, ...],
        confidence: float,
    ) -> Iterable[EvaluationFinding]:
        capability_mismatch = tuple(
            fact
            for fact in facts
            if fact.kind == "tool.capability_mismatch"
            or (
                fact.kind == "tool.selection_failed"
                and fact.payload.get("reason") == "capability_mismatch"
            )
            or fact.payload.get("reason_code") == "capability_mismatch"
        )
        timeouts = tuple(
            fact for fact in facts if fact.kind == "tool.attempt_timed_out"
        )
        unavailable = tuple(
            fact for fact in facts if fact.kind == "tool.provider_unavailable"
        )
        retries = tuple(
            fact for fact in facts if fact.kind == "tool.retry_scheduled"
        )
        failed = tuple(
            fact
            for fact in facts
            if fact.kind == "tool.execution_finished"
            and self._finished_status(fact.payload)
            in {"failed", "timed_out", "provider_unavailable"}
        )
        if capability_mismatch:
            yield self._finding(
                trace_id,
                code="component.tool.capability_mismatch",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.TOOL,
                summary="Tool selection did not satisfy the Capability requirement.",
                evidence=capability_mismatch,
                confidence=confidence,
            )
        if timeouts:
            yield self._finding(
                trace_id,
                code="component.tool.timeout",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.TOOL,
                summary="Tool execution timed out.",
                evidence=timeouts,
                confidence=confidence,
            )
        if unavailable:
            yield self._finding(
                trace_id,
                code="component.tool.provider_unavailable",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.TOOL,
                summary="A Tool Provider was unavailable.",
                evidence=unavailable,
                confidence=confidence,
            )
        if failed:
            yield self._finding(
                trace_id,
                code="component.tool.execution_failed",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.TOOL,
                summary="A Tool call did not recover successfully.",
                evidence=failed,
                confidence=confidence,
            )
        if retries:
            yield self._finding(
                trace_id,
                code="component.tool.retry",
                severity=FindingSeverity.WARNING,
                component=EvaluationComponent.TOOL,
                summary="Tool execution required retry handling.",
                evidence=retries,
                confidence=confidence,
            )

    @staticmethod
    def _finished_status(payload: Mapping[str, JsonValue]) -> object:
        data = payload.get("data")
        return data.get("status") if isinstance(data, Mapping) else None


class ContextMemoryComponentEvaluator(_DeterministicComponentEvaluator):
    module_id = "evaluation.component.context_memory"
    component = EvaluationComponent.CONTEXT_MEMORY
    observed_components = (
        EvaluationComponent.CONTEXT,
        EvaluationComponent.MEMORY,
    )

    def _findings(
        self,
        trace_id: UUID,
        facts: tuple[EvaluationFact, ...],
        confidence: float,
    ) -> Iterable[EvaluationFinding]:
        context_failures = tuple(
            fact
            for fact in facts
            if fact.component is EvaluationComponent.CONTEXT
            and (
                fact.kind in {"context.failed", "context.operation_failed"}
                or fact.payload.get("status") == "failed"
            )
        )
        memory_conflicts = tuple(
            fact
            for fact in facts
            if fact.component is EvaluationComponent.MEMORY
            and (
                fact.kind == "memory.conflict"
                or fact.payload.get("status") == "conflicted"
                or fact.payload.get("evolution") == "conflict"
            )
        )
        if context_failures:
            yield self._finding(
                trace_id,
                code="component.context.operation_failed",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.CONTEXT,
                summary="A Context lifecycle or assembly operation failed.",
                evidence=context_failures,
                confidence=confidence,
            )
        if memory_conflicts:
            yield self._finding(
                trace_id,
                code="component.memory.conflict",
                severity=FindingSeverity.ERROR,
                component=EvaluationComponent.MEMORY,
                summary="Memory consolidation produced unresolved conflict.",
                evidence=memory_conflicts,
                confidence=confidence,
            )
