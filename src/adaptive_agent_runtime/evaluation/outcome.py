"""Deterministic result-level evaluation without an LLM judge."""

from __future__ import annotations

from typing import Mapping
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.evaluation.contracts import OutputQualityEvaluator
from adaptive_agent_runtime.evaluation.models import (
    EvaluationComponent,
    EvaluationCriteria,
    EvaluationFinding,
    EvaluationResult,
    EvaluationScope,
    EvaluationSubject,
    EvaluationVerdict,
    ExecutionStatus,
    FindingSeverity,
    OutputQualityAssessment,
    TraceCategory,
    TraceCompleteness,
    canonical_evaluation_json,
    stable_evaluation_id,
)


class DeterministicOutputQualityEvaluator:
    module_id = "evaluation.output_quality.deterministic"

    def assess(
        self,
        output: JsonValue,
        criteria: EvaluationCriteria,
    ) -> OutputQualityAssessment:
        expected = criteria.expected_output_values
        required = tuple(
            dict.fromkeys((*criteria.required_output_keys, *expected.keys()))
        )
        if output is None:
            score = 0.0 if criteria.require_output or required else 1.0
            return OutputQualityAssessment(
                score=score,
                missing_keys=required,
                metrics={"requirement_count": len(required), "satisfied_count": 0},
            )
        if not required:
            return OutputQualityAssessment(
                score=1.0,
                metrics={"requirement_count": 0, "satisfied_count": 0},
            )
        if not isinstance(output, Mapping):
            return OutputQualityAssessment(
                score=0.0,
                missing_keys=required,
                metrics={"requirement_count": len(required), "satisfied_count": 0},
            )

        missing = tuple(key for key in required if key not in output)
        mismatched = tuple(
            key
            for key, value in expected.items()
            if key in output and output[key] != value
        )
        satisfied = len(required) - len(missing) - len(mismatched)
        return OutputQualityAssessment(
            score=max(0.0, satisfied / len(required)),
            missing_keys=missing,
            mismatched_keys=mismatched,
            metrics={
                "requirement_count": len(required),
                "satisfied_count": satisfied,
            },
        )


class DeterministicOutcomeEvaluator:
    module_id = "evaluation.outcome.deterministic"
    evaluator_version = "1"

    def __init__(
        self,
        quality: OutputQualityEvaluator | None = None,
    ) -> None:
        self._quality = quality or DeterministicOutputQualityEvaluator()

    def evaluate(
        self,
        subject: EvaluationSubject,
        criteria: EvaluationCriteria,
    ) -> EvaluationResult:
        trace = subject.trace
        terminal_facts = tuple(
            fact
            for fact in trace.facts
            if fact.category is TraceCategory.FINAL_RESULT
        )
        evidence_ids = tuple(fact.fact_id for fact in terminal_facts)
        runtime_coverage = trace.coverage_for(EvaluationComponent.RUNTIME)
        assessment_confidence = 0.6
        if runtime_coverage is not None:
            assessment_confidence = {
                TraceCompleteness.COMPLETE: 1.0,
                TraceCompleteness.PARTIAL: 0.8,
                TraceCompleteness.MISSING: 0.6,
            }[runtime_coverage.completeness]

        dumped_result = subject.result.model_dump(mode="json")
        quality = self._quality.assess(dumped_result["output"], criteria)
        snapshot_completed = (
            subject.state.status is ExecutionStatus.COMPLETED
            and subject.result.succeeded
        )
        completed_trace = tuple(
            fact for fact in terminal_facts if fact.kind == "runtime.completed"
        )
        failed_trace = tuple(
            fact for fact in terminal_facts if fact.kind == "runtime.failed"
        )
        evidence_conflict = (
            bool(completed_trace and failed_trace)
            or bool(failed_trace and snapshot_completed)
            or (bool(completed_trace) and not snapshot_completed)
        )
        completed = snapshot_completed and not failed_trace and not evidence_conflict
        completion_score = 1.0 if completed else 0.0
        has_explicit_goal = bool(
            criteria.required_output_keys or criteria.expected_output_values
        )
        goal_score = quality.score if has_explicit_goal else completion_score
        overall_score = (
            (completion_score * 0.4)
            + (quality.score * 0.3)
            + (goal_score * 0.3)
        )

        failed = (
            not completed
            or (criteria.require_output and quality.score < 1.0)
            or (has_explicit_goal and goal_score < 1.0)
        )
        findings: list[EvaluationFinding] = []
        if not completed:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="outcome.runtime_failed",
                    severity=FindingSeverity.CRITICAL,
                    summary="Runtime did not complete the task successfully.",
                    evidence_ids=evidence_ids,
                    confidence=assessment_confidence,
                    details={"error": subject.result.error},
                )
            )
        if evidence_conflict:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="outcome.execution_evidence_conflict",
                    severity=FindingSeverity.CRITICAL,
                    summary=(
                        "Runtime terminal Trace and execution snapshots disagree."
                    ),
                    evidence_ids=evidence_ids,
                    confidence=assessment_confidence,
                    details={
                        "snapshot_succeeded": snapshot_completed,
                        "completed_trace_count": len(completed_trace),
                        "failed_trace_count": len(failed_trace),
                    },
                )
            )
        if criteria.require_output and dumped_result["output"] is None:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="outcome.output_missing",
                    severity=FindingSeverity.ERROR,
                    summary="The completed result has no required output.",
                    evidence_ids=evidence_ids,
                    confidence=assessment_confidence,
                )
            )
        if has_explicit_goal and goal_score < 1.0:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="outcome.goal_not_satisfied",
                    severity=FindingSeverity.ERROR,
                    summary="Deterministic output requirements were not satisfied.",
                    evidence_ids=evidence_ids,
                    confidence=assessment_confidence,
                    details={
                        "missing_keys": list(quality.missing_keys),
                        "mismatched_keys": list(quality.mismatched_keys),
                    },
                )
            )

        verdict = EvaluationVerdict.FAIL if failed else EvaluationVerdict.PASS
        findings_tuple = tuple(sorted(findings, key=lambda item: item.code))
        evaluation_id = stable_evaluation_id(
            self.module_id,
            self.evaluator_version,
            trace.trace_id,
            canonical_evaluation_json(criteria),
            verdict.value,
            *(item.finding_id for item in findings_tuple),
        )
        return EvaluationResult(
            evaluation_id=evaluation_id,
            trace_id=trace.trace_id,
            run_id=trace.run_id,
            task_id=trace.task_id,
            evaluator_id=self.module_id,
            evaluator_version=self.evaluator_version,
            scope=EvaluationScope.OUTCOME,
            verdict=verdict,
            score=overall_score,
            assessment_confidence=assessment_confidence,
            metrics={
                "completion_score": completion_score,
                "output_quality_score": quality.score,
                "goal_satisfaction_score": goal_score,
                "missing_output_keys": list(quality.missing_keys),
                "mismatched_output_keys": list(quality.mismatched_keys),
            },
            findings=findings_tuple,
            evaluated_at=subject.result.completed_at,
        )

    @staticmethod
    def _finding(
        trace_id: object,
        *,
        code: str,
        severity: FindingSeverity,
        summary: str,
        evidence_ids: tuple[object, ...],
        confidence: float,
        details: Mapping[str, JsonValue] | None = None,
    ) -> EvaluationFinding:
        typed_evidence = tuple(
            item for item in evidence_ids if isinstance(item, UUID)
        )
        return EvaluationFinding(
            finding_id=stable_evaluation_id(
                "finding",
                trace_id,
                code,
                *typed_evidence,
            ),
            code=code,
            severity=severity,
            component=EvaluationComponent.RUNTIME,
            summary=summary,
            assessment_confidence=confidence,
            evidence_fact_ids=typed_evidence,
            details=details or {},
        )
