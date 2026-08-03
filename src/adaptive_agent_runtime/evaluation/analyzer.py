"""Convert structured failures into root-cause hypotheses and candidates."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import fmean
from typing import Sequence
from uuid import UUID

from adaptive_agent_runtime.evaluation.models import (
    EvaluationComponent,
    EvaluationFinding,
    EvaluationResult,
    FailureAnalysis,
    FailureOccurrence,
    FailurePattern,
    FindingSeverity,
    OptimizationCandidate,
    RootCauseHypothesis,
    stable_evaluation_id,
)


_EMPTY_ANALYSIS_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)


class DeterministicFailureAnalyzer:
    module_id = "evaluation.failure_analyzer.deterministic"

    def analyze(
        self,
        results: Sequence[EvaluationResult],
    ) -> FailureAnalysis:
        unique_results = self._deduplicate_results(results)
        grouped: defaultdict[
            tuple[EvaluationComponent, str],
            list[tuple[EvaluationResult, EvaluationFinding]],
        ] = defaultdict(list)
        for result in unique_results:
            for finding in result.findings:
                if finding.severity in {
                    FindingSeverity.ERROR,
                    FindingSeverity.CRITICAL,
                }:
                    grouped[(finding.component, finding.code)].append(
                        (result, finding)
                    )

        patterns: list[FailurePattern] = []
        candidates: list[OptimizationCandidate] = []
        for (component, code), observations in sorted(
            grouped.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        ):
            occurrences = tuple(
                FailureOccurrence(
                    occurrence_id=stable_evaluation_id(
                        "failure-occurrence",
                        result.evaluation_id,
                        code,
                        finding.finding_id,
                    ),
                    run_id=result.run_id,
                    trace_id=result.trace_id,
                    evaluation_id=result.evaluation_id,
                    finding_ids=(finding.finding_id,),
                    observed_at=result.evaluated_at,
                )
                for result, finding in sorted(
                    observations,
                    key=lambda item: (
                        item[0].evaluated_at,
                        str(item[0].run_id),
                        str(item[0].evaluation_id),
                    ),
                )
            )
            affected_runs = {item.run_id for item in occurrences}
            base_confidence = fmean(
                finding.assessment_confidence
                for _, finding in observations
            )
            recurrence_factor = min(1.0, 0.6 + (0.2 * len(affected_runs)))
            confidence = base_confidence * recurrence_factor
            root_code, root_description = self._root_cause(code)
            pattern_id = stable_evaluation_id(
                "failure-pattern",
                component.value,
                code,
            )
            pattern = FailurePattern(
                pattern_id=pattern_id,
                pattern_key=f"{component.value}:{code}",
                component=component,
                finding_code=code,
                description=observations[0][1].summary,
                root_cause=RootCauseHypothesis(
                    code=root_code,
                    description=root_description,
                    assessment_confidence=confidence,
                    evidence_finding_codes=(code,),
                ),
                occurrences=occurrences,
                pattern_confidence=confidence,
                first_seen=min(item.observed_at for item in occurrences),
                last_seen=max(item.observed_at for item in occurrences),
            )
            patterns.append(pattern)

            evidence_ids = tuple(
                sorted(
                    {
                        finding_id
                        for occurrence in occurrences
                        for finding_id in occurrence.finding_ids
                    },
                    key=str,
                )
            )
            candidates.append(
                OptimizationCandidate(
                    candidate_id=stable_evaluation_id(
                        "optimization-candidate",
                        pattern_id,
                        confidence,
                        *(item.occurrence_id for item in occurrences),
                        *evidence_ids,
                    ),
                    pattern_id=pattern_id,
                    target_component=component,
                    objective=f"Reduce recurrence of {code}",
                    change_kind=self._change_kind(component, code),
                    rationale=(
                        f"Root-cause hypothesis '{root_code}' is supported by "
                        f"{len(affected_runs)} distinct run(s)."
                    ),
                    expected_benefit=(
                        "Reduce repeated execution failures while preserving "
                        "the current Runtime contract."
                    ),
                    expected_benefit_score=(
                        0.75 if len(affected_runs) >= 2 else 0.4
                    ),
                    assessment_confidence=confidence,
                    affected_run_ids=tuple(sorted(affected_runs, key=str)),
                    evidence_finding_ids=evidence_ids,
                )
            )

        analyzed_at = (
            max(item.evaluated_at for item in unique_results)
            if unique_results
            else _EMPTY_ANALYSIS_TIME
        )
        return FailureAnalysis(
            patterns=tuple(patterns),
            candidates=tuple(candidates),
            analyzed_at=analyzed_at,
        )

    @staticmethod
    def _deduplicate_results(
        results: Sequence[EvaluationResult],
    ) -> tuple[EvaluationResult, ...]:
        by_id: dict[UUID, EvaluationResult] = {}
        for result in results:
            existing = by_id.get(result.evaluation_id)
            if existing is None:
                by_id[result.evaluation_id] = result
            elif existing != result:
                raise ValueError(
                    f"evaluation_id '{result.evaluation_id}' has conflicting results"
                )
        return tuple(
            sorted(
                by_id.values(),
                key=lambda item: (
                    item.evaluated_at,
                    str(item.run_id),
                    str(item.evaluation_id),
                ),
            )
        )

    @staticmethod
    def _root_cause(code: str) -> tuple[str, str]:
        mappings = (
            (
                "capability_mismatch",
                "tool.capability_mismatch",
                "The selected Provider does not satisfy the Capability requirement.",
            ),
            (
                "provider_unavailable",
                "tool.provider_availability",
                "The required external Provider was unavailable.",
            ),
            (
                "timeout",
                "tool.provider_timeout",
                "The Provider did not finish within its execution policy.",
            ),
            (
                "node_failure",
                "orchestration.node_execution",
                "A Task Node execution failed or became blocked.",
            ),
            (
                "memory.conflict",
                "memory.conflicting_evidence",
                "Memory evidence produced an unresolved conditional conflict.",
            ),
            (
                "runtime_failed",
                "runtime.execution_failure",
                "The Runtime reached a failed terminal state.",
            ),
        )
        for marker, root_code, description in mappings:
            if marker in code:
                return root_code, description
        return (
            f"{code}.root_cause",
            "The structured finding indicates a repeatable component failure.",
        )

    @staticmethod
    def _change_kind(
        component: EvaluationComponent,
        code: str,
    ) -> str:
        if component is EvaluationComponent.TOOL:
            if "capability_mismatch" in code:
                return "tool.selection_policy.review"
            return "tool.execution_policy.review"
        if component is EvaluationComponent.ORCHESTRATION:
            return "orchestration.strategy.review"
        if component in {
            EvaluationComponent.CONTEXT,
            EvaluationComponent.MEMORY,
            EvaluationComponent.CONTEXT_MEMORY,
        }:
            return "context_memory.policy.review"
        return "runtime.configuration.review"
