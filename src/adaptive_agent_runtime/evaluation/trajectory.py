"""Deterministic trajectory evaluation over normalized trace facts."""

from __future__ import annotations

from typing import Mapping
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
    TraceCategory,
    TraceCompleteness,
    stable_evaluation_id,
)


class DeterministicTrajectoryEvaluator:
    module_id = "evaluation.trajectory.deterministic"
    evaluator_version = "1"

    def evaluate(self, subject: EvaluationSubject) -> EvaluationResult:
        trace = subject.trace
        if not trace.facts:
            return EvaluationResult(
                evaluation_id=stable_evaluation_id(
                    self.module_id,
                    self.evaluator_version,
                    trace.trace_id,
                    "inconclusive",
                ),
                trace_id=trace.trace_id,
                run_id=trace.run_id,
                task_id=trace.task_id,
                evaluator_id=self.module_id,
                evaluator_version=self.evaluator_version,
                scope=EvaluationScope.TRAJECTORY,
                verdict=EvaluationVerdict.INCONCLUSIVE,
                assessment_confidence=0.0,
                metrics={"fact_count": 0},
                evaluated_at=subject.result.completed_at,
            )

        retries = self._facts(trace.facts, "tool.retry_scheduled")
        attempt_failures = self._facts(trace.facts, "tool.attempt_failed")
        timeouts = self._facts(trace.facts, "tool.attempt_timed_out")
        unavailable = self._facts(trace.facts, "tool.provider_unavailable")
        runtime_failures = (
            self._facts(trace.facts, "runtime.failed")
            + self._facts(trace.facts, "runtime.terminated")
        )

        action_facts = tuple(
            fact for fact in trace.facts if fact.kind == "action.started"
        )
        observation_facts = tuple(
            fact for fact in trace.facts if fact.kind == "observation.received"
        )
        observed_actions = {
            fact.correlation.action_id
            for fact in observation_facts
            if fact.correlation.action_id is not None
        }
        missing_observations = tuple(
            fact
            for fact in action_facts
            if fact.correlation.action_id is not None
            and fact.correlation.action_id not in observed_actions
        )
        invalid_action_order = self._invalid_action_observation_order(
            action_facts,
            observation_facts,
        )
        invalid_tool_order = self._invalid_tool_order(trace.facts)
        node_failures = tuple(
            fact
            for fact in trace.facts
            if fact.component is EvaluationComponent.ORCHESTRATION
            and self._has_failed_execution(fact.payload)
        )

        findings: list[EvaluationFinding] = []
        confidence = self._assessment_confidence(subject)
        if retries:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.tool_retry",
                    severity=FindingSeverity.WARNING,
                    component=EvaluationComponent.TOOL,
                    summary="Tool execution required one or more retries.",
                    evidence=retries,
                    confidence=confidence,
                )
            )
        if attempt_failures:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.tool_attempt_failure",
                    severity=FindingSeverity.WARNING,
                    component=EvaluationComponent.TOOL,
                    summary="A Tool attempt failed during the trajectory.",
                    evidence=attempt_failures,
                    confidence=confidence,
                )
            )
        if timeouts:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.tool_timeout",
                    severity=FindingSeverity.ERROR,
                    component=EvaluationComponent.TOOL,
                    summary="A Tool attempt exceeded its timeout.",
                    evidence=timeouts,
                    confidence=confidence,
                )
            )
        if unavailable:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.provider_unavailable",
                    severity=FindingSeverity.ERROR,
                    component=EvaluationComponent.TOOL,
                    summary="A selected Tool Provider was unavailable.",
                    evidence=unavailable,
                    confidence=confidence,
                )
            )
        if missing_observations:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.missing_observation",
                    severity=FindingSeverity.ERROR,
                    component=EvaluationComponent.ORCHESTRATION,
                    summary="An action has no correlated observation.",
                    evidence=missing_observations,
                    confidence=confidence,
                )
            )
        if invalid_action_order:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.invalid_action_observation_order",
                    severity=FindingSeverity.ERROR,
                    component=EvaluationComponent.ORCHESTRATION,
                    summary=(
                        "An observation precedes its action in the same source trace."
                    ),
                    evidence=invalid_action_order,
                    confidence=confidence,
                )
            )
        if invalid_tool_order:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.invalid_tool_event_order",
                    severity=FindingSeverity.ERROR,
                    component=EvaluationComponent.TOOL,
                    summary=(
                        "Tool attempt, retry, or completion events violate their "
                        "source order."
                    ),
                    evidence=invalid_tool_order,
                    confidence=confidence,
                )
            )
        if node_failures:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.node_failure",
                    severity=FindingSeverity.ERROR,
                    component=EvaluationComponent.ORCHESTRATION,
                    summary="A Task Graph node ended in failure.",
                    evidence=node_failures,
                    confidence=confidence,
                )
            )
        if runtime_failures:
            findings.append(
                self._finding(
                    trace.trace_id,
                    code="trajectory.runtime_failed",
                    severity=FindingSeverity.CRITICAL,
                    component=EvaluationComponent.RUNTIME,
                    summary="The Runtime trajectory ended in failure.",
                    evidence=runtime_failures,
                    confidence=confidence,
                )
            )

        penalty = min(
            1.0,
            (len(retries) * 0.05)
            + (len(attempt_failures) * 0.05)
            + (len(timeouts) * 0.20)
            + (len(unavailable) * 0.20)
            + (len(missing_observations) * 0.25)
            + (0.25 if invalid_action_order else 0.0)
            + (0.25 if invalid_tool_order else 0.0)
            + (len(node_failures) * 0.20)
            + (len(runtime_failures) * 0.35),
        )
        score = 1.0 - penalty
        has_failure_finding = any(
            finding.severity
            in {FindingSeverity.ERROR, FindingSeverity.CRITICAL}
            for finding in findings
        )
        failure_detected = (
            not subject.result.succeeded
            or has_failure_finding
            or score < 0.5
        )
        coverage_sufficient = self._coverage_is_sufficient(subject)
        if failure_detected:
            verdict = EvaluationVerdict.FAIL
        elif not coverage_sufficient:
            verdict = EvaluationVerdict.INCONCLUSIVE
        else:
            verdict = EvaluationVerdict.PASS
        reported_score = (
            None if verdict is EvaluationVerdict.INCONCLUSIVE else score
        )
        findings_tuple = tuple(sorted(findings, key=lambda item: item.code))
        evaluation_id = stable_evaluation_id(
            self.module_id,
            self.evaluator_version,
            trace.trace_id,
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
            scope=EvaluationScope.TRAJECTORY,
            verdict=verdict,
            score=reported_score,
            assessment_confidence=confidence,
            metrics={
                "fact_count": len(trace.facts),
                "task_graph_fact_count": sum(
                    fact.category is TraceCategory.TASK_GRAPH
                    for fact in trace.facts
                ),
                "node_execution_fact_count": sum(
                    fact.category is TraceCategory.NODE_EXECUTION
                    for fact in trace.facts
                ),
                "tool_call_fact_count": sum(
                    fact.category is TraceCategory.TOOL_CALL
                    for fact in trace.facts
                ),
                "retry_count": len(retries),
                "timeout_count": len(timeouts),
                "ordering_violation_fact_count": (
                    len(invalid_action_order) + len(invalid_tool_order)
                ),
                "observed_coverage_sufficient": coverage_sufficient,
                "context_fact_count": len(
                    trace.facts_for(EvaluationComponent.CONTEXT)
                ),
                "memory_fact_count": len(
                    trace.facts_for(EvaluationComponent.MEMORY)
                ),
            },
            findings=findings_tuple,
            evaluated_at=subject.result.completed_at,
        )

    @staticmethod
    def _facts(
        facts: tuple[EvaluationFact, ...],
        kind: str,
    ) -> tuple[EvaluationFact, ...]:
        return tuple(fact for fact in facts if fact.kind == kind)

    @staticmethod
    def _has_failed_execution(payload: Mapping[str, JsonValue]) -> bool:
        observation = payload.get("observation")
        if isinstance(observation, Mapping):
            if observation.get("succeeded") is False:
                return True
        graph = payload.get("graph")
        if isinstance(graph, Mapping):
            nodes = graph.get("nodes")
            if isinstance(nodes, (list, tuple)):
                for node in nodes:
                    if isinstance(node, Mapping) and node.get("status") in {
                        "failed",
                        "blocked",
                    }:
                        return True
        return payload.get("status") in {"failed", "blocked"}

    @classmethod
    def _invalid_action_observation_order(
        cls,
        actions: tuple[EvaluationFact, ...],
        observations: tuple[EvaluationFact, ...],
    ) -> tuple[EvaluationFact, ...]:
        evidence: dict[UUID, EvaluationFact] = {}
        for action in actions:
            if action.correlation.action_id is None:
                continue
            matches = tuple(
                observation
                for observation in observations
                if observation.correlation.action_id
                == action.correlation.action_id
            )
            for observation in matches:
                if cls._precedes(action, observation) is False:
                    evidence[action.fact_id] = action
                    evidence[observation.fact_id] = observation
        return tuple(evidence[key] for key in sorted(evidence, key=str))

    @classmethod
    def _invalid_tool_order(
        cls,
        facts: tuple[EvaluationFact, ...],
    ) -> tuple[EvaluationFact, ...]:
        invocations: dict[UUID, list[EvaluationFact]] = {}
        for fact in facts:
            invocation_id = fact.correlation.invocation_id
            if fact.component is EvaluationComponent.TOOL and invocation_id:
                invocations.setdefault(invocation_id, []).append(fact)

        evidence: dict[UUID, EvaluationFact] = {}
        terminal_attempt_kinds = {
            "tool.attempt_succeeded",
            "tool.attempt_failed",
            "tool.attempt_timed_out",
        }
        failed_attempt_kinds = {
            "tool.attempt_failed",
            "tool.attempt_timed_out",
        }
        for invocation_facts in invocations.values():
            for later in invocation_facts:
                required_kind: str | None = None
                if later.kind in terminal_attempt_kinds:
                    required_kind = "tool.attempt_started"
                elif later.kind == "tool.retry_scheduled":
                    required_kind = "tool.attempt_failed"
                elif later.kind == "tool.execution_finished":
                    required_kind = "tool.execution_started"
                if required_kind is None:
                    continue

                candidates = tuple(
                    earlier
                    for earlier in invocation_facts
                    if (
                        earlier.kind == required_kind
                        or (
                            later.kind == "tool.retry_scheduled"
                            and earlier.kind in failed_attempt_kinds
                        )
                    )
                    and (
                        later.payload.get("attempt_number") is None
                        or earlier.payload.get("attempt_number")
                        == later.payload.get("attempt_number")
                    )
                )
                comparisons = tuple(
                    (candidate, cls._precedes(candidate, later))
                    for candidate in candidates
                )
                if comparisons and not any(
                    result is True for _, result in comparisons
                ):
                    comparable = tuple(
                        candidate
                        for candidate, result in comparisons
                        if result is False
                    )
                    if comparable:
                        evidence[later.fact_id] = later
                        for candidate in comparable:
                            evidence[candidate.fact_id] = candidate
        return tuple(evidence[key] for key in sorted(evidence, key=str))

    @staticmethod
    def _precedes(
        earlier: EvaluationFact,
        later: EvaluationFact,
    ) -> bool | None:
        if earlier.source_scope != later.source_scope:
            return None
        if (
            earlier.source_sequence is not None
            and later.source_sequence is not None
        ):
            return earlier.source_sequence < later.source_sequence
        if earlier.source == later.source:
            return earlier.occurred_at <= later.occurred_at
        return None

    @staticmethod
    def _observed_components(
        subject: EvaluationSubject,
    ) -> set[EvaluationComponent]:
        return {
            EvaluationComponent.RUNTIME,
            *(
                fact.component
                for fact in subject.trace.facts
                if fact.component
                in {
                    EvaluationComponent.ORCHESTRATION,
                    EvaluationComponent.TOOL,
                    EvaluationComponent.CONTEXT,
                    EvaluationComponent.MEMORY,
                }
            ),
        }

    @classmethod
    def _coverage_is_sufficient(cls, subject: EvaluationSubject) -> bool:
        coverage = tuple(
            subject.trace.coverage_for(component)
            for component in cls._observed_components(subject)
        )
        return all(
            item is not None
            and item.completeness is not TraceCompleteness.MISSING
            for item in coverage
        )

    @classmethod
    def _assessment_confidence(cls, subject: EvaluationSubject) -> float:
        observed_components = cls._observed_components(subject)
        scores = {
            TraceCompleteness.COMPLETE: 1.0,
            TraceCompleteness.PARTIAL: 0.75,
            TraceCompleteness.MISSING: 0.0,
        }
        coverage_scores = tuple(
            scores[coverage.completeness] if coverage is not None else 0.0
            for component in observed_components
            if (coverage := subject.trace.coverage_for(component)) is not None
        )
        missing_count = len(observed_components) - len(coverage_scores)
        return sum(coverage_scores) / (
            len(coverage_scores) + missing_count
        )

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
