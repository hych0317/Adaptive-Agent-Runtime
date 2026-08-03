"""Deterministic outcome, trajectory, and component evaluation tests."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Mapping
from uuid import UUID

from pydantic import JsonValue, ValidationError

from adaptive_agent_runtime.evaluation import (
    AgentEvaluationPipeline,
    AgentExecutionTrace,
    ContextMemoryComponentEvaluator,
    DeterministicOutcomeEvaluator,
    DeterministicTrajectoryEvaluator,
    EvaluationComponent,
    EvaluationCorrelation,
    EvaluationCriteria,
    EvaluationFact,
    EvaluationStateSnapshot,
    EvaluationSubject,
    EvaluationVerdict,
    ExecutionResultSnapshot,
    ExecutionStatus,
    OrchestrationComponentEvaluator,
    ToolComponentEvaluator,
    TraceCategory,
    TraceCompleteness,
    TraceCoverage,
)


_NOW = datetime(2026, 8, 2, 1, 0, tzinfo=timezone.utc)
_RUN_ID = UUID("00000000-0000-0000-0000-000000000101")
_TASK_ID = UUID("00000000-0000-0000-0000-000000000102")
_ACTION_ID = UUID("00000000-0000-0000-0000-000000000103")


def _fact(
    index: int,
    *,
    component: EvaluationComponent,
    category: TraceCategory,
    kind: str,
    payload: Mapping[str, JsonValue] | None = None,
    action_id: UUID | None = None,
    invocation_id: UUID | None = None,
) -> EvaluationFact:
    return EvaluationFact(
        fact_id=UUID(int=1000 + index),
        component=component,
        category=category,
        kind=kind,
        source="test",
        occurred_at=_NOW + timedelta(seconds=index),
        correlation=EvaluationCorrelation(
            run_id=_RUN_ID,
            task_id=_TASK_ID,
            action_id=action_id,
            invocation_id=invocation_id,
        ),
        source_scope=f"test:{component.value}",
        source_sequence=index,
        source_record_id=str(index),
        payload=payload or {},
    )


def _subject(
    facts: tuple[EvaluationFact, ...],
    *,
    succeeded: bool = True,
    output: JsonValue = None,
    coverage: tuple[TraceCoverage, ...] | None = None,
) -> EvaluationSubject:
    status = ExecutionStatus.COMPLETED if succeeded else ExecutionStatus.FAILED
    error = None if succeeded else "execution failed"
    if coverage is None:
        observed_components = {
            EvaluationComponent.RUNTIME,
            *(fact.component for fact in facts),
        }
        coverage = tuple(
            TraceCoverage(
                run_id=_RUN_ID,
                component=component,
                completeness=(
                    TraceCompleteness.COMPLETE
                    if component is EvaluationComponent.RUNTIME
                    else TraceCompleteness.PARTIAL
                ),
            )
            for component in sorted(observed_components, key=lambda item: item.value)
        )
    trace = AgentExecutionTrace(
        trace_id=UUID("00000000-0000-0000-0000-000000000104"),
        run_id=_RUN_ID,
        task_id=_TASK_ID,
        facts=facts,
        coverage=coverage,
        collected_at=_NOW + timedelta(minutes=1),
    )
    return EvaluationSubject(
        trace=trace,
        state=EvaluationStateSnapshot(
            run_id=_RUN_ID,
            task_id=_TASK_ID,
            task_description="Evaluate a deterministic result",
            status=status,
            revision=3,
            step_count=1,
            output=output if succeeded else None,
            error=error,
            captured_at=_NOW + timedelta(minutes=1),
        ),
        result=ExecutionResultSnapshot(
            run_id=_RUN_ID,
            task_id=_TASK_ID,
            succeeded=succeeded,
            output=output if succeeded else None,
            error=error,
            completed_at=_NOW + timedelta(minutes=1),
        ),
    )


class OutcomeEvaluationTests(unittest.TestCase):
    def test_terminal_trace_conflict_cannot_pass(self) -> None:
        failed_trace = _fact(
            1,
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.failed",
        )
        subject = _subject((failed_trace,), output={"answer": 42})

        result = DeterministicOutcomeEvaluator().evaluate(
            subject,
            EvaluationCriteria(expected_output_values={"answer": 42}),
        )

        self.assertEqual(result.verdict, EvaluationVerdict.FAIL)
        self.assertIn(
            "outcome.execution_evidence_conflict",
            {finding.code for finding in result.findings},
        )
        conflict = next(
            finding
            for finding in result.findings
            if finding.code == "outcome.execution_evidence_conflict"
        )
        self.assertEqual(conflict.evidence_fact_ids, (failed_trace.fact_id,))

    def test_failed_execution_cannot_pass_when_requirements_are_optional(self) -> None:
        terminal = _fact(
            1,
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.failed",
        )
        subject = _subject((terminal,), succeeded=False)

        result = DeterministicOutcomeEvaluator().evaluate(
            subject,
            EvaluationCriteria(require_completion=False, require_output=False),
        )

        self.assertEqual(result.verdict, EvaluationVerdict.FAIL)
        self.assertIn(
            "outcome.runtime_failed",
            {finding.code for finding in result.findings},
        )

    def test_completed_result_satisfying_explicit_goal_passes(self) -> None:
        terminal = _fact(
            1,
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.completed",
        )
        subject = _subject((terminal,), output={"answer": 42, "source": "mock"})
        criteria = EvaluationCriteria(
            required_output_keys=("answer", "source"),
            expected_output_values={"answer": 42},
        )

        result = DeterministicOutcomeEvaluator().evaluate(subject, criteria)

        self.assertEqual(result.verdict, EvaluationVerdict.PASS)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metrics["goal_satisfaction_score"], 1.0)
        self.assertEqual(result.findings, ())

    def test_failed_result_reports_completion_and_output_failures(self) -> None:
        terminal = _fact(
            1,
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.failed",
            payload={"error": "execution failed"},
        )
        subject = _subject((terminal,), succeeded=False)

        result = DeterministicOutcomeEvaluator().evaluate(
            subject,
            EvaluationCriteria(required_output_keys=("answer",)),
        )

        self.assertEqual(result.verdict, EvaluationVerdict.FAIL)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(
            {finding.code for finding in result.findings},
            {
                "outcome.goal_not_satisfied",
                "outcome.output_missing",
                "outcome.runtime_failed",
            },
        )
        self.assertTrue(
            all(terminal.fact_id in item.evidence_fact_ids for item in result.findings)
        )


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_invalid_core_and_tool_source_order_is_detected(self) -> None:
        observation = _fact(
            1,
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.NODE_EXECUTION,
            kind="observation.received",
            action_id=_ACTION_ID,
        )
        action = _fact(
            2,
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.NODE_EXECUTION,
            kind="action.started",
            action_id=_ACTION_ID,
        )
        invocation_id = UUID("00000000-0000-0000-0000-000000000105")
        retry = _fact(
            3,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.retry_scheduled",
            payload={"attempt_number": 1},
            invocation_id=invocation_id,
        )
        attempt_failure = _fact(
            4,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.attempt_failed",
            payload={"attempt_number": 1},
            invocation_id=invocation_id,
        )
        subject = _subject(
            (observation, action, retry, attempt_failure),
            output={"answer": 42},
        )

        result = DeterministicTrajectoryEvaluator().evaluate(subject)

        self.assertEqual(result.verdict, EvaluationVerdict.FAIL)
        self.assertEqual(
            {
                "trajectory.invalid_action_observation_order",
                "trajectory.invalid_tool_event_order",
            },
            {
                finding.code
                for finding in result.findings
                if finding.code.startswith("trajectory.invalid_")
            },
        )

    def test_clean_recovered_and_broken_trajectories_are_ordered(self) -> None:
        action = _fact(
            1,
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.NODE_EXECUTION,
            kind="action.started",
            action_id=_ACTION_ID,
        )
        observation = _fact(
            2,
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.NODE_EXECUTION,
            kind="observation.received",
            action_id=_ACTION_ID,
        )
        clean_subject = _subject((action, observation), output={"answer": 42})

        attempt_failure = _fact(
            3,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.attempt_failed",
            action_id=_ACTION_ID,
        )
        retry = _fact(
            4,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.retry_scheduled",
            action_id=_ACTION_ID,
        )
        recovered_subject = _subject(
            (action, observation, retry, attempt_failure),
            output={"answer": 42},
        )

        timeout = _fact(
            5,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.attempt_timed_out",
            action_id=_ACTION_ID,
        )
        failed_node = _fact(
            6,
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.NODE_EXECUTION,
            kind="orchestration.node_finished",
            payload={"status": "failed"},
        )
        broken_subject = _subject(
            (action, timeout, failed_node),
            succeeded=False,
        )
        evaluator = DeterministicTrajectoryEvaluator()

        clean = evaluator.evaluate(clean_subject)
        recovered = evaluator.evaluate(recovered_subject)
        broken = evaluator.evaluate(broken_subject)

        self.assertGreater(clean.score or 0.0, recovered.score or 0.0)
        self.assertGreater(recovered.score or 0.0, broken.score or 0.0)
        self.assertEqual(clean.verdict, EvaluationVerdict.PASS)
        self.assertEqual(recovered.verdict, EvaluationVerdict.PASS)
        self.assertEqual(broken.verdict, EvaluationVerdict.FAIL)
        self.assertIn(
            "trajectory.missing_observation",
            {finding.code for finding in broken.findings},
        )
        self.assertIn(
            "trajectory.node_failure",
            {finding.code for finding in broken.findings},
        )


class ComponentEvaluationTests(unittest.TestCase):
    def test_observed_fact_with_missing_coverage_is_inconclusive(self) -> None:
        tool = _fact(
            1,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.execution_finished",
            payload={"data": {"status": "succeeded"}},
        )
        subject = _subject(
            (tool,),
            output={"answer": 42},
            coverage=(
                TraceCoverage(
                    run_id=_RUN_ID,
                    component=EvaluationComponent.RUNTIME,
                    completeness=TraceCompleteness.COMPLETE,
                ),
                TraceCoverage(
                    run_id=_RUN_ID,
                    component=EvaluationComponent.TOOL,
                    completeness=TraceCompleteness.MISSING,
                ),
            ),
        )

        result = ToolComponentEvaluator().evaluate(subject)
        trajectory = DeterministicTrajectoryEvaluator().evaluate(subject)

        self.assertEqual(result.verdict, EvaluationVerdict.INCONCLUSIVE)
        self.assertIsNone(result.score)
        self.assertEqual(result.assessment_confidence, 0.0)
        self.assertEqual(trajectory.verdict, EvaluationVerdict.INCONCLUSIVE)
        self.assertIsNone(trajectory.score)

    def test_component_evaluators_use_only_scoped_trace_evidence(self) -> None:
        orchestration = _fact(
            1,
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.NODE_EXECUTION,
            kind="orchestration.node_finished",
            payload={"status": "failed"},
        )
        tool = _fact(
            2,
            component=EvaluationComponent.TOOL,
            category=TraceCategory.TOOL_CALL,
            kind="tool.selection_failed",
            payload={"reason_code": "capability_mismatch"},
        )
        context = _fact(
            3,
            component=EvaluationComponent.CONTEXT,
            category=TraceCategory.CONTEXT_CHANGE,
            kind="context.failed",
        )
        memory = _fact(
            4,
            component=EvaluationComponent.MEMORY,
            category=TraceCategory.MEMORY_OPERATION,
            kind="memory.conflict",
            payload={"status": "conflicted"},
        )
        coverage = tuple(
            TraceCoverage(
                run_id=_RUN_ID,
                component=component,
                completeness=TraceCompleteness.PARTIAL,
            )
            for component in (
                EvaluationComponent.RUNTIME,
                EvaluationComponent.ORCHESTRATION,
                EvaluationComponent.TOOL,
                EvaluationComponent.CONTEXT,
                EvaluationComponent.MEMORY,
            )
        )
        subject = _subject(
            (orchestration, tool, context, memory),
            succeeded=False,
            coverage=coverage,
        )

        orchestration_result = OrchestrationComponentEvaluator().evaluate(subject)
        tool_result = ToolComponentEvaluator().evaluate(subject)
        context_memory_result = ContextMemoryComponentEvaluator().evaluate(subject)

        for result in (
            orchestration_result,
            tool_result,
            context_memory_result,
        ):
            self.assertEqual(result.verdict, EvaluationVerdict.FAIL)
        self.assertEqual(
            {item.code for item in tool_result.findings},
            {"component.tool.capability_mismatch"},
        )
        self.assertEqual(tool_result.findings[0].evidence_fact_ids, (tool.fact_id,))
        self.assertEqual(
            {item.code for item in context_memory_result.findings},
            {"component.context.operation_failed", "component.memory.conflict"},
        )
        self.assertEqual(
            {
                finding.component
                for finding in context_memory_result.findings
            },
            {EvaluationComponent.CONTEXT, EvaluationComponent.MEMORY},
        )

    def test_unobserved_component_is_inconclusive(self) -> None:
        runtime = _fact(
            1,
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.completed",
        )
        subject = _subject((runtime,), output={"answer": 42})

        result = ToolComponentEvaluator().evaluate(subject)

        self.assertEqual(result.verdict, EvaluationVerdict.INCONCLUSIVE)
        self.assertIsNone(result.score)
        self.assertEqual(result.metrics["observed_fact_count"], 0)

    def test_pipeline_keeps_outcome_trajectory_and_components_separate(self) -> None:
        final = _fact(
            1,
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.completed",
        )
        subject = _subject((final,), output={"answer": 42})
        pipeline = AgentEvaluationPipeline(
            outcome=DeterministicOutcomeEvaluator(),
            trajectory=DeterministicTrajectoryEvaluator(),
            components=(ToolComponentEvaluator(),),
        )

        report = pipeline.evaluate(
            subject,
            EvaluationCriteria(expected_output_values={"answer": 42}),
        )

        self.assertEqual(report.outcome.verdict, EvaluationVerdict.PASS)
        self.assertEqual(report.trajectory.verdict, EvaluationVerdict.PASS)
        self.assertEqual(report.components[0].verdict, EvaluationVerdict.INCONCLUSIVE)
        self.assertEqual(report.results, (
            report.outcome,
            report.trajectory,
            report.components[0],
        ))

    def test_subject_rejects_inconsistent_state_and_execution_result(self) -> None:
        valid = _subject((), output={"answer": 42})
        failed_result = ExecutionResultSnapshot(
            run_id=_RUN_ID,
            task_id=_TASK_ID,
            succeeded=False,
            error="failed",
            completed_at=_NOW + timedelta(minutes=1),
        )

        with self.assertRaises(ValidationError):
            EvaluationSubject(
                trace=valid.trace,
                state=valid.state,
                result=failed_result,
            )


if __name__ == "__main__":
    unittest.main()
