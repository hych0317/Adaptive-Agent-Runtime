"""Adapters from completed Evaluation runs into durable Replay workloads."""

from __future__ import annotations

from adaptive_agent_runtime.evaluation import EvaluationReport, EvaluationSubject
from adaptive_agent_runtime.evolution.models import ReplayCase, stable_evolution_id


class EvaluationReplayCaseFactory:
    module_id = "evolution.replay_case.evaluation"

    def create(
        self,
        subject: EvaluationSubject,
        report: EvaluationReport,
        *,
        minimum_score: float = 0.0,
    ) -> ReplayCase:
        if (
            subject.state.run_id != report.run_id
            or subject.state.task_id != report.task_id
        ):
            raise ValueError("Evaluation subject and report identify another run")
        scores = tuple(
            item.score
            for item in report.results
            if item.score is not None
        )
        if not scores:
            raise ValueError("Replay case requires a scored Evaluation report")
        baseline_score = (
            report.overall_score
            if report.overall_score is not None
            else sum(scores) / len(scores)
        )
        return ReplayCase(
            case_id=stable_evolution_id(
                "replay-case",
                report.run_id,
                report.report_id,
            ),
            source_run_id=report.run_id,
            task_id=report.task_id,
            input_payload={
                "task_description": subject.state.task_description,
                "trace_id": str(subject.trace.trace_id),
                "report_id": str(report.report_id),
                "source_state_revision": subject.state.revision,
            },
            baseline_score=baseline_score,
            minimum_score=minimum_score,
            created_at=report.created_at,
        )

