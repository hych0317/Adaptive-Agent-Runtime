"""Read-only contracts for trace collection, evaluation, and proposals."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.evaluation.models import (
    AgentExecutionTrace,
    ConservativeOptimizationPolicy,
    EvaluationComponent,
    EvaluationCriteria,
    EvaluationResult,
    EvaluationSubject,
    FailureAnalysis,
    OptimizationProposal,
    OutputQualityAssessment,
    TraceBatch,
)


@runtime_checkable
class TraceCollector(RuntimeModule, Protocol):
    def collect(
        self,
        *,
        run_id: UUID,
        task_id: UUID,
        batches: Sequence[TraceBatch],
    ) -> AgentExecutionTrace: ...


@runtime_checkable
class OutputQualityEvaluator(RuntimeModule, Protocol):
    def assess(
        self,
        output: JsonValue,
        criteria: EvaluationCriteria,
    ) -> OutputQualityAssessment: ...


@runtime_checkable
class OutcomeEvaluator(RuntimeModule, Protocol):
    def evaluate(
        self,
        subject: EvaluationSubject,
        criteria: EvaluationCriteria,
    ) -> EvaluationResult: ...


@runtime_checkable
class TrajectoryEvaluator(RuntimeModule, Protocol):
    def evaluate(self, subject: EvaluationSubject) -> EvaluationResult: ...


@runtime_checkable
class ComponentEvaluator(RuntimeModule, Protocol):
    @property
    def component(self) -> EvaluationComponent: ...

    def evaluate(self, subject: EvaluationSubject) -> EvaluationResult: ...


@runtime_checkable
class FailureAnalyzer(RuntimeModule, Protocol):
    def analyze(
        self,
        results: Sequence[EvaluationResult],
    ) -> FailureAnalysis: ...


@runtime_checkable
class OptimizationAgent(RuntimeModule, Protocol):
    def propose(
        self,
        analysis: FailureAnalysis,
        policy: ConservativeOptimizationPolicy,
    ) -> tuple[OptimizationProposal, ...]: ...
