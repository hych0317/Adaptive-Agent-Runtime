"""Replay execution and deterministic non-regression validation."""

from __future__ import annotations

from typing import Sequence

from adaptive_agent_runtime.evolution.contracts import (
    ReplayValidationPolicy,
    ReplayWorkloadExecutor,
)
from adaptive_agent_runtime.evolution.models import (
    ReplayCase,
    ReplayObservation,
    ReplayValidation,
    RuntimeConfigurationSnapshot,
)


class RuntimeReplayRunner:
    """Execute persisted workloads in an application-supplied isolated Runtime."""

    module_id = "evolution.replay.runtime"

    def __init__(self, executor: ReplayWorkloadExecutor) -> None:
        self._executor = executor

    async def replay(
        self,
        cases: Sequence[ReplayCase],
        candidate: RuntimeConfigurationSnapshot,
    ) -> tuple[ReplayObservation, ...]:
        observations: list[ReplayObservation] = []
        for case in cases:
            result = await self._executor.execute(case, candidate)
            observations.append(
                ReplayObservation(
                    case_id=case.case_id,
                    source_run_id=case.source_run_id,
                    candidate_component=candidate.component,
                    candidate_version=candidate.version,
                    execution=result,
                )
            )
        return tuple(observations)


class DeterministicReplayValidator:
    module_id = "evolution.replay_validator.deterministic"

    def __init__(self, *, max_mean_regression: float = 0.0) -> None:
        if not 0.0 <= max_mean_regression <= 1.0:
            raise ValueError("maximum replay regression must be between zero and one")
        self._max_mean_regression = max_mean_regression

    def validate(
        self,
        cases: Sequence[ReplayCase],
        results: Sequence[ReplayObservation],
        candidate: RuntimeConfigurationSnapshot,
    ) -> ReplayValidation:
        if not cases or len(cases) != len(results):
            raise ValueError("Replay validation requires one result per case")
        case_by_id = {case.case_id: case for case in cases}
        if len(case_by_id) != len(cases):
            raise ValueError("Replay cases must be unique")
        findings: list[str] = []
        for observation in results:
            if (
                observation.candidate_component != candidate.component
                or observation.candidate_version != candidate.version
            ):
                raise ValueError(
                    "Replay result was produced for another candidate version"
                )
            case = case_by_id.get(observation.case_id)
            if case is None:
                raise ValueError("Replay result references an unknown case")
            if not observation.execution.succeeded:
                findings.append(f"case:{case.case_id}:execution_failed")
            if observation.execution.score < case.minimum_score:
                findings.append(f"case:{case.case_id}:below_minimum")
        baseline = sum(case.baseline_score for case in cases) / len(cases)
        replayed = sum(item.execution.score for item in results) / len(results)
        if replayed + self._max_mean_regression < baseline:
            findings.append("candidate_mean_score_regressed")
        return ReplayValidation(
            passed=not findings,
            baseline_score=baseline,
            candidate_score=replayed,
            findings=tuple(findings),
            observations=tuple(results),
        )
