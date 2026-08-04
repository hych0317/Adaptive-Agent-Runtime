"""Append-only storage for validated semantic Root Cause assessments."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from adaptive_agent_runtime.evaluation.root_cause import RootCauseAssessment


class InMemoryRootCauseAssessmentStore:
    module_id = "evaluation.root_cause_store.in_memory"

    def __init__(self) -> None:
        self._by_id: dict[UUID, RootCauseAssessment] = {}
        self._by_run: defaultdict[UUID, list[UUID]] = defaultdict(list)

    async def record(self, assessment: RootCauseAssessment) -> None:
        existing = self._by_id.get(assessment.assessment_id)
        if existing is not None:
            if existing != assessment:
                raise ValueError("Root Cause assessment id has conflicting content")
            return
        self._by_id[assessment.assessment_id] = assessment
        self._by_run[assessment.correlation.run_id].append(
            assessment.assessment_id
        )

    async def load(self, assessment_id: UUID) -> RootCauseAssessment | None:
        return self._by_id.get(assessment_id)

    async def list_for_run(self, run_id: UUID) -> tuple[RootCauseAssessment, ...]:
        return tuple(self._by_id[item] for item in self._by_run.get(run_id, ()))

    async def list_all(self) -> tuple[RootCauseAssessment, ...]:
        return tuple(
            sorted(
                self._by_id.values(),
                key=lambda item: (item.assessed_at, str(item.assessment_id)),
            )
        )
