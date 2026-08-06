from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sqlite3
import sys
from uuid import UUID

from adaptive_agent_runtime.decisioning import DecisionFaultPoint
from adaptive_agent_runtime.experience_learning import EXPERIENCE_LEARNING_DECISION_TYPE
from adaptive_agent_runtime.llm import CapabilityTurnKind, CapabilityTurnResult
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.cognition import ResearchCognitiveCapabilities
from applications.research_agent.experience_learning import (
    DeterministicLearningAssessmentCapability,
)


class LedgerLearningCapability(DeterministicLearningAssessmentCapability):
    module_id = "test.learning.ledger"
    capability_id = "test.learning.ledger"

    def __init__(self, database: Path) -> None:
        self._database = database

    async def assess_learning(self, request, *, invocation=None):  # type: ignore[no-untyped-def]
        connection = sqlite3.connect(self._database)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS learning_agent_invocations "
                "(singleton INTEGER PRIMARY KEY, invocation_count INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO learning_agent_invocations(singleton, invocation_count) "
                "VALUES (1, 1) ON CONFLICT(singleton) DO UPDATE SET "
                "invocation_count = invocation_count + 1"
            )
            connection.commit()
        finally:
            connection.close()
        return await super().assess_learning(request, invocation=invocation)


def crash(
    database: Path,
    marker: Path,
    fault_target: DecisionFaultPoint,
) -> None:
    async def execute() -> None:
        first = ResearchAgent(persistence_path=database)
        try:
            await first.run("分析 Acme 的投资价值")
        finally:
            first.close()

        def fault(point, checkpoint):  # type: ignore[no-untyped-def]
            if (
                point is fault_target
                and checkpoint.request.decision_type
                == EXPERIENCE_LEARNING_DECISION_TYPE
            ):
                marker.write_text(str(checkpoint.run_id), encoding="utf-8")
                os._exit(73)

        second = ResearchAgent(
            persistence_path=database,
            cognitive_capabilities=ResearchCognitiveCapabilities(
                experience_learner=LedgerLearningCapability(database)
            ),
            decision_fault_injector=fault,
        )
        await second.run("分析 Acme 的长期投资价值")

    asyncio.run(execute())
    raise SystemExit("fault point was not reached")


def resume(database: Path, marker: Path) -> None:
    run_id = UUID(marker.read_text(encoding="utf-8").strip())
    agent = ResearchAgent(
        persistence_path=database,
        cognitive_capabilities=ResearchCognitiveCapabilities(
            experience_learner=LedgerLearningCapability(database)
        ),
    )
    try:
        result = asyncio.run(agent.resume(run_id))
    finally:
        agent.close()
    connection = sqlite3.connect(database)
    try:
        invocation_count = connection.execute(
            "SELECT invocation_count FROM learning_agent_invocations WHERE singleton = 1"
        ).fetchone()[0]
        insight_count = connection.execute(
            "SELECT COUNT(*) FROM learning_insights"
        ).fetchone()[0]
    finally:
        connection.close()
    insight = result.learning_insights[0]
    print(
        json.dumps(
            {
                "run_id": str(run_id),
                "invocation_count": invocation_count,
                "insight_count": insight_count,
                "learning_insight_id": str(insight.learning_insight_id),
                "effect_fingerprint": insight.effect_fingerprint,
                "evidence_set_fingerprint": insight.evidence_set_fingerprint,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    database = Path(sys.argv[2])
    marker = Path(sys.argv[3])
    if mode == "crash":
        crash(database, marker, DecisionFaultPoint(sys.argv[4]))
    elif mode == "resume":
        resume(database, marker)
    else:
        raise SystemExit(f"unknown mode: {mode}")
