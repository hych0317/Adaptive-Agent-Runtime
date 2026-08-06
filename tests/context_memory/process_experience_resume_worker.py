from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    ExperienceAssessmentAgentRequest,
    ExperienceAssessmentDraft,
)
from adaptive_agent_runtime.decisioning import DecisionFaultPoint
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
)
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.cognition import ResearchCognitiveCapabilities


class PersistentExperienceAssessor:
    module_id = "test.experience.assessor.persistent"
    capability_id = "experience.assessment.persistent"

    def __init__(self, database: Path) -> None:
        self._database = database

    async def assess_experience(
        self,
        request: ExperienceAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ExperienceAssessmentDraft]:
        del request, invocation
        connection = sqlite3.connect(self._database)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS experience_agent_invocations ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                "calls INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO experience_agent_invocations(singleton, calls) "
                "VALUES (1, 1) ON CONFLICT(singleton) "
                "DO UPDATE SET calls = calls + 1"
            )
            connection.commit()
        finally:
            connection.close()
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=ExperienceAssessmentDraft(
                observed_pattern="A persistent completed execution was observed.",
                possible_relevance="May be relevant to another research run.",
                explanation="Advisory only.",
            ),
        )


async def run(args: argparse.Namespace) -> None:
    path = Path(args.database).resolve()

    def fault(point, checkpoint):  # type: ignore[no-untyped-def]
        if (
            args.action == "start"
            and point is DecisionFaultPoint.EFFECT_COMMITTED
            and checkpoint.request.decision_type == "experience.assessment"
        ):
            os._exit(94)

    agent = ResearchAgent(
        persistence_path=path,
        cognitive_capabilities=ResearchCognitiveCapabilities(
            experience_assessor=PersistentExperienceAssessor(path)
        ),
        decision_fault_injector=fault,
    )
    try:
        if args.action == "start":
            result = await agent.run("分析 Acme 的投资价值")
        else:
            if args.run_id is None:
                raise RuntimeError("resume requires run id")
            result = await agent.resume(UUID(args.run_id))
        metadata = result.experience_metadata
        print(
            json.dumps(
                {
                    "run_id": str(result.runtime_result.final_state.run_id),
                    "experience_id": str(metadata.experience_id) if metadata else None,
                    "effect_fingerprint": (
                        metadata.effect_fingerprint if metadata else None
                    ),
                    "version": metadata.version if metadata else None,
                }
            )
        )
    finally:
        agent.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--action", choices=("start", "resume"), required=True)
    parser.add_argument("--run-id")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
