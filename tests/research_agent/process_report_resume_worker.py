from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sqlite3
from uuid import UUID

from adaptive_agent_runtime.decisioning import DecisionFaultPoint
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    GeneratedArtifactDraft,
    GenerationRequest,
)
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.cognition import ResearchCognitiveCapabilities
from applications.research_agent.report import ReportSection, ResearchReport


class PersistentReportGenerator:
    module_id = "test.report_generator.persistent"
    capability_id = "generation"

    def __init__(self, database: Path) -> None:
        self._database = database

    async def generate(
        self,
        request: GenerationRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GeneratedArtifactDraft]:
        del invocation
        connection = sqlite3.connect(self._database)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS closeout_report_invocations ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), calls INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO closeout_report_invocations(singleton, calls) VALUES (1, 1) "
                "ON CONFLICT(singleton) DO UPDATE SET calls = calls + 1"
            )
            connection.commit()
        finally:
            connection.close()
        context = request.context
        assert isinstance(context, Mapping)
        company = str(context["company"])
        report = ResearchReport(
            company=company,
            executive_summary="Persistent report generated once.",
            sections=(
                ReportSection(
                    title="Analysis",
                    findings=("Evidence-bound finding.",),
                ),
            ),
            risk_factors=("Bounded risk.",),
            investment_view="Balanced view.",
            markdown=f"# {company} Persistent Report",
        )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=GeneratedArtifactDraft(
                media_type=request.media_type,
                content=report.model_dump(mode="json"),
                evidence_reference_ids=tuple(
                    item.reference_id for item in request.evidence
                ),
            ),
        )


async def run(args: argparse.Namespace) -> None:
    path = Path(args.database).resolve()
    generator = PersistentReportGenerator(path)

    def fault(point, checkpoint):  # type: ignore[no-untyped-def]
        if (
            args.action == "start"
            and point is DecisionFaultPoint.EFFECT_COMMITTED
            and checkpoint.request.decision_type == "artifact.report_commit"
        ):
            os._exit(92)

    agent = ResearchAgent(
        persistence_path=path,
        cognitive_capabilities=ResearchCognitiveCapabilities(
            report_generator=generator
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
        receipt = result.report_commit_receipt
        print(
            json.dumps(
                {
                    "run_id": str(result.runtime_result.final_state.run_id),
                    "status": result.runtime_result.final_state.status.value,
                    "effect_fingerprint": (
                        receipt.effect_fingerprint if receipt is not None else None
                    ),
                    "artifact_fingerprint": (
                        receipt.artifact_fingerprint if receipt is not None else None
                    ),
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
