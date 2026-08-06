from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from uuid import UUID, uuid4

from adaptive_agent_runtime.context_memory import (
    MemoryCondition,
    MemoryEvidence,
    MemoryRecallDraft,
    MemoryRecallEffect,
    MemoryRecallRequest,
    MemoryScope,
    MemoryUnit,
)
from adaptive_agent_runtime.decisioning import DecisionCheckpoint, DecisionFaultPoint
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import CapabilityTurnKind, CapabilityTurnResult
from adaptive_agent_runtime.persistence import SQLitePersistence

from applications.research_agent.memory_recall import (
    ResearchInitialMemoryRecallHandler,
)


class LedgerRecallAgent:
    module_id = "test.process.memory_recall"
    capability_id = "test.process.memory_recall"

    def __init__(self, persistence: SQLitePersistence) -> None:
        self._persistence = persistence

    async def propose_memory_recall(self, request, *, invocation=None):
        del invocation
        with self._persistence.database.transaction() as cursor:
            cursor.execute(
                "UPDATE closeout_recall_invocations SET calls = calls + 1 "
                "WHERE singleton = 1"
            )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=MemoryRecallDraft(
                selected_candidate_refs=(request.candidates[0].candidate_ref,),
                selection_reason="Select the persisted eligible candidate.",
            ),
        )


def handler(persistence: SQLitePersistence, *, crash: bool):
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )

    def fault(point, checkpoint):
        if crash and point is DecisionFaultPoint.EFFECT_COMMITTED:
            with persistence.database.transaction() as cursor:
                cursor.execute(
                    "INSERT OR REPLACE INTO closeout_recall_control "
                    "(singleton, request_id, run_id) VALUES (1, ?, ?)",
                    (str(checkpoint.request_id), str(checkpoint.run_id)),
                )
            os._exit(93)

    return ResearchInitialMemoryRecallHandler(
        memory_store=persistence.memory_store,
        bundle_store=persistence.memory_recall_bundle_store,
        capability=LedgerRecallAgent(persistence),
        governance=governance,
        reviews=persistence.human_review_service,
        issuer=persistence.authorization_issuer,
        operation_executor=persistence.operation_executor,
        trace_sink=persistence.trace_sink,
        checkpoint_store=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                MemoryRecallRequest, MemoryRecallDraft, MemoryRecallEffect
            ]
        ),
        fault_injector=fault,
    )


async def run(args) -> None:
    persistence = SQLitePersistence(Path(args.database))
    try:
        with persistence.database.transaction() as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS closeout_recall_invocations "
                "(singleton INTEGER PRIMARY KEY, calls INTEGER NOT NULL)"
            )
            cursor.execute(
                "INSERT OR IGNORE INTO closeout_recall_invocations VALUES (1, 0)"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS closeout_recall_control "
                "(singleton INTEGER PRIMARY KEY, request_id TEXT NOT NULL, run_id TEXT NOT NULL)"
            )
        if args.action == "start":
            candidate_id = uuid4()
            source = MemoryUnit(
                memory_key="research.process_resume",
                content={"principle": "preserve the original bundle"},
                condition=MemoryCondition(
                    facts={"domain": "financial_research"},
                    required_tags=("research",),
                ),
                evidence=(
                    MemoryEvidence(
                        source_reference="process-a",
                        note="cross-process provenance",
                    ),
                ),
                confidence=0.95,
                scope=MemoryScope(project_id="research", agent_scope="planner"),
                last_candidate_id=candidate_id,
                last_candidate_fingerprint="c" * 64,
            )
            await persistence.memory_store.save(source, expected_revision=None)
            run_id = uuid4()
            await handler(persistence, crash=True).recall(
                goal="分析 Acme 投资价值",
                run_id=run_id,
                task_id=uuid4(),
                scope=MemoryScope(project_id="research", agent_scope="planner"),
                facts={"domain": "financial_research"},
                tags=("research",),
            )
            raise AssertionError("fault injector did not terminate Process A")
        with persistence.database.reader() as cursor:
            row = cursor.execute(
                "SELECT request_id FROM closeout_recall_control WHERE singleton = 1"
            ).fetchone()
        assert row is not None
        result = await handler(persistence, crash=False).resume(
            UUID(row["request_id"])
        )
        assert result is not None
        print(result.bundle.model_dump_json())
    finally:
        persistence.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--action", choices=("start", "resume"), required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
