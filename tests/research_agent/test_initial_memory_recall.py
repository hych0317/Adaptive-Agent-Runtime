from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from adaptive_agent_runtime.context_memory import MemoryRecallDraft
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    TaskGraphDraft,
    TaskPlanningRequest,
)
from adaptive_agent_runtime.persistence import SQLitePersistence

from applications.research_agent.agent import ResearchAgent
from applications.research_agent.cognition import ResearchCognitiveCapabilities
from applications.research_agent.tasks import (
    COMPANY_RESEARCH,
    build_research_task_draft,
)


class SelectingRecallAgent:
    module_id = "test.research.memory_recall"
    capability_id = "test.research.memory_recall"

    def __init__(self) -> None:
        self.requests = []

    async def propose_memory_recall(
        self,
        request,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ):
        del invocation
        self.requests.append(request)
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=MemoryRecallDraft(
                selected_candidate_refs=(request.candidates[0].candidate_ref,),
                selection_reason="Use one prior planning experience.",
            ),
        )


class RecallAwarePlanner:
    module_id = "test.research.recall_aware_planner"
    capability_id = "test.research.recall_aware_planner"

    def __init__(self, company: str) -> None:
        self.company = company
        self.requests: list[TaskPlanningRequest] = []

    async def propose(
        self,
        request: TaskPlanningRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[TaskGraphDraft]:
        del invocation
        self.requests.append(request)
        draft = build_research_task_draft(self.company)
        if request.committed_memory_recall_bundle is not None:
            nodes = tuple(
                (
                    node.model_copy(
                        update={
                            "goal": node.goal + " using committed historical experience"
                        }
                    )
                    if node.node_key == COMPANY_RESEARCH
                    else node
                )
                for node in draft.nodes
            )
            draft = draft.model_copy(
                update={
                    "nodes": nodes,
                    "rationale": "Use the committed Recall Bundle as advisory context.",
                }
            )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=draft,
        )


def planning_checkpoint(path: Path) -> dict:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT requests.canonical_payload AS request_json, "
            "proposals.canonical_payload AS proposal_json "
            "FROM decision_current AS current "
            "JOIN decision_requests AS requests "
            "ON requests.request_id = current.request_ref "
            "JOIN decision_proposals AS proposals "
            "ON proposals.proposal_id = current.proposal_ref "
            "WHERE current.decision_type = 'planning.task_graph.initialize' "
            "ORDER BY current.updated_at DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return {
        "request": json.loads(row["request_json"]),
        "proposal": json.loads(row["proposal_json"]),
    }


class InitialPlanningMemoryRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_candidate_planning_continues_without_recall(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            recall = SelectingRecallAgent()
            planner = RecallAwarePlanner("Acme")
            agent = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    memory_recall=recall,
                    task_planner=planner,
                ),
            )
            result = await agent.run("分析 Acme 投资价值")
            self.assertTrue(result.runtime_result.succeeded)
            self.assertIsNone(result.memory_recall_bundle)
            self.assertEqual(recall.requests, [])
            self.assertIsNone(planner.requests[0].committed_memory_recall_bundle)
            agent.close()

    async def test_cross_run_memory_is_recalled_for_initial_planning(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            first = ResearchAgent(persistence_path=path)
            run_a = await first.run("分析 Acme 投资价值")
            self.assertIsNone(run_a.memory_recall_bundle)
            first.close()

            recall_agent = SelectingRecallAgent()
            planner = RecallAwarePlanner("Acme")
            second = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    memory_recall=recall_agent,
                    task_planner=planner,
                ),
            )
            run_b = await second.run("分析 Acme 投资价值")
            self.assertIsNotNone(run_b.memory_recall_bundle)
            self.assertEqual(len(recall_agent.requests), 1)
            self.assertEqual(len(planner.requests), 1)
            projected = planner.requests[0].committed_memory_recall_bundle
            self.assertIsNotNone(projected)
            assert run_b.memory_recall_bundle is not None
            self.assertEqual(
                projected["bundle_fingerprint"],  # type: ignore[index]
                decision_fingerprint(run_b.memory_recall_bundle),
            )
            company_node = next(
                node for node in run_b.task_graph.nodes
                if "committed historical experience" in node.goal
            )
            self.assertIn("committed historical experience", company_node.goal)
            verification = SQLitePersistence(path)
            persisted = await verification.memory_recall_bundle_store.load_for_run(
                run_b.runtime_result.final_state.run_id
            )
            self.assertEqual(persisted, run_b.memory_recall_bundle)
            second.close()
            verification.close()

    async def test_planning_context_binds_recall_bundle_fingerprint(self) -> None:
        with TemporaryDirectory() as recalled_dir, TemporaryDirectory() as empty_dir:
            recalled_path = Path(recalled_dir) / "runtime.sqlite3"
            seed = ResearchAgent(persistence_path=recalled_path)
            await seed.run("分析 Acme 投资价值")
            seed.close()

            recalled_planner = RecallAwarePlanner("Acme")
            recalled = ResearchAgent(
                persistence_path=recalled_path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    memory_recall=SelectingRecallAgent(),
                    task_planner=recalled_planner,
                ),
            )
            recalled_result = await recalled.run("分析 Acme 投资价值")
            recalled_checkpoint = planning_checkpoint(recalled_path)
            recalled.close()

            empty_path = Path(empty_dir) / "runtime.sqlite3"
            empty_planner = RecallAwarePlanner("Acme")
            empty = ResearchAgent(
                persistence_path=empty_path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    memory_recall=SelectingRecallAgent(),
                    task_planner=empty_planner,
                ),
            )
            empty_result = await empty.run("分析 Acme 投资价值")
            empty_checkpoint = planning_checkpoint(empty_path)
            empty.close()

            self.assertIsNotNone(recalled_result.memory_recall_bundle)
            self.assertIsNone(empty_result.memory_recall_bundle)
            self.assertNotEqual(
                recalled_checkpoint["proposal"]["context_fingerprint"],
                empty_checkpoint["proposal"]["context_fingerprint"],
            )
            self.assertNotEqual(
                recalled_checkpoint["request"]["basis"]["snapshot_fingerprint"],
                empty_checkpoint["request"]["basis"]["snapshot_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
