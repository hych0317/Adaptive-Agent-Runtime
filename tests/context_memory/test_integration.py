from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    Observation,
    RunStatus,
)
from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    ContextAssembler,
    ContextLayer,
    ContextMemoryCoordinator,
    ContextMetadata,
    ContextScheduler,
    ContextSource,
    ContextUnit,
    DefaultTaskContextRequirementProvider,
    EvidenceDrivenMemoryConsolidator,
    ExplicitMemoryCandidateFactory,
    InMemoryContextStore,
    InMemoryMemoryStore,
    MemoryCondition,
    MemoryCandidate,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryRecallQuery,
    ObservationContextAdapter,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    MockExecutionStrategy,
    NodeExecutionResult,
    StrategyActionExecutor,
    TaskNode,
    TaskNodeStatus,
)


def task_node(goal: str) -> TaskNode:
    return TaskNode(
        goal=goal,
        expected_output=f"{goal} output",
        strategy_id="mock",
    )


class ContextMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_access_timestamp_does_not_regress_with_wall_clock(self) -> None:
        now = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
        run_id = uuid4()
        unit = ContextUnit(
            content={"fact": "stable"},
            metadata=ContextMetadata(
                source=ContextSource.DOCUMENT,
                layer=ContextLayer.TASK,
                run_id=run_id,
                created_at=now,
                updated_at=now,
            ),
        )
        store = InMemoryContextStore()
        await store.save(unit, expected_revision=None)
        coordinator = ContextMemoryCoordinator(
            context_store=store,
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=DefaultTaskContextRequirementProvider(),
            memory_recall=ConditionalMemoryRecall(InMemoryMemoryStore()),
        )

        with patch(
            "adaptive_agent_runtime.context_memory.integration.utc_now",
            return_value=now - timedelta(milliseconds=25),
        ):
            await coordinator._mark_accessed({unit.context_id})

        updated = await store.load(unit.context_id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.metadata.created_at, now)
        self.assertEqual(updated.metadata.updated_at, now)
        self.assertEqual(updated.metadata.last_accessed_at, now)
        self.assertEqual(updated.metadata.access_count, 1)

    async def test_observation_memory_recall_and_next_context_flow(self) -> None:
        first_node = task_node("collect evidence")
        next_node = task_node("analyze evidence")
        first_snapshot = first_node.model_dump(mode="json")
        planner = DynamicTaskGraphPlanner(DynamicTaskGraph(nodes=(first_node,)))
        strategy = MockExecutionStrategy(
            {
                first_node.node_id: NodeExecutionResult.ok(
                    output={"finding": "audited evidence is reliable"}
                )
            }
        )
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            max_steps=4,
        )
        result = await runtime.run(AgentTask(description="research"))
        state = result.final_state
        observation = state.last_observation
        assert observation is not None
        completed_graph = planner.graph_for(state.run_id)
        graph_snapshot = completed_graph.model_dump(mode="json")

        self.assertEqual(state.status, RunStatus.COMPLETED)
        self.assertEqual(strategy.executed_node_ids, [first_node.node_id])
        self.assertEqual(completed_graph.nodes[0].status, TaskNodeStatus.COMPLETED)

        context_store = InMemoryContextStore()
        memory_store = InMemoryMemoryStore()
        coordinator = ContextMemoryCoordinator(
            context_store=context_store,
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=DefaultTaskContextRequirementProvider(),
            memory_recall=ConditionalMemoryRecall(memory_store),
        )
        observation_context = await coordinator.record_observation(
            observation,
            state,
            node=first_node,
        )
        candidate = ExplicitMemoryCandidateFactory().from_context(
            observation_context,
            memory_key="research.source_policy",
            content="prefer audited evidence",
            condition=MemoryCondition(
                facts={"market": "CN"},
                required_tags=("research",),
            ),
            confidence=0.8,
            evolution=MemoryEvolutionType.EXTEND,
            note="reliable evidence affected source selection",
        )
        self.assertEqual(candidate.content, "prefer audited evidence")
        self.assertEqual(candidate.evidence[0].source_context_id, observation_context.context_id)
        self.assertEqual(await memory_store.list_all(), ())
        await EvidenceDrivenMemoryConsolidator(memory_store).consolidate(candidate)

        assembly = await coordinator.prepare_next_context(
            state,
            next_node,
            memory_query=MemoryRecallQuery(
                facts={"market": "CN"},
                tags=("research",),
            ),
        )

        self.assertEqual(
            tuple(unit.metadata.source for unit in assembly.units),
            (ContextSource.OBSERVATION, ContextSource.MEMORY_RECALL),
        )
        self.assertEqual(assembly.requirement.node_id, next_node.node_id)
        self.assertEqual(
            planner.graph_for(state.run_id).model_dump(mode="json"),
            graph_snapshot,
        )
        self.assertEqual(first_node.model_dump(mode="json"), first_snapshot)
        self.assertTrue(
            all(
                "context" not in name.lower() and "memory" not in name.lower()
                for name in TaskNode.model_fields
            )
        )
        self.assertTrue(
            all(
                "context" not in name.lower() and "memory" not in name.lower()
                for name in DynamicTaskGraph.model_fields
            )
        )

    async def test_conflicted_recall_projection_preserves_uncertainty(self) -> None:
        condition = MemoryCondition(facts={"market": "CN"})
        memory_store = InMemoryMemoryStore()
        consolidator = EvidenceDrivenMemoryConsolidator(memory_store)
        created = await consolidator.consolidate(
            MemoryCandidate(
                memory_key="policy",
                content="use primary sources",
                condition=condition,
                evidence=(MemoryEvidence(source_reference="source:1", note="initial"),),
                confidence=0.8,
                evolution=MemoryEvolutionType.EXTEND,
            )
        )
        await consolidator.consolidate(
            MemoryCandidate(
                memory_key="policy",
                content="use secondary sources",
                condition=condition,
                evidence=(MemoryEvidence(source_reference="source:2", note="counter"),),
                confidence=0.9,
                evolution=MemoryEvolutionType.CONFLICT,
                target_memory_id=created.memory.memory_id,
            )
        )
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="conflict projection"),
            status=RunStatus.RUNNING,
        )
        coordinator = ContextMemoryCoordinator(
            context_store=InMemoryContextStore(),
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=DefaultTaskContextRequirementProvider(),
            memory_recall=ConditionalMemoryRecall(memory_store),
        )

        assembly = await coordinator.prepare_next_context(
            state,
            task_node("resolve conflict"),
            memory_query=MemoryRecallQuery(
                facts={"market": "CN"},
                include_conflicted=True,
            ),
        )

        self.assertEqual(len(assembly.semantic_context), 1)
        content = cast(dict[str, Any], assembly.semantic_context[0].content)
        self.assertEqual(content["status"], "conflicted")
        self.assertEqual(content["conflicts"][0]["content"], "use secondary sources")

    async def test_failed_observation_becomes_task_context(self) -> None:
        node = task_node("failing task")
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="failure"),
            status=RunStatus.RUNNING,
        )
        observation = Observation.failed(uuid4(), error="expected failure")

        unit = ObservationContextAdapter().convert(
            observation,
            state,
            node=node,
        )

        self.assertEqual(unit.metadata.source, ContextSource.OBSERVATION)
        self.assertIn("failure", unit.metadata.tags)
        content = cast(dict[str, Any], unit.content)
        self.assertEqual(content["error"], "expected failure")

    async def test_recording_the_same_observation_is_idempotent(self) -> None:
        node = task_node("stable observation")
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="idempotent recording"),
            status=RunStatus.RUNNING,
        )
        store = InMemoryContextStore()
        coordinator = ContextMemoryCoordinator(
            context_store=store,
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=DefaultTaskContextRequirementProvider(),
            memory_recall=ConditionalMemoryRecall(InMemoryMemoryStore()),
        )
        observation = Observation.ok(uuid4(), output="stable")

        first = await coordinator.record_observation(observation, state, node=node)
        replay = await coordinator.record_observation(observation, state, node=node)

        self.assertEqual(replay, first)
        self.assertEqual(store.history_for(first.context_id), (first,))

    async def test_memory_projection_preserves_recall_recency_order(self) -> None:
        memory_store = InMemoryMemoryStore()
        consolidator = EvidenceDrivenMemoryConsolidator(memory_store)
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        old = MemoryCandidate(
            memory_key="old",
            content="old claim",
            condition=MemoryCondition(),
            evidence=(MemoryEvidence(source_reference="old", note="old"),),
            confidence=0.7,
            evolution=MemoryEvolutionType.EXTEND,
            created_at=base_time,
        )
        new = MemoryCandidate(
            memory_key="new",
            content="new claim",
            condition=MemoryCondition(),
            evidence=(MemoryEvidence(source_reference="new", note="new"),),
            confidence=0.7,
            evolution=MemoryEvolutionType.EXTEND,
            created_at=base_time + timedelta(days=1),
        )
        await consolidator.consolidate(old)
        await consolidator.consolidate(new)
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="recency"),
            status=RunStatus.RUNNING,
        )
        coordinator = ContextMemoryCoordinator(
            context_store=InMemoryContextStore(),
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=DefaultTaskContextRequirementProvider(max_units=1),
            memory_recall=ConditionalMemoryRecall(memory_store),
        )

        assembly = await coordinator.prepare_next_context(
            state,
            task_node("use newest"),
            memory_query=MemoryRecallQuery(),
        )

        self.assertEqual(len(assembly.semantic_context), 1)
        content = cast(dict[str, Any], assembly.semantic_context[0].content)
        self.assertEqual(content["memory_key"], "new")


if __name__ == "__main__":
    unittest.main()
