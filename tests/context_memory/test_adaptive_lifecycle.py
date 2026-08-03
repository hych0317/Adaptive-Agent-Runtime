from __future__ import annotations

import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime import AgentState, AgentTask, RunStatus
from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    ContextAssembler,
    ContextCompressionResult,
    ContextLayer,
    ContextLifecycleAction,
    ContextLifecycleManager,
    ContextLifecycleRuntime,
    ContextMemoryCoordinator,
    ContextMetadata,
    ContextRequirement,
    ContextScheduler,
    ContextSource,
    ContextUnit,
    DeterministicContextLifecyclePolicy,
    DeterministicContextPressureMonitor,
    DirectContextLifecycleExecutor,
    InMemoryContextArchive,
    InMemoryContextStore,
    InMemoryMemoryStore,
    MemoryRecallQuery,
    ResidencyPolicy,
)
from adaptive_agent_runtime.orchestration import TaskNode


class FixedCompressor:
    module_id = "test.context_compressor.fixed"

    async def compress(self, unit: ContextUnit) -> ContextCompressionResult:
        return ContextCompressionResult(
            content={"summary_of": str(unit.context_id)},
            core_conclusions=("retained conclusion",),
            estimated_tokens=min(10, unit.metadata.estimated_tokens),
        )


class FixedRequirementProvider:
    module_id = "test.context_requirement.fixed"

    def __init__(self, requirement: ContextRequirement) -> None:
        self._requirement = requirement

    def requirement_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> ContextRequirement:
        return self._requirement.model_copy(
            update={
                "run_id": state.run_id,
                "task_id": state.task.task_id,
                "node_id": node.node_id,
            }
        )


def unit(
    run_id: UUID,
    *,
    tokens: int,
    importance: float,
    residency: ResidencyPolicy = ResidencyPolicy.SESSION,
) -> ContextUnit:
    return ContextUnit(
        content={"tokens": tokens},
        metadata=ContextMetadata(
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.TASK,
            run_id=run_id,
            importance=importance,
            estimated_tokens=tokens,
        ),
        residency_policy=residency,
    )


class AdaptiveContextLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryContextStore()
        self.archive = InMemoryContextArchive()
        self.manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=FixedCompressor(),
        )

    def runtime(
        self,
        *,
        max_tokens: int,
        max_compressed_units: int = 1,
    ) -> ContextLifecycleRuntime:
        return ContextLifecycleRuntime(
            store=self.store,
            pressure_monitor=DeterministicContextPressureMonitor(
                max_resident_tokens=max_tokens
            ),
            policy=DeterministicContextLifecyclePolicy(
                max_compressed_units=max_compressed_units
            ),
            executor=DirectContextLifecycleExecutor(self.manager),
        )

    async def test_pressure_automatically_compresses_then_archives(self) -> None:
        run_id = uuid4()
        pinned = unit(
            run_id,
            tokens=20,
            importance=1.0,
            residency=ResidencyPolicy.PINNED,
        )
        low_value = unit(
            run_id,
            tokens=60,
            importance=0.2,
            residency=ResidencyPolicy.TRANSIENT,
        )
        medium_value = unit(run_id, tokens=40, importance=0.6)
        for item in (pinned, low_value, medium_value):
            await self.manager.add(item)
        requirement = ContextRequirement(
            run_id=run_id,
            goal="assemble under pressure",
            max_tokens=100,
        )

        before, after, results = await self.runtime(
            max_tokens=100,
            max_compressed_units=0,
        ).reconcile(requirement)

        self.assertEqual(before.resident_tokens, 120)
        self.assertLess(after.resident_tokens, before.resident_tokens)
        self.assertEqual(
            tuple(item.decision.action for item in results),
            (
                ContextLifecycleAction.COMPRESS,
                ContextLifecycleAction.ARCHIVE,
            ),
        )
        self.assertEqual(await self.store.load(pinned.context_id), pinned)
        self.assertIsNone(await self.store.load(low_value.context_id))
        self.assertIsNotNone(await self.archive.find_latest(low_value.context_id))

    async def test_required_archived_context_restores_before_assembly(self) -> None:
        run_id = uuid4()
        original = unit(run_id, tokens=40, importance=0.7)
        await self.manager.add(original)
        await self.manager.compress(original.context_id)
        await self.manager.archive(original.context_id)
        requirement = ContextRequirement(
            run_id=run_id,
            goal="reuse archived evidence",
            required_context_ids=(original.context_id,),
            max_tokens=100,
        )
        state = AgentState(
            run_id=run_id,
            task=AgentTask(description="restore"),
            status=RunStatus.RUNNING,
        )
        node = TaskNode(
            goal="reuse evidence",
            expected_output="result",
            strategy_id="mock",
        )
        coordinator = ContextMemoryCoordinator(
            context_store=self.store,
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=FixedRequirementProvider(requirement),
            memory_recall=ConditionalMemoryRecall(InMemoryMemoryStore()),
            lifecycle=self.runtime(max_tokens=100),
        )

        preparation = await coordinator.prepare(
            state,
            node,
            memory_query=MemoryRecallQuery(),
        )

        self.assertEqual(
            tuple(
                item.decision.action for item in preparation.lifecycle_results
            ),
            (ContextLifecycleAction.RESTORE,),
        )
        self.assertEqual(
            tuple(item.context_id for item in preparation.assembly.units),
            (original.context_id,),
        )
        resident = await self.store.load(original.context_id)
        assert resident is not None
        self.assertEqual(resident.metadata.access_count, 1)
        self.assertIsNotNone(resident.metadata.last_accessed_at)
        self.assertEqual(
            preparation.assembly.units[0].revision,
            resident.revision,
        )


if __name__ == "__main__":
    unittest.main()
