from __future__ import annotations

import unittest
from collections.abc import Callable
from uuid import UUID, uuid4

from adaptive_agent_runtime import (
    AgentState,
    AgentTask,
    InMemoryTraceSink,
    RunStatus,
)
from adaptive_agent_runtime.context_memory import (
    ContextCompressionExecutionPolicy,
    ContextLayer,
    ContextLifecycleManager,
    ContextLifecycleRuntime,
    ContextLifecycleState,
    ContextMetadata,
    ContextRequirement,
    ContextSource,
    ContextStore,
    ContextUnit,
    DeterministicContextLifecyclePolicy,
    DeterministicContextPressureMonitor,
    DirectContextLifecycleExecutor,
    InMemoryContextArchive,
    InMemoryContextStore,
    InMemoryMemoryStore,
    ResidencyPolicy,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    CompressedContextDraft,
    CompressionRequest,
    ToolIntentDraft,
)
from applications.research_agent.context_compression import (
    ResearchContextCompressionDecisionHandler,
)
from applications.research_agent.strategies import ResearchWorkspace
from applications.research_agent.tasks import build_research_task


class FakeCompressionAgent:
    module_id = "test.context_compression.agent"
    capability_id = "semantic_compression"

    def __init__(
        self,
        draft: Callable[[CompressionRequest], CompressedContextDraft] | None = None,
    ) -> None:
        self._draft = draft or self._valid_draft
        self.requests: list[CompressionRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def compress(
        self,
        request: CompressionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[CompressedContextDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        return CapabilityTurnResult[CompressedContextDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=self._draft(request),
        )

    @staticmethod
    def _valid_draft(request: CompressionRequest) -> CompressedContextDraft:
        return CompressedContextDraft(
            content={"summary": "source-bound semantic summary"},
            core_conclusions=("retained conclusion",),
            source_reference_ids=(request.source_reference_id,),
            estimated_tokens=request.target_max_tokens,
        )


class ToolIntentCompressionAgent:
    module_id = "test.context_compression.tool_intent_agent"
    capability_id = "semantic_compression"

    async def compress(
        self,
        request: CompressionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[CompressedContextDraft]:
        del request, invocation
        return CapabilityTurnResult[CompressedContextDraft](
            kind=CapabilityTurnKind.TOOL_INTENT,
            tool_intents=(
                ToolIntentDraft(
                    call_key="compression-escape",
                    capability_id="runtime.state_mutation",
                    arguments={"operation": "delete_context"},
                ),
            ),
        )


class ApplyLoadIsStaleStore(ContextStore):
    module_id = "test.context_store.apply_load_stale"

    def __init__(self, delegate: InMemoryContextStore) -> None:
        self._delegate = delegate
        self._loads: dict[UUID, int] = {}

    async def save(
        self,
        unit: ContextUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        await self._delegate.save(unit, expected_revision=expected_revision)

    async def load(self, context_id: UUID) -> ContextUnit | None:
        unit = await self._delegate.load(context_id)
        count = self._loads.get(context_id, 0) + 1
        self._loads[context_id] = count
        if unit is not None and count >= 3:
            return unit.model_copy(update={"revision": unit.revision + 1})
        return unit

    async def delete(
        self,
        context_id: UUID,
        *,
        expected_revision: int,
    ) -> None:
        await self._delegate.delete(
            context_id,
            expected_revision=expected_revision,
        )

    async def list_for_run(self, run_id: UUID) -> tuple[ContextUnit, ...]:
        return await self._delegate.list_for_run(run_id)


def context_unit(
    run_id: UUID,
    *,
    tokens: int = 80,
    residency: ResidencyPolicy = ResidencyPolicy.SESSION,
) -> ContextUnit:
    return ContextUnit(
        content={"evidence": "bounded source content"},
        metadata=ContextMetadata(
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.TASK,
            run_id=run_id,
            source_reference=f"document:{run_id}",
            tags=("research",),
            estimated_tokens=tokens,
        ),
        residency_policy=residency,
    )


class ContextCompressionDecisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = InMemoryContextStore()
        self.archive = InMemoryContextArchive()
        self.trace = InMemoryTraceSink()
        self.workspace = ResearchWorkspace(build_research_task("Acme"))
        self.reviews = InMemoryHumanReviewService()
        self.governance = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self.reviews,
        )
        self.operation_executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=InMemoryAuthorizationConsumptionStore(),
        )

    def handler(
        self,
        agent: FakeCompressionAgent,
        *,
        store: ContextStore | None = None,
    ) -> ResearchContextCompressionDecisionHandler:
        return ResearchContextCompressionDecisionHandler(
            capability=agent,
            store=store or self.store,
            execution_policy=ContextCompressionExecutionPolicy(
                timeout_seconds=2.0,
                target_token_ratio=0.5,
            ),
            governance=self.governance,
            reviews=self.reviews,
            issuer=GovernanceAuthorizationIssuer(),
            operation_executor=self.operation_executor,
            trace_sink=self.trace,
            workspace=self.workspace,
        )

    async def test_pressure_runs_full_decision_lifecycle_then_context_commit(
        self,
    ) -> None:
        run_id = uuid4()
        original = context_unit(run_id)
        agent = FakeCompressionAgent()
        manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=self.handler(agent),
        )
        await manager.add(original)
        runtime = ContextLifecycleRuntime(
            store=self.store,
            pressure_monitor=DeterministicContextPressureMonitor(
                max_resident_tokens=50
            ),
            policy=DeterministicContextLifecyclePolicy(
                compression_trigger_ratio=0.8,
                target_ratio=0.6,
                max_compressed_units=1,
            ),
            executor=DirectContextLifecycleExecutor(manager),
            max_passes=1,
        )

        before, after, results = await runtime.reconcile(
            ContextRequirement(
                run_id=run_id,
                goal="fit Context into the Runtime token budget",
                max_tokens=100,
            )
        )

        self.assertEqual(len(agent.requests), 1)
        request = agent.requests[0]
        self.assertEqual(request.original_estimated_tokens, 80)
        self.assertEqual(request.target_max_tokens, 40)
        projected = request.model_dump(mode="json")["content"]
        self.assertEqual(
            set(projected),
            {"context_id", "content", "source", "layer", "source_reference", "tags"},
        )
        self.assertNotIn("revision", projected)
        self.assertNotIn("governance", projected)
        self.assertNotIn("memory", projected)
        self.assertEqual(len(results), 1)
        compressed = results[0].unit
        assert compressed is not None
        self.assertEqual(compressed.lifecycle_state, ContextLifecycleState.COMPRESSED)
        self.assertEqual(compressed.revision, original.revision + 1)
        self.assertEqual(compressed.metadata.estimated_tokens, 40)
        self.assertIsNotNone(compressed.recovery_reference)
        self.assertLess(after.resident_tokens, before.resident_tokens)
        self.assertEqual(
            tuple(entry.event.kind for entry in self.trace.entries_for(run_id)),
            (
                "decision.requested",
                "decision.context_projected",
                "decision.proposed",
                "decision.validation_passed",
                "decision.governance_requested",
                "decision.authorized",
                "decision.apply_started",
                "decision.applied",
            ),
        )
        record = self.workspace.governance_records[0]
        self.assertEqual(record.scenario, "context_compression_agent")
        self.assertEqual(record.request.operation, "context.compress")

    async def test_required_and_pinned_context_never_reach_agent(self) -> None:
        run_id = uuid4()
        required = context_unit(run_id, tokens=60)
        pinned = context_unit(
            run_id,
            tokens=60,
            residency=ResidencyPolicy.PINNED,
        )
        agent = FakeCompressionAgent()
        manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=self.handler(agent),
        )
        await manager.add(required)
        await manager.add(pinned)
        runtime = ContextLifecycleRuntime(
            store=self.store,
            pressure_monitor=DeterministicContextPressureMonitor(
                max_resident_tokens=100
            ),
            policy=DeterministicContextLifecyclePolicy(),
            executor=DirectContextLifecycleExecutor(manager),
            max_passes=1,
        )

        _, _, results = await runtime.reconcile(
            ContextRequirement(
                run_id=run_id,
                goal="preserve mandatory Context",
                required_context_ids=(required.context_id,),
                max_tokens=120,
            )
        )

        self.assertEqual(results, ())
        self.assertEqual(agent.requests, [])
        self.assertEqual(await self.store.load(required.context_id), required)
        self.assertEqual(await self.store.load(pinned.context_id), pinned)

    async def test_wrong_source_reference_is_rejected_without_commit(self) -> None:
        run_id = uuid4()
        original = context_unit(run_id)
        agent = FakeCompressionAgent(
            lambda request: CompressedContextDraft(
                content={"summary": "unbound"},
                core_conclusions=("unbound",),
                source_reference_ids=("another-context",),
                estimated_tokens=request.target_max_tokens,
            )
        )
        manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=self.handler(agent),
        )
        await manager.add(original)

        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            await manager.compress(original.context_id)

        self.assertEqual(await self.store.load(original.context_id), original)
        self.assertIsNone(await self.archive.find_latest(original.context_id))
        kinds = tuple(entry.event.kind for entry in self.trace.entries_for(run_id))
        self.assertIn("decision.validation_failed", kinds)
        self.assertNotIn("decision.apply_started", kinds)

    async def test_non_reducing_draft_is_rejected(self) -> None:
        run_id = uuid4()
        original = context_unit(run_id)
        agent = FakeCompressionAgent(
            lambda request: CompressedContextDraft(
                content={"summary": "too large"},
                core_conclusions=("too large",),
                source_reference_ids=(request.source_reference_id,),
                estimated_tokens=request.original_estimated_tokens,
            )
        )
        manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=self.handler(agent),
        )
        await manager.add(original)

        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            await manager.compress(original.context_id)

        self.assertEqual(await self.store.load(original.context_id), original)
        self.assertIsNone(await self.archive.find_latest(original.context_id))

    async def test_tool_intent_is_rejected_without_context_commit(self) -> None:
        run_id = uuid4()
        original = context_unit(run_id)
        manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=ResearchContextCompressionDecisionHandler(
                capability=ToolIntentCompressionAgent(),
                store=self.store,
                execution_policy=ContextCompressionExecutionPolicy(
                    timeout_seconds=2.0
                ),
                governance=self.governance,
                reviews=self.reviews,
                issuer=GovernanceAuthorizationIssuer(),
                operation_executor=self.operation_executor,
                trace_sink=self.trace,
                workspace=self.workspace,
            ),
        )
        await manager.add(original)

        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            await manager.compress(original.context_id)

        self.assertEqual(await self.store.load(original.context_id), original)
        self.assertIsNone(await self.archive.find_latest(original.context_id))

    async def test_apply_revalidates_revision_after_governance(self) -> None:
        run_id = uuid4()
        original = context_unit(run_id)
        await self.store.save(original, expected_revision=None)
        sequenced = ApplyLoadIsStaleStore(self.store)
        agent = FakeCompressionAgent()

        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            await self.handler(agent, store=sequenced).compress(original)

        self.assertEqual(await self.store.load(original.context_id), original)
        kinds = tuple(entry.event.kind for entry in self.trace.entries_for(run_id))
        self.assertIn("decision.validation_passed", kinds)
        self.assertIn("decision.apply_started", kinds)
        self.assertEqual(kinds[-1], "decision.failed")

    async def test_compression_does_not_change_task_state_or_memory(self) -> None:
        run_id = uuid4()
        state = AgentState(
            run_id=run_id,
            task=AgentTask(description="unchanged task"),
            status=RunStatus.RUNNING,
        )
        memory = InMemoryMemoryStore()
        original = context_unit(run_id)
        manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=self.handler(FakeCompressionAgent()),
        )
        await manager.add(original)

        await manager.compress(original.context_id)

        self.assertEqual(state.revision, 0)
        self.assertEqual(state.status, RunStatus.RUNNING)
        self.assertEqual(await memory.list_all(), ())


if __name__ == "__main__":
    unittest.main()
