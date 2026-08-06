from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from adaptive_agent_runtime.core import InMemoryTraceSink
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionGovernanceOutcome,
    InMemoryDecisionCheckpointStore,
    DecisionLifecycleCoordinator,
    DecisionResultStatus,
    DecisionTraceEvent,
    DecisionTraceKind,
    PolicyAgentContextBuilder,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from tests.decisioning.fakes import (
    FakeAgent,
    FakeApplier,
    FakeBasisProvider,
    FakeEffectPayload,
    FakeGovernance,
    FakeNormalizer,
    FakeProposalPayload,
    FakeRequestPayload,
    FakeTraceWriter,
    NOW,
    make_policy,
    make_request,
    make_sources,
)


CHECKPOINT_TYPE = DecisionCheckpoint[
    FakeRequestPayload,
    FakeProposalPayload,
    FakeEffectPayload,
]


class DecisionPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_checkpoint_survives_reopen_and_resumes_without_agent(self) -> None:
        request = make_request()
        with TemporaryDirectory() as directory:
            path = f"{directory}/decision.sqlite3"
            first = SQLitePersistence(path)
            first_store = first.create_decision_checkpoint_store(CHECKPOINT_TYPE)
            first_agent = FakeAgent()
            first_coordinator = DecisionLifecycleCoordinator[
                FakeRequestPayload,
                FakeProposalPayload,
                FakeEffectPayload,
                str,
            ](
                context_builder=PolicyAgentContextBuilder(),
                proposal_producer=first_agent,
                basis_provider=FakeBasisProvider(),
                validator=RuntimeDecisionValidator(normalizer=FakeNormalizer()),
                governance=FakeGovernance(
                    DecisionGovernanceOutcome.REVIEW_REQUIRED
                ),
                applier=FakeApplier(),
                checkpoint_store=first_store,
                checkpoint_type=CHECKPOINT_TYPE,
                trace_writer=FakeTraceWriter(),
            )
            pending = await first_coordinator.run(
                request,
                sources=make_sources(),
                policy=make_policy(),
            )
            first.close()

            reopened = SQLitePersistence(path)
            reopened_store = reopened.create_decision_checkpoint_store(
                CHECKPOINT_TYPE
            )
            resumed_agent = FakeAgent()
            resumed_applier = FakeApplier()
            resumed = DecisionLifecycleCoordinator[
                FakeRequestPayload,
                FakeProposalPayload,
                FakeEffectPayload,
                str,
            ](
                context_builder=PolicyAgentContextBuilder(),
                proposal_producer=resumed_agent,
                basis_provider=FakeBasisProvider(),
                validator=RuntimeDecisionValidator(normalizer=FakeNormalizer()),
                governance=FakeGovernance(
                    DecisionGovernanceOutcome.REVIEW_REQUIRED,
                    resume_outcome=DecisionGovernanceOutcome.ALLOW,
                ),
                applier=resumed_applier,
                checkpoint_store=reopened_store,
                checkpoint_type=CHECKPOINT_TYPE,
                trace_writer=FakeTraceWriter(),
            )
            completed = await resumed.resume_review(request.request_id)
            history = await reopened_store.transitions_for(request.request_id)
            reopened.close()

        self.assertEqual(pending.stage, DecisionCheckpointStage.REVIEW_PENDING)
        assert completed.result is not None
        self.assertEqual(completed.result.status, DecisionResultStatus.APPLIED)
        self.assertEqual(resumed_agent.calls, 0)
        self.assertEqual(resumed_applier.calls, 1)
        self.assertGreater(len(history), 1)
        self.assertIsInstance(completed.request.payload, FakeRequestPayload)
        assert completed.proposal is not None
        self.assertIsInstance(completed.proposal.payload, FakeProposalPayload)
        assert completed.validated_decision is not None
        self.assertIsInstance(
            completed.validated_decision.normalized_effect.payload,
            FakeEffectPayload,
        )


class DecisionTraceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_decision_trace_maps_to_existing_runtime_trace(self) -> None:
        sink = InMemoryTraceSink()
        writer = RuntimeDecisionTraceWriter(sink)
        event = DecisionTraceEvent(
            kind=DecisionTraceKind.PROPOSED,
            run_id=UUID(int=1),
            source="test",
            request_id=UUID(int=2),
            task_id=UUID(int=3),
            proposal_id=UUID(int=4),
            occurred_at=NOW,
            payload={"selected_action": "apply"},
        )

        await writer.record(event)

        entries = sink.entries_for(UUID(int=1))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].event.kind, "decision.proposed")
        correlation = entries[0].event.payload["correlation"]
        self.assertEqual(correlation["request_id"], str(UUID(int=2)))
        self.assertEqual(correlation["task_id"], str(UUID(int=3)))
        self.assertEqual(correlation["proposal_id"], str(UUID(int=4)))
        self.assertNotIn("context", entries[0].event.payload)

    async def test_full_lifecycle_is_ordered_in_runtime_trace_sink(self) -> None:
        sink = InMemoryTraceSink()
        request = make_request()
        coordinator = DecisionLifecycleCoordinator[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
            str,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=FakeAgent(),
            basis_provider=FakeBasisProvider(),
            validator=RuntimeDecisionValidator(normalizer=FakeNormalizer()),
            governance=FakeGovernance(),
            applier=FakeApplier(),
            checkpoint_store=InMemoryDecisionCheckpointStore(),
            checkpoint_type=CHECKPOINT_TYPE,
            trace_writer=RuntimeDecisionTraceWriter(sink),
        )

        await coordinator.run(
            request,
            sources=make_sources(),
            policy=make_policy(),
        )

        entries = sink.entries_for(request.correlation.run_id)
        self.assertEqual(
            [entry.event.kind for entry in entries],
            [
                "decision.requested",
                "decision.context_projected",
                "decision.proposed",
                "decision.validation_passed",
                "decision.governance_requested",
                "decision.authorized",
                "decision.apply_started",
                "decision.applied",
            ],
        )
        self.assertTrue(
            all(
                entry.event.payload["correlation"]["request_id"]
                == str(request.request_id)
                for entry in entries
            )
        )
        self.assertNotIn("remove-me", str(entries))


if __name__ == "__main__":
    unittest.main()
