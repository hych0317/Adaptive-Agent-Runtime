from __future__ import annotations

import unittest

from adaptive_agent_runtime.decisioning import (
    DecisionBasis,
    DecisionApplyReceipt,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionGovernanceOutcome,
    DecisionLifecycleCoordinator,
    DecisionResultStatus,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionTraceKind,
    InMemoryDecisionCheckpointStore,
    PolicyAgentContextBuilder,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from tests.decisioning.fakes import (
    FAKE_BASIS,
    FakeAgent,
    FakeApplier,
    FakeBasisProvider,
    FakeEffectPayload,
    FakeGovernance,
    FakeNormalizer,
    FakeProposalPayload,
    FakeRequestPayload,
    FakeTraceWriter,
    make_policy,
    make_request,
    make_sources,
)


def make_coordinator(
    *,
    agent: FakeAgent | None = None,
    governance: FakeGovernance | None = None,
    applier: FakeApplier | None = None,
    basis: FakeBasisProvider | None = None,
):
    store = InMemoryDecisionCheckpointStore[
        FakeRequestPayload,
        FakeProposalPayload,
        FakeEffectPayload,
    ]()
    trace = FakeTraceWriter()
    selected_agent = agent or FakeAgent()
    selected_governance = governance or FakeGovernance()
    selected_applier = applier or FakeApplier()
    selected_basis = basis or FakeBasisProvider()
    coordinator = DecisionLifecycleCoordinator[
        FakeRequestPayload,
        FakeProposalPayload,
        FakeEffectPayload,
        str,
    ](
        context_builder=PolicyAgentContextBuilder(),
        proposal_producer=selected_agent,
        basis_provider=selected_basis,
        validator=RuntimeDecisionValidator(normalizer=FakeNormalizer()),
        governance=selected_governance,
        applier=selected_applier,
        checkpoint_store=store,
        checkpoint_type=DecisionCheckpoint[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
        ],
        trace_writer=trace,
    )
    return (
        coordinator,
        selected_agent,
        selected_governance,
        selected_applier,
        selected_basis,
        store,
        trace,
    )


class DecisionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_committed_apply_reconciles_without_agent_recall_or_reapply(
        self,
    ) -> None:
        class CrashAfterCommitApplier(FakeApplier):
            def __init__(self) -> None:
                super().__init__()
                self.committed = False
                self.reconciliations = 0

            async def apply(self, decision, approval):  # type: ignore[no-untyped-def]
                del approval
                self.calls += 1
                self.committed = True
                self.effect_fingerprint = decision.normalized_effect.effect_fingerprint
                raise KeyboardInterrupt("simulated process loss after commit")

            async def reconcile(self, decision, approval):  # type: ignore[no-untyped-def]
                del approval
                self.reconciliations += 1
                self.assert_same = decision.normalized_effect.effect_fingerprint
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.COMMITTED,
                    reason="authoritative state contains the effect",
                    apply_receipt=DecisionApplyReceipt(
                        effect_fingerprint=self.effect_fingerprint,
                        committed_state_fingerprint=decision_fingerprint(
                            {"value": decision.normalized_effect.payload.value}
                        ),
                        result={"value": decision.normalized_effect.payload.value},
                    ),
                )

        applier = CrashAfterCommitApplier()
        coordinator, agent, _, _, _, store, _ = make_coordinator(applier=applier)
        request = make_request()

        with self.assertRaises(KeyboardInterrupt):
            await coordinator.run(
                request, sources=make_sources(), policy=make_policy()
            )
        interrupted = await store.load(request.request_id)
        assert interrupted is not None
        self.assertEqual(interrupted.stage, DecisionCheckpointStage.APPLYING)

        completed = await coordinator.run(
            request, sources=make_sources(), policy=make_policy()
        )

        assert completed.result is not None
        self.assertEqual(completed.result.status, DecisionResultStatus.APPLIED)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(applier.calls, 1)
        self.assertEqual(applier.reconciliations, 1)

    async def test_fake_agent_end_to_end_lifecycle_applies_and_traces(self) -> None:
        coordinator, agent, governance, applier, _, store, trace = make_coordinator()
        request = make_request()

        checkpoint = await coordinator.run(
            request,
            sources=make_sources(),
            policy=make_policy(),
        )

        self.assertEqual(checkpoint.stage, DecisionCheckpointStage.COMPLETED)
        self.assertIsNotNone(checkpoint.result)
        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.APPLIED)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(governance.evaluations, 1)
        self.assertEqual(applier.calls, 1)
        self.assertEqual(applier.values, [2])
        self.assertEqual(
            [event.kind for event in trace.events],
            [
                DecisionTraceKind.REQUESTED,
                DecisionTraceKind.CONTEXT_PROJECTED,
                DecisionTraceKind.PROPOSED,
                DecisionTraceKind.VALIDATION_PASSED,
                DecisionTraceKind.GOVERNANCE_REQUESTED,
                DecisionTraceKind.AUTHORIZED,
                DecisionTraceKind.APPLY_STARTED,
                DecisionTraceKind.APPLIED,
            ],
        )
        transitions = store.transitions_for(request.request_id)
        self.assertEqual(len(transitions), 8)
        self.assertEqual(
            transitions[-2].to_stage,
            DecisionCheckpointStage.EFFECT_COMMITTED,
        )
        self.assertNotIn(
            "secret",
            str(checkpoint.context_manifest.model_dump(mode="json")),
        )

    async def test_illegal_agent_action_is_rejected_before_governance(self) -> None:
        agent = FakeAgent(selected_action="not-allowed")
        coordinator, _, governance, applier, _, _, trace = make_coordinator(
            agent=agent
        )

        checkpoint = await coordinator.run(
            make_request(), sources=make_sources(), policy=make_policy()
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.REJECTED)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)
        self.assertIn(DecisionTraceKind.VALIDATION_FAILED, [e.kind for e in trace.events])

    async def test_context_fingerprint_tampering_is_rejected(self) -> None:
        agent = FakeAgent(context_fingerprint="0" * 64)
        coordinator, _, governance, applier, _, _, _ = make_coordinator(agent=agent)

        checkpoint = await coordinator.run(
            make_request(), sources=make_sources(), policy=make_policy()
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.REJECTED)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)

    async def test_zero_agent_budget_stops_before_agent_call(self) -> None:
        budget = DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=0,
            max_revision_count=0,
        )
        coordinator, agent, governance, applier, _, _, _ = make_coordinator()

        checkpoint = await coordinator.run(
            make_request(budget=budget),
            sources=make_sources(),
            policy=make_policy(),
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.EXPIRED)
        self.assertEqual(agent.calls, 0)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)

    async def test_zero_cycle_budget_stops_before_context_and_agent(self) -> None:
        budget = DecisionBudget(
            max_decision_cycles=0,
            max_agent_calls=1,
            max_revision_count=0,
        )
        coordinator, agent, governance, applier, _, _, trace = make_coordinator()

        checkpoint = await coordinator.run(
            make_request(budget=budget),
            sources=make_sources(),
            policy=make_policy(),
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.EXPIRED)
        self.assertEqual(agent.calls, 0)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)
        self.assertEqual(
            [item.kind for item in trace.events],
            [DecisionTraceKind.REQUESTED, DecisionTraceKind.EXPIRED],
        )

    async def test_cost_budget_is_checked_after_agent_call(self) -> None:
        budget = DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=1,
            max_revision_count=0,
            max_cost_units=0.1,
        )
        coordinator, agent, governance, applier, _, _, _ = make_coordinator(
            agent=FakeAgent(cost_units=0.5)
        )

        checkpoint = await coordinator.run(
            make_request(budget=budget),
            sources=make_sources(),
            policy=make_policy(),
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.EXPIRED)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)

    async def test_time_budget_is_checked_after_agent_call(self) -> None:
        budget = DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=1,
            max_revision_count=0,
            max_elapsed_seconds=0.1,
        )
        coordinator, agent, governance, applier, _, _, _ = make_coordinator(
            agent=FakeAgent(elapsed_seconds=0.2)
        )

        checkpoint = await coordinator.run(
            make_request(budget=budget),
            sources=make_sources(),
            policy=make_policy(),
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.EXPIRED)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)

    async def test_revision_budget_rejects_agent_revision(self) -> None:
        coordinator, agent, governance, applier, _, _, _ = make_coordinator(
            agent=FakeAgent(revision=1)
        )

        checkpoint = await coordinator.run(
            make_request(), sources=make_sources(), policy=make_policy()
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.REJECTED)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)

    async def test_stale_runtime_basis_expires_proposal(self) -> None:
        stale = FakeBasisProvider(
            DecisionBasis(
                snapshot_fingerprint=decision_fingerprint({"state_revision": 3}),
                state_revision=3,
                graph_version=FAKE_BASIS.graph_version,
                configuration_revision=FAKE_BASIS.configuration_revision,
            )
        )
        coordinator, _, governance, applier, _, _, _ = make_coordinator(basis=stale)

        checkpoint = await coordinator.run(
            make_request(), sources=make_sources(), policy=make_policy()
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.EXPIRED)
        self.assertEqual(governance.evaluations, 0)
        self.assertEqual(applier.calls, 0)

    async def test_governance_deny_never_calls_apply(self) -> None:
        coordinator, _, governance, applier, _, _, _ = make_coordinator(
            governance=FakeGovernance(DecisionGovernanceOutcome.DENY)
        )

        checkpoint = await coordinator.run(
            make_request(), sources=make_sources(), policy=make_policy()
        )

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.REJECTED)
        self.assertEqual(governance.evaluations, 1)
        self.assertEqual(applier.calls, 0)

    async def test_review_checkpoint_resumes_without_second_agent_call(self) -> None:
        governance = FakeGovernance(
            DecisionGovernanceOutcome.REVIEW_REQUIRED,
            resume_outcome=DecisionGovernanceOutcome.ALLOW,
        )
        coordinator, agent, _, applier, _, _, _ = make_coordinator(
            governance=governance
        )
        request = make_request()

        pending = await coordinator.run(
            request, sources=make_sources(), policy=make_policy()
        )
        completed = await coordinator.resume_review(request.request_id)
        repeated = await coordinator.resume_review(request.request_id)

        self.assertEqual(pending.stage, DecisionCheckpointStage.REVIEW_PENDING)
        self.assertIsNone(pending.result)
        assert completed.result is not None
        self.assertEqual(completed.result.status, DecisionResultStatus.APPLIED)
        self.assertEqual(repeated, completed)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(governance.resumes, 1)
        self.assertEqual(applier.calls, 1)

    async def test_reviewed_proposal_expires_if_basis_changes(self) -> None:
        basis = FakeBasisProvider()
        governance = FakeGovernance(
            DecisionGovernanceOutcome.REVIEW_REQUIRED,
            resume_outcome=DecisionGovernanceOutcome.ALLOW,
        )
        coordinator, agent, _, applier, _, _, _ = make_coordinator(
            governance=governance,
            basis=basis,
        )
        request = make_request()
        await coordinator.run(request, sources=make_sources(), policy=make_policy())
        basis.basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint({"state_revision": 4}),
            state_revision=4,
            graph_version=4,
            configuration_revision=1,
        )

        checkpoint = await coordinator.resume_review(request.request_id)

        assert checkpoint.result is not None
        self.assertEqual(checkpoint.result.status, DecisionResultStatus.EXPIRED)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(governance.resumes, 0)
        self.assertEqual(applier.calls, 0)

    async def test_apply_failure_is_terminal_and_not_replayed(self) -> None:
        coordinator, _, _, applier, _, _, _ = make_coordinator(
            applier=FakeApplier(fail=True)
        )
        request = make_request()

        failed = await coordinator.run(
            request, sources=make_sources(), policy=make_policy()
        )
        repeated = await coordinator.resume_review(request.request_id)

        assert failed.result is not None
        self.assertEqual(failed.result.status, DecisionResultStatus.FAILED)
        self.assertEqual(repeated, failed)
        self.assertEqual(applier.calls, 1)


if __name__ == "__main__":
    unittest.main()
