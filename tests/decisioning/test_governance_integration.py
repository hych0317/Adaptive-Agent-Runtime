from __future__ import annotations

import unittest

from adaptive_agent_runtime.decisioning import (
    DecisionBudgetUsage,
    DecisionGovernanceOutcome,
    PolicyAgentContextBuilder,
    RuntimeDecisionValidator,
    ValidatedDecision,
)
from adaptive_agent_runtime.governance import (
    AuthorizationReplayError,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RuntimeDecisionGovernanceAdapter,
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    StrictAuthorizationVerifier,
    default_governance_policy,
    governance_fingerprint,
)
from tests.decisioning.fakes import (
    FAKE_BASIS,
    NOW,
    FakeAgent,
    FakeEffectPayload,
    FakeNormalizer,
    FakeProposalPayload,
    FakeRequestPayload,
    make_policy,
    make_request,
    make_sources,
)


async def validated_decision() -> ValidatedDecision[
    FakeRequestPayload,
    FakeProposalPayload,
    FakeEffectPayload,
]:
    request = make_request()
    context = PolicyAgentContextBuilder().build(
        request, make_sources(), make_policy()
    )
    proposal = (await FakeAgent().propose(context)).proposal
    outcome = RuntimeDecisionValidator(normalizer=FakeNormalizer()).validate(
        request,
        context,
        proposal,
        current_basis=FAKE_BASIS,
        usage=DecisionBudgetUsage(
            decision_cycles=1,
            agent_calls=1,
            elapsed_seconds=0.1,
            consumed_cost_units=0.5,
        ),
    )
    assert outcome.normalized_effect is not None
    return ValidatedDecision[
        FakeRequestPayload,
        FakeProposalPayload,
        FakeEffectPayload,
    ](
        request=request,
        proposal=proposal,
        validation=outcome.validation,
        normalized_effect=outcome.normalized_effect,
    )


class GovernanceDecisionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_governance_authorizes_exact_normalized_effect_once(self) -> None:
        decision = await validated_decision()
        reviews = InMemoryHumanReviewService(clock=lambda: NOW)
        evaluator = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=reviews,
            clock=lambda: NOW,
        )
        adapter = RuntimeDecisionGovernanceAdapter[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
        ](
            evaluator=evaluator,
            authorization_issuer=GovernanceAuthorizationIssuer(clock=lambda: NOW),
            review_service=reviews,
        )

        resolution = await adapter.evaluate(decision)

        self.assertEqual(
            resolution.receipt.outcome,
            DecisionGovernanceOutcome.ALLOW,
        )
        self.assertIsNotNone(resolution.approval)
        assert resolution.approval is not None
        self.assertEqual(
            resolution.approval.request.attributes[
                SUBJECT_FINGERPRINT_ATTRIBUTE
            ],
            governance_fingerprint(decision.normalized_effect),
        )

        applied_values: list[int] = []

        async def apply_effect(effect: FakeEffectPayload):
            applied_values.append(effect.value)
            return {"value": effect.value}

        applier = GovernedDecisionApplier[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
        ](
            executor=GovernedOperationExecutor(
                verifier=StrictAuthorizationVerifier(),
                consumption_store=InMemoryAuthorizationConsumptionStore(),
                clock=lambda: NOW,
            ),
            apply_effect=apply_effect,
            clock=lambda: NOW,
        )

        receipt = await applier.apply(decision, resolution.approval)
        self.assertEqual(receipt.effect_fingerprint, decision.normalized_effect.effect_fingerprint)
        self.assertEqual(applied_values, [2])
        with self.assertRaises(AuthorizationReplayError):
            await applier.apply(decision, resolution.approval)
        self.assertEqual(applied_values, [2])


if __name__ == "__main__":
    unittest.main()
