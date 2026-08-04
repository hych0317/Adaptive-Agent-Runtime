from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from pydantic import ValidationError

from adaptive_agent_runtime.decisioning import (
    DecisionBudget,
    DecisionBudgetUsage,
    DecisionProposal,
    NormalizedDecisionEffect,
    budget_violations,
    consume_agent_call,
    consume_decision_cycle,
)
from tests.decisioning.fakes import (
    FakeAgent,
    FakeEffectPayload,
    FakeNormalizer,
    make_policy,
    make_request,
    make_sources,
)
from adaptive_agent_runtime.decisioning import PolicyAgentContextBuilder


class DecisionContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposal_is_immutable_and_rejects_authority_fields(self) -> None:
        context = PolicyAgentContextBuilder().build(
            make_request(), make_sources(), make_policy()
        )
        proposal = (await FakeAgent().propose(context)).proposal
        with self.assertRaises(ValidationError):
            DecisionProposal[proposal.payload.__class__].model_validate(
                {
                    **proposal.model_dump(mode="json"),
                    "authorization_id": "not-agent-authority",
                }
            )
        with self.assertRaises(ValidationError):
            proposal.confidence = 0.1  # type: ignore[misc]

    def test_budget_accounts_cycles_calls_time_and_cost(self) -> None:
        budget = DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=1,
            max_revision_count=0,
            max_elapsed_seconds=1.0,
            max_cost_units=1.0,
        )
        usage = consume_decision_cycle(DecisionBudgetUsage())
        usage = consume_agent_call(
            usage,
            elapsed_seconds=1.1,
            cost_units=1.2,
        )
        self.assertEqual(
            budget_violations(budget, usage),
            ("max_elapsed_seconds", "max_cost_units"),
        )

    def test_normalized_effect_fingerprint_covers_runtime_metadata(self) -> None:
        request = make_request()
        context = PolicyAgentContextBuilder().build(
            request, make_sources(), make_policy()
        )
        proposal = self._proposal(context)
        effect = FakeNormalizer().normalize(request, proposal)
        with self.assertRaises(ValidationError):
            NormalizedDecisionEffect[FakeEffectPayload](
                **{
                    **effect.model_dump(mode="python"),
                    "operation": "changed.operation",
                }
            )

    @staticmethod
    def _proposal(context: object):
        import asyncio

        return asyncio.run(FakeAgent().propose(context)).proposal  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
