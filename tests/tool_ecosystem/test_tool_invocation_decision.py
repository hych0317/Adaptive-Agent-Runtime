from __future__ import annotations

import unittest
from uuid import uuid4

from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionBasis,
    DecisionBudget,
    DecisionBudgetUsage,
    DecisionCorrelation,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionTarget,
    DecisionValidationStatus,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.llm import (
    BoundToolInvocationProposalProducer,
    ToolInvocationEffectNormalizer,
    build_tool_invocation_agent_input,
)
from adaptive_agent_runtime.tool_ecosystem import (
    TOOL_INVOCATION_DECISION_TYPE,
    TOOL_INVOCATION_INPUT_SOURCE_TYPE,
    TOOL_INVOCATION_OPERATION,
    CapabilityRequirement,
    ProviderAvailability,
    ToolCorrelation,
    ToolInvocation,
    ToolInvocationDecisionPayload,
    ToolInvocationProposalDraft,
    ToolProviderMetadata,
    tool_invocation_candidate_set_fingerprint,
    tool_invocation_fingerprint,
)


class ToolInvocationDecisionTests(unittest.IsolatedAsyncioTestCase):
    def _fixture(self) -> tuple[
        DecisionRequest[ToolInvocationDecisionPayload],
        ProjectionSources,
        ContextProjectionPolicy,
        ToolInvocationProposalDraft,
    ]:
        run_id = uuid4()
        task_id = uuid4()
        node_id = uuid4()
        invocation_id = uuid4()
        requirement = CapabilityRequirement(
            capability_id="research.retrieve",
            required_provider_tags=("industry",),
        )
        draft = ToolInvocationProposalDraft(
            call_key="industry-once",
            capability_id=requirement.capability_id,
            arguments={
                "company": "Tesla",
                "scope": "industry",
                "query": "industry evidence",
            },
        )
        metadata = ToolProviderMetadata(
            provider_id="fixture.industry",
            name="Fixture Industry Provider",
            capability_id=requirement.capability_id,
            description="Return bounded fixture evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "scope": {"const": "industry"},
                    "query": {"type": "string"},
                },
                "required": ["company", "scope", "query"],
                "additionalProperties": False,
            },
            availability=ProviderAvailability.AVAILABLE,
            tags=("industry",),
        )
        invocation = ToolInvocation(
            invocation_id=invocation_id,
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=metadata.provider_id,
            arguments=draft.arguments,
            correlation=ToolCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=invocation_id,
            ),
        )
        payload = ToolInvocationDecisionPayload(
            task_description="Analyze Tesla",
            node_goal="Analyze industry evidence",
            invocation=invocation,
            requirement=requirement,
            provider_metadata=metadata,
            provider_metadata_fingerprint=tool_invocation_fingerprint(metadata),
            candidate_set_fingerprint=tool_invocation_candidate_set_fingerprint(
                (metadata,)
            ),
            selection_fingerprint=tool_invocation_fingerprint(
                {"provider_id": metadata.provider_id}
            ),
            proposal_fingerprint=tool_invocation_fingerprint(draft),
            exact_argument_constraints={
                "company": "Tesla",
                "scope": "industry",
            },
            state_revision=3,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(payload),
            state_revision=payload.state_revision,
        )
        request = DecisionRequest[ToolInvocationDecisionPayload](
            decision_type=TOOL_INVOCATION_DECISION_TYPE,
            target=DecisionTarget(
                target_type="tool_provider",
                target_id=metadata.provider_id,
            ),
            correlation=DecisionCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=invocation_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(TOOL_INVOCATION_OPERATION,),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
            ),
        )
        projected = build_tool_invocation_agent_input(payload)
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="tool-invocation-input",
                    source_type=TOOL_INVOCATION_INPUT_SOURCE_TYPE,
                    agent_scope="tool_invocation",
                    content=projected.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    estimated_tokens=128,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="test.tool-invocation",
            version="1",
            agent_scope="tool_invocation",
            allowed_decision_types=frozenset({TOOL_INVOCATION_DECISION_TYPE}),
            allowed_source_types=frozenset({TOOL_INVOCATION_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            max_items=1,
            max_context_tokens=256,
        )
        return request, sources, policy, draft

    async def test_valid_proposal_is_isolated_and_normalized_to_runtime_effect(
        self,
    ) -> None:
        request, sources, policy, draft = self._fixture()
        context = PolicyAgentContextBuilder().build(request, sources, policy)
        call = await BoundToolInvocationProposalProducer(
            draft=draft,
            producer_id="reasoning",
        ).propose(context)
        outcome = RuntimeDecisionValidator(
            normalizer=ToolInvocationEffectNormalizer()
        ).validate(
            request,
            context,
            call.proposal,
            current_basis=request.basis,
            usage=DecisionBudgetUsage(decision_cycles=1, agent_calls=1),
        )

        self.assertEqual(outcome.validation.status, DecisionValidationStatus.PASSED)
        self.assertIsNotNone(outcome.normalized_effect)
        assert outcome.normalized_effect is not None
        self.assertEqual(outcome.normalized_effect.operation, "tool.call")
        self.assertEqual(
            outcome.normalized_effect.payload.invocation,
            request.payload.invocation,
        )
        projected = context.blocks[0].content
        self.assertNotIn("provider_id", projected)
        self.assertNotIn("credential", projected)
        self.assertNotIn("authorization", projected)

    async def test_runtime_rejects_proposal_changed_after_capture(self) -> None:
        request, sources, policy, _ = self._fixture()
        context = PolicyAgentContextBuilder().build(request, sources, policy)
        tampered = ToolInvocationProposalDraft(
            call_key="industry-once",
            capability_id=request.payload.requirement.capability_id,
            arguments={
                "company": "Another Company",
                "scope": "industry",
                "query": "industry evidence",
            },
        )
        proposal = DecisionProposal[ToolInvocationProposalDraft](
            request_id=request.request_id,
            proposal_type=request.decision_type,
            producer=DecisionProducer(
                producer_id="reasoning",
                capability="tool_invocation_proposal",
            ),
            input_snapshot_fingerprint=request.basis.snapshot_fingerprint,
            context_fingerprint=context.context_fingerprint,
            selected_action=TOOL_INVOCATION_OPERATION,
            payload=tampered,
            rationale="Attempt to change Runtime-bound arguments.",
            confidence=1.0,
        )
        outcome = RuntimeDecisionValidator(
            normalizer=ToolInvocationEffectNormalizer()
        ).validate(
            request,
            context,
            proposal,
            current_basis=request.basis,
            usage=DecisionBudgetUsage(decision_cycles=1, agent_calls=1),
        )

        self.assertEqual(outcome.validation.status, DecisionValidationStatus.FAILED)
        self.assertIsNone(outcome.normalized_effect)
        self.assertIn(
            "effect_normalization_failed",
            {item.code for item in outcome.validation.violations},
        )


if __name__ == "__main__":
    unittest.main()
