"""Tool, Memory, and Evaluation Governance adapter tests."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
)
from adaptive_agent_runtime.evaluation import (
    EvaluationComponent,
    OptimizationProposal,
)
from adaptive_agent_runtime.governance import (
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceHistory,
    GovernancePolicy,
    HumanReviewDecision,
    InMemoryHumanReviewService,
    MemoryGovernanceAdapter,
    ReviewOutcome,
    RiskLevel,
    RuntimeGovernanceEvaluator,
    ToolGovernanceAdapter,
    default_governance_policy,
    governance_fingerprint,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
)
from adaptive_agent_runtime.legacy.optimization_governance import (
    LegacyOptimizationApplyGovernanceAdapter,
)
from adaptive_agent_runtime.tool_ecosystem import ToolCorrelation, ToolInvocation


NOW = datetime(2026, 8, 2, 5, 0, tzinfo=timezone.utc)


def governor(
    policy: GovernancePolicy,
) -> tuple[RuntimeGovernanceEvaluator, InMemoryHumanReviewService]:
    reviews = InMemoryHumanReviewService(clock=lambda: NOW)
    return (
        RuntimeGovernanceEvaluator(
            policy=policy,
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=reviews,
            clock=lambda: NOW,
        ),
        reviews,
    )


def memory_candidate(
    candidate_id: int,
    evidence_count: int,
) -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=UUID(int=candidate_id),
        memory_key="research.source_preference",
        content="prefer audited sources",
        condition=MemoryCondition(),
        evidence=tuple(
            MemoryEvidence(
                evidence_id=UUID(int=candidate_id + index + 1),
                source_context_id=UUID(int=1000 + index),
                note=f"supporting observation {index + 1}",
                weight=0.95,
                observed_at=NOW + timedelta(seconds=index),
            )
            for index in range(evidence_count)
        ),
        confidence=0.95,
        evolution=MemoryEvolutionType.EXTEND,
        created_at=NOW,
    )


class ToolGovernanceIntegrationTests(unittest.TestCase):
    def test_ordinary_tool_call_is_allowed_and_privileged_call_is_reviewed(
        self,
    ) -> None:
        invocation = ToolInvocation(
            invocation_id=UUID(int=30),
            requirement_id=UUID(int=31),
            capability_id="financial_information",
            provider_id="provider-a",
            arguments={"symbol": "EXAMPLE"},
            correlation=ToolCorrelation(
                run_id=UUID(int=32),
                task_id=UUID(int=33),
                node_id=UUID(int=34),
                action_id=UUID(int=35),
            ),
            requested_at=NOW,
        )
        adapter = ToolGovernanceAdapter()
        runtime_governor, _ = governor(default_governance_policy())

        ordinary_request = adapter.to_request(invocation)
        privileged_request = adapter.to_request(invocation, privileged=True)
        ordinary = runtime_governor.evaluate(ordinary_request)
        privileged = runtime_governor.evaluate(privileged_request)

        self.assertEqual(ordinary.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(privileged.outcome, DecisionOutcome.REVIEW_REQUIRED)
        self.assertEqual(privileged_request.risk, RiskLevel.HIGH)
        self.assertTrue(privileged_request.attributes["privileged"])
        self.assertEqual(
            ordinary_request.attributes[SUBJECT_FINGERPRINT_ATTRIBUTE],
            governance_fingerprint(invocation),
        )
        self.assertEqual(
            ordinary_request.correlation.action_id,
            invocation.correlation.action_id,
        )


class MemoryGovernanceIntegrationTests(unittest.TestCase):
    def test_single_evidence_requires_review_but_repeated_evidence_can_allow(
        self,
    ) -> None:
        runtime_governor, _ = governor(
            GovernancePolicy(policy_id="test", version="1")
        )
        adapter = MemoryGovernanceAdapter()
        one_evidence = adapter.to_request(
            memory_candidate(40, 1),
            run_id=UUID(int=41),
            history=GovernanceHistory(successful_similar=5),
        )
        repeated_evidence = adapter.to_request(
            memory_candidate(50, 2),
            run_id=UUID(int=51),
            history=GovernanceHistory(successful_similar=5),
        )

        uncertain = runtime_governor.evaluate(one_evidence)
        supported = runtime_governor.evaluate(repeated_evidence)

        self.assertEqual(uncertain.outcome, DecisionOutcome.REVIEW_REQUIRED)
        self.assertEqual(supported.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(one_evidence.operation, "memory.write")
        self.assertEqual(one_evidence.attributes["evidence_count"], 1)
        self.assertEqual(repeated_evidence.attributes["evidence_count"], 2)
        self.assertEqual(
            one_evidence.attributes[SUBJECT_FINGERPRINT_ATTRIBUTE],
            governance_fingerprint(memory_candidate(40, 1)),
        )


class OptimizationGovernanceIntegrationTests(unittest.TestCase):
    def test_optimization_proposal_requires_approval_before_authorization(
        self,
    ) -> None:
        proposal = OptimizationProposal(
            proposal_id=UUID(int=60),
            source_pattern_ids=(UUID(int=61),),
            source_candidate_ids=(UUID(int=62),),
            target_component=EvaluationComponent.TOOL,
            change_kind="tool.selection_policy.review",
            change_spec={"proposal_only": True},
            rationale="Repeated capability mismatch.",
            expected_benefit="Reduce selection failures.",
            proposal_confidence=0.95,
            validation_plan=("Replay isolated traces.",),
            rollback_plan=("Restore the prior policy.",),
            created_at=NOW,
        )
        adapter = LegacyOptimizationApplyGovernanceAdapter()
        request = adapter.to_request(proposal)
        runtime_governor, reviews = governor(
            GovernancePolicy(policy_id="test", version="1")
        )

        preliminary = runtime_governor.evaluate(request)

        self.assertEqual(request.risk, RiskLevel.HIGH)
        self.assertEqual(preliminary.outcome, DecisionOutcome.REVIEW_REQUIRED)
        assert preliminary.review_request_id is not None
        approved = reviews.resolve(
            preliminary.review_request_id,
            HumanReviewDecision(
                outcome=ReviewOutcome.APPROVE,
                reviewer_id="reviewer-3",
                rationale="Proposal is bounded and reversible.",
                decided_at=NOW + timedelta(minutes=1),
            ),
        )
        final = runtime_governor.finalize_review(request, approved)
        authorization = GovernanceAuthorizationIssuer(
            clock=lambda: NOW + timedelta(minutes=1)
        ).issue(request, final)

        self.assertEqual(final.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(authorization.operation, "optimization.apply")
        self.assertEqual(
            authorization.target.target_id,
            str(proposal.proposal_id),
        )
        self.assertEqual(proposal.change_spec["proposal_only"], True)


if __name__ == "__main__":
    unittest.main()
