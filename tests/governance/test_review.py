"""Human Review lifecycle and authorization tests."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from adaptive_agent_runtime.governance import (
    ConfidenceSignals,
    DecisionLevel,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceHistory,
    GovernanceInvariantError,
    GovernancePolicy,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    HumanReviewDecision,
    ImpactAssessment,
    InMemoryHumanReviewService,
    InvalidReviewTransitionError,
    ReviewOutcome,
    ReviewStatus,
    RiskLevel,
    RuntimeGovernanceEvaluator,
)


NOW = datetime(2026, 8, 2, 4, 0, tzinfo=timezone.utc)


def high_risk_request(value: int) -> GovernanceRequest:
    return GovernanceRequest(
        request_id=UUID(int=value),
        scope=GovernanceScope.EVOLUTION,
        operation="runtime.policy_update",
        target=GovernanceTarget(target_type="policy", target_id="planner"),
        risk=RiskLevel.HIGH,
        signals=ConfidenceSignals(
            stated_confidence=0.95,
            impact=ImpactAssessment(
                score=0.9,
                reversible=True,
                description="runtime behavior change",
            ),
            history=GovernanceHistory(successful_similar=2),
        ),
        requested_at=NOW,
    )


class HumanReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviews = InMemoryHumanReviewService(clock=lambda: NOW)
        self.governor = RuntimeGovernanceEvaluator(
            policy=GovernancePolicy(policy_id="test", version="1"),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self.reviews,
            clock=lambda: NOW,
        )
        self.issuer = GovernanceAuthorizationIssuer(clock=lambda: NOW)

    def test_high_risk_review_approval_and_authorization(self) -> None:
        request = high_risk_request(20)

        preliminary = self.governor.evaluate(request)

        self.assertEqual(preliminary.outcome, DecisionOutcome.REVIEW_REQUIRED)
        self.assertEqual(preliminary.level, DecisionLevel.REVIEW)
        assert preliminary.review_request_id is not None
        pending = self.reviews.get(preliminary.review_request_id)
        assert pending is not None
        self.assertEqual(pending.status, ReviewStatus.PENDING)
        with self.assertRaises(GovernanceInvariantError):
            self.governor.finalize_review(request, pending)

        human = HumanReviewDecision(
            outcome=ReviewOutcome.APPROVE,
            reviewer_id="reviewer-1",
            rationale="Evidence and rollback plan are sufficient.",
            decided_at=NOW + timedelta(minutes=1),
        )
        approved = self.reviews.resolve(pending.review_request_id, human)
        replayed = self.reviews.resolve(pending.review_request_id, human)
        final = self.governor.finalize_review(request, approved)
        authorization = self.issuer.issue(request, final)
        replayed_evaluation = self.governor.evaluate(request)

        self.assertEqual(approved, replayed)
        self.assertEqual(approved.status, ReviewStatus.APPROVED)
        self.assertEqual(approved.revision, 1)
        self.assertEqual(final.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(replayed_evaluation, final)
        self.assertEqual(final.level, DecisionLevel.REVIEW)
        self.assertEqual(authorization.request_id, request.request_id)
        self.assertEqual(authorization.decision_id, final.decision_id)

        changed = human.model_copy(
            update={
                "outcome": ReviewOutcome.REJECT,
                "rationale": "changed decision",
            }
        )
        with self.assertRaises(InvalidReviewTransitionError):
            self.reviews.resolve(pending.review_request_id, changed)

        changed_request = request.model_copy(
            update={"operation": "runtime.another_policy_update"}
        )
        with self.assertRaises(GovernanceInvariantError):
            self.issuer.issue(changed_request, final)

    def test_rejected_review_cannot_issue_authorization(self) -> None:
        request = high_risk_request(21)
        preliminary = self.governor.evaluate(request)
        assert preliminary.review_request_id is not None
        rejected = self.reviews.resolve(
            preliminary.review_request_id,
            HumanReviewDecision(
                outcome=ReviewOutcome.REJECT,
                reviewer_id="reviewer-2",
                rationale="Impact is too broad.",
                decided_at=NOW + timedelta(minutes=1),
            ),
        )

        final = self.governor.finalize_review(request, rejected)

        self.assertEqual(rejected.status, ReviewStatus.REJECTED)
        self.assertEqual(final.outcome, DecisionOutcome.DENY)
        with self.assertRaises(GovernanceInvariantError):
            self.issuer.issue(request, final)

    def test_review_is_bound_to_governance_policy_version(self) -> None:
        request = high_risk_request(22)
        version_one = self.governor.evaluate(request)
        assert version_one.review_request_id is not None
        approved_under_v1 = self.reviews.resolve(
            version_one.review_request_id,
            HumanReviewDecision(
                outcome=ReviewOutcome.APPROVE,
                reviewer_id="reviewer-4",
                rationale="Approved under policy version one.",
                decided_at=NOW + timedelta(minutes=1),
            ),
        )
        version_two_governor = RuntimeGovernanceEvaluator(
            policy=GovernancePolicy(policy_id="test", version="2"),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self.reviews,
            clock=lambda: NOW,
        )

        version_two = version_two_governor.evaluate(request)

        self.assertNotEqual(
            version_one.review_request_id,
            version_two.review_request_id,
        )
        with self.assertRaises(GovernanceInvariantError):
            version_two_governor.finalize_review(request, approved_under_v1)


if __name__ == "__main__":
    unittest.main()
