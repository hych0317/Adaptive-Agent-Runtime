"""Rule and Confidence Governance decision tests."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID

from adaptive_agent_runtime.governance import (
    ConfidencePolicy,
    ConfidenceSignals,
    DecisionLevel,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceEvidence,
    GovernanceHistory,
    GovernancePolicy,
    GovernanceRequest,
    GovernanceRule,
    GovernanceScope,
    GovernanceTarget,
    ImpactAssessment,
    InMemoryHumanReviewService,
    RiskLevel,
    RuleEffect,
    RuntimeGovernanceEvaluator,
)


NOW = datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc)


def request(
    request_id: int,
    *,
    risk: RiskLevel,
    operation: str = "tool.call",
    confidence: float = 0.9,
    reliabilities: tuple[float, ...] = (0.9, 0.9),
    impact: float = 0.2,
    reversible: bool = True,
    history: GovernanceHistory | None = None,
) -> GovernanceRequest:
    return GovernanceRequest(
        request_id=UUID(int=request_id),
        scope=GovernanceScope.ACTION,
        operation=operation,
        target=GovernanceTarget(target_type="test", target_id="target"),
        risk=risk,
        signals=ConfidenceSignals(
            stated_confidence=confidence,
            evidence=tuple(
                GovernanceEvidence(
                    evidence_id=f"evidence-{index}",
                    kind="test.evidence",
                    source="test",
                    reliability=reliability,
                    summary="deterministic evidence",
                )
                for index, reliability in enumerate(reliabilities, start=1)
            ),
            impact=ImpactAssessment(
                score=impact,
                reversible=reversible,
                description="test impact",
            ),
            history=history or GovernanceHistory(),
        ),
        requested_at=NOW,
    )


def evaluator(
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


class RuleGovernanceTests(unittest.TestCase):
    def test_low_risk_allow_rule_is_automatic(self) -> None:
        policy = GovernancePolicy(
            policy_id="test",
            version="1",
            rules=(
                GovernanceRule(
                    rule_id="allow.tool",
                    description="allow ordinary Tool calls",
                    effect=RuleEffect.ALLOW,
                    scopes=(GovernanceScope.ACTION,),
                    operations=("tool.call",),
                    risk_levels=(RiskLevel.LOW,),
                ),
            ),
        )
        governor, _ = evaluator(policy)

        decision = governor.evaluate(request(1, risk=RiskLevel.LOW))

        self.assertEqual(decision.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(decision.level, DecisionLevel.RULE)
        self.assertEqual(decision.matched_rule_ids, ("allow.tool",))
        self.assertIsNone(decision.review_request_id)

    def test_deny_rule_overrides_risk_level(self) -> None:
        policy = GovernancePolicy(
            policy_id="test",
            version="1",
            rules=(
                GovernanceRule(
                    rule_id="deny.blocked",
                    description="blocked operation",
                    effect=RuleEffect.DENY,
                    operations=("tool.blocked",),
                    priority=100,
                ),
            ),
        )
        governor, reviews = evaluator(policy)

        decision = governor.evaluate(
            request(
                2,
                risk=RiskLevel.HIGH,
                operation="tool.blocked",
            )
        )

        self.assertEqual(decision.outcome, DecisionOutcome.DENY)
        self.assertEqual(decision.level, DecisionLevel.RULE)
        self.assertIsNone(decision.review_request_id)
        self.assertIsNone(reviews.get(UUID(int=999)))

    def test_low_risk_without_allow_rule_is_denied(self) -> None:
        governor, _ = evaluator(
            GovernancePolicy(policy_id="test", version="1")
        )

        decision = governor.evaluate(request(3, risk=RiskLevel.LOW))

        self.assertEqual(decision.outcome, DecisionOutcome.DENY)
        self.assertEqual(decision.level, DecisionLevel.RULE)


class ConfidenceGovernanceTests(unittest.TestCase):
    def test_confidence_thresholds_allow_deny_and_require_review(self) -> None:
        policy = GovernancePolicy(
            policy_id="test",
            version="1",
            confidence=ConfidencePolicy(
                allow_threshold=0.75,
                deny_threshold=0.35,
                min_evidence_for_allow=2,
                max_impact_for_allow=0.6,
            ),
        )
        governor, reviews = evaluator(policy)
        high = request(
            10,
            risk=RiskLevel.MEDIUM,
            confidence=0.95,
            reliabilities=(0.95, 0.95),
            impact=0.1,
            history=GovernanceHistory(successful_similar=5),
        )
        low = request(
            11,
            risk=RiskLevel.MEDIUM,
            confidence=0.05,
            reliabilities=(),
            impact=0.9,
            reversible=False,
            history=GovernanceHistory(failed_similar=4, prior_denials=2),
        )
        uncertain = request(
            12,
            risk=RiskLevel.MEDIUM,
            confidence=0.6,
            reliabilities=(0.6,),
            impact=0.5,
        )

        allowed = governor.evaluate(high)
        denied = governor.evaluate(low)
        pending = governor.evaluate(uncertain)

        self.assertEqual(allowed.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(denied.outcome, DecisionOutcome.DENY)
        self.assertEqual(pending.outcome, DecisionOutcome.REVIEW_REQUIRED)
        self.assertEqual(allowed.level, DecisionLevel.CONFIDENCE)
        self.assertEqual(denied.level, DecisionLevel.CONFIDENCE)
        self.assertEqual(pending.level, DecisionLevel.CONFIDENCE)
        self.assertGreater(allowed.confidence_score or 0.0, 0.75)
        self.assertLess(denied.confidence_score or 1.0, 0.35)
        self.assertIsNotNone(pending.review_request_id)
        assert pending.review_request_id is not None
        self.assertIsNotNone(reviews.get(pending.review_request_id))


if __name__ == "__main__":
    unittest.main()
