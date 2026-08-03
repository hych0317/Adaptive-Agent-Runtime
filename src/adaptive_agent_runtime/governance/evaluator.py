"""Rule + Confidence + Review decision pipeline."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from adaptive_agent_runtime.governance.contracts import (
    ConfidenceEvaluator,
    HumanReviewService,
    RuleEvaluator,
)
from adaptive_agent_runtime.governance.errors import GovernanceInvariantError
from adaptive_agent_runtime.governance.models import (
    ConfidenceAssessment,
    ConfidencePolicy,
    DecisionLevel,
    DecisionOutcome,
    GovernanceAuthorization,
    GovernanceDecision,
    GovernancePolicy,
    GovernanceRequest,
    ReviewOutcome,
    ReviewRequest,
    ReviewStatus,
    RiskLevel,
    RuleEffect,
    RuleEvaluation,
    governance_fingerprint,
    stable_governance_id,
    utc_now,
)


class DeterministicConfidenceEvaluator:
    module_id = "governance.confidence_evaluator.deterministic"

    def evaluate(
        self,
        request: GovernanceRequest,
        policy: ConfidencePolicy,
    ) -> ConfidenceAssessment:
        signals = request.signals
        evidence_score = (
            sum(item.reliability for item in signals.evidence)
            / len(signals.evidence)
            if signals.evidence
            else 0.0
        )
        history_total = (
            signals.history.successful_similar
            + signals.history.failed_similar
            + signals.history.prior_denials
        )
        history_score = (
            signals.history.successful_similar / history_total
            if history_total
            else 0.5
        )
        safety_score = 1.0 - signals.impact.score
        if not signals.impact.reversible:
            safety_score *= 0.5
        score = round(
            (signals.stated_confidence * 0.45)
            + (evidence_score * 0.25)
            + (history_score * 0.15)
            + (safety_score * 0.15),
            6,
        )
        allow_gates = (
            len(signals.evidence) >= policy.min_evidence_for_allow
            and signals.impact.score <= policy.max_impact_for_allow
        )
        if score < policy.deny_threshold:
            outcome = DecisionOutcome.DENY
            reason = "Confidence score is below the deny threshold."
        elif score >= policy.allow_threshold and allow_gates:
            outcome = DecisionOutcome.ALLOW
            reason = "Confidence and evidence satisfy automatic allow gates."
        else:
            outcome = DecisionOutcome.REVIEW_REQUIRED
            reason = (
                "Confidence, evidence, impact, or history requires Human Review."
            )
        return ConfidenceAssessment(
            score=score,
            outcome=outcome,
            evidence_score=evidence_score,
            history_score=history_score,
            safety_score=safety_score,
            reason=reason,
        )


class RuntimeGovernanceEvaluator:
    module_id = "governance.evaluator.runtime"

    def __init__(
        self,
        *,
        policy: GovernancePolicy,
        rule_evaluator: RuleEvaluator,
        confidence_evaluator: ConfidenceEvaluator,
        review_service: HumanReviewService,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._policy = policy
        self._rule_evaluator = rule_evaluator
        self._confidence_evaluator = confidence_evaluator
        self._review_service = review_service
        self._clock = clock

    def evaluate(self, request: GovernanceRequest) -> GovernanceDecision:
        rule = self._rule_evaluator.evaluate(request, self._policy)
        if rule.effect is RuleEffect.DENY:
            return self._decision(
                request,
                outcome=DecisionOutcome.DENY,
                level=DecisionLevel.RULE,
                reason=rule.reason,
                rule=rule,
            )
        if request.risk is RiskLevel.LOW:
            if rule.effect is RuleEffect.ALLOW:
                return self._decision(
                    request,
                    outcome=DecisionOutcome.ALLOW,
                    level=DecisionLevel.RULE,
                    reason=rule.reason,
                    rule=rule,
                )
            return self._decision(
                request,
                outcome=DecisionOutcome.DENY,
                level=DecisionLevel.RULE,
                reason="Low-risk request has no explicit allow rule.",
                rule=rule,
            )
        if request.risk is RiskLevel.HIGH:
            return self._require_review(
                request,
                level=DecisionLevel.REVIEW,
                reason="High-risk operation requires Human Review.",
                rule=rule,
            )

        assessment = self._confidence_evaluator.evaluate(
            request,
            self._policy.confidence,
        )
        if assessment.outcome is DecisionOutcome.REVIEW_REQUIRED:
            return self._require_review(
                request,
                level=DecisionLevel.CONFIDENCE,
                reason=assessment.reason,
                rule=rule,
                confidence_score=assessment.score,
            )
        return self._decision(
            request,
            outcome=assessment.outcome,
            level=DecisionLevel.CONFIDENCE,
            reason=assessment.reason,
            rule=rule,
            confidence_score=assessment.score,
        )

    def finalize_review(
        self,
        request: GovernanceRequest,
        review: ReviewRequest,
    ) -> GovernanceDecision:
        if review.governance_request_id != request.request_id:
            raise GovernanceInvariantError(
                "review and Governance request identities differ"
            )
        if review.request_fingerprint != governance_fingerprint(request):
            raise GovernanceInvariantError(
                "Human Review belongs to another request snapshot"
            )
        if (
            review.policy_id != self._policy.policy_id
            or review.policy_version != self._policy.version
        ):
            raise GovernanceInvariantError(
                "Human Review belongs to another Governance policy"
            )
        if review.status is ReviewStatus.PENDING or review.decision is None:
            raise GovernanceInvariantError("Human Review is not resolved")
        outcome = (
            DecisionOutcome.ALLOW
            if review.decision.outcome is ReviewOutcome.APPROVE
            else DecisionOutcome.DENY
        )
        return self._decision(
            request,
            outcome=outcome,
            level=DecisionLevel.REVIEW,
            reason=f"Human reviewer: {review.decision.rationale}",
            review_request_id=review.review_request_id,
            decided_at=review.decision.decided_at,
        )

    def _require_review(
        self,
        request: GovernanceRequest,
        *,
        level: DecisionLevel,
        reason: str,
        rule: RuleEvaluation,
        confidence_score: float | None = None,
    ) -> GovernanceDecision:
        review = self._review_service.open(
            request,
            policy_id=self._policy.policy_id,
            policy_version=self._policy.version,
            reason=reason,
        )
        if review.status is not ReviewStatus.PENDING:
            return self.finalize_review(request, review)
        return self._decision(
            request,
            outcome=DecisionOutcome.REVIEW_REQUIRED,
            level=level,
            reason=reason,
            rule=rule,
            confidence_score=confidence_score,
            review_request_id=review.review_request_id,
        )

    def _decision(
        self,
        request: GovernanceRequest,
        *,
        outcome: DecisionOutcome,
        level: DecisionLevel,
        reason: str,
        rule: RuleEvaluation | None = None,
        confidence_score: float | None = None,
        review_request_id: UUID | None = None,
        decided_at: datetime | None = None,
    ) -> GovernanceDecision:
        matched_rule_ids = rule.matched_rule_ids if rule is not None else ()
        decision_time = decided_at or self._clock()
        request_fingerprint = governance_fingerprint(request)
        decision_id = stable_governance_id(
            "governance-decision",
            request.request_id,
            request_fingerprint,
            self._policy.policy_id,
            self._policy.version,
            outcome.value,
            level.value,
            *matched_rule_ids,
            confidence_score,
            review_request_id,
            reason,
            decision_time.isoformat(),
        )
        return GovernanceDecision(
            decision_id=decision_id,
            request_id=request.request_id,
            request_fingerprint=request_fingerprint,
            policy_id=self._policy.policy_id,
            policy_version=self._policy.version,
            outcome=outcome,
            level=level,
            reason=reason,
            matched_rule_ids=matched_rule_ids,
            confidence_score=confidence_score,
            review_request_id=review_request_id,
            decided_at=decision_time,
        )


class GovernanceAuthorizationIssuer:
    module_id = "governance.authorization_issuer"

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._clock = clock

    def issue(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
    ) -> GovernanceAuthorization:
        if decision.request_id != request.request_id:
            raise GovernanceInvariantError(
                "Governance decision belongs to another request"
            )
        request_fingerprint = governance_fingerprint(request)
        if decision.request_fingerprint != request_fingerprint:
            raise GovernanceInvariantError(
                "Governance decision belongs to another request snapshot"
            )
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise GovernanceInvariantError(
                "only an allowed decision can issue authorization"
            )
        authorization_id = stable_governance_id(
            "governance-authorization",
            request.request_id,
            decision.decision_id,
        )
        return GovernanceAuthorization(
            authorization_id=authorization_id,
            request_id=request.request_id,
            request_fingerprint=request_fingerprint,
            decision_id=decision.decision_id,
            policy_id=decision.policy_id,
            policy_version=decision.policy_version,
            operation=request.operation,
            target=request.target,
            issued_at=self._clock(),
        )
