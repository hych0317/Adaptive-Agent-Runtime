"""Small recording wrappers shared by Research Decision handlers."""

from __future__ import annotations

from adaptive_agent_runtime.governance import (
    GovernanceAuthorization,
    GovernanceAuthorizationIssuer,
    GovernanceDecision,
    GovernanceEvaluator,
    GovernanceRequest,
    ReviewRequest,
)


class RecordingGovernanceEvaluator:
    module_id = "research_agent.decision.governance_recorder"

    def __init__(self, delegate: GovernanceEvaluator) -> None:
        self._delegate = delegate
        self.request: GovernanceRequest | None = None
        self.preliminary: GovernanceDecision | None = None
        self.final: GovernanceDecision | None = None
        self.review: ReviewRequest | None = None

    def evaluate(self, request: GovernanceRequest) -> GovernanceDecision:
        decision = self._delegate.evaluate(request)
        self.request = request
        self.preliminary = decision
        self.final = decision
        return decision

    def finalize_review(
        self,
        request: GovernanceRequest,
        review: ReviewRequest,
    ) -> GovernanceDecision:
        decision = self._delegate.finalize_review(request, review)
        self.review = review
        self.final = decision
        return decision


class RecordingAuthorizationIssuer:
    module_id = "research_agent.decision.authorization_recorder"

    def __init__(self, delegate: GovernanceAuthorizationIssuer) -> None:
        self._delegate = delegate
        self.authorization: GovernanceAuthorization | None = None

    def issue(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
    ) -> GovernanceAuthorization:
        authorization = self._delegate.issue(request, decision)
        self.authorization = authorization
        return authorization
