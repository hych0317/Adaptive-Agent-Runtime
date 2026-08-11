"""In-memory Human Review lifecycle without UI or approval-system coupling."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import UUID

from adaptive_agent_runtime.governance.errors import (
    GovernanceInvariantError,
    InvalidReviewTransitionError,
    ReviewNotFoundError,
)
from adaptive_agent_runtime.governance.models import (
    GovernanceRequest,
    HumanReviewDecision,
    ReviewOutcome,
    ReviewRequest,
    ReviewStatus,
    governance_fingerprint,
    stable_governance_id,
    utc_now,
)

_MAX_REVIEW_CLOCK_SKEW = timedelta(seconds=1)


def normalize_review_decision_time(
    current: ReviewRequest,
    decision: HumanReviewDecision,
) -> HumanReviewDecision:
    """Clamp small local clock regressions while rejecting stale decisions."""

    if decision.decided_at >= current.requested_at:
        return decision
    if current.requested_at - decision.decided_at > _MAX_REVIEW_CLOCK_SKEW:
        raise GovernanceInvariantError(
            "human decision cannot precede the review request"
        )
    return decision.model_copy(update={"decided_at": current.requested_at})


class InMemoryHumanReviewService:
    module_id = "governance.human_review.in_memory"

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._clock = clock
        self._reviews: dict[UUID, ReviewRequest] = {}

    def open(
        self,
        request: GovernanceRequest,
        *,
        policy_id: str,
        policy_version: str,
        reason: str,
    ) -> ReviewRequest:
        request_fingerprint = governance_fingerprint(request)
        review_request_id = stable_governance_id(
            "human-review",
            request.request_id,
            request_fingerprint,
            policy_id,
            policy_version,
        )
        existing = self._reviews.get(review_request_id)
        if existing is not None:
            if existing.governance_request_id != request.request_id:
                raise GovernanceInvariantError(
                    "review id is already associated with another request"
                )
            if existing.request_fingerprint != request_fingerprint:
                raise GovernanceInvariantError(
                    "review request snapshot differs from the existing request"
                )
            if (
                existing.policy_id != policy_id
                or existing.policy_version != policy_version
            ):
                raise GovernanceInvariantError(
                    "review id is already associated with another policy"
                )
            return existing
        now = self._clock()
        review = ReviewRequest(
            review_request_id=review_request_id,
            governance_request_id=request.request_id,
            request_fingerprint=request_fingerprint,
            policy_id=policy_id,
            policy_version=policy_version,
            reason=reason,
            requested_at=now,
            updated_at=now,
        )
        self._reviews[review_request_id] = review
        return review

    def get(self, review_request_id: UUID) -> ReviewRequest | None:
        return self._reviews.get(review_request_id)

    def resolve(
        self,
        review_request_id: UUID,
        decision: HumanReviewDecision,
    ) -> ReviewRequest:
        current = self._reviews.get(review_request_id)
        if current is None:
            raise ReviewNotFoundError(
                f"review request '{review_request_id}' does not exist"
            )
        decision = normalize_review_decision_time(current, decision)
        if current.status is not ReviewStatus.PENDING:
            if current.decision == decision:
                return current
            raise InvalidReviewTransitionError(
                f"review request '{review_request_id}' is already resolved"
            )
        status = (
            ReviewStatus.APPROVED
            if decision.outcome is ReviewOutcome.APPROVE
            else ReviewStatus.REJECTED
        )
        resolved = current.model_copy(
            update={
                "status": status,
                "revision": current.revision + 1,
                "updated_at": decision.decided_at,
                "decision": decision,
            }
        )
        self._reviews[review_request_id] = resolved
        return resolved
