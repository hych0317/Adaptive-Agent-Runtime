"""Durable Human Review and authorization-consumption stores."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
from uuid import UUID

from adaptive_agent_runtime.governance import (
    AuthorizationReplayError,
    AuthorizationUse,
    AuthorizationUseConflictError,
    AuthorizationUseStatus,
    GovernanceInvariantError,
    GovernanceRequest,
    HumanReviewDecision,
    InvalidReviewTransitionError,
    ReviewNotFoundError,
    ReviewOutcome,
    ReviewRequest,
    ReviewStatus,
)
from adaptive_agent_runtime.governance.models import (
    governance_fingerprint,
    stable_governance_id,
    utc_now,
)
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


def _model_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class SQLiteHumanReviewService:
    """Persistent Review lifecycle implementing the existing synchronous port."""

    module_id = "governance.human_review.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._database = database
        self._clock = clock

    def open(
        self,
        request: GovernanceRequest,
        *,
        policy_id: str,
        policy_version: str,
        reason: str,
    ) -> ReviewRequest:
        request_fingerprint = governance_fingerprint(request)
        review_id = stable_governance_id(
            "human-review",
            request.request_id,
            request_fingerprint,
            policy_id,
            policy_version,
        )
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT snapshot_json FROM governance_reviews "
                "WHERE review_request_id = ?",
                (str(review_id),),
            ).fetchone()
            if row is not None:
                existing = ReviewRequest.model_validate_json(row["snapshot_json"])
                self._validate_existing(
                    existing,
                    request=request,
                    request_fingerprint=request_fingerprint,
                    policy_id=policy_id,
                    policy_version=policy_version,
                )
                return existing
            now = self._clock()
            review = ReviewRequest(
                review_request_id=review_id,
                governance_request_id=request.request_id,
                request_fingerprint=request_fingerprint,
                policy_id=policy_id,
                policy_version=policy_version,
                reason=reason,
                requested_at=now,
                updated_at=now,
            )
            payload = _model_json(review)
            cursor.execute(
                "INSERT INTO governance_reviews "
                "(review_request_id, revision, snapshot_json) VALUES (?, ?, ?)",
                (str(review_id), review.revision, payload),
            )
            cursor.execute(
                "INSERT INTO governance_review_history "
                "(review_request_id, revision, snapshot_json) VALUES (?, ?, ?)",
                (str(review_id), review.revision, payload),
            )
        return review

    def get(self, review_request_id: UUID) -> ReviewRequest | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshot_json FROM governance_reviews "
                "WHERE review_request_id = ?",
                (str(review_request_id),),
            ).fetchone()
        if row is None:
            return None
        return ReviewRequest.model_validate_json(row["snapshot_json"])

    def resolve(
        self,
        review_request_id: UUID,
        decision: HumanReviewDecision,
    ) -> ReviewRequest:
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT snapshot_json FROM governance_reviews "
                "WHERE review_request_id = ?",
                (str(review_request_id),),
            ).fetchone()
            if row is None:
                raise ReviewNotFoundError(
                    f"review request '{review_request_id}' does not exist"
                )
            current = ReviewRequest.model_validate_json(row["snapshot_json"])
            if current.status is not ReviewStatus.PENDING:
                if current.decision == decision:
                    return current
                raise InvalidReviewTransitionError(
                    f"review request '{review_request_id}' is already resolved"
                )
            if decision.decided_at < current.requested_at:
                raise GovernanceInvariantError(
                    "human decision cannot precede the review request"
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
            payload = _model_json(resolved)
            updated = cursor.execute(
                "UPDATE governance_reviews SET revision = ?, snapshot_json = ? "
                "WHERE review_request_id = ? AND revision = ?",
                (
                    resolved.revision,
                    payload,
                    str(review_request_id),
                    current.revision,
                ),
            )
            if updated.rowcount != 1:
                raise InvalidReviewTransitionError(
                    "review changed while the decision was being recorded"
                )
            cursor.execute(
                "INSERT INTO governance_review_history "
                "(review_request_id, revision, snapshot_json) VALUES (?, ?, ?)",
                (str(review_request_id), resolved.revision, payload),
            )
        return resolved

    @staticmethod
    def _validate_existing(
        existing: ReviewRequest,
        *,
        request: GovernanceRequest,
        request_fingerprint: str,
        policy_id: str,
        policy_version: str,
    ) -> None:
        if existing.governance_request_id != request.request_id:
            raise GovernanceInvariantError(
                "review id is associated with another Governance request"
            )
        if existing.request_fingerprint != request_fingerprint:
            raise GovernanceInvariantError(
                "review id is associated with another request snapshot"
            )
        if (
            existing.policy_id != policy_id
            or existing.policy_version != policy_version
        ):
            raise GovernanceInvariantError(
                "review id is associated with another Governance policy"
            )


class SQLiteAuthorizationConsumptionStore:
    """Durable anti-replay store; a reserved token is never silently retried."""

    module_id = "governance.authorization_use.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def reserve(self, use: AuthorizationUse) -> None:
        if use.status is not AuthorizationUseStatus.RESERVED or use.revision != 0:
            raise AuthorizationUseConflictError(
                "new authorization use must be reserved at revision 0"
            )
        payload = _model_json(use)
        with self._database.transaction() as cursor:
            existing = cursor.execute(
                "SELECT status FROM governance_authorization_uses "
                "WHERE authorization_id = ?",
                (str(use.authorization_id),),
            ).fetchone()
            if existing is not None:
                raise AuthorizationReplayError(
                    "authorization was already reserved or consumed: "
                    f"{existing['status']}"
                )
            cursor.execute(
                "INSERT INTO governance_authorization_uses "
                "(authorization_id, revision, status, use_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(use.authorization_id),
                    use.revision,
                    use.status.value,
                    payload,
                ),
            )
            cursor.execute(
                "INSERT INTO governance_authorization_use_history "
                "(authorization_id, revision, use_json) VALUES (?, ?, ?)",
                (str(use.authorization_id), use.revision, payload),
            )

    async def resolve(
        self,
        use: AuthorizationUse,
        *,
        expected_revision: int,
    ) -> None:
        payload = _model_json(use)
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT revision, status FROM governance_authorization_uses "
                "WHERE authorization_id = ?",
                (str(use.authorization_id),),
            ).fetchone()
            if (
                row is None
                or row["status"] != AuthorizationUseStatus.RESERVED.value
                or int(row["revision"]) != expected_revision
                or use.revision != expected_revision + 1
            ):
                raise AuthorizationUseConflictError(
                    "authorization resolution is based on a stale reservation"
                )
            cursor.execute(
                "UPDATE governance_authorization_uses "
                "SET revision = ?, status = ?, use_json = ? "
                "WHERE authorization_id = ?",
                (
                    use.revision,
                    use.status.value,
                    payload,
                    str(use.authorization_id),
                ),
            )
            cursor.execute(
                "INSERT INTO governance_authorization_use_history "
                "(authorization_id, revision, use_json) VALUES (?, ?, ?)",
                (str(use.authorization_id), use.revision, payload),
            )

    async def load(self, authorization_id: UUID) -> AuthorizationUse | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT use_json FROM governance_authorization_uses "
                "WHERE authorization_id = ?",
                (str(authorization_id),),
            ).fetchone()
        if row is None:
            return None
        return AuthorizationUse.model_validate_json(row["use_json"])

    async def history_for(
        self,
        authorization_id: UUID,
    ) -> tuple[AuthorizationUse, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT use_json FROM governance_authorization_use_history "
                "WHERE authorization_id = ? ORDER BY revision",
                (str(authorization_id),),
            ).fetchall()
        return tuple(
            AuthorizationUse.model_validate_json(row["use_json"])
            for row in rows
        )
