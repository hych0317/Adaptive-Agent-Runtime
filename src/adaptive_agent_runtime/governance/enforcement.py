"""Apply-point authorization verification and single-use enforcement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from adaptive_agent_runtime.governance.contracts import (
    AuthorizationConsumptionStore,
    GovernedOperationTarget,
)
from adaptive_agent_runtime.governance.errors import (
    AuthorizationReplayError,
    AuthorizationUseConflictError,
    AuthorizationVerificationError,
    GovernedOperationError,
)
from adaptive_agent_runtime.governance.models import (
    AuthorizationUse,
    AuthorizationUseStatus,
    DecisionOutcome,
    GovernanceAuthorization,
    GovernanceDecision,
    GovernanceRequest,
    GovernanceTarget,
    governance_fingerprint,
    utc_now,
)


SUBJECT_FINGERPRINT_ATTRIBUTE = "subject_fingerprint"
T = TypeVar("T")


class BoundGovernedOperation(Generic[T]):
    """Bind a validated subject to the only callable allowed to apply it."""

    def __init__(
        self,
        *,
        module_id: str,
        operation: str,
        target: GovernanceTarget,
        subject: object,
        apply: Callable[[], Awaitable[T]],
    ) -> None:
        self.module_id = module_id
        self._operation = operation
        self._target = target
        self._subject_fingerprint = governance_fingerprint(subject)
        self._apply = apply

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def target(self) -> GovernanceTarget:
        return self._target

    @property
    def subject_fingerprint(self) -> str:
        return self._subject_fingerprint

    async def apply(self) -> T:
        return await self._apply()


class StrictAuthorizationVerifier:
    """Fail closed unless request, decision, token, target, and subject agree."""

    module_id = "governance.authorization_verifier.strict"

    def verify(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
        authorization: GovernanceAuthorization,
        target: GovernedOperationTarget[object],
    ) -> None:
        request_fingerprint = governance_fingerprint(request)
        expected_subject = request.attributes.get(SUBJECT_FINGERPRINT_ATTRIBUTE)
        checks = (
            (decision.outcome is DecisionOutcome.ALLOW, "decision is not ALLOW"),
            (decision.request_id == request.request_id, "decision request mismatch"),
            (
                decision.request_fingerprint == request_fingerprint,
                "decision snapshot mismatch",
            ),
            (
                authorization.request_id == request.request_id,
                "authorization request mismatch",
            ),
            (
                authorization.request_fingerprint == request_fingerprint,
                "authorization snapshot mismatch",
            ),
            (
                authorization.decision_id == decision.decision_id,
                "authorization decision mismatch",
            ),
            (
                authorization.policy_id == decision.policy_id
                and authorization.policy_version == decision.policy_version,
                "authorization policy mismatch",
            ),
            (
                authorization.operation == request.operation == target.operation,
                "operation mismatch",
            ),
            (
                authorization.target == request.target == target.target,
                "target mismatch",
            ),
            (
                expected_subject == target.subject_fingerprint,
                "subject fingerprint mismatch",
            ),
            (
                authorization.issued_at >= decision.decided_at,
                "authorization predates its decision",
            ),
        )
        for valid, message in checks:
            if not valid:
                raise AuthorizationVerificationError(message)


class InMemoryAuthorizationConsumptionStore:
    module_id = "governance.authorization_use.in_memory"

    def __init__(self) -> None:
        self._current: dict[UUID, AuthorizationUse] = {}
        self._history: defaultdict[UUID, list[AuthorizationUse]] = defaultdict(list)

    async def reserve(self, use: AuthorizationUse) -> None:
        existing = self._current.get(use.authorization_id)
        if existing is not None:
            raise AuthorizationReplayError(
                "authorization was already reserved or consumed: "
                f"{existing.status.value}"
            )
        if use.status is not AuthorizationUseStatus.RESERVED or use.revision != 0:
            raise AuthorizationUseConflictError(
                "new authorization use must be reserved at revision 0"
            )
        self._current[use.authorization_id] = use
        self._history[use.authorization_id].append(use)

    async def resolve(
        self,
        use: AuthorizationUse,
        *,
        expected_revision: int,
    ) -> None:
        current = self._current.get(use.authorization_id)
        if (
            current is None
            or current.status is not AuthorizationUseStatus.RESERVED
            or current.revision != expected_revision
            or use.revision != expected_revision + 1
        ):
            raise AuthorizationUseConflictError(
                "authorization resolution is based on a stale reservation"
            )
        self._current[use.authorization_id] = use
        self._history[use.authorization_id].append(use)

    async def load(self, authorization_id: UUID) -> AuthorizationUse | None:
        return self._current.get(authorization_id)

    def history_for(self, authorization_id: UUID) -> tuple[AuthorizationUse, ...]:
        return tuple(self._history.get(authorization_id, ()))


class GovernedOperationExecutor:
    """Reserve authorization before apply and consume it exactly once."""

    module_id = "governance.operation_executor"

    def __init__(
        self,
        *,
        verifier: StrictAuthorizationVerifier,
        consumption_store: AuthorizationConsumptionStore,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._verifier = verifier
        self._consumption_store = consumption_store
        self._clock = clock

    async def execute(
        self,
        *,
        request: GovernanceRequest,
        decision: GovernanceDecision,
        authorization: GovernanceAuthorization,
        target: GovernedOperationTarget[T],
    ) -> T:
        self._verifier.verify(request, decision, authorization, target)
        now = self._clock()
        reserved = AuthorizationUse(
            authorization_id=authorization.authorization_id,
            request_id=request.request_id,
            decision_id=decision.decision_id,
            operation=target.operation,
            target=target.target,
            subject_fingerprint=target.subject_fingerprint,
            status=AuthorizationUseStatus.RESERVED,
            reserved_at=now,
            updated_at=now,
        )
        await self._consumption_store.reserve(reserved)
        try:
            result = await target.apply()
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            failed = reserved.model_copy(
                update={
                    "status": AuthorizationUseStatus.FAILED,
                    "revision": reserved.revision + 1,
                    "updated_at": self._clock(),
                    "error": f"{exc.__class__.__name__}: {detail}",
                }
            )
            await self._consumption_store.resolve(
                failed,
                expected_revision=reserved.revision,
            )
            raise GovernedOperationError(
                f"governed operation '{target.operation}' failed: {detail}"
            ) from exc
        applied = reserved.model_copy(
            update={
                "status": AuthorizationUseStatus.APPLIED,
                "revision": reserved.revision + 1,
                "updated_at": self._clock(),
                "result_fingerprint": governance_fingerprint(result),
            }
        )
        await self._consumption_store.resolve(
            applied,
            expected_revision=reserved.revision,
        )
        return result
