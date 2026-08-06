"""Apply-point authorization verification and single-use enforcement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime
import hashlib
import hmac
import secrets
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
    RuntimeCommitPermit,
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
        apply_with_permit: Callable[[RuntimeCommitPermit], Awaitable[T]] | None = None,
    ) -> None:
        self.module_id = module_id
        self._operation = operation
        self._target = target
        self._subject_fingerprint = governance_fingerprint(subject)
        self._apply = apply
        self._apply_with_permit = apply_with_permit

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def target(self) -> GovernanceTarget:
        return self._target

    @property
    def subject_fingerprint(self) -> str:
        return self._subject_fingerprint

    async def apply(self, permit: RuntimeCommitPermit) -> T:
        if self._apply_with_permit is not None:
            return await self._apply_with_permit(permit)
        return await self._apply()


class HMACCommitPermitAuthority:
    """Authenticate Governance authorizations and commit capabilities."""

    module_id = "governance.commit_permit.hmac"

    def __init__(
        self,
        secret: bytes | None = None,
        *,
        allow_unsealed_authorizations: bool = False,
    ) -> None:
        self._secret = secret or secrets.token_bytes(32)
        self._allow_unsealed_authorizations = allow_unsealed_authorizations
        if len(self._secret) < 32:
            raise ValueError("commit Permit secret must contain at least 32 bytes")

    def seal_authorization(
        self,
        authorization: GovernanceAuthorization,
    ) -> GovernanceAuthorization:
        seal = self._seal(
            authorization.model_dump(mode="json", exclude={"integrity_seal"})
        )
        return authorization.model_copy(update={"integrity_seal": seal})

    def verify_authorization(self, authorization: GovernanceAuthorization) -> None:
        seal = authorization.integrity_seal
        if seal is None and self._allow_unsealed_authorizations:
            return
        expected = self._seal(
            authorization.model_dump(mode="json", exclude={"integrity_seal"})
        )
        if seal is None or not hmac.compare_digest(seal, expected):
            raise AuthorizationVerificationError(
                "Governance authorization integrity seal is invalid"
            )

    def issue_permit(
        self,
        authorization: GovernanceAuthorization,
        target: GovernedOperationTarget[object],
        *,
        issued_at: datetime,
    ) -> RuntimeCommitPermit:
        self.verify_authorization(authorization)
        unsigned = RuntimeCommitPermit(
            authorization_id=authorization.authorization_id,
            request_id=authorization.request_id,
            decision_id=authorization.decision_id,
            operation=target.operation,
            target=target.target,
            subject_fingerprint=target.subject_fingerprint,
            issued_at=issued_at,
            integrity_seal="0" * 64,
        )
        return unsigned.model_copy(
            update={
                "integrity_seal": self._seal(
                    unsigned.model_dump(mode="json", exclude={"integrity_seal"})
                )
            }
        )

    def verify_permit(
        self,
        permit: RuntimeCommitPermit,
        *,
        operation: str,
        target: GovernanceTarget,
        subject_fingerprint: str,
    ) -> None:
        expected = self._seal(
            permit.model_dump(mode="json", exclude={"integrity_seal"})
        )
        checks = (
            (hmac.compare_digest(permit.integrity_seal, expected), "Permit seal mismatch"),
            (permit.operation == operation, "Permit operation mismatch"),
            (permit.target == target, "Permit target mismatch"),
            (
                permit.subject_fingerprint == subject_fingerprint,
                "Permit Effect fingerprint mismatch",
            ),
        )
        for valid, message in checks:
            if not valid:
                raise AuthorizationVerificationError(message)

    def _seal(self, value: object) -> str:
        payload = governance_fingerprint(value).encode("ascii")
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()


class CommitPermitVerifier:
    """Validate cryptographic binding plus durable single-use reservation."""

    module_id = "governance.commit_permit_verifier"

    def __init__(
        self,
        *,
        authority: HMACCommitPermitAuthority,
        consumption_store: AuthorizationConsumptionStore,
    ) -> None:
        self._authority = authority
        self._consumption_store = consumption_store

    async def verify(
        self,
        permit: RuntimeCommitPermit,
        *,
        operation: str,
        target: GovernanceTarget,
        subject_fingerprint: str,
    ) -> None:
        if not isinstance(permit, RuntimeCommitPermit):
            raise AuthorizationVerificationError(
                "authoritative commit requires a Runtime-issued Permit"
            )
        self._authority.verify_permit(
            permit,
            operation=operation,
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        use = await self._consumption_store.load(permit.authorization_id)
        if (
            use is None
            or use.status is not AuthorizationUseStatus.RESERVED
            or use.request_id != permit.request_id
            or use.decision_id != permit.decision_id
            or use.operation != operation
            or use.target != target
            or use.subject_fingerprint != subject_fingerprint
        ):
            raise AuthorizationReplayError(
                "commit Permit has no matching active authorization reservation"
            )


class StrictAuthorizationVerifier:
    """Fail closed unless request, decision, token, target, and subject agree."""

    module_id = "governance.authorization_verifier.strict"

    def __init__(self, authority: HMACCommitPermitAuthority | None = None) -> None:
        self._authority = authority

    def verify(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
        authorization: GovernanceAuthorization,
        target: GovernedOperationTarget[object],
    ) -> None:
        if self._authority is not None:
            self._authority.verify_authorization(authorization)
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
        permit_authority: HMACCommitPermitAuthority | None = None,
    ) -> None:
        self._verifier = verifier
        self._consumption_store = consumption_store
        self._clock = clock
        self._permit_authority = permit_authority or HMACCommitPermitAuthority(
            allow_unsealed_authorizations=True
        )

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
        permit = self._permit_authority.issue_permit(
            authorization,
            target,
            issued_at=now,
        )
        await self._consumption_store.reserve(reserved)
        try:
            result = await target.apply(permit)
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

    async def resume_reserved(
        self,
        *,
        request: GovernanceRequest,
        decision: GovernanceDecision,
        authorization: GovernanceAuthorization,
        target: GovernedOperationTarget[T],
    ) -> T:
        """Resume a reserved Apply, or reserve it if interruption preceded reservation."""

        self._verifier.verify(request, decision, authorization, target)
        reserved = await self._consumption_store.load(authorization.authorization_id)
        if reserved is None:
            return await self.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=target,
            )
        if reserved.status is not AuthorizationUseStatus.RESERVED:
            raise AuthorizationReplayError(
                "only an existing reserved authorization can resume Apply"
            )
        permit = self._permit_authority.issue_permit(
            authorization,
            target,
            issued_at=self._clock(),
        )
        try:
            result = await target.apply(permit)
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
                failed, expected_revision=reserved.revision
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
            applied, expected_revision=reserved.revision
        )
        return result

    async def reconcile_committed(
        self,
        *,
        request: GovernanceRequest,
        decision: GovernanceDecision,
        authorization: GovernanceAuthorization,
        result: object,
    ) -> None:
        """Resolve a durable reservation only after domain read-back proved commit."""

        expected = governance_fingerprint(result)
        current = await self._consumption_store.load(authorization.authorization_id)
        if current is None:
            raise AuthorizationUseConflictError(
                "committed reconciliation has no authorization reservation"
            )
        if current.status is AuthorizationUseStatus.APPLIED:
            if current.result_fingerprint != expected:
                raise AuthorizationUseConflictError(
                    "authorization result conflicts with committed read-back"
                )
            return
        if current.status is not AuthorizationUseStatus.RESERVED:
            raise AuthorizationUseConflictError(
                "failed authorization cannot reconcile as committed"
            )
        applied = current.model_copy(
            update={
                "status": AuthorizationUseStatus.APPLIED,
                "revision": current.revision + 1,
                "updated_at": self._clock(),
                "result_fingerprint": expected,
            }
        )
        await self._consumption_store.resolve(
            applied, expected_revision=current.revision
        )
