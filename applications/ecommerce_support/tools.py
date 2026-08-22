"""Domain tools that bind trusted Principal to authoritative operations."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from applications.ecommerce_support.contracts import FaultHook
from applications.ecommerce_support.models import (
    AuthenticatedPrincipal,
    CouponGrantRecord,
    MemoryWriteRequest,
    OperationReceipt,
    OrderRecord,
    RefundOperationRecord,
    RefundOperationStatus,
)
from applications.ecommerce_support.payment_gateway import FakePaymentGateway
from applications.ecommerce_support.persistence import EcommerceSQLiteStore
from applications.ecommerce_support.policies import (
    DomainPolicyError,
    EcommercePolicy,
    MemoryCandidateWritePolicy,
)
from applications.governance_scenario_suite.contracts import (
    AuditEventType,
    EffectSpec,
    OperationType,
    ReasonCode,
    ReconciliationStatus,
)


class EcommerceTools:
    """Trusted application facade; Principal is never taken from Effect arguments."""

    def __init__(
        self,
        *,
        store: EcommerceSQLiteStore,
        gateway: FakePaymentGateway,
        policy: EcommercePolicy,
        clock: Callable[[], datetime],
        memory_write_policy: MemoryCandidateWritePolicy | None = None,
    ) -> None:
        self._store = store
        self._gateway = gateway
        self._policy = policy
        self._clock = clock
        self._memory_write_policy = memory_write_policy or MemoryCandidateWritePolicy()

    def get_order(
        self,
        principal: AuthenticatedPrincipal,
        order_id: str,
    ) -> OrderRecord:
        return self._store.get_order(principal, order_id, now=self._clock())

    def change_shipping_address(
        self,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
    ) -> OperationReceipt:
        return self._store.change_address(principal, effect, now=self._clock())

    def request_refund(
        self,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
        *,
        approval_id: str | None = None,
        fault_hook: FaultHook | None = None,
    ) -> RefundOperationRecord:
        if effect.operation is not OperationType.REFUND:
            raise ValueError("request_refund requires a REFUND effect")
        assert effect.amount_cents is not None
        assert effect.idempotency_key is not None
        prepared = self._store.prepare_refund(
            principal,
            effect,
            now=self._clock(),
            require_approval=self._policy.refund_requires_approval(
                effect.amount_cents
            ),
            approval_id=approval_id,
            policy_version=self._policy.policy_version,
        )
        if prepared.status is RefundOperationStatus.COMMITTED:
            return prepared
        self._store.append_audit(
            occurred_at=self._clock(),
            event_type=AuditEventType.EXTERNAL_SEND_STARTED,
            effect_fingerprint=prepared.effect_fingerprint,
            idempotency_key=prepared.idempotency_key,
            details={"operation": OperationType.REFUND.value},
        )
        result = self._gateway.refund(
            order_id=prepared.order_id,
            amount_cents=prepared.amount_cents,
            idempotency_key=prepared.idempotency_key,
            now=self._clock(),
            fault_hook=fault_hook,
        )
        self._store.append_audit(
            occurred_at=self._clock(),
            event_type=AuditEventType.EXTERNAL_EFFECT_COMMITTED,
            effect_fingerprint=prepared.effect_fingerprint,
            idempotency_key=prepared.idempotency_key,
            details={"external_ref": result.external_ref},
        )
        return self._store.complete_refund(
            prepared.idempotency_key,
            result,
            now=self._clock(),
        )

    def reconcile_refund(
        self,
        idempotency_key: str,
        *,
        retry_if_not_committed: bool,
        fault_hook: FaultHook | None = None,
    ) -> RefundOperationRecord:
        operation = self._store.load_refund_by_idempotency_key(idempotency_key)
        if operation is None:
            raise ValueError("refund operation does not exist")
        self._store.append_audit(
            occurred_at=self._clock(),
            event_type=AuditEventType.RECONCILIATION_STARTED,
            effect_fingerprint=operation.effect_fingerprint,
            idempotency_key=operation.idempotency_key,
        )
        reconciliation = self._gateway.reconcile(
            idempotency_key,
            fault_hook=fault_hook,
        )
        if reconciliation.status is ReconciliationStatus.COMMITTED:
            assert reconciliation.result is not None
            completed = self._store.complete_refund(
                idempotency_key,
                reconciliation.result,
                now=self._clock(),
            )
            self._record_reconciliation(
                operation,
                ReasonCode.RECONCILIATION_COMMITTED,
            )
            return completed
        if reconciliation.status is ReconciliationStatus.UNKNOWN:
            self._record_reconciliation(
                operation,
                ReasonCode.RECONCILIATION_UNKNOWN,
            )
            raise DomainPolicyError(
                ReasonCode.RECONCILIATION_UNKNOWN,
                "Refund outcome is unknown; automatic retry is blocked.",
            )
        self._record_reconciliation(
            operation,
            ReasonCode.RECONCILIATION_NOT_COMMITTED,
        )
        if not retry_if_not_committed:
            raise DomainPolicyError(
                ReasonCode.RECONCILIATION_NOT_COMMITTED,
                "Refund was not committed and retry is disabled.",
            )
        result = self._gateway.refund(
            order_id=operation.order_id,
            amount_cents=operation.amount_cents,
            idempotency_key=operation.idempotency_key,
            now=self._clock(),
            fault_hook=fault_hook,
        )
        return self._store.complete_refund(
            operation.idempotency_key,
            result,
            now=self._clock(),
        )

    def grant_coupon(
        self,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
    ) -> CouponGrantRecord:
        assert effect.amount_cents is not None
        return self._store.grant_coupon(
            principal,
            effect,
            now=self._clock(),
            allow_automatic=self._policy.coupon_is_automatic(effect.amount_cents),
        )

    def validate_memory_write(
        self,
        principal: AuthenticatedPrincipal,
        candidate: MemoryWriteRequest,
    ) -> None:
        try:
            self._memory_write_policy.validate(principal, candidate)
        except DomainPolicyError as exc:
            self._store.append_audit(
                occurred_at=self._clock(),
                event_type=AuditEventType.MEMORY_WRITE_DENIED,
                reason_code=exc.reason_code,
                details={"category": candidate.category},
            )
            raise

    def _record_reconciliation(
        self,
        operation: RefundOperationRecord,
        reason_code: ReasonCode,
    ) -> None:
        self._store.append_audit(
            occurred_at=self._clock(),
            event_type=AuditEventType.RECONCILIATION_RESOLVED,
            reason_code=reason_code,
            effect_fingerprint=operation.effect_fingerprint,
            idempotency_key=operation.idempotency_key,
        )
