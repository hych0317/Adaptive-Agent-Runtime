"""SQLite authoritative store for the governance evaluation domain."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any
from uuid import UUID, uuid4, uuid5

from pydantic import JsonValue

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.ecommerce_support.models import (
    ApprovalRecord,
    AuditEventRecord,
    AuthenticatedPrincipal,
    CouponGrantRecord,
    CouponGrantStatus,
    DomainSnapshot,
    GatewayRefundResult,
    OperationReceipt,
    OrderRecord,
    PaymentRecord,
    RefundOperationRecord,
    RefundOperationStatus,
    UserRecord,
)
from applications.ecommerce_support.policies import DomainPolicyError
from applications.governance_scenario_suite.contracts import (
    AuditEventType,
    AuthoritativeStateSpec,
    EffectSpec,
    OperationType,
    OrderStatus,
    ReasonCode,
)


_OPERATION_NAMESPACE = UUID("a5743a19-b223-4dd9-948b-89d15e55e120")


class EcommerceSQLiteStore:
    """Own authoritative ecommerce state with transactional CAS and receipts."""

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved
        self._connection = sqlite3.connect(
            str(resolved),
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> EcommerceSQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def seed(self, state: AuthoritativeStateSpec) -> None:
        """Seed an empty scenario database atomically."""

        with self._transaction() as connection:
            for user in state.users:
                connection.execute(
                    "INSERT INTO users(tenant_id, user_id) VALUES (?, ?)",
                    (user.tenant_id, user.user_id),
                )
            for order in state.orders:
                connection.execute(
                    """
                    INSERT INTO orders(
                        order_id, tenant_id, owner_user_id, paid_amount_cents,
                        refunded_amount_cents, status, address_ref, description,
                        version
                    ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        order.order_id,
                        order.tenant_id,
                        order.owner_user_id,
                        order.paid_amount_cents,
                        order.status.value,
                        order.address_ref,
                        order.description,
                        order.version,
                    ),
                )
            for payment in state.payments:
                connection.execute(
                    """
                    INSERT INTO payments(
                        payment_id, order_id, display_reference, sensitive_canary
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        payment.payment_id,
                        payment.order_id,
                        payment.display_reference,
                        payment.sensitive_canary,
                    ),
                )

    def create_approval(
        self,
        *,
        approval_id: str,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
        policy_version: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> ApprovalRecord:
        effect_fingerprint = decision_fingerprint(effect)
        record = ApprovalRecord(
            approval_id=approval_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            effect=effect,
            effect_fingerprint=effect_fingerprint,
            policy_version=policy_version,
            created_at=created_at,
            expires_at=expires_at,
        )
        with self._transaction() as connection:
            self._assert_effect_target_in_scope(connection, principal, effect)
            connection.execute(
                """
                INSERT INTO approvals(
                    approval_id, tenant_id, user_id, effect_json,
                    effect_fingerprint, policy_version, created_at, expires_at,
                    consumed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record.approval_id,
                    record.tenant_id,
                    record.user_id,
                    _json_dump(record.effect.model_dump(mode="json")),
                    record.effect_fingerprint,
                    record.policy_version,
                    record.created_at.isoformat(),
                    record.expires_at.isoformat(),
                ),
            )
        return record

    def get_order(
        self,
        principal: AuthenticatedPrincipal,
        order_id: str,
        *,
        now: datetime,
    ) -> OrderRecord:
        """Return a scoped order without revealing whether a denied order exists."""

        order: OrderRecord | None = None
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is None:
                self._append_audit(
                    connection,
                    occurred_at=now,
                    event_type=AuditEventType.RESOURCE_SCOPE_DENIED,
                    reason_code=ReasonCode.RESOURCE_NOT_AVAILABLE,
                    details={"resource_type": "order"},
                )
            elif (
                row["tenant_id"] != principal.tenant_id
                or row["owner_user_id"] != principal.user_id
            ):
                self._append_audit(
                    connection,
                    occurred_at=now,
                    event_type=AuditEventType.RESOURCE_SCOPE_DENIED,
                    reason_code=ReasonCode.RESOURCE_SCOPE_DENIED,
                    details={"resource_type": "order"},
                )
            else:
                order = _order_from_row(row)
        if order is None:
            raise _resource_not_available()
        return order

    def load_order_authoritative(self, order_id: str) -> OrderRecord | None:
        row = self._connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return _order_from_row(row) if row is not None else None

    def change_address(
        self,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
        *,
        now: datetime,
    ) -> OperationReceipt:
        if effect.operation is not OperationType.CHANGE_ADDRESS:
            raise ValueError("change_address requires a CHANGE_ADDRESS effect")
        assert effect.idempotency_key is not None
        request_fingerprint = decision_fingerprint(effect)
        with self._transaction() as connection:
            existing = self._load_receipt(connection, effect.idempotency_key)
            if existing is not None:
                if existing.request_fingerprint != request_fingerprint:
                    raise _idempotency_conflict()
                return existing
            order = self._scoped_order(connection, principal, effect.order_id)
            assert effect.state_version is not None
            if order.version != effect.state_version:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed; a new decision is required.",
                )
            if order.status is not OrderStatus.PAID:
                raise DomainPolicyError(
                    ReasonCode.OPERATION_NOT_ALLOWED_IN_STATE,
                    "The requested operation is not allowed in the current state.",
                )
            assert effect.address_ref is not None
            updated = connection.execute(
                """
                UPDATE orders
                SET address_ref = ?, version = version + 1
                WHERE order_id = ? AND version = ? AND status = ?
                """,
                (
                    effect.address_ref,
                    order.order_id,
                    order.version,
                    OrderStatus.PAID.value,
                ),
            )
            if updated.rowcount != 1:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed; a new decision is required.",
                )
            result: dict[str, JsonValue] = {
                "order_id": order.order_id,
                "address_ref": effect.address_ref,
                "version": order.version + 1,
            }
            return self._insert_receipt(
                connection,
                effect=effect,
                request_fingerprint=request_fingerprint,
                result=result,
                committed_at=now,
            )

    def fulfill_order(
        self,
        effect: EffectSpec,
        *,
        now: datetime,
    ) -> OperationReceipt:
        """Apply an idempotent external fulfillment event for TOCTOU fixtures."""

        if effect.operation is not OperationType.FULFILL_ORDER:
            raise ValueError("fulfill_order requires a FULFILL_ORDER effect")
        assert effect.order_id is not None
        assert effect.state_version is not None
        assert effect.idempotency_key is not None
        request_fingerprint = decision_fingerprint(effect)
        with self._transaction() as connection:
            existing = self._load_receipt(connection, effect.idempotency_key)
            if existing is not None:
                if existing.request_fingerprint != request_fingerprint:
                    raise _idempotency_conflict()
                return existing
            row = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?", (effect.order_id,)
            ).fetchone()
            if row is None:
                raise _resource_not_available()
            order = _order_from_row(row)
            if order.version != effect.state_version:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed; a new decision is required.",
                )
            if order.status is not OrderStatus.PAID:
                raise DomainPolicyError(
                    ReasonCode.OPERATION_NOT_ALLOWED_IN_STATE,
                    "The requested operation is not allowed in the current state.",
                )
            updated = connection.execute(
                """
                UPDATE orders
                SET status = ?, version = version + 1
                WHERE order_id = ? AND version = ? AND status = ?
                """,
                (
                    OrderStatus.SHIPPED.value,
                    order.order_id,
                    order.version,
                    OrderStatus.PAID.value,
                ),
            )
            if updated.rowcount != 1:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed; a new decision is required.",
                )
            return self._insert_receipt(
                connection,
                effect=effect,
                request_fingerprint=request_fingerprint,
                result={
                    "order_id": order.order_id,
                    "status": OrderStatus.SHIPPED.value,
                    "version": order.version + 1,
                },
                committed_at=now,
            )

    def prepare_refund(
        self,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
        *,
        now: datetime,
        require_approval: bool,
        approval_id: str | None,
        policy_version: str,
    ) -> RefundOperationRecord:
        if effect.operation is not OperationType.REFUND:
            raise ValueError("prepare_refund requires a REFUND effect")
        assert effect.idempotency_key is not None
        request_fingerprint = decision_fingerprint(effect)
        effect_fingerprint = request_fingerprint
        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM refund_operations WHERE idempotency_key = ?",
                (effect.idempotency_key,),
            ).fetchone()
            if existing_row is not None:
                existing = _refund_from_row(existing_row)
                if existing.request_fingerprint != request_fingerprint:
                    raise _idempotency_conflict()
                return existing

            order = self._scoped_order(connection, principal, effect.order_id)
            assert effect.state_version is not None
            assert effect.amount_cents is not None
            if effect.amount_cents > (
                order.paid_amount_cents - order.refunded_amount_cents
            ):
                raise DomainPolicyError(
                    ReasonCode.REFUND_EXCEEDS_PAID_AMOUNT,
                    "Refund amount exceeds the refundable paid amount.",
                )
            if require_approval:
                self._consume_approval(
                    connection,
                    principal=principal,
                    approval_id=approval_id,
                    effect=effect,
                    effect_fingerprint=effect_fingerprint,
                    policy_version=policy_version,
                    now=now,
                )
            if order.version != effect.state_version:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed; a new decision is required.",
                )
            if order.status is not OrderStatus.PAID:
                raise DomainPolicyError(
                    ReasonCode.OPERATION_NOT_ALLOWED_IN_STATE,
                    "The requested operation is not allowed in the current state.",
                )

            operation_id = str(uuid5(_OPERATION_NAMESPACE, effect.idempotency_key))
            connection.execute(
                """
                INSERT INTO refund_operations(
                    operation_id, tenant_id, user_id, order_id, amount_cents,
                    original_order_version, idempotency_key, request_fingerprint,
                    effect_fingerprint, status, external_ref, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    operation_id,
                    principal.tenant_id,
                    principal.user_id,
                    order.order_id,
                    effect.amount_cents,
                    order.version,
                    effect.idempotency_key,
                    request_fingerprint,
                    effect_fingerprint,
                    RefundOperationStatus.PREPARED.value,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            updated = connection.execute(
                """
                UPDATE orders
                SET status = ?, version = version + 1
                WHERE order_id = ? AND version = ? AND status = ?
                """,
                (
                    OrderStatus.REFUND_PENDING.value,
                    order.order_id,
                    order.version,
                    OrderStatus.PAID.value,
                ),
            )
            if updated.rowcount != 1:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed; a new decision is required.",
                )
            row = connection.execute(
                "SELECT * FROM refund_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            assert row is not None
            return _refund_from_row(row)

    def complete_refund(
        self,
        idempotency_key: str,
        result: GatewayRefundResult,
        *,
        now: datetime,
    ) -> RefundOperationRecord:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM refund_operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise ValueError("refund operation was not prepared")
            operation = _refund_from_row(row)
            if operation.status is RefundOperationStatus.COMMITTED:
                if operation.external_ref != result.external_ref:
                    raise _idempotency_conflict()
                return operation
            if operation.status is not RefundOperationStatus.PREPARED:
                raise ValueError("only a prepared refund can be completed")
            if (
                operation.order_id != result.order_id
                or operation.amount_cents != result.amount_cents
                or operation.idempotency_key != result.idempotency_key
            ):
                raise _idempotency_conflict()

            current_version = operation.original_order_version + 1
            updated = connection.execute(
                """
                UPDATE orders
                SET status = ?,
                    refunded_amount_cents = refunded_amount_cents + ?,
                    version = version + 1
                WHERE order_id = ? AND version = ? AND status = ?
                """,
                (
                    OrderStatus.REFUNDED.value,
                    operation.amount_cents,
                    operation.order_id,
                    current_version,
                    OrderStatus.REFUND_PENDING.value,
                ),
            )
            if updated.rowcount != 1:
                raise DomainPolicyError(
                    ReasonCode.STATE_VERSION_STALE,
                    "Order state changed while committing the refund.",
                )
            result_payload = result.model_dump(mode="json")
            connection.execute(
                """
                UPDATE refund_operations
                SET status = ?, external_ref = ?, result_json = ?, updated_at = ?
                WHERE operation_id = ? AND status = ?
                """,
                (
                    RefundOperationStatus.COMMITTED.value,
                    result.external_ref,
                    _json_dump(result_payload),
                    now.isoformat(),
                    operation.operation_id,
                    RefundOperationStatus.PREPARED.value,
                ),
            )
            effect = EffectSpec(
                operation=OperationType.REFUND,
                order_id=operation.order_id,
                amount_cents=operation.amount_cents,
                state_version=operation.original_order_version,
                idempotency_key=operation.idempotency_key,
            )
            self._insert_receipt(
                connection,
                effect=effect,
                request_fingerprint=operation.request_fingerprint,
                result=result_payload,
                committed_at=now,
            )
            self._append_audit(
                connection,
                occurred_at=now,
                event_type=AuditEventType.APPLY_RECEIPT_COMMITTED,
                effect_fingerprint=operation.effect_fingerprint,
                idempotency_key=operation.idempotency_key,
                details={"operation": OperationType.REFUND.value},
            )
            committed = connection.execute(
                "SELECT * FROM refund_operations WHERE operation_id = ?",
                (operation.operation_id,),
            ).fetchone()
            assert committed is not None
            return _refund_from_row(committed)

    def load_refund_by_idempotency_key(
        self, idempotency_key: str
    ) -> RefundOperationRecord | None:
        row = self._connection.execute(
            "SELECT * FROM refund_operations WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return _refund_from_row(row) if row is not None else None

    def grant_coupon(
        self,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
        *,
        now: datetime,
        allow_automatic: bool,
    ) -> CouponGrantRecord:
        if effect.operation is not OperationType.GRANT_COUPON:
            raise ValueError("grant_coupon requires a GRANT_COUPON effect")
        assert effect.idempotency_key is not None
        assert effect.user_id is not None
        assert effect.amount_cents is not None
        if effect.user_id != principal.user_id:
            raise DomainPolicyError(
                ReasonCode.RESOURCE_SCOPE_DENIED,
                "The requested resource is not available.",
            )
        if not allow_automatic:
            raise DomainPolicyError(
                ReasonCode.APPROVAL_REQUIRED,
                "This compensation requires approval.",
            )
        request_fingerprint = decision_fingerprint(effect)
        with self._transaction() as connection:
            existing_row = connection.execute(
                "SELECT * FROM coupon_grants WHERE idempotency_key = ?",
                (effect.idempotency_key,),
            ).fetchone()
            if existing_row is not None:
                existing = _coupon_from_row(existing_row)
                if existing.request_fingerprint != request_fingerprint:
                    raise _idempotency_conflict()
                return existing
            user = connection.execute(
                "SELECT tenant_id FROM users WHERE tenant_id = ? AND user_id = ?",
                (principal.tenant_id, principal.user_id),
            ).fetchone()
            if user is None:
                raise _resource_not_available()
            grant_id = str(uuid5(_OPERATION_NAMESPACE, effect.idempotency_key))
            connection.execute(
                """
                INSERT INTO coupon_grants(
                    grant_id, tenant_id, user_id, amount_cents,
                    idempotency_key, request_fingerprint, effect_fingerprint,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    grant_id,
                    principal.tenant_id,
                    principal.user_id,
                    effect.amount_cents,
                    effect.idempotency_key,
                    request_fingerprint,
                    request_fingerprint,
                    CouponGrantStatus.COMMITTED.value,
                    now.isoformat(),
                ),
            )
            self._insert_receipt(
                connection,
                effect=effect,
                request_fingerprint=request_fingerprint,
                result={
                    "grant_id": grant_id,
                    "user_id": principal.user_id,
                    "amount_cents": effect.amount_cents,
                },
                committed_at=now,
            )
            row = connection.execute(
                "SELECT * FROM coupon_grants WHERE grant_id = ?", (grant_id,)
            ).fetchone()
            assert row is not None
            return _coupon_from_row(row)

    def append_audit(
        self,
        *,
        occurred_at: datetime,
        event_type: AuditEventType,
        reason_code: ReasonCode | None = None,
        effect_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> AuditEventRecord:
        with self._transaction() as connection:
            return self._append_audit(
                connection,
                occurred_at=occurred_at,
                event_type=event_type,
                reason_code=reason_code,
                effect_fingerprint=effect_fingerprint,
                idempotency_key=idempotency_key,
                details=details,
            )

    def snapshot(self) -> DomainSnapshot:
        return DomainSnapshot(
            users=tuple(
                UserRecord(tenant_id=row["tenant_id"], user_id=row["user_id"])
                for row in self._connection.execute(
                    "SELECT * FROM users ORDER BY tenant_id, user_id"
                )
            ),
            orders=tuple(
                _order_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM orders ORDER BY order_id"
                )
            ),
            payments=tuple(
                _payment_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM payments ORDER BY payment_id"
                )
            ),
            approvals=tuple(
                _approval_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM approvals ORDER BY approval_id"
                )
            ),
            refund_operations=tuple(
                _refund_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM refund_operations ORDER BY operation_id"
                )
            ),
            coupon_grants=tuple(
                _coupon_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM coupon_grants ORDER BY grant_id"
                )
            ),
            operation_receipts=tuple(
                _receipt_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM operation_receipts ORDER BY idempotency_key"
                )
            ),
            audit_events=tuple(
                _audit_from_row(row)
                for row in self._connection.execute(
                    "SELECT * FROM audit_events ORDER BY sequence"
                )
            ),
        )

    def _consume_approval(
        self,
        connection: sqlite3.Connection,
        *,
        principal: AuthenticatedPrincipal,
        approval_id: str | None,
        effect: EffectSpec,
        effect_fingerprint: str,
        policy_version: str,
        now: datetime,
    ) -> None:
        if approval_id is None:
            raise DomainPolicyError(
                ReasonCode.APPROVAL_REQUIRED,
                "This refund requires approval.",
            )
        row = connection.execute(
            "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        if row is None:
            raise DomainPolicyError(
                ReasonCode.APPROVAL_REQUIRED,
                "This refund requires approval.",
            )
        approval = _approval_from_row(row)
        if (
            approval.tenant_id != principal.tenant_id
            or approval.user_id != principal.user_id
        ):
            raise DomainPolicyError(
                ReasonCode.RESOURCE_SCOPE_DENIED,
                "The requested resource is not available.",
            )
        if approval.consumed_at is not None:
            raise DomainPolicyError(
                ReasonCode.AUTHORIZATION_ALREADY_CONSUMED,
                "Approval has already been consumed.",
            )
        if approval.expires_at <= now:
            raise DomainPolicyError(
                ReasonCode.APPROVAL_EXPIRED,
                "Approval has expired.",
            )
        if approval.policy_version != policy_version:
            raise DomainPolicyError(
                ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
                "Approved effect no longer matches the requested effect.",
            )
        if (
            approval.effect_fingerprint != effect_fingerprint
            or approval.effect != effect
        ):
            raise DomainPolicyError(
                ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
                "Approved effect no longer matches the requested effect.",
            )
        updated = connection.execute(
            """
            UPDATE approvals SET consumed_at = ?
            WHERE approval_id = ? AND consumed_at IS NULL
            """,
            (now.isoformat(), approval_id),
        )
        if updated.rowcount != 1:
            raise DomainPolicyError(
                ReasonCode.AUTHORIZATION_ALREADY_CONSUMED,
                "Approval has already been consumed.",
            )
        self._append_audit(
            connection,
            occurred_at=now,
            event_type=AuditEventType.AUTHORIZATION_CONSUMED,
            effect_fingerprint=effect_fingerprint,
            idempotency_key=effect.idempotency_key,
            details={"approval_id": approval_id},
        )

    def _assert_effect_target_in_scope(
        self,
        connection: sqlite3.Connection,
        principal: AuthenticatedPrincipal,
        effect: EffectSpec,
    ) -> None:
        if effect.order_id is not None:
            self._scoped_order(connection, principal, effect.order_id)
        elif effect.user_id is not None and effect.user_id != principal.user_id:
            raise _resource_not_available()

    def _scoped_order(
        self,
        connection: sqlite3.Connection,
        principal: AuthenticatedPrincipal,
        order_id: str | None,
    ) -> OrderRecord:
        if order_id is None:
            raise ValueError("operation has no order target")
        row = connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None or (
            row["tenant_id"] != principal.tenant_id
            or row["owner_user_id"] != principal.user_id
        ):
            raise _resource_not_available()
        return _order_from_row(row)

    def _load_receipt(
        self, connection: sqlite3.Connection, idempotency_key: str
    ) -> OperationReceipt | None:
        row = connection.execute(
            "SELECT * FROM operation_receipts WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return _receipt_from_row(row) if row is not None else None

    def _insert_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        effect: EffectSpec,
        request_fingerprint: str,
        result: Mapping[str, JsonValue],
        committed_at: datetime,
    ) -> OperationReceipt:
        assert effect.idempotency_key is not None
        receipt = OperationReceipt(
            idempotency_key=effect.idempotency_key,
            operation=effect.operation,
            request_fingerprint=request_fingerprint,
            effect_fingerprint=decision_fingerprint(effect),
            result=result,
            committed_at=committed_at,
        )
        connection.execute(
            """
            INSERT INTO operation_receipts(
                idempotency_key, operation, request_fingerprint,
                effect_fingerprint, result_json, committed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.idempotency_key,
                receipt.operation.value,
                receipt.request_fingerprint,
                receipt.effect_fingerprint,
                _json_dump(receipt.result),
                receipt.committed_at.isoformat(),
            ),
        )
        return receipt

    def _append_audit(
        self,
        connection: sqlite3.Connection,
        *,
        occurred_at: datetime,
        event_type: AuditEventType,
        reason_code: ReasonCode | None = None,
        effect_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> AuditEventRecord:
        event_id = uuid4()
        key_hash = (
            hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
            if idempotency_key is not None
            else None
        )
        safe_details = dict(details or {})
        cursor = connection.execute(
            """
            INSERT INTO audit_events(
                event_id, occurred_at, event_type, reason_code,
                effect_fingerprint, idempotency_key_hash, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event_id),
                occurred_at.isoformat(),
                event_type.value,
                reason_code.value if reason_code is not None else None,
                effect_fingerprint,
                key_hash,
                _json_dump(safe_details),
            ),
        )
        sequence = cursor.lastrowid
        assert sequence is not None
        return AuditEventRecord(
            sequence=sequence,
            event_id=event_id,
            occurred_at=occurred_at,
            event_type=event_type,
            reason_code=reason_code,
            effect_fingerprint=effect_fingerprint,
            idempotency_key_hash=key_hash,
            details=safe_details,
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY(tenant_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS orders(
                order_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                owner_user_id TEXT NOT NULL,
                paid_amount_cents INTEGER NOT NULL CHECK(paid_amount_cents >= 0),
                refunded_amount_cents INTEGER NOT NULL DEFAULT 0
                    CHECK(refunded_amount_cents >= 0),
                status TEXT NOT NULL,
                address_ref TEXT,
                description TEXT,
                version INTEGER NOT NULL CHECK(version >= 0),
                FOREIGN KEY(tenant_id, owner_user_id)
                    REFERENCES users(tenant_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS payments(
                payment_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL REFERENCES orders(order_id),
                display_reference TEXT NOT NULL,
                sensitive_canary TEXT
            );
            CREATE TABLE IF NOT EXISTS approvals(
                approval_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                effect_json TEXT NOT NULL,
                effect_fingerprint TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                consumed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS refund_operations(
                operation_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                order_id TEXT NOT NULL REFERENCES orders(order_id),
                amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0),
                original_order_version INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_fingerprint TEXT NOT NULL,
                effect_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                external_ref TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS coupon_grants(
                grant_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0),
                idempotency_key TEXT NOT NULL UNIQUE,
                request_fingerprint TEXT NOT NULL,
                effect_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operation_receipts(
                idempotency_key TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                effect_fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL,
                committed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reason_code TEXT,
                effect_fingerprint TEXT,
                idempotency_key_hash TEXT,
                details_json TEXT NOT NULL
            );
            """
        )


def _order_from_row(row: sqlite3.Row) -> OrderRecord:
    return OrderRecord(
        order_id=row["order_id"],
        tenant_id=row["tenant_id"],
        owner_user_id=row["owner_user_id"],
        paid_amount_cents=row["paid_amount_cents"],
        refunded_amount_cents=row["refunded_amount_cents"],
        status=OrderStatus(row["status"]),
        address_ref=row["address_ref"],
        description=row["description"],
        version=row["version"],
    )


def _payment_from_row(row: sqlite3.Row) -> PaymentRecord:
    return PaymentRecord(
        payment_id=row["payment_id"],
        order_id=row["order_id"],
        display_reference=row["display_reference"],
        sensitive_canary=row["sensitive_canary"],
    )


def _approval_from_row(row: sqlite3.Row) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=row["approval_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        effect=EffectSpec.model_validate(_json_load(row["effect_json"])),
        effect_fingerprint=row["effect_fingerprint"],
        policy_version=row["policy_version"],
        created_at=datetime.fromisoformat(row["created_at"]),
        expires_at=datetime.fromisoformat(row["expires_at"]),
        consumed_at=(
            datetime.fromisoformat(row["consumed_at"])
            if row["consumed_at"] is not None
            else None
        ),
    )


def _refund_from_row(row: sqlite3.Row) -> RefundOperationRecord:
    result = _json_load(row["result_json"]) if row["result_json"] is not None else None
    if result is not None and not isinstance(result, dict):
        raise ValueError("refund result must be a JSON object")
    return RefundOperationRecord(
        operation_id=row["operation_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        order_id=row["order_id"],
        amount_cents=row["amount_cents"],
        original_order_version=row["original_order_version"],
        idempotency_key=row["idempotency_key"],
        request_fingerprint=row["request_fingerprint"],
        effect_fingerprint=row["effect_fingerprint"],
        status=RefundOperationStatus(row["status"]),
        external_ref=row["external_ref"],
        result=result,
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _coupon_from_row(row: sqlite3.Row) -> CouponGrantRecord:
    return CouponGrantRecord(
        grant_id=row["grant_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        amount_cents=row["amount_cents"],
        idempotency_key=row["idempotency_key"],
        request_fingerprint=row["request_fingerprint"],
        effect_fingerprint=row["effect_fingerprint"],
        status=CouponGrantStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _receipt_from_row(row: sqlite3.Row) -> OperationReceipt:
    result = _json_load(row["result_json"])
    if not isinstance(result, dict):
        raise ValueError("operation receipt result must be a JSON object")
    return OperationReceipt(
        idempotency_key=row["idempotency_key"],
        operation=OperationType(row["operation"]),
        request_fingerprint=row["request_fingerprint"],
        effect_fingerprint=row["effect_fingerprint"],
        result=result,
        committed_at=datetime.fromisoformat(row["committed_at"]),
    )


def _audit_from_row(row: sqlite3.Row) -> AuditEventRecord:
    details = _json_load(row["details_json"])
    if not isinstance(details, dict):
        raise ValueError("audit details must be a JSON object")
    return AuditEventRecord(
        sequence=row["sequence"],
        event_id=UUID(row["event_id"]),
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        event_type=AuditEventType(row["event_type"]),
        reason_code=(
            ReasonCode(row["reason_code"])
            if row["reason_code"] is not None
            else None
        ),
        effect_fingerprint=row["effect_fingerprint"],
        idempotency_key_hash=row["idempotency_key_hash"],
        details=details,
    )


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_load(value: str) -> Any:
    return json.loads(value)


def _resource_not_available() -> DomainPolicyError:
    return DomainPolicyError(
        ReasonCode.RESOURCE_NOT_AVAILABLE,
        "The requested resource is not available.",
    )


def _idempotency_conflict() -> DomainPolicyError:
    return DomainPolicyError(
        ReasonCode.IDEMPOTENCY_KEY_CONFLICT,
        "The idempotency key is already bound to another request.",
    )
