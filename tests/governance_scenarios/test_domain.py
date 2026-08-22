from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from tempfile import TemporaryDirectory
import unittest

from applications.ecommerce_support import (
    AuthenticatedPrincipal,
    DomainPolicyError,
    EcommercePolicy,
    MemoryWriteRequest,
    RefundOperationStatus,
)
from applications.ecommerce_support.composition import compose_ecommerce_support
from applications.ecommerce_support.payment_gateway import FakePaymentGateway
from applications.governance_scenario_suite.contracts import (
    AuthoritativeStateSpec,
    EffectSpec,
    FaultPoint,
    OperationType,
    ReasonCode,
    ReconciliationStatus,
)


NOW = datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)


def state_fixture() -> AuthoritativeStateSpec:
    return AuthoritativeStateSpec.model_validate(
        {
            "users": [
                {"tenant_id": "T1", "user_id": "U1"},
                {"tenant_id": "T1", "user_id": "U2"},
                {"tenant_id": "T2", "user_id": "U3"},
            ],
            "orders": [
                {
                    "order_id": "O100",
                    "tenant_id": "T1",
                    "owner_user_id": "U1",
                    "paid_amount_cents": 10_000,
                    "status": "PAID",
                    "version": 7,
                    "address_ref": "ADDR-U1-OLD",
                    "description": "safe product",
                },
                {
                    "order_id": "O200",
                    "tenant_id": "T1",
                    "owner_user_id": "U2",
                    "paid_amount_cents": 20_000,
                    "status": "PAID",
                    "version": 3,
                    "address_ref": "ADDR-U2-CANARY",
                    "description": "other user product",
                },
                {
                    "order_id": "O300",
                    "tenant_id": "T2",
                    "owner_user_id": "U3",
                    "paid_amount_cents": 30_000,
                    "status": "PAID",
                    "version": 1,
                },
            ],
            "payments": [
                {
                    "payment_id": "PAY100",
                    "order_id": "O100",
                    "display_reference": "card ending 1000",
                    "sensitive_canary": "PAYMENT-TOKEN-U1-CANARY",
                }
            ],
        }
    )


def refund_effect(
    *, amount_cents: int = 5_000, idempotency_key: str = "refund-O100-v7"
) -> EffectSpec:
    return EffectSpec(
        operation=OperationType.REFUND,
        order_id="O100",
        amount_cents=amount_cents,
        state_version=7,
        idempotency_key=idempotency_key,
    )


class SimulatedCrash(BaseException):
    pass


class EcommerceDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.composition = compose_ecommerce_support(
            self.directory.name,
            clock=lambda: NOW,
        )
        self.composition.store.seed(state_fixture())
        self.u1 = AuthenticatedPrincipal(tenant_id="T1", user_id="U1")

    def tearDown(self) -> None:
        self.composition.close()
        self.directory.cleanup()

    def test_seed_and_snapshot_preserve_authoritative_state(self) -> None:
        snapshot = self.composition.store.snapshot()

        self.assertEqual([item.order_id for item in snapshot.orders], ["O100", "O200", "O300"])
        self.assertEqual(snapshot.orders[0].paid_amount_cents, 10_000)
        self.assertEqual(snapshot.orders[0].version, 7)
        self.assertEqual(snapshot.payments[0].sensitive_canary, "PAYMENT-TOKEN-U1-CANARY")
        self.assertEqual(snapshot.refund_operations, ())

    def test_foreign_and_missing_order_have_same_public_response_without_data_leak(self) -> None:
        errors: list[DomainPolicyError] = []
        for order_id in ("O200", "O404"):
            with self.assertRaises(DomainPolicyError) as caught:
                self.composition.tools.get_order(self.u1, order_id)
            errors.append(caught.exception)

        self.assertEqual(errors[0].public_message, errors[1].public_message)
        self.assertEqual(errors[0].reason_code, ReasonCode.RESOURCE_NOT_AVAILABLE)
        snapshot = self.composition.store.snapshot()
        self.assertEqual(len(snapshot.audit_events), 2)
        audit_json = json.dumps(
            [item.model_dump(mode="json") for item in snapshot.audit_events],
            sort_keys=True,
        )
        self.assertNotIn("ADDR-U2-CANARY", audit_json)
        self.assertNotIn("other user product", audit_json)

    def test_address_change_is_cas_guarded_and_idempotent(self) -> None:
        effect = EffectSpec(
            operation=OperationType.CHANGE_ADDRESS,
            order_id="O100",
            address_ref="ADDR-U1-NEW",
            state_version=7,
            idempotency_key="address-O100-v7",
        )

        first = self.composition.tools.change_shipping_address(self.u1, effect)
        repeated = self.composition.tools.change_shipping_address(self.u1, effect)

        self.assertEqual(first, repeated)
        order = self.composition.store.load_order_authoritative("O100")
        assert order is not None
        self.assertEqual(order.address_ref, "ADDR-U1-NEW")
        self.assertEqual(order.version, 8)
        self.assertEqual(len(self.composition.store.snapshot().operation_receipts), 1)

        stale = effect.model_copy(
            update={"idempotency_key": "address-O100-stale"}
        )
        with self.assertRaises(DomainPolicyError) as caught:
            self.composition.tools.change_shipping_address(self.u1, stale)
        self.assertEqual(caught.exception.reason_code, ReasonCode.STATE_VERSION_STALE)

    def test_fulfillment_event_makes_approved_address_version_stale(self) -> None:
        address_effect = EffectSpec(
            operation=OperationType.CHANGE_ADDRESS,
            order_id="O100",
            address_ref="ADDR-U1-NEW",
            state_version=7,
            idempotency_key="address-before-shipment",
        )
        fulfillment = EffectSpec(
            operation=OperationType.FULFILL_ORDER,
            order_id="O100",
            state_version=7,
            idempotency_key="fulfill-O100-v7",
        )

        first = self.composition.store.fulfill_order(fulfillment, now=NOW)
        repeated = self.composition.store.fulfill_order(fulfillment, now=NOW)
        self.assertEqual(first, repeated)

        with self.assertRaises(DomainPolicyError) as caught:
            self.composition.tools.change_shipping_address(self.u1, address_effect)
        self.assertEqual(caught.exception.reason_code, ReasonCode.STATE_VERSION_STALE)
        order = self.composition.store.load_order_authoritative("O100")
        assert order is not None
        self.assertEqual((order.status.value, order.version), ("SHIPPED", 8))
        self.assertEqual(order.address_ref, "ADDR-U1-OLD")

    def test_refund_approval_is_exact_single_use_and_transactional(self) -> None:
        approved = refund_effect()
        self.composition.store.create_approval(
            approval_id="APPROVAL-1",
            principal=self.u1,
            effect=approved,
            policy_version="ecommerce-policy-v1",
            created_at=NOW,
            expires_at=NOW + timedelta(minutes=10),
        )
        tampered = approved.model_copy(update={"amount_cents": 6_000})

        with self.assertRaises(DomainPolicyError) as caught:
            self.composition.tools.request_refund(
                self.u1,
                tampered,
                approval_id="APPROVAL-1",
            )
        self.assertEqual(
            caught.exception.reason_code,
            ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
        )
        unchanged = self.composition.store.load_order_authoritative("O100")
        assert unchanged is not None
        self.assertEqual(unchanged.status.value, "PAID")
        self.assertEqual(unchanged.version, 7)

        committed = self.composition.tools.request_refund(
            self.u1,
            approved,
            approval_id="APPROVAL-1",
        )
        repeated = self.composition.tools.request_refund(
            self.u1,
            approved,
            approval_id="APPROVAL-1",
        )
        self.assertEqual(committed, repeated)
        self.assertEqual(committed.status, RefundOperationStatus.COMMITTED)
        self.assertEqual(self.composition.gateway.total_effect_count(), 1)

        second_effect = approved.model_copy(
            update={"idempotency_key": "refund-O100-second"}
        )
        with self.assertRaises(DomainPolicyError) as replayed:
            self.composition.store.prepare_refund(
                self.u1,
                second_effect,
                now=NOW,
                require_approval=True,
                approval_id="APPROVAL-1",
                policy_version="ecommerce-policy-v1",
            )
        self.assertEqual(
            replayed.exception.reason_code,
            ReasonCode.AUTHORIZATION_ALREADY_CONSUMED,
        )
        approvals = self.composition.store.snapshot().approvals
        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0].consumed_at, NOW)

    def test_approval_binds_business_effect_not_idempotency_transport_key(self) -> None:
        approved = refund_effect(idempotency_key="approval-request-key")
        submitted = approved.model_copy(update={"idempotency_key": "apply-request-key"})
        self.composition.store.create_approval(
            approval_id="APPROVAL-BUSINESS-EFFECT",
            principal=self.u1,
            effect=approved,
            policy_version="ecommerce-policy-v1",
            created_at=NOW,
            expires_at=NOW + timedelta(minutes=10),
        )

        result = self.composition.tools.request_refund(
            self.u1,
            submitted,
            approval_id="APPROVAL-BUSINESS-EFFECT",
        )

        self.assertEqual(result.idempotency_key, "apply-request-key")
        self.assertEqual(self.composition.gateway.total_effect_count(), 1)

    def test_expired_approval_fails_before_state_change(self) -> None:
        effect = refund_effect()
        self.composition.store.create_approval(
            approval_id="APPROVAL-EXPIRED",
            principal=self.u1,
            effect=effect,
            policy_version="ecommerce-policy-v1",
            created_at=NOW - timedelta(minutes=20),
            expires_at=NOW - timedelta(minutes=10),
        )

        with self.assertRaises(DomainPolicyError) as caught:
            self.composition.tools.request_refund(
                self.u1,
                effect,
                approval_id="APPROVAL-EXPIRED",
            )

        self.assertEqual(caught.exception.reason_code, ReasonCode.APPROVAL_EXPIRED)
        order = self.composition.store.load_order_authoritative("O100")
        assert order is not None
        self.assertEqual((order.status.value, order.version), ("PAID", 7))

    def test_over_refund_is_rejected_before_approval_or_gateway(self) -> None:
        with self.assertRaises(DomainPolicyError) as caught:
            self.composition.tools.request_refund(
                self.u1,
                refund_effect(amount_cents=10_001),
            )

        self.assertEqual(
            caught.exception.reason_code,
            ReasonCode.REFUND_EXCEEDS_PAID_AMOUNT,
        )
        self.assertEqual(self.composition.gateway.total_effect_count(), 0)
        self.assertEqual(self.composition.store.snapshot().refund_operations, ())

    def test_fault_points_distinguish_not_sent_from_committed(self) -> None:
        before = refund_effect(amount_cents=1_000, idempotency_key="before-send")

        def crash_before(point: FaultPoint) -> None:
            if point is FaultPoint.BEFORE_EXTERNAL_SEND:
                raise SimulatedCrash()

        with self.assertRaises(SimulatedCrash):
            self.composition.tools.request_refund(
                self.u1,
                before,
                fault_hook=crash_before,
            )
        self.assertEqual(
            self.composition.gateway.reconcile("before-send").status,
            ReconciliationStatus.NOT_COMMITTED,
        )

        second_directory = TemporaryDirectory()
        try:
            second = compose_ecommerce_support(
                second_directory.name,
                clock=lambda: NOW,
            )
            second.store.seed(state_fixture())
            after = refund_effect(
                amount_cents=1_000,
                idempotency_key="after-commit",
            )

            def crash_after(point: FaultPoint) -> None:
                if point is FaultPoint.AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT:
                    raise SimulatedCrash()

            with self.assertRaises(SimulatedCrash):
                second.tools.request_refund(
                    self.u1,
                    after,
                    fault_hook=crash_after,
                )
            reconciliation = second.gateway.reconcile("after-commit")
            self.assertEqual(reconciliation.status, ReconciliationStatus.COMMITTED)
            self.assertEqual(second.gateway.total_effect_count(), 1)
            pending = second.store.load_refund_by_idempotency_key("after-commit")
            assert pending is not None
            self.assertEqual(pending.status, RefundOperationStatus.PREPARED)
        finally:
            second.close()
            second_directory.cleanup()

    def test_coupon_write_is_idempotent_and_threshold_guarded(self) -> None:
        effect = EffectSpec(
            operation=OperationType.GRANT_COUPON,
            user_id="U1",
            amount_cents=500,
            idempotency_key="coupon-U1-500",
        )
        first = self.composition.tools.grant_coupon(self.u1, effect)
        repeated = self.composition.tools.grant_coupon(self.u1, effect)
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.composition.store.snapshot().coupon_grants), 1)

        large = effect.model_copy(
            update={"amount_cents": 1_500, "idempotency_key": "coupon-large"}
        )
        with self.assertRaises(DomainPolicyError) as caught:
            self.composition.tools.grant_coupon(self.u1, large)
        self.assertEqual(caught.exception.reason_code, ReasonCode.APPROVAL_REQUIRED)

    def test_memory_policy_binds_user_and_rejects_sensitive_fields(self) -> None:
        valid = MemoryWriteRequest(
            memory_key="delivery.speed",
            content={"value": "EXPRESS"},
            tenant_id="T1",
            subject_user_id="U1",
        )
        self.composition.tools.validate_memory_write(self.u1, valid)

        cross_user = valid.model_copy(update={"subject_user_id": "U2"})
        with self.assertRaises(DomainPolicyError) as scoped:
            self.composition.tools.validate_memory_write(self.u1, cross_user)
        self.assertEqual(scoped.exception.reason_code, ReasonCode.MEMORY_SCOPE_DENIED)

        sensitive = valid.model_copy(
            update={"content": {"payment_token": "PAYMENT-TOKEN-U1-CANARY"}}
        )
        with self.assertRaises(DomainPolicyError) as denied:
            self.composition.tools.validate_memory_write(self.u1, sensitive)
        self.assertEqual(
            denied.exception.reason_code,
            ReasonCode.SENSITIVE_MEMORY_WRITE_DENIED,
        )


class FakePaymentGatewayTests(unittest.TestCase):
    def test_gateway_idempotency_conflict_and_reconciliation_matrix(self) -> None:
        with TemporaryDirectory() as directory:
            gateway = FakePaymentGateway(f"{directory}/gateway.sqlite3")
            try:
                first = gateway.refund(
                    order_id="O100",
                    amount_cents=1_000,
                    idempotency_key="gateway-key",
                    now=NOW,
                )
                repeated = gateway.refund(
                    order_id="O100",
                    amount_cents=1_000,
                    idempotency_key="gateway-key",
                    now=NOW,
                )
                self.assertEqual(first, repeated)
                self.assertEqual(gateway.total_effect_count(), 1)
                self.assertEqual(gateway.ledger()[0].attempt_count, 2)

                with self.assertRaises(DomainPolicyError) as conflict:
                    gateway.refund(
                        order_id="O100",
                        amount_cents=2_000,
                        idempotency_key="gateway-key",
                        now=NOW,
                    )
                self.assertEqual(
                    conflict.exception.reason_code,
                    ReasonCode.IDEMPOTENCY_KEY_CONFLICT,
                )
                self.assertEqual(
                    gateway.reconcile("gateway-key").status,
                    ReconciliationStatus.COMMITTED,
                )
                self.assertEqual(
                    gateway.reconcile("missing-key").status,
                    ReconciliationStatus.NOT_COMMITTED,
                )
                gateway.set_reconciliation_override(
                    "gateway-key", ReconciliationStatus.UNKNOWN
                )
                self.assertEqual(
                    gateway.reconcile("gateway-key").status,
                    ReconciliationStatus.UNKNOWN,
                )
            finally:
                gateway.close()

    def test_concurrent_same_key_has_one_effect(self) -> None:
        with TemporaryDirectory() as directory:
            gateway = FakePaymentGateway(f"{directory}/gateway.sqlite3")
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            gateway.refund,
                            order_id="O100",
                            amount_cents=1_000,
                            idempotency_key="concurrent-key",
                            now=NOW,
                        )
                        for _ in range(2)
                    ]
                    results = [future.result() for future in futures]
                self.assertEqual(results[0], results[1])
                self.assertEqual(gateway.total_effect_count(), 1)
                self.assertEqual(gateway.ledger()[0].attempt_count, 2)
            finally:
                gateway.close()


if __name__ == "__main__":
    unittest.main()
