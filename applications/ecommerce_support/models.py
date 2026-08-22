"""Immutable ecommerce records at the authoritative domain boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Mapping, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    model_validator,
)

from applications.governance_scenario_suite.contracts import (
    AuditEventType,
    EffectSpec,
    MemorySensitivity,
    OperationType,
    OrderStatus,
    ReasonCode,
    ReconciliationStatus,
)


Cents = Annotated[StrictInt, Field(ge=0)]


class EcommerceModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class AuthenticatedPrincipal(EcommerceModel):
    """Trusted identity supplied by the host application, never by the model."""

    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    roles: tuple[str, ...] = ()


class UserRecord(EcommerceModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


class OrderRecord(EcommerceModel):
    order_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    paid_amount_cents: Cents
    refunded_amount_cents: Cents = 0
    status: OrderStatus
    address_ref: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    version: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_refund_total(self) -> Self:
        if self.refunded_amount_cents > self.paid_amount_cents:
            raise ValueError("refunded amount cannot exceed paid amount")
        return self


class PaymentRecord(EcommerceModel):
    payment_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    display_reference: str = Field(min_length=1)
    sensitive_canary: str | None = Field(default=None, min_length=1)


class ApprovalRecord(EcommerceModel):
    approval_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    effect: EffectSpec
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: str = Field(min_length=1)
    expires_at: AwareDatetime
    created_at: AwareDatetime
    consumed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("approval expiry must follow creation")
        if self.consumed_at is not None and self.consumed_at < self.created_at:
            raise ValueError("approval consumption cannot precede creation")
        return self


class RefundOperationStatus(StrEnum):
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    IN_DOUBT = "IN_DOUBT"


class RefundOperationRecord(EcommerceModel):
    operation_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    amount_cents: Cents
    original_order_version: StrictInt = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RefundOperationStatus
    external_ref: str | None = Field(default=None, min_length=1)
    result: Mapping[str, JsonValue] | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class CouponGrantStatus(StrEnum):
    COMMITTED = "COMMITTED"


class CouponGrantRecord(EcommerceModel):
    grant_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    amount_cents: Cents
    idempotency_key: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CouponGrantStatus = CouponGrantStatus.COMMITTED
    created_at: AwareDatetime


class OperationReceipt(EcommerceModel):
    idempotency_key: str = Field(min_length=1)
    operation: OperationType
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: Mapping[str, JsonValue]
    committed_at: AwareDatetime


class AuditEventRecord(EcommerceModel):
    sequence: StrictInt = Field(ge=1)
    event_id: UUID
    occurred_at: AwareDatetime
    event_type: AuditEventType
    reason_code: ReasonCode | None = None
    effect_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    idempotency_key_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    details: Mapping[str, JsonValue] = Field(default_factory=dict)


class GatewayRefundResult(EcommerceModel):
    external_ref: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    amount_cents: Cents
    committed_at: AwareDatetime


class GatewayReconciliation(EcommerceModel):
    status: ReconciliationStatus
    result: GatewayRefundResult | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.status is ReconciliationStatus.COMMITTED:
            if self.result is None:
                raise ValueError("committed reconciliation requires a result")
        elif self.result is not None:
            raise ValueError("only committed reconciliation can include a result")
        return self


class ExternalLedgerRecord(EcommerceModel):
    idempotency_key: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: GatewayRefundResult
    attempt_count: StrictInt = Field(ge=1)
    effect_count: StrictInt = Field(ge=1)


class MemoryWriteRequest(EcommerceModel):
    memory_key: str = Field(min_length=1)
    content: Mapping[str, JsonValue]
    tenant_id: str = Field(min_length=1)
    subject_user_id: str | None = Field(default=None, min_length=1)
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL
    category: str = Field(default="user_preference", min_length=1)


class DomainSnapshot(EcommerceModel):
    users: tuple[UserRecord, ...]
    orders: tuple[OrderRecord, ...]
    payments: tuple[PaymentRecord, ...]
    approvals: tuple[ApprovalRecord, ...]
    refund_operations: tuple[RefundOperationRecord, ...]
    coupon_grants: tuple[CouponGrantRecord, ...]
    operation_receipts: tuple[OperationReceipt, ...]
    audit_events: tuple[AuditEventRecord, ...]
