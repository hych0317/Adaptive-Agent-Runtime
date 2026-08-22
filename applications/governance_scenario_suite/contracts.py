"""Strict, versioned contracts for governance scenario specifications."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Mapping, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    model_validator,
)


ScenarioIdentifier = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]+$")]
Cents = Annotated[StrictInt, Field(ge=0)]
PositiveStrictInt = Annotated[StrictInt, Field(ge=1)]


class ScenarioContractModel(BaseModel):
    """Immutable strict model used by every checked-in scenario contract."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class ScenarioProfile(StrEnum):
    PLAIN_AGENT = "plain_agent"
    FULL_AAR = "full_aar"
    NO_EXACT_EFFECT_BINDING = "no_exact_effect_binding"
    NO_MEMORY_SCOPE_FILTER = "no_memory_scope_filter"
    NO_AUTHORITATIVE_CONSTRAINT_CHECK = "no_authoritative_constraint_check"
    NO_RECONCILE_FAIL_CLOSED = "no_reconcile_fail_closed"
    NO_RECONCILE_BLIND_RETRY = "no_reconcile_blind_retry"


class EvaluationVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class ScenarioVerdict(StrEnum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    EXPIRE = "EXPIRE"
    FAIL_CLOSED = "FAIL_CLOSED"
    WAIT_FOR_REVIEW = "WAIT_FOR_REVIEW"


class ReasonCode(StrEnum):
    RESOURCE_SCOPE_DENIED = "RESOURCE_SCOPE_DENIED"
    RESOURCE_NOT_AVAILABLE = "RESOURCE_NOT_AVAILABLE"
    EFFECT_NOT_EQUAL_TO_APPROVAL = "EFFECT_NOT_EQUAL_TO_APPROVAL"
    AUTHORIZATION_ALREADY_CONSUMED = "AUTHORIZATION_ALREADY_CONSUMED"
    IDEMPOTENT_RESULT_REUSED = "IDEMPOTENT_RESULT_REUSED"
    IDEMPOTENCY_KEY_CONFLICT = "IDEMPOTENCY_KEY_CONFLICT"
    STATE_VERSION_STALE = "STATE_VERSION_STALE"
    REFUND_EXCEEDS_PAID_AMOUNT = "REFUND_EXCEEDS_PAID_AMOUNT"
    OPERATION_NOT_ALLOWED_IN_STATE = "OPERATION_NOT_ALLOWED_IN_STATE"
    RECONCILIATION_COMMITTED = "RECONCILIATION_COMMITTED"
    RECONCILIATION_NOT_COMMITTED = "RECONCILIATION_NOT_COMMITTED"
    RECONCILIATION_UNKNOWN = "RECONCILIATION_UNKNOWN"
    MEMORY_SCOPE_DENIED = "MEMORY_SCOPE_DENIED"
    SENSITIVE_MEMORY_WRITE_DENIED = "SENSITIVE_MEMORY_WRITE_DENIED"
    PROMPT_INJECTION_EFFECT_DENIED = "PROMPT_INJECTION_EFFECT_DENIED"


class FaultPoint(StrEnum):
    BEFORE_EXTERNAL_SEND = "BEFORE_EXTERNAL_SEND"
    AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT = (
        "AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT"
    )
    DURING_RECONCILIATION_QUERY = "DURING_RECONCILIATION_QUERY"


class ReconciliationStatus(StrEnum):
    COMMITTED = "COMMITTED"
    NOT_COMMITTED = "NOT_COMMITTED"
    UNKNOWN = "UNKNOWN"


class AuditEventType(StrEnum):
    PROPOSAL_RECORDED = "PROPOSAL_RECORDED"
    EFFECT_MISMATCH_REJECTED = "EFFECT_MISMATCH_REJECTED"
    RESOURCE_SCOPE_DENIED = "RESOURCE_SCOPE_DENIED"
    AUTHORIZATION_RESERVED = "AUTHORIZATION_RESERVED"
    AUTHORIZATION_CONSUMED = "AUTHORIZATION_CONSUMED"
    AUTHORIZATION_REPLAY_REJECTED = "AUTHORIZATION_REPLAY_REJECTED"
    EXTERNAL_SEND_STARTED = "EXTERNAL_SEND_STARTED"
    EXTERNAL_EFFECT_COMMITTED = "EXTERNAL_EFFECT_COMMITTED"
    APPLY_RECEIPT_COMMITTED = "APPLY_RECEIPT_COMMITTED"
    RECONCILIATION_STARTED = "RECONCILIATION_STARTED"
    RECONCILIATION_RESOLVED = "RECONCILIATION_RESOLVED"
    MEMORY_WRITE_DENIED = "MEMORY_WRITE_DENIED"
    MEMORY_RECALL_FILTERED = "MEMORY_RECALL_FILTERED"


class OperationType(StrEnum):
    GET_ORDER = "GET_ORDER"
    CHANGE_ADDRESS = "CHANGE_ADDRESS"
    REFUND = "REFUND"
    REFUND_TO_CREDIT = "REFUND_TO_CREDIT"
    GRANT_COUPON = "GRANT_COUPON"
    GET_OPERATION_RESULT = "GET_OPERATION_RESULT"


class OrderStatus(StrEnum):
    PAID = "PAID"
    SHIPPED = "SHIPPED"
    REFUND_PENDING = "REFUND_PENDING"
    REFUNDED = "REFUNDED"


class MemorySensitivity(StrEnum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class PrincipalSpec(ScenarioContractModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    roles: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_roles(self) -> Self:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("principal roles must be unique")
        if any(not item for item in self.roles):
            raise ValueError("principal roles cannot be empty")
        return self


class UserFixture(ScenarioContractModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


class OrderFixture(ScenarioContractModel):
    order_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    paid_amount_cents: Cents
    status: OrderStatus
    version: StrictInt = Field(ge=0)
    address_ref: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)


class PaymentFixture(ScenarioContractModel):
    payment_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    display_reference: str = Field(min_length=1)
    sensitive_canary: str | None = Field(default=None, min_length=1)


class MemoryFixture(ScenarioContractModel):
    memory_key: str = Field(min_length=1)
    content: JsonValue
    tenant_id: str = Field(min_length=1)
    project_id: str = Field(default="ecommerce_support", min_length=1)
    agent_scope: str = Field(default="support", min_length=1)
    subject_user_id: str | None = Field(default=None, min_length=1)
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL
    condition_facts: Mapping[str, JsonValue] = Field(default_factory=dict)
    required_tags: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, gt=0.0, le=1.0)
    canary: str | None = Field(default=None, min_length=1)


class AuthoritativeStateSpec(ScenarioContractModel):
    users: tuple[UserFixture, ...] = ()
    orders: tuple[OrderFixture, ...] = ()
    payments: tuple[PaymentFixture, ...] = ()
    memories: tuple[MemoryFixture, ...] = ()

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        _require_unique((item.user_id for item in self.users), "user fixture ids")
        _require_unique((item.order_id for item in self.orders), "order fixture ids")
        _require_unique(
            (item.payment_id for item in self.payments), "payment fixture ids"
        )
        order_ids = {item.order_id for item in self.orders}
        missing_payment_orders = {
            item.order_id for item in self.payments if item.order_id not in order_ids
        }
        if missing_payment_orders:
            raise ValueError(
                "payments reference missing orders: "
                + ", ".join(sorted(missing_payment_orders))
            )
        return self


class EffectSpec(ScenarioContractModel):
    operation: OperationType
    order_id: str | None = Field(default=None, min_length=1)
    user_id: str | None = Field(default=None, min_length=1)
    amount_cents: Cents | None = None
    address_ref: str | None = Field(default=None, min_length=1)
    state_version: StrictInt | None = Field(default=None, ge=0)
    idempotency_key: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_operation_fields(self) -> Self:
        required: dict[OperationType, frozenset[str]] = {
            OperationType.GET_ORDER: frozenset({"order_id"}),
            OperationType.CHANGE_ADDRESS: frozenset(
                {"order_id", "address_ref", "state_version", "idempotency_key"}
            ),
            OperationType.REFUND: frozenset(
                {"order_id", "amount_cents", "state_version", "idempotency_key"}
            ),
            OperationType.REFUND_TO_CREDIT: frozenset(
                {"order_id", "amount_cents", "state_version", "idempotency_key"}
            ),
            OperationType.GRANT_COUPON: frozenset(
                {"user_id", "amount_cents", "idempotency_key"}
            ),
            OperationType.GET_OPERATION_RESULT: frozenset({"idempotency_key"}),
        }
        present = {
            name
            for name in (
                "order_id",
                "user_id",
                "amount_cents",
                "address_ref",
                "state_version",
                "idempotency_key",
            )
            if getattr(self, name) is not None
        }
        missing = required[self.operation] - present
        if missing:
            raise ValueError(
                f"{self.operation.value} effect is missing fields: "
                + ", ".join(sorted(missing))
            )
        allowed = required[self.operation]
        extras = present - allowed
        if extras:
            raise ValueError(
                f"{self.operation.value} effect contains unrelated fields: "
                + ", ".join(sorted(extras))
            )
        return self


class ModelScriptSpec(ScenarioContractModel):
    proposals: tuple[EffectSpec, ...] = Field(min_length=1)
    final_response: str | None = Field(default=None, min_length=1)


class FaultScheduleSpec(ScenarioContractModel):
    point: FaultPoint
    occurrence: PositiveStrictInt = 1
    reconciliation_status: ReconciliationStatus | None = None

    @model_validator(mode="after")
    def validate_reconciliation_fault(self) -> Self:
        if (
            self.point is FaultPoint.DURING_RECONCILIATION_QUERY
            and self.reconciliation_status is None
        ):
            raise ValueError(
                "a reconciliation-query fault requires reconciliation_status"
            )
        return self


class MutationExpectation(ScenarioContractModel):
    proposal_index: StrictInt = Field(default=0, ge=0)
    changed_fields: tuple[str, ...] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_field_names(self) -> Self:
        allowed = set(EffectSpec.model_fields)
        if any(item not in allowed for item in self.changed_fields):
            raise ValueError("mutation expectation names an unknown Effect field")
        return self


class ExpectedOrderState(ScenarioContractModel):
    order_id: str = Field(min_length=1)
    status: OrderStatus | None = None
    version: StrictInt | None = Field(default=None, ge=0)
    address_ref: str | None = Field(default=None, min_length=1)
    refunded_amount_cents: Cents | None = None


class AuthoritativeStateExpectation(ScenarioContractModel):
    orders: tuple[ExpectedOrderState, ...] = ()
    refund_operation_count: StrictInt | None = Field(default=None, ge=0)
    coupon_grant_count: StrictInt | None = Field(default=None, ge=0)


class ExternalLedgerExpectation(ScenarioContractModel):
    effect_count: StrictInt = Field(ge=0)
    attempt_count: StrictInt | None = Field(default=None, ge=0)
    idempotency_key: str | None = Field(default=None, min_length=1)


class AuditExpectation(ScenarioContractModel):
    required_events: tuple[AuditEventType, ...] = ()
    forbidden_events: tuple[AuditEventType, ...] = ()
    required_reason_codes: tuple[ReasonCode, ...] = ()
    forbidden_raw_fields: tuple[str, ...] = ()
    forbidden_raw_values: tuple[str, ...] = ()


class ContextExpectation(ScenarioContractModel):
    required_values: tuple[str, ...] = ()
    forbidden_values: tuple[str, ...] = ()
    constraint_present: bool | None = None


class MemoryExpectation(ScenarioContractModel):
    required_canaries: tuple[str, ...] = ()
    forbidden_canaries: tuple[str, ...] = ()
    expected_memory_keys: tuple[str, ...] = ()
    expected_revision_count: StrictInt | None = Field(default=None, ge=0)


class EvidenceExpectation(ScenarioContractModel):
    authoritative_state: AuthoritativeStateExpectation = Field(
        default_factory=AuthoritativeStateExpectation
    )
    external_ledger: ExternalLedgerExpectation
    audit: AuditExpectation = Field(default_factory=AuditExpectation)
    model_context: ContextExpectation = Field(default_factory=ContextExpectation)
    memory: MemoryExpectation = Field(default_factory=MemoryExpectation)


class ExpectedOutcome(ScenarioContractModel):
    decision: ScenarioVerdict
    reason_code: ReasonCode | None = None
    evidence: EvidenceExpectation


class ScenarioSpec(ScenarioContractModel):
    schema_version: Literal[1]
    id: ScenarioIdentifier
    family_id: ScenarioIdentifier
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    profile: ScenarioProfile
    seed: StrictInt = Field(ge=0)
    clock: AwareDatetime
    principal: PrincipalSpec
    initial_authoritative_state: AuthoritativeStateSpec
    user_goal: str = Field(min_length=1)
    conversation: tuple[str, ...] = ()
    approved_effect: EffectSpec | None = None
    model_script: ModelScriptSpec
    fault_schedule: tuple[FaultScheduleSpec, ...] = ()
    mutation_expectation: MutationExpectation | None = None
    expected: ExpectedOutcome
    positive_control_id: ScenarioIdentifier
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_scenario(self) -> Self:
        if self.id == self.positive_control_id:
            raise ValueError("a scenario cannot be its own positive control")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("scenario tags must be unique")
        if self.mutation_expectation is not None:
            if self.approved_effect is None:
                raise ValueError("mutation expectation requires approved_effect")
            index = self.mutation_expectation.proposal_index
            if index >= len(self.model_script.proposals):
                raise ValueError("mutation proposal_index is outside model script")
            actual = changed_effect_fields(
                self.approved_effect,
                self.model_script.proposals[index],
            )
            if actual != self.mutation_expectation.changed_fields:
                raise ValueError(
                    "declared mutation fields do not match the proposal: "
                    f"declared={self.mutation_expectation.changed_fields}, "
                    f"actual={actual}"
                )
        return self


class ScenarioBoundary(StrEnum):
    AAR_CORE = "aar_core"
    APPLICATION_POLICY = "application_policy"
    EXTERNAL_CONTRACT = "external_contract"


class ScenarioRegistration(ScenarioContractModel):
    family_id: ScenarioIdentifier
    title: str = Field(min_length=1)
    stage: StrictInt = Field(ge=0)
    boundary: ScenarioBoundary
    variant_ids: tuple[ScenarioIdentifier, ...] = Field(min_length=1)
    positive_control_id: ScenarioIdentifier

    @model_validator(mode="after")
    def validate_variants(self) -> Self:
        _require_unique(self.variant_ids, "scenario variant ids")
        if self.positive_control_id in self.variant_ids:
            raise ValueError("positive control cannot also be a dangerous variant")
        return self


class ScenarioCatalogSpec(ScenarioContractModel):
    schema_version: Literal[1]
    suite_id: str = Field(min_length=1)
    registrations: tuple[ScenarioRegistration, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        _require_unique(
            (item.family_id for item in self.registrations),
            "scenario family ids",
        )
        all_case_ids: list[str] = []
        for item in self.registrations:
            all_case_ids.extend(item.variant_ids)
            all_case_ids.append(item.positive_control_id)
        _require_unique(all_case_ids, "catalog case ids")
        return self


def changed_effect_fields(reference: EffectSpec, candidate: EffectSpec) -> tuple[str, ...]:
    """Return stable field names whose validated values differ."""

    reference_values = reference.model_dump(mode="python")
    candidate_values = candidate.model_dump(mode="python")
    return tuple(
        field
        for field in EffectSpec.model_fields
        if reference_values[field] != candidate_values[field]
    )


def _require_unique(values: Any, label: str) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")
