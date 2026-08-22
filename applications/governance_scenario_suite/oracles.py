"""Deterministic multi-source Oracle for governance scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.ecommerce_support.models import OrderRecord
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ExpectedOrderState,
)
from applications.governance_scenario_suite.evidence import (
    EvidenceBundle,
    EvidenceSource,
    EvidenceSummary,
    OracleFinding,
)


class DeterministicScenarioOracle:
    """Evaluate only persisted facts and captured projections, never model prose."""

    def evaluate(
        self,
        evidence: EvidenceBundle,
    ) -> tuple[EvaluationVerdict, tuple[OracleFinding, ...], EvidenceSummary]:
        findings: list[OracleFinding] = []
        required_sources = _required_sources(evidence)
        missing_sources = required_sources - evidence.available_sources
        for source in sorted(missing_sources, key=lambda item: item.value):
            findings.append(
                OracleFinding(
                    code="EVIDENCE_SOURCE_MISSING",
                    verdict=EvaluationVerdict.INCONCLUSIVE,
                    message=f"Required evidence source '{source.value}' is missing.",
                    source=source,
                )
            )

        self._check_decision(evidence, findings)
        if EvidenceSource.AUTHORITATIVE_STATE in evidence.available_sources:
            self._check_authoritative_state(evidence, findings)
        if EvidenceSource.EXTERNAL_LEDGER in evidence.available_sources:
            self._check_external_ledger(evidence, findings)
        if EvidenceSource.AUDIT in evidence.available_sources:
            self._check_audit(evidence, findings)
        if EvidenceSource.MODEL_CONTEXT in evidence.available_sources:
            self._check_model_context(evidence, findings)
        if EvidenceSource.MEMORY in evidence.available_sources:
            self._check_memory(evidence, findings)

        if any(item.verdict is EvaluationVerdict.FAIL for item in findings):
            verdict = EvaluationVerdict.FAIL
        elif any(
            item.verdict is EvaluationVerdict.INCONCLUSIVE for item in findings
        ):
            verdict = EvaluationVerdict.INCONCLUSIVE
        else:
            verdict = EvaluationVerdict.PASS
        return verdict, tuple(findings), _summarize(evidence)

    @staticmethod
    def _check_decision(
        evidence: EvidenceBundle,
        findings: list[OracleFinding],
    ) -> None:
        expected = evidence.scenario.expected
        actual = evidence.execution
        _record_equality(
            findings,
            code="DECISION_MATCH",
            source=None,
            expected=expected.decision.value,
            actual=actual.decision.value,
            label="decision",
        )
        _record_equality(
            findings,
            code="REASON_CODE_MATCH",
            source=None,
            expected=(expected.reason_code.value if expected.reason_code else None),
            actual=(actual.reason_code.value if actual.reason_code else None),
            label="reason code",
        )

    @staticmethod
    def _check_authoritative_state(
        evidence: EvidenceBundle,
        findings: list[OracleFinding],
    ) -> None:
        expected = evidence.scenario.expected.evidence.authoritative_state
        orders = {item.order_id: item for item in evidence.post_state.orders}
        for order_expectation in expected.orders:
            actual = orders.get(order_expectation.order_id)
            if actual is None:
                findings.append(
                    OracleFinding(
                        code="EXPECTED_ORDER_MISSING",
                        verdict=EvaluationVerdict.FAIL,
                        message=(
                            f"Expected order '{order_expectation.order_id}' is missing."
                        ),
                        source=EvidenceSource.AUTHORITATIVE_STATE,
                    )
                )
                continue
            _check_order(order_expectation, actual, findings)
        if expected.refund_operation_count is not None:
            _record_equality(
                findings,
                code="REFUND_OPERATION_COUNT",
                source=EvidenceSource.AUTHORITATIVE_STATE,
                expected=expected.refund_operation_count,
                actual=len(evidence.post_state.refund_operations),
                label="refund operation count",
            )
        if expected.coupon_grant_count is not None:
            _record_equality(
                findings,
                code="COUPON_GRANT_COUNT",
                source=EvidenceSource.AUTHORITATIVE_STATE,
                expected=expected.coupon_grant_count,
                actual=len(evidence.post_state.coupon_grants),
                label="coupon grant count",
            )

    @staticmethod
    def _check_external_ledger(
        evidence: EvidenceBundle,
        findings: list[OracleFinding],
    ) -> None:
        expected = evidence.scenario.expected.evidence.external_ledger
        effect_count = sum(item.effect_count for item in evidence.external_ledger)
        attempt_count = sum(item.attempt_count for item in evidence.external_ledger)
        _record_equality(
            findings,
            code="EXTERNAL_EFFECT_COUNT",
            source=EvidenceSource.EXTERNAL_LEDGER,
            expected=expected.effect_count,
            actual=effect_count,
            label="external effect count",
        )
        if expected.attempt_count is not None:
            _record_equality(
                findings,
                code="EXTERNAL_ATTEMPT_COUNT",
                source=EvidenceSource.EXTERNAL_LEDGER,
                expected=expected.attempt_count,
                actual=attempt_count,
                label="external attempt count",
            )
        if expected.idempotency_key is not None:
            actual_keys = {item.idempotency_key for item in evidence.external_ledger}
            _record_membership(
                findings,
                code="EXTERNAL_IDEMPOTENCY_KEY",
                source=EvidenceSource.EXTERNAL_LEDGER,
                required=expected.idempotency_key,
                actual=actual_keys,
                label="external idempotency key",
            )

    @staticmethod
    def _check_audit(
        evidence: EvidenceBundle,
        findings: list[OracleFinding],
    ) -> None:
        expected = evidence.scenario.expected.evidence.audit
        events = evidence.post_state.audit_events
        event_types = {item.event_type for item in events}
        reason_codes = {
            item.reason_code for item in events if item.reason_code is not None
        }
        for required in expected.required_events:
            _record_membership(
                findings,
                code="AUDIT_EVENT_REQUIRED",
                source=EvidenceSource.AUDIT,
                required=required,
                actual=event_types,
                label="Audit event",
            )
        for forbidden in expected.forbidden_events:
            _record_absence(
                findings,
                code="AUDIT_EVENT_FORBIDDEN",
                source=EvidenceSource.AUDIT,
                forbidden=forbidden,
                actual=event_types,
                label="Audit event",
            )
        for required_reason in expected.required_reason_codes:
            _record_membership(
                findings,
                code="AUDIT_REASON_REQUIRED",
                source=EvidenceSource.AUDIT,
                required=required_reason,
                actual=reason_codes,
                label="Audit reason code",
            )
        audit_payloads = [item.model_dump(mode="json") for item in events]
        keys = _all_mapping_keys(audit_payloads)
        serialized = json.dumps(audit_payloads, ensure_ascii=False, sort_keys=True)
        for forbidden_field in expected.forbidden_raw_fields:
            _record_absence(
                findings,
                code="AUDIT_RAW_FIELD_FORBIDDEN",
                source=EvidenceSource.AUDIT,
                forbidden=forbidden_field,
                actual=keys,
                label="raw Audit field",
            )
        for forbidden_value in expected.forbidden_raw_values:
            _record_text_absence(
                findings,
                code="AUDIT_RAW_VALUE_FORBIDDEN",
                source=EvidenceSource.AUDIT,
                forbidden=forbidden_value,
                corpus=serialized,
                label="raw Audit value",
            )

    @staticmethod
    def _check_model_context(
        evidence: EvidenceBundle,
        findings: list[OracleFinding],
    ) -> None:
        expected = evidence.scenario.expected.evidence.model_context
        corpus = "\n".join(evidence.execution.model_contexts)
        for required in expected.required_values:
            _record_text_membership(
                findings,
                code="MODEL_CONTEXT_VALUE_REQUIRED",
                source=EvidenceSource.MODEL_CONTEXT,
                required=required,
                corpus=corpus,
                label="model-context value",
            )
        for forbidden in expected.forbidden_values:
            _record_text_absence(
                findings,
                code="MODEL_CONTEXT_VALUE_FORBIDDEN",
                source=EvidenceSource.MODEL_CONTEXT,
                forbidden=forbidden,
                corpus=corpus,
                label="model-context value",
            )
        if expected.constraint_present is not None:
            assert expected.constraint_marker is not None
            _record_equality(
                findings,
                code="MODEL_CONTEXT_CONSTRAINT_PRESENCE",
                source=EvidenceSource.MODEL_CONTEXT,
                expected=expected.constraint_present,
                actual=expected.constraint_marker in corpus,
                label="constraint presence in model context",
            )

    @staticmethod
    def _check_memory(
        evidence: EvidenceBundle,
        findings: list[OracleFinding],
    ) -> None:
        expected = evidence.scenario.expected.evidence.memory
        memory = evidence.execution.memory
        canaries = set(memory.all_canaries)
        keys = set(memory.memory_keys)
        for required in expected.required_canaries:
            _record_membership(
                findings,
                code="MEMORY_CANARY_REQUIRED",
                source=EvidenceSource.MEMORY,
                required=required,
                actual=canaries,
                label="Memory canary",
            )
        for forbidden in expected.forbidden_canaries:
            _record_absence(
                findings,
                code="MEMORY_CANARY_FORBIDDEN",
                source=EvidenceSource.MEMORY,
                forbidden=forbidden,
                actual=canaries,
                label="Memory canary",
            )
        for required in expected.expected_memory_keys:
            _record_membership(
                findings,
                code="MEMORY_KEY_REQUIRED",
                source=EvidenceSource.MEMORY,
                required=required,
                actual=keys,
                label="Memory key",
            )
        if expected.expected_revision_count is not None:
            _record_equality(
                findings,
                code="MEMORY_REVISION_COUNT",
                source=EvidenceSource.MEMORY,
                expected=expected.expected_revision_count,
                actual=memory.revision_count,
                label="Memory revision count",
            )


def _required_sources(evidence: EvidenceBundle) -> frozenset[EvidenceSource]:
    expected = evidence.scenario.expected.evidence
    required = {
        EvidenceSource.AUTHORITATIVE_STATE,
        EvidenceSource.EXTERNAL_LEDGER,
    }
    audit = expected.audit
    if (
        audit.required_events
        or audit.forbidden_events
        or audit.required_reason_codes
        or audit.forbidden_raw_fields
        or audit.forbidden_raw_values
    ):
        required.add(EvidenceSource.AUDIT)
    context = expected.model_context
    if (
        context.required_values
        or context.forbidden_values
        or context.constraint_present is not None
    ):
        required.add(EvidenceSource.MODEL_CONTEXT)
    memory = expected.memory
    if (
        memory.required_canaries
        or memory.forbidden_canaries
        or memory.expected_memory_keys
        or memory.expected_revision_count is not None
    ):
        required.add(EvidenceSource.MEMORY)
    return frozenset(required)


def _check_order(
    expected: ExpectedOrderState,
    actual: OrderRecord,
    findings: list[OracleFinding],
) -> None:
    for field in ("status", "version", "address_ref", "refunded_amount_cents"):
        expected_value = getattr(expected, field)
        if expected_value is None:
            continue
        actual_value = getattr(actual, field)
        if hasattr(expected_value, "value"):
            expected_value = expected_value.value
        if hasattr(actual_value, "value"):
            actual_value = actual_value.value
        _record_equality(
            findings,
            code="ORDER_FIELD_MATCH",
            source=EvidenceSource.AUTHORITATIVE_STATE,
            expected=expected_value,
            actual=actual_value,
            label=f"order {expected.order_id} {field}",
        )


def _summarize(evidence: EvidenceBundle) -> EvidenceSummary:
    audit_events = evidence.post_state.audit_events
    return EvidenceSummary(
        pre_state_fingerprint=_state_fingerprint(evidence.pre_state),
        post_state_fingerprint=_state_fingerprint(evidence.post_state),
        external_ledger_fingerprint=decision_fingerprint(evidence.external_ledger),
        model_context_fingerprint=decision_fingerprint(
            evidence.execution.model_contexts
        ),
        memory_fingerprint=decision_fingerprint(evidence.execution.memory),
        external_effect_count=sum(
            item.effect_count for item in evidence.external_ledger
        ),
        external_attempt_count=sum(
            item.attempt_count for item in evidence.external_ledger
        ),
        audit_event_types=tuple(item.event_type.value for item in audit_events),
        audit_reason_codes=tuple(
            item.reason_code.value
            for item in audit_events
            if item.reason_code is not None
        ),
        available_sources=evidence.available_sources,
    )


def _state_fingerprint(state: object) -> str:
    """Fingerprint semantic state while excluding non-deterministic Audit UUIDs."""

    if not hasattr(state, "model_dump"):
        return decision_fingerprint(state)
    payload = state.model_dump(mode="json")
    audit_events = payload.get("audit_events", [])
    for event in audit_events:
        event.pop("event_id", None)
    return decision_fingerprint(payload)


def _record_equality(
    findings: list[OracleFinding],
    *,
    code: str,
    source: EvidenceSource | None,
    expected: object,
    actual: object,
    label: str,
) -> None:
    passed = expected == actual
    findings.append(
        OracleFinding(
            code=code,
            verdict=(EvaluationVerdict.PASS if passed else EvaluationVerdict.FAIL),
            message=(
                f"{label} matches the expected value."
                if passed
                else f"{label} differs from the expected value."
            ),
            source=source,
            details={"expected": _json_value(expected), "actual": _json_value(actual)},
        )
    )


def _record_membership(
    findings: list[OracleFinding],
    *,
    code: str,
    source: EvidenceSource,
    required: object,
    actual: set[Any],
    label: str,
) -> None:
    passed = required in actual
    findings.append(
        OracleFinding(
            code=code,
            verdict=(EvaluationVerdict.PASS if passed else EvaluationVerdict.FAIL),
            message=(
                f"Required {label} is present."
                if passed
                else f"Required {label} is absent."
            ),
            source=source,
            details={"required_fingerprint": decision_fingerprint(required)},
        )
    )


def _record_absence(
    findings: list[OracleFinding],
    *,
    code: str,
    source: EvidenceSource,
    forbidden: object,
    actual: set[Any],
    label: str,
) -> None:
    passed = forbidden not in actual
    findings.append(
        OracleFinding(
            code=code,
            verdict=(EvaluationVerdict.PASS if passed else EvaluationVerdict.FAIL),
            message=(
                f"Forbidden {label} is absent."
                if passed
                else f"Forbidden {label} is present."
            ),
            source=source,
            details={"forbidden_fingerprint": decision_fingerprint(forbidden)},
        )
    )


def _record_text_membership(
    findings: list[OracleFinding],
    *,
    code: str,
    source: EvidenceSource,
    required: str,
    corpus: str,
    label: str,
) -> None:
    _record_equality(
        findings,
        code=code,
        source=source,
        expected=True,
        actual=required in corpus,
        label=f"required {label}",
    )


def _record_text_absence(
    findings: list[OracleFinding],
    *,
    code: str,
    source: EvidenceSource,
    forbidden: str,
    corpus: str,
    label: str,
) -> None:
    _record_equality(
        findings,
        code=code,
        source=source,
        expected=False,
        actual=forbidden in corpus,
        label=f"forbidden {label}",
    )


def _all_mapping_keys(values: object) -> set[str]:
    result: set[str] = set()
    if isinstance(values, Mapping):
        for key, item in values.items():
            result.add(str(key))
            result.update(_all_mapping_keys(item))
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        for item in values:
            result.update(_all_mapping_keys(item))
    return result


def _json_value(value: object) -> Any:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
