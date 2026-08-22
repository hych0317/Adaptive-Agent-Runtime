"""Subprocess worker used to prove durable ecommerce recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from applications.ecommerce_support.composition import (
    EcommerceComposition,
    compose_ecommerce_support,
)
from applications.ecommerce_support.models import AuthenticatedPrincipal
from applications.ecommerce_support.policies import DomainPolicyError
from applications.governance_scenario_suite.contracts import (
    FaultPoint,
    ReasonCode,
    ReconciliationStatus,
    ScenarioProfile,
    ScenarioSpec,
    ScenarioVerdict,
)


def _load(path: Path) -> ScenarioSpec:
    return ScenarioSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _write_outcome(
    root: Path,
    decision: ScenarioVerdict,
    reason: ReasonCode | None,
) -> None:
    (root / "outcome.json").write_text(
        json.dumps(
            {
                "decision": decision.value,
                "reason_code": reason.value if reason is not None else None,
                "model_call_count": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _initialize(
    root: Path,
    scenario: ScenarioSpec,
) -> tuple[EcommerceComposition, str]:
    composition = compose_ecommerce_support(root, clock=lambda: scenario.clock)
    composition.store.seed(scenario.initial_authoritative_state)
    principal = AuthenticatedPrincipal(
        tenant_id=scenario.principal.tenant_id,
        user_id=scenario.principal.user_id,
        roles=scenario.principal.roles,
    )
    approval_id = f"approval-{scenario.id.lower()}"
    assert scenario.approved_effect is not None
    from datetime import timedelta

    composition.store.create_approval(
        approval_id=approval_id,
        principal=principal,
        effect=scenario.approved_effect,
        policy_version="ecommerce-policy-v1",
        created_at=scenario.clock,
        expires_at=scenario.clock + timedelta(minutes=10),
    )
    (root / "pre_state.json").write_text(
        composition.store.snapshot().model_dump_json(indent=2),
        encoding="utf-8",
    )
    (root / "recovery_state.json").write_text(
        json.dumps(
            {
                "approval_id": approval_id,
                "idempotency_key": scenario.approved_effect.idempotency_key,
                "model_call_count": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return composition, approval_id


def start(root: Path, scenario: ScenarioSpec, *, crash: bool) -> None:
    composition, approval_id = _initialize(root, scenario)
    assert scenario.approved_effect is not None
    principal = AuthenticatedPrincipal(
        tenant_id=scenario.principal.tenant_id,
        user_id=scenario.principal.user_id,
        roles=scenario.principal.roles,
    )

    def fault(point: FaultPoint) -> None:
        if not crash:
            return
        target = scenario.fault_schedule[0].point
        if point is target:
            os._exit(91 if point is FaultPoint.AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT else 92)

    try:
        composition.tools.request_refund(
            principal,
            scenario.approved_effect,
            approval_id=approval_id,
            fault_hook=fault,
        )
        _write_outcome(root, ScenarioVerdict.ALLOW, None)
    finally:
        composition.close()


def recover(root: Path, scenario: ScenarioSpec) -> None:
    composition = compose_ecommerce_support(root, clock=lambda: scenario.clock)
    assert scenario.approved_effect is not None
    key = scenario.approved_effect.idempotency_key
    assert key is not None
    fault = scenario.fault_schedule[0].point
    try:
        if scenario.profile in {
            ScenarioProfile.NO_RECONCILE_BLIND_RETRY,
            ScenarioProfile.PLAIN_AGENT,
        }:
            assert scenario.approved_effect.order_id is not None
            assert scenario.approved_effect.amount_cents is not None
            composition.gateway.refund(
                order_id=scenario.approved_effect.order_id,
                amount_cents=scenario.approved_effect.amount_cents,
                idempotency_key=f"{key}-blind-retry",
                now=scenario.clock,
            )
        if fault is FaultPoint.BEFORE_EXTERNAL_SEND and scenario.id.startswith("R2B"):
            composition.gateway.set_reconciliation_override(
                key,
                ReconciliationStatus.UNKNOWN,
            )
        retry = scenario.profile is not ScenarioProfile.NO_RECONCILE_FAIL_CLOSED
        try:
            composition.tools.reconcile_refund(key, retry_if_not_committed=retry)
        except DomainPolicyError as exc:
            _write_outcome(root, ScenarioVerdict.FAIL_CLOSED, exc.reason_code)
            return
        reason = (
            ReasonCode.RECONCILIATION_COMMITTED
            if fault is FaultPoint.AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT
            else ReasonCode.RECONCILIATION_NOT_COMMITTED
        )
        _write_outcome(root, ScenarioVerdict.ALLOW, reason)
    finally:
        composition.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--action", choices=("normal", "crash", "recover"), required=True)
    args = parser.parse_args()
    scenario = _load(args.scenario)
    if args.action == "recover":
        recover(args.root, scenario)
    else:
        start(args.root, scenario, crash=args.action == "crash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
