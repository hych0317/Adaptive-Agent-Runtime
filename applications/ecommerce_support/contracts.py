"""Narrow ports used by the ecommerce scenario application."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from applications.ecommerce_support.models import (
    AuthenticatedPrincipal,
    DomainSnapshot,
    GatewayReconciliation,
    GatewayRefundResult,
    OrderRecord,
)
from applications.governance_scenario_suite.contracts import FaultPoint


FaultHook = Callable[[FaultPoint], None]


class PaymentGateway(Protocol):
    def refund(
        self,
        *,
        order_id: str,
        amount_cents: int,
        idempotency_key: str,
        now: datetime,
        fault_hook: FaultHook | None = None,
    ) -> GatewayRefundResult: ...

    def reconcile(
        self,
        idempotency_key: str,
        *,
        fault_hook: FaultHook | None = None,
    ) -> GatewayReconciliation: ...


class EcommerceReadStore(Protocol):
    def get_order(
        self,
        principal: AuthenticatedPrincipal,
        order_id: str,
        *,
        now: datetime,
    ) -> OrderRecord: ...

    def snapshot(self) -> DomainSnapshot: ...
