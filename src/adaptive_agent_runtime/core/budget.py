"""Aggregate, run-scoped inference usage accounting."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import asyncio
from collections.abc import Iterator
from uuid import UUID, uuid4

from adaptive_agent_runtime.core.errors import (
    RunBudgetExhaustedError,
    RunUsageAccountingError,
)
from adaptive_agent_runtime.core.models import RunStopPolicy, RunUsage


@dataclass(frozen=True)
class RunBudgetReservation:
    request_id: UUID
    reserved_tokens: int
    reserved_cost: float
    currency: str | None


class RunBudgetLedger:
    """Serialize reservations and accumulate authoritative usage for one Run."""

    def __init__(
        self,
        *,
        run_id: UUID,
        policy: RunStopPolicy,
        initial_usage: RunUsage | None = None,
    ) -> None:
        self.run_id = run_id
        self.policy = policy
        self._usage = initial_usage or RunUsage(currency=policy.currency)
        if (
            policy.currency is not None
            and self._usage.currency not in {None, policy.currency}
        ):
            raise ValueError("persisted Run usage has another currency")
        self._reserved_tokens = 0
        self._reserved_cost = 0.0
        self._reservations: dict[UUID, RunBudgetReservation] = {}
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        request_id: UUID,
        *,
        requested_max_tokens: int | None,
        requested_max_cost: float | None,
        currency: str | None,
    ) -> RunBudgetReservation:
        async with self._lock:
            if request_id in self._reservations:
                return self._reservations[request_id]
            token_reservation = 0
            if self.policy.max_total_tokens is not None:
                remaining_tokens = (
                    self.policy.max_total_tokens
                    - self._usage.total_tokens
                    - self._reserved_tokens
                )
                if remaining_tokens <= 0:
                    raise RunBudgetExhaustedError(
                        "tokens",
                        "Run token budget is exhausted",
                    )
                token_reservation = min(
                    remaining_tokens,
                    (
                        requested_max_tokens
                        if requested_max_tokens is not None
                        else remaining_tokens
                    ),
                )
            cost_reservation = 0.0
            if self.policy.max_monetary_cost is not None:
                if currency not in {None, self.policy.currency}:
                    raise RunUsageAccountingError(
                        "inference currency does not match the Run budget"
                    )
                remaining_cost = (
                    self.policy.max_monetary_cost
                    - self._usage.monetary_cost
                    - self._reserved_cost
                )
                if remaining_cost <= 0.0:
                    raise RunBudgetExhaustedError(
                        "cost",
                        "Run cost budget is exhausted",
                    )
                cost_reservation = min(
                    remaining_cost,
                    (
                        requested_max_cost
                        if requested_max_cost is not None
                        else remaining_cost
                    ),
                )
            reservation = RunBudgetReservation(
                request_id=request_id,
                reserved_tokens=token_reservation,
                reserved_cost=cost_reservation,
                currency=self.policy.currency,
            )
            self._reservations[request_id] = reservation
            self._reserved_tokens += token_reservation
            self._reserved_cost += cost_reservation
            return reservation

    async def commit(
        self,
        reservation: RunBudgetReservation,
        *,
        total_tokens: int | None,
        monetary_cost: float | None,
        currency: str | None,
    ) -> RunUsage:
        async with self._lock:
            if self.policy.max_total_tokens is not None and total_tokens is None:
                raise RunUsageAccountingError(
                    "token usage is missing for a budgeted Run"
                )
            if self.policy.max_monetary_cost is not None:
                if monetary_cost is None or currency is None:
                    raise RunUsageAccountingError(
                        "monetary usage is missing for a budgeted Run"
                    )
                if currency != self.policy.currency:
                    raise RunUsageAccountingError(
                        "inference currency does not match the Run budget"
                    )
            if (
                self.policy.max_total_tokens is not None
                and total_tokens is not None
                and total_tokens > reservation.reserved_tokens
            ):
                self._consume_reserved_locked(reservation)
                raise RunBudgetExhaustedError(
                    "tokens",
                    "reported token usage exceeded the reserved Run budget",
                )
            if (
                self.policy.max_monetary_cost is not None
                and monetary_cost is not None
                and monetary_cost > reservation.reserved_cost
            ):
                self._consume_reserved_locked(reservation)
                raise RunBudgetExhaustedError(
                    "cost",
                    "reported monetary usage exceeded the reserved Run budget",
                )
            current = self._pop_reservation(reservation)
            next_currency = self._usage.currency or currency or self.policy.currency
            if (
                monetary_cost is not None
                and monetary_cost > 0.0
                and next_currency is None
            ):
                raise RunUsageAccountingError("monetary usage requires a currency")
            self._usage = RunUsage(
                total_tokens=self._usage.total_tokens + (total_tokens or 0),
                monetary_cost=(
                    self._usage.monetary_cost + (monetary_cost or 0.0)
                ),
                currency=next_currency,
            )
            del current
            return self._usage

    async def consume_reservation(
        self,
        reservation: RunBudgetReservation,
    ) -> RunUsage:
        """Fail closed when a response exceeded or omitted budgeted usage."""

        return await self.commit(
            reservation,
            total_tokens=(
                reservation.reserved_tokens
                if self.policy.max_total_tokens is not None
                else 0
            ),
            monetary_cost=(
                reservation.reserved_cost
                if self.policy.max_monetary_cost is not None
                else 0.0
            ),
            currency=self.policy.currency,
        )

    async def release(self, reservation: RunBudgetReservation) -> None:
        async with self._lock:
            if reservation.request_id not in self._reservations:
                return
            self._pop_reservation(reservation)

    async def record_usage(
        self,
        *,
        total_tokens: int | None,
        monetary_cost: float | None,
        currency: str | None,
    ) -> RunUsage:
        reservation = await self.reserve(
            uuid4(),
            requested_max_tokens=total_tokens,
            requested_max_cost=monetary_cost,
            currency=currency,
        )
        return await self.commit(
            reservation,
            total_tokens=total_tokens,
            monetary_cost=monetary_cost,
            currency=currency,
        )

    async def snapshot(self) -> RunUsage:
        async with self._lock:
            return self._usage

    def _pop_reservation(
        self,
        reservation: RunBudgetReservation,
    ) -> RunBudgetReservation:
        current = self._reservations.pop(reservation.request_id, None)
        if current != reservation:
            raise RunUsageAccountingError("Run budget reservation is not active")
        self._reserved_tokens -= reservation.reserved_tokens
        self._reserved_cost -= reservation.reserved_cost
        return current

    def _consume_reserved_locked(
        self,
        reservation: RunBudgetReservation,
    ) -> None:
        """Charge the full reservation when reported usage is out of bounds."""

        self._pop_reservation(reservation)
        self._usage = RunUsage(
            total_tokens=(
                self._usage.total_tokens + reservation.reserved_tokens
            ),
            monetary_cost=(
                self._usage.monetary_cost + reservation.reserved_cost
            ),
            currency=self._usage.currency or reservation.currency,
        )


_CURRENT_LEDGER: ContextVar[RunBudgetLedger | None] = ContextVar(
    "adaptive_agent_runtime_run_budget_ledger",
    default=None,
)


@contextmanager
def bind_run_budget_ledger(ledger: RunBudgetLedger) -> Iterator[None]:
    token: Token[RunBudgetLedger | None] = _CURRENT_LEDGER.set(ledger)
    try:
        yield
    finally:
        _CURRENT_LEDGER.reset(token)


def current_run_budget_ledger() -> RunBudgetLedger | None:
    return _CURRENT_LEDGER.get()
