"""Independent, durable fake payment gateway with explicit idempotency."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import UUID, uuid5

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.ecommerce_support.contracts import FaultHook
from applications.ecommerce_support.models import (
    ExternalLedgerRecord,
    GatewayReconciliation,
    GatewayRefundResult,
)
from applications.ecommerce_support.policies import DomainPolicyError
from applications.governance_scenario_suite.contracts import (
    FaultPoint,
    ReasonCode,
    ReconciliationStatus,
)


_GATEWAY_NAMESPACE = UUID("7c5f4740-c927-46de-a3c9-f7947d8dc897")


class FakePaymentGateway:
    """A separate SQLite service boundary used as external source of truth."""

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
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> FakePaymentGateway:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def refund(
        self,
        *,
        order_id: str,
        amount_cents: int,
        idempotency_key: str,
        now: datetime,
        fault_hook: FaultHook | None = None,
    ) -> GatewayRefundResult:
        if fault_hook is not None:
            fault_hook(FaultPoint.BEFORE_EXTERNAL_SEND)
        request_fingerprint = decision_fingerprint(
            {
                "operation": "refund",
                "order_id": order_id,
                "amount_cents": amount_cents,
            }
        )
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM external_refunds WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row["request_fingerprint"] != request_fingerprint:
                    raise DomainPolicyError(
                        ReasonCode.IDEMPOTENCY_KEY_CONFLICT,
                        "The idempotency key is already bound to another request.",
                    )
                connection.execute(
                    """
                    UPDATE external_refunds
                    SET attempt_count = attempt_count + 1
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                )
                result = _result_from_row(row)
            else:
                result = GatewayRefundResult(
                    external_ref=str(uuid5(_GATEWAY_NAMESPACE, idempotency_key)),
                    idempotency_key=idempotency_key,
                    order_id=order_id,
                    amount_cents=amount_cents,
                    committed_at=now,
                )
                connection.execute(
                    """
                    INSERT INTO external_refunds(
                        idempotency_key, request_fingerprint, external_ref,
                        order_id, amount_cents, committed_at,
                        attempt_count, effect_count
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, 1)
                    """,
                    (
                        idempotency_key,
                        request_fingerprint,
                        result.external_ref,
                        order_id,
                        amount_cents,
                        now.isoformat(),
                    ),
                )
        if fault_hook is not None:
            fault_hook(FaultPoint.AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT)
        return result

    def reconcile(
        self,
        idempotency_key: str,
        *,
        fault_hook: FaultHook | None = None,
    ) -> GatewayReconciliation:
        if fault_hook is not None:
            fault_hook(FaultPoint.DURING_RECONCILIATION_QUERY)
        override = self._connection.execute(
            "SELECT status FROM reconciliation_overrides WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if override is not None:
            status = ReconciliationStatus(override["status"])
            if status is ReconciliationStatus.UNKNOWN:
                return GatewayReconciliation(
                    status=status,
                    reason="gateway reconciliation is unavailable or inconsistent",
                )
            if status is ReconciliationStatus.NOT_COMMITTED:
                return GatewayReconciliation(
                    status=status,
                    reason="gateway has no committed effect for this key",
                )
        row = self._connection.execute(
            "SELECT * FROM external_refunds WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return GatewayReconciliation(
                status=ReconciliationStatus.NOT_COMMITTED,
                reason="gateway has no committed effect for this key",
            )
        return GatewayReconciliation(
            status=ReconciliationStatus.COMMITTED,
            result=_result_from_row(row),
            reason="gateway ledger contains the committed effect",
        )

    def set_reconciliation_override(
        self,
        idempotency_key: str,
        status: ReconciliationStatus,
    ) -> None:
        if status is ReconciliationStatus.COMMITTED:
            row = self._connection.execute(
                "SELECT 1 FROM external_refunds WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is None:
                raise ValueError("COMMITTED override requires a ledger result")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO reconciliation_overrides(idempotency_key, status)
                VALUES (?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET status = excluded.status
                """,
                (idempotency_key, status.value),
            )

    def ledger(self) -> tuple[ExternalLedgerRecord, ...]:
        return tuple(
            ExternalLedgerRecord(
                idempotency_key=row["idempotency_key"],
                request_fingerprint=row["request_fingerprint"],
                result=_result_from_row(row),
                attempt_count=row["attempt_count"],
                effect_count=row["effect_count"],
            )
            for row in self._connection.execute(
                "SELECT * FROM external_refunds ORDER BY idempotency_key"
            )
        )

    def total_effect_count(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(SUM(effect_count), 0) AS total FROM external_refunds"
        ).fetchone()
        assert row is not None
        return int(row["total"])

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
            CREATE TABLE IF NOT EXISTS external_refunds(
                idempotency_key TEXT PRIMARY KEY,
                request_fingerprint TEXT NOT NULL,
                external_ref TEXT NOT NULL UNIQUE,
                order_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0),
                committed_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                effect_count INTEGER NOT NULL CHECK(effect_count = 1)
            );
            CREATE TABLE IF NOT EXISTS reconciliation_overrides(
                idempotency_key TEXT PRIMARY KEY,
                status TEXT NOT NULL
            );
            """
        )


def _result_from_row(row: sqlite3.Row) -> GatewayRefundResult:
    return GatewayRefundResult(
        external_ref=row["external_ref"],
        idempotency_key=row["idempotency_key"],
        order_id=row["order_id"],
        amount_cents=row["amount_cents"],
        committed_at=datetime.fromisoformat(row["committed_at"]),
    )
