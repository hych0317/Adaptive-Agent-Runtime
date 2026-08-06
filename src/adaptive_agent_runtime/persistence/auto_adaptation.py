"""Durable single-trigger ledger for controlled Runtime auto-adaptation."""

from __future__ import annotations

import sqlite3
from uuid import UUID

from adaptive_agent_runtime.optimization import (
    AutoAdaptationSkipReason,
    AutoAdaptationStatus,
    AutoAdaptationTriggerRecord,
)
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class SQLiteAutoAdaptationTriggerStore:
    module_id = "optimization.auto_adaptation_trigger.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def load_for_run(
        self,
        trigger_run_id: UUID,
    ) -> AutoAdaptationTriggerRecord | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT trigger_json FROM auto_adaptation_triggers "
                "WHERE trigger_run_id = ?",
                (str(trigger_run_id),),
            ).fetchone()
        return _record_from_row(row)

    async def claim(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        """Atomically claim a Run and, when selected, a Proposal."""

        if not record.outcome_persisted:
            raise PersistenceConflictError(
                "A non-persistent adaptation outcome cannot enter the trigger ledger"
            )

        with self._database.transaction() as cursor:
            existing = _load_for_run(cursor, record.trigger_run_id)
            if existing is not None:
                return existing
            claimed = record
            if record.selected_proposal_id is not None:
                owner = cursor.execute(
                    "SELECT trigger_run_id FROM auto_adaptation_triggers "
                    "WHERE selected_proposal_id = ?",
                    (str(record.selected_proposal_id),),
                ).fetchone()
                if owner is not None:
                    claimed = record.model_copy(
                        update={
                            "status": AutoAdaptationStatus.SKIPPED,
                            "selected_proposal_id": None,
                            "trigger_identity": None,
                            "apply_request_id": None,
                            "skip_reason": (
                                AutoAdaptationSkipReason.PROPOSAL_ALREADY_APPLIED
                            ),
                            "candidate_rejection_reasons": (
                                "proposal already claimed by another completed run",
                            ),
                        }
                    )
            try:
                _insert_current(cursor, claimed)
                _insert_history(cursor, claimed)
            except sqlite3.IntegrityError as exc:
                # A concurrent connection may have won after the check.  The
                # transaction is rolled back by SQLiteDatabase; callers retry a
                # read rather than selecting a different Proposal.
                raise PersistenceConflictError(
                    "Auto-adaptation trigger claim conflicted"
                ) from exc
        return claimed

    async def transition(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        if not record.outcome_persisted:
            raise PersistenceConflictError(
                "A non-persistent adaptation outcome cannot enter the trigger ledger"
            )
        with self._database.transaction() as cursor:
            current = _load_for_run(cursor, record.trigger_run_id)
            if current is None:
                raise PersistenceConflictError(
                    "Auto-adaptation trigger transition has no durable claim"
                )
            if current == record:
                return current
            if current.status not in {
                AutoAdaptationStatus.SELECTED,
                AutoAdaptationStatus.REVIEW_PENDING,
            }:
                # Terminal outcomes never reopen or trigger another Apply.
                return current
            if (
                current.status is AutoAdaptationStatus.REVIEW_PENDING
                and record.status is AutoAdaptationStatus.SELECTED
            ):
                raise PersistenceConflictError(
                    "Human Review trigger cannot return to selected"
                )
            if record.revision != current.revision + 1:
                raise PersistenceConflictError(
                    "Auto-adaptation trigger revision is stale"
                )
            immutable = (
                "attempt_id",
                "trigger_run_id",
                "policy_fingerprint",
                "baseline_revision",
                "baseline_fingerprint",
                "candidate_set_fingerprint",
                "selected_proposal_id",
                "trigger_identity",
                "apply_request_id",
                "created_at",
            )
            if any(getattr(current, name) != getattr(record, name) for name in immutable):
                raise PersistenceConflictError(
                    "Auto-adaptation transition changed its trigger authority"
                )
            updated = cursor.execute(
                "UPDATE auto_adaptation_triggers SET revision = ?, status = ?, "
                "trigger_json = ? WHERE trigger_run_id = ? AND revision = ?",
                (
                    record.revision,
                    record.status.value,
                    record.model_dump_json(),
                    str(record.trigger_run_id),
                    current.revision,
                ),
            ).rowcount
            if updated != 1:
                raise PersistenceConflictError(
                    "Auto-adaptation trigger transition CAS failed"
                )
            _insert_history(cursor, record)
        return record


def _record_from_row(row: object | None) -> AutoAdaptationTriggerRecord | None:
    if row is None:
        return None
    return AutoAdaptationTriggerRecord.model_validate_json(
        row["trigger_json"]  # type: ignore[index]
    )


def _load_for_run(
    cursor: object,
    trigger_run_id: UUID,
) -> AutoAdaptationTriggerRecord | None:
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT trigger_json FROM auto_adaptation_triggers WHERE trigger_run_id = ?",
        (str(trigger_run_id),),
    ).fetchone()
    return _record_from_row(row)


def _insert_current(cursor: object, record: AutoAdaptationTriggerRecord) -> None:
    cursor.execute(  # type: ignore[attr-defined]
        "INSERT INTO auto_adaptation_triggers "
        "(trigger_run_id, attempt_id, revision, status, policy_fingerprint, "
        "selected_proposal_id, trigger_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(record.trigger_run_id),
            str(record.attempt_id),
            record.revision,
            record.status.value,
            record.policy_fingerprint,
            (
                str(record.selected_proposal_id)
                if record.selected_proposal_id is not None
                else None
            ),
            record.model_dump_json(),
        ),
    )


def _insert_history(cursor: object, record: AutoAdaptationTriggerRecord) -> None:
    cursor.execute(  # type: ignore[attr-defined]
        "INSERT INTO auto_adaptation_trigger_history "
        "(trigger_run_id, revision, status, trigger_json) VALUES (?, ?, ?, ?)",
        (
            str(record.trigger_run_id),
            record.revision,
            record.status.value,
            record.model_dump_json(),
        ),
    )
