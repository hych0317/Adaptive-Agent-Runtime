"""Explicit, fail-closed cleanup for disposable Runtime runs."""

from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Row
from typing import Sequence
from uuid import UUID

from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


@dataclass(frozen=True)
class PurgeRunReport:
    run_id: UUID
    run_kind: str
    decision_count: int
    dry_run: bool
    deleted: bool


class SQLiteDisposableRunCleaner:
    """Delete only an explicitly disposable, structurally unreferenced run."""

    module_id = "runtime.retention.disposable.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def purge_run(
        self,
        run_id: UUID,
        *,
        require_disposable: bool = True,
        dry_run: bool = True,
    ) -> PurgeRunReport:
        run = str(run_id)
        with self._database.transaction() as cursor:
            metadata = cursor.execute(
                "SELECT run_kind, disposable FROM runtime_run_metadata "
                "WHERE run_id = ?",
                (run,),
            ).fetchone()
            if metadata is None:
                raise PersistenceConflictError("Run has no authoritative retention metadata")
            if require_disposable and int(metadata["disposable"]) != 1:
                raise PersistenceConflictError("Run is not explicitly disposable")

            decisions = cursor.execute(
                "SELECT request_id, proposal_ref, effect_ref, commit_receipt_ref, "
                "stage, reconciliation_status FROM decision_current WHERE run_id = ?",
                (run,),
            ).fetchall()
            unsafe = {
                "review_pending",
                "applying",
                "effect_committed",
            }
            if any(
                row["stage"] in unsafe or row["reconciliation_status"] == "unknown"
                for row in decisions
            ):
                raise PersistenceConflictError(
                    "Disposable Run has a non-purgeable Decision state"
                )
            request_ids = tuple(str(row["request_id"]) for row in decisions)
            proposal_ids = tuple(
                str(row["proposal_ref"])
                for row in decisions
                if row["proposal_ref"] is not None
            )

            self._reject_structured_external_references(
                cursor, run, request_ids, proposal_ids
            )
            self._reject_unstructured_provenance(cursor, run)

            report = PurgeRunReport(
                run_id=run_id,
                run_kind=str(metadata["run_kind"]),
                decision_count=len(decisions),
                dry_run=dry_run,
                deleted=not dry_run,
            )
            if dry_run:
                return report
            self._delete_run(cursor, run, decisions)
            return report

    @staticmethod
    def _reject_structured_external_references(
        cursor: object,
        run_id: str,
        request_ids: tuple[str, ...],
        proposal_ids: tuple[str, ...],
    ) -> None:
        if request_ids:
            placeholders = ", ".join("?" for _ in request_ids)
            feedback = cursor.execute(  # type: ignore[attr-defined]
                "SELECT 1 FROM decision_feedback WHERE subject_decision_id IN "
                f"({placeholders}) AND source_run_id != ? LIMIT 1",
                (*request_ids, run_id),
            ).fetchone()
            if feedback is not None:
                raise PersistenceConflictError(
                    "Run Decisions are referenced by external Decision Feedback"
                )
            proposal = cursor.execute(  # type: ignore[attr-defined]
                "SELECT 1 FROM optimization_proposals "
                "WHERE source_decision_request_id IN "
                f"({placeholders}) LIMIT 1",
                request_ids,
            ).fetchone()
            if proposal is not None:
                raise PersistenceConflictError(
                    "Run Decisions are referenced by Optimization provenance"
                )
        if proposal_ids:
            placeholders = ", ".join("?" for _ in proposal_ids)
            trigger = cursor.execute(  # type: ignore[attr-defined]
                "SELECT 1 FROM auto_adaptation_triggers "
                f"WHERE selected_proposal_id IN ({placeholders}) LIMIT 1",
                proposal_ids,
            ).fetchone()
            if trigger is not None:
                raise PersistenceConflictError(
                    "Run Proposals are referenced by Auto-adaptation provenance"
                )

    @staticmethod
    def _reject_unstructured_provenance(cursor: object, run_id: str) -> None:
        # These domains still contain provenance inside their own immutable JSON.
        # Until they expose authoritative reference columns, cleanup refuses any
        # cross-run population instead of guessing or parsing payloads.
        for table, run_column in (
            ("experience_metadata", "source_run_id"),
            ("workspace_artifacts", "run_id"),
            ("decision_feedback", "source_run_id"),
        ):
            row = cursor.execute(  # type: ignore[attr-defined]
                f"SELECT 1 FROM {table} WHERE {run_column} != ? LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is not None:
                raise PersistenceConflictError(
                    f"Cannot prove {table} has no cross-run provenance reference"
                )
        for table in ("learning_insights", "optimization_configuration_receipts"):
            row = cursor.execute(  # type: ignore[attr-defined]
                f"SELECT 1 FROM {table} LIMIT 1"
            ).fetchone()
            if row is not None:
                raise PersistenceConflictError(
                    f"Cannot safely purge while {table} provenance exists"
                )

    @staticmethod
    def _delete_run(
        cursor: object, run_id: str, decisions: Sequence[Row]
    ) -> None:
        decision_rows = tuple(decisions)
        request_ids = tuple(str(row["request_id"]) for row in decision_rows)
        effect_refs = tuple(
            str(row["effect_ref"])
            for row in decision_rows
            if row["effect_ref"] is not None
        )
        commit_refs = tuple(
            str(row["commit_receipt_ref"])
            for row in decision_rows
            if row["commit_receipt_ref"] is not None
        )
        evidence_refs: set[str] = set()
        if request_ids:
            placeholders = ", ".join("?" for _ in request_ids)
            rows = cursor.execute(  # type: ignore[attr-defined]
                "SELECT evidence_snapshot_ref FROM decision_requests "
                f"WHERE request_id IN ({placeholders}) "
                "AND evidence_snapshot_ref IS NOT NULL",
                request_ids,
            ).fetchall()
            evidence_refs.update(str(row[0]) for row in rows)
            if effect_refs:
                effect_placeholders = ", ".join("?" for _ in effect_refs)
                rows = cursor.execute(  # type: ignore[attr-defined]
                    "SELECT evidence_snapshot_ref FROM decision_effects "
                    f"WHERE effect_fingerprint IN ({effect_placeholders}) "
                    "AND evidence_snapshot_ref IS NOT NULL",
                    effect_refs,
                ).fetchall()
                evidence_refs.update(str(row[0]) for row in rows)

            cursor.execute(  # type: ignore[attr-defined]
                f"DELETE FROM decision_transitions WHERE request_id IN ({placeholders})",
                request_ids,
            )
            cursor.execute(  # type: ignore[attr-defined]
                f"DELETE FROM decision_current WHERE request_id IN ({placeholders})",
                request_ids,
            )
            for table in (
                "decision_results",
                "decision_governance_receipts",
                "decision_validations",
                "decision_proposals",
                "decision_requests",
            ):
                cursor.execute(  # type: ignore[attr-defined]
                    f"DELETE FROM {table} WHERE request_id IN ({placeholders})",
                    request_ids,
                )
            for reference in commit_refs:
                used = cursor.execute(  # type: ignore[attr-defined]
                    "SELECT 1 FROM decision_current WHERE commit_receipt_ref = ? "
                    "UNION ALL SELECT 1 FROM decision_results "
                    "WHERE commit_receipt_ref = ? LIMIT 1",
                    (reference, reference),
                ).fetchone()
                if used is None:
                    cursor.execute(  # type: ignore[attr-defined]
                        "DELETE FROM decision_commit_receipts "
                        "WHERE commit_receipt_id = ?",
                        (reference,),
                    )
            for reference in effect_refs:
                used = cursor.execute(  # type: ignore[attr-defined]
                    "SELECT 1 FROM decision_current WHERE effect_ref = ? "
                    "UNION ALL SELECT 1 FROM decision_validations "
                    "WHERE effect_fingerprint = ? "
                    "UNION ALL SELECT 1 FROM decision_commit_receipts "
                    "WHERE effect_fingerprint = ? LIMIT 1",
                    (reference, reference, reference),
                ).fetchone()
                if used is None:
                    cursor.execute(  # type: ignore[attr-defined]
                        "DELETE FROM decision_effects WHERE effect_fingerprint = ?",
                        (reference,),
                    )
            for reference in evidence_refs:
                used = cursor.execute(  # type: ignore[attr-defined]
                    "SELECT 1 FROM decision_requests WHERE evidence_snapshot_ref = ? "
                    "UNION ALL SELECT 1 FROM decision_effects "
                    "WHERE evidence_snapshot_ref = ? LIMIT 1",
                    (reference, reference),
                ).fetchone()
                if used is None:
                    cursor.execute(  # type: ignore[attr-defined]
                        "DELETE FROM decision_evidence_snapshots "
                        "WHERE storage_payload_fingerprint = ?",
                        (reference,),
                    )

        experience_effects = tuple(
            str(row[0])
            for row in cursor.execute(  # type: ignore[attr-defined]
                "SELECT effect_fingerprint FROM experience_metadata "
                "WHERE source_run_id = ?",
                (run_id,),
            ).fetchall()
        )
        if experience_effects:
            placeholders = ", ".join("?" for _ in experience_effects)
            cursor.execute(  # type: ignore[attr-defined]
                f"DELETE FROM experience_memory_links WHERE effect_fingerprint IN "
                f"({placeholders})",
                experience_effects,
            )
        context_ids = tuple(
            str(row[0])
            for row in cursor.execute(  # type: ignore[attr-defined]
                "SELECT DISTINCT context_id FROM context_snapshots WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        )
        if context_ids:
            placeholders = ", ".join("?" for _ in context_ids)
            cursor.execute(  # type: ignore[attr-defined]
                f"DELETE FROM context_archive_transactions WHERE context_id IN "
                f"({placeholders})",
                context_ids,
            )
            cursor.execute(  # type: ignore[attr-defined]
                f"DELETE FROM context_archives WHERE context_id IN ({placeholders})",
                context_ids,
            )
            cursor.execute(  # type: ignore[attr-defined]
                f"DELETE FROM context_current WHERE context_id IN ({placeholders})",
                context_ids,
            )
        for sql in (
            "DELETE FROM context_snapshots WHERE run_id = ?",
            "DELETE FROM runtime_trace WHERE run_id = ?",
            "DELETE FROM task_graph_checkpoint_journal WHERE run_id = ?",
            "DELETE FROM task_graph_history WHERE run_id = ?",
            "DELETE FROM task_graph_checkpoints WHERE run_id = ?",
            "DELETE FROM workspace_artifacts WHERE run_id = ?",
            "DELETE FROM memory_recall_bundles WHERE run_id = ?",
            "DELETE FROM evaluation_reports WHERE run_id = ?",
            "DELETE FROM decision_feedback WHERE source_run_id = ?",
            "DELETE FROM experience_metadata WHERE source_run_id = ?",
            "DELETE FROM application_run_manifests WHERE run_id = ?",
            "DELETE FROM agent_state_current WHERE run_id = ?",
            "DELETE FROM agent_state_snapshots WHERE run_id = ?",
            "DELETE FROM runtime_run_metadata WHERE run_id = ?",
        ):
            cursor.execute(sql, (run_id,))  # type: ignore[attr-defined]
