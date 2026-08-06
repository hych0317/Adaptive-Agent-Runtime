"""SQLite commit boundary for immutable governed Memory Recall bundles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    MemoryRecallBundle,
    MemoryRecallBundleItem,
    MemoryRecallCommitReceipt,
    MemoryRecallEffect,
    MemoryStatus,
    MemoryUnit,
    stable_recall_bundle_id,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance import (
    AuthorizationVerificationError,
    CommitPermitValidation,
    GovernanceTarget,
    RuntimeCommitPermit,
)
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class SQLiteMemoryRecallBundleStore:
    module_id = "memory.recall_bundle_store.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def commit(
        self,
        effect: MemoryRecallEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> MemoryRecallBundle:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Memory Recall Bundle commit requires a Runtime Permit"
            )
        await self._permit_verifier.verify(
            permit,
            operation="memory.recall.commit",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        payload_fingerprint = decision_fingerprint(effect)
        bundle_id = stable_recall_bundle_id(effect_fingerprint)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT payload_fingerprint, bundle_json FROM memory_recall_bundles "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                if prior["payload_fingerprint"] != payload_fingerprint:
                    raise PersistenceConflictError(
                        "Recall Effect fingerprint conflicts with stored payload"
                    )
                return MemoryRecallBundle.model_validate_json(prior["bundle_json"])

            for source in effect.sources:
                row = cursor.execute(
                    "SELECT snapshots.snapshot_json FROM memory_current AS current "
                    "JOIN memory_snapshots AS snapshots "
                    "ON snapshots.memory_id = current.memory_id "
                    "AND snapshots.revision = current.revision "
                    "WHERE current.memory_id = ?",
                    (str(source.memory_id),),
                ).fetchone()
                if row is None:
                    raise PersistenceConflictError(
                        "Recall source Memory no longer exists"
                    )
                memory = MemoryUnit.model_validate_json(row["snapshot_json"])
                if (
                    memory.revision != source.memory_revision
                    or decision_fingerprint(memory) != source.source_fingerprint
                    or memory.scope != source.scope
                    or memory.sensitivity is not source.sensitivity
                    or memory.status is not MemoryStatus.ACTIVE
                    or (
                        memory.expires_at is not None
                        and memory.expires_at <= datetime.now(timezone.utc)
                    )
                ):
                    raise PersistenceConflictError(
                        "Recall source Memory changed after proposal"
                    )

            bundle = MemoryRecallBundle(
                bundle_id=bundle_id,
                recall_decision_id=effect.recall_decision_id,
                run_id=effect.run_id,
                task_id=effect.task_id,
                items=tuple(
                    MemoryRecallBundleItem(
                        source_memory_ref=source.candidate_ref,
                        source_memory_version=source.memory_revision,
                        content=source.sanitized_content,
                        memory_category=source.memory_category,
                        provenance=source.provenance_summary,
                    )
                    for source in effect.sources
                ),
                scope=effect.scope,
                max_items=effect.max_items,
                max_tokens=effect.max_tokens,
                used_tokens=sum(item.estimated_tokens for item in effect.sources),
                candidate_set_fingerprint=effect.candidate_set_fingerprint,
                effect_fingerprint=effect_fingerprint,
                provenance=tuple(
                    dict.fromkeys(
                        item
                        for source in effect.sources
                        for item in source.provenance_summary
                    )
                ),
            )
            bundle_fingerprint = decision_fingerprint(bundle)
            receipt = MemoryRecallCommitReceipt(
                bundle_id=bundle.bundle_id,
                recall_decision_id=bundle.recall_decision_id,
                effect_fingerprint=effect_fingerprint,
                bundle_fingerprint=bundle_fingerprint,
                source_count=len(bundle.items),
                committed_at=bundle.committed_at,
            )
            cursor.execute(
                "INSERT INTO memory_recall_bundles "
                "(effect_fingerprint, bundle_id, run_id, task_id, "
                "payload_fingerprint, bundle_json, receipt_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    effect_fingerprint,
                    str(bundle.bundle_id),
                    str(bundle.run_id),
                    str(bundle.task_id),
                    payload_fingerprint,
                    bundle.model_dump_json(),
                    receipt.model_dump_json(),
                ),
            )
            return bundle

    async def load_by_effect(
        self,
        effect_fingerprint: str,
    ) -> MemoryRecallBundle | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT bundle_json, receipt_json FROM memory_recall_bundles "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        bundle = MemoryRecallBundle.model_validate_json(row["bundle_json"])
        receipt = MemoryRecallCommitReceipt.model_validate_json(
            row["receipt_json"]
        )
        if (
            bundle.effect_fingerprint != effect_fingerprint
            or receipt.effect_fingerprint != effect_fingerprint
            or receipt.bundle_id != bundle.bundle_id
            or receipt.bundle_fingerprint != decision_fingerprint(bundle)
        ):
            raise PersistenceConflictError("Recall Bundle read-back is corrupted")
        return bundle

    async def load_receipt(
        self,
        effect_fingerprint: str,
    ) -> MemoryRecallCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT receipt_json FROM memory_recall_bundles "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        return (
            None
            if row is None
            else MemoryRecallCommitReceipt.model_validate_json(row["receipt_json"])
        )

    async def load_for_run(self, run_id: UUID) -> MemoryRecallBundle | None:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT bundle_json FROM memory_recall_bundles WHERE run_id = ?",
                (str(run_id),),
            ).fetchall()
        if len(rows) > 1:
            raise PersistenceConflictError(
                "Initial Planning run has multiple Recall Bundles"
            )
        return (
            None
            if not rows
            else MemoryRecallBundle.model_validate_json(rows[0]["bundle_json"])
        )

    async def count_for_run(self, run_id: UUID) -> int:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT COUNT(*) AS count FROM memory_recall_bundles WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return int(row["count"])
