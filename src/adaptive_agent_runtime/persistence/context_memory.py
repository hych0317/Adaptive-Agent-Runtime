"""Durable Context, Archive, and evidence-driven Memory stores."""

from __future__ import annotations

import json
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    ContextArchiveReference,
    ContextLifecycleState,
    ContextRecoveryError,
    ContextSnapshotConflictError,
    ContextTransitionError,
    ContextUnit,
    MemorySnapshotConflictError,
    MemoryUnit,
    MemoryBatchWrite,
    MemoryBatchCommitReceipt,
)
from adaptive_agent_runtime.context_memory.json_types import utc_now
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance.contracts import CommitPermitValidation
from adaptive_agent_runtime.governance.errors import AuthorizationVerificationError
from adaptive_agent_runtime.governance.models import GovernanceTarget, RuntimeCommitPermit
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


def _model_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class SQLiteContextStore:
    """Versioned resident Context Unit storage."""

    module_id = "context.store.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def save(
        self,
        unit: ContextUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        if unit.lifecycle_state is ContextLifecycleState.ARCHIVED:
            raise ContextTransitionError("archived context cannot remain resident")
        context_id = str(unit.context_id)
        payload = _model_json(unit)
        with self._database.transaction() as cursor:
            current = cursor.execute(
                "SELECT revision FROM context_current WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            latest = cursor.execute(
                "SELECT MAX(revision) AS revision FROM context_snapshots "
                "WHERE context_id = ?",
                (context_id,),
            ).fetchone()
            latest_revision = (
                None if latest["revision"] is None else int(latest["revision"])
            )
            if expected_revision is None:
                if current is not None:
                    existing = cursor.execute(
                        "SELECT snapshot_json FROM context_snapshots "
                        "WHERE context_id = ? AND revision = ?",
                        (context_id, int(current["revision"])),
                    ).fetchone()
                    if existing is not None and existing["snapshot_json"] == payload:
                        return
                    raise ContextSnapshotConflictError(
                        "context already has a resident snapshot"
                    )
                if latest_revision is not None:
                    raise ContextSnapshotConflictError(
                        "archived context requires a versioned restore write"
                    )
            else:
                base_revision = (
                    int(current["revision"])
                    if current is not None
                    else latest_revision
                )
                if (
                    base_revision != expected_revision
                    or unit.revision != expected_revision + 1
                ):
                    raise ContextSnapshotConflictError(
                        "context write is based on a stale snapshot"
                    )
            existing_revision = cursor.execute(
                "SELECT snapshot_json FROM context_snapshots "
                "WHERE context_id = ? AND revision = ?",
                (context_id, unit.revision),
            ).fetchone()
            if existing_revision is not None:
                if existing_revision["snapshot_json"] == payload:
                    cursor.execute(
                        "INSERT INTO context_current(context_id, revision) "
                        "VALUES (?, ?) ON CONFLICT(context_id) DO UPDATE SET "
                        "revision = excluded.revision",
                        (context_id, unit.revision),
                    )
                    return
                raise ContextSnapshotConflictError(
                    "context revision was reused with different content"
                )
            cursor.execute(
                "INSERT INTO context_snapshots "
                "(context_id, revision, run_id, snapshot_json) VALUES (?, ?, ?, ?)",
                (context_id, unit.revision, str(unit.metadata.run_id), payload),
            )
            cursor.execute(
                "INSERT INTO context_current(context_id, revision) VALUES (?, ?) "
                "ON CONFLICT(context_id) DO UPDATE SET revision = excluded.revision",
                (context_id, unit.revision),
            )

    async def load(self, context_id: UUID) -> ContextUnit | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshots.snapshot_json FROM context_current AS current "
                "JOIN context_snapshots AS snapshots "
                "ON snapshots.context_id = current.context_id "
                "AND snapshots.revision = current.revision "
                "WHERE current.context_id = ?",
                (str(context_id),),
            ).fetchone()
        if row is None:
            return None
        return ContextUnit.model_validate_json(row["snapshot_json"])

    async def delete(
        self,
        context_id: UUID,
        *,
        expected_revision: int,
    ) -> None:
        with self._database.transaction() as cursor:
            current = cursor.execute(
                "SELECT revision FROM context_current WHERE context_id = ?",
                (str(context_id),),
            ).fetchone()
            if current is None or int(current["revision"]) != expected_revision:
                raise ContextSnapshotConflictError(
                    "context delete is based on a stale snapshot"
                )
            cursor.execute(
                "DELETE FROM context_current WHERE context_id = ?",
                (str(context_id),),
            )

    async def list_for_run(self, run_id: UUID) -> tuple[ContextUnit, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT snapshots.snapshot_json FROM context_current AS current "
                "JOIN context_snapshots AS snapshots "
                "ON snapshots.context_id = current.context_id "
                "AND snapshots.revision = current.revision "
                "WHERE snapshots.run_id = ? ORDER BY snapshots.context_id",
                (str(run_id),),
            ).fetchall()
        return tuple(
            ContextUnit.model_validate_json(row["snapshot_json"])
            for row in rows
        )

    async def history_for(self, context_id: UUID) -> tuple[ContextUnit, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT snapshot_json FROM context_snapshots "
                "WHERE context_id = ? ORDER BY revision",
                (str(context_id),),
            ).fetchall()
        return tuple(
            ContextUnit.model_validate_json(row["snapshot_json"])
            for row in rows
        )


class SQLiteContextArchive:
    """Archive payloads survive process and resident-store restarts."""

    module_id = "context.archive.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def archive(
        self,
        unit: ContextUnit,
        *,
        reference: ContextArchiveReference | None = None,
    ) -> ContextArchiveReference:
        reference = reference or ContextArchiveReference(context_id=unit.context_id)
        if reference.context_id != unit.context_id:
            raise ContextRecoveryError("archive reference has a different Context identity")
        with self._database.reader() as cursor:
            existing_row = cursor.execute(
                "SELECT context_id, snapshot_json FROM context_archives "
                "WHERE archive_id = ?",
                (str(reference.archive_id),),
            ).fetchone()
        if existing_row is not None:
            archived = ContextUnit.model_validate_json(existing_row["snapshot_json"])
            if UUID(existing_row["context_id"]) != unit.context_id:
                raise ContextRecoveryError("archive id is bound to another Context")
            return ContextArchiveReference(
                archive_id=reference.archive_id,
                context_id=unit.context_id,
                archived_at=archived.metadata.updated_at,
            )
        values = unit.model_dump(mode="python")
        values["lifecycle_state"] = ContextLifecycleState.ARCHIVED
        metadata = unit.metadata.model_dump(mode="python")
        metadata["updated_at"] = utc_now()
        values["metadata"] = metadata
        archived = ContextUnit.model_validate(values)
        with self._database.transaction() as cursor:
            cursor.execute(
                "INSERT INTO context_archives "
                "(archive_id, context_id, snapshot_json) VALUES (?, ?, ?)",
                (
                    str(reference.archive_id),
                    str(reference.context_id),
                    _model_json(archived),
                ),
            )
        return reference

    async def stage_compression(
        self,
        unit: ContextUnit,
        *,
        reference: ContextArchiveReference,
        effect_fingerprint: str,
        source_fingerprint: str,
    ) -> ContextArchiveReference:
        """Persist an invisible, fingerprint-bound recovery snapshot."""

        if (
            reference.context_id != unit.context_id
            or decision_fingerprint(unit) != source_fingerprint
        ):
            raise ContextSnapshotConflictError(
                "compression Archive stage does not match its source snapshot"
            )
        values = unit.model_dump(mode="python")
        values["lifecycle_state"] = ContextLifecycleState.ARCHIVED
        metadata = unit.metadata.model_dump(mode="python")
        metadata["updated_at"] = utc_now()
        values["metadata"] = metadata
        archived = ContextUnit.model_validate(values)
        payload = _model_json(archived)
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT archive_id, context_id, source_fingerprint, archive_json "
                "FROM context_archive_transactions WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if row is not None:
                if (
                    row["archive_id"] != str(reference.archive_id)
                    or row["context_id"] != str(unit.context_id)
                    or row["source_fingerprint"] != source_fingerprint
                ):
                    raise ContextSnapshotConflictError(
                        "compression Effect is bound to another pending Archive"
                    )
                existing = ContextUnit.model_validate_json(row["archive_json"])
                if (
                    existing.context_id != unit.context_id
                    or existing.revision != unit.revision
                    or existing.content != unit.content
                ):
                    raise ContextSnapshotConflictError(
                        "pending Archive source payload changed"
                    )
                return ContextArchiveReference(
                    archive_id=reference.archive_id,
                    context_id=unit.context_id,
                    archived_at=existing.metadata.updated_at,
                )
            cursor.execute(
                "INSERT INTO context_archive_transactions "
                "(effect_fingerprint, archive_id, context_id, source_fingerprint, "
                "status, archive_json) VALUES (?, ?, ?, ?, 'pending', ?)",
                (
                    effect_fingerprint,
                    str(reference.archive_id),
                    str(unit.context_id),
                    source_fingerprint,
                    payload,
                ),
            )
        return ContextArchiveReference(
            archive_id=reference.archive_id,
            context_id=unit.context_id,
            archived_at=archived.metadata.updated_at,
        )

    async def commit_compression(
        self,
        source: ContextUnit,
        compressed: ContextUnit,
        *,
        reference: ContextArchiveReference,
        effect_fingerprint: str,
        source_fingerprint: str,
        permit: RuntimeCommitPermit,
        target: GovernanceTarget,
        subject_fingerprint: str,
    ) -> ContextUnit:
        """Publish Archive and resident revision in one SQLite transaction."""

        await self._permit_verifier.verify(
            permit,
            operation="context.compress",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )

        if (
            decision_fingerprint(source) != source_fingerprint
            or compressed.context_id != source.context_id
            or compressed.revision != source.revision + 1
            or compressed.recovery_reference is None
            or compressed.recovery_reference.archive_id != reference.archive_id
            or compressed.last_effect_fingerprint != effect_fingerprint
        ):
            raise ContextSnapshotConflictError(
                "compression commit payload is not bound to the staged Effect"
            )
        context_id = str(source.context_id)
        compressed_json = _model_json(compressed)
        with self._database.transaction() as cursor:
            transaction = cursor.execute(
                "SELECT archive_id, context_id, source_fingerprint, status, archive_json "
                "FROM context_archive_transactions WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if transaction is None:
                raise ContextSnapshotConflictError(
                    "compression commit has no pending Archive"
                )
            if (
                transaction["archive_id"] != str(reference.archive_id)
                or transaction["context_id"] != context_id
                or transaction["source_fingerprint"] != source_fingerprint
            ):
                raise ContextSnapshotConflictError(
                    "pending Archive identity conflicts with compression Effect"
                )
            current = cursor.execute(
                "SELECT current.revision, snapshots.snapshot_json "
                "FROM context_current AS current JOIN context_snapshots AS snapshots "
                "ON snapshots.context_id = current.context_id "
                "AND snapshots.revision = current.revision "
                "WHERE current.context_id = ?",
                (context_id,),
            ).fetchone()
            if transaction["status"] == "committed":
                if current is None or current["snapshot_json"] != compressed_json:
                    raise ContextSnapshotConflictError(
                        "committed compression failed authoritative read-back"
                    )
                return ContextUnit.model_validate_json(current["snapshot_json"])
            if (
                current is None
                or int(current["revision"]) != source.revision
                or current["snapshot_json"] != _model_json(source)
            ):
                raise ContextSnapshotConflictError(
                    "compression source is no longer authoritative"
                )
            cursor.execute(
                "INSERT INTO context_archives "
                "(archive_id, context_id, snapshot_json) VALUES (?, ?, ?)",
                (
                    str(reference.archive_id),
                    context_id,
                    transaction["archive_json"],
                ),
            )
            cursor.execute(
                "INSERT INTO context_snapshots "
                "(context_id, revision, run_id, snapshot_json) VALUES (?, ?, ?, ?)",
                (
                    context_id,
                    compressed.revision,
                    str(compressed.metadata.run_id),
                    compressed_json,
                ),
            )
            cursor.execute(
                "UPDATE context_current SET revision = ? WHERE context_id = ?",
                (compressed.revision, context_id),
            )
            cursor.execute(
                "UPDATE context_archive_transactions SET status = 'committed' "
                "WHERE effect_fingerprint = ? AND status = 'pending'",
                (effect_fingerprint,),
            )
        readback = await SQLiteContextStore(self._database).load(source.context_id)
        if readback != compressed:
            raise ContextSnapshotConflictError(
                "compression transaction failed final resident read-back"
            )
        restored = await self.restore(reference)
        if (
            restored.context_id != source.context_id
            or restored.revision != source.revision
            or decision_fingerprint(source) != source_fingerprint
        ):
            raise ContextSnapshotConflictError(
                "compression transaction failed Archive read-back"
            )
        return readback

    async def pending_count(self) -> int:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT COUNT(*) AS count FROM context_archive_transactions "
                "WHERE status = 'pending'"
            ).fetchone()
        return int(row["count"])

    async def verify_compression(
        self,
        unit: ContextUnit,
        *,
        effect_fingerprint: str,
        source_fingerprint: str,
    ) -> bool:
        reference = unit.recovery_reference
        if reference is None:
            return False
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT tx.archive_id, tx.context_id, tx.source_fingerprint, "
                "tx.status, archives.snapshot_json "
                "FROM context_archive_transactions AS tx "
                "LEFT JOIN context_archives AS archives "
                "ON archives.archive_id = tx.archive_id "
                "WHERE tx.effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if (
            row is None
            or row["status"] != "committed"
            or row["archive_id"] != str(reference.archive_id)
            or row["context_id"] != str(unit.context_id)
            or row["source_fingerprint"] != source_fingerprint
            or row["snapshot_json"] is None
        ):
            return False
        archived = ContextUnit.model_validate_json(row["snapshot_json"])
        return (
            archived.context_id == unit.context_id
            and archived.revision + 1 == unit.revision
            and unit.last_effect_fingerprint == effect_fingerprint
        )

    async def discard(self, reference: ContextArchiveReference) -> None:
        with self._database.transaction() as cursor:
            cursor.execute(
                "DELETE FROM context_archives "
                "WHERE archive_id = ? AND context_id = ?",
                (str(reference.archive_id), str(reference.context_id)),
            )

    async def restore(self, reference: ContextArchiveReference) -> ContextUnit:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshot_json FROM context_archives "
                "WHERE archive_id = ? AND context_id = ?",
                (str(reference.archive_id), str(reference.context_id)),
            ).fetchone()
        if row is None:
            raise ContextRecoveryError(
                f"archive reference '{reference.archive_id}' is unavailable"
            )
        return ContextUnit.model_validate_json(row["snapshot_json"])

    async def find_latest(
        self,
        context_id: UUID,
    ) -> ContextArchiveReference | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT archive_id, snapshot_json FROM context_archives "
                "WHERE context_id = ? ORDER BY rowid DESC LIMIT 1",
                (str(context_id),),
            ).fetchone()
        if row is None:
            return None
        archived = ContextUnit.model_validate_json(row["snapshot_json"])
        return ContextArchiveReference(
            archive_id=UUID(row["archive_id"]),
            context_id=context_id,
            archived_at=archived.metadata.updated_at,
        )

    async def record_count(self) -> int:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT COUNT(*) AS count FROM context_archives"
            ).fetchone()
        return int(row["count"])


class SQLiteMemoryStore:
    """Versioned Memory Unit storage with durable candidate idempotency."""

    module_id = "memory.store.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def save(
        self,
        memory: MemoryUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        candidate_id = memory.last_candidate_id
        if candidate_id is None:
            raise MemorySnapshotConflictError(
                "memory writes require an originating candidate"
            )
        memory_id = str(memory.memory_id)
        payload = _model_json(memory)
        with self._database.transaction() as cursor:
            applied = cursor.execute(
                "SELECT memory_id, revision FROM memory_applied_candidates "
                "WHERE candidate_id = ?",
                (str(candidate_id),),
            ).fetchone()
            if applied is not None:
                existing = cursor.execute(
                    "SELECT snapshot_json FROM memory_snapshots "
                    "WHERE memory_id = ? AND revision = ?",
                    (applied["memory_id"], int(applied["revision"])),
                ).fetchone()
                if existing is not None and existing["snapshot_json"] == payload:
                    return
                raise MemorySnapshotConflictError(
                    "candidate id was already applied with different content"
                )
            current = cursor.execute(
                "SELECT revision FROM memory_current WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if expected_revision is None:
                if current is not None:
                    raise MemorySnapshotConflictError(
                        "memory already has a current snapshot"
                    )
            elif (
                current is None
                or int(current["revision"]) != expected_revision
                or memory.revision != expected_revision + 1
            ):
                raise MemorySnapshotConflictError(
                    "memory write is based on a stale snapshot"
                )
            cursor.execute(
                "INSERT INTO memory_snapshots "
                "(memory_id, revision, snapshot_json) VALUES (?, ?, ?)",
                (memory_id, memory.revision, payload),
            )
            cursor.execute(
                "INSERT INTO memory_current(memory_id, revision) VALUES (?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET revision = excluded.revision",
                (memory_id, memory.revision),
            )
            cursor.execute(
                "INSERT INTO memory_applied_candidates "
                "(candidate_id, memory_id, revision) VALUES (?, ?, ?)",
                (str(candidate_id), memory_id, memory.revision),
            )

    async def load(self, memory_id: UUID) -> MemoryUnit | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshots.snapshot_json FROM memory_current AS current "
                "JOIN memory_snapshots AS snapshots "
                "ON snapshots.memory_id = current.memory_id "
                "AND snapshots.revision = current.revision "
                "WHERE current.memory_id = ?",
                (str(memory_id),),
            ).fetchone()
        if row is None:
            return None
        return MemoryUnit.model_validate_json(row["snapshot_json"])

    async def load_applied_candidate(
        self,
        candidate_id: UUID,
    ) -> MemoryUnit | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshots.snapshot_json "
                "FROM memory_applied_candidates AS applied "
                "JOIN memory_snapshots AS snapshots "
                "ON snapshots.memory_id = applied.memory_id "
                "AND snapshots.revision = applied.revision "
                "WHERE applied.candidate_id = ?",
                (str(candidate_id),),
            ).fetchone()
        if row is None:
            return None
        return MemoryUnit.model_validate_json(row["snapshot_json"])

    async def list_all(self) -> tuple[MemoryUnit, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT snapshots.snapshot_json FROM memory_current AS current "
                "JOIN memory_snapshots AS snapshots "
                "ON snapshots.memory_id = current.memory_id "
                "AND snapshots.revision = current.revision "
                "ORDER BY snapshots.memory_id"
            ).fetchall()
        return tuple(
            MemoryUnit.model_validate_json(row["snapshot_json"])
            for row in rows
        )

    async def save_batch(
        self,
        writes: tuple[MemoryBatchWrite, ...],
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> tuple[MemoryUnit, ...]:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Memory batch commit requires a Runtime Permit"
            )
        await self._permit_verifier.verify(
            permit,
            operation="memory.write",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        payload_fingerprint = decision_fingerprint(writes)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT payload_fingerprint, result_json FROM memory_batch_receipts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                if prior["payload_fingerprint"] != payload_fingerprint:
                    raise MemorySnapshotConflictError(
                        "Memory Effect fingerprint was reused with another batch"
                    )
                raw = json.loads(prior["result_json"])
                return tuple(MemoryUnit.model_validate(item) for item in raw)
            legacy = cursor.execute(
                "SELECT result_json FROM memory_applied_effects "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if legacy is not None:
                raw = json.loads(legacy["result_json"])
                committed_legacy = tuple(
                    MemoryUnit.model_validate(item) for item in raw
                )
                if committed_legacy != tuple(write.memory for write in writes):
                    raise MemorySnapshotConflictError(
                        "legacy Memory Effect conflicts with requested batch"
                    )
                receipt = MemoryBatchCommitReceipt(
                    effect_fingerprint=effect_fingerprint,
                    payload_fingerprint=payload_fingerprint,
                    result_fingerprint=decision_fingerprint(committed_legacy),
                    memory_count=len(committed_legacy),
                )
                cursor.execute(
                    "INSERT INTO memory_batch_receipts "
                    "(effect_fingerprint, payload_fingerprint, result_json, "
                    "receipt_json) VALUES (?, ?, ?, ?)",
                    (
                        effect_fingerprint,
                        payload_fingerprint,
                        legacy["result_json"],
                        _model_json(receipt),
                    ),
                )
                return committed_legacy
            committed: list[MemoryUnit] = []
            for write in writes:
                memory = write.memory
                candidate_id = memory.last_candidate_id
                if candidate_id is None:
                    raise MemorySnapshotConflictError(
                        "Memory batch write has no originating candidate"
                    )
                if cursor.execute(
                    "SELECT 1 FROM memory_applied_candidates WHERE candidate_id = ?",
                    (str(candidate_id),),
                ).fetchone() is not None:
                    raise MemorySnapshotConflictError(
                        "Memory batch candidate was already applied"
                    )
                memory_id = str(memory.memory_id)
                current = cursor.execute(
                    "SELECT revision FROM memory_current WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                if write.expected_revision is None:
                    if current is not None:
                        raise MemorySnapshotConflictError(
                            "Memory batch create collides with current state"
                        )
                elif (
                    current is None
                    or int(current["revision"]) != write.expected_revision
                    or memory.revision != write.expected_revision + 1
                ):
                    raise MemorySnapshotConflictError(
                        "Memory batch write is based on a stale snapshot"
                    )
                cursor.execute(
                    "INSERT INTO memory_snapshots "
                    "(memory_id, revision, snapshot_json) VALUES (?, ?, ?)",
                    (memory_id, memory.revision, _model_json(memory)),
                )
                cursor.execute(
                    "INSERT INTO memory_current(memory_id, revision) VALUES (?, ?) "
                    "ON CONFLICT(memory_id) DO UPDATE SET revision = excluded.revision",
                    (memory_id, memory.revision),
                )
                cursor.execute(
                    "INSERT INTO memory_applied_candidates "
                    "(candidate_id, memory_id, revision) VALUES (?, ?, ?)",
                    (str(candidate_id), memory_id, memory.revision),
                )
                committed.append(memory)
            result = tuple(committed)
            result_json = json.dumps(
                [item.model_dump(mode="json") for item in result],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            receipt = MemoryBatchCommitReceipt(
                effect_fingerprint=effect_fingerprint,
                payload_fingerprint=payload_fingerprint,
                result_fingerprint=decision_fingerprint(result),
                memory_count=len(result),
            )
            cursor.execute(
                "INSERT INTO memory_applied_effects "
                "(effect_fingerprint, result_json) VALUES (?, ?)",
                (effect_fingerprint, result_json),
            )
            cursor.execute(
                "INSERT INTO memory_batch_receipts "
                "(effect_fingerprint, payload_fingerprint, result_json, receipt_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    effect_fingerprint,
                    payload_fingerprint,
                    result_json,
                    _model_json(receipt),
                ),
            )
        return result

    async def load_applied_effect(
        self,
        effect_fingerprint: str,
    ) -> tuple[MemoryUnit, ...] | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT result_json FROM memory_applied_effects "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        raw = json.loads(row["result_json"])
        return tuple(MemoryUnit.model_validate(item) for item in raw)

    async def load_batch_receipt(
        self,
        effect_fingerprint: str,
    ) -> MemoryBatchCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT receipt_json FROM memory_batch_receipts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        return MemoryBatchCommitReceipt.model_validate_json(row["receipt_json"])

    async def history_for(self, memory_id: UUID) -> tuple[MemoryUnit, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT snapshot_json FROM memory_snapshots "
                "WHERE memory_id = ? ORDER BY revision",
                (str(memory_id),),
            ).fetchall()
        return tuple(
            MemoryUnit.model_validate_json(row["snapshot_json"])
            for row in rows
        )
