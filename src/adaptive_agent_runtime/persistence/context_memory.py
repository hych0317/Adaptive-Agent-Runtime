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
)
from adaptive_agent_runtime.context_memory.json_types import utc_now
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

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def archive(self, unit: ContextUnit) -> ContextArchiveReference:
        reference = ContextArchiveReference(context_id=unit.context_id)
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

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

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
