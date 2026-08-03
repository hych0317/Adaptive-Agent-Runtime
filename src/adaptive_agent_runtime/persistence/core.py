"""Durable implementations of Runtime Core storage contracts."""

from __future__ import annotations

import json
from uuid import UUID

from adaptive_agent_runtime import AgentState, RuntimeEvent, TraceEntry
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


def _json_text(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class SQLiteStateStore:
    """Append immutable AgentState revisions and point to the latest one."""

    module_id = "state.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def save(self, state: AgentState) -> None:
        run_id = str(state.run_id)
        payload = _json_text(state)
        with self._database.transaction() as cursor:
            current = cursor.execute(
                "SELECT revision FROM agent_state_current WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if current is None:
                if state.revision != 0:
                    raise PersistenceConflictError(
                        "a new run must begin at state revision 0"
                    )
            else:
                current_revision = int(current["revision"])
                if current_revision == state.revision:
                    existing = cursor.execute(
                        "SELECT snapshot_json FROM agent_state_snapshots "
                        "WHERE run_id = ? AND revision = ?",
                        (run_id, state.revision),
                    ).fetchone()
                    if existing is not None and existing["snapshot_json"] == payload:
                        return
                    raise PersistenceConflictError(
                        "state revision was reused with different content"
                    )
                if state.revision != current_revision + 1:
                    raise PersistenceConflictError(
                        "state write is not the next immutable revision"
                    )
            cursor.execute(
                "INSERT INTO agent_state_snapshots "
                "(run_id, revision, snapshot_json) VALUES (?, ?, ?)",
                (run_id, state.revision, payload),
            )
            cursor.execute(
                "INSERT INTO agent_state_current(run_id, revision) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET revision = excluded.revision",
                (run_id, state.revision),
            )

    async def load(self, run_id: UUID) -> AgentState | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshots.snapshot_json "
                "FROM agent_state_current AS current "
                "JOIN agent_state_snapshots AS snapshots "
                "ON snapshots.run_id = current.run_id "
                "AND snapshots.revision = current.revision "
                "WHERE current.run_id = ?",
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        return AgentState.model_validate_json(row["snapshot_json"])

    async def history_for(self, run_id: UUID) -> tuple[AgentState, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT snapshot_json FROM agent_state_snapshots "
                "WHERE run_id = ? ORDER BY revision",
                (str(run_id),),
            ).fetchall()
        return tuple(
            AgentState.model_validate_json(row["snapshot_json"])
            for row in rows
        )


class SQLiteTraceSink:
    """Append trace entries with durable, per-run monotonic sequencing."""

    module_id = "trace.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def record(self, event: RuntimeEvent) -> TraceEntry:
        with self._database.transaction() as cursor:
            existing = cursor.execute(
                "SELECT entry_json FROM runtime_trace WHERE event_id = ?",
                (str(event.event_id),),
            ).fetchone()
            if existing is not None:
                entry = TraceEntry.model_validate_json(existing["entry_json"])
                if entry.event != event:
                    raise PersistenceConflictError(
                        "trace event id was reused with different content"
                    )
                return entry
            row = cursor.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS latest "
                "FROM runtime_trace WHERE run_id = ?",
                (str(event.run_id),),
            ).fetchone()
            sequence = int(row["latest"]) + 1
            entry = TraceEntry(sequence=sequence, event=event)
            cursor.execute(
                "INSERT INTO runtime_trace "
                "(run_id, sequence, event_id, entry_json) VALUES (?, ?, ?, ?)",
                (
                    str(event.run_id),
                    sequence,
                    str(event.event_id),
                    _json_text(entry),
                ),
            )
        return entry

    async def entries_for(self, run_id: UUID) -> tuple[TraceEntry, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT entry_json FROM runtime_trace "
                "WHERE run_id = ? ORDER BY sequence",
                (str(run_id),),
            ).fetchall()
        return tuple(
            TraceEntry.model_validate_json(row["entry_json"])
            for row in rows
        )
