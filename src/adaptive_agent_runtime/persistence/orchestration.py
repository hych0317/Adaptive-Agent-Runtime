"""SQLite adapter for the orchestration checkpoint contract."""

from __future__ import annotations

import json
from uuid import UUID

from adaptive_agent_runtime.orchestration import TaskGraphCheckpoint
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


def _checkpoint_json(checkpoint: TaskGraphCheckpoint) -> str:
    return json.dumps(
        checkpoint.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class SQLiteTaskGraphStore:
    """Persist the graph, in-flight actions, and processed-action cursor."""

    module_id = "orchestration.graph_store.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def save(self, checkpoint: TaskGraphCheckpoint) -> None:
        run_id = str(checkpoint.run_id)
        version = checkpoint.graph.version
        payload = _checkpoint_json(checkpoint)
        with self._database.transaction() as cursor:
            current = cursor.execute(
                "SELECT graph_version, checkpoint_json "
                "FROM task_graph_checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if current is not None:
                current_version = int(current["graph_version"])
                if version < current_version:
                    raise PersistenceConflictError(
                        "task graph checkpoint would move backwards"
                    )
                if version == current_version:
                    if current["checkpoint_json"] == payload:
                        return
                    raise PersistenceConflictError(
                        "task graph version was reused with different content"
                    )
            cursor.execute(
                "INSERT INTO task_graph_history "
                "(run_id, graph_version, state_revision, checkpoint_json) "
                "VALUES (?, ?, ?, ?)",
                (run_id, version, checkpoint.state_revision, payload),
            )
            cursor.execute(
                "INSERT INTO task_graph_checkpoints "
                "(run_id, graph_version, state_revision, checkpoint_json) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "graph_version = excluded.graph_version, "
                "state_revision = excluded.state_revision, "
                "checkpoint_json = excluded.checkpoint_json",
                (run_id, version, checkpoint.state_revision, payload),
            )

    async def load(self, run_id: UUID) -> TaskGraphCheckpoint | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT checkpoint_json FROM task_graph_checkpoints "
                "WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        return TaskGraphCheckpoint.model_validate_json(row["checkpoint_json"])

    async def history_for(
        self,
        run_id: UUID,
    ) -> tuple[TaskGraphCheckpoint, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT checkpoint_json FROM task_graph_history "
                "WHERE run_id = ? ORDER BY graph_version",
                (str(run_id),),
            ).fetchall()
        return tuple(
            TaskGraphCheckpoint.model_validate_json(row["checkpoint_json"])
            for row in rows
        )
