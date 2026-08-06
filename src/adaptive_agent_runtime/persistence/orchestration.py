"""SQLite adapter for the orchestration checkpoint contract."""

from __future__ import annotations

import json
from uuid import UUID

from adaptive_agent_runtime.orchestration import TaskGraphCheckpoint
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance.contracts import CommitPermitValidation
from adaptive_agent_runtime.governance.errors import AuthorizationVerificationError
from adaptive_agent_runtime.governance.models import GovernanceTarget, RuntimeCommitPermit
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

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation | None = None,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def save(
        self,
        checkpoint: TaskGraphCheckpoint,
        *,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> None:
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
                current_checkpoint = TaskGraphCheckpoint.model_validate_json(
                    current["checkpoint_json"]
                )
                if checkpoint.checkpoint_revision < current_checkpoint.checkpoint_revision:
                    raise PersistenceConflictError(
                        "task graph checkpoint would move backwards"
                    )
                if checkpoint.checkpoint_revision == current_checkpoint.checkpoint_revision:
                    if current["checkpoint_json"] == payload:
                        return
                    raise PersistenceConflictError(
                        "task graph checkpoint revision was reused with different content"
                    )
            structural_change = current is None
            if current is not None:
                structural_change = _graph_structure_fingerprint(
                    current_checkpoint
                ) != _graph_structure_fingerprint(checkpoint)
            if structural_change and self._permit_verifier is not None:
                if permit is None or target is None or subject_fingerprint is None:
                    raise AuthorizationVerificationError(
                        "structural Graph commit requires a Runtime Permit"
                    )
                await self._permit_verifier.verify(
                    permit,
                    operation=(
                        permit.operation
                        if permit.operation
                        in {"graph.initialize", "graph.mutate", "recovery.apply"}
                        else "graph.invalid_operation"
                    ),
                    target=target,
                    subject_fingerprint=subject_fingerprint,
                )
            cursor.execute(
                "INSERT INTO task_graph_checkpoint_journal "
                "(run_id, checkpoint_revision, graph_version, state_revision, checkpoint_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    checkpoint.checkpoint_revision,
                    version,
                    checkpoint.state_revision,
                    payload,
                ),
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
                "SELECT checkpoint_json FROM task_graph_checkpoint_journal "
                "WHERE run_id = ? ORDER BY checkpoint_revision",
                (str(run_id),),
            ).fetchall()
        return tuple(
            TaskGraphCheckpoint.model_validate_json(row["checkpoint_json"])
            for row in rows
        )


def _graph_structure_fingerprint(checkpoint: TaskGraphCheckpoint) -> str:
    return decision_fingerprint(
        {
            "graph_id": checkpoint.graph.graph_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "goal": node.goal,
                    "dependencies": node.dependencies,
                    "expected_output": node.expected_output,
                    "strategy_id": node.strategy_id,
                }
                for node in checkpoint.graph.nodes
            ],
        }
    )
