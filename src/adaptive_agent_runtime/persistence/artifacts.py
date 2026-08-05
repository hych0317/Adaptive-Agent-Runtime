"""Durable, fingerprint-idempotent workspace artifact commits."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class WorkspaceArtifactCommitReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID
    node_id: UUID
    artifact_type: str = Field(min_length=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: tuple[str, ...] = Field(min_length=1)
    committed_at: AwareDatetime


class SQLiteWorkspaceArtifactStore:
    """The mutation method is intentionally private to Runtime committers."""

    module_id = "workspace.artifact_store.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def _commit(
        self,
        *,
        run_id: UUID,
        node_id: UUID,
        artifact_type: str,
        effect_fingerprint: str,
        artifact: JsonValue,
        provenance: tuple[str, ...],
        committed_at: datetime | None = None,
    ) -> WorkspaceArtifactCommitReceipt:
        receipt = WorkspaceArtifactCommitReceipt(
            run_id=run_id,
            node_id=node_id,
            artifact_type=artifact_type,
            effect_fingerprint=effect_fingerprint,
            artifact_fingerprint=decision_fingerprint(artifact),
            provenance=provenance,
            committed_at=committed_at or datetime.now(timezone.utc),
        )
        artifact_json = json.dumps(
            artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        receipt_json = receipt.model_dump_json()
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT artifact_json, receipt_json FROM workspace_artifacts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                existing = WorkspaceArtifactCommitReceipt.model_validate_json(
                    prior["receipt_json"]
                )
                if prior["artifact_json"] != artifact_json or existing != receipt:
                    if (
                        prior["artifact_json"] == artifact_json
                        and existing.model_copy(update={"committed_at": receipt.committed_at})
                        == receipt
                    ):
                        return existing
                    raise PersistenceConflictError(
                        "artifact effect fingerprint was reused"
                    )
                return existing
            target = cursor.execute(
                "SELECT effect_fingerprint FROM workspace_artifacts "
                "WHERE run_id = ? AND node_id = ? AND artifact_type = ?",
                (str(run_id), str(node_id), artifact_type),
            ).fetchone()
            if target is not None:
                raise PersistenceConflictError(
                    "workspace artifact target already has another effect"
                )
            cursor.execute(
                "INSERT INTO workspace_artifacts "
                "(run_id, node_id, artifact_type, effect_fingerprint, "
                "artifact_json, receipt_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(run_id),
                    str(node_id),
                    artifact_type,
                    effect_fingerprint,
                    artifact_json,
                    receipt_json,
                ),
            )
        return receipt

    def load_by_effect(
        self,
        effect_fingerprint: str,
    ) -> tuple[JsonValue, WorkspaceArtifactCommitReceipt] | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT artifact_json, receipt_json FROM workspace_artifacts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        return (
            json.loads(row["artifact_json"]),
            WorkspaceArtifactCommitReceipt.model_validate_json(row["receipt_json"]),
        )

    def load(
        self,
        *,
        run_id: UUID,
        node_id: UUID,
        artifact_type: str,
    ) -> tuple[JsonValue, WorkspaceArtifactCommitReceipt] | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT artifact_json, receipt_json FROM workspace_artifacts "
                "WHERE run_id = ? AND node_id = ? AND artifact_type = ?",
                (str(run_id), str(node_id), artifact_type),
            ).fetchone()
        if row is None:
            return None
        return (
            json.loads(row["artifact_json"]),
            WorkspaceArtifactCommitReceipt.model_validate_json(row["receipt_json"]),
        )


class WorkspaceArtifactCommitter:
    """Narrow commit capability handed only to governed Decision handlers."""

    module_id = "workspace.artifact_committer"

    def __init__(self, store: SQLiteWorkspaceArtifactStore) -> None:
        self._store = store

    def commit(self, **values: object) -> WorkspaceArtifactCommitReceipt:
        return self._store._commit(**values)  # type: ignore[arg-type]

    def load_by_effect(
        self, effect_fingerprint: str
    ) -> tuple[JsonValue, WorkspaceArtifactCommitReceipt] | None:
        return self._store.load_by_effect(effect_fingerprint)
