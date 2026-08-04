"""Durable CAS checkpoints for one configured typed decision lifecycle."""

from __future__ import annotations

import json
from typing import Generic
from uuid import UUID

from adaptive_agent_runtime.decisioning.errors import (
    DecisionCheckpointConflictError,
)
from adaptive_agent_runtime.decisioning.models import (
    DecisionCheckpoint,
    EffectPayloadT,
    ProposalPayloadT,
    RequestPayloadT,
)
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class SQLiteDecisionCheckpointStore(
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
):
    """Persist a single typed decision definition without a global registry."""

    module_id = "decision.checkpoint.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        checkpoint_type: type[
            DecisionCheckpoint[
                RequestPayloadT,
                ProposalPayloadT,
                EffectPayloadT,
            ]
        ],
    ) -> None:
        self._database = database
        self._checkpoint_type = checkpoint_type

    async def save(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        *,
        expected_revision: int | None,
    ) -> None:
        payload = json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        request_id = str(checkpoint.request_id)
        with self._database.transaction() as cursor:
            current = cursor.execute(
                "SELECT revision FROM decision_checkpoint_current "
                "WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if current is None:
                if expected_revision is not None or checkpoint.revision != 0:
                    raise DecisionCheckpointConflictError(
                        "a new decision checkpoint must begin at revision 0"
                    )
            else:
                current_revision = int(current["revision"])
                if current_revision == checkpoint.revision:
                    existing = cursor.execute(
                        "SELECT checkpoint_json FROM decision_checkpoints "
                        "WHERE request_id = ? AND revision = ?",
                        (request_id, checkpoint.revision),
                    ).fetchone()
                    if existing is not None and existing["checkpoint_json"] == payload:
                        return
                    raise DecisionCheckpointConflictError(
                        "decision checkpoint revision was reused with different content"
                    )
                if expected_revision is None or current_revision != expected_revision:
                    raise DecisionCheckpointConflictError(
                        "decision checkpoint write is based on a stale revision"
                    )
                if checkpoint.revision != current_revision + 1:
                    raise DecisionCheckpointConflictError(
                        "decision checkpoint write is not the next revision"
                    )
            cursor.execute(
                "INSERT INTO decision_checkpoints "
                "(request_id, revision, run_id, stage, checkpoint_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    request_id,
                    checkpoint.revision,
                    str(checkpoint.run_id),
                    checkpoint.stage.value,
                    payload,
                ),
            )
            cursor.execute(
                "INSERT INTO decision_checkpoint_current(request_id, revision) "
                "VALUES (?, ?) ON CONFLICT(request_id) DO UPDATE SET "
                "revision = excluded.revision",
                (request_id, checkpoint.revision),
            )

    async def load(
        self,
        request_id: UUID,
    ) -> (
        DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
        | None
    ):
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT checkpoints.checkpoint_json "
                "FROM decision_checkpoint_current AS current "
                "JOIN decision_checkpoints AS checkpoints "
                "ON checkpoints.request_id = current.request_id "
                "AND checkpoints.revision = current.revision "
                "WHERE current.request_id = ?",
                (str(request_id),),
            ).fetchone()
        if row is None:
            return None
        return self._checkpoint_type.model_validate_json(row["checkpoint_json"])

    async def history_for(
        self,
        request_id: UUID,
    ) -> tuple[
        DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT],
        ...,
    ]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT checkpoint_json FROM decision_checkpoints "
                "WHERE request_id = ? ORDER BY revision",
                (str(request_id),),
            ).fetchall()
        return tuple(
            self._checkpoint_type.model_validate_json(row["checkpoint_json"])
            for row in rows
        )
