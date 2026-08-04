"""In-memory Decision checkpoint storage behind a narrow CAS contract."""

from __future__ import annotations

from collections import defaultdict
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


class InMemoryDecisionCheckpointStore(
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
):
    module_id = "decision.checkpoint.in_memory"

    def __init__(self) -> None:
        self._current: dict[
            UUID,
            DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT],
        ] = {}
        self._history: defaultdict[
            UUID,
            list[
                DecisionCheckpoint[
                    RequestPayloadT,
                    ProposalPayloadT,
                    EffectPayloadT,
                ]
            ],
        ] = defaultdict(list)

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
        current = self._current.get(checkpoint.request_id)
        if current is None:
            if expected_revision is not None or checkpoint.revision != 0:
                raise DecisionCheckpointConflictError(
                    "a new decision checkpoint must begin at revision 0"
                )
        else:
            if current == checkpoint:
                return
            if expected_revision is None or current.revision != expected_revision:
                raise DecisionCheckpointConflictError(
                    "decision checkpoint write is based on a stale revision"
                )
            if checkpoint.revision != current.revision + 1:
                raise DecisionCheckpointConflictError(
                    "decision checkpoint write is not the next revision"
                )
        self._current[checkpoint.request_id] = checkpoint
        self._history[checkpoint.request_id].append(checkpoint)

    async def load(
        self,
        request_id: UUID,
    ) -> (
        DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
        | None
    ):
        return self._current.get(request_id)

    def history_for(
        self,
        request_id: UUID,
    ) -> tuple[
        DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT],
        ...,
    ]:
        return tuple(self._history.get(request_id, ()))
