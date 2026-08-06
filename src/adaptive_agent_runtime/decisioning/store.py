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
    DecisionTransition,
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
        self._transitions: defaultdict[UUID, list[DecisionTransition]] = defaultdict(
            list
        )

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
        transition = DecisionTransition(
            request_id=checkpoint.request_id,
            from_revision=-1 if current is None else current.revision,
            to_revision=checkpoint.revision,
            from_stage=None if current is None else current.stage,
            to_stage=checkpoint.stage,
            occurred_at=checkpoint.updated_at,
            request_ref=str(checkpoint.request_id),
            proposal_ref=(
                str(checkpoint.proposal.proposal_id)
                if checkpoint.proposal is not None
                else None
            ),
            validation_ref=(
                str(checkpoint.validation.validation_id)
                if checkpoint.validation is not None
                else None
            ),
            effect_ref=(
                checkpoint.validated_decision.normalized_effect.effect_fingerprint
                if checkpoint.validated_decision is not None
                else None
            ),
            governance_receipt_ref=(
                str(checkpoint.governance_receipt.governance_decision_id)
                if checkpoint.governance_receipt is not None
                else None
            ),
            authorization_id=(
                checkpoint.governance_receipt.authorization_id
                if checkpoint.governance_receipt is not None
                else None
            ),
            commit_receipt_ref=(
                checkpoint.commit_receipt.effect_fingerprint
                if checkpoint.commit_receipt is not None
                else None
            ),
            result_ref=(
                str(checkpoint.result.result_id)
                if checkpoint.result is not None
                else None
            ),
            result_status=(
                checkpoint.result.status if checkpoint.result is not None else None
            ),
        )
        self._current[checkpoint.request_id] = checkpoint
        self._transitions[checkpoint.request_id].append(transition)

    async def load(
        self,
        request_id: UUID,
    ) -> (
        DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
        | None
    ):
        return self._current.get(request_id)

    def transitions_for(
        self,
        request_id: UUID,
    ) -> tuple[DecisionTransition, ...]:
        return tuple(self._transitions.get(request_id, ()))
