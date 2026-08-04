"""Narrow ports used by the Decision Infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from pydantic import BaseModel

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.decisioning.context import (
    AgentContext,
    ContextProjectionPolicy,
    ProjectionSources,
)
from adaptive_agent_runtime.decisioning.models import (
    AgentCallResult,
    DecisionApplyReceipt,
    DecisionBasis,
    DecisionBudgetUsage,
    DecisionCheckpoint,
    DecisionGovernanceReceipt,
    DecisionProposal,
    DecisionRequest,
    DecisionTraceEvent,
    DecisionValidationOutcome,
    NormalizedDecisionEffect,
    ValidatedDecision,
)


RequestT = TypeVar("RequestT", bound=BaseModel)
ProposalT = TypeVar("ProposalT", bound=BaseModel)
EffectT = TypeVar("EffectT", bound=BaseModel)
ApprovalCoT = TypeVar("ApprovalCoT", covariant=True)
ApprovalContraT = TypeVar("ApprovalContraT", contravariant=True)


@runtime_checkable
class AgentContextBuilder(RuntimeModule, Protocol):
    def build(
        self,
        request: DecisionRequest[Any],
        sources: ProjectionSources,
        policy: ContextProjectionPolicy,
    ) -> AgentContext: ...


@runtime_checkable
class DecisionProposalProducer(RuntimeModule, Protocol[ProposalT]):
    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[ProposalT]: ...


@runtime_checkable
class DecisionBasisProvider(RuntimeModule, Protocol):
    async def current_basis(
        self,
        request: DecisionRequest[Any],
    ) -> DecisionBasis: ...


@runtime_checkable
class DecisionEffectNormalizer(
    RuntimeModule,
    Protocol[RequestT, ProposalT, EffectT],
):
    def normalize(
        self,
        request: DecisionRequest[RequestT],
        proposal: DecisionProposal[ProposalT],
    ) -> NormalizedDecisionEffect[EffectT]: ...


@runtime_checkable
class DecisionValidator(
    RuntimeModule,
    Protocol[RequestT, ProposalT, EffectT],
):
    def validate(
        self,
        request: DecisionRequest[RequestT],
        context: AgentContext,
        proposal: DecisionProposal[ProposalT],
        *,
        current_basis: DecisionBasis,
        usage: DecisionBudgetUsage,
    ) -> DecisionValidationOutcome[EffectT]: ...


@dataclass(frozen=True)
class DecisionGovernanceResolution(Generic[ApprovalCoT]):
    """Serializable receipt plus an opaque, non-Agent approval binding."""

    receipt: DecisionGovernanceReceipt
    approval: ApprovalCoT | None = None


@runtime_checkable
class DecisionGovernancePort(
    RuntimeModule,
    Protocol[RequestT, ProposalT, EffectT, ApprovalCoT],
):
    async def evaluate(
        self,
        decision: ValidatedDecision[RequestT, ProposalT, EffectT],
    ) -> DecisionGovernanceResolution[ApprovalCoT]: ...

    async def resume_review(
        self,
        decision: ValidatedDecision[RequestT, ProposalT, EffectT],
        receipt: DecisionGovernanceReceipt,
    ) -> DecisionGovernanceResolution[ApprovalCoT]: ...


@runtime_checkable
class DecisionApplier(
    RuntimeModule,
    Protocol[RequestT, ProposalT, EffectT, ApprovalContraT],
):
    async def apply(
        self,
        decision: ValidatedDecision[RequestT, ProposalT, EffectT],
        approval: ApprovalContraT,
    ) -> DecisionApplyReceipt: ...


@runtime_checkable
class DecisionTraceWriter(RuntimeModule, Protocol):
    async def record(self, event: DecisionTraceEvent) -> None: ...


@runtime_checkable
class DecisionCheckpointStore(
    RuntimeModule,
    Protocol[RequestT, ProposalT, EffectT],
):
    async def save(
        self,
        checkpoint: DecisionCheckpoint[RequestT, ProposalT, EffectT],
        *,
        expected_revision: int | None,
    ) -> None: ...

    async def load(
        self,
        request_id: UUID,
    ) -> DecisionCheckpoint[RequestT, ProposalT, EffectT] | None: ...
