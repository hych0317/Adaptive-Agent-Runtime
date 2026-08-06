"""Narrow read and Agent ports for Phase 4-A Optimization assessment."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from adaptive_agent_runtime.decisioning import AgentCallResult, AgentContext
from adaptive_agent_runtime.optimization.models import (
    OptimizationApplyCommitReceipt,
    OptimizationApplyEffect,
    OptimizationEvidenceBinding,
    OptimizationProposalDraft,
    OptimizationProposal,
    OptimizationRollbackEffect,
    OptimizationScope,
    OptimizationTarget,
    OptimizationTargetKey,
    RuntimeConfigurationSnapshot,
)
from adaptive_agent_runtime.governance import GovernanceTarget, RuntimeCommitPermit


class OptimizationEvidenceResolver(Protocol):
    async def resolve(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationEvidenceBinding, ...]: ...


class OptimizationBaselineProvider(Protocol):
    async def targets(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationTarget, ...]: ...


class OptimizationProposalProducer(Protocol):
    module_id: str

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[OptimizationProposalDraft]: ...


class OptimizationProposalQuery(Protocol):
    async def load_by_id(self, proposal_id: UUID) -> OptimizationProposal | None: ...

    async def list_for_scope(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationProposal, ...]: ...

    async def verify_phase3_provenance(self, proposal_id: UUID) -> bool: ...


class RuntimeConfigurationQuery(Protocol):
    async def load_active(
        self,
        scope: OptimizationScope,
        target_key: OptimizationTargetKey,
    ) -> RuntimeConfigurationSnapshot | None: ...

    async def load_snapshot(
        self,
        scope: OptimizationScope,
        target_key: OptimizationTargetKey,
        revision: int,
    ) -> RuntimeConfigurationSnapshot | None: ...

    async def was_proposal_applied(self, proposal_id: UUID) -> bool: ...


class GovernedRuntimeConfigurationCommitPort(RuntimeConfigurationQuery, Protocol):
    async def commit_apply(
        self,
        effect: OptimizationApplyEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> OptimizationApplyCommitReceipt: ...

    async def commit_rollback(
        self,
        effect: OptimizationRollbackEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> OptimizationApplyCommitReceipt: ...

    async def load_receipt(
        self,
        effect_fingerprint: str,
    ) -> OptimizationApplyCommitReceipt | None: ...
