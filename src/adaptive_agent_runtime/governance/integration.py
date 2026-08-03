"""Read-only adapters from Runtime DTOs into Governance requests."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from adaptive_agent_runtime.context_memory.context_models import (
    ContextArchiveReference,
    ContextUnit,
)
from adaptive_agent_runtime.context_memory.memory_models import (
    MemoryCandidate,
    MemoryEvolutionType,
)
from adaptive_agent_runtime.evaluation.models import OptimizationProposal
from adaptive_agent_runtime.evolution.models import OptimizationApplication
from adaptive_agent_runtime.governance.models import (
    ConfidenceSignals,
    GovernanceCorrelation,
    GovernanceEvidence,
    GovernanceHistory,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    ImpactAssessment,
    RiskLevel,
    stable_governance_id,
    governance_fingerprint,
)
from adaptive_agent_runtime.governance.enforcement import (
    SUBJECT_FINGERPRINT_ATTRIBUTE,
)
from adaptive_agent_runtime.orchestration.models import GraphMutation
from adaptive_agent_runtime.orchestration.recovery import RecoveryPlan
from adaptive_agent_runtime.tool_ecosystem.models import ToolInvocation


class ToolGovernanceAdapter:
    module_id = "governance.adapter.tool"

    def to_request(
        self,
        invocation: ToolInvocation,
        *,
        risk: RiskLevel = RiskLevel.LOW,
        privileged: bool = False,
        confidence: float = 1.0,
        impact_score: float = 0.1,
        reversible: bool = True,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        effective_risk = RiskLevel.HIGH if privileged else risk
        effective_history = history or GovernanceHistory()
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                invocation.invocation_id,
                invocation.model_dump_json(),
                effective_risk.value,
                privileged,
                confidence,
                impact_score,
                reversible,
                effective_history.successful_similar,
                effective_history.failed_similar,
                effective_history.prior_denials,
            ),
            scope=GovernanceScope.ACTION,
            operation="tool.call",
            target=GovernanceTarget(
                target_type="tool_provider",
                target_id=invocation.provider_id,
            ),
            risk=effective_risk,
            signals=ConfidenceSignals(
                stated_confidence=confidence,
                impact=ImpactAssessment(
                    score=impact_score,
                    reversible=reversible,
                    description="Potential impact of invoking the Tool Provider.",
                ),
                history=effective_history,
            ),
            correlation=GovernanceCorrelation(
                run_id=invocation.correlation.run_id,
                task_id=invocation.correlation.task_id,
                node_id=invocation.correlation.node_id,
                action_id=invocation.correlation.action_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(invocation),
                "invocation_id": str(invocation.invocation_id),
                "requirement_id": str(invocation.requirement_id),
                "capability_id": invocation.capability_id,
                "provider_id": invocation.provider_id,
                "privileged": privileged,
            },
            requested_at=invocation.requested_at,
        )


class MemoryGovernanceAdapter:
    module_id = "governance.adapter.memory"

    _IMPACT_BY_EVOLUTION = {
        MemoryEvolutionType.SUPPORT: 0.3,
        MemoryEvolutionType.EXTEND: 0.5,
        MemoryEvolutionType.MODIFY: 0.6,
        MemoryEvolutionType.CONFLICT: 0.7,
    }

    def to_request(
        self,
        candidate: MemoryCandidate,
        *,
        run_id: UUID | None = None,
        task_id: UUID | None = None,
        node_id: UUID | None = None,
        risk: RiskLevel = RiskLevel.MEDIUM,
        impact_score: float | None = None,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        resolved_impact = (
            impact_score
            if impact_score is not None
            else self._IMPACT_BY_EVOLUTION[candidate.evolution]
        )
        effective_history = history or GovernanceHistory()
        evidence = tuple(
            GovernanceEvidence(
                evidence_id=str(item.evidence_id),
                kind="memory.evidence",
                source=(
                    str(item.source_context_id)
                    if item.source_context_id is not None
                    else item.source_reference or "unknown"
                ),
                reliability=item.weight,
                summary=item.note,
            )
            for item in candidate.evidence
        )
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                candidate.candidate_id,
                candidate.model_dump_json(),
                risk.value,
                resolved_impact,
                run_id,
                task_id,
                node_id,
                effective_history.successful_similar,
                effective_history.failed_similar,
                effective_history.prior_denials,
            ),
            scope=GovernanceScope.STATE,
            operation="memory.write",
            target=GovernanceTarget(
                target_type="memory_candidate",
                target_id=str(candidate.candidate_id),
            ),
            risk=risk,
            signals=ConfidenceSignals(
                stated_confidence=candidate.confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=resolved_impact,
                    reversible=True,
                    description="Impact of changing long-term Memory.",
                ),
                history=effective_history,
            ),
            correlation=GovernanceCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(candidate),
                "candidate_id": str(candidate.candidate_id),
                "memory_key": candidate.memory_key,
                "evolution": candidate.evolution.value,
                "target_memory_id": (
                    str(candidate.target_memory_id)
                    if candidate.target_memory_id is not None
                    else None
                ),
                "evidence_count": len(candidate.evidence),
            },
            requested_at=candidate.created_at,
        )


class OptimizationGovernanceAdapter:
    module_id = "governance.adapter.optimization"

    def to_request(
        self,
        proposal: OptimizationProposal,
        *,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        risk = RiskLevel.HIGH
        effective_history = history or GovernanceHistory()
        evidence = tuple(
            GovernanceEvidence(
                evidence_id=f"pattern:{pattern_id}",
                kind="evaluation.failure_pattern",
                source="agent_evaluation",
                reliability=proposal.proposal_confidence,
                summary="Failure Pattern supporting an Optimization Proposal.",
            )
            for pattern_id in proposal.source_pattern_ids
        ) + tuple(
            GovernanceEvidence(
                evidence_id=f"candidate:{candidate_id}",
                kind="evaluation.optimization_candidate",
                source="agent_evaluation",
                reliability=proposal.proposal_confidence,
                summary="Optimization Candidate supporting the Proposal.",
            )
            for candidate_id in proposal.source_candidate_ids
        )
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                proposal.proposal_id,
                proposal.model_dump_json(),
                risk.value,
                effective_history.successful_similar,
                effective_history.failed_similar,
                effective_history.prior_denials,
            ),
            scope=GovernanceScope.EVOLUTION,
            operation="optimization.apply",
            target=GovernanceTarget(
                target_type="optimization_proposal",
                target_id=str(proposal.proposal_id),
            ),
            risk=risk,
            signals=ConfidenceSignals(
                stated_confidence=proposal.proposal_confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=0.9,
                    reversible=bool(proposal.rollback_plan),
                    description="Impact of changing Runtime behavior or policy.",
                ),
                history=effective_history,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(proposal),
                "proposal_id": str(proposal.proposal_id),
                "target_component": proposal.target_component.value,
                "change_kind": proposal.change_kind,
                "source_pattern_ids": [
                    str(item) for item in proposal.source_pattern_ids
                ],
                "source_candidate_ids": [
                    str(item) for item in proposal.source_candidate_ids
                ],
            },
            requested_at=proposal.created_at,
        )


class OptimizationRollbackGovernanceAdapter:
    """Govern rollback as a separate, one-shot Evolution mutation."""

    module_id = "governance.adapter.optimization_rollback"

    def to_request(
        self,
        application: OptimizationApplication,
        *,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        effective_history = history or GovernanceHistory()
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                application.application_id,
                application.revision,
                governance_fingerprint(application),
            ),
            scope=GovernanceScope.EVOLUTION,
            operation="optimization.rollback",
            target=GovernanceTarget(
                target_type="optimization_application",
                target_id=str(application.application_id),
            ),
            risk=RiskLevel.HIGH,
            signals=ConfidenceSignals(
                stated_confidence=1.0,
                impact=ImpactAssessment(
                    score=0.7,
                    reversible=True,
                    description="Restore the replayed configuration baseline.",
                ),
                history=effective_history,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(application),
                "application_id": str(application.application_id),
                "candidate_version": application.deployment.candidate.version,
                "baseline_version": application.deployment.baseline.version,
            },
        )


class GraphMutationGovernanceAdapter:
    """Translate one validated graph-mutation proposal into a state request."""

    module_id = "governance.adapter.graph_mutation"

    def to_request(
        self,
        mutation: GraphMutation,
        *,
        run_id: UUID,
        task_id: UUID,
        node_id: UUID | None = None,
        action_id: UUID | None = None,
        risk: RiskLevel = RiskLevel.MEDIUM,
        confidence: float = 0.8,
        evidence: tuple[GovernanceEvidence, ...] = (),
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        effective_history = history or GovernanceHistory()
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                mutation.mutation_id,
                mutation.model_dump_json(),
                run_id,
                task_id,
                node_id,
                action_id,
                risk.value,
                confidence,
                *[item.evidence_id for item in evidence],
            ),
            scope=GovernanceScope.STATE,
            operation="graph.mutate",
            target=GovernanceTarget(
                target_type="task_graph_mutation",
                target_id=str(mutation.mutation_id),
            ),
            risk=risk,
            signals=ConfidenceSignals(
                stated_confidence=confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=0.55,
                    reversible=True,
                    description="Impact of changing the active Task Graph.",
                ),
                history=effective_history,
            ),
            correlation=GovernanceCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=action_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(mutation),
                "mutation_id": str(mutation.mutation_id),
                "mutation_type": mutation.mutation_type.value,
            },
        )


class ContextGovernanceOperation(StrEnum):
    COMPRESS = "context.compress"
    ARCHIVE = "context.archive"
    RESTORE = "context.restore"


class ContextGovernanceAdapter:
    """Create requests for semantic Context lifecycle state changes."""

    module_id = "governance.adapter.context"

    def to_request(
        self,
        subject: ContextUnit | ContextArchiveReference,
        *,
        operation: ContextGovernanceOperation,
        run_id: UUID,
        task_id: UUID | None = None,
        node_id: UUID | None = None,
        risk: RiskLevel | None = None,
        confidence: float = 1.0,
        evidence: tuple[GovernanceEvidence, ...] = (),
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        context_id = subject.context_id
        effective_risk = risk or (
            RiskLevel.LOW
            if operation in {
                ContextGovernanceOperation.COMPRESS,
                ContextGovernanceOperation.RESTORE,
            }
            else RiskLevel.MEDIUM
        )
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                operation.value,
                governance_fingerprint(subject),
                run_id,
                task_id,
                node_id,
                effective_risk.value,
            ),
            scope=GovernanceScope.STATE,
            operation=operation.value,
            target=GovernanceTarget(
                target_type="context_unit",
                target_id=str(context_id),
            ),
            risk=effective_risk,
            signals=ConfidenceSignals(
                stated_confidence=confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=(0.3 if effective_risk is RiskLevel.LOW else 0.55),
                    reversible=True,
                    description="Impact of a semantic Context lifecycle change.",
                ),
                history=history or GovernanceHistory(),
            ),
            correlation=GovernanceCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(subject),
                "context_id": str(context_id),
                "lifecycle_operation": operation.value,
            },
        )


class RecoveryGovernanceAdapter:
    """Govern failure-driven graph recovery as an adaptive state change."""

    module_id = "governance.adapter.recovery"

    def to_request(
        self,
        plan: RecoveryPlan,
        *,
        run_id: UUID,
        task_id: UUID,
        action_id: UUID,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        evidence = tuple(
            GovernanceEvidence(
                evidence_id=f"failure:{plan.analysis.analysis_id}:{index}",
                kind="orchestration.failure_analysis",
                source="failure_classifier",
                reliability=plan.analysis.confidence,
                summary=item,
            )
            for index, item in enumerate(plan.analysis.evidence, start=1)
        ) + (
            GovernanceEvidence(
                evidence_id=f"recovery-plan:{plan.plan_id}",
                kind="orchestration.recovery_plan",
                source="failure_driven_replanner",
                reliability=1.0,
                summary=(
                    f"Bounded attempt {plan.attempt_number}/{plan.max_attempts}."
                ),
            ),
        )
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                plan.plan_id,
                governance_fingerprint(plan),
                run_id,
                task_id,
                action_id,
            ),
            scope=GovernanceScope.STATE,
            operation="graph.recover",
            target=GovernanceTarget(
                target_type="task_node",
                target_id=str(plan.analysis.node_id),
            ),
            risk=RiskLevel.MEDIUM,
            signals=ConfidenceSignals(
                stated_confidence=plan.analysis.confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=0.6,
                    reversible=True,
                    description="Impact of modifying the graph after failure.",
                ),
                history=history or GovernanceHistory(),
            ),
            correlation=GovernanceCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=plan.analysis.node_id,
                action_id=action_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(plan),
                "plan_id": str(plan.plan_id),
                "failure_kind": plan.analysis.kind.value,
                "attempt_number": plan.attempt_number,
                "max_attempts": plan.max_attempts,
                "actions": [item.action_type.value for item in plan.actions],
            },
        )
