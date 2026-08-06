"""Legacy Phase 5/6 demo adapters, isolated from Governance dependencies.

These adapters authorize historical deployment operations and exist only for
compatibility tests.  Applications and the default Runtime must not import
this module during Phase 4-A.
"""

from __future__ import annotations

from adaptive_agent_runtime.evaluation.models import OptimizationProposal
from adaptive_agent_runtime.evolution.models import OptimizationApplication
from adaptive_agent_runtime.governance.enforcement import (
    SUBJECT_FINGERPRINT_ATTRIBUTE,
)
from adaptive_agent_runtime.governance.models import (
    ConfidenceSignals,
    GovernanceEvidence,
    GovernanceHistory,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    ImpactAssessment,
    RiskLevel,
    governance_fingerprint,
    stable_governance_id,
)


class LegacyOptimizationApplyGovernanceAdapter:
    module_id = "governance.adapter.legacy_optimization_apply"

    def to_request(
        self,
        proposal: OptimizationProposal,
        *,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
        effective_history = history or GovernanceHistory()
        evidence = tuple(
            GovernanceEvidence(
                evidence_id=f"pattern:{pattern_id}",
                kind="evaluation.failure_pattern",
                source="legacy_agent_evaluation",
                reliability=proposal.proposal_confidence,
                summary="Legacy Failure Pattern supporting a demo proposal.",
            )
            for pattern_id in proposal.source_pattern_ids
        )
        return GovernanceRequest(
            request_id=stable_governance_id(
                self.module_id,
                proposal.proposal_id,
                proposal.model_dump_json(),
            ),
            scope=GovernanceScope.EVOLUTION,
            operation="optimization.apply",
            target=GovernanceTarget(
                target_type="optimization_proposal",
                target_id=str(proposal.proposal_id),
            ),
            risk=RiskLevel.HIGH,
            signals=ConfidenceSignals(
                stated_confidence=proposal.proposal_confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=0.9,
                    reversible=bool(proposal.rollback_plan),
                    description="Legacy Runtime configuration deployment.",
                ),
                history=effective_history,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(proposal),
                "proposal_id": str(proposal.proposal_id),
            },
            requested_at=proposal.created_at,
        )


class LegacyOptimizationRollbackGovernanceAdapter:
    module_id = "governance.adapter.legacy_optimization_rollback"

    def to_request(
        self,
        application: OptimizationApplication,
        *,
        history: GovernanceHistory | None = None,
    ) -> GovernanceRequest:
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
                    description="Legacy configuration rollback.",
                ),
                history=history or GovernanceHistory(),
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(application),
                "application_id": str(application.application_id),
            },
        )

