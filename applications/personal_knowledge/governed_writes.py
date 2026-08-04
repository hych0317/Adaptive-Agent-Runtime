"""Governed apply-point for already human-confirmed knowledge changes."""

from __future__ import annotations

from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    ConfidenceSignals,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceEvidence,
    GovernanceHistory,
    GovernancePolicy,
    GovernanceRequest,
    GovernanceRule,
    GovernanceScope,
    GovernanceTarget,
    GovernedOperationExecutor,
    ImpactAssessment,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RiskLevel,
    RuleEffect,
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    StrictAuthorizationVerifier,
    governance_fingerprint,
)

from applications.personal_knowledge.contracts import KnowledgeEntryStore
from applications.personal_knowledge.models import (
    ConfirmedKnowledgeChange,
    KnowledgeEntry,
)


KNOWLEDGE_APPLY_OPERATION = "personal_knowledge.apply_confirmed_change"


class GovernedKnowledgeWriter:
    """Enforce exact-payload authorization at the database apply point."""

    def __init__(self, store: KnowledgeEntryStore) -> None:
        self._store = store
        policy = GovernancePolicy(
            policy_id="personal-knowledge-confirmed-write",
            version="1",
            rules=(
                GovernanceRule(
                    rule_id="allow-explicitly-confirmed-knowledge-change",
                    description=(
                        "Allow only the application operation whose subject is an "
                        "immutable user-confirmed change."
                    ),
                    effect=RuleEffect.ALLOW,
                    scopes=(GovernanceScope.STATE,),
                    operations=(KNOWLEDGE_APPLY_OPERATION,),
                    risk_levels=(RiskLevel.LOW,),
                    priority=100,
                ),
            ),
        )
        self._evaluator = RuntimeGovernanceEvaluator(
            policy=policy,
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=InMemoryHumanReviewService(),
        )
        self._issuer = GovernanceAuthorizationIssuer()
        self._executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=InMemoryAuthorizationConsumptionStore(),
        )

    async def apply(self, change: ConfirmedKnowledgeChange) -> KnowledgeEntry:
        request = GovernanceRequest(
            scope=GovernanceScope.STATE,
            operation=KNOWLEDGE_APPLY_OPERATION,
            target=GovernanceTarget(
                target_type="knowledge_confirmation",
                target_id=str(change.confirmation_id),
            ),
            risk=RiskLevel.LOW,
            signals=ConfidenceSignals(
                stated_confidence=1.0,
                evidence=(
                    GovernanceEvidence(
                        evidence_id=str(change.confirmation_id),
                        kind="human.ui_confirmation",
                        source="personal_knowledge.confirmation_dialog",
                        reliability=1.0,
                        summary="User confirmed the exact editable payload.",
                    ),
                ),
                impact=ImpactAssessment(
                    score=0.3,
                    reversible=True,
                    description="Versioned knowledge write with recoverable trash.",
                ),
                history=GovernanceHistory(),
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(change),
                "confirmation_id": str(change.confirmation_id),
                "change_type": change.change_type.value,
            },
            requested_at=change.confirmed_at,
        )
        decision = self._evaluator.evaluate(request)
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise PermissionError(f"knowledge write denied: {decision.reason}")
        authorization = self._issuer.issue(request, decision)

        async def persist() -> KnowledgeEntry:
            return await self._store.apply_knowledge_change(change)

        return await self._executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="personal_knowledge.confirmed_write",
                operation=KNOWLEDGE_APPLY_OPERATION,
                target=request.target,
                subject=change,
                apply=persist,
            ),
        )
