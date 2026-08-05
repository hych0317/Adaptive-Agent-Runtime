"""Thin adapters from validated Decision effects to Runtime Governance."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Generic

from pydantic import BaseModel, JsonValue


from adaptive_agent_runtime.decisioning.contracts import (
    DecisionGovernanceResolution,
)
from adaptive_agent_runtime.decisioning.errors import DecisionInvariantError
from adaptive_agent_runtime.decisioning.models import (
    DecisionApplyReceipt,
    DecisionGovernanceOutcome,
    DecisionGovernanceReceipt,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    EffectPayloadT,
    ProposalPayloadT,
    RequestPayloadT,
    NormalizedDecisionEffect,
    ValidatedDecision,
)
from adaptive_agent_runtime.governance.contracts import (
    AuthorizationIssuer,
    GovernanceEvaluator,
    HumanReviewService,
)
from adaptive_agent_runtime.governance.enforcement import (
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    BoundGovernedOperation,
    GovernedOperationExecutor,
)
from adaptive_agent_runtime.governance.models import (
    ConfidenceSignals,
    DecisionOutcome,
    GovernanceAuthorization,
    GovernanceCorrelation,
    GovernanceDecision,
    GovernanceEvidence,
    GovernanceHistory,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    ImpactAssessment,
    RiskLevel,
    governance_fingerprint,
    stable_governance_id,
    utc_now,
)


@dataclass(frozen=True)
class DecisionGovernanceBinding:
    """Opaque authorization objects never exposed to the producing Agent."""

    request: GovernanceRequest
    decision: GovernanceDecision
    authorization: GovernanceAuthorization


class RuntimeDecisionGovernanceAdapter(
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
):
    module_id = "governance.adapter.decision"

    def __init__(
        self,
        *,
        evaluator: GovernanceEvaluator,
        authorization_issuer: AuthorizationIssuer,
        review_service: HumanReviewService,
    ) -> None:
        self._evaluator = evaluator
        self._authorization_issuer = authorization_issuer
        self._review_service = review_service

    async def evaluate(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
    ) -> DecisionGovernanceResolution[DecisionGovernanceBinding]:
        request = self._request_for(decision)
        governance_decision = self._evaluator.evaluate(request)
        return self._resolution(request, governance_decision)

    async def resume_review(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        receipt: DecisionGovernanceReceipt,
    ) -> DecisionGovernanceResolution[DecisionGovernanceBinding]:
        if receipt.review_request_id is None:
            raise DecisionInvariantError("decision receipt has no Human Review")
        request = self._request_for(decision)
        if receipt.governance_request_id != request.request_id:
            raise DecisionInvariantError(
                "persisted Governance receipt belongs to another request"
            )
        review = self._review_service.get(receipt.review_request_id)
        if review is None:
            raise DecisionInvariantError("Human Review request no longer exists")
        if review.status.value == "pending":
            return DecisionGovernanceResolution(receipt=receipt)
        final = self._evaluator.finalize_review(request, review)
        return self._resolution(request, final)

    def _request_for(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
    ) -> GovernanceRequest:
        request = decision.request
        proposal = decision.proposal
        effect = decision.normalized_effect
        evidence_by_id = {item.evidence_id: item for item in request.evidence}
        evidence = tuple(
            GovernanceEvidence(
                evidence_id=item.evidence_id,
                kind=item.kind,
                source=item.source,
                reliability=item.reliability,
                summary=item.summary,
            )
            for evidence_id in proposal.evidence_refs
            if (item := evidence_by_id.get(evidence_id)) is not None
        )
        request_id = stable_governance_id(
            "decision-lifecycle-governance",
            request.request_id,
            proposal.proposal_id,
            effect.effect_fingerprint,
        )
        return GovernanceRequest(
            request_id=request_id,
            scope=GovernanceScope(effect.governance_scope.value),
            operation=effect.operation,
            target=GovernanceTarget(
                target_type=effect.target.target_type,
                target_id=effect.target.target_id,
            ),
            risk=RiskLevel(effect.risk.value),
            signals=ConfidenceSignals(
                stated_confidence=proposal.confidence,
                evidence=evidence,
                impact=ImpactAssessment(
                    score=effect.impact_score,
                    reversible=effect.reversible,
                    description=effect.impact_description,
                ),
                history=GovernanceHistory(),
            ),
            correlation=GovernanceCorrelation(
                run_id=request.correlation.run_id,
                task_id=request.correlation.task_id,
                node_id=request.correlation.node_id,
                action_id=request.correlation.action_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(effect),
                "decision_request_id": str(request.request_id),
                "decision_proposal_id": str(proposal.proposal_id),
                "decision_validation_id": str(decision.validation.validation_id),
                "decision_effect_fingerprint": effect.effect_fingerprint,
            },
            requested_at=decision.validation.created_at,
        )

    def _resolution(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
    ) -> DecisionGovernanceResolution[DecisionGovernanceBinding]:
        authorization: GovernanceAuthorization | None = None
        if decision.outcome is DecisionOutcome.ALLOW:
            authorization = self._authorization_issuer.issue(request, decision)
        receipt = DecisionGovernanceReceipt(
            outcome=DecisionGovernanceOutcome(decision.outcome.value),
            governance_request_id=request.request_id,
            governance_decision_id=decision.decision_id,
            authorization_id=(
                authorization.authorization_id
                if authorization is not None
                else None
            ),
            review_request_id=decision.review_request_id,
            reason=decision.reason,
            decided_at=decision.decided_at,
            approval_snapshot=(
                {
                    "request": request.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                    "authorization": authorization.model_dump(mode="json"),
                }
                if authorization is not None
                else None
            ),
        )
        binding = (
            DecisionGovernanceBinding(
                request=request,
                decision=decision,
                authorization=authorization,
            )
            if authorization is not None
            else None
        )
        return DecisionGovernanceResolution(receipt=receipt, approval=binding)

    def restore_approval(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        receipt: DecisionGovernanceReceipt,
    ) -> DecisionGovernanceBinding:
        snapshot = receipt.approval_snapshot
        if snapshot is None:
            raise DecisionInvariantError("Governance approval snapshot is missing")
        request = GovernanceRequest.model_validate(snapshot.get("request"))
        governance_decision = GovernanceDecision.model_validate(
            snapshot.get("decision")
        )
        authorization = GovernanceAuthorization.model_validate(
            snapshot.get("authorization")
        )
        expected = self._request_for(decision)
        if request != expected:
            raise DecisionInvariantError("restored Governance request is stale")
        if (
            governance_decision.decision_id != receipt.governance_decision_id
            or authorization.authorization_id != receipt.authorization_id
        ):
            raise DecisionInvariantError("restored Governance binding identity mismatch")
        return DecisionGovernanceBinding(request, governance_decision, authorization)


class GovernedDecisionApplier(Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT]):
    """Apply exactly the normalized effect authorized by Governance."""

    module_id = "governance.applier.decision"

    def __init__(
        self,
        *,
        executor: GovernedOperationExecutor,
        apply_effect: Callable[[EffectPayloadT], Awaitable[JsonValue]],
        apply_normalized_effect: Callable[
            [NormalizedDecisionEffect[EffectPayloadT]], Awaitable[JsonValue]
        ] | None = None,
        reconcile_effect: Callable[
            [NormalizedDecisionEffect[EffectPayloadT]], Awaitable[DecisionReconciliation]
        ] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._executor = executor
        self._apply_effect = apply_effect
        self._apply_normalized_effect = apply_normalized_effect
        self._reconcile_effect = reconcile_effect
        self._clock = clock

    async def apply(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        approval: DecisionGovernanceBinding,
    ) -> DecisionApplyReceipt:
        effect = decision.normalized_effect

        async def apply_bound_effect() -> JsonValue:
            if self._apply_normalized_effect is not None:
                return await self._apply_normalized_effect(effect)
            return await self._apply_effect(effect.payload)

        result = await self._executor.execute(
            request=approval.request,
            decision=approval.decision,
            authorization=approval.authorization,
            target=BoundGovernedOperation(
                module_id=self.module_id,
                operation=effect.operation,
                target=GovernanceTarget(
                    target_type=effect.target.target_type,
                    target_id=effect.target.target_id,
                ),
                subject=effect,
                apply=apply_bound_effect,
            ),
        )
        return DecisionApplyReceipt(
            effect_fingerprint=effect.effect_fingerprint,
            committed_state_fingerprint=governance_fingerprint(result),
            result=result,
            applied_at=self._clock(),
        )

    async def resume_apply(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        approval: DecisionGovernanceBinding,
    ) -> DecisionApplyReceipt:
        effect = decision.normalized_effect

        async def apply_bound_effect() -> JsonValue:
            if self._apply_normalized_effect is not None:
                return await self._apply_normalized_effect(effect)
            return await self._apply_effect(effect.payload)

        result = await self._executor.resume_reserved(
            request=approval.request,
            decision=approval.decision,
            authorization=approval.authorization,
            target=BoundGovernedOperation(
                module_id=self.module_id,
                operation=effect.operation,
                target=GovernanceTarget(
                    target_type=effect.target.target_type,
                    target_id=effect.target.target_id,
                ),
                subject=effect,
                apply=apply_bound_effect,
            ),
        )
        return DecisionApplyReceipt(
            effect_fingerprint=effect.effect_fingerprint,
            committed_state_fingerprint=governance_fingerprint(result),
            result=result,
            applied_at=self._clock(),
        )

    async def reconcile(
        self,
        decision: ValidatedDecision[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        approval: DecisionGovernanceBinding,
    ) -> DecisionReconciliation:
        if self._reconcile_effect is None:
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.UNKNOWN,
                reason="the effect has no authoritative reconciliation reader",
            )
        reconciled = await self._reconcile_effect(decision.normalized_effect)
        if (
            reconciled.status is DecisionReconciliationStatus.COMMITTED
            and reconciled.apply_receipt is not None
        ):
            await self._executor.reconcile_committed(
                request=approval.request,
                decision=approval.decision,
                authorization=approval.authorization,
                result=reconciled.apply_receipt.result,
            )
        return reconciled
