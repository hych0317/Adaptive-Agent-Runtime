from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import Field

from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionApplyReceipt,
    DecisionBasis,
    DecisionBudget,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionGovernanceOutcome,
    DecisionGovernanceReceipt,
    DecisionGovernanceResolution,
    DecisionGovernanceScope,
    DecisionModel,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionRiskLevel,
    DecisionTarget,
    DecisionTraceEvent,
    NormalizedDecisionEffect,
    ProjectionSource,
    ProjectionSources,
    ValidatedDecision,
    decision_fingerprint,
)


NOW = datetime(2026, 8, 5, 1, 0, tzinfo=timezone.utc)


class FakeRequestPayload(DecisionModel):
    goal: str = Field(min_length=1)


class FakeProposalPayload(DecisionModel):
    value: int


class FakeEffectPayload(DecisionModel):
    value: int


FAKE_BASIS = DecisionBasis(
    snapshot_fingerprint=decision_fingerprint({"state_revision": 2}),
    state_revision=2,
    graph_version=3,
    configuration_revision=1,
)


def make_request(*, budget: DecisionBudget | None = None) -> DecisionRequest[FakeRequestPayload]:
    return DecisionRequest[FakeRequestPayload](
        request_id=UUID(int=100),
        decision_type="fake.change",
        schema_version="1",
        target=DecisionTarget(target_type="fake_state", target_id="state-1"),
        correlation=DecisionCorrelation(
            run_id=UUID(int=200),
            task_id=UUID(int=201),
            node_id=UUID(int=202),
            action_id=UUID(int=203),
        ),
        basis=FAKE_BASIS,
        payload=FakeRequestPayload(goal="increment the bounded value"),
        allowed_actions=("apply",),
        evidence=(
            DecisionEvidenceReference(
                evidence_id="evidence-1",
                kind="test",
                source="runtime",
                reliability=0.95,
                summary="bounded test evidence",
            ),
        ),
        budget=budget or DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=1,
            max_revision_count=0,
            max_elapsed_seconds=10.0,
            max_cost_units=2.0,
        ),
        created_at=NOW,
    )


def make_sources() -> ProjectionSources:
    return ProjectionSources(
        items=(
            ProjectionSource(
                source_id="goal",
                source_type="goal_summary",
                agent_scope="runtime_shared",
                content={"goal": "increment", "secret": "remove-me"},
                sensitivity=ContextSensitivity.INTERNAL,
                priority=10,
                estimated_tokens=10,
            ),
            ProjectionSource(
                source_id="evidence",
                source_type="evidence",
                agent_scope="fake_agent",
                content={"value": 1},
                sensitivity=ContextSensitivity.INTERNAL,
                evidence_id="evidence-1",
                priority=5,
                estimated_tokens=5,
            ),
        )
    )


def make_policy() -> ContextProjectionPolicy:
    return ContextProjectionPolicy(
        policy_id="fake-agent",
        version="1",
        agent_scope="fake_agent",
        allowed_decision_types=frozenset({"fake.change"}),
        allowed_source_types=frozenset({"goal_summary", "evidence"}),
        allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
        redact_keys=frozenset({"secret"}),
        max_items=4,
        max_context_tokens=100,
    )


class FakeBasisProvider:
    module_id = "test.basis"

    def __init__(self, basis: DecisionBasis = FAKE_BASIS) -> None:
        self.basis = basis

    async def current_basis(self, request: DecisionRequest[object]) -> DecisionBasis:
        del request
        return self.basis


class FakeNormalizer:
    module_id = "test.normalizer"

    def normalize(
        self,
        request: DecisionRequest[FakeRequestPayload],
        proposal: DecisionProposal[FakeProposalPayload],
    ) -> NormalizedDecisionEffect[FakeEffectPayload]:
        return NormalizedDecisionEffect[FakeEffectPayload].create(
            payload=FakeEffectPayload(value=proposal.payload.value),
            operation="tool.call",
            target=request.target,
            governance_scope=DecisionGovernanceScope.ACTION,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.1,
            reversible=True,
            impact_description="bounded fake state change",
        )


class FakeAgent:
    module_id = "test.agent"

    def __init__(
        self,
        *,
        selected_action: str = "apply",
        context_fingerprint: str | None = None,
        cost_units: float = 0.5,
        elapsed_seconds: float = 0.1,
        revision: int = 0,
        fail: bool = False,
    ) -> None:
        self.selected_action = selected_action
        self.context_fingerprint = context_fingerprint
        self.cost_units = cost_units
        self.elapsed_seconds = elapsed_seconds
        self.revision = revision
        self.fail = fail
        self.calls = 0
        self.contexts: list[AgentContext] = []

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[FakeProposalPayload]:
        self.calls += 1
        self.contexts.append(context)
        if self.fail:
            raise RuntimeError("fake Agent failed")
        proposal = DecisionProposal[FakeProposalPayload](
            proposal_id=UUID(int=300 + self.calls),
            request_id=context.request_id,
            proposal_type=context.decision_type,
            schema_version="1",
            producer=DecisionProducer(
                producer_id="fake-agent",
                capability="test.propose",
            ),
            input_snapshot_fingerprint=context.basis_fingerprint,
            context_fingerprint=(
                self.context_fingerprint or context.context_fingerprint
            ),
            revision=self.revision,
            selected_action=self.selected_action,
            payload=FakeProposalPayload(value=2),
            rationale="bounded fake proposal",
            evidence_refs=tuple(item.evidence_id for item in context.evidence),
            confidence=0.95,
            created_at=NOW,
        )
        return AgentCallResult(
            proposal=proposal,
            elapsed_seconds=self.elapsed_seconds,
            cost_units=self.cost_units,
        )


class FakeGovernance:
    module_id = "test.governance"

    def __init__(
        self,
        outcome: DecisionGovernanceOutcome = DecisionGovernanceOutcome.ALLOW,
        *,
        resume_outcome: DecisionGovernanceOutcome | None = None,
    ) -> None:
        self.outcome = outcome
        self.resume_outcome = resume_outcome or outcome
        self.evaluations = 0
        self.resumes = 0

    def _resolution(
        self,
        outcome: DecisionGovernanceOutcome,
        *,
        reviewed: bool = False,
    ) -> DecisionGovernanceResolution[str]:
        receipt = DecisionGovernanceReceipt(
            outcome=outcome,
            governance_request_id=UUID(int=401),
            governance_decision_id=UUID(int=402 + self.evaluations + self.resumes),
            authorization_id=(UUID(int=403) if outcome is DecisionGovernanceOutcome.ALLOW else None),
            review_request_id=(
                UUID(int=404)
                if outcome is DecisionGovernanceOutcome.REVIEW_REQUIRED or reviewed
                else None
            ),
            reason=f"fake Governance {outcome.value}",
            decided_at=NOW,
        )
        return DecisionGovernanceResolution(
            receipt=receipt,
            approval="approved" if outcome is DecisionGovernanceOutcome.ALLOW else None,
        )

    async def evaluate(
        self,
        decision: ValidatedDecision[object, object, object],
    ) -> DecisionGovernanceResolution[str]:
        del decision
        self.evaluations += 1
        return self._resolution(self.outcome)

    async def resume_review(
        self,
        decision: ValidatedDecision[object, object, object],
        receipt: DecisionGovernanceReceipt,
    ) -> DecisionGovernanceResolution[str]:
        del decision, receipt
        self.resumes += 1
        return self._resolution(self.resume_outcome, reviewed=True)

    def restore_approval(
        self,
        decision: ValidatedDecision[object, object, object],
        receipt: DecisionGovernanceReceipt,
    ) -> str:
        del decision, receipt
        return "approved"


class FakeApplier:
    module_id = "test.applier"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.values: list[int] = []

    async def apply(
        self,
        decision: ValidatedDecision[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
        ],
        approval: str,
    ) -> DecisionApplyReceipt:
        if approval != "approved":
            raise AssertionError("unexpected approval")
        self.calls += 1
        if self.fail:
            raise RuntimeError("fake apply failed")
        self.values.append(decision.normalized_effect.payload.value)
        return DecisionApplyReceipt(
            effect_fingerprint=decision.normalized_effect.effect_fingerprint,
            committed_state_fingerprint=decision_fingerprint(
                {"value": decision.normalized_effect.payload.value}
            ),
            result={"value": decision.normalized_effect.payload.value},
            applied_at=NOW,
        )

    async def resume_apply(
        self,
        decision: ValidatedDecision[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
        ],
        approval: str,
    ) -> DecisionApplyReceipt:
        return await self.apply(decision, approval)

    async def reconcile(
        self,
        decision: ValidatedDecision[
            FakeRequestPayload,
            FakeProposalPayload,
            FakeEffectPayload,
        ],
        approval: str,
    ) -> DecisionReconciliation:
        del decision, approval
        return DecisionReconciliation(
            status=DecisionReconciliationStatus.NOT_COMMITTED,
            reason="fake authoritative state has no commit",
        )


class FakeTraceWriter:
    module_id = "test.trace"

    def __init__(self) -> None:
        self.events: list[DecisionTraceEvent] = []

    async def record(self, event: DecisionTraceEvent) -> None:
        self.events.append(event)
