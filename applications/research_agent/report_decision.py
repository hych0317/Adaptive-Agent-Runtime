"""Governed commit lifecycle for the final Research report artifact."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import Field, JsonValue

from adaptive_agent_runtime import TraceSink
from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionApplyReceipt,
    DecisionBasis,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionCheckpointStore,
    DecisionConstraint,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionGovernanceScope,
    DecisionLifecycleCoordinator,
    DecisionModel,
    DecisionProducer,
    DecisionProposal,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionRequest,
    DecisionResultStatus,
    DecisionRiskLevel,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    NormalizedDecisionEffect,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewDecision,
    HumanReviewService,
    ReviewOutcome,
    RuntimeDecisionGovernanceAdapter,
    governance_fingerprint,
)
from adaptive_agent_runtime.persistence import (
    WorkspaceArtifactCommitReceipt,
    WorkspaceArtifactCommitter,
)

from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord, ResearchReport


REPORT_COMMIT_DECISION_TYPE = "artifact.report_commit"
REPORT_COMMIT_OPERATION = "workspace.report.commit"


class ReportDraft(DecisionModel):
    report: ResearchReport
    provenance: tuple[str, ...] = Field(min_length=1)


class ReportDecisionPayload(DecisionModel):
    run_id: UUID
    task_id: UUID
    node_id: UUID
    draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: tuple[str, ...] = Field(min_length=1)


class ReportArtifactEffect(DecisionModel):
    run_id: UUID
    task_id: UUID
    node_id: UUID
    report: ResearchReport
    provenance: tuple[str, ...] = Field(min_length=1)
    draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class _FixedBasisProvider:
    module_id = "research_agent.report.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class _DraftProducer:
    module_id = "research_agent.report.draft_producer"

    def __init__(self, draft: ReportDraft) -> None:
        self._draft = draft

    async def propose(self, context: AgentContext) -> AgentCallResult[ReportDraft]:
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=REPORT_COMMIT_DECISION_TYPE,
                producer=DecisionProducer(
                    producer_id="research-report-generator",
                    capability="artifact.report_generation",
                    implementation_version="1",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=REPORT_COMMIT_OPERATION,
                payload=self._draft,
                rationale="Submit the generated report as a non-executable artifact draft.",
                evidence_refs=self._draft.provenance,
                confidence=0.9,
            )
        )


class _EffectNormalizer:
    module_id = "research_agent.report.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[ReportDecisionPayload],
        proposal: DecisionProposal[ReportDraft],
    ) -> NormalizedDecisionEffect[ReportArtifactEffect]:
        if decision_fingerprint(proposal.payload) != request.payload.draft_fingerprint:
            raise ValueError("Report Draft changed after Runtime request creation")
        if proposal.payload.provenance != request.payload.provenance:
            raise ValueError("Report Draft provenance changed")
        effect = ReportArtifactEffect(
            run_id=request.payload.run_id,
            task_id=request.payload.task_id,
            node_id=request.payload.node_id,
            report=proposal.payload.report,
            provenance=proposal.payload.provenance,
            draft_fingerprint=request.payload.draft_fingerprint,
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect.create(
            payload=effect,
            operation=REPORT_COMMIT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.25,
            reversible=True,
            impact_description="Commit one validated report artifact to the run workspace.",
        )


class ResearchReportDecisionHandler:
    module_id = "research_agent.report_decision_handler"

    def __init__(
        self,
        *,
        committer: WorkspaceArtifactCommitter,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        checkpoint_store: DecisionCheckpointStore[
            ReportDecisionPayload, ReportDraft, ReportArtifactEffect
        ] | None = None,
    ) -> None:
        self._committer = committer
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._checkpoint_store = checkpoint_store or InMemoryDecisionCheckpointStore()

    async def commit(
        self,
        *,
        run_id: UUID,
        task_id: UUID,
        node_id: UUID,
        report: ResearchReport,
        provenance: tuple[str, ...],
    ) -> tuple[ResearchReport, WorkspaceArtifactCommitReceipt, GovernanceRecord]:
        draft = ReportDraft(report=report, provenance=provenance)
        draft_fingerprint = decision_fingerprint(draft)
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "node_id": node_id,
                    "draft_fingerprint": draft_fingerprint,
                    "provenance": provenance,
                }
            )
        )
        payload = ReportDecisionPayload(
            run_id=run_id,
            task_id=task_id,
            node_id=node_id,
            draft_fingerprint=draft_fingerprint,
            provenance=provenance,
        )
        request = DecisionRequest(
            decision_type=REPORT_COMMIT_DECISION_TYPE,
            target=DecisionTarget(
                target_type="workspace_report",
                target_id=f"{run_id}:{node_id}",
            ),
            correlation=DecisionCorrelation(
                run_id=run_id, task_id=task_id, node_id=node_id
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(REPORT_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="report-artifact-only",
                    description="The proposal may contain only report content and provenance.",
                ),
            ),
            evidence=tuple(
                DecisionEvidenceReference(
                    evidence_id=item,
                    kind="runtime.accepted_output",
                    source="runtime_workspace",
                    reliability=1.0,
                    summary="Accepted upstream research output.",
                )
                for item in provenance
            ),
            budget=DecisionBudget(max_agent_calls=1, max_decision_cycles=1),
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="report-draft",
                    source_type="report_draft",
                    agent_scope="report_commit",
                    content={
                        "draft_fingerprint": draft_fingerprint,
                        "provenance": list(provenance),
                    },
                    sensitivity=ContextSensitivity.INTERNAL,
                    evidence_id=provenance[0],
                    priority=100,
                    estimated_tokens=32,
                ),
                *(
                    ProjectionSource(
                        source_id=f"report-evidence:{evidence_id}",
                        source_type="report_evidence",
                        agent_scope="report_commit",
                        content={"reference_id": evidence_id},
                        sensitivity=ContextSensitivity.INTERNAL,
                        evidence_id=evidence_id,
                        priority=90,
                        estimated_tokens=8,
                    )
                    for evidence_id in provenance[1:]
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.report_commit.context",
            version="1",
            agent_scope="report_commit",
            allowed_decision_types=frozenset({REPORT_COMMIT_DECISION_TYPE}),
            allowed_source_types=frozenset({"report_draft", "report_evidence"}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            max_items=len(provenance),
            max_context_tokens=64 + 8 * len(provenance),
        )
        evaluator = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)
        committed_receipt: WorkspaceArtifactCommitReceipt | None = None

        async def apply_effect(effect: ReportArtifactEffect) -> JsonValue:
            nonlocal committed_receipt
            raise RuntimeError("Report Apply requires normalized effect fingerprint")

        async def apply_normalized(
            normalized: NormalizedDecisionEffect[ReportArtifactEffect],
        ) -> JsonValue:
            nonlocal committed_receipt
            effect = normalized.payload
            committed_receipt = self._committer.commit(
                run_id=effect.run_id,
                node_id=effect.node_id,
                artifact_type="research_report",
                effect_fingerprint=normalized.effect_fingerprint,
                artifact=effect.report.model_dump(mode="json"),
                provenance=effect.provenance,
            )
            readback = self._committer.load_by_effect(normalized.effect_fingerprint)
            if readback is None or readback[1] != committed_receipt:
                raise RuntimeError("Report artifact commit failed read-back")
            return committed_receipt.model_dump(mode="json")

        async def reconcile(
            normalized: NormalizedDecisionEffect[ReportArtifactEffect],
        ) -> DecisionReconciliation:
            found = self._committer.load_by_effect(normalized.effect_fingerprint)
            if found is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Report artifact effect has no commit receipt",
                )
            artifact, receipt = found
            if (
                artifact != normalized.payload.report.model_dump(mode="json")
                or receipt.artifact_fingerprint != decision_fingerprint(artifact)
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Report artifact read-back conflicts with the effect",
                )
            result = receipt.model_dump(mode="json")
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Report artifact was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=governance_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            ReportDecisionPayload,
            ReportDraft,
            ReportArtifactEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=_DraftProducer(draft),
            basis_provider=_FixedBasisProvider(basis),
            validator=RuntimeDecisionValidator(normalizer=_EffectNormalizer()),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=evaluator,
                authorization_issuer=issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
                apply_normalized_effect=apply_normalized,
                reconcile_effect=reconcile,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                ReportDecisionPayload, ReportDraft, ReportArtifactEffect
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            governance_receipt = checkpoint.governance_receipt
            if governance_receipt is None or governance_receipt.review_request_id is None:
                raise RuntimeError("Report review checkpoint has no review request")
            review_id = governance_receipt.review_request_id
            self._reviews.resolve(
                review_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="The report artifact is source-bound and reversible.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            checkpoint = await coordinator.resume_review(request.request_id)
        if (
            committed_receipt is None
            and checkpoint.result is not None
            and checkpoint.result.status is DecisionResultStatus.APPLIED
            and checkpoint.validated_decision is not None
        ):
            found = self._committer.load_by_effect(
                checkpoint.validated_decision.normalized_effect.effect_fingerprint
            )
            if found is not None:
                committed_receipt = found[1]
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or committed_receipt is None
            or evaluator.request is None
            or evaluator.preliminary is None
            or evaluator.final is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Report Decision did not commit: {reason}")
        return (
            report,
            committed_receipt,
            GovernanceRecord(
                scenario="report_commit",
                request=evaluator.request,
                preliminary=evaluator.preliminary,
                final=evaluator.final,
                authorization=issuer.authorization,
                review=evaluator.review,
            ),
        )
