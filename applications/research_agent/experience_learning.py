"""Governed cross-run Learning Insight composition for Initial Planning."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import TraceSink
from adaptive_agent_runtime.context_memory import MemoryScope
from adaptive_agent_runtime.decision_feedback import DecisionFeedbackAttributionType
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
    DecisionFaultPoint,
    DecisionGovernanceScope,
    DecisionLifecycleCoordinator,
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
from adaptive_agent_runtime.experience_learning import (
    EXPERIENCE_LEARNING_DECISION_TYPE,
    LEARNING_INSIGHT_COMMIT_OPERATION,
    LearningAssessmentAgentRequest,
    LearningAssessmentRequest,
    LearningEvidenceAgentView,
    LearningEvidenceBinding,
    LearningEvidenceCandidate,
    LearningInsight,
    LearningInsightDraft,
    LearningInsightEffect,
    LearningInsightStore,
    stable_learning_insight_id,
    stable_learning_request_id,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernanceTarget,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewService,
    RuntimeCommitPermit,
    RuntimeDecisionGovernanceAdapter,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    InferenceCorrelation,
)

from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord


LEARNING_ASSESSMENT_INPUT_SOURCE_TYPE = "learning_assessment_input"


class LearningAssessmentCapability(Protocol):
    module_id: str
    capability_id: str

    async def assess_learning(
        self,
        request: LearningAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[LearningInsightDraft]: ...


class DeterministicLearningAssessmentCapability:
    """Offline Agent substitute that produces an association-only Draft."""

    module_id = "research_agent.experience_learning.deterministic"
    capability_id = "experience_learning.deterministic"

    async def assess_learning(
        self,
        request: LearningAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[LearningInsightDraft]:
        del invocation
        eligible = tuple(item for item in request.evidence if item.conclusion_eligible)
        succeeded = tuple(
            item for item in eligible if item.outcome.value == "succeeded"
        )
        failed = tuple(item for item in eligible if item.outcome.value == "failed")
        if succeeded and failed:
            supporting = succeeded
            counter = failed
            observed = (
                "Completed and failed outcomes were both observed across verified "
                "Initial Planning Decisions with complete provenance."
            )
        elif succeeded:
            supporting = succeeded
            counter = ()
            observed = (
                "Completed outcomes were observed across multiple verified Initial "
                "Planning Decisions with complete provenance."
            )
        else:
            supporting = failed
            counter = ()
            observed = (
                "Failed outcomes were observed across multiple verified Initial "
                "Planning Decisions with complete provenance."
            )
        limitation_count = sum(
            1 for item in request.evidence if not item.conclusion_eligible
        )
        limitations = [
            "The evidence is associative and does not establish causality, reward, "
            "strategy quality, or a required Runtime change."
        ]
        if limitation_count:
            limitations.append(
                f"{limitation_count} inconclusive record(s) are limitation-only and "
                "were not used as supporting or counterevidence."
            )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=LearningInsightDraft(
                observed_pattern=observed,
                applicable_conditions=(
                    "Initial Planning in the same tenant, project, and planner scope.",
                    "Only persisted terminal runs with verified provenance are in scope.",
                ),
                limitations=tuple(limitations),
                supporting_candidate_refs=tuple(
                    item.candidate_ref for item in supporting
                ),
                counterevidence_candidate_refs=tuple(
                    item.candidate_ref for item in counter
                ),
            ),
        )


@dataclass(frozen=True)
class ResearchLearningAssessmentResult:
    insight: LearningInsight | None
    governance_record: GovernanceRecord | None = None
    request_id: UUID | None = None
    proposal: LearningInsightDraft | None = None
    eligible_candidate_count: int = 0


class _LearningProposalProducer:
    module_id = "research_agent.experience_learning.proposal_producer"

    def __init__(
        self,
        capability: LearningAssessmentCapability,
        *,
        timeout_seconds: float,
        correlation: InferenceCorrelation,
    ) -> None:
        self._capability = capability
        self._timeout_seconds = timeout_seconds
        self._correlation = correlation

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[LearningInsightDraft]:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == LEARNING_ASSESSMENT_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1 or not isinstance(blocks[0].content, Mapping):
            raise ValueError("Learning Agent requires one isolated evidence block")
        request = LearningAssessmentAgentRequest.model_validate(dict(blocks[0].content))
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.assess_learning(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "experience.learning.propose"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Learning Assessment cannot return ToolIntents")
        if turn.result is None:
            raise RuntimeError("Learning Assessment returned no Draft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="experience_learning",
                    implementation_version="phase-3d",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=LEARNING_INSIGHT_COMMIT_OPERATION,
                payload=draft,
                rationale=(
                    "Association-only cross-run assessment; Runtime validates evidence."
                ),
                evidence_refs=(),
                confidence=0.5,
            ),
            elapsed_seconds=monotonic() - started,
        )


class _LearningEffectNormalizer:
    module_id = "research_agent.experience_learning.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[LearningAssessmentRequest],
        proposal: DecisionProposal[LearningInsightDraft],
    ) -> NormalizedDecisionEffect[LearningInsightEffect]:
        source = request.payload
        draft = proposal.payload
        _validate_non_behavioral_draft(draft)
        by_ref = {item.candidate_ref: item for item in source.candidates}
        selected_refs = (
            *draft.supporting_candidate_refs,
            *draft.counterevidence_candidate_refs,
        )
        if any(item not in by_ref for item in selected_refs):
            raise ValueError("Learning Agent selected evidence outside Runtime candidates")
        selected_candidates = tuple(by_ref[item] for item in selected_refs)
        if any(
            item.attribution_type is not DecisionFeedbackAttributionType.ASSOCIATED
            for item in selected_candidates
        ):
            raise ValueError(
                "INSUFFICIENT_EVIDENCE cannot support or refute a Learning Insight"
            )
        if len({item.source_run_id for item in selected_candidates}) < 2:
            raise ValueError("Learning Draft requires two independent selected runs")
        bindings = {item.candidate_ref: _binding(item) for item in source.candidates}
        supporting = tuple(bindings[item] for item in draft.supporting_candidate_refs)
        counter = tuple(
            bindings[item] for item in draft.counterevidence_candidate_refs
        )
        selected = (*supporting, *counter)
        effect = LearningInsightEffect(
            learning_insight_id=stable_learning_insight_id(request.request_id),
            scope=source.scope,
            subject_decision_type=source.subject_decision_type,
            observed_pattern=draft.observed_pattern,
            applicable_conditions=draft.applicable_conditions,
            limitations=draft.limitations,
            evidence_snapshot=tuple(
                bindings[item.candidate_ref] for item in source.candidates
            ),
            supporting_evidence=supporting,
            counterevidence=counter,
            source_run_refs=tuple(
                dict.fromkeys(item.source_run_id for item in selected)
            ),
            source_feedback_refs=tuple(
                dict.fromkeys(item.feedback_id for item in selected)
            ),
            source_experience_refs=tuple(
                dict.fromkeys(item.experience_id for item in selected)
            ),
            evidence_set_fingerprint=source.evidence_set_fingerprint,
            proposal_fingerprint=decision_fingerprint(draft),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[LearningInsightEffect].create(
            payload=effect,
            operation=LEARNING_INSIGHT_COMMIT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.1,
            reversible=False,
            impact_description=(
                "Append a non-behavioral, association-only Learning Insight."
            ),
        )


class _LearningBasisProvider:
    module_id = "research_agent.experience_learning.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        # The immutable candidate snapshot is retained by the Decision checkpoint.
        # The authority Store independently revalidates every source at commit.
        return self._basis


class ResearchExperienceLearningHandler:
    module_id = "research_agent.experience_learning"

    def __init__(
        self,
        *,
        store: LearningInsightStore,
        capability: LearningAssessmentCapability,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        checkpoint_store: DecisionCheckpointStore[
            LearningAssessmentRequest,
            LearningInsightDraft,
            LearningInsightEffect,
        ]
        | None = None,
        fault_injector: Callable[[DecisionFaultPoint, Any], None] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._store = store
        self._capability = capability
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._checkpoints = checkpoint_store or InMemoryDecisionCheckpointStore()
        self._fault_injector = fault_injector
        self._timeout_seconds = timeout_seconds

    async def assess(
        self,
        *,
        trigger_run_id: UUID,
        task_id: UUID,
        scope: MemoryScope,
        subject_decision_type: str,
    ) -> ResearchLearningAssessmentResult:
        request_id = stable_learning_request_id(
            trigger_run_id, scope, subject_decision_type
        )
        existing = await self._checkpoints.load(request_id)
        if existing is not None:
            coordinator, recorder, issuer = self._coordinator(
                existing.request,
                existing.request.basis,
            )
            checkpoint = await coordinator.resume(request_id)
            return await self._result(checkpoint, recorder, issuer)

        candidates = await self._store.resolve_candidates(
            scope=scope,
            subject_decision_type=subject_decision_type,
        )
        eligible = tuple(
            item
            for item in candidates
            if item.attribution_type is DecisionFeedbackAttributionType.ASSOCIATED
        )
        if len({item.source_run_id for item in eligible}) < 2:
            return ResearchLearningAssessmentResult(
                insight=None,
                eligible_candidate_count=len(eligible),
            )
        evidence_set_fingerprint = decision_fingerprint(
            tuple(item.candidate_fingerprint for item in candidates)
        )
        payload = LearningAssessmentRequest(
            scope=scope,
            subject_decision_type=subject_decision_type,
            trigger_run_id=trigger_run_id,
            candidates=candidates,
            evidence_set_fingerprint=evidence_set_fingerprint,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(payload),
            state_revision=0,
            configuration_revision=1,
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=EXPERIENCE_LEARNING_DECISION_TYPE,
            target=DecisionTarget(
                target_type="learning_insight",
                target_id=str(stable_learning_insight_id(request_id)),
            ),
            correlation=DecisionCorrelation(run_id=trigger_run_id, task_id=task_id),
            basis=basis,
            payload=payload,
            allowed_actions=(LEARNING_INSIGHT_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="learning-association-only",
                    description="No causal, reward, ranking, or policy claim is allowed.",
                ),
                DecisionConstraint(
                    constraint_id="learning-no-behavior-change",
                    description="The Insight cannot modify Runtime behavior or sources.",
                ),
                DecisionConstraint(
                    constraint_id="learning-two-independent-runs",
                    description="At least two verified associated runs must be selected.",
                ),
            ),
            evidence=tuple(
                DecisionEvidenceReference(
                    evidence_id=item.candidate_ref,
                    kind="runtime.verified.learning_candidate",
                    source="learning_evidence_resolver",
                    reliability=(
                        1.0
                        if item.attribution_type
                        is DecisionFeedbackAttributionType.ASSOCIATED
                        else 0.0
                    ),
                    summary=(
                        "Verified associated cross-run evidence."
                        if item.attribution_type
                        is DecisionFeedbackAttributionType.ASSOCIATED
                        else "Inconclusive evidence retained only as a limitation."
                    ),
                )
                for item in candidates
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
                max_revision_count=0,
                max_elapsed_seconds=self._timeout_seconds,
            ),
        )
        agent_request = LearningAssessmentAgentRequest(
            subject_decision_type=subject_decision_type,
            evidence=tuple(_agent_view(item) for item in candidates),
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="learning-evidence-set",
                    source_type=LEARNING_ASSESSMENT_INPUT_SOURCE_TYPE,
                    agent_scope="experience_learning",
                    content=agent_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=1024,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.experience_learning.context",
            version="1",
            agent_scope="experience_learning",
            allowed_decision_types=frozenset({EXPERIENCE_LEARNING_DECISION_TYPE}),
            allowed_source_types=frozenset({LEARNING_ASSESSMENT_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {
                    "api_key",
                    "authorization",
                    "credential",
                    "database",
                    "password",
                    "secret",
                    "token",
                    "trace",
                }
            ),
            max_items=1,
            max_context_tokens=2048,
        )
        coordinator, recorder, issuer = self._coordinator(request, basis)
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        return await self._result(checkpoint, recorder, issuer)

    def _coordinator(
        self,
        request: DecisionRequest[LearningAssessmentRequest],
        basis: DecisionBasis,
    ) -> tuple[
        DecisionLifecycleCoordinator[
            LearningAssessmentRequest,
            LearningInsightDraft,
            LearningInsightEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        recorder = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)

        async def forbidden_apply(effect: LearningInsightEffect) -> JsonValue:
            del effect
            raise RuntimeError("Learning Insight commit requires a Runtime Permit")

        async def apply_authorized(
            normalized: NormalizedDecisionEffect[LearningInsightEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            insight = await self._store.commit(
                normalized.payload,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            readback = await self._store.load_by_effect(normalized.effect_fingerprint)
            if readback is None or readback != insight:
                raise RuntimeError("Learning Insight commit read-back failed")
            return self._insight_result(readback)

        async def reconcile(
            normalized: NormalizedDecisionEffect[LearningInsightEffect],
        ) -> DecisionReconciliation:
            try:
                insight = await self._store.load_by_effect(
                    normalized.effect_fingerprint
                )
            except Exception as exc:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=(
                        "Learning Insight read-back is indeterminate: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if insight is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Learning Insight has not been committed",
                )
            effect = normalized.payload
            if (
                insight.learning_insight_id != effect.learning_insight_id
                or insight.evidence_set_fingerprint
                != effect.evidence_set_fingerprint
                or insight.effect_fingerprint != normalized.effect_fingerprint
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Learning Insight conflicts with the authorized Effect",
                )
            result = self._insight_result(insight)
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Learning Insight was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=decision_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            LearningAssessmentRequest,
            LearningInsightDraft,
            LearningInsightEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=_LearningProposalProducer(
                self._capability,
                timeout_seconds=self._timeout_seconds,
                correlation=InferenceCorrelation(
                    run_id=request.correlation.run_id,
                    task_id=request.correlation.task_id,
                ),
            ),
            basis_provider=_LearningBasisProvider(basis),
            validator=RuntimeDecisionValidator(normalizer=_LearningEffectNormalizer()),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=recorder,
                authorization_issuer=issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=forbidden_apply,
                apply_authorized_effect=apply_authorized,
                reconcile_effect=reconcile,
            ),
            checkpoint_store=self._checkpoints,
            checkpoint_type=DecisionCheckpoint[
                LearningAssessmentRequest,
                LearningInsightDraft,
                LearningInsightEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
            fault_injector=self._fault_injector,
        )
        return coordinator, recorder, issuer

    async def _result(
        self,
        checkpoint: DecisionCheckpoint[
            LearningAssessmentRequest,
            LearningInsightDraft,
            LearningInsightEffect,
        ],
        recorder: RecordingGovernanceEvaluator,
        issuer: RecordingAuthorizationIssuer,
    ) -> ResearchLearningAssessmentResult:
        eligible_count = sum(
            1
            for item in checkpoint.request.payload.candidates
            if item.attribution_type is DecisionFeedbackAttributionType.ASSOCIATED
        )
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            return ResearchLearningAssessmentResult(
                insight=None,
                request_id=checkpoint.request_id,
                proposal=(checkpoint.proposal.payload if checkpoint.proposal else None),
                eligible_candidate_count=eligible_count,
            )
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Learning Assessment did not apply: {reason}")
        normalized = checkpoint.validated_decision.normalized_effect
        insight = await self._store.load_by_effect(normalized.effect_fingerprint)
        if insight is None:
            raise RuntimeError("APPLIED Learning Decision has no committed Insight")
        governance_record = (
            GovernanceRecord(
                scenario="experience_learning",
                request=recorder.request,
                preliminary=recorder.preliminary,
                final=recorder.final,
                authorization=issuer.authorization,
                review=None,
            )
            if recorder.request is not None
            and recorder.preliminary is not None
            and recorder.final is not None
            else None
        )
        return ResearchLearningAssessmentResult(
            insight=insight,
            governance_record=governance_record,
            request_id=checkpoint.request_id,
            proposal=(checkpoint.proposal.payload if checkpoint.proposal else None),
            eligible_candidate_count=eligible_count,
        )

    @staticmethod
    def _insight_result(insight: LearningInsight) -> JsonValue:
        return {
            "learning_insight_id": str(insight.learning_insight_id),
            "version": insight.version,
            "effect_fingerprint": insight.effect_fingerprint,
            "evidence_set_fingerprint": insight.evidence_set_fingerprint,
            "insight_fingerprint": decision_fingerprint(insight),
        }


def _binding(candidate: LearningEvidenceCandidate) -> LearningEvidenceBinding:
    return LearningEvidenceBinding(
        candidate_ref=candidate.candidate_ref,
        candidate_fingerprint=candidate.candidate_fingerprint,
        source_run_id=candidate.source_run_id,
        feedback_id=candidate.feedback_id,
        feedback_effect_fingerprint=candidate.feedback_effect_fingerprint,
        experience_id=candidate.experience.experience_id,
        experience_effect_fingerprint=candidate.experience.effect_fingerprint,
        evaluation_report_id=candidate.evaluation.report_id,
        evaluation_report_fingerprint=candidate.evaluation.report_fingerprint,
        artifact_effect_fingerprint=candidate.artifact.effect_fingerprint,
        runtime_outcome=candidate.runtime_outcome,
        attribution_type=candidate.attribution_type,
    )


def _agent_view(candidate: LearningEvidenceCandidate) -> LearningEvidenceAgentView:
    return LearningEvidenceAgentView(
        candidate_ref=candidate.candidate_ref,
        outcome=candidate.runtime_outcome,
        evaluation_verdict=candidate.evaluation_verdict,
        observed_pattern=str(
            candidate.experience_summary.get(
                "observed_pattern", "Verified execution evidence is present."
            )
        ),
        possible_relevance=str(
            candidate.experience_summary.get(
                "possible_relevance", "May be relevant to similarly scoped planning."
            )
        ),
        limitation=(
            "Limitation-only: deterministic Evaluation was inconclusive."
            if candidate.attribution_type
            is DecisionFeedbackAttributionType.INSUFFICIENT_EVIDENCE
            else "Association only; provenance is verified but causality is not."
        ),
        conclusion_eligible=(
            candidate.attribution_type is DecisionFeedbackAttributionType.ASSOCIATED
        ),
    )


def _validate_non_behavioral_draft(draft: LearningInsightDraft) -> None:
    authoritative_phrases = (
        "always",
        "automatically",
        "because",
        "caused",
        "causes",
        "change configuration",
        "modify runtime",
        "must",
        "prompt rewrite",
        "rank strategy",
        "replace strategy",
        "reward",
        "set configuration",
    )
    asserted_text = " ".join(
        (draft.observed_pattern, *draft.applicable_conditions)
    ).lower()
    found = tuple(item for item in authoritative_phrases if item in asserted_text)
    if found:
        raise ValueError(
            "Learning Draft contains causal or behavior-changing semantics: "
            + ", ".join(found)
        )
