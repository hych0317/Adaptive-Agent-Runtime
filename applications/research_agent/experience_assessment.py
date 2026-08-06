"""Governed post-execution Experience Assessment composition."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import AgentState, RunStatus, StateStore, TraceSink
from adaptive_agent_runtime.context_memory import (
    EXPERIENCE_ASSESSMENT_DECISION_TYPE,
    EXPERIENCE_METADATA_COMMIT_OPERATION,
    ExperienceArtifactReference,
    ExperienceAssessmentAgentRequest,
    ExperienceAssessmentDraft,
    ExperienceAssessmentRequest,
    ExperienceEvaluationReference,
    ExperienceExecutionOutcome,
    ExperienceMemorySource,
    ExperienceMetadata,
    ExperienceMetadataEffect,
    ExperienceMetadataStore,
    MemoryUnit,
    RuntimeExecutionObservation,
    stable_experience_id,
    stable_experience_request_id,
)
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
from adaptive_agent_runtime.evaluation import EvaluationReport
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
from adaptive_agent_runtime.persistence import WorkspaceArtifactCommitReceipt

from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord


EXPERIENCE_ASSESSMENT_INPUT_SOURCE_TYPE = "experience_assessment_input"


class ExperienceAssessmentCapability(Protocol):
    module_id: str
    capability_id: str

    async def assess_experience(
        self,
        request: ExperienceAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ExperienceAssessmentDraft]: ...


class ExperienceArtifactReadback(Protocol):
    def load(
        self,
        *,
        run_id: UUID,
        node_id: UUID,
        artifact_type: str,
    ) -> tuple[JsonValue, WorkspaceArtifactCommitReceipt] | None: ...


class DeterministicExperienceAssessmentCapability:
    """Offline proposal-only fallback; Runtime remains the outcome authority."""

    module_id = "research_agent.experience_assessment.deterministic"
    capability_id = "experience_assessment.deterministic"

    async def assess_experience(
        self,
        request: ExperienceAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ExperienceAssessmentDraft]:
        del invocation
        scopes = ", ".join(
            str(item.get("scope", "unknown")) for item in request.evaluation_summary
        )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=ExperienceAssessmentDraft(
                observed_pattern=f"Completed evidence exists across {scopes} evaluation scopes.",
                possible_relevance=(
                    "May explain when similarly scoped research evidence is useful."
                ),
                explanation=(
                    "Assessment is advisory; Runtime observations determine outcome signals."
                ),
            ),
        )


@dataclass(frozen=True)
class ResearchExperienceAssessmentResult:
    metadata: ExperienceMetadata
    governance_record: GovernanceRecord | None
    request_id: UUID
    proposal: ExperienceAssessmentDraft


class _ExperienceProposalProducer:
    module_id = "research_agent.experience_assessment.proposal_producer"

    def __init__(
        self,
        capability: ExperienceAssessmentCapability,
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
    ) -> AgentCallResult[ExperienceAssessmentDraft]:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == EXPERIENCE_ASSESSMENT_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1 or not isinstance(blocks[0].content, Mapping):
            raise ValueError("Experience Agent requires one isolated input block")
        request = ExperienceAssessmentAgentRequest.model_validate(
            dict(blocks[0].content)
        )
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.assess_experience(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "experience.assessment.propose"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Experience Assessment cannot return ToolIntents")
        if turn.result is None:
            raise RuntimeError("Experience Assessment returned no Draft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="experience_assessment",
                    implementation_version="phase-3b",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=EXPERIENCE_METADATA_COMMIT_OPERATION,
                payload=draft,
                rationale=draft.explanation,
                confidence=0.5,
            ),
            elapsed_seconds=monotonic() - started,
        )


class _ExperienceEffectNormalizer:
    module_id = "research_agent.experience_assessment.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[ExperienceAssessmentRequest],
        proposal: DecisionProposal[ExperienceAssessmentDraft],
    ) -> NormalizedDecisionEffect[ExperienceMetadataEffect]:
        source = request.payload
        effect = ExperienceMetadataEffect(
            experience_id=stable_experience_id(request.request_id),
            source_execution_id=source.execution_id,
            source_run_id=source.run_id,
            source_task_id=source.task_id,
            source_decision_ids=source.related_decision_ids,
            source_artifacts=source.source_artifacts,
            source_evaluations=source.evaluation_refs,
            source_memories=source.source_memories,
            runtime_observation=source.runtime_observation,
            execution_outcome=source.execution_outcome,
            success_signal=source.success_signal,
            failure_signal=source.failure_signal,
            evidence_refs=tuple(
                [
                    *(f"evaluation:{item.evaluation_id}" for item in source.evaluation_refs),
                    *(
                        f"artifact:{item.effect_fingerprint}"
                        for item in source.source_artifacts
                    ),
                ]
            ),
            agent_assessment_summary={
                "observed_pattern": proposal.payload.observed_pattern,
                "possible_relevance": proposal.payload.possible_relevance,
                "explanation": proposal.payload.explanation,
            },
            proposal_fingerprint=decision_fingerprint(proposal.payload),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[ExperienceMetadataEffect].create(
            payload=effect,
            operation=EXPERIENCE_METADATA_COMMIT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.15,
            reversible=False,
            impact_description=(
                "Append Runtime-evidenced Experience Metadata without changing Memory."
            ),
        )


class _ExperienceBasisProvider:
    module_id = "research_agent.experience_assessment.basis"

    def __init__(self, state_store: StateStore, basis: DecisionBasis) -> None:
        self._state_store = state_store
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        state = await self._state_store.load(request.correlation.run_id)
        if (
            state is not None
            and state.revision == request.payload.runtime_observation.state_revision
            and decision_fingerprint(state)
            == request.payload.runtime_observation.state_fingerprint
        ):
            return self._basis
        return self._basis.model_copy(
            update={"snapshot_fingerprint": decision_fingerprint(state)}
        )


class ResearchExperienceAssessmentHandler:
    module_id = "research_agent.experience_assessment"

    def __init__(
        self,
        *,
        state_store: StateStore,
        artifact_readback: ExperienceArtifactReadback,
        metadata_store: ExperienceMetadataStore,
        capability: ExperienceAssessmentCapability,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        checkpoint_store: DecisionCheckpointStore[
            ExperienceAssessmentRequest,
            ExperienceAssessmentDraft,
            ExperienceMetadataEffect,
        ]
        | None = None,
        timeout_seconds: float = 5.0,
        fault_injector: Callable[[DecisionFaultPoint, Any], None] | None = None,
    ) -> None:
        self._state_store = state_store
        self._artifact_readback = artifact_readback
        self._metadata_store = metadata_store
        self._capability = capability
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._checkpoints = checkpoint_store or InMemoryDecisionCheckpointStore()
        self._timeout_seconds = timeout_seconds
        self._fault_injector = fault_injector

    async def assess(
        self,
        *,
        final_state: AgentState,
        evaluation: EvaluationReport,
        artifact_receipt: WorkspaceArtifactCommitReceipt,
        source_memories: tuple[MemoryUnit, ...] = (),
        assessment_revision: int = 1,
    ) -> ResearchExperienceAssessmentResult:
        await self._validate_sources(final_state, evaluation, artifact_receipt)
        source_decision_id = artifact_receipt.source_decision_request_id
        if source_decision_id is None:
            raise ValueError("Experience Artifact has no source Decision")
        request_id = stable_experience_request_id(
            final_state.run_id,
            assessment_revision,
        )
        artifact = ExperienceArtifactReference(
            node_id=artifact_receipt.node_id,
            artifact_type=artifact_receipt.artifact_type,
            effect_fingerprint=artifact_receipt.effect_fingerprint,
            artifact_fingerprint=artifact_receipt.artifact_fingerprint,
            source_decision_id=source_decision_id,
        )
        evaluations = tuple(
            ExperienceEvaluationReference(
                evaluation_id=item.evaluation_id,
                evaluator_id=item.evaluator_id,
                scope=item.scope.value,
                verdict=item.verdict.value,
                score=item.score,
                evaluation_fingerprint=decision_fingerprint(item),
            )
            for item in evaluation.results
        )
        memory_sources = tuple(
            ExperienceMemorySource(
                memory_id=item.memory_id,
                revision=item.revision,
                memory_fingerprint=decision_fingerprint(item),
            )
            for item in source_memories
        )
        succeeded = final_state.status is RunStatus.COMPLETED
        observation = RuntimeExecutionObservation(
            state_revision=final_state.revision,
            state_fingerprint=decision_fingerprint(final_state),
            final_status=final_state.status.value,
            step_count=final_state.step_count,
            output_fingerprint=(
                decision_fingerprint(final_state.output)
                if final_state.output is not None
                else None
            ),
            artifact_verified=True,
            evaluation_available=True,
        )
        payload = ExperienceAssessmentRequest(
            execution_id=final_state.run_id,
            run_id=final_state.run_id,
            task_id=final_state.task.task_id,
            related_decision_ids=(artifact.source_decision_id,),
            source_artifacts=(artifact,),
            evaluation_refs=evaluations,
            source_memories=memory_sources,
            runtime_observation=observation,
            execution_outcome=(
                ExperienceExecutionOutcome.SUCCEEDED
                if succeeded
                else ExperienceExecutionOutcome.FAILED
            ),
            success_signal=succeeded,
            failure_signal=not succeeded,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {
                    "state": observation,
                    "artifact": artifact,
                    "evaluations": evaluations,
                    "memories": memory_sources,
                }
            ),
            state_revision=final_state.revision,
            configuration_revision=1,
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=EXPERIENCE_ASSESSMENT_DECISION_TYPE,
            target=DecisionTarget(
                target_type="experience_metadata",
                target_id=f"{final_state.run_id}:{assessment_revision}",
            ),
            correlation=DecisionCorrelation(
                run_id=final_state.run_id,
                task_id=final_state.task.task_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(EXPERIENCE_METADATA_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="experience-runtime-outcome",
                    description="Runtime observations are authoritative over Agent opinion.",
                ),
                DecisionConstraint(
                    constraint_id="experience-append-only",
                    description="Experience Metadata is append-only and cannot mutate Memory.",
                ),
            ),
            evidence=tuple(
                [
                    *(
                        DecisionEvidenceReference(
                            evidence_id=f"evaluation:{item.evaluation_id}",
                            kind="runtime.evaluation",
                            source="evaluation_runtime",
                            reliability=0.9,
                            summary="Persisted Decision input derived from Runtime Evaluation.",
                        )
                        for item in evaluations
                    ),
                    DecisionEvidenceReference(
                        evidence_id=f"artifact:{artifact.effect_fingerprint}",
                        kind="workspace.artifact_receipt",
                        source="persistence",
                        reliability=1.0,
                        summary="Verified committed report Artifact.",
                    ),
                ]
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
                max_revision_count=0,
                max_elapsed_seconds=self._timeout_seconds,
            ),
        )
        agent_request = ExperienceAssessmentAgentRequest(
            execution_summary={
                "final_status": final_state.status.value,
                "step_count": final_state.step_count,
                "artifact_verified": True,
            },
            evaluation_summary=tuple(
                {
                    "scope": item.scope,
                    "verdict": item.verdict,
                    "score": item.score,
                }
                for item in evaluations
            ),
            artifact_summary=(
                {
                    "artifact_type": artifact.artifact_type,
                    "verified": True,
                },
            ),
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="experience-assessment-input",
                    source_type=EXPERIENCE_ASSESSMENT_INPUT_SOURCE_TYPE,
                    agent_scope="experience_assessment",
                    content=agent_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=256,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.experience_assessment.context",
            version="1",
            agent_scope="experience_assessment",
            allowed_decision_types=frozenset({EXPERIENCE_ASSESSMENT_DECISION_TYPE}),
            allowed_source_types=frozenset(
                {EXPERIENCE_ASSESSMENT_INPUT_SOURCE_TYPE}
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {
                    "api_key",
                    "authorization",
                    "credential",
                    "password",
                    "secret",
                    "token",
                    "memory_id",
                    "governance",
                    "policy",
                }
            ),
            max_items=1,
            max_context_tokens=512,
        )
        coordinator, recorder, issuer = self._coordinator(request, basis)
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        result = await self._result(checkpoint, recorder, issuer)
        if result is None:
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Experience Assessment did not apply: {reason}")
        return result

    async def resume(
        self,
        run_id: UUID,
        *,
        assessment_revision: int = 1,
    ) -> ResearchExperienceAssessmentResult | None:
        request_id = stable_experience_request_id(run_id, assessment_revision)
        checkpoint = await self._checkpoints.load(request_id)
        if checkpoint is None:
            return None
        coordinator, recorder, issuer = self._coordinator(
            checkpoint.request,
            checkpoint.request.basis,
        )
        resumed = await coordinator.resume(request_id)
        return await self._result(resumed, recorder, issuer)

    def _coordinator(
        self,
        request: DecisionRequest[ExperienceAssessmentRequest],
        basis: DecisionBasis,
    ) -> tuple[
        DecisionLifecycleCoordinator[
            ExperienceAssessmentRequest,
            ExperienceAssessmentDraft,
            ExperienceMetadataEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        recorder = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)

        async def forbidden_apply(effect: ExperienceMetadataEffect) -> JsonValue:
            del effect
            raise RuntimeError("Experience commit requires a Runtime Permit")

        async def apply_authorized(
            normalized: NormalizedDecisionEffect[ExperienceMetadataEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            metadata = await self._metadata_store.commit(
                normalized.payload,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            return self._metadata_result(metadata)

        async def reconcile(
            normalized: NormalizedDecisionEffect[ExperienceMetadataEffect],
        ) -> DecisionReconciliation:
            try:
                metadata = await self._metadata_store.load_by_effect(
                    normalized.effect_fingerprint
                )
            except Exception as exc:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=(
                        "Experience read-back is indeterminate: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if metadata is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Experience Metadata has not been committed",
                )
            effect = normalized.payload
            if (
                metadata.experience_id != effect.experience_id
                or metadata.source_run_id != effect.source_run_id
                or metadata.source_decision_ids != effect.source_decision_ids
                or metadata.effect_fingerprint != normalized.effect_fingerprint
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Experience Metadata conflicts with the authorized Effect",
                )
            result = self._metadata_result(metadata)
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Experience Metadata was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=decision_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            ExperienceAssessmentRequest,
            ExperienceAssessmentDraft,
            ExperienceMetadataEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=_ExperienceProposalProducer(
                self._capability,
                timeout_seconds=self._timeout_seconds,
                correlation=InferenceCorrelation(
                    run_id=request.correlation.run_id,
                    task_id=request.correlation.task_id,
                ),
            ),
            basis_provider=_ExperienceBasisProvider(self._state_store, basis),
            validator=RuntimeDecisionValidator(
                normalizer=_ExperienceEffectNormalizer()
            ),
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
                ExperienceAssessmentRequest,
                ExperienceAssessmentDraft,
                ExperienceMetadataEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
            fault_injector=self._fault_injector,
        )
        return coordinator, recorder, issuer

    async def _result(
        self,
        checkpoint: DecisionCheckpoint[
            ExperienceAssessmentRequest,
            ExperienceAssessmentDraft,
            ExperienceMetadataEffect,
        ],
        recorder: RecordingGovernanceEvaluator,
        issuer: RecordingAuthorizationIssuer,
    ) -> ResearchExperienceAssessmentResult | None:
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            return None
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
            or checkpoint.proposal is None
        ):
            return None
        normalized = checkpoint.validated_decision.normalized_effect
        metadata = await self._metadata_store.load_by_effect(
            normalized.effect_fingerprint
        )
        if metadata is None:
            raise RuntimeError("APPLIED Experience Decision has no Metadata")
        governance_record = (
            GovernanceRecord(
                scenario="experience_assessment",
                request=recorder.request,
                preliminary=recorder.preliminary,
                final=recorder.final,
                authorization=issuer.authorization,
                review=None,
            )
            if (
                recorder.request is not None
                and recorder.preliminary is not None
                and recorder.final is not None
            )
            else None
        )
        return ResearchExperienceAssessmentResult(
            metadata=metadata,
            governance_record=governance_record,
            request_id=checkpoint.request.request_id,
            proposal=checkpoint.proposal.payload,
        )

    async def _validate_sources(
        self,
        final_state: AgentState,
        evaluation: EvaluationReport,
        artifact: WorkspaceArtifactCommitReceipt,
    ) -> None:
        if final_state.status not in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.TERMINATED,
        }:
            raise ValueError("Experience requires a terminal persisted execution")
        if evaluation.run_id != final_state.run_id:
            raise ValueError("Experience Evaluation belongs to another run")
        if evaluation.task_id != final_state.task.task_id:
            raise ValueError("Experience Evaluation belongs to another task")
        if artifact.run_id != final_state.run_id:
            raise ValueError("Experience Artifact belongs to another run")
        if artifact.source_decision_request_id is None:
            raise ValueError("Experience Artifact has no source Decision")
        persisted_state = await self._state_store.load(final_state.run_id)
        if (
            persisted_state is None
            or persisted_state.revision != final_state.revision
            or decision_fingerprint(persisted_state)
            != decision_fingerprint(final_state)
        ):
            raise ValueError("Experience execution outcome is not persisted")
        committed = self._artifact_readback.load(
            run_id=artifact.run_id,
            node_id=artifact.node_id,
            artifact_type=artifact.artifact_type,
        )
        if committed is None or committed[1] != artifact:
            raise ValueError("Experience Artifact is not committed or verified")

    @staticmethod
    def _metadata_result(metadata: ExperienceMetadata) -> JsonValue:
        return {
            "experience_id": str(metadata.experience_id),
            "version": metadata.version,
            "effect_fingerprint": metadata.effect_fingerprint,
            "metadata_fingerprint": decision_fingerprint(metadata),
        }
