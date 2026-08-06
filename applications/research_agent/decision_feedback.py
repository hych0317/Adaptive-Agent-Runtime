"""Governed deterministic Decision outcome feedback for completed research runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import AgentState, RunStatus, StateStore, TraceSink
from adaptive_agent_runtime.context_memory import (
    ExperienceMetadata,
    MemoryRecallBundle,
)
from adaptive_agent_runtime.decision_feedback import (
    DECISION_FEEDBACK_COMMIT_OPERATION,
    DECISION_FEEDBACK_DECISION_TYPE,
    DecisionFeedbackArtifactReference,
    DecisionFeedbackAttributionType,
    DecisionFeedbackDraft,
    DecisionFeedbackEffect,
    DecisionFeedbackEvaluationReference,
    DecisionFeedbackExperienceReference,
    DecisionFeedbackRecallReference,
    DecisionFeedbackRecord,
    DecisionFeedbackRequest,
    DecisionFeedbackRuntimeObservation,
    DecisionFeedbackStore,
    DecisionFeedbackSubjectReference,
    FeedbackRuntimeOutcome,
    stable_feedback_id,
    stable_feedback_request_id,
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
from adaptive_agent_runtime.evaluation import (
    EvaluationReport,
    EvaluationVerdict,
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
from adaptive_agent_runtime.orchestration import PLANNING_DECISION_TYPE
from adaptive_agent_runtime.persistence import (
    SQLiteEvaluationReportStore,
    WorkspaceArtifactCommitReceipt,
)

from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord


FEEDBACK_INPUT_SOURCE_TYPE = "decision_feedback_input"


@dataclass(frozen=True)
class ResearchDecisionFeedbackResult:
    records: tuple[DecisionFeedbackRecord, ...]
    governance_records: tuple[GovernanceRecord, ...]


class _DeterministicFeedbackProducer:
    module_id = "research_agent.decision_feedback.deterministic_builder"

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[DecisionFeedbackDraft]:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == FEEDBACK_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1 or not isinstance(blocks[0].content, Mapping):
            raise ValueError("Feedback Builder requires one isolated Runtime block")
        request = DecisionFeedbackRequest.model_validate(dict(blocks[0].content))
        attribution = (
            DecisionFeedbackAttributionType.INSUFFICIENT_EVIDENCE
            if request.evaluation.verdict == EvaluationVerdict.INCONCLUSIVE.value
            else DecisionFeedbackAttributionType.ASSOCIATED
        )
        if request.recall is None:
            summary = (
                "The completed run outcome is associated with the applied Initial "
                "Planning Decision; no optimality or agent-quality claim is made."
            )
        else:
            summary = (
                "The committed Recall Bundle participated in the applied Planning "
                "Decision; no causal success or failure claim is made."
            )
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    f"decision:{request.subject.decision_id}",
                    f"evaluation:{request.evaluation.report_id}",
                    f"experience:{request.experience.experience_id}",
                    f"artifact:{request.artifact.effect_fingerprint}",
                    *(
                        (f"recall-bundle:{request.recall.bundle_id}",)
                        if request.recall is not None
                        else ()
                    ),
                )
            )
        )
        draft = DecisionFeedbackDraft(
            attribution_type=attribution,
            summary=summary,
            evidence_refs=evidence_refs,
        )
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self.module_id,
                    capability="deterministic_decision_feedback",
                    implementation_version="phase-3c",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=DECISION_FEEDBACK_COMMIT_OPERATION,
                payload=draft,
                rationale=draft.summary,
                evidence_refs=(),
                confidence=1.0,
            )
        )


class _FeedbackEffectNormalizer:
    module_id = "research_agent.decision_feedback.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[DecisionFeedbackRequest],
        proposal: DecisionProposal[DecisionFeedbackDraft],
    ) -> NormalizedDecisionEffect[DecisionFeedbackEffect]:
        source = request.payload
        effect = DecisionFeedbackEffect(
            feedback_id=stable_feedback_id(request.request_id),
            source_run_id=source.source_run_id,
            source_task_id=source.source_task_id,
            subject=source.subject,
            planning_subject=source.planning_subject,
            evaluation=source.evaluation,
            experience=source.experience,
            artifact=source.artifact,
            runtime_observation=source.runtime_observation,
            runtime_outcome=source.runtime_outcome,
            evaluation_verdict=source.evaluation.verdict,
            evidence_refs=proposal.payload.evidence_refs,
            attribution_type=proposal.payload.attribution_type,
            recall=source.recall,
            proposal_fingerprint=decision_fingerprint(proposal.payload),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[DecisionFeedbackEffect].create(
            payload=effect,
            operation=DECISION_FEEDBACK_COMMIT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.1,
            reversible=False,
            impact_description=(
                "Append non-behavioral outcome attribution without mutating sources."
            ),
        )


class _FeedbackBasisProvider:
    module_id = "research_agent.decision_feedback.basis"

    def __init__(self, state_store: StateStore, basis: DecisionBasis) -> None:
        self._state_store = state_store
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        state = await self._state_store.load(request.correlation.run_id)
        observation = request.payload.runtime_observation
        if (
            state is not None
            and state.revision == observation.state_revision
            and decision_fingerprint(state) == observation.state_fingerprint
        ):
            return self._basis
        return self._basis.model_copy(
            update={"snapshot_fingerprint": decision_fingerprint(state)}
        )


class ResearchDecisionFeedbackHandler:
    module_id = "research_agent.decision_feedback"

    def __init__(
        self,
        *,
        state_store: StateStore,
        evaluation_store: SQLiteEvaluationReportStore,
        feedback_store: DecisionFeedbackStore,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        checkpoint_store: DecisionCheckpointStore[
            DecisionFeedbackRequest,
            DecisionFeedbackDraft,
            DecisionFeedbackEffect,
        ]
        | None = None,
        fault_injector: Callable[[DecisionFaultPoint, Any], None] | None = None,
    ) -> None:
        self._state_store = state_store
        self._evaluation_store = evaluation_store
        self._feedback_store = feedback_store
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._checkpoints = checkpoint_store or InMemoryDecisionCheckpointStore()
        self._fault_injector = fault_injector

    async def record_for_run(
        self,
        *,
        final_state: AgentState,
        evaluation: EvaluationReport,
        artifact_receipt: WorkspaceArtifactCommitReceipt,
        experience: ExperienceMetadata,
        recall_bundle: MemoryRecallBundle | None,
    ) -> ResearchDecisionFeedbackResult:
        await self._validate_sources(
            final_state, evaluation, artifact_receipt, experience
        )
        planning = await self._feedback_store.find_applied_subject(
            final_state.run_id, PLANNING_DECISION_TYPE
        )
        if planning is None:
            raise ValueError("Planning Feedback requires an APPLIED Planning Decision")
        subjects = [planning]
        if recall_bundle is not None:
            recall = await self._feedback_store.find_applied_subject(
                final_state.run_id, "memory.recall"
            )
            if recall is None:
                raise ValueError("Recall Feedback requires an APPLIED Recall Decision")
            subjects.append(recall)
        records: list[DecisionFeedbackRecord] = []
        governance_records: list[GovernanceRecord] = []
        for subject in subjects:
            result = await self._record_one(
                final_state=final_state,
                evaluation=evaluation,
                artifact_receipt=artifact_receipt,
                experience=experience,
                subject=subject,
                planning=planning,
                recall_bundle=(
                    recall_bundle if subject.decision_type == "memory.recall" else None
                ),
            )
            records.append(result[0])
            if result[1] is not None:
                governance_records.append(result[1])
        return ResearchDecisionFeedbackResult(
            records=tuple(records), governance_records=tuple(governance_records)
        )

    async def _record_one(
        self,
        *,
        final_state: AgentState,
        evaluation: EvaluationReport,
        artifact_receipt: WorkspaceArtifactCommitReceipt,
        experience: ExperienceMetadata,
        subject: DecisionFeedbackSubjectReference,
        planning: DecisionFeedbackSubjectReference,
        recall_bundle: MemoryRecallBundle | None,
    ) -> tuple[DecisionFeedbackRecord, GovernanceRecord | None]:
        request_id = stable_feedback_request_id(final_state.run_id, subject.decision_id)
        existing = await self._checkpoints.load(request_id)
        if existing is not None:
            coordinator, recorder, issuer = self._coordinator(
                existing.request, existing.request.basis
            )
            checkpoint = await coordinator.resume(request_id)
            result = await self._result(checkpoint, recorder, issuer)
            if result is None:
                reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
                raise RuntimeError(f"Decision Feedback did not apply: {reason}")
            return result

        evaluation_ref = DecisionFeedbackEvaluationReference(
            report_id=evaluation.report_id,
            trace_id=evaluation.trace_id,
            run_id=evaluation.run_id,
            task_id=evaluation.task_id,
            outcome_evaluation_id=evaluation.outcome.evaluation_id,
            outcome_fingerprint=decision_fingerprint(evaluation.outcome),
            report_fingerprint=decision_fingerprint(evaluation),
            verdict=evaluation.outcome.verdict.value,
            score=evaluation.outcome.score,
        )
        artifact_ref = DecisionFeedbackArtifactReference(
            node_id=artifact_receipt.node_id,
            artifact_type=artifact_receipt.artifact_type,
            effect_fingerprint=artifact_receipt.effect_fingerprint,
            artifact_fingerprint=artifact_receipt.artifact_fingerprint,
            source_decision_id=(
                artifact_receipt.source_decision_request_id
                if artifact_receipt.source_decision_request_id is not None
                else _missing_artifact_decision()
            ),
        )
        experience_ref = DecisionFeedbackExperienceReference(
            experience_id=experience.experience_id,
            version=experience.version,
            effect_fingerprint=experience.effect_fingerprint,
            metadata_fingerprint=decision_fingerprint(experience),
        )
        retry_count = _metric_int(evaluation.trajectory.metrics, "retry_count")
        failure_count = sum(
            1
            for item in evaluation.results
            for finding in item.findings
            if finding.severity.value in {"error", "critical"}
        )
        observation = DecisionFeedbackRuntimeObservation(
            state_revision=final_state.revision,
            state_fingerprint=decision_fingerprint(final_state),
            final_status=final_state.status.value,
            failure_count=failure_count,
            retry_count=retry_count,
            artifact_committed=True,
        )
        recall_ref = (
            DecisionFeedbackRecallReference(
                recall_decision_id=subject.decision_id,
                recall_effect_fingerprint=subject.effect_fingerprint,
                bundle_id=recall_bundle.bundle_id,
                bundle_fingerprint=decision_fingerprint(recall_bundle),
                planning_decision_id=planning.decision_id,
                planning_effect_fingerprint=planning.effect_fingerprint,
            )
            if recall_bundle is not None
            else None
        )
        payload = DecisionFeedbackRequest(
            source_run_id=final_state.run_id,
            source_task_id=final_state.task.task_id,
            subject=subject,
            planning_subject=planning,
            evaluation=evaluation_ref,
            experience=experience_ref,
            artifact=artifact_ref,
            runtime_observation=observation,
            runtime_outcome=(
                FeedbackRuntimeOutcome.SUCCEEDED
                if final_state.status is RunStatus.COMPLETED
                else FeedbackRuntimeOutcome.FAILED
            ),
            recall=recall_ref,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(payload),
            state_revision=final_state.revision,
            configuration_revision=1,
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=DECISION_FEEDBACK_DECISION_TYPE,
            target=DecisionTarget(
                target_type="decision_feedback",
                target_id=f"{final_state.run_id}:{subject.decision_id}",
            ),
            correlation=DecisionCorrelation(
                run_id=final_state.run_id,
                task_id=final_state.task.task_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(DECISION_FEEDBACK_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="feedback-association-only",
                    description="Feedback cannot claim causality, reward, or policy quality.",
                ),
                DecisionConstraint(
                    constraint_id="feedback-append-only",
                    description="Feedback cannot mutate any source object or behavior.",
                ),
            ),
            evidence=(
                DecisionEvidenceReference(
                    evidence_id=f"decision:{subject.decision_id}",
                    kind="decision.applied",
                    source="decision_checkpoint",
                    reliability=1.0,
                    summary="Persisted APPLIED subject Decision.",
                ),
                DecisionEvidenceReference(
                    evidence_id=f"evaluation:{evaluation.report_id}",
                    kind="evaluation.report",
                    source="evaluation_store",
                    reliability=1.0,
                    summary="Persisted deterministic Evaluation report.",
                ),
                DecisionEvidenceReference(
                    evidence_id=f"experience:{experience.experience_id}",
                    kind="experience.metadata",
                    source="experience_store",
                    reliability=1.0,
                    summary="Committed Runtime-evidenced Experience Metadata.",
                ),
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
                max_revision_count=0,
                max_elapsed_seconds=5.0,
            ),
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="decision-feedback-input",
                    source_type=FEEDBACK_INPUT_SOURCE_TYPE,
                    agent_scope="deterministic_feedback_builder",
                    content=payload.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=512,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.decision_feedback.context",
            version="1",
            agent_scope="deterministic_feedback_builder",
            allowed_decision_types=frozenset({DECISION_FEEDBACK_DECISION_TYPE}),
            allowed_source_types=frozenset({FEEDBACK_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {"api_key", "authorization", "credential", "password", "secret", "token"}
            ),
            max_items=1,
            max_context_tokens=1024,
        )
        coordinator, recorder, issuer = self._coordinator(request, basis)
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        result = await self._result(checkpoint, recorder, issuer)
        if result is None:
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Decision Feedback did not apply: {reason}")
        return result

    def _coordinator(
        self,
        request: DecisionRequest[DecisionFeedbackRequest],
        basis: DecisionBasis,
    ) -> tuple[
        DecisionLifecycleCoordinator[
            DecisionFeedbackRequest,
            DecisionFeedbackDraft,
            DecisionFeedbackEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        recorder = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)

        async def forbidden_apply(effect: DecisionFeedbackEffect) -> JsonValue:
            del effect
            raise RuntimeError("Decision Feedback commit requires a Runtime Permit")

        async def apply_authorized(
            normalized: NormalizedDecisionEffect[DecisionFeedbackEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            record = await self._feedback_store.commit(
                normalized.payload,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            readback = await self._feedback_store.load_by_effect(
                normalized.effect_fingerprint
            )
            if readback is None or readback != record:
                raise RuntimeError("Decision Feedback commit read-back failed")
            return self._record_result(readback)

        async def reconcile(
            normalized: NormalizedDecisionEffect[DecisionFeedbackEffect],
        ) -> DecisionReconciliation:
            try:
                record = await self._feedback_store.load_by_effect(
                    normalized.effect_fingerprint
                )
            except Exception as exc:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=f"Feedback read-back is indeterminate: {type(exc).__name__}: {exc}",
                )
            if record is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Decision Feedback has not been committed",
                )
            effect = normalized.payload
            if (
                record.feedback_id != effect.feedback_id
                or record.subject_decision_id != effect.subject.decision_id
                or record.effect_fingerprint != normalized.effect_fingerprint
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Decision Feedback conflicts with the authorized Effect",
                )
            result = self._record_result(record)
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Decision Feedback was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=decision_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            DecisionFeedbackRequest,
            DecisionFeedbackDraft,
            DecisionFeedbackEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=_DeterministicFeedbackProducer(),
            basis_provider=_FeedbackBasisProvider(self._state_store, basis),
            validator=RuntimeDecisionValidator(normalizer=_FeedbackEffectNormalizer()),
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
                DecisionFeedbackRequest,
                DecisionFeedbackDraft,
                DecisionFeedbackEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
            fault_injector=self._fault_injector,
        )
        return coordinator, recorder, issuer

    async def _result(
        self,
        checkpoint: DecisionCheckpoint[
            DecisionFeedbackRequest,
            DecisionFeedbackDraft,
            DecisionFeedbackEffect,
        ],
        recorder: RecordingGovernanceEvaluator,
        issuer: RecordingAuthorizationIssuer,
    ) -> tuple[DecisionFeedbackRecord, GovernanceRecord | None] | None:
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            return None
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
        ):
            return None
        normalized = checkpoint.validated_decision.normalized_effect
        record = await self._feedback_store.load_by_effect(
            normalized.effect_fingerprint
        )
        if record is None:
            raise RuntimeError("APPLIED Decision Feedback has no committed Record")
        governance_record = (
            GovernanceRecord(
                scenario="decision_feedback",
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
        return record, governance_record

    async def _validate_sources(
        self,
        state: AgentState,
        evaluation: EvaluationReport,
        artifact: WorkspaceArtifactCommitReceipt,
        experience: ExperienceMetadata,
    ) -> None:
        if state.status not in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise ValueError("Decision Feedback requires a completed Run")
        if evaluation.run_id != state.run_id or evaluation.task_id != state.task.task_id:
            raise ValueError("Decision Feedback Evaluation identity is mismatched")
        if artifact.run_id != state.run_id:
            raise ValueError("Decision Feedback Artifact belongs to another run")
        if artifact.source_decision_request_id is None:
            raise ValueError("Decision Feedback Artifact lacks provenance")
        if experience.source_run_id != state.run_id:
            raise ValueError("Decision Feedback Experience belongs to another run")
        persisted_state = await self._state_store.load(state.run_id)
        if persisted_state is None or decision_fingerprint(persisted_state) != decision_fingerprint(state):
            raise ValueError("Decision Feedback Run outcome is not persisted")
        persisted_evaluation = await self._evaluation_store.load(evaluation.report_id)
        if persisted_evaluation is None or persisted_evaluation != evaluation:
            raise ValueError("Decision Feedback Evaluation is not persisted")

    @staticmethod
    def _record_result(record: DecisionFeedbackRecord) -> JsonValue:
        return {
            "feedback_id": str(record.feedback_id),
            "version": record.version,
            "subject_decision_id": str(record.subject_decision_id),
            "effect_fingerprint": record.effect_fingerprint,
            "record_fingerprint": decision_fingerprint(record),
        }


def _metric_int(metrics: Mapping[str, JsonValue], key: str) -> int:
    value = metrics.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _missing_artifact_decision() -> UUID:
    raise ValueError("Decision Feedback Artifact lacks a source Decision")
