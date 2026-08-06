"""Research Application composition for governed Context compression Decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from adaptive_agent_runtime import TraceSink
from adaptive_agent_runtime.context_memory import (
    CONTEXT_COMPRESSION_APPLY_OPERATION,
    CONTEXT_COMPRESSION_DECISION_TYPE,
    ContextAssembly,
    ContextArchive,
    ContextCompressionCommitter,
    ContextCompressionDecisionPayload,
    ContextCompressionEffect,
    ContextCompressionExecutionPolicy,
    ContextCompressionResult,
    ContextLayer,
    ContextLifecycleState,
    ContextRequirement,
    ContextSnapshotConflictError,
    ContextStore,
    ContextUnit,
    ResidencyPolicy,
    stable_context_compression_id,
)
from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionBasis,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionCheckpointStore,
    DecisionConstraint,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionLifecycleCoordinator,
    DecisionRequest,
    DecisionApplyReceipt,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionResultStatus,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    NormalizedDecisionEffect,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorization,
    GovernanceAuthorizationIssuer,
    GovernanceDecision,
    GovernanceEvaluator,
    GovernanceTarget,
    GovernanceRequest,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewDecision,
    HumanReviewService,
    ReviewOutcome,
    ReviewRequest,
    RuntimeDecisionGovernanceAdapter,
    RuntimeCommitPermit,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    CONTEXT_COMPRESSION_INPUT_SOURCE_TYPE,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    CompressedContextDraft,
    CompressionRequest,
    ContextCompressionDecisionProposalProducer,
    ContextCompressionEffectNormalizer,
    InferenceCorrelation,
    LLMContextPackage,
    SemanticCompressionCapability,
    build_context_compression_proposal_request,
)

from applications.research_agent.cognition import ResearchContextProjection
from applications.research_agent.report import GovernanceRecord


if TYPE_CHECKING:
    from applications.research_agent.strategies import ResearchWorkspace


_REDACT_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


class DeterministicCompressionProposalCapability:
    """Offline proposal fallback; it has no Context mutation authority."""

    module_id = "research_agent.compression_proposal.deterministic"
    capability_id = "semantic_compression.deterministic"

    async def compress(
        self,
        request: CompressionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[CompressedContextDraft]:
        del invocation
        return CapabilityTurnResult[CompressedContextDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=CompressedContextDraft(
                content={
                    "summary": (
                        "Research evidence retained as a compact Runtime "
                        "conclusion."
                    ),
                    "source_reference": request.source_reference_id,
                },
                core_conclusions=(
                    "The evidence was consumed by the current research run.",
                ),
                source_reference_ids=(request.source_reference_id,),
                estimated_tokens=request.target_max_tokens,
            ),
        )


class _CurrentCompressionBasisProvider:
    module_id = "research_agent.context_compression.basis"

    def __init__(
        self,
        *,
        store: ContextStore,
        payload: ContextCompressionDecisionPayload,
    ) -> None:
        self._store = store
        self._payload = payload

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        current = await self._store.load(self._payload.context_id)
        if current is None:
            return DecisionBasis(
                snapshot_fingerprint=decision_fingerprint(
                    {"missing_context_id": self._payload.context_id}
                ),
                configuration_revision=1,
            )
        return _compression_basis(
            current,
            self._payload.execution_policy,
            self._payload.target_max_tokens,
        )


class _RecordingGovernanceEvaluator:
    module_id = "research_agent.context_compression.governance_recorder"

    def __init__(self, delegate: GovernanceEvaluator) -> None:
        self._delegate = delegate
        self.request: GovernanceRequest | None = None
        self.preliminary: GovernanceDecision | None = None
        self.final: GovernanceDecision | None = None
        self.review: ReviewRequest | None = None

    def evaluate(self, request: GovernanceRequest) -> GovernanceDecision:
        decision = self._delegate.evaluate(request)
        self.request = request
        self.preliminary = decision
        self.final = decision
        return decision

    def finalize_review(
        self,
        request: GovernanceRequest,
        review: ReviewRequest,
    ) -> GovernanceDecision:
        decision = self._delegate.finalize_review(request, review)
        self.review = review
        self.final = decision
        return decision


class _RecordingAuthorizationIssuer:
    module_id = "research_agent.context_compression.authorization_recorder"

    def __init__(self, delegate: GovernanceAuthorizationIssuer) -> None:
        self._delegate = delegate
        self.authorization: GovernanceAuthorization | None = None

    def issue(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
    ) -> GovernanceAuthorization:
        authorization = self._delegate.issue(request, decision)
        self.authorization = authorization
        return authorization


def _compression_basis(
    unit: ContextUnit,
    policy: ContextCompressionExecutionPolicy,
    target_max_tokens: int,
) -> DecisionBasis:
    return DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {
                "source_context": unit,
                "target_max_tokens": target_max_tokens,
                "execution_policy": policy,
                "configuration_revision": 1,
            }
        ),
        state_revision=unit.revision,
        configuration_revision=1,
    )


def _single_unit_assembly(unit: ContextUnit) -> ContextAssembly:
    working = (unit,) if unit.metadata.layer is ContextLayer.WORKING else ()
    task = (unit,) if unit.metadata.layer is ContextLayer.TASK else ()
    semantic = (unit,) if unit.metadata.layer is ContextLayer.SEMANTIC else ()
    return ContextAssembly(
        requirement=ContextRequirement(
            run_id=unit.metadata.run_id,
            task_id=unit.metadata.task_id,
            node_id=unit.metadata.node_id,
            goal="Compress one Runtime-approved Context Unit.",
            layers=(unit.metadata.layer,),
            required_context_ids=(unit.context_id,),
            max_units=1,
            max_tokens=unit.metadata.estimated_tokens,
        ),
        units=(unit,),
        working_context=working,
        task_context=task,
        semantic_context=semantic,
        used_tokens=unit.metadata.estimated_tokens,
    )


class ResearchContextCompressionDecisionHandler:
    """ContextCompressor implementation backed by the Decision Lifecycle."""

    module_id = "research_agent.context_compression_decision_handler"

    def __init__(
        self,
        *,
        capability: SemanticCompressionCapability,
        store: ContextStore,
        archive: ContextArchive,
        execution_policy: ContextCompressionExecutionPolicy,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        workspace: ResearchWorkspace,
        context_projection: ResearchContextProjection | None = None,
        checkpoint_store: DecisionCheckpointStore[
            ContextCompressionDecisionPayload,
            CompressedContextDraft,
            ContextCompressionEffect,
        ]
        | None = None,
    ) -> None:
        self._capability = capability
        self._store = store
        self._committer = ContextCompressionCommitter(store=store, archive=archive)
        self._execution_policy = execution_policy
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._workspace = workspace
        self._context_projection = context_projection
        self._checkpoint_store = (
            checkpoint_store
            or InMemoryDecisionCheckpointStore[
                ContextCompressionDecisionPayload,
                CompressedContextDraft,
                ContextCompressionEffect,
            ]()
        )

    async def compress(self, unit: ContextUnit) -> ContextCompressionResult:
        self._validate_source(unit)
        target_max_tokens = self._execution_policy.target_tokens(
            unit.metadata.estimated_tokens
        )
        agent_input, package = self._project_agent_input(unit)
        if package is not None:
            self._workspace.llm_context_packages.append(package)
        payload = ContextCompressionDecisionPayload(
            context_id=unit.context_id,
            source_revision=unit.revision,
            source_snapshot_fingerprint=decision_fingerprint(unit),
            original_estimated_tokens=unit.metadata.estimated_tokens,
            target_max_tokens=target_max_tokens,
            agent_input=agent_input,
            execution_policy=self._execution_policy,
        )
        basis = _compression_basis(
            unit,
            self._execution_policy,
            target_max_tokens,
        )
        evidence_id = f"context:{unit.context_id}:revision:{unit.revision}"
        request_id = stable_context_compression_id(
            "decision-request",
            unit.context_id,
            unit.revision,
            basis.snapshot_fingerprint,
        )
        request = DecisionRequest[ContextCompressionDecisionPayload](
            request_id=request_id,
            decision_type=CONTEXT_COMPRESSION_DECISION_TYPE,
            target=DecisionTarget(
                target_type="context_unit",
                target_id=str(unit.context_id),
            ),
            correlation=DecisionCorrelation(
                run_id=unit.metadata.run_id,
                task_id=unit.metadata.task_id,
                node_id=unit.metadata.node_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(CONTEXT_COMPRESSION_APPLY_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="compression-source",
                    description=(
                        "Compress only the single Runtime-projected source Context."
                    ),
                ),
                DecisionConstraint(
                    constraint_id="compression-token-budget",
                    description=(
                        "Return a smaller semantic representation within the "
                        "Runtime token target."
                    ),
                    value=target_max_tokens,
                ),
            ),
            evidence=(
                DecisionEvidenceReference(
                    evidence_id=evidence_id,
                    kind="context.snapshot",
                    source="context_runtime",
                    reliability=1.0,
                    summary=(
                        "The Runtime supplied one immutable Context Unit revision."
                    ),
                ),
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=self._execution_policy.max_agent_calls,
                max_revision_count=self._execution_policy.max_revision_count,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        projected_request = build_context_compression_proposal_request(payload)
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="context-compression-input",
                    source_type=CONTEXT_COMPRESSION_INPUT_SOURCE_TYPE,
                    agent_scope="context_compression",
                    content=projected_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    evidence_id=evidence_id,
                    priority=100,
                    estimated_tokens=unit.metadata.estimated_tokens,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.context_compression.context",
            version="1",
            agent_scope="context_compression",
            allowed_decision_types=frozenset(
                {CONTEXT_COMPRESSION_DECISION_TYPE}
            ),
            allowed_source_types=frozenset(
                {CONTEXT_COMPRESSION_INPUT_SOURCE_TYPE}
            ),
            allowed_sensitivity_levels=frozenset(
                {ContextSensitivity.INTERNAL}
            ),
            redact_keys=_REDACT_KEYS,
            max_items=1,
            max_context_tokens=max(1, unit.metadata.estimated_tokens),
        )
        recording_evaluator = _RecordingGovernanceEvaluator(self._governance)
        recording_issuer = _RecordingAuthorizationIssuer(self._issuer)
        governance_adapter = RuntimeDecisionGovernanceAdapter[
            ContextCompressionDecisionPayload,
            CompressedContextDraft,
            ContextCompressionEffect,
        ](
            evaluator=recording_evaluator,
            authorization_issuer=recording_issuer,
            review_service=self._reviews,
        )

        async def apply_effect(effect: ContextCompressionEffect) -> JsonValue:
            await self._validate_current_effect(effect)
            return {
                "context_id": str(effect.context_id),
                "source_revision": effect.source_revision,
                "original_estimated_tokens": effect.original_estimated_tokens,
                "estimated_tokens": effect.estimated_tokens,
            }

        async def apply_authorized_effect(
            normalized: NormalizedDecisionEffect[ContextCompressionEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            effect = normalized.payload
            await self._validate_current_effect(effect)
            current = await self._store.load(effect.context_id)
            if current is None:
                raise ContextSnapshotConflictError("Context source disappeared")
            committed = await self._committer.commit(
                current,
                effect,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            return {
                "context_id": str(committed.context_id),
                "source_revision": effect.source_revision,
                "committed_revision": committed.revision,
                "original_estimated_tokens": effect.original_estimated_tokens,
                "estimated_tokens": committed.metadata.estimated_tokens,
            }

        async def reconcile_effect(
            normalized: NormalizedDecisionEffect[ContextCompressionEffect],
        ) -> DecisionReconciliation:
            effect = normalized.payload
            committed = await self._committer.load_effect(
                effect.context_id,
                normalized.effect_fingerprint,
                source_fingerprint=effect.source_snapshot_fingerprint,
            )
            if committed is not None:
                if (
                    committed.revision != effect.source_revision + 1
                    or committed.metadata.estimated_tokens != effect.estimated_tokens
                    or committed.content != effect.content
                    or committed.recovery_reference is None
                ):
                    return DecisionReconciliation(
                        status=DecisionReconciliationStatus.UNKNOWN,
                        reason="Context read-back conflicts with the authorized compression",
                    )
                result: JsonValue = {
                    "context_id": str(committed.context_id),
                    "source_revision": effect.source_revision,
                    "committed_revision": committed.revision,
                    "original_estimated_tokens": effect.original_estimated_tokens,
                    "estimated_tokens": committed.metadata.estimated_tokens,
                }
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.COMMITTED,
                    reason="compressed Context was committed and read back",
                    apply_receipt=DecisionApplyReceipt(
                        effect_fingerprint=normalized.effect_fingerprint,
                        committed_state_fingerprint=governance_fingerprint(result),
                        result=result,
                    ),
                )
            current = await self._store.load(effect.context_id)
            if (
                current is not None
                and current.revision == effect.source_revision
                and decision_fingerprint(current) == effect.source_snapshot_fingerprint
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="the authoritative Context remains at its source revision",
                )
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.UNKNOWN,
                reason="Context state cannot prove commit or non-commit",
            )

        coordinator = DecisionLifecycleCoordinator[
            ContextCompressionDecisionPayload,
            CompressedContextDraft,
            ContextCompressionEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=ContextCompressionDecisionProposalProducer(
                capability=self._capability,
                correlation=InferenceCorrelation(
                    run_id=unit.metadata.run_id,
                    task_id=unit.metadata.task_id,
                    node_id=unit.metadata.node_id,
                ),
                timeout_seconds=self._execution_policy.timeout_seconds,
            ),
            basis_provider=_CurrentCompressionBasisProvider(
                store=self._store,
                payload=payload,
            ),
            validator=RuntimeDecisionValidator(
                normalizer=ContextCompressionEffectNormalizer()
            ),
            governance=governance_adapter,
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
                apply_authorized_effect=apply_authorized_effect,
                reconcile_effect=reconcile_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                ContextCompressionDecisionPayload,
                CompressedContextDraft,
                ContextCompressionEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(
            request,
            sources=sources,
            policy=policy,
        )
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = checkpoint.governance_receipt
            if receipt is None or receipt.review_request_id is None:
                raise RuntimeError("Compression review has no Review Request")
            review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale=(
                        "The source-bound compression is reversible through Archive."
                    ),
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            recording_evaluator.review = review
            checkpoint = await coordinator.resume_review(request.request_id)
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
        ):
            reason = (
                checkpoint.result.reason
                if checkpoint.result is not None
                else checkpoint.stage
            )
            raise RuntimeError(f"Context Compression decision did not apply: {reason}")
        governance_request = recording_evaluator.request
        preliminary = recording_evaluator.preliminary
        final = recording_evaluator.final
        if governance_request is not None and preliminary is not None and final is not None:
            self._workspace.governance_records.append(
                GovernanceRecord(
                    scenario="context_compression_agent",
                    request=governance_request,
                    preliminary=preliminary,
                    final=final,
                    authorization=recording_issuer.authorization,
                    review=recording_evaluator.review,
                )
            )
        return checkpoint.validated_decision.normalized_effect.payload.to_result()

    def _project_agent_input(
        self,
        unit: ContextUnit,
    ) -> tuple[JsonValue, LLMContextPackage | None]:
        if self._context_projection is None:
            return (
                {
                    "context_id": str(unit.context_id),
                    "content": unit.model_dump(mode="json")["content"],
                    "source": unit.metadata.source.value,
                    "layer": unit.metadata.layer.value,
                    "source_reference": unit.metadata.source_reference,
                    "tags": list(unit.metadata.tags),
                },
                None,
            )
        package = self._context_projection.project(_single_unit_assembly(unit))
        return package.model_dump(mode="json"), package

    @staticmethod
    def _validate_source(unit: ContextUnit) -> None:
        if unit.lifecycle_state is not ContextLifecycleState.ACTIVE:
            raise ValueError("Compression Agent requires active Context")
        if unit.residency_policy is ResidencyPolicy.PINNED:
            raise ValueError("Pinned Context cannot be compressed")
        if unit.metadata.estimated_tokens <= 1:
            raise ValueError("Context Unit is too small for compression")

    async def _validate_current_effect(
        self,
        effect: ContextCompressionEffect,
    ) -> None:
        current = await self._store.load(effect.context_id)
        if current is None:
            raise ContextSnapshotConflictError(
                "compression source is no longer resident"
            )
        if current.revision != effect.source_revision:
            raise ContextSnapshotConflictError(
                "compression source revision changed before Apply"
            )
        if decision_fingerprint(current) != effect.source_snapshot_fingerprint:
            raise ContextSnapshotConflictError(
                "compression source changed before Apply"
            )
        self._validate_source(current)
        if current.metadata.estimated_tokens != effect.original_estimated_tokens:
            raise ContextSnapshotConflictError(
                "compression source token estimate changed before Apply"
            )
        if (
            effect.estimated_tokens >= current.metadata.estimated_tokens
            or effect.estimated_tokens > effect.target_max_tokens
        ):
            raise ValueError(
                "compression effect does not satisfy the Runtime token reduction"
            )
