"""Research composition for governed Agent-proposed Memory extraction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import TraceSink
from adaptive_agent_runtime.context_memory import (
    MEMORY_CANDIDATES_APPLY_OPERATION,
    MEMORY_EXTRACTION_DECISION_TYPE,
    ContextAssembly,
    ContextLayer,
    ContextRequirement,
    ContextSource,
    EvidenceDrivenMemoryConsolidator,
    ExistingMemoryBinding,
    MemoryEvidenceBinding,
    MemoryExtractionDecisionOutcome,
    MemoryExtractionDecisionPayload,
    MemoryExtractionEffect,
    MemoryExtractionExecutionPolicy,
    MemoryStore,
    MemoryUpdateResult,
    stable_memory_extraction_id,
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
from adaptive_agent_runtime.evaluation import ContextMemoryTraceAdapter, EvaluationCorrelation
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
from adaptive_agent_runtime.llm import (
    MEMORY_EXTRACTION_EVIDENCE_SOURCE_TYPE,
    MEMORY_EXTRACTION_INPUT_SOURCE_TYPE,
    EvidenceReference,
    InferenceCorrelation,
    MemoryCandidateBatchDraft,
    MemoryExtractionCapability,
    MemoryExtractionEffectNormalizer,
    MemoryExtractionProposalProducer,
    MemoryExtractionRequest,
)

from applications.research_agent.cognition import ResearchContextProjection
from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord


if TYPE_CHECKING:
    from applications.research_agent.strategies import ResearchWorkspace


_REDACT_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


class _CurrentMemoryExtractionBasisProvider:
    module_id = "research_agent.memory_extraction.basis"

    def __init__(
        self,
        *,
        store: MemoryStore,
        payload: MemoryExtractionDecisionPayload,
    ) -> None:
        self._store = store
        self._payload = payload

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        current = await self._store.list_all()
        return _memory_basis(
            state_revision=self._payload.state_revision,
            evidence=self._payload.evidence,
            existing=current,
            execution_policy=self._payload.execution_policy,
        )


def _memory_basis(
    *,
    state_revision: int,
    evidence: tuple[MemoryEvidenceBinding, ...],
    existing: tuple[Any, ...],
    execution_policy: MemoryExtractionExecutionPolicy,
) -> DecisionBasis:
    return DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {
                "state_revision": state_revision,
                "evidence": evidence,
                "existing_memories": existing,
                "execution_policy": execution_policy,
            }
        ),
        state_revision=state_revision,
    )


class ResearchMemoryExtractionDecisionHandler:
    module_id = "research_agent.memory_extraction_decision_handler"

    def __init__(
        self,
        *,
        capability: MemoryExtractionCapability,
        context_projection: ResearchContextProjection,
        memory_store: MemoryStore,
        consolidator: EvidenceDrivenMemoryConsolidator,
        execution_policy: MemoryExtractionExecutionPolicy,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        workspace: ResearchWorkspace,
        checkpoint_store: DecisionCheckpointStore[
            MemoryExtractionDecisionPayload,
            MemoryCandidateBatchDraft,
            MemoryExtractionEffect,
        ]
        | None = None,
    ) -> None:
        self._capability = capability
        self._context_projection = context_projection
        self._memory_store = memory_store
        self._consolidator = consolidator
        self._execution_policy = execution_policy
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._workspace = workspace
        self._checkpoint_store = checkpoint_store or InMemoryDecisionCheckpointStore()

    async def extract(
        self,
        *,
        run_id: UUID,
        task_id: UUID,
        state_revision: int,
        context_trace: ContextMemoryTraceAdapter,
    ) -> MemoryExtractionDecisionOutcome | None:
        successful = tuple(item for item in self._workspace.tool_observations if item.succeeded)
        if not successful:
            return None
        existing = await self._memory_store.list_all()
        successful_ids = {str(item.invocation_id) for item in successful}
        projected_units = []
        seen_references: set[str] = set()
        for unit in self._workspace.context_units:
            if unit.metadata.source is not ContextSource.OBSERVATION:
                continue
            content = unit.content
            if not isinstance(content, dict):
                continue
            metadata = content.get("metadata")
            tool = metadata.get("tool") if isinstance(metadata, dict) else None
            invocation_id = tool.get("invocation_id") if isinstance(tool, dict) else None
            if not isinstance(invocation_id, str) or invocation_id not in successful_ids:
                continue
            reference = unit.metadata.source_reference or str(unit.context_id)
            if reference not in seen_references:
                seen_references.add(reference)
                projected_units.append(unit)
        for assembly in self._workspace.context_assemblies:
            for unit in assembly.units:
                if unit.metadata.source is not ContextSource.MEMORY_RECALL:
                    continue
                reference = unit.metadata.source_reference or str(unit.context_id)
                if reference not in seen_references:
                    seen_references.add(reference)
                    projected_units.append(unit)
        if not projected_units:
            raise RuntimeError("Memory Extraction has no Runtime Context evidence")
        working = tuple(item for item in projected_units if item.metadata.layer is ContextLayer.WORKING)
        task_units = tuple(item for item in projected_units if item.metadata.layer is ContextLayer.TASK)
        semantic = tuple(item for item in projected_units if item.metadata.layer is ContextLayer.SEMANTIC)
        ordered = (*working, *task_units, *semantic)
        tokens = sum(item.metadata.estimated_tokens for item in ordered)
        assembly = ContextAssembly(
            requirement=ContextRequirement(
                run_id=run_id,
                task_id=task_id,
                goal="Extract evidence-bound Memory Candidates from approved Runtime Context.",
                max_units=len(ordered),
                max_tokens=tokens,
            ),
            units=ordered,
            working_context=working,
            task_context=task_units,
            semantic_context=semantic,
            used_tokens=tokens,
        )
        package = self._context_projection.project(assembly)
        self._workspace.llm_context_packages.append(package)
        evidence_refs = tuple(
            EvidenceReference(
                reference_id=f"tool:{item.invocation_id}",
                kind="tool.observation",
                summary=f"Accepted output from capability '{item.capability_id}'.",
                reliability=1.0,
            )
            for item in successful
        )
        extraction_request = MemoryExtractionRequest(
            observations={
                "context_package": package.model_dump(mode="json"),
                "observation_context_ids": [
                    str(block.context_id)
                    for block in package.blocks
                    if block.source is ContextSource.OBSERVATION
                ],
            },
            evidence_catalog=evidence_refs,
            existing_memories={
                "memory_context_ids": [
                    str(block.context_id)
                    for block in package.blocks
                    if block.source is ContextSource.MEMORY_RECALL
                ],
                "memory_reference_ids": [str(item.memory_id) for item in existing],
            },
            existing_memory_reference_ids=tuple(str(item.memory_id) for item in existing),
        )
        evidence_bindings = tuple(
            MemoryEvidenceBinding(
                reference_id=item.reference_id,
                kind=item.kind,
                summary=item.summary or "Accepted Tool observation.",
                source_reference=item.reference_id,
                reliability=item.reliability or 1.0,
            )
            for item in evidence_refs
        )
        existing_bindings = tuple(
            ExistingMemoryBinding(
                reference_id=str(item.memory_id),
                memory_id=item.memory_id,
                revision=item.revision,
                memory_fingerprint=decision_fingerprint(item),
            )
            for item in existing
        )
        payload = MemoryExtractionDecisionPayload(
            agent_input=extraction_request.model_dump(mode="json"),
            evidence=evidence_bindings,
            existing_memories=existing_bindings,
            state_revision=state_revision,
            execution_policy=self._execution_policy,
        )
        basis = _memory_basis(
            state_revision=state_revision,
            evidence=evidence_bindings,
            existing=existing,
            execution_policy=self._execution_policy,
        )
        request = DecisionRequest[MemoryExtractionDecisionPayload](
            request_id=stable_memory_extraction_id(
                "decision-request", run_id, state_revision, basis.snapshot_fingerprint
            ),
            decision_type=MEMORY_EXTRACTION_DECISION_TYPE,
            target=DecisionTarget(target_type="runtime_memory", target_id=str(run_id)),
            correlation=DecisionCorrelation(run_id=run_id, task_id=task_id),
            basis=basis,
            payload=payload,
            allowed_actions=(MEMORY_CANDIDATES_APPLY_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="memory-evidence",
                    description="Every candidate must cite Runtime-projected evidence.",
                ),
                DecisionConstraint(
                    constraint_id="memory-no-write",
                    description="Agent proposes candidates; Runtime alone consolidates Memory.",
                ),
                DecisionConstraint(
                    constraint_id="memory-candidate-budget",
                    description="Candidate count is bounded by Runtime policy.",
                    value=self._execution_policy.max_candidates,
                ),
            ),
            evidence=tuple(
                DecisionEvidenceReference(
                    evidence_id=item.reference_id,
                    kind=item.kind,
                    source="context_runtime",
                    reliability=item.reliability,
                    summary=item.summary,
                )
                for item in evidence_bindings
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=self._execution_policy.max_agent_calls,
                max_revision_count=self._execution_policy.max_revision_count,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="memory-extraction-input",
                    source_type=MEMORY_EXTRACTION_INPUT_SOURCE_TYPE,
                    agent_scope="memory_extraction",
                    content=extraction_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=max(512, tokens),
                ),
                *(
                    ProjectionSource(
                        source_id=f"memory-extraction-evidence:{item.reference_id}",
                        source_type=MEMORY_EXTRACTION_EVIDENCE_SOURCE_TYPE,
                        agent_scope="memory_extraction",
                        content={
                            "reference_id": item.reference_id,
                            "kind": item.kind,
                            "summary": item.summary,
                            "reliability": item.reliability,
                        },
                        sensitivity=ContextSensitivity.INTERNAL,
                        evidence_id=item.reference_id,
                        priority=90,
                        estimated_tokens=32,
                    )
                    for item in evidence_bindings
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.memory_extraction.context",
            version="1",
            agent_scope="memory_extraction",
            allowed_decision_types=frozenset({MEMORY_EXTRACTION_DECISION_TYPE}),
            allowed_source_types=frozenset(
                {
                    MEMORY_EXTRACTION_INPUT_SOURCE_TYPE,
                    MEMORY_EXTRACTION_EVIDENCE_SOURCE_TYPE,
                }
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=_REDACT_KEYS,
            max_items=1 + len(evidence_bindings),
            max_context_tokens=max(1024, tokens) + 32 * len(evidence_bindings),
        )
        recording_evaluator = RecordingGovernanceEvaluator(self._governance)
        recording_issuer = RecordingAuthorizationIssuer(self._issuer)
        updates: list[MemoryUpdateResult] = []

        async def apply_effect(effect: MemoryExtractionEffect) -> JsonValue:
            for candidate in effect.candidates:
                updates.append(await self._consolidator.consolidate(candidate))
            return {
                "candidate_ids": [str(item.candidate_id) for item in effect.candidates],
                "memory_ids": [str(item.memory.memory_id) for item in updates],
            }

        async def apply_normalized_effect(
            normalized: NormalizedDecisionEffect[MemoryExtractionEffect],
        ) -> JsonValue:
            batch = await self._consolidator.consolidate_batch(
                normalized.payload.candidates,
                effect_fingerprint=normalized.effect_fingerprint,
            )
            updates.clear()
            updates.extend(batch)
            return {
                "candidate_ids": [
                    str(item.candidate_id) for item in normalized.payload.candidates
                ],
                "memory_ids": [str(item.memory.memory_id) for item in batch],
            }

        async def reconcile_effect(
            normalized: NormalizedDecisionEffect[MemoryExtractionEffect],
        ) -> DecisionReconciliation:
            committed = await self._memory_store.load_applied_effect(
                normalized.effect_fingerprint
            )
            if committed is not None:
                result: JsonValue = {
                    "candidate_ids": [
                        str(item.candidate_id)
                        for item in normalized.payload.candidates
                    ],
                    "memory_ids": [str(item.memory_id) for item in committed],
                }
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.COMMITTED,
                    reason="Memory batch was committed atomically and read back",
                    apply_receipt=DecisionApplyReceipt(
                        effect_fingerprint=normalized.effect_fingerprint,
                        committed_state_fingerprint=governance_fingerprint(result),
                        result=result,
                    ),
                )
            partial = tuple(
                item
                for candidate in normalized.payload.candidates
                if (item := await self._memory_store.load_applied_candidate(
                    candidate.candidate_id
                )) is not None
            )
            if partial:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="candidate writes exist without an atomic Memory batch receipt",
                )
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.NOT_COMMITTED,
                reason="no candidate from the Memory batch was committed",
            )

        coordinator = DecisionLifecycleCoordinator[
            MemoryExtractionDecisionPayload,
            MemoryCandidateBatchDraft,
            MemoryExtractionEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=MemoryExtractionProposalProducer(
                capability=self._capability,
                timeout_seconds=self._execution_policy.timeout_seconds,
                correlation=InferenceCorrelation(run_id=run_id, task_id=task_id),
            ),
            basis_provider=_CurrentMemoryExtractionBasisProvider(
                store=self._memory_store,
                payload=payload,
            ),
            validator=RuntimeDecisionValidator(
                normalizer=MemoryExtractionEffectNormalizer()
            ),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=recording_evaluator,
                authorization_issuer=recording_issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
                apply_normalized_effect=apply_normalized_effect,
                reconcile_effect=reconcile_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                MemoryExtractionDecisionPayload,
                MemoryCandidateBatchDraft,
                MemoryExtractionEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = checkpoint.governance_receipt
            if receipt is None or receipt.review_request_id is None:
                raise RuntimeError("Memory Extraction review has no Review Request")
            recording_evaluator.review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="Candidates are evidence-bound and auditable.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            checkpoint = await coordinator.resume_review(request.request_id)
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
            or checkpoint.proposal is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Memory Extraction decision did not apply: {reason}")
        if (
            recording_evaluator.request is None
            or recording_evaluator.preliminary is None
            or recording_evaluator.final is None
        ):
            raise RuntimeError("Memory Extraction Governance record is incomplete")
        self._workspace.memory_candidate_drafts.extend(checkpoint.proposal.payload.candidates)
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="llm_memory",
                request=recording_evaluator.request,
                preliminary=recording_evaluator.preliminary,
                final=recording_evaluator.final,
                authorization=recording_issuer.authorization,
                review=recording_evaluator.review,
            )
        )
        for update in updates:
            self._workspace.trace_batches.append(
                context_trace.memory_update(
                    update,
                    correlation=EvaluationCorrelation(run_id=run_id, task_id=task_id),
                )
            )
        effect = checkpoint.validated_decision.normalized_effect.payload
        return MemoryExtractionDecisionOutcome(
            request_id=request.request_id,
            effect=effect,
            updates=tuple(updates),
        )
