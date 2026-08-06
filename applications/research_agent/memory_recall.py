"""Governed Initial Planning Memory Recall composition."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import JsonValue

from adaptive_agent_runtime import TraceSink
from adaptive_agent_runtime.context_memory import (
    MEMORY_RECALL_COMMIT_OPERATION,
    MEMORY_RECALL_DECISION_TYPE,
    MemoryRecallAgentRequest,
    MemoryRecallBundle,
    MemoryRecallBundleStore,
    MemoryRecallCandidate,
    MemoryRecallDraft,
    MemoryRecallEffect,
    MemoryRecallEligibilityPolicy,
    MemoryRecallRequest,
    MemoryRecallSourceSnapshot,
    MemoryScope,
    MemoryStore,
    ExperienceMetadataStore,
    RuntimeMemoryCandidateResolver,
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
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernanceTarget,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewService,
    RuntimeDecisionGovernanceAdapter,
    RuntimeCommitPermit,
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


MEMORY_RECALL_INPUT_SOURCE_TYPE = "memory_recall_input"


class MemoryRecallProposalCapability(Protocol):
    module_id: str
    capability_id: str

    async def propose_memory_recall(
        self,
        request: MemoryRecallAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[MemoryRecallDraft]: ...


class DeterministicMemoryRecallCapability:
    """Bounded fallback; it has the same proposal-only authority as an Agent."""

    module_id = "research_agent.memory_recall.deterministic"
    capability_id = "memory_recall.deterministic"

    async def propose_memory_recall(
        self,
        request: MemoryRecallAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[MemoryRecallDraft]:
        del invocation
        selected: list[str] = []
        tokens = 0
        for candidate in request.candidates:
            if len(selected) >= request.max_selected_items:
                break
            if selected and tokens + candidate.estimated_tokens > request.max_selected_tokens:
                continue
            if candidate.estimated_tokens > request.max_selected_tokens:
                continue
            selected.append(candidate.candidate_ref)
            tokens += candidate.estimated_tokens
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=MemoryRecallDraft(
                selected_candidate_refs=tuple(selected),
                selection_reason=(
                    "Select the highest-ranked Runtime-eligible planning memories."
                ),
            ),
        )


@dataclass(frozen=True)
class ResearchMemoryRecallResult:
    bundle: MemoryRecallBundle
    governance_record: GovernanceRecord | None
    request_id: UUID
    proposal: MemoryRecallDraft


class _RecallProposalProducer:
    module_id = "research_agent.memory_recall.proposal_producer"

    def __init__(
        self,
        capability: MemoryRecallProposalCapability,
        *,
        timeout_seconds: float,
        correlation: InferenceCorrelation,
    ) -> None:
        self._capability = capability
        self._timeout_seconds = timeout_seconds
        self._correlation = correlation

    async def propose(self, context: AgentContext) -> AgentCallResult[MemoryRecallDraft]:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == MEMORY_RECALL_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1 or not isinstance(blocks[0].content, Mapping):
            raise ValueError("Recall Agent requires one isolated input block")
        request = MemoryRecallAgentRequest.model_validate(dict(blocks[0].content))
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.propose_memory_recall(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "memory.recall.propose"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Memory Recall Agent cannot return ToolIntents")
        if turn.result is None:
            raise RuntimeError("Memory Recall Agent returned no Draft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="memory_recall_selection",
                    implementation_version="phase-3a",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=MEMORY_RECALL_COMMIT_OPERATION,
                payload=draft,
                rationale=draft.selection_reason,
                evidence_refs=draft.selected_candidate_refs,
                confidence=0.8,
            ),
            elapsed_seconds=monotonic() - started,
        )


class _RecallEffectNormalizer:
    module_id = "research_agent.memory_recall.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[MemoryRecallRequest],
        proposal: DecisionProposal[MemoryRecallDraft],
    ) -> NormalizedDecisionEffect[MemoryRecallEffect]:
        selected_refs = proposal.payload.selected_candidate_refs
        bindings = {
            item.candidate.candidate_ref: item for item in request.payload.candidates
        }
        unknown = set(selected_refs) - set(bindings)
        if unknown:
            raise ValueError("Recall Agent selected refs outside the Runtime candidate set")
        if not selected_refs:
            raise ValueError("Recall Agent selected no candidate")
        if len(selected_refs) > request.payload.max_selected_items:
            raise ValueError("Recall selection exceeds the Runtime item budget")
        selected = tuple(bindings[item] for item in selected_refs)
        used_tokens = sum(item.candidate.estimated_tokens for item in selected)
        if used_tokens > request.payload.max_selected_tokens:
            raise ValueError("Recall selection exceeds the Runtime token budget")
        sources = tuple(
            MemoryRecallSourceSnapshot(
                candidate_ref=item.candidate.candidate_ref,
                memory_id=item.memory_id,
                memory_revision=item.memory_revision,
                source_fingerprint=item.source_fingerprint,
                sanitized_content=item.candidate.sanitized_content,
                memory_category=item.candidate.memory_category,
                confidence_band=item.candidate.confidence_band,
                provenance_summary=item.candidate.provenance_summary,
                estimated_tokens=item.candidate.estimated_tokens,
                scope=item.scope,
                sensitivity=item.sensitivity,
            )
            for item in selected
        )
        if request.correlation.task_id is None:
            raise ValueError("Initial Memory Recall requires a task correlation")
        effect = MemoryRecallEffect(
            recall_decision_id=request.request_id,
            run_id=request.correlation.run_id,
            task_id=request.correlation.task_id,
            scope=request.payload.scope,
            sources=sources,
            max_items=request.payload.max_selected_items,
            max_tokens=request.payload.max_selected_tokens,
            candidate_set_fingerprint=request.payload.candidate_set_fingerprint,
            proposal_fingerprint=decision_fingerprint(proposal.payload),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[MemoryRecallEffect].create(
            payload=effect,
            operation=MEMORY_RECALL_COMMIT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.2,
            reversible=True,
            impact_description=(
                "Commit an immutable, scope-filtered Memory Recall Bundle for planning."
            ),
        )


class _RecallBasisProvider:
    module_id = "research_agent.memory_recall.basis"

    def __init__(self, store: MemoryStore, basis: DecisionBasis) -> None:
        self._store = store
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        payload = MemoryRecallRequest.model_validate(request.payload)
        current: list[object] = []
        valid = True
        for binding in payload.candidates:
            memory = await self._store.load(binding.memory_id)
            fingerprint = decision_fingerprint(memory) if memory is not None else None
            current.append((str(binding.memory_id), fingerprint))
            if (
                memory is None
                or memory.revision != binding.memory_revision
                or fingerprint != binding.source_fingerprint
            ):
                valid = False
        if valid:
            return self._basis
        return self._basis.model_copy(
            update={"snapshot_fingerprint": decision_fingerprint(current)}
        )


class ResearchInitialMemoryRecallHandler:
    module_id = "research_agent.initial_memory_recall"

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        bundle_store: MemoryRecallBundleStore,
        experience_store: ExperienceMetadataStore | None = None,
        capability: MemoryRecallProposalCapability,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        checkpoint_store: DecisionCheckpointStore[
            MemoryRecallRequest, MemoryRecallDraft, MemoryRecallEffect
        ]
        | None = None,
        timeout_seconds: float = 5.0,
        fault_injector: Callable[[DecisionFaultPoint, Any], None] | None = None,
    ) -> None:
        self._memory_store = memory_store
        self._bundle_store = bundle_store
        self._experience_store = experience_store
        self._capability = capability
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._checkpoints = checkpoint_store or InMemoryDecisionCheckpointStore()
        self._timeout_seconds = timeout_seconds
        self._fault_injector = fault_injector

    async def recall(
        self,
        *,
        goal: str,
        run_id: UUID,
        task_id: UUID,
        scope: MemoryScope,
        facts: Mapping[str, JsonValue],
        tags: tuple[str, ...],
        max_items: int = 3,
        max_tokens: int = 512,
    ) -> ResearchMemoryRecallResult | None:
        request_id = uuid4()
        policy = MemoryRecallEligibilityPolicy(
            scope=scope,
            max_candidates=8,
            max_candidate_tokens=1024,
        )
        candidates = await RuntimeMemoryCandidateResolver(
            self._memory_store,
            self._experience_store,
        ).resolve(
            request_id=request_id,
            goal=goal,
            facts=facts,
            tags=tags,
            policy=policy,
        )
        if not candidates:
            return None
        payload = MemoryRecallRequest(
            goal=goal,
            scope=scope,
            facts=dict(facts),
            tags=tags,
            candidates=candidates,
            candidate_set_fingerprint=decision_fingerprint(candidates),
            max_selected_items=max_items,
            max_selected_tokens=max_tokens,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {"goal": goal, "scope": scope, "candidates": candidates}
            ),
            configuration_revision=1,
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=MEMORY_RECALL_DECISION_TYPE,
            target=DecisionTarget(
                target_type="memory_recall_bundle",
                target_id=f"{run_id}:{task_id}",
            ),
            correlation=DecisionCorrelation(run_id=run_id, task_id=task_id),
            basis=basis,
            payload=payload,
            allowed_actions=(MEMORY_RECALL_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="recall-candidate-set",
                    description="Select only opaque Runtime-eligible candidate refs.",
                ),
                DecisionConstraint(
                    constraint_id="recall-advisory",
                    description="Historical Memory cannot override goal or Runtime policy.",
                ),
            ),
            evidence=tuple(
                DecisionEvidenceReference(
                    evidence_id=item.candidate.candidate_ref,
                    kind="memory.candidate",
                    source="memory_runtime",
                    reliability=(
                        0.9 if item.candidate.confidence_band.value == "high" else 0.7
                    ),
                    summary=(
                        "Runtime-eligible historical planning memory; not a current fact."
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
        agent_request = MemoryRecallAgentRequest(
            goal=goal,
            candidates=tuple(item.candidate for item in candidates),
            max_selected_items=max_items,
            max_selected_tokens=max_tokens,
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="memory-recall-input",
                    source_type=MEMORY_RECALL_INPUT_SOURCE_TYPE,
                    agent_scope="memory_recall",
                    content=agent_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=min(1024, 64 + sum(
                        item.candidate.estimated_tokens for item in candidates
                    )),
                ),
                *(
                    ProjectionSource(
                        source_id=f"memory-recall-evidence:{item.candidate.candidate_ref}",
                        source_type="memory_recall_evidence",
                        agent_scope="memory_recall",
                        content={
                            "candidate_ref": item.candidate.candidate_ref,
                            "memory_category": item.candidate.memory_category,
                            "confidence_band": item.candidate.confidence_band.value,
                        },
                        sensitivity=ContextSensitivity.INTERNAL,
                        evidence_id=item.candidate.candidate_ref,
                        priority=90,
                        estimated_tokens=16,
                    )
                    for item in candidates
                ),
            )
        )
        projection = ContextProjectionPolicy(
            policy_id="research.initial_memory_recall.context",
            version="1",
            agent_scope="memory_recall",
            allowed_decision_types=frozenset({MEMORY_RECALL_DECISION_TYPE}),
            allowed_source_types=frozenset(
                {MEMORY_RECALL_INPUT_SOURCE_TYPE, "memory_recall_evidence"}
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {"api_key", "authorization", "credential", "password", "secret", "token"}
            ),
            max_items=1 + len(candidates),
            max_context_tokens=1200 + 16 * len(candidates),
        )
        return await self._run(request, sources=sources, projection=projection)

    async def resume(self, request_id: UUID) -> ResearchMemoryRecallResult | None:
        checkpoint = await self._checkpoints.load(request_id)
        if checkpoint is None:
            raise KeyError(f"Recall Decision '{request_id}' was not found")
        coordinator, recorder, issuer = self._coordinator(
            checkpoint.request,
            checkpoint.request.basis,
        )
        resumed = await coordinator.resume(request_id)
        return await self._result(resumed, recorder, issuer)

    async def _run(
        self,
        request: DecisionRequest[MemoryRecallRequest],
        *,
        sources: ProjectionSources,
        projection: ContextProjectionPolicy,
    ) -> ResearchMemoryRecallResult | None:
        coordinator, recorder, issuer = self._coordinator(request, request.basis)
        checkpoint = await coordinator.run(request, sources=sources, policy=projection)
        return await self._result(checkpoint, recorder, issuer)

    def _coordinator(
        self,
        request: DecisionRequest[MemoryRecallRequest],
        basis: DecisionBasis,
    ) -> tuple[
        DecisionLifecycleCoordinator[
            MemoryRecallRequest,
            MemoryRecallDraft,
            MemoryRecallEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        recorder = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)

        async def forbidden_apply(effect: MemoryRecallEffect) -> JsonValue:
            del effect
            raise RuntimeError("Recall Bundle commit requires a Runtime Permit")

        async def apply_authorized(
            normalized: NormalizedDecisionEffect[MemoryRecallEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            bundle = await self._bundle_store.commit(
                normalized.payload,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            return self._bundle_result(bundle)

        async def reconcile(
            normalized: NormalizedDecisionEffect[MemoryRecallEffect],
        ) -> DecisionReconciliation:
            try:
                bundle = await self._bundle_store.load_by_effect(
                    normalized.effect_fingerprint
                )
            except Exception as exc:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=(
                        "Recall Bundle read-back is indeterminate: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if bundle is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Recall Bundle has not been committed",
                )
            effect = normalized.payload
            expected_refs = tuple(item.candidate_ref for item in effect.sources)
            observed_refs = tuple(item.source_memory_ref for item in bundle.items)
            if (
                bundle.run_id != effect.run_id
                or bundle.task_id != effect.task_id
                or bundle.recall_decision_id != effect.recall_decision_id
                or bundle.candidate_set_fingerprint != effect.candidate_set_fingerprint
                or bundle.effect_fingerprint != normalized.effect_fingerprint
                or observed_refs != expected_refs
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Recall Bundle conflicts with the authorized Effect",
                )
            result = self._bundle_result(bundle)
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Recall Bundle was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=decision_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            MemoryRecallRequest,
            MemoryRecallDraft,
            MemoryRecallEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=_RecallProposalProducer(
                self._capability,
                timeout_seconds=self._timeout_seconds,
                correlation=InferenceCorrelation(
                    run_id=request.correlation.run_id,
                    task_id=request.correlation.task_id,
                ),
            ),
            basis_provider=_RecallBasisProvider(self._memory_store, basis),
            validator=RuntimeDecisionValidator(normalizer=_RecallEffectNormalizer()),
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
                MemoryRecallRequest, MemoryRecallDraft, MemoryRecallEffect
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
            fault_injector=self._fault_injector,
        )
        return coordinator, recorder, issuer

    async def _result(
        self,
        checkpoint: DecisionCheckpoint[
            MemoryRecallRequest, MemoryRecallDraft, MemoryRecallEffect
        ],
        recorder: RecordingGovernanceEvaluator,
        issuer: RecordingAuthorizationIssuer,
    ) -> ResearchMemoryRecallResult | None:
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
        bundle = await self._bundle_store.load_by_effect(normalized.effect_fingerprint)
        if bundle is None:
            raise RuntimeError("APPLIED Recall Decision has no committed Bundle")
        governance_record = (
            GovernanceRecord(
                scenario="memory_recall",
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
        return ResearchMemoryRecallResult(
            bundle=bundle,
            governance_record=governance_record,
            request_id=checkpoint.request.request_id,
            proposal=checkpoint.proposal.payload,
        )

    @staticmethod
    def _bundle_result(bundle: MemoryRecallBundle) -> JsonValue:
        return {
            "bundle_id": str(bundle.bundle_id),
            "bundle_fingerprint": decision_fingerprint(bundle),
            "effect_fingerprint": bundle.effect_fingerprint,
            "source_count": len(bundle.items),
        }
