"""Research Application composition for governed adaptive Tool selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import AgentState, TraceSink
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
    DecisionLifecycleCoordinator,
    DecisionRequest,
    DecisionResultStatus,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorization,
    GovernanceAuthorizationIssuer,
    GovernanceDecision,
    GovernanceEvaluator,
    GovernanceRequest,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewDecision,
    HumanReviewService,
    ReviewOutcome,
    ReviewRequest,
    RuntimeDecisionGovernanceAdapter,
)
from adaptive_agent_runtime.llm import (
    TOOL_SELECTION_INPUT_SOURCE_TYPE,
    InferenceCorrelation,
    ToolSelectionDecisionProposalProducer,
    ToolSelectionDraft,
    ToolSelectionEffectNormalizer,
    ToolSelectionProposalCapability,
    build_tool_selection_proposal_request,
)
from adaptive_agent_runtime.orchestration import TaskNode
from adaptive_agent_runtime.tool_ecosystem import (
    TOOL_SELECTION_BIND_OPERATION,
    TOOL_SELECTION_DECISION_TYPE,
    CapabilityCandidateResolver,
    CapabilityRequest,
    ToolProviderMetadata,
    ToolSelection,
    ToolSelectionDecisionHandler,
    ToolSelectionDecisionOutcome,
    ToolSelectionDecisionPayload,
    ToolSelectionEffect,
    ToolSelectionError,
    ToolSelectionExecutionPolicy,
    candidate_set_fingerprint,
    eligible_tool_candidate_bindings,
    stable_tool_selection_id,
    tool_selection_fingerprint,
)

from applications.research_agent.report import GovernanceRecord
from applications.research_agent.strategies import ResearchWorkspace


_REDACT_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


def _selection_basis(payload: ToolSelectionDecisionPayload) -> DecisionBasis:
    return DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {
                "invocation_id": payload.invocation_id,
                "requirement": payload.requirement,
                "arguments_fingerprint": payload.arguments_fingerprint,
                "selection_context": payload.selection_context,
                "candidate_set_fingerprint": payload.candidate_set_fingerprint,
                "state_revision": payload.state_revision,
            }
        ),
        state_revision=payload.state_revision,
    )


class _CurrentToolSelectionBasisProvider:
    module_id = "research_agent.tool_selection.basis"

    def __init__(
        self,
        *,
        resolver: CapabilityCandidateResolver,
        capability_request: CapabilityRequest,
        payload: ToolSelectionDecisionPayload,
    ) -> None:
        self._resolver = resolver
        self._capability_request = capability_request
        self._payload = payload

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        current = self._resolver.candidates(self._capability_request.requirement)
        bindings = eligible_tool_candidate_bindings(
            invocation_id=self._payload.invocation_id,
            request=self._capability_request,
            candidates=current,
        )
        return DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {
                    "invocation_id": self._payload.invocation_id,
                    "requirement": self._payload.requirement,
                    "arguments_fingerprint": self._payload.arguments_fingerprint,
                    "selection_context": self._payload.selection_context,
                    "candidate_set_fingerprint": candidate_set_fingerprint(bindings),
                    "state_revision": self._payload.state_revision,
                }
            ),
            state_revision=self._payload.state_revision,
        )


class _RecordingGovernanceEvaluator:
    module_id = "research_agent.tool_selection.governance_recorder"

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
    module_id = "research_agent.tool_selection.authorization_recorder"

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


class ResearchToolSelectionDecisionHandler(ToolSelectionDecisionHandler):
    """Compose selection Decisions without exposing or invoking Providers."""

    module_id = "research_agent.tool_selection_decision_handler"

    def __init__(
        self,
        *,
        capability: ToolSelectionProposalCapability,
        resolver: CapabilityCandidateResolver,
        execution_policy: ToolSelectionExecutionPolicy,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        workspace: ResearchWorkspace,
        checkpoint_store: DecisionCheckpointStore[
            ToolSelectionDecisionPayload,
            ToolSelectionDraft,
            ToolSelectionEffect,
        ]
        | None = None,
    ) -> None:
        self._capability = capability
        self._resolver = resolver
        self._execution_policy = execution_policy
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._workspace = workspace
        self._checkpoint_store = (
            checkpoint_store
            or InMemoryDecisionCheckpointStore[
                ToolSelectionDecisionPayload,
                ToolSelectionDraft,
                ToolSelectionEffect,
            ]()
        )

    async def handle(
        self,
        *,
        request: CapabilityRequest,
        candidates: tuple[ToolProviderMetadata, ...],
        node: TaskNode,
        state: AgentState,
        invocation_id: UUID,
    ) -> ToolSelectionDecisionOutcome:
        bindings = eligible_tool_candidate_bindings(
            invocation_id=invocation_id,
            request=request,
            candidates=candidates,
        )
        if not bindings:
            raise ToolSelectionError(
                "no available Provider satisfies capability, tags, and input schema"
            )
        payload = ToolSelectionDecisionPayload(
            invocation_id=invocation_id,
            task_description=state.task.description,
            node_goal=node.goal,
            requirement=request.requirement,
            arguments_fingerprint=tool_selection_fingerprint(
                request.model_dump(mode="json")["arguments"]
            ),
            selection_context=request.selection_context,
            candidates=bindings,
            candidate_set_fingerprint=candidate_set_fingerprint(bindings),
            state_revision=state.revision,
            execution_policy=self._execution_policy,
        )
        basis = _selection_basis(payload)
        request_id = stable_tool_selection_id(
            "decision-request",
            invocation_id,
            request.requirement.requirement_id,
            state.revision,
        )
        decision_request = DecisionRequest[ToolSelectionDecisionPayload](
            request_id=request_id,
            decision_type=TOOL_SELECTION_DECISION_TYPE,
            target=DecisionTarget(
                target_type="tool_invocation",
                target_id=str(invocation_id),
            ),
            correlation=DecisionCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(TOOL_SELECTION_BIND_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="tool-selection-candidate-set",
                    description="Select exactly one Runtime-projected candidate.",
                ),
                DecisionConstraint(
                    constraint_id="tool-selection-no-execution",
                    description="Propose a Provider preference; do not invoke it.",
                ),
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=self._execution_policy.max_agent_calls,
                max_revision_count=self._execution_policy.max_revision_count,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        projected = build_tool_selection_proposal_request(payload, ())
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="tool-selection-input",
                    source_type=TOOL_SELECTION_INPUT_SOURCE_TYPE,
                    agent_scope="tool_selection",
                    content=projected.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=256 + (64 * len(bindings)),
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.tool_selection.context",
            version="1",
            agent_scope="tool_selection",
            allowed_decision_types=frozenset({TOOL_SELECTION_DECISION_TYPE}),
            allowed_source_types=frozenset({TOOL_SELECTION_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=_REDACT_KEYS,
            max_items=1,
            max_context_tokens=512 + (64 * len(bindings)),
        )
        recording_evaluator = _RecordingGovernanceEvaluator(self._governance)
        recording_issuer = _RecordingAuthorizationIssuer(self._issuer)
        governance_adapter = RuntimeDecisionGovernanceAdapter[
            ToolSelectionDecisionPayload,
            ToolSelectionDraft,
            ToolSelectionEffect,
        ](
            evaluator=recording_evaluator,
            authorization_issuer=recording_issuer,
            review_service=self._reviews,
        )

        async def apply_effect(effect: ToolSelectionEffect) -> JsonValue:
            return {
                "invocation_id": str(effect.invocation_id),
                "requirement_id": str(effect.requirement_id),
                "provider_id": effect.provider_id,
            }

        coordinator = DecisionLifecycleCoordinator[
            ToolSelectionDecisionPayload,
            ToolSelectionDraft,
            ToolSelectionEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=ToolSelectionDecisionProposalProducer(
                capability=self._capability,
                execution_policy=self._execution_policy,
                correlation=InferenceCorrelation(
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                    node_id=node.node_id,
                ),
            ),
            basis_provider=_CurrentToolSelectionBasisProvider(
                resolver=self._resolver,
                capability_request=request,
                payload=payload,
            ),
            validator=RuntimeDecisionValidator(
                normalizer=ToolSelectionEffectNormalizer()
            ),
            governance=governance_adapter,
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                ToolSelectionDecisionPayload,
                ToolSelectionDraft,
                ToolSelectionEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(
            decision_request,
            sources=sources,
            policy=policy,
        )
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = checkpoint.governance_receipt
            if receipt is None or receipt.review_request_id is None:
                raise RuntimeError("Tool Selection review has no Review Request")
            review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="The Provider binding is reversible and does not execute.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            recording_evaluator.review = review
            checkpoint = await coordinator.resume_review(decision_request.request_id)
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
            raise ToolSelectionError(f"Tool Selection decision did not apply: {reason}")
        governance_request = recording_evaluator.request
        preliminary = recording_evaluator.preliminary
        final = recording_evaluator.final
        if governance_request is None or preliminary is None or final is None:
            raise RuntimeError("Tool Selection Governance record is incomplete")
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="tool_selection_agent",
                request=governance_request,
                preliminary=preliminary,
                final=final,
                authorization=recording_issuer.authorization,
                review=recording_evaluator.review,
            )
        )
        effect = checkpoint.validated_decision.normalized_effect.payload
        return ToolSelectionDecisionOutcome(
            request_id=decision_request.request_id,
            effect=effect,
            selection=ToolSelection(
                requirement_id=effect.requirement_id,
                capability_id=effect.capability_id,
                provider_id=effect.provider_id,
                reason=effect.selection_reason,
            ),
        )
