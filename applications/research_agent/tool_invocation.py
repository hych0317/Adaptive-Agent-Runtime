"""Research composition for governed Reasoner Tool invocation Decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

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
    HumanReviewDecision,
    HumanReviewService,
    ReviewOutcome,
    RuntimeDecisionGovernanceAdapter,
    RuntimeCommitPermit,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    BoundToolInvocationProposalProducer,
    ToolInvocationEffectNormalizer,
    build_tool_invocation_agent_input,
)
from adaptive_agent_runtime.orchestration import TaskNode
from adaptive_agent_runtime.tool_ecosystem import (
    TOOL_INVOCATION_DECISION_TYPE,
    TOOL_INVOCATION_INPUT_SOURCE_TYPE,
    TOOL_INVOCATION_OPERATION,
    CapabilityCandidateResolver,
    CapabilityRequest,
    CapabilityRequirement,
    ToolExecutionPolicy,
    PermitBoundToolExecutor,
    ToolIntegrationError,
    ToolInvocation,
    ToolInvocationDecisionOutcome,
    ToolInvocationDecisionPayload,
    ToolInvocationEffect,
    ToolInvocationProposalDraft,
    ToolCorrelation,
    ToolObservation,
    ToolProviderMetadata,
    ToolSelector,
    ToolSelection,
    ToolSelectionContext,
    ToolSelectionDecisionHandler,
    eligible_tool_candidate_bindings,
    stable_tool_invocation_decision_id,
    tool_invocation_candidate_set_fingerprint,
    tool_invocation_fingerprint,
)

from applications.research_agent.capabilities import (
    PRIVILEGED_INFORMATION_PROVIDERS,
)
from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.llm_tools import (
    RESEARCH_INFORMATION_RETRIEVAL_TOOL,
    RESEARCH_RETRIEVAL_SCOPES,
)
from applications.research_agent.report import GovernanceRecord


if TYPE_CHECKING:
    from applications.research_agent.strategies import ResearchWorkspace


_REDACT_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


def _invocation_basis(
    payload: ToolInvocationDecisionPayload,
    *,
    candidate_fingerprint: str | None = None,
    provider_fingerprint: str | None = None,
) -> DecisionBasis:
    return DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {
                "invocation": payload.invocation,
                "requirement": payload.requirement,
                "candidate_set_fingerprint": (
                    candidate_fingerprint or payload.candidate_set_fingerprint
                ),
                "provider_metadata_fingerprint": (
                    provider_fingerprint or payload.provider_metadata_fingerprint
                ),
                "selection_fingerprint": payload.selection_fingerprint,
                "proposal_fingerprint": payload.proposal_fingerprint,
                "exact_argument_constraints": payload.exact_argument_constraints,
                "state_revision": payload.state_revision,
                "privileged": payload.privileged,
            }
        ),
        state_revision=payload.state_revision,
    )


class _CurrentToolInvocationBasisProvider:
    module_id = "research_agent.tool_invocation.basis"

    def __init__(
        self,
        *,
        resolver: CapabilityCandidateResolver,
        capability_request: CapabilityRequest,
        payload: ToolInvocationDecisionPayload,
    ) -> None:
        self._resolver = resolver
        self._capability_request = capability_request
        self._payload = payload

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        current = self._resolver.candidates(self._capability_request.requirement)
        bindings = eligible_tool_candidate_bindings(
            invocation_id=self._payload.invocation.invocation_id,
            request=self._capability_request,
            candidates=current,
        )
        current_candidates = tuple(item.metadata for item in bindings)
        provider = next(
            (
                item
                for item in current_candidates
                if item.provider_id == self._payload.invocation.provider_id
            ),
            None,
        )
        provider_fingerprint = (
            tool_invocation_fingerprint(provider)
            if provider is not None
            else tool_invocation_fingerprint(
                {"missing_provider_id": self._payload.invocation.provider_id}
            )
        )
        return _invocation_basis(
            self._payload,
            candidate_fingerprint=tool_invocation_candidate_set_fingerprint(
                current_candidates
            ),
            provider_fingerprint=provider_fingerprint,
        )


class ResearchToolInvocationDecisionHandler:
    """Turn one powerless Reasoner intent into a governed Runtime invocation."""

    module_id = "research_agent.tool_invocation_decision_handler"

    def __init__(
        self,
        *,
        resolver: CapabilityCandidateResolver,
        selector: ToolSelector | None = None,
        selection_handler: ToolSelectionDecisionHandler | None = None,
        executor: PermitBoundToolExecutor,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        workspace: ResearchWorkspace,
        execution_policy: ToolExecutionPolicy | None = None,
        checkpoint_store: DecisionCheckpointStore[
            ToolInvocationDecisionPayload,
            ToolInvocationProposalDraft,
            ToolInvocationEffect,
        ]
        | None = None,
    ) -> None:
        if (selector is None) == (selection_handler is None):
            raise ValueError(
                "exactly one of selector or selection_handler must be configured"
            )
        self._resolver = resolver
        self._selector = selector
        self._selection_handler = selection_handler
        self._executor = executor
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._workspace = workspace
        self._execution_policy = execution_policy or ToolExecutionPolicy()
        self._checkpoint_store = (
            checkpoint_store
            or InMemoryDecisionCheckpointStore[
                ToolInvocationDecisionPayload,
                ToolInvocationProposalDraft,
                ToolInvocationEffect,
            ]()
        )

    async def handle(
        self,
        *,
        proposal: ToolInvocationProposalDraft,
        producer_id: str,
        node: TaskNode,
        state: AgentState,
    ) -> ToolInvocationDecisionOutcome:
        arguments = proposal.model_dump(mode="json")["arguments"]
        if proposal.capability_id != RESEARCH_INFORMATION_RETRIEVAL_TOOL.capability_id:
            raise ToolIntegrationError(
                f"ToolIntent capability '{proposal.capability_id}' is not allowed"
            )
        company = arguments.get("company")
        scope = arguments.get("scope")
        if company != self._workspace.definition.company:
            raise ToolIntegrationError(
                "ToolIntent cannot change the company task boundary"
            )
        if not isinstance(scope, str) or scope not in RESEARCH_RETRIEVAL_SCOPES:
            raise ToolIntegrationError("ToolIntent retrieval scope is not allowed")

        requirement = CapabilityRequirement(
            capability_id=proposal.capability_id,
            required_provider_tags=(scope,),
        )
        selection_context = ToolSelectionContext(tags=("llm_tool_intent", scope))
        capability_request = CapabilityRequest(
            requirement=requirement,
            arguments=arguments,
            selection_context=selection_context,
        )
        invocation_id = uuid4()
        bindings = eligible_tool_candidate_bindings(
            invocation_id=invocation_id,
            request=capability_request,
            candidates=self._resolver.candidates(requirement),
        )
        candidates = tuple(item.metadata for item in bindings)
        if not candidates:
            raise ToolIntegrationError(
                "no available Provider accepts the ToolIntent arguments"
            )
        selection = await self._select_provider(
            request=capability_request,
            candidates=candidates,
            node=node,
            state=state,
            invocation_id=invocation_id,
        )
        provider = self._selected_provider(selection, requirement, candidates)
        invocation = ToolInvocation(
            invocation_id=invocation_id,
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=provider.provider_id,
            arguments=arguments,
            correlation=ToolCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                action_id=invocation_id,
            ),
        )
        payload = ToolInvocationDecisionPayload(
            task_description=state.task.description,
            node_goal=node.goal,
            invocation=invocation,
            requirement=requirement,
            provider_metadata=provider,
            provider_metadata_fingerprint=tool_invocation_fingerprint(provider),
            candidate_set_fingerprint=tool_invocation_candidate_set_fingerprint(
                candidates
            ),
            selection_fingerprint=tool_invocation_fingerprint(selection),
            proposal_fingerprint=tool_invocation_fingerprint(proposal),
            exact_argument_constraints={"company": company, "scope": scope},
            state_revision=state.revision,
            privileged=provider.provider_id in PRIVILEGED_INFORMATION_PROVIDERS,
        )
        basis = _invocation_basis(payload)
        request_id = stable_tool_invocation_decision_id(
            "decision-request",
            invocation_id,
            proposal.call_key,
            state.revision,
        )
        decision_request = DecisionRequest[ToolInvocationDecisionPayload](
            request_id=request_id,
            decision_type=TOOL_INVOCATION_DECISION_TYPE,
            target=DecisionTarget(
                target_type="tool_provider",
                target_id=provider.provider_id,
            ),
            correlation=DecisionCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                action_id=invocation_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(TOOL_INVOCATION_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="tool-invocation-capability",
                    description="Use only the Runtime-allowlisted capability.",
                    value=requirement.capability_id,
                ),
                DecisionConstraint(
                    constraint_id="tool-invocation-provider",
                    description=(
                        "Runtime owns Provider selection and invocation identity."
                    ),
                ),
                DecisionConstraint(
                    constraint_id="tool-invocation-arguments",
                    description=(
                        "Arguments must satisfy Provider schema and Runtime bounds."
                    ),
                ),
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
                max_revision_count=0,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        projected = build_tool_invocation_agent_input(payload)
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="tool-invocation-input",
                    source_type=TOOL_INVOCATION_INPUT_SOURCE_TYPE,
                    agent_scope="tool_invocation",
                    content=projected.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=384,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.tool_invocation.context",
            version="1",
            agent_scope="tool_invocation",
            allowed_decision_types=frozenset({TOOL_INVOCATION_DECISION_TYPE}),
            allowed_source_types=frozenset({TOOL_INVOCATION_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=_REDACT_KEYS,
            max_items=1,
            max_context_tokens=512,
        )
        recording_evaluator = RecordingGovernanceEvaluator(self._governance)
        recording_issuer = RecordingAuthorizationIssuer(self._issuer)

        async def apply_effect(effect: ToolInvocationEffect) -> JsonValue:
            raise RuntimeError("raw Tool execution requires a Runtime Permit")

        async def apply_authorized_effect(
            normalized: NormalizedDecisionEffect[ToolInvocationEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            effect = normalized.payload
            observation = await self._executor.execute(
                effect.invocation,
                self._execution_policy,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            return observation.model_dump(mode="json")

        coordinator = DecisionLifecycleCoordinator[
            ToolInvocationDecisionPayload,
            ToolInvocationProposalDraft,
            ToolInvocationEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=BoundToolInvocationProposalProducer(
                draft=proposal,
                producer_id=producer_id,
            ),
            basis_provider=_CurrentToolInvocationBasisProvider(
                resolver=self._resolver,
                capability_request=capability_request,
                payload=payload,
            ),
            validator=RuntimeDecisionValidator(
                normalizer=ToolInvocationEffectNormalizer()
            ),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=recording_evaluator,
                authorization_issuer=recording_issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
                apply_authorized_effect=apply_authorized_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                ToolInvocationDecisionPayload,
                ToolInvocationProposalDraft,
                ToolInvocationEffect,
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
                raise RuntimeError("Tool Invocation review has no Review Request")
            recording_evaluator.review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="The Tool invocation is bounded and auditable.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            checkpoint = await coordinator.resume_review(decision_request.request_id)
        result = checkpoint.result
        if (
            result is None
            or result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
            or result.apply_receipt is None
        ):
            reason = (
                result.reason
                if result is not None
                else checkpoint.stage
            )
            raise ToolIntegrationError(
                f"Tool Invocation decision did not apply: {reason}"
            )
        if (
            recording_evaluator.request is None
            or recording_evaluator.preliminary is None
            or recording_evaluator.final is None
        ):
            raise RuntimeError("Tool Invocation Governance record is incomplete")
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="tool_invocation_decision",
                request=recording_evaluator.request,
                preliminary=recording_evaluator.preliminary,
                final=recording_evaluator.final,
                authorization=recording_issuer.authorization,
                review=recording_evaluator.review,
            )
        )
        effect = checkpoint.validated_decision.normalized_effect.payload
        observation = ToolObservation.model_validate(result.apply_receipt.result)
        return ToolInvocationDecisionOutcome(
            request_id=decision_request.request_id,
            effect=effect,
            observation=observation,
        )

    async def _select_provider(
        self,
        *,
        request: CapabilityRequest,
        candidates: tuple[ToolProviderMetadata, ...],
        node: TaskNode,
        state: AgentState,
        invocation_id: UUID,
    ) -> ToolSelection:
        if self._selection_handler is None:
            assert self._selector is not None
            return self._selector.select(
                request.requirement,
                candidates,
                request.selection_context,
            )
        outcome = await self._selection_handler.handle(
            request=request,
            candidates=candidates,
            node=node,
            state=state,
            invocation_id=invocation_id,
        )
        if outcome.effect.invocation_id != invocation_id:
            raise ToolIntegrationError(
                "Tool Selection effect belongs to another invocation"
            )
        return outcome.selection

    @staticmethod
    def _selected_provider(
        selection: ToolSelection,
        requirement: CapabilityRequirement,
        candidates: tuple[ToolProviderMetadata, ...],
    ) -> ToolProviderMetadata:
        if selection.requirement_id != requirement.requirement_id:
            raise ToolIntegrationError("Tool selector returned a mismatched requirement")
        if selection.capability_id != requirement.capability_id:
            raise ToolIntegrationError("Tool selector changed the capability")
        provider = next(
            (
                candidate
                for candidate in candidates
                if candidate.provider_id == selection.provider_id
            ),
            None,
        )
        if provider is None:
            raise ToolIntegrationError(
                "Tool selector escaped Runtime-filtered candidates"
            )
        return provider
