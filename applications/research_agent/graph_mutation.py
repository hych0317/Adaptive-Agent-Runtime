"""Research composition for governed Agent-proposed Task Graph mutation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import AgentState, Observation, TraceSink
from adaptive_agent_runtime.context_memory import (
    ContextAssembly,
    ContextLayer,
    ContextRequirement,
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
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewDecision,
    HumanReviewService,
    ReviewOutcome,
    RuntimeDecisionGovernanceAdapter,
)
from adaptive_agent_runtime.llm import (
    GRAPH_MUTATION_EVIDENCE_SOURCE_TYPE,
    GRAPH_MUTATION_INPUT_SOURCE_TYPE,
    EvidenceReference,
    GraphMutationEffectNormalizer,
    GraphMutationProposalCapability,
    GraphMutationProposalDraft,
    GraphMutationProposalProducer,
    GraphMutationProposalRequest,
    InferenceCorrelation,
)
from adaptive_agent_runtime.orchestration import (
    GRAPH_MUTATION_APPLY_OPERATION,
    GRAPH_MUTATION_DECISION_TYPE,
    DynamicTaskGraph,
    GraphMutationDecisionOutcome,
    GraphMutationDecisionPayload,
    GraphMutationEffect,
    GraphMutationExecutionPolicy,
    apply_graph_mutation_effect,
)

from applications.research_agent.capabilities import INFORMATION_RETRIEVAL
from applications.research_agent.cognition import ResearchContextProjection
from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord
from applications.research_agent.tasks import (
    COMPANY_RESEARCH,
    NEWS_ANALYSIS,
    RESEARCH_NODE_ROLES,
    RESEARCH_STRATEGY_ID,
    REVIEW_STRATEGY_ID,
    build_research_mutations_from_draft,
)


if TYPE_CHECKING:
    from applications.research_agent.strategies import ResearchWorkspace


_REDACT_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


class _FixedGraphMutationBasisProvider:
    module_id = "research_agent.graph_mutation.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class ResearchGraphMutationDecisionHandler:
    module_id = "research_agent.graph_mutation_decision_handler"

    def __init__(
        self,
        *,
        capability: GraphMutationProposalCapability,
        context_projection: ResearchContextProjection,
        execution_policy: GraphMutationExecutionPolicy,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        workspace: ResearchWorkspace,
        checkpoint_store: DecisionCheckpointStore[
            GraphMutationDecisionPayload,
            GraphMutationProposalDraft,
            GraphMutationEffect,
        ]
        | None = None,
    ) -> None:
        self._capability = capability
        self._context_projection = context_projection
        self._execution_policy = execution_policy
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._workspace = workspace
        self._checkpoint_store = checkpoint_store or InMemoryDecisionCheckpointStore()

    async def handle(
        self,
        *,
        graph: DynamicTaskGraph,
        state: AgentState,
        source_node_id: UUID,
        observation: Observation,
    ) -> GraphMutationDecisionOutcome | None:
        source_node = graph.get_node(source_node_id)
        if self._workspace.role_for(source_node) != COMPANY_RESEARCH:
            return None
        units = tuple(
            unit
            for unit in self._workspace.context_units
            if unit.metadata.node_id == source_node_id
        )
        if not units:
            raise RuntimeError("Graph Mutation Agent has no accepted Context evidence")
        working = tuple(item for item in units if item.metadata.layer is ContextLayer.WORKING)
        task_units = tuple(item for item in units if item.metadata.layer is ContextLayer.TASK)
        semantic = tuple(item for item in units if item.metadata.layer is ContextLayer.SEMANTIC)
        ordered = (*working, *task_units, *semantic)
        used_tokens = sum(item.metadata.estimated_tokens for item in ordered)
        assembly = ContextAssembly(
            requirement=ContextRequirement(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=source_node_id,
                goal="Propose bounded graph mutations from company evidence.",
                max_units=len(ordered),
                max_tokens=used_tokens,
            ),
            units=ordered,
            working_context=working,
            task_context=task_units,
            semantic_context=semantic,
            used_tokens=used_tokens,
        )
        package = self._context_projection.project(assembly)
        if not package.blocks:
            raise RuntimeError("Graph Mutation Context policy omitted all evidence")
        self._workspace.llm_context_packages.append(package)
        evidence = tuple(
            EvidenceReference(
                reference_id=f"context:{block.context_id}",
                kind="tool.observation",
                summary="Policy-approved company research observation.",
                reliability=1.0,
            )
            for block in package.blocks
        )
        proposal_request = GraphMutationProposalRequest(
            task=state.task.description,
            trigger_node_key=COMPANY_RESEARCH,
            trigger_observation={
                "context_package": package.model_dump(mode="json"),
                "accepted_output_present": observation.output is not None,
            },
            existing_node_keys=tuple(
                role for role in RESEARCH_NODE_ROLES if role != NEWS_ANALYSIS
            ),
            allowed_new_node_keys=(NEWS_ANALYSIS,),
            available_strategies=(RESEARCH_STRATEGY_ID, REVIEW_STRATEGY_ID),
            available_execution_capability_ids=(INFORMATION_RETRIEVAL,),
            evidence=evidence,
        )
        observation_fingerprint = decision_fingerprint(observation)
        payload = GraphMutationDecisionPayload(
            graph=graph,
            source_node_id=source_node_id,
            source_action_id=observation.action_id,
            source_observation_fingerprint=observation_fingerprint,
            agent_input=proposal_request.model_dump(mode="json"),
            execution_policy=self._execution_policy,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {
                    "graph": graph,
                    "observation_fingerprint": observation_fingerprint,
                    "execution_policy": self._execution_policy,
                }
            ),
            state_revision=state.revision,
            graph_version=graph.version,
        )
        request = DecisionRequest[GraphMutationDecisionPayload](
            decision_type=GRAPH_MUTATION_DECISION_TYPE,
            target=DecisionTarget(target_type="task_graph", target_id=str(graph.graph_id)),
            correlation=DecisionCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=source_node_id,
                action_id=observation.action_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(GRAPH_MUTATION_APPLY_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="graph-mutation-bounds",
                    description="Use only Runtime-allowed nodes, dependencies, and strategies.",
                ),
                DecisionConstraint(
                    constraint_id="graph-mutation-no-apply",
                    description="Propose symbolic mutations; Runtime owns IDs and Apply.",
                ),
            ),
            evidence=tuple(
                DecisionEvidenceReference(
                    evidence_id=item.reference_id,
                    kind=item.kind,
                    source="context_runtime",
                    reliability=item.reliability or 1.0,
                    summary=item.summary or "Accepted Runtime evidence.",
                )
                for item in evidence
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
                    source_id="graph-mutation-input",
                    source_type=GRAPH_MUTATION_INPUT_SOURCE_TYPE,
                    agent_scope="graph_mutation",
                    content=proposal_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=max(256, used_tokens),
                ),
                *(
                    ProjectionSource(
                        source_id=f"graph-mutation-evidence:{item.reference_id}",
                        source_type=GRAPH_MUTATION_EVIDENCE_SOURCE_TYPE,
                        agent_scope="graph_mutation",
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
                    for item in evidence
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.graph_mutation.context",
            version="1",
            agent_scope="graph_mutation",
            allowed_decision_types=frozenset({GRAPH_MUTATION_DECISION_TYPE}),
            allowed_source_types=frozenset(
                {
                    GRAPH_MUTATION_INPUT_SOURCE_TYPE,
                    GRAPH_MUTATION_EVIDENCE_SOURCE_TYPE,
                }
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=_REDACT_KEYS,
            max_items=1 + len(evidence),
            max_context_tokens=max(512, used_tokens) + 32 * len(evidence),
        )
        recording_evaluator = RecordingGovernanceEvaluator(self._governance)
        recording_issuer = RecordingAuthorizationIssuer(self._issuer)
        applied_graph: DynamicTaskGraph | None = None

        async def apply_effect(effect: GraphMutationEffect) -> JsonValue:
            nonlocal applied_graph
            applied_graph = apply_graph_mutation_effect(graph, effect)
            return {
                "graph_id": str(applied_graph.graph_id),
                "graph_version": applied_graph.version,
                "mutation_ids": [str(item.mutation_id) for item in effect.mutations],
            }

        coordinator = DecisionLifecycleCoordinator[
            GraphMutationDecisionPayload,
            GraphMutationProposalDraft,
            GraphMutationEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=GraphMutationProposalProducer(
                capability=self._capability,
                timeout_seconds=self._execution_policy.timeout_seconds,
                correlation=InferenceCorrelation(
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                    node_id=source_node_id,
                    action_id=observation.action_id,
                ),
            ),
            basis_provider=_FixedGraphMutationBasisProvider(basis),
            validator=RuntimeDecisionValidator(
                normalizer=GraphMutationEffectNormalizer(
                    mutation_projector=lambda draft: build_research_mutations_from_draft(
                        self._workspace.definition,
                        draft,
                    )
                )
            ),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=recording_evaluator,
                authorization_issuer=recording_issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                GraphMutationDecisionPayload,
                GraphMutationProposalDraft,
                GraphMutationEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = checkpoint.governance_receipt
            if receipt is None or receipt.review_request_id is None:
                raise RuntimeError("Graph Mutation review has no Review Request")
            recording_evaluator.review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="The mutation batch is bounded and replayable.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            checkpoint = await coordinator.resume_review(request.request_id)
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
            or applied_graph is None
            or checkpoint.proposal is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Graph Mutation decision did not apply: {reason}")
        if (
            recording_evaluator.request is None
            or recording_evaluator.preliminary is None
            or recording_evaluator.final is None
        ):
            raise RuntimeError("Graph Mutation Governance record is incomplete")
        self._workspace.graph_mutation_proposals.append(checkpoint.proposal.payload)
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="graph_mutation",
                request=recording_evaluator.request,
                preliminary=recording_evaluator.preliminary,
                final=recording_evaluator.final,
                authorization=recording_issuer.authorization,
                review=recording_evaluator.review,
            )
        )
        effect = checkpoint.validated_decision.normalized_effect.payload
        return GraphMutationDecisionOutcome(
            request_id=request.request_id,
            effect=effect,
            graph=applied_graph,
        )
