"""Research composition for governed adaptive ready-node selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
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
    READY_NODE_SELECTION_INPUT_SOURCE_TYPE,
    ActionProposalCapability,
    ActionProposalDraft,
    InferenceCorrelation,
    ReadyNodeSelectionEffectNormalizer,
    ReadyNodeSelectionProposalProducer,
    build_ready_node_action_request,
)
from adaptive_agent_runtime.orchestration import (
    READY_NODE_SELECT_OPERATION,
    READY_NODE_SELECTION_DECISION_TYPE,
    ReadyNodeCandidateBinding,
    ReadyNodeSelectionDecisionPayload,
    ReadyNodeSelectionEffect,
    ReadyNodeSelectionExecutionPolicy,
    TaskNode,
    ready_node_candidate_set_fingerprint,
    ready_node_selection_fingerprint,
    stable_ready_node_selection_id,
)

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


class _FixedReadyNodeBasisProvider:
    module_id = "research_agent.ready_node_selection.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class LLMReadyTaskNodeSelector:
    """ReadyTaskNodeSelector backed by the shared Decision Lifecycle."""

    module_id = "research_agent.ready_node_selector.decision"

    def __init__(
        self,
        *,
        capability: ActionProposalCapability,
        workspace: ResearchWorkspace,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        execution_policy: ReadyNodeSelectionExecutionPolicy | None = None,
        checkpoint_store: DecisionCheckpointStore[
            ReadyNodeSelectionDecisionPayload,
            ActionProposalDraft,
            ReadyNodeSelectionEffect,
        ]
        | None = None,
    ) -> None:
        self._capability = capability
        self._workspace = workspace
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._execution_policy = execution_policy or ReadyNodeSelectionExecutionPolicy()
        self._checkpoint_store = checkpoint_store or InMemoryDecisionCheckpointStore()

    async def select_node_id(
        self,
        ready_nodes: tuple[TaskNode, ...],
        state: AgentState,
    ) -> UUID:
        if not ready_nodes:
            raise RuntimeError("Ready Node selection requires candidates")
        if len(ready_nodes) == 1:
            return ready_nodes[0].node_id
        bindings = tuple(
            ReadyNodeCandidateBinding(
                node_key=self._workspace.role_for(node),
                node=node,
                node_fingerprint=ready_node_selection_fingerprint(node),
            )
            for node in ready_nodes
        )
        payload = ReadyNodeSelectionDecisionPayload(
            task_description=state.task.description,
            candidates=bindings,
            candidate_set_fingerprint=ready_node_candidate_set_fingerprint(bindings),
            state_revision=state.revision,
            execution_policy=self._execution_policy,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {
                    "run_id": state.run_id,
                    "state_revision": state.revision,
                    "candidate_set_fingerprint": payload.candidate_set_fingerprint,
                }
            ),
            state_revision=state.revision,
        )
        evidence_id = f"runtime:ready:{state.run_id}:{state.revision}"
        request = DecisionRequest[ReadyNodeSelectionDecisionPayload](
            request_id=stable_ready_node_selection_id(
                "decision-request",
                state.run_id,
                state.revision,
                payload.candidate_set_fingerprint,
            ),
            decision_type=READY_NODE_SELECTION_DECISION_TYPE,
            target=DecisionTarget(target_type="ready_set", target_id=str(state.run_id)),
            correlation=DecisionCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(READY_NODE_SELECT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="ready-node-candidate-set",
                    description="Select exactly one Runtime-computed ready node.",
                ),
                DecisionConstraint(
                    constraint_id="ready-node-no-dispatch",
                    description="Selection cannot dispatch a node or modify the DAG.",
                ),
            ),
            evidence=(
                DecisionEvidenceReference(
                    evidence_id=evidence_id,
                    kind="runtime.ready_set",
                    source="graph_scheduler",
                    reliability=1.0,
                    summary="Runtime scheduler computed this immutable candidate set.",
                ),
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=self._execution_policy.max_agent_calls,
                max_revision_count=self._execution_policy.max_revision_count,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        projected = build_ready_node_action_request(payload, request.evidence)
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="ready-node-selection-input",
                    source_type=READY_NODE_SELECTION_INPUT_SOURCE_TYPE,
                    agent_scope="ready_node_selection",
                    content=projected.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    evidence_id=evidence_id,
                    priority=100,
                    estimated_tokens=128 + 64 * len(bindings),
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.ready_node_selection.context",
            version="1",
            agent_scope="ready_node_selection",
            allowed_decision_types=frozenset({READY_NODE_SELECTION_DECISION_TYPE}),
            allowed_source_types=frozenset({READY_NODE_SELECTION_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=_REDACT_KEYS,
            max_items=1,
            max_context_tokens=256 + 64 * len(bindings),
        )
        recording_evaluator = RecordingGovernanceEvaluator(self._governance)
        recording_issuer = RecordingAuthorizationIssuer(self._issuer)

        async def apply_effect(effect: ReadyNodeSelectionEffect) -> JsonValue:
            binding = payload.binding_for(effect.node_key)
            if binding.node.node_id != effect.node_id:
                raise RuntimeError("Ready Node effect no longer matches its binding")
            if binding.node_fingerprint != effect.node_fingerprint:
                raise RuntimeError("Ready Node effect fingerprint is stale")
            return {"node_id": str(effect.node_id), "node_key": effect.node_key}

        coordinator = DecisionLifecycleCoordinator[
            ReadyNodeSelectionDecisionPayload,
            ActionProposalDraft,
            ReadyNodeSelectionEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=ReadyNodeSelectionProposalProducer(
                capability=self._capability,
                timeout_seconds=self._execution_policy.timeout_seconds,
                correlation=InferenceCorrelation(
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                ),
            ),
            basis_provider=_FixedReadyNodeBasisProvider(basis),
            validator=RuntimeDecisionValidator(
                normalizer=ReadyNodeSelectionEffectNormalizer()
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
                ReadyNodeSelectionDecisionPayload,
                ActionProposalDraft,
                ReadyNodeSelectionEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = checkpoint.governance_receipt
            if receipt is None or receipt.review_request_id is None:
                raise RuntimeError("Ready Node review has no Review Request")
            recording_evaluator.review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="The choice is bounded to Runtime-ready nodes.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            checkpoint = await coordinator.resume_review(request.request_id)
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.proposal is None
            or checkpoint.validated_decision is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Ready Node decision did not apply: {reason}")
        if (
            recording_evaluator.request is None
            or recording_evaluator.preliminary is None
            or recording_evaluator.final is None
        ):
            raise RuntimeError("Ready Node Governance record is incomplete")
        self._workspace.action_proposals.append(checkpoint.proposal.payload)
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="action_proposal",
                request=recording_evaluator.request,
                preliminary=recording_evaluator.preliminary,
                final=recording_evaluator.final,
                authorization=recording_issuer.authorization,
                review=recording_evaluator.review,
            )
        )
        return checkpoint.validated_decision.normalized_effect.payload.node_id
