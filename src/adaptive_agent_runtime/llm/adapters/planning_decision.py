"""Planner Agent adapter into the Runtime-owned Decision Lifecycle."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from time import monotonic

from pydantic import JsonValue

from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    DecisionGovernanceScope,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionRiskLevel,
    NormalizedDecisionEffect,
    decision_fingerprint,
)
from adaptive_agent_runtime.llm.capabilities.contracts import (
    TaskGraphProposalCapability,
)
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    TaskGraphDraft,
    TaskPlanningRequest,
)
from adaptive_agent_runtime.llm.models import InferenceCorrelation
from adaptive_agent_runtime.orchestration.planning import (
    GRAPH_INITIALIZE_OPERATION,
    PLANNING_DECISION_TYPE,
    PlanningDecisionPayload,
    PlanningExecutionPolicy,
    PlanningGraphEffect,
    PlanningGraphEffectProjector,
    PlanningGraphValidator,
    PlanningStrategyRisk,
    SymbolicTaskGraph,
    SymbolicTaskNode,
)


PLANNING_INPUT_SOURCE_TYPE = "planning_input"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:^|\s)(?:execute_tool|runtime_state_change|state_patch|"
    r"runtime\.apply|tool\.call)\s*\("
)


class PlannerTaskRequestAdapter:
    """Build the cognitive request from projected blocks, never Runtime state."""

    module_id = "llm.adapter.planning.task_request"

    def to_request(self, context: AgentContext) -> TaskPlanningRequest:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == PLANNING_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError("Planner context requires one planning_input block")
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("planning_input content must be an object")
        goal = self._string(content, "goal")
        strategies = self._string_tuple(content, "available_strategies")
        capabilities = self._string_tuple(
            content,
            "available_execution_capability_ids",
            required=False,
        )
        projected_constraints = self._string_tuple(
            content,
            "constraints",
            required=False,
        )
        runtime_constraints = tuple(item.description for item in context.constraints)
        evidence = tuple(
            EvidenceReference(
                reference_id=item.evidence_id,
                kind=item.kind,
                summary=item.summary,
                reliability=item.reliability,
            )
            for item in context.evidence
        )
        return TaskPlanningRequest(
            task=goal,
            constraints=tuple(dict.fromkeys((*projected_constraints, *runtime_constraints))),
            available_strategies=strategies,
            available_execution_capability_ids=capabilities,
            evidence=evidence,
        )

    @staticmethod
    def _string(content: Mapping[str, JsonValue], key: str) -> str:
        value = content.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"planning_input '{key}' must be a non-empty string")
        return value

    @staticmethod
    def _string_tuple(
        content: Mapping[str, JsonValue],
        key: str,
        *,
        required: bool = True,
    ) -> tuple[str, ...]:
        value = content.get(key)
        if value is None and not required:
            return ()
        if not isinstance(value, (tuple, list)) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"planning_input '{key}' must contain strings")
        result = tuple(str(item) for item in value)
        if required and not result:
            raise ValueError(f"planning_input '{key}' cannot be empty")
        return result


class PlannerDecisionProposalProducer:
    """Make one bounded Agent call and wrap only its TaskGraphDraft."""

    module_id = "llm.adapter.planning.proposal_producer"

    def __init__(
        self,
        *,
        capability: TaskGraphProposalCapability,
        execution_policy: PlanningExecutionPolicy,
        correlation: InferenceCorrelation,
        confidence: float = 0.9,
        request_adapter: PlannerTaskRequestAdapter | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Planner proposal confidence must be between zero and one")
        self._capability = capability
        self._execution_policy = execution_policy
        self._correlation = correlation
        self._confidence = confidence
        self._request_adapter = request_adapter or PlannerTaskRequestAdapter()
        self._called_request_ids: set[object] = set()

    async def propose(self, context: AgentContext) -> AgentCallResult[TaskGraphDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("PlanningExecutionPolicy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = self._request_adapter.to_request(context)
        started = monotonic()
        invocation = CapabilityInvocationMetadata(
            correlation=self._correlation,
            trace_attributes={
                "operation": "graph.initialize.propose",
                "planning_model": self._execution_policy.model,
                "planning_temperature": self._execution_policy.temperature,
                "planning_timeout_seconds": self._execution_policy.timeout_seconds,
                "planning_max_agent_calls": self._execution_policy.max_agent_calls,
                "planning_max_retries": (
                    self._execution_policy.retry_policy.max_retries
                ),
            },
        )
        turn = await asyncio.wait_for(
            self._capability.propose(request, invocation=invocation),
            timeout=self._execution_policy.timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Planner Agent cannot return ToolIntent in Phase 1-A")
        if turn.result is None or not isinstance(turn.result, TaskGraphDraft):
            raise RuntimeError("Planner Agent must return only TaskGraphDraft")
        draft = turn.result
        proposal = DecisionProposal[TaskGraphDraft](
            request_id=context.request_id,
            proposal_type=context.decision_type,
            producer=DecisionProducer(
                producer_id=self._capability.capability_id,
                capability="task_graph_proposal",
                model_id=self._execution_policy.model,
                implementation_version="phase-1-a",
            ),
            input_snapshot_fingerprint=context.basis_fingerprint,
            context_fingerprint=context.context_fingerprint,
            revision=0,
            selected_action=GRAPH_INITIALIZE_OPERATION,
            payload=draft,
            rationale=draft.rationale or "Planner Agent supplied a task graph draft.",
            evidence_refs=tuple(item.evidence_id for item in context.evidence),
            confidence=self._confidence,
        )
        return AgentCallResult(
            proposal=proposal,
            elapsed_seconds=elapsed,
            cost_units=0.0,
        )


class TaskGraphDraftAdapter:
    """Convert an authority-free draft into Runtime symbolic data."""

    module_id = "llm.adapter.planning.task_graph_draft"

    def __init__(
        self,
        *,
        domain_validator: Callable[[TaskGraphDraft], None] | None = None,
    ) -> None:
        self._domain_validator = domain_validator

    def adapt(self, draft: TaskGraphDraft) -> SymbolicTaskGraph:
        if self._domain_validator is not None:
            self._domain_validator(draft)
        for node in draft.nodes:
            for value in (node.goal, node.expected_output):
                if _CONTROL_DIRECTIVE.search(value):
                    raise ValueError(
                        f"node '{node.node_key}' contains a reserved Runtime directive"
                    )
        if draft.rationale is not None and _CONTROL_DIRECTIVE.search(draft.rationale):
            raise ValueError("graph rationale contains a reserved Runtime directive")
        return SymbolicTaskGraph(
            nodes=tuple(
                SymbolicTaskNode(
                    node_key=node.node_key,
                    goal=node.goal,
                    dependency_keys=node.dependency_keys,
                    expected_output=node.expected_output,
                    requested_strategy_id=node.requested_strategy_id,
                )
                for node in draft.nodes
            )
        )


class PlanningGraphEffectNormalizer:
    """Validate the Agent draft and let Runtime create the final Graph Effect."""

    module_id = "llm.adapter.planning.effect_normalizer"

    def __init__(
        self,
        *,
        draft_adapter: TaskGraphDraftAdapter | None = None,
        graph_validator: PlanningGraphValidator | None = None,
        effect_projector: PlanningGraphEffectProjector | None = None,
    ) -> None:
        self._draft_adapter = draft_adapter or TaskGraphDraftAdapter()
        self._graph_validator = graph_validator or PlanningGraphValidator()
        self._effect_projector = effect_projector or PlanningGraphEffectProjector()

    def normalize(
        self,
        request: DecisionRequest[PlanningDecisionPayload],
        proposal: DecisionProposal[TaskGraphDraft],
    ) -> NormalizedDecisionEffect[PlanningGraphEffect]:
        if request.decision_type != PLANNING_DECISION_TYPE:
            raise ValueError("Planning normalizer received another decision type")
        if request.correlation.task_id is None:
            raise ValueError("Planning decision requires a task_id")
        draft_fingerprint = decision_fingerprint(proposal.payload)
        symbolic = self._draft_adapter.adapt(proposal.payload)
        validated = self._graph_validator.validate(
            request.payload,
            symbolic,
            source_draft_fingerprint=draft_fingerprint,
        )
        effect = self._effect_projector.project(
            request.payload,
            validated,
            request_id=request.request_id,
            run_id=request.correlation.run_id,
            task_id=request.correlation.task_id,
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        risk = {
            PlanningStrategyRisk.LOW: DecisionRiskLevel.LOW,
            PlanningStrategyRisk.MEDIUM: DecisionRiskLevel.MEDIUM,
            PlanningStrategyRisk.HIGH: DecisionRiskLevel.HIGH,
        }[effect.risk_assessment.risk]
        return NormalizedDecisionEffect[PlanningGraphEffect].create(
            payload=effect,
            operation=GRAPH_INITIALIZE_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=risk,
            impact_score=effect.risk_assessment.impact_score,
            reversible=effect.risk_assessment.reversible,
            impact_description=effect.risk_assessment.description,
        )
