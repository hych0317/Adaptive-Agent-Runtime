"""Ready-node Agent adapter into the Runtime-owned Decision Lifecycle."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from time import monotonic

from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    DecisionEvidenceReference,
    DecisionGovernanceScope,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionRiskLevel,
    NormalizedDecisionEffect,
)
from adaptive_agent_runtime.llm.capabilities.contracts import ActionProposalCapability
from adaptive_agent_runtime.llm.capabilities.models import (
    ActionProposalDraft,
    ActionProposalRequest,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    PlanningActionCandidate,
)
from adaptive_agent_runtime.llm.capabilities.validation import CapabilityDraftValidator
from adaptive_agent_runtime.llm.models import InferenceCorrelation
from adaptive_agent_runtime.orchestration.selection_decision import (
    READY_NODE_SELECT_OPERATION,
    READY_NODE_SELECTION_DECISION_TYPE,
    ReadyNodeSelectionDecisionPayload,
    ReadyNodeSelectionEffect,
)


READY_NODE_SELECTION_INPUT_SOURCE_TYPE = "ready_node_selection_input"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:execute_tool|runtime_state_change|state_patch|runtime\.apply|"
    r"graph\.mutate|dispatch_node)\s*\("
)


def build_ready_node_action_request(
    payload: ReadyNodeSelectionDecisionPayload,
    evidence: tuple[DecisionEvidenceReference, ...],
) -> ActionProposalRequest:
    return ActionProposalRequest(
        task=payload.task_description,
        candidates=tuple(
            PlanningActionCandidate(
                node_key=item.node_key,
                goal=item.node.goal,
                expected_output=item.node.expected_output,
                strategy_id=item.node.strategy_id,
            )
            for item in payload.candidates
        ),
        evidence=tuple(
            EvidenceReference(
                reference_id=item.evidence_id,
                kind=item.kind,
                summary=item.summary,
                reliability=item.reliability,
            )
            for item in evidence
        ),
    )


class ReadyNodeSelectionRequestAdapter:
    module_id = "llm.adapter.ready_node_selection.request"

    def to_request(self, context: AgentContext) -> ActionProposalRequest:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == READY_NODE_SELECTION_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError("Ready Node Agent context requires one input block")
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("Ready Node input must be an object")
        return ActionProposalRequest.model_validate(dict(content))


class ReadyNodeSelectionProposalProducer:
    module_id = "llm.adapter.ready_node_selection.proposal_producer"

    def __init__(
        self,
        *,
        capability: ActionProposalCapability,
        timeout_seconds: float,
        correlation: InferenceCorrelation,
    ) -> None:
        self._capability = capability
        self._timeout_seconds = timeout_seconds
        self._correlation = correlation
        self._called_request_ids: set[object] = set()

    async def propose(self, context: AgentContext) -> AgentCallResult[ActionProposalDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("Ready Node policy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = ReadyNodeSelectionRequestAdapter().to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.propose_action(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "node.select.propose"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Ready Node Agent cannot execute ToolIntents")
        if turn.result is None or not isinstance(turn.result, ActionProposalDraft):
            raise RuntimeError("Ready Node Agent must return ActionProposalDraft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal[ActionProposalDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="ready_node_selection",
                    implementation_version="phase-2d",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=READY_NODE_SELECT_OPERATION,
                payload=draft,
                rationale=draft.rationale,
                evidence_refs=draft.evidence_reference_ids,
                confidence=0.5,
            ),
            elapsed_seconds=elapsed,
        )


class ReadyNodeSelectionEffectNormalizer:
    module_id = "llm.adapter.ready_node_selection.effect_normalizer"

    def __init__(self) -> None:
        self._draft_validator = CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[ReadyNodeSelectionDecisionPayload],
        proposal: DecisionProposal[ActionProposalDraft],
    ) -> NormalizedDecisionEffect[ReadyNodeSelectionEffect]:
        if request.decision_type != READY_NODE_SELECTION_DECISION_TYPE:
            raise ValueError("Ready Node normalizer received another decision type")
        projected = build_ready_node_action_request(request.payload, request.evidence)
        draft = self._draft_validator.validate_action_proposal(
            projected,
            proposal.payload,
        )
        if _CONTROL_DIRECTIVE.search(draft.rationale):
            raise ValueError("Ready Node Draft contains an execution directive")
        binding = request.payload.binding_for(draft.node_key)
        effect = ReadyNodeSelectionEffect(
            node_id=binding.node.node_id,
            node_key=binding.node_key,
            node_fingerprint=binding.node_fingerprint,
            candidate_set_fingerprint=request.payload.candidate_set_fingerprint,
            basis_fingerprint=request.basis.snapshot_fingerprint,
            selection_reason=draft.rationale,
        )
        return NormalizedDecisionEffect[ReadyNodeSelectionEffect].create(
            payload=effect,
            operation=READY_NODE_SELECT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.ACTION,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.1,
            reversible=True,
            impact_description=(
                "Order one Runtime-computed ready node; the effect cannot dispatch it."
            ),
        )
