"""Graph Mutation Agent adapter into the unified Decision Lifecycle."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from time import monotonic

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
from adaptive_agent_runtime.llm.capabilities.contracts import GraphMutationProposalCapability
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    GraphMutationProposalDraft,
    GraphMutationProposalRequest,
)
from adaptive_agent_runtime.llm.capabilities.validation import CapabilityDraftValidator
from adaptive_agent_runtime.llm.models import InferenceCorrelation
from adaptive_agent_runtime.orchestration.models import GraphMutation
from adaptive_agent_runtime.orchestration.mutation_decision import (
    GRAPH_MUTATION_APPLY_OPERATION,
    GRAPH_MUTATION_DECISION_TYPE,
    GraphMutationDecisionPayload,
    GraphMutationEffect,
)


GRAPH_MUTATION_INPUT_SOURCE_TYPE = "graph_mutation_input"
GRAPH_MUTATION_EVIDENCE_SOURCE_TYPE = "graph_mutation_evidence"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:execute_tool|runtime_state_change|state_patch|runtime\.apply|"
    r"governance\.(?:allow|deny)|authorize)\s*\("
)


class GraphMutationRequestAdapter:
    module_id = "llm.adapter.graph_mutation.request"

    def to_request(self, context: AgentContext) -> GraphMutationProposalRequest:
        blocks = tuple(
            item for item in context.blocks if item.source_type == GRAPH_MUTATION_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError("Graph Mutation Agent context requires one input block")
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("Graph Mutation input must be an object")
        return GraphMutationProposalRequest.model_validate(dict(content))


class GraphMutationProposalProducer:
    module_id = "llm.adapter.graph_mutation.proposal_producer"

    def __init__(
        self,
        *,
        capability: GraphMutationProposalCapability,
        timeout_seconds: float,
        correlation: InferenceCorrelation,
    ) -> None:
        self._capability = capability
        self._timeout_seconds = timeout_seconds
        self._correlation = correlation
        self._called_request_ids: set[object] = set()

    async def propose(self, context: AgentContext) -> AgentCallResult[GraphMutationProposalDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("Graph Mutation policy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = GraphMutationRequestAdapter().to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.propose_mutations(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "graph.mutate.propose"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Graph Mutation Agent cannot execute ToolIntents")
        if turn.result is None or not isinstance(turn.result, GraphMutationProposalDraft):
            raise RuntimeError("Graph Mutation Agent must return GraphMutationProposalDraft")
        draft = turn.result
        evidence_refs = tuple(
            dict.fromkeys(
                reference
                for operation in draft.operations
                for reference in operation.evidence_reference_ids
            )
        )
        return AgentCallResult(
            proposal=DecisionProposal[GraphMutationProposalDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="graph_mutation_proposal",
                    implementation_version="phase-2d",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=GRAPH_MUTATION_APPLY_OPERATION,
                payload=draft,
                rationale=draft.rationale,
                evidence_refs=evidence_refs,
                confidence=0.5,
            ),
            elapsed_seconds=elapsed,
        )


class GraphMutationEffectNormalizer:
    module_id = "llm.adapter.graph_mutation.effect_normalizer"

    def __init__(
        self,
        *,
        mutation_projector: Callable[[GraphMutationProposalDraft], tuple[GraphMutation, ...]],
    ) -> None:
        self._mutation_projector = mutation_projector
        self._draft_validator = CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[GraphMutationDecisionPayload],
        proposal: DecisionProposal[GraphMutationProposalDraft],
    ) -> NormalizedDecisionEffect[GraphMutationEffect]:
        if request.decision_type != GRAPH_MUTATION_DECISION_TYPE:
            raise ValueError("Graph Mutation normalizer received another decision type")
        projected = GraphMutationProposalRequest.model_validate(request.payload.agent_input)
        draft = self._draft_validator.validate_graph_mutation_proposal(
            projected,
            proposal.payload,
        )
        serialized = json.dumps(draft.model_dump(mode="json"), ensure_ascii=False)
        if _CONTROL_DIRECTIVE.search(serialized):
            raise ValueError("Graph Mutation Draft contains a Runtime directive")
        mutations = self._mutation_projector(draft)
        if not mutations:
            raise ValueError("Graph Mutation Draft normalized to no mutations")
        preview = request.payload.graph
        for mutation in mutations:
            preview = preview.apply_mutation(mutation)
        effect = GraphMutationEffect(
            graph_id=request.payload.graph.graph_id,
            graph_version_before=request.payload.graph.version,
            graph_version_after=preview.version,
            source_node_id=request.payload.source_node_id,
            source_action_id=request.payload.source_action_id,
            mutations=mutations,
            proposal_fingerprint=decision_fingerprint(draft),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[GraphMutationEffect].create(
            payload=effect,
            operation=GRAPH_MUTATION_APPLY_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.MEDIUM,
            impact_score=0.4,
            reversible=True,
            impact_description="Apply a validated mutation batch to this run's Task Graph.",
        )
