"""Tool Selection Agent adapter into the Runtime-owned Decision Lifecycle."""

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
from adaptive_agent_runtime.llm.capabilities.contracts import (
    ToolSelectionProposalCapability,
)
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    ToolSelectionCandidate,
    ToolSelectionDraft,
    ToolSelectionProposalRequest,
)
from adaptive_agent_runtime.llm.capabilities.validation import (
    CapabilityDraftValidator,
)
from adaptive_agent_runtime.llm.models import InferenceCorrelation
from adaptive_agent_runtime.tool_ecosystem.selection_decision import (
    TOOL_SELECTION_BIND_OPERATION,
    TOOL_SELECTION_DECISION_TYPE,
    ToolSelectionDecisionPayload,
    ToolSelectionEffect,
    ToolSelectionExecutionPolicy,
)


TOOL_SELECTION_INPUT_SOURCE_TYPE = "tool_selection_input"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:^|\s)(?:execute_tool|runtime_state_change|state_patch|"
    r"runtime\.apply|tool\.call|provider\.invoke)\s*\("
)


def build_tool_selection_proposal_request(
    payload: ToolSelectionDecisionPayload,
    evidence: tuple[DecisionEvidenceReference, ...],
) -> ToolSelectionProposalRequest:
    """Project only an allowlisted, credential-free candidate view."""

    return ToolSelectionProposalRequest(
        task=payload.task_description,
        node_goal=payload.node_goal,
        capability_id=payload.requirement.capability_id,
        preferred_tags=payload.requirement.preferred_provider_tags,
        context_tags=payload.selection_context.tags,
        candidates=tuple(
            ToolSelectionCandidate(
                candidate_ref=item.candidate_ref,
                name=item.metadata.name,
                description=item.metadata.description,
                capability_id=item.metadata.capability_id,
                tags=item.metadata.tags,
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


class ToolSelectionRequestAdapter:
    module_id = "llm.adapter.tool_selection.request"

    def to_request(self, context: AgentContext) -> ToolSelectionProposalRequest:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == TOOL_SELECTION_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError(
                "Tool Selection Agent context requires one tool_selection_input block"
            )
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("tool_selection_input content must be an object")
        return ToolSelectionProposalRequest.model_validate(dict(content))


class ToolSelectionDecisionProposalProducer:
    """Make one bounded Agent call; its Draft cannot invoke a Provider."""

    module_id = "llm.adapter.tool_selection.proposal_producer"

    def __init__(
        self,
        *,
        capability: ToolSelectionProposalCapability,
        execution_policy: ToolSelectionExecutionPolicy,
        correlation: InferenceCorrelation,
        request_adapter: ToolSelectionRequestAdapter | None = None,
    ) -> None:
        self._capability = capability
        self._execution_policy = execution_policy
        self._correlation = correlation
        self._request_adapter = request_adapter or ToolSelectionRequestAdapter()
        self._called_request_ids: set[object] = set()

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[ToolSelectionDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("ToolSelectionExecutionPolicy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = self._request_adapter.to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.propose_tool_selection(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={
                        "operation": "tool.selection.propose",
                        "tool_selection_timeout_seconds": (
                            self._execution_policy.timeout_seconds
                        ),
                        "tool_selection_max_agent_calls": (
                            self._execution_policy.max_agent_calls
                        ),
                    },
                ),
            ),
            timeout=self._execution_policy.timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Tool Selection Agent cannot execute ToolIntents")
        if turn.result is None or not isinstance(turn.result, ToolSelectionDraft):
            raise RuntimeError("Tool Selection Agent must return only ToolSelectionDraft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal[ToolSelectionDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="tool_selection_proposal",
                    implementation_version="phase-2b-tool-selection",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                revision=0,
                selected_action=TOOL_SELECTION_BIND_OPERATION,
                payload=draft,
                rationale=draft.rationale,
                evidence_refs=draft.evidence_reference_ids,
                confidence=draft.confidence,
            ),
            elapsed_seconds=elapsed,
            cost_units=0.0,
        )


class ToolSelectionEffectNormalizer:
    """Bind one opaque Agent choice to a Runtime-owned Provider effect."""

    module_id = "llm.adapter.tool_selection.effect_normalizer"

    def __init__(
        self,
        *,
        draft_validator: CapabilityDraftValidator | None = None,
    ) -> None:
        self._draft_validator = draft_validator or CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[ToolSelectionDecisionPayload],
        proposal: DecisionProposal[ToolSelectionDraft],
    ) -> NormalizedDecisionEffect[ToolSelectionEffect]:
        if request.decision_type != TOOL_SELECTION_DECISION_TYPE:
            raise ValueError("Tool Selection normalizer received another decision type")
        draft = proposal.payload
        projected_request = build_tool_selection_proposal_request(
            request.payload,
            request.evidence,
        )
        self._draft_validator.validate_tool_selection(projected_request, draft)
        self._reject_control_directives(draft)
        binding = request.payload.binding_for(draft.selected_candidate_ref)
        effect = ToolSelectionEffect(
            invocation_id=request.payload.invocation_id,
            requirement_id=request.payload.requirement.requirement_id,
            capability_id=request.payload.requirement.capability_id,
            candidate_ref=binding.candidate_ref,
            provider_id=binding.metadata.provider_id,
            candidate_set_fingerprint=request.payload.candidate_set_fingerprint,
            provider_metadata_fingerprint=binding.metadata_fingerprint,
            basis_fingerprint=request.basis.snapshot_fingerprint,
            selection_reason=draft.rationale,
            agent_confidence=draft.confidence,
        )
        return NormalizedDecisionEffect[ToolSelectionEffect].create(
            payload=effect,
            operation=TOOL_SELECTION_BIND_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.ACTION,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.1,
            reversible=True,
            impact_description=(
                "Bind one Runtime-filtered Provider to a pending Tool invocation; "
                "the binding does not execute the Provider."
            ),
        )

    @staticmethod
    def _reject_control_directives(draft: ToolSelectionDraft) -> None:
        if _CONTROL_DIRECTIVE.search(draft.rationale):
            raise ValueError("Tool Selection Draft contains an execution directive")
