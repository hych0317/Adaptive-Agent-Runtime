"""Compression Agent adapter into the Runtime-owned Decision Lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import monotonic

from adaptive_agent_runtime.context_memory import (
    CONTEXT_COMPRESSION_APPLY_OPERATION,
    CONTEXT_COMPRESSION_DECISION_TYPE,
    ContextCompressionDecisionPayload,
    ContextCompressionEffect,
)
from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    DecisionGovernanceScope,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionRiskLevel,
    NormalizedDecisionEffect,
)
from adaptive_agent_runtime.llm.capabilities.contracts import (
    SemanticCompressionCapability,
)
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CompressedContextDraft,
    CompressionRequest,
)
from adaptive_agent_runtime.llm.capabilities.validation import (
    CapabilityDraftValidator,
)
from adaptive_agent_runtime.llm.models import InferenceCorrelation


CONTEXT_COMPRESSION_INPUT_SOURCE_TYPE = "context_compression_input"


def build_context_compression_proposal_request(
    payload: ContextCompressionDecisionPayload,
) -> CompressionRequest:
    """Expose only one Runtime-approved Context projection and its token limit."""

    return CompressionRequest(
        source_reference_id=str(payload.context_id),
        content=payload.agent_input,
        original_estimated_tokens=payload.original_estimated_tokens,
        target_max_tokens=payload.target_max_tokens,
    )


class ContextCompressionRequestAdapter:
    module_id = "llm.adapter.context_compression.request"

    def to_request(self, context: AgentContext) -> CompressionRequest:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == CONTEXT_COMPRESSION_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError(
                "Compression Agent context requires one context_compression_input block"
            )
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("context_compression_input content must be an object")
        return CompressionRequest.model_validate(dict(content))


class ContextCompressionDecisionProposalProducer:
    """Make one bounded Agent call that can return only a compression Draft."""

    module_id = "llm.adapter.context_compression.proposal_producer"

    def __init__(
        self,
        *,
        capability: SemanticCompressionCapability,
        correlation: InferenceCorrelation,
        timeout_seconds: float,
        confidence: float = 0.8,
        request_adapter: ContextCompressionRequestAdapter | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Compression Agent timeout must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Compression Agent confidence must be normalized")
        self._capability = capability
        self._correlation = correlation
        self._timeout_seconds = timeout_seconds
        self._confidence = confidence
        self._request_adapter = (
            request_adapter or ContextCompressionRequestAdapter()
        )
        self._called_request_ids: set[object] = set()

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[CompressedContextDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("Compression policy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = self._request_adapter.to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.compress(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "context.compress"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Compression Agent returned ToolIntent to Runtime")
        draft = turn.result
        if draft is None:
            raise RuntimeError("Compression Agent returned no draft")
        return AgentCallResult[CompressedContextDraft](
            proposal=DecisionProposal[CompressedContextDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="semantic_compression",
                    implementation_version="phase-2c-context-compression",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                revision=0,
                selected_action=CONTEXT_COMPRESSION_APPLY_OPERATION,
                payload=draft,
                rationale=(
                    "Retain the proposed semantic summary and its source-bound "
                    "core conclusions."
                ),
                evidence_refs=tuple(
                    item.evidence_id for item in context.evidence
                ),
                confidence=self._confidence,
            ),
            elapsed_seconds=elapsed,
            cost_units=0.0,
        )


class ContextCompressionEffectNormalizer:
    """Validate a Draft and create the only compression effect Runtime may apply."""

    module_id = "llm.adapter.context_compression.effect_normalizer"

    def __init__(
        self,
        *,
        draft_validator: CapabilityDraftValidator | None = None,
    ) -> None:
        self._draft_validator = draft_validator or CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[ContextCompressionDecisionPayload],
        proposal: DecisionProposal[CompressedContextDraft],
    ) -> NormalizedDecisionEffect[ContextCompressionEffect]:
        if request.decision_type != CONTEXT_COMPRESSION_DECISION_TYPE:
            raise ValueError("Compression normalizer received another decision type")
        projected_request = build_context_compression_proposal_request(
            request.payload
        )
        draft = self._draft_validator.validate_compression(
            projected_request,
            proposal.payload,
        )
        if draft.estimated_tokens >= request.payload.original_estimated_tokens:
            raise ValueError("compressed Context must reduce estimated token usage")
        effect = ContextCompressionEffect(
            context_id=request.payload.context_id,
            source_revision=request.payload.source_revision,
            source_snapshot_fingerprint=(
                request.payload.source_snapshot_fingerprint
            ),
            basis_fingerprint=request.basis.snapshot_fingerprint,
            content=draft.content,
            core_conclusions=draft.core_conclusions,
            original_estimated_tokens=(
                request.payload.original_estimated_tokens
            ),
            target_max_tokens=request.payload.target_max_tokens,
            estimated_tokens=draft.estimated_tokens,
        )
        return NormalizedDecisionEffect[ContextCompressionEffect].create(
            payload=effect,
            operation=CONTEXT_COMPRESSION_APPLY_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.3,
            reversible=True,
            impact_description=(
                "Accept a source-bound semantic compression Draft; the existing "
                "Context Lifecycle remains responsible for archive and state commit."
            ),
        )
