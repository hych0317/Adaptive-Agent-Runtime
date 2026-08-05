"""Memory Extraction Agent adapter into the unified Decision Lifecycle."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from time import monotonic

from adaptive_agent_runtime.context_memory import (
    MEMORY_CANDIDATES_APPLY_OPERATION,
    MEMORY_EXTRACTION_DECISION_TYPE,
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryExtractionDecisionPayload,
    MemoryExtractionEffect,
    memory_decision_fingerprint,
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
    decision_fingerprint,
)
from adaptive_agent_runtime.llm.capabilities.contracts import MemoryExtractionCapability
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    MemoryCandidateBatchDraft,
    MemoryExtractionRequest,
)
from adaptive_agent_runtime.llm.capabilities.validation import CapabilityDraftValidator
from adaptive_agent_runtime.llm.models import InferenceCorrelation


MEMORY_EXTRACTION_INPUT_SOURCE_TYPE = "memory_extraction_input"
MEMORY_EXTRACTION_EVIDENCE_SOURCE_TYPE = "memory_extraction_evidence"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:execute_tool|runtime_state_change|state_patch|runtime\.apply|"
    r"memory\.(?:write|delete)|governance\.(?:allow|deny))\s*\("
)


def build_memory_extraction_request(
    payload: MemoryExtractionDecisionPayload,
) -> MemoryExtractionRequest:
    return MemoryExtractionRequest.model_validate(payload.agent_input)


class MemoryExtractionRequestAdapter:
    module_id = "llm.adapter.memory_extraction.request"

    def to_request(self, context: AgentContext) -> MemoryExtractionRequest:
        blocks = tuple(
            item for item in context.blocks if item.source_type == MEMORY_EXTRACTION_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError("Memory Extraction Agent context requires one input block")
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("Memory Extraction input must be an object")
        return MemoryExtractionRequest.model_validate(dict(content))


class MemoryExtractionProposalProducer:
    module_id = "llm.adapter.memory_extraction.proposal_producer"

    def __init__(
        self,
        *,
        capability: MemoryExtractionCapability,
        timeout_seconds: float,
        correlation: InferenceCorrelation,
    ) -> None:
        self._capability = capability
        self._timeout_seconds = timeout_seconds
        self._correlation = correlation
        self._called_request_ids: set[object] = set()

    async def propose(self, context: AgentContext) -> AgentCallResult[MemoryCandidateBatchDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("Memory Extraction policy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = MemoryExtractionRequestAdapter().to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.extract(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "memory.extract"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Memory Extraction Agent cannot execute ToolIntents")
        if turn.result is None:
            raise RuntimeError("Memory Extraction Agent returned no result")
        draft = MemoryCandidateBatchDraft(candidates=tuple(turn.result))
        evidence_refs = tuple(
            dict.fromkeys(
                reference
                for candidate in draft.candidates
                for reference in candidate.evidence_reference_ids
            )
        )
        return AgentCallResult(
            proposal=DecisionProposal[MemoryCandidateBatchDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="memory_extraction",
                    implementation_version="phase-2d",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=MEMORY_CANDIDATES_APPLY_OPERATION,
                payload=draft,
                rationale="Propose evidence-bound Memory candidates.",
                evidence_refs=evidence_refs,
                confidence=(
                    min((item.confidence for item in draft.candidates), default=0.5)
                ),
            ),
            elapsed_seconds=elapsed,
        )


class MemoryExtractionEffectNormalizer:
    module_id = "llm.adapter.memory_extraction.effect_normalizer"

    def __init__(self) -> None:
        self._draft_validator = CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[MemoryExtractionDecisionPayload],
        proposal: DecisionProposal[MemoryCandidateBatchDraft],
    ) -> NormalizedDecisionEffect[MemoryExtractionEffect]:
        if request.decision_type != MEMORY_EXTRACTION_DECISION_TYPE:
            raise ValueError("Memory Extraction normalizer received another decision type")
        projected = build_memory_extraction_request(request.payload)
        drafts = self._draft_validator.validate_memory_extraction(
            projected,
            proposal.payload.candidates,
        )
        if len(drafts) > request.payload.execution_policy.max_candidates:
            raise ValueError("Memory Extraction exceeded Runtime candidate budget")
        serialized = json.dumps(proposal.payload.model_dump(mode="json"), ensure_ascii=False)
        if _CONTROL_DIRECTIVE.search(serialized):
            raise ValueError("Memory Draft contains a Runtime directive")
        evidence_by_ref = {item.reference_id: item for item in request.payload.evidence}
        candidates: list[MemoryCandidate] = []
        for draft in drafts:
            target_id = (
                request.payload.memory_id_for(draft.target_memory_reference)
                if draft.target_memory_reference is not None
                else None
            )
            candidates.append(
                MemoryCandidate(
                    memory_key=draft.memory_key,
                    content=draft.content,
                    condition=MemoryCondition(
                        facts=draft.condition.facts,
                        required_tags=draft.condition.required_tags,
                        description=draft.condition.description,
                    ),
                    evidence=tuple(
                        MemoryEvidence(
                            source_context_id=evidence_by_ref[reference].source_context_id,
                            source_reference=evidence_by_ref[reference].source_reference,
                            note=evidence_by_ref[reference].summary,
                            weight=evidence_by_ref[reference].reliability,
                        )
                        for reference in draft.evidence_reference_ids
                    ),
                    confidence=draft.confidence,
                    evolution=MemoryEvolutionType(draft.evolution.value),
                    target_memory_id=target_id,
                )
            )
        candidate_tuple = tuple(candidates)
        effect = MemoryExtractionEffect(
            candidates=candidate_tuple,
            candidate_batch_fingerprint=memory_decision_fingerprint(
                candidate_tuple
            ),
            proposal_fingerprint=decision_fingerprint(proposal.payload),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[MemoryExtractionEffect].create(
            payload=effect,
            operation=MEMORY_CANDIDATES_APPLY_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.MEDIUM if candidate_tuple else DecisionRiskLevel.LOW,
            impact_score=0.4 if candidate_tuple else 0.0,
            reversible=True,
            impact_description=(
                "Consolidate the exact evidence-bound Memory candidate batch."
                if candidate_tuple
                else "Record that extraction produced no persistent Memory candidates."
            ),
        )
