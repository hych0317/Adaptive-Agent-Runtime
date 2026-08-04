"""Semantic Root Cause Agent adapter into the Runtime Decision Lifecycle."""

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
    decision_fingerprint,
)
from adaptive_agent_runtime.evaluation import TraceCompleteness
from adaptive_agent_runtime.evaluation.root_cause import (
    ROOT_CAUSE_DECISION_TYPE,
    ROOT_CAUSE_RECORD_OPERATION,
    RootCauseAlternative,
    RootCauseAssessment,
    RootCauseAssessmentEffect,
    RootCauseConclusion,
    RootCauseDecisionPayload,
    RootCauseDeterministicAlignment,
    RootCauseEvidenceStrength,
    RootCauseExecutionPolicy,
    RootCauseTrigger,
    stable_root_cause_id,
)
from adaptive_agent_runtime.llm.capabilities.contracts import (
    RootCauseAnalysisCapability,
)
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    RootCauseAnalysisRequest,
    RootCauseConclusionDraft,
    RootCauseDeterministicFinding,
    RootCauseDraft,
)
from adaptive_agent_runtime.llm.capabilities.validation import (
    CapabilityDraftValidator,
)
from adaptive_agent_runtime.llm.models import InferenceCorrelation


ROOT_CAUSE_INPUT_SOURCE_TYPE = "root_cause_input"
ROOT_CAUSE_EVIDENCE_SOURCE_TYPE = "root_cause_evidence"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:^|\s)(?:execute_tool|runtime_state_change|state_patch|"
    r"runtime\.apply|tool\.call|graph\.mutate|memory\.write)\s*\("
)


def build_root_cause_analysis_request(
    payload: RootCauseDecisionPayload,
    evidence: tuple[DecisionEvidenceReference, ...],
) -> RootCauseAnalysisRequest:
    """Project semantic evidence without Runtime, Governance, or producer identity."""

    return RootCauseAnalysisRequest(
        trigger=payload.trigger.value,
        failure_summary=payload.failure_summary,
        trace_completeness=payload.trace_completeness.value,
        deterministic_findings=tuple(
            RootCauseDeterministicFinding(
                code=item.code,
                component=item.component.value,
                severity=item.severity.value,
                summary=item.summary,
                evidence_reference_ids=item.evidence_ids,
            )
            for item in payload.deterministic_findings
        ),
        evidence_catalog=tuple(
            EvidenceReference(
                reference_id=item.evidence_id,
                kind=item.kind,
                summary=item.summary,
                reliability=item.reliability,
            )
            for item in evidence
        ),
        constraints=(
            "Cite only evidence from the supplied catalog.",
            "Return inconclusive when the evidence cannot support one cause.",
            "Treat deterministic findings as immutable facts, not editable scores.",
            "Do not propose or execute Recovery, Tool, Graph, Memory, or State actions.",
        ),
    )


class RootCauseRequestAdapter:
    module_id = "llm.adapter.root_cause.request"

    def to_request(self, context: AgentContext) -> RootCauseAnalysisRequest:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == ROOT_CAUSE_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError("Root Cause context requires one root_cause_input block")
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("root_cause_input content must be an object")
        return RootCauseAnalysisRequest.model_validate(dict(content))


class RootCauseDecisionProposalProducer:
    """Make one isolated Agent call and return an authority-free hypothesis."""

    module_id = "llm.adapter.root_cause.proposal_producer"

    def __init__(
        self,
        *,
        capability: RootCauseAnalysisCapability,
        execution_policy: RootCauseExecutionPolicy,
        correlation: InferenceCorrelation,
        request_adapter: RootCauseRequestAdapter | None = None,
    ) -> None:
        self._capability = capability
        self._execution_policy = execution_policy
        self._correlation = correlation
        self._request_adapter = request_adapter or RootCauseRequestAdapter()
        self._called_request_ids: set[object] = set()

    async def propose(self, context: AgentContext) -> AgentCallResult[RootCauseDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("Root Cause policy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = self._request_adapter.to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.analyze_root_cause(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={
                        "operation": "evaluation.root_cause.analyze",
                        "root_cause_timeout_seconds": (
                            self._execution_policy.timeout_seconds
                        ),
                        "root_cause_max_agent_calls": (
                            self._execution_policy.max_agent_calls
                        ),
                    },
                ),
            ),
            timeout=self._execution_policy.timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Root Cause Agent cannot execute ToolIntents")
        if turn.result is None or not isinstance(turn.result, RootCauseDraft):
            raise RuntimeError("Root Cause Agent must return only RootCauseDraft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal[RootCauseDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="root_cause_analysis",
                    implementation_version="phase-2a",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                revision=0,
                selected_action=ROOT_CAUSE_RECORD_OPERATION,
                payload=draft,
                rationale=draft.rationale,
                evidence_refs=draft.evidence_reference_ids,
                confidence=draft.confidence,
            ),
            elapsed_seconds=elapsed,
            cost_units=0.0,
        )


class RootCauseEffectNormalizer:
    """Verify semantic evidence and create a Runtime-owned advisory record."""

    module_id = "llm.adapter.root_cause.effect_normalizer"

    def __init__(
        self,
        *,
        draft_validator: CapabilityDraftValidator | None = None,
    ) -> None:
        self._draft_validator = draft_validator or CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[RootCauseDecisionPayload],
        proposal: DecisionProposal[RootCauseDraft],
    ) -> NormalizedDecisionEffect[RootCauseAssessmentEffect]:
        if request.decision_type != ROOT_CAUSE_DECISION_TYPE:
            raise ValueError("Root Cause normalizer received another decision type")
        draft = proposal.payload
        projected = build_root_cause_analysis_request(
            request.payload,
            request.evidence,
        )
        self._draft_validator.validate_root_cause(projected, draft)
        self._reject_control_directives(draft)
        assessment = self._assessment(request, proposal)
        effect = RootCauseAssessmentEffect(
            assessment=assessment,
            source_draft_fingerprint=decision_fingerprint(draft),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[RootCauseAssessmentEffect].create(
            payload=effect,
            operation=ROOT_CAUSE_RECORD_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.1,
            reversible=True,
            impact_description=(
                "Append one validated advisory diagnosis without changing metrics, "
                "Recovery, Graph, Memory, or Runtime State."
            ),
        )

    def _assessment(
        self,
        request: DecisionRequest[RootCauseDecisionPayload],
        proposal: DecisionProposal[RootCauseDraft],
    ) -> RootCauseAssessment:
        payload = request.payload
        draft = proposal.payload
        known = {item.evidence_id for item in payload.evidence_bindings}
        cited = set(draft.evidence_reference_ids)
        if not cited.issubset(known):
            raise ValueError("Root Cause cites evidence outside the Runtime snapshot")
        conclusion = RootCauseConclusion(draft.conclusion.value)
        primary = draft.primary
        supporting = (
            primary.supporting_evidence_reference_ids
            if primary is not None
            else ()
        )
        counter = (
            primary.counter_evidence_reference_ids
            if primary is not None
            else ()
        )
        direct_ids = {
            item.evidence_id
            for item in payload.evidence_bindings
            if item.direct_failure
        }
        if conclusion is RootCauseConclusion.SUPPORTED:
            if payload.trace_completeness is TraceCompleteness.MISSING:
                raise ValueError(
                    "missing Trace coverage permits only an inconclusive Root Cause"
                )
            if (
                payload.trigger is RootCauseTrigger.INLINE_FAILURE
                and not direct_ids.intersection(supporting)
            ):
                raise ValueError(
                    "inline Root Cause must cite the direct failure Observation"
                )
        strength = self._evidence_strength(
            request.evidence,
            supporting,
            conclusion=conclusion,
            direct_ids=direct_ids,
        )
        alignment = self._deterministic_alignment(payload, draft)
        alternatives = tuple(
            RootCauseAlternative(
                code=item.code,
                description=item.description,
                supporting_evidence_ids=item.supporting_evidence_reference_ids,
            )
            for item in draft.alternatives
        )
        return RootCauseAssessment(
            assessment_id=stable_root_cause_id(
                "assessment",
                request.request_id,
                proposal.proposal_id,
                decision_fingerprint(draft),
            ),
            request_id=request.request_id,
            proposal_id=proposal.proposal_id,
            correlation=payload.correlation,
            trigger=payload.trigger,
            failure_signature=payload.failure_signature,
            conclusion=conclusion,
            primary_code=(primary.code if primary is not None else None),
            primary_description=(
                primary.description if primary is not None else None
            ),
            alternatives=alternatives,
            supporting_evidence_ids=supporting,
            counter_evidence_ids=counter,
            assumptions=draft.assumptions,
            unresolved_questions=draft.unresolved_questions,
            rationale=draft.rationale,
            stated_confidence=draft.confidence,
            evidence_strength=strength,
            deterministic_alignment=alignment,
            trace_completeness=payload.trace_completeness,
            deterministic_finding_codes=tuple(
                item.code for item in payload.deterministic_findings
            ),
        )

    @staticmethod
    def _evidence_strength(
        evidence: tuple[DecisionEvidenceReference, ...],
        supporting: tuple[str, ...],
        *,
        conclusion: RootCauseConclusion,
        direct_ids: set[str],
    ) -> RootCauseEvidenceStrength:
        if conclusion is RootCauseConclusion.INCONCLUSIVE:
            return RootCauseEvidenceStrength.INSUFFICIENT
        by_id = {item.evidence_id: item for item in evidence}
        reliability = tuple(by_id[item].reliability for item in supporting)
        if (
            len(supporting) >= 2
            and bool(direct_ids.intersection(supporting))
            and sum(reliability) / len(reliability) >= 0.75
        ):
            return RootCauseEvidenceStrength.CORROBORATED
        return RootCauseEvidenceStrength.LIMITED

    @staticmethod
    def _deterministic_alignment(
        payload: RootCauseDecisionPayload,
        draft: RootCauseDraft,
    ) -> RootCauseDeterministicAlignment:
        findings = payload.deterministic_findings
        if not findings:
            return RootCauseDeterministicAlignment.NOT_AVAILABLE
        if draft.primary is None:
            return RootCauseDeterministicAlignment.PARTIAL
        deterministic_evidence = {
            evidence_id for finding in findings for evidence_id in finding.evidence_ids
        }
        supporting = set(draft.primary.supporting_evidence_reference_ids)
        if draft.primary.code in {item.code for item in findings}:
            return RootCauseDeterministicAlignment.ALIGNED
        if deterministic_evidence.intersection(supporting):
            return RootCauseDeterministicAlignment.PARTIAL
        return RootCauseDeterministicAlignment.UNVERIFIED

    @staticmethod
    def _reject_control_directives(draft: RootCauseDraft) -> None:
        values = [draft.rationale, *draft.assumptions, *draft.unresolved_questions]
        if draft.primary is not None:
            values.extend([draft.primary.code, draft.primary.description])
        for item in draft.alternatives:
            values.extend([item.code, item.description])
        if any(_CONTROL_DIRECTIVE.search(value) for value in values):
            raise ValueError("Root Cause Draft contains a reserved Runtime directive")
