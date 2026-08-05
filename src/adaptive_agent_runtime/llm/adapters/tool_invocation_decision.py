"""Bind a Reasoner ToolIntent to the Runtime Decision Lifecycle."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

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
from adaptive_agent_runtime.tool_ecosystem.invocation_decision import (
    TOOL_INVOCATION_DECISION_TYPE,
    TOOL_INVOCATION_INPUT_SOURCE_TYPE,
    TOOL_INVOCATION_OPERATION,
    ToolInvocationDecisionPayload,
    ToolInvocationEffect,
    ToolInvocationProposalDraft,
    tool_invocation_fingerprint,
)
from adaptive_agent_runtime.tool_ecosystem.models import (
    ImmutableJsonObject,
    ToolModel,
)


class ToolInvocationAgentInput(ToolModel):
    """Credential-free projection used to bind an upstream semantic proposal."""

    task_description: str = Field(min_length=1)
    node_goal: str = Field(min_length=1)
    allowed_capability_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_description: str = Field(min_length=1)
    input_schema: ImmutableJsonObject = Field(default_factory=dict)
    exact_argument_constraints: ImmutableJsonObject = Field(default_factory=dict)
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_tool_invocation_agent_input(
    payload: ToolInvocationDecisionPayload,
) -> ToolInvocationAgentInput:
    metadata = payload.provider_metadata
    return ToolInvocationAgentInput(
        task_description=payload.task_description,
        node_goal=payload.node_goal,
        allowed_capability_id=payload.requirement.capability_id,
        tool_name=metadata.name,
        tool_description=metadata.description,
        input_schema=metadata.input_schema,
        exact_argument_constraints=payload.exact_argument_constraints,
        proposal_fingerprint=payload.proposal_fingerprint,
    )


class BoundToolInvocationProposalProducer:
    """Import one already-produced Agent ToolIntent as a powerless Proposal."""

    module_id = "llm.adapter.tool_invocation.proposal_producer"

    def __init__(
        self,
        *,
        draft: ToolInvocationProposalDraft,
        producer_id: str,
    ) -> None:
        self._draft = draft
        self._producer_id = producer_id
        self._called_request_ids: set[object] = set()

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[ToolInvocationProposalDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("Tool Invocation accepts one Agent proposal")
        self._called_request_ids.add(context.request_id)
        projected = self._projected_input(context)
        if projected.proposal_fingerprint != tool_invocation_fingerprint(self._draft):
            raise RuntimeError("ToolIntent does not match the isolated Agent context")
        rationale = (
            "Reasoner proposed a Runtime-governed Tool invocation with call key "
            f"'{self._draft.call_key}'."
        )
        return AgentCallResult(
            proposal=DecisionProposal[ToolInvocationProposalDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._producer_id,
                    capability="tool_invocation_proposal",
                    implementation_version="phase-2d-tool-invocation",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                revision=0,
                selected_action=TOOL_INVOCATION_OPERATION,
                payload=self._draft,
                rationale=rationale,
                confidence=0.5,
            ),
        )

    @staticmethod
    def _projected_input(context: AgentContext) -> ToolInvocationAgentInput:
        blocks = tuple(
            block
            for block in context.blocks
            if block.source_type == TOOL_INVOCATION_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError(
                "Tool Invocation Agent context requires one tool_invocation_input block"
            )
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("tool_invocation_input content must be an object")
        return ToolInvocationAgentInput.model_validate(dict(content))


class ToolInvocationEffectNormalizer:
    """Validate an Agent intent and produce the final Runtime invocation."""

    module_id = "llm.adapter.tool_invocation.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[ToolInvocationDecisionPayload],
        proposal: DecisionProposal[ToolInvocationProposalDraft],
    ) -> NormalizedDecisionEffect[ToolInvocationEffect]:
        if request.decision_type != TOOL_INVOCATION_DECISION_TYPE:
            raise ValueError("Tool Invocation normalizer received another decision type")
        payload = ToolInvocationDecisionPayload.model_validate(
            request.payload.model_dump(mode="python")
        )
        draft = proposal.payload
        if tool_invocation_fingerprint(draft) != payload.proposal_fingerprint:
            raise ValueError("Tool Invocation proposal changed after Runtime capture")
        invocation = payload.invocation
        if draft.capability_id != invocation.capability_id:
            raise ValueError("Tool Invocation proposal changed the capability")
        if draft.arguments != invocation.arguments:
            raise ValueError("Tool Invocation proposal changed Runtime arguments")
        if request.target.target_type != "tool_provider":
            raise ValueError("Tool Invocation target must be a Tool Provider")
        if request.target.target_id != invocation.provider_id:
            raise ValueError("Tool Invocation target does not match selected Provider")
        effect = ToolInvocationEffect(
            invocation=invocation,
            agent_call_key=draft.call_key,
            proposal_fingerprint=payload.proposal_fingerprint,
            provider_metadata_fingerprint=payload.provider_metadata_fingerprint,
            candidate_set_fingerprint=payload.candidate_set_fingerprint,
            selection_fingerprint=payload.selection_fingerprint,
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        privileged = payload.privileged
        return NormalizedDecisionEffect[ToolInvocationEffect].create(
            payload=effect,
            operation=TOOL_INVOCATION_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.ACTION,
            risk=(
                DecisionRiskLevel.HIGH if privileged else DecisionRiskLevel.LOW
            ),
            impact_score=0.7 if privileged else 0.1,
            reversible=not privileged,
            impact_description=(
                "Invoke the Runtime-selected Tool Provider using schema-validated "
                "arguments from an Agent proposal."
            ),
        )
