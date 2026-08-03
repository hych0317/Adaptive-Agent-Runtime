"""Typed cognitive capability inputs and authority-free result drafts."""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import Field, model_validator

from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    ImmutableJsonValue,
    LLMModel,
)
from adaptive_agent_runtime.llm.models import InferenceCorrelation, ToolIntentDraft


CapabilityResultT = TypeVar("CapabilityResultT")


class CapabilityInvocationMetadata(LLMModel):
    """Runtime-owned call identity kept outside cognitive payloads."""

    correlation: InferenceCorrelation = Field(
        default_factory=InferenceCorrelation
    )
    trace_attributes: ImmutableJsonObject = Field(default_factory=dict)


class CapabilityTurnKind(StrEnum):
    COMPLETED = "completed"
    TOOL_INTENT = "tool_intent"


class CapabilityTurnResult(LLMModel, Generic[CapabilityResultT]):
    """One Runtime-driven cognitive turn: a result or executable proposals."""

    kind: CapabilityTurnKind
    result: CapabilityResultT | None = None
    tool_intents: tuple[ToolIntentDraft, ...] = ()

    @model_validator(mode="after")
    def validate_turn(self) -> CapabilityTurnResult[CapabilityResultT]:
        if self.kind is CapabilityTurnKind.COMPLETED:
            if self.result is None:
                raise ValueError("a completed capability turn requires a result")
            if self.tool_intents:
                raise ValueError(
                    "a completed capability turn cannot contain tool intents"
                )
        else:
            if self.result is not None:
                raise ValueError("a tool-intent turn cannot contain a result")
            if not self.tool_intents:
                raise ValueError("a tool-intent turn requires tool intents")
        call_keys = tuple(intent.call_key for intent in self.tool_intents)
        if len(set(call_keys)) != len(call_keys):
            raise ValueError("capability tool intent call keys must be unique")
        return self


class EvidenceReference(LLMModel):
    reference_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    summary: str | None = Field(default=None, min_length=1)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)


class ReasoningContext(LLMModel):
    goal: str = Field(min_length=1)
    context: ImmutableJsonValue = None
    evidence: tuple[EvidenceReference, ...] = ()
    constraints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_context(self) -> ReasoningContext:
        _ensure_unique_references(self.evidence)
        _ensure_unique_strings(self.constraints, "reasoning constraints")
        return self


class ReasoningResult(LLMModel):
    conclusions: tuple[str, ...] = Field(min_length=1)
    assumptions: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    rejected_alternatives: tuple[str, ...] = ()
    decision_rationale: str | None = Field(default=None, min_length=1)
    evidence_reference_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> ReasoningResult:
        _ensure_unique_strings(self.conclusions, "reasoning conclusions")
        _ensure_unique_strings(self.assumptions, "reasoning assumptions")
        _ensure_unique_strings(self.uncertainties, "reasoning uncertainties")
        _ensure_unique_strings(
            self.rejected_alternatives,
            "rejected reasoning alternatives",
        )
        _ensure_unique_strings(
            self.evidence_reference_ids,
            "reasoning evidence references",
        )
        return self


class TaskPlanningRequest(LLMModel):
    task: str = Field(min_length=1)
    constraints: tuple[str, ...] = ()
    available_strategies: tuple[str, ...] = Field(min_length=1)
    available_execution_capability_ids: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> TaskPlanningRequest:
        _ensure_unique_strings(self.constraints, "planning constraints")
        _ensure_unique_strings(
            self.available_strategies,
            "available strategies",
        )
        _ensure_unique_strings(
            self.available_execution_capability_ids,
            "available execution capability identifiers",
        )
        _ensure_unique_references(self.evidence)
        return self


class TaskNodeDraft(LLMModel):
    """A symbolic node proposal; Runtime assigns the real node UUID."""

    node_key: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    dependency_keys: tuple[str, ...] = ()
    expected_output: str = Field(min_length=1)
    requested_strategy_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_node(self) -> TaskNodeDraft:
        _ensure_unique_strings(self.dependency_keys, "node dependencies")
        if self.node_key in self.dependency_keys:
            raise ValueError("a task node draft cannot depend on itself")
        return self


class TaskGraphDraft(LLMModel):
    nodes: tuple[TaskNodeDraft, ...] = Field(min_length=1)
    rationale: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> TaskGraphDraft:
        node_keys = tuple(node.node_key for node in self.nodes)
        if len(set(node_keys)) != len(node_keys):
            raise ValueError("task graph draft node keys must be unique")
        known = set(node_keys)
        unknown = {
            dependency
            for node in self.nodes
            for dependency in node.dependency_keys
            if dependency not in known
        }
        if unknown:
            raise ValueError(
                "task graph draft references unknown dependencies: "
                + ", ".join(sorted(unknown))
            )
        remaining: dict[str, set[str]] = {
            node.node_key: set(node.dependency_keys) for node in self.nodes
        }
        while remaining:
            ready = tuple(
                key for key, dependencies in remaining.items() if not dependencies
            )
            if not ready:
                raise ValueError("task graph draft must be acyclic")
            for key in ready:
                del remaining[key]
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return self


class PlanningActionCandidate(LLMModel):
    """One Runtime-approved ready node that a planner may select."""

    node_key: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)


class ActionProposalRequest(LLMModel):
    task: str = Field(min_length=1)
    candidates: tuple[PlanningActionCandidate, ...] = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_request(self) -> ActionProposalRequest:
        _ensure_unique_strings(
            tuple(candidate.node_key for candidate in self.candidates),
            "action candidate keys",
        )
        _ensure_unique_references(self.evidence)
        return self


class ActionProposalDraft(LLMModel):
    """A ready-node selection proposal; it cannot dispatch the node."""

    node_key: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_draft(self) -> ActionProposalDraft:
        _ensure_unique_strings(
            self.evidence_reference_ids,
            "action proposal evidence references",
        )
        return self


class GraphMutationOperationKind(StrEnum):
    ADD_NODE = "add_node"
    ADD_DEPENDENCY = "add_dependency"


class GraphMutationOperationDraft(LLMModel):
    """One symbolic graph change with no Runtime UUID or apply authority."""

    kind: GraphMutationOperationKind
    node: TaskNodeDraft | None = None
    node_key: str | None = Field(default=None, min_length=1)
    dependency_key: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operation(self) -> GraphMutationOperationDraft:
        if self.kind is GraphMutationOperationKind.ADD_NODE:
            if self.node is None:
                raise ValueError("add_node mutation proposal requires a node")
            if self.node_key is not None or self.dependency_key is not None:
                raise ValueError(
                    "add_node mutation proposal cannot contain dependency fields"
                )
        else:
            if self.node is not None:
                raise ValueError(
                    "add_dependency mutation proposal cannot contain a node"
                )
            if self.node_key is None or self.dependency_key is None:
                raise ValueError(
                    "add_dependency mutation proposal requires both node keys"
                )
            if self.node_key == self.dependency_key:
                raise ValueError(
                    "graph mutation proposal cannot add a self dependency"
                )
        _ensure_unique_strings(
            self.evidence_reference_ids,
            "graph mutation evidence references",
        )
        return self


class GraphMutationProposalRequest(LLMModel):
    task: str = Field(min_length=1)
    trigger_node_key: str = Field(min_length=1)
    trigger_observation: ImmutableJsonValue
    existing_node_keys: tuple[str, ...] = Field(min_length=1)
    allowed_new_node_keys: tuple[str, ...] = ()
    available_strategies: tuple[str, ...] = Field(min_length=1)
    available_execution_capability_ids: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_request(self) -> GraphMutationProposalRequest:
        _ensure_unique_strings(
            self.existing_node_keys,
            "existing graph node keys",
        )
        _ensure_unique_strings(
            self.allowed_new_node_keys,
            "allowed new graph node keys",
        )
        _ensure_unique_strings(
            self.available_strategies,
            "graph mutation strategies",
        )
        _ensure_unique_strings(
            self.available_execution_capability_ids,
            "graph mutation execution capabilities",
        )
        if set(self.existing_node_keys).intersection(self.allowed_new_node_keys):
            raise ValueError(
                "existing and allowed new graph node keys must be disjoint"
            )
        if self.trigger_node_key not in self.existing_node_keys:
            raise ValueError("graph mutation trigger must be an existing node")
        _ensure_unique_references(self.evidence)
        return self


class GraphMutationProposalDraft(LLMModel):
    """A batch of symbolic changes that Runtime must validate and govern."""

    operations: tuple[GraphMutationOperationDraft, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class GenerationRequest(LLMModel):
    instruction: str = Field(min_length=1)
    context: ImmutableJsonValue = None
    media_type: str = Field(default="text/plain", min_length=1)
    output_schema: ImmutableJsonObject | None = None
    evidence: tuple[EvidenceReference, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> GenerationRequest:
        _ensure_unique_references(self.evidence)
        return self


class GeneratedArtifactDraft(LLMModel):
    """Generated content only; storage and publication remain external effects."""

    media_type: str = Field(min_length=1)
    content: ImmutableJsonValue
    evidence_reference_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_artifact(self) -> GeneratedArtifactDraft:
        if self.content is None:
            raise ValueError("a generated artifact requires content")
        _ensure_unique_strings(
            self.evidence_reference_ids,
            "artifact evidence references",
        )
        _ensure_unique_strings(self.warnings, "artifact warnings")
        return self


class JudgeSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class JudgeRequest(LLMModel):
    subject: ImmutableJsonValue
    criteria: tuple[str, ...] = Field(min_length=1)
    evidence_catalog: tuple[EvidenceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_request(self) -> JudgeRequest:
        _ensure_unique_strings(self.criteria, "judge criteria")
        _ensure_unique_references(self.evidence_catalog)
        return self


class JudgeFindingDraft(LLMModel):
    """A semantic finding; Runtime supplies stable identity and timestamps."""

    code: str = Field(min_length=1)
    severity: JudgeSeverity
    summary: str = Field(min_length=1)
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)
    details: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_finding(self) -> JudgeFindingDraft:
        _ensure_unique_strings(
            self.evidence_reference_ids,
            "judge finding evidence references",
        )
        return self


class JudgeAssessmentDraft(LLMModel):
    summary: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    findings: tuple[JudgeFindingDraft, ...] = ()


class CompressionRequest(LLMModel):
    """Request semantic compression of one Context Unit, not an assembly."""

    source_reference_id: str = Field(min_length=1)
    content: ImmutableJsonValue
    original_estimated_tokens: int = Field(ge=1)
    target_max_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_budget(self) -> CompressionRequest:
        if self.target_max_tokens >= self.original_estimated_tokens:
            raise ValueError(
                "compression target must be smaller than the original estimate"
            )
        return self


class CompressedContextDraft(LLMModel):
    content: ImmutableJsonValue
    core_conclusions: tuple[str, ...] = Field(min_length=1)
    source_reference_ids: tuple[str, ...] = Field(min_length=1)
    estimated_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_draft(self) -> CompressedContextDraft:
        _ensure_unique_strings(
            self.core_conclusions,
            "compressed context conclusions",
        )
        _ensure_unique_strings(
            self.source_reference_ids,
            "compressed context source references",
        )
        return self


class MemoryEvolutionDraft(StrEnum):
    SUPPORT = "support"
    MODIFY = "modify"
    EXTEND = "extend"
    CONFLICT = "conflict"


class MemoryConditionDraft(LLMModel):
    facts: ImmutableJsonObject = Field(default_factory=dict)
    required_tags: tuple[str, ...] = ()
    description: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_condition(self) -> MemoryConditionDraft:
        _ensure_unique_strings(self.required_tags, "memory condition tags")
        return self


class MemoryCandidateDraft(LLMModel):
    memory_key: str = Field(min_length=1)
    content: ImmutableJsonValue
    condition: MemoryConditionDraft
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(gt=0.0, le=1.0)
    evolution: MemoryEvolutionDraft
    target_memory_reference: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_candidate(self) -> MemoryCandidateDraft:
        _ensure_unique_strings(
            self.evidence_reference_ids,
            "memory candidate evidence references",
        )
        if self.evolution is MemoryEvolutionDraft.EXTEND:
            if self.target_memory_reference is not None:
                raise ValueError(
                    "extend memory drafts cannot target existing memory"
                )
        elif self.target_memory_reference is None:
            raise ValueError(
                f"{self.evolution.value} memory drafts require a target reference"
            )
        return self


class MemoryCandidateBatchDraft(LLMModel):
    candidates: tuple[MemoryCandidateDraft, ...] = ()


class MemoryExtractionRequest(LLMModel):
    observations: ImmutableJsonValue
    evidence_catalog: tuple[EvidenceReference, ...] = Field(min_length=1)
    existing_memories: ImmutableJsonValue = None
    existing_memory_reference_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> MemoryExtractionRequest:
        _ensure_unique_references(self.evidence_catalog)
        _ensure_unique_strings(
            self.existing_memory_reference_ids,
            "existing memory references",
        )
        return self


def _ensure_unique_references(references: tuple[EvidenceReference, ...]) -> None:
    _ensure_unique_strings(
        tuple(item.reference_id for item in references),
        "evidence references",
    )


def _ensure_unique_strings(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    if any(not value for value in values):
        raise ValueError(f"{name} cannot contain empty values")
