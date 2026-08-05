"""Runtime-owned graph mutation Decision contracts and effects."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import Field, JsonValue, model_validator

from adaptive_agent_runtime import AgentState, Observation, RuntimeModule
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import GraphMutation, OrchestrationModel


GRAPH_MUTATION_DECISION_TYPE = "orchestration.graph_mutation"
GRAPH_MUTATION_APPLY_OPERATION = "graph.mutate"


class GraphMutationExecutionPolicy(OrchestrationModel):
    timeout_seconds: float = Field(default=20.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_revision_count: int = Field(default=0, ge=0, le=0)


class GraphMutationDecisionPayload(OrchestrationModel):
    """Runtime snapshot; only ``agent_input`` is projected to the Agent."""

    graph: DynamicTaskGraph
    source_node_id: UUID
    source_action_id: UUID
    source_observation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_input: JsonValue
    execution_policy: GraphMutationExecutionPolicy = Field(
        default_factory=GraphMutationExecutionPolicy
    )

    @model_validator(mode="after")
    def validate_source(self) -> GraphMutationDecisionPayload:
        self.graph.get_node(self.source_node_id)
        return self


class GraphMutationEffect(OrchestrationModel):
    """Normalized, fingerprint-bound batch that Governance and Apply share."""

    graph_id: UUID
    graph_version_before: int = Field(ge=0)
    graph_version_after: int = Field(ge=1)
    source_node_id: UUID
    source_action_id: UUID
    mutations: tuple[GraphMutation, ...] = Field(min_length=1)
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_versions(self) -> GraphMutationEffect:
        if self.graph_version_after <= self.graph_version_before:
            raise ValueError("graph mutation effect must advance graph version")
        return self


class GraphMutationDecisionOutcome(OrchestrationModel):
    request_id: UUID
    effect: GraphMutationEffect
    graph: DynamicTaskGraph

    @model_validator(mode="after")
    def validate_graph(self) -> GraphMutationDecisionOutcome:
        if self.graph.graph_id != self.effect.graph_id:
            raise ValueError("graph mutation outcome changed graph identity")
        if self.graph.version != self.effect.graph_version_after:
            raise ValueError("graph mutation outcome version differs from effect")
        return self


@runtime_checkable
class GraphMutationDecisionHandler(RuntimeModule, Protocol):
    """Optional domain Decision seam invoked after a successful Observation."""

    async def handle(
        self,
        *,
        graph: DynamicTaskGraph,
        state: AgentState,
        source_node_id: UUID,
        observation: Observation,
    ) -> GraphMutationDecisionOutcome | None: ...


def apply_graph_mutation_effect(
    graph: DynamicTaskGraph,
    effect: GraphMutationEffect,
) -> DynamicTaskGraph:
    """Apply the exact normalized batch while checking its Runtime basis."""

    if graph.graph_id != effect.graph_id:
        raise ValueError("graph mutation effect targets another graph")
    if graph.version != effect.graph_version_before:
        raise ValueError("graph mutation effect is stale")
    updated = graph
    for mutation in effect.mutations:
        updated = updated.apply_mutation(mutation)
    if updated.version != effect.graph_version_after:
        raise ValueError("applied graph version differs from normalized effect")
    return updated
