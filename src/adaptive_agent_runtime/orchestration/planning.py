"""Runtime-owned contracts for one governed initial planning decision."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import (
    OrchestrationModel,
    TaskNode,
)


GRAPH_INITIALIZE_OPERATION = "graph.initialize"
PLANNING_DECISION_TYPE = "planning.task_graph.initialize"


class PlanningStrategyRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlanningRetryPolicy(OrchestrationModel):
    """Phase 1-A deliberately permits no repeated graph generation."""

    max_retries: int = Field(default=0, ge=0, le=0)


class PlanningExecutionPolicy(OrchestrationModel):
    """Runtime-owned controls around the single Planner Agent call."""

    model: str = Field(min_length=1)
    temperature: float = Field(default=0.1, ge=0.0, le=0.3)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    retry_policy: PlanningRetryPolicy = Field(default_factory=PlanningRetryPolicy)


class PlanningGraphLimits(OrchestrationModel):
    max_nodes: int = Field(default=32, ge=1)
    max_depth: int = Field(default=12, ge=1)
    max_fan_out: int = Field(default=12, ge=1)


class PlanningStrategyDescriptor(OrchestrationModel):
    strategy_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required_capability_ids: tuple[str, ...] = ()
    risk: PlanningStrategyRisk = PlanningStrategyRisk.MEDIUM

    @model_validator(mode="after")
    def validate_capabilities(self) -> PlanningStrategyDescriptor:
        if len(set(self.required_capability_ids)) != len(
            self.required_capability_ids
        ):
            raise ValueError("strategy capability identifiers must be unique")
        return self


class PlanningDecisionPayload(OrchestrationModel):
    """Runtime input retained in DecisionRequest, not exposed wholesale to Agent."""

    goal: str = Field(min_length=1)
    constraints: tuple[str, ...] = ()
    strategies: tuple[PlanningStrategyDescriptor, ...] = Field(min_length=1)
    available_execution_capability_ids: tuple[str, ...] = ()
    graph_limits: PlanningGraphLimits = Field(default_factory=PlanningGraphLimits)
    execution_policy: PlanningExecutionPolicy

    @model_validator(mode="after")
    def validate_catalog(self) -> PlanningDecisionPayload:
        strategy_ids = tuple(item.strategy_id for item in self.strategies)
        if len(set(strategy_ids)) != len(strategy_ids):
            raise ValueError("planning strategy identifiers must be unique")
        if len(set(self.constraints)) != len(self.constraints):
            raise ValueError("planning constraints must be unique")
        if len(set(self.available_execution_capability_ids)) != len(
            self.available_execution_capability_ids
        ):
            raise ValueError("planning capability identifiers must be unique")
        return self


class SymbolicTaskNode(OrchestrationModel):
    node_key: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    dependency_keys: tuple[str, ...] = ()
    expected_output: str = Field(min_length=1)
    requested_strategy_id: str = Field(min_length=1)


class SymbolicTaskGraph(OrchestrationModel):
    nodes: tuple[SymbolicTaskNode, ...] = Field(min_length=1)


class GraphValidationReceipt(OrchestrationModel):
    source_draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_count: int = Field(ge=1)
    root_count: int = Field(ge=1)
    terminal_count: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    max_fan_out: int = Field(ge=0)
    strategy_ids: tuple[str, ...] = Field(min_length=1)
    checks: tuple[str, ...] = Field(min_length=1)


class ValidatedSymbolicTaskGraph(OrchestrationModel):
    graph: SymbolicTaskGraph
    receipt: GraphValidationReceipt


class PlanningNodeBinding(OrchestrationModel):
    node_key: str = Field(min_length=1)
    node_id: UUID


class PlanningRiskAssessment(OrchestrationModel):
    risk: PlanningStrategyRisk
    impact_score: float = Field(ge=0.0, le=1.0)
    reversible: bool
    description: str = Field(min_length=1)


class PlanningGraphEffect(OrchestrationModel):
    """Final graph initialization effect created exclusively by Runtime."""

    run_id: UUID
    task_id: UUID
    graph: DynamicTaskGraph
    node_bindings: tuple[PlanningNodeBinding, ...] = Field(min_length=1)
    source_draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation: GraphValidationReceipt
    risk_assessment: PlanningRiskAssessment

    @model_validator(mode="after")
    def validate_bindings(self) -> PlanningGraphEffect:
        keys = tuple(item.node_key for item in self.node_bindings)
        ids = tuple(item.node_id for item in self.node_bindings)
        if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
            raise ValueError("planning graph bindings must be one-to-one")
        if set(ids) != {node.node_id for node in self.graph.nodes}:
            raise ValueError("planning graph bindings must cover every graph node")
        if self.source_draft_fingerprint != self.validation.source_draft_fingerprint:
            raise ValueError("planning effect and validation draft fingerprints differ")
        return self


class PlanningGraphValidator:
    """Validate an untrusted symbolic graph against current Runtime policy."""

    module_id = "orchestration.planning.graph_validator"

    def validate(
        self,
        payload: PlanningDecisionPayload,
        graph: SymbolicTaskGraph,
        *,
        source_draft_fingerprint: str,
    ) -> ValidatedSymbolicTaskGraph:
        nodes_by_key = {node.node_key: node for node in graph.nodes}
        if len(nodes_by_key) != len(graph.nodes):
            raise ValueError("planning graph node keys must be unique")
        if len(graph.nodes) > payload.graph_limits.max_nodes:
            raise ValueError("planning graph exceeds the Runtime node budget")

        known = set(nodes_by_key)
        for node in graph.nodes:
            if len(set(node.dependency_keys)) != len(node.dependency_keys):
                raise ValueError(f"node '{node.node_key}' has duplicate dependencies")
            if node.node_key in node.dependency_keys:
                raise ValueError(f"node '{node.node_key}' depends on itself")
            missing = set(node.dependency_keys) - known
            if missing:
                raise ValueError(
                    f"node '{node.node_key}' has unknown dependencies: "
                    + ", ".join(sorted(missing))
                )

        ordered_keys, depth_by_key = self._canonical_order(nodes_by_key)
        max_depth = max(depth_by_key.values())
        if max_depth > payload.graph_limits.max_depth:
            raise ValueError("planning graph exceeds the Runtime depth budget")

        child_counts = {key: 0 for key in known}
        for node in graph.nodes:
            for dependency in node.dependency_keys:
                child_counts[dependency] += 1
        max_fan_out = max(child_counts.values(), default=0)
        if max_fan_out > payload.graph_limits.max_fan_out:
            raise ValueError("planning graph exceeds the Runtime fan-out budget")

        strategies = {item.strategy_id: item for item in payload.strategies}
        available_capabilities = set(payload.available_execution_capability_ids)
        used_strategy_ids: set[str] = set()
        for node in graph.nodes:
            descriptor = strategies.get(node.requested_strategy_id)
            if descriptor is None:
                raise ValueError(
                    f"node '{node.node_key}' requests an unavailable strategy"
                )
            unavailable = (
                set(descriptor.required_capability_ids) - available_capabilities
            )
            if unavailable:
                raise ValueError(
                    f"strategy '{descriptor.strategy_id}' requires unavailable "
                    f"capabilities: {', '.join(sorted(unavailable))}"
                )
            used_strategy_ids.add(descriptor.strategy_id)

        depended_on = {
            dependency
            for node in graph.nodes
            for dependency in node.dependency_keys
        }
        roots = tuple(node for node in graph.nodes if not node.dependency_keys)
        terminals = tuple(node for node in graph.nodes if node.node_key not in depended_on)
        if not roots or not terminals:
            raise ValueError("planning graph requires roots and terminal nodes")

        canonical = SymbolicTaskGraph(
            nodes=tuple(nodes_by_key[key] for key in ordered_keys)
        )
        receipt = GraphValidationReceipt(
            source_draft_fingerprint=source_draft_fingerprint,
            node_count=len(graph.nodes),
            root_count=len(roots),
            terminal_count=len(terminals),
            max_depth=max_depth,
            max_fan_out=max_fan_out,
            strategy_ids=tuple(sorted(used_strategy_ids)),
            checks=(
                "symbolic DAG is acyclic",
                "graph complexity is within Runtime limits",
                "strategies and capabilities are currently available",
                "graph has Runtime-valid roots and terminals",
            ),
        )
        return ValidatedSymbolicTaskGraph(graph=canonical, receipt=receipt)

    @staticmethod
    def _canonical_order(
        nodes_by_key: dict[str, SymbolicTaskNode],
    ) -> tuple[tuple[str, ...], dict[str, int]]:
        remaining = {
            key: set(node.dependency_keys) for key, node in nodes_by_key.items()
        }
        ordered: list[str] = []
        depth_by_key: dict[str, int] = {}
        while remaining:
            ready = tuple(
                sorted(key for key, dependencies in remaining.items() if not dependencies)
            )
            if not ready:
                raise ValueError("planning graph must be acyclic")
            for key in ready:
                node = nodes_by_key[key]
                depth_by_key[key] = (
                    1
                    if not node.dependency_keys
                    else 1 + max(depth_by_key[item] for item in node.dependency_keys)
                )
                ordered.append(key)
                del remaining[key]
            for dependencies in remaining.values():
                dependencies.difference_update(ready)
        return tuple(ordered), depth_by_key


class PlanningGraphEffectProjector:
    """Assign Runtime identity and construct the immutable graph effect."""

    module_id = "orchestration.planning.effect_projector"

    def project(
        self,
        payload: PlanningDecisionPayload,
        validated: ValidatedSymbolicTaskGraph,
        *,
        request_id: UUID,
        run_id: UUID,
        task_id: UUID,
        basis_fingerprint: str,
    ) -> PlanningGraphEffect:
        graph_id = uuid5(request_id, "initial-graph")
        node_ids = {
            node.node_key: uuid5(graph_id, node.node_key)
            for node in validated.graph.nodes
        }
        nodes = tuple(
            TaskNode(
                node_id=node_ids[node.node_key],
                goal=node.goal,
                dependencies=tuple(
                    node_ids[key] for key in sorted(node.dependency_keys)
                ),
                expected_output=node.expected_output,
                strategy_id=node.requested_strategy_id,
            )
            for node in validated.graph.nodes
        )
        graph = DynamicTaskGraph(graph_id=graph_id, version=0, nodes=nodes)
        risk = self._risk_assessment(payload, validated)
        return PlanningGraphEffect(
            run_id=run_id,
            task_id=task_id,
            graph=graph,
            node_bindings=tuple(
                PlanningNodeBinding(node_key=key, node_id=node_ids[key])
                for key in sorted(node_ids)
            ),
            source_draft_fingerprint=validated.receipt.source_draft_fingerprint,
            basis_fingerprint=basis_fingerprint,
            validation=validated.receipt,
            risk_assessment=risk,
        )

    @staticmethod
    def _risk_assessment(
        payload: PlanningDecisionPayload,
        validated: ValidatedSymbolicTaskGraph,
    ) -> PlanningRiskAssessment:
        descriptors = {item.strategy_id: item for item in payload.strategies}
        risks = {
            descriptors[node.requested_strategy_id].risk
            for node in validated.graph.nodes
        }
        if PlanningStrategyRisk.HIGH in risks:
            risk = PlanningStrategyRisk.HIGH
            risk_weight = 0.4
        elif len(validated.graph.nodes) > 1 or PlanningStrategyRisk.MEDIUM in risks:
            risk = PlanningStrategyRisk.MEDIUM
            risk_weight = 0.2
        else:
            risk = PlanningStrategyRisk.LOW
            risk_weight = 0.05
        size_ratio = min(
            1.0,
            len(validated.graph.nodes) / payload.graph_limits.max_nodes,
        )
        impact = round(min(1.0, 0.2 + (size_ratio * 0.4) + risk_weight), 6)
        return PlanningRiskAssessment(
            risk=risk,
            impact_score=impact,
            reversible=True,
            description=(
                "Runtime-computed impact of initializing this run's validated "
                f"{len(validated.graph.nodes)}-node task graph."
            ),
        )
