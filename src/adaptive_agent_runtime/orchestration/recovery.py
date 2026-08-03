"""Failure analysis and bounded recovery-plan proposals."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime import AgentState, Observation, RuntimeModule
from adaptive_agent_runtime.core.models import utc_now
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import (
    OrchestrationModel,
    TaskNode,
)


_RECOVERY_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/orchestration/recovery",
)


def stable_recovery_id(*parts: object) -> UUID:
    return uuid5(_RECOVERY_NAMESPACE, "|".join(str(item) for item in parts))


class FailureKind(StrEnum):
    TRANSIENT = "transient"
    STRATEGY_UNAVAILABLE = "strategy_unavailable"
    INVALID_INPUT = "invalid_input"
    POLICY_DENIED = "policy_denied"
    NON_RECOVERABLE = "non_recoverable"
    UNKNOWN = "unknown"


class RecoveryActionType(StrEnum):
    RETRY_NODE = "retry_node"
    REPLACE_STRATEGY = "replace_strategy"
    ADD_RECOVERY_NODE = "add_recovery_node"
    REWIRE_DEPENDENCY = "rewire_dependency"
    ABORT = "abort"


class FailureAnalysis(OrchestrationModel):
    analysis_id: UUID = Field(default_factory=uuid4)
    node_id: UUID
    action_id: UUID
    kind: FailureKind
    retryable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = Field(min_length=1)


class RecoveryAction(OrchestrationModel):
    action_type: RecoveryActionType
    node_id: UUID
    replacement_strategy_id: str | None = None
    recovery_node: TaskNode | None = None
    dependent_node_id: UUID | None = None
    old_dependency_id: UUID | None = None
    new_dependency_id: UUID | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> RecoveryAction:
        if self.action_type is RecoveryActionType.REPLACE_STRATEGY:
            if not self.replacement_strategy_id:
                raise ValueError("replace_strategy requires a strategy id")
        elif self.replacement_strategy_id is not None:
            raise ValueError("only replace_strategy accepts a strategy id")
        if self.action_type is RecoveryActionType.ADD_RECOVERY_NODE:
            if self.recovery_node is None:
                raise ValueError("add_recovery_node requires a TaskNode")
        elif self.recovery_node is not None:
            raise ValueError("only add_recovery_node accepts a TaskNode")
        dependency_fields = (
            self.old_dependency_id,
            self.new_dependency_id,
        )
        if self.action_type is RecoveryActionType.REWIRE_DEPENDENCY:
            if self.dependent_node_id is None or any(
                item is None for item in dependency_fields
            ):
                raise ValueError(
                    "rewire_dependency requires dependent, old, and new ids"
                )
        elif self.dependent_node_id is not None or any(
            item is not None for item in dependency_fields
        ):
            raise ValueError("only rewire_dependency accepts dependency ids")
        return self


class RecoveryPlan(OrchestrationModel):
    plan_id: UUID = Field(default_factory=uuid4)
    analysis: FailureAnalysis
    actions: tuple[RecoveryAction, ...] = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    max_attempts: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_plan(self) -> RecoveryPlan:
        if self.attempt_number > self.max_attempts:
            raise ValueError("recovery attempt exceeds its budget")
        aborts = tuple(
            item for item in self.actions
            if item.action_type is RecoveryActionType.ABORT
        )
        if aborts and len(self.actions) != 1:
            raise ValueError("abort must be the only recovery action")
        if any(item.node_id != self.analysis.node_id for item in self.actions):
            raise ValueError("recovery actions must target the analyzed failure")
        return self

    @property
    def aborts(self) -> bool:
        return self.actions[0].action_type is RecoveryActionType.ABORT


class RecoveryContext(OrchestrationModel):
    graph: DynamicTaskGraph
    state: AgentState
    failed_node: TaskNode
    observation: Observation
    prior_attempts: int = Field(ge=0)


class RecoveryRecord(OrchestrationModel):
    plan: RecoveryPlan
    graph_version_before: int = Field(ge=0)
    graph_version_after: int = Field(ge=0)
    recorded_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_versions(self) -> RecoveryRecord:
        if self.plan.aborts:
            if self.graph_version_after != self.graph_version_before:
                raise ValueError("abort cannot change graph version")
        elif self.graph_version_after <= self.graph_version_before:
            raise ValueError("applied recovery must advance graph version")
        return self


@runtime_checkable
class FailureClassifier(RuntimeModule, Protocol):
    def classify(self, context: RecoveryContext) -> FailureAnalysis: ...


@runtime_checkable
class FailureDrivenReplanner(RuntimeModule, Protocol):
    async def replan(self, context: RecoveryContext) -> RecoveryPlan: ...


@runtime_checkable
class RecoveryPlanApplier(RuntimeModule, Protocol):
    async def apply(
        self,
        graph: DynamicTaskGraph,
        plan: RecoveryPlan,
        *,
        state: AgentState,
    ) -> DynamicTaskGraph: ...


class DeterministicFailureClassifier:
    module_id = "orchestration.failure_classifier.deterministic"

    _TRANSIENT_MARKERS = (
        "timeout",
        "timed out",
        "temporar",
        "rate limit",
        "connection reset",
    )

    def classify(self, context: RecoveryContext) -> FailureAnalysis:
        error = context.observation.error or "unknown failure"
        normalized = error.lower()
        metadata = context.observation.metadata
        explicit = metadata.get("failure_kind")
        if isinstance(explicit, str):
            try:
                kind = FailureKind(explicit)
            except ValueError:
                kind = FailureKind.UNKNOWN
        elif "strategy" in normalized and "unavailable" in normalized:
            kind = FailureKind.STRATEGY_UNAVAILABLE
        elif any(marker in normalized for marker in self._TRANSIENT_MARKERS):
            kind = FailureKind.TRANSIENT
        elif "governance denied" in normalized or "policy denied" in normalized:
            kind = FailureKind.POLICY_DENIED
        elif "invalid" in normalized or "validation" in normalized:
            kind = FailureKind.INVALID_INPUT
        elif "non-recoverable" in normalized or "fatal" in normalized:
            kind = FailureKind.NON_RECOVERABLE
        else:
            kind = FailureKind.UNKNOWN
        retryable = kind in {
            FailureKind.TRANSIENT,
            FailureKind.STRATEGY_UNAVAILABLE,
        }
        return FailureAnalysis(
            analysis_id=stable_recovery_id(
                "failure-analysis",
                context.state.run_id,
                context.failed_node.node_id,
                context.observation.action_id,
                kind.value,
                error,
            ),
            node_id=context.failed_node.node_id,
            action_id=context.observation.action_id,
            kind=kind,
            retryable=retryable,
            confidence=0.95 if kind is not FailureKind.UNKNOWN else 0.5,
            evidence=(error,),
        )


class DeterministicFailureDrivenReplanner:
    """Propose bounded graph recovery; never applies or retries a Tool call."""

    module_id = "orchestration.replanner.failure_driven"

    def __init__(
        self,
        *,
        classifier: FailureClassifier | None = None,
        max_attempts_per_node: int = 2,
        alternate_strategies: Mapping[str, str] | None = None,
    ) -> None:
        if max_attempts_per_node < 1:
            raise ValueError("max_attempts_per_node must be at least 1")
        self._classifier = classifier or DeterministicFailureClassifier()
        self._max_attempts = max_attempts_per_node
        self._alternate_strategies = dict(alternate_strategies or {})

    async def replan(self, context: RecoveryContext) -> RecoveryPlan:
        analysis = self._classifier.classify(context)
        attempt = context.prior_attempts + 1
        if attempt > self._max_attempts or not analysis.retryable:
            action = RecoveryAction(
                action_type=RecoveryActionType.ABORT,
                node_id=context.failed_node.node_id,
                reason=(
                    "Recovery budget exhausted."
                    if attempt > self._max_attempts
                    else f"Failure kind '{analysis.kind.value}' is not recoverable."
                ),
            )
            return RecoveryPlan(
                plan_id=stable_recovery_id(
                    "recovery-plan",
                    analysis.analysis_id,
                    min(attempt, self._max_attempts),
                    RecoveryActionType.ABORT.value,
                ),
                analysis=analysis,
                actions=(action,),
                attempt_number=min(attempt, self._max_attempts),
                max_attempts=self._max_attempts,
            )
        alternate = self._alternate_strategies.get(
            context.failed_node.strategy_id
        )
        if analysis.kind is FailureKind.STRATEGY_UNAVAILABLE and alternate:
            action = RecoveryAction(
                action_type=RecoveryActionType.REPLACE_STRATEGY,
                node_id=context.failed_node.node_id,
                replacement_strategy_id=alternate,
                reason="Use the configured alternate execution strategy.",
            )
        else:
            action = RecoveryAction(
                action_type=RecoveryActionType.RETRY_NODE,
                node_id=context.failed_node.node_id,
                reason="Retry the node after a classified transient failure.",
            )
        return RecoveryPlan(
            plan_id=stable_recovery_id(
                "recovery-plan",
                analysis.analysis_id,
                attempt,
                action.action_type.value,
                action.replacement_strategy_id,
            ),
            analysis=analysis,
            actions=(action,),
            attempt_number=attempt,
            max_attempts=self._max_attempts,
        )


def apply_recovery_plan(
    graph: DynamicTaskGraph,
    plan: RecoveryPlan,
) -> DynamicTaskGraph:
    """Apply only validated graph transitions; ABORT returns the same graph."""

    current = graph
    for action in plan.actions:
        if action.action_type is RecoveryActionType.ABORT:
            return current
        if action.action_type is RecoveryActionType.RETRY_NODE:
            current = current.retry_failed(action.node_id)
        elif action.action_type is RecoveryActionType.REPLACE_STRATEGY:
            assert action.replacement_strategy_id is not None
            current = current.replace_failed_strategy(
                action.node_id,
                action.replacement_strategy_id,
            )
        elif action.action_type is RecoveryActionType.ADD_RECOVERY_NODE:
            assert action.recovery_node is not None
            current = current.add_recovery_node(
                action.node_id,
                action.recovery_node,
            )
        else:
            assert action.dependent_node_id is not None
            assert action.old_dependency_id is not None
            assert action.new_dependency_id is not None
            current = current.rewire_dependency(
                action.dependent_node_id,
                action.old_dependency_id,
                action.new_dependency_id,
            )
    return current
