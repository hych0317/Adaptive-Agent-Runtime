"""Runtime-owned models for governed ready-node selection decisions."""

from __future__ import annotations

import hashlib
import json
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator

from adaptive_agent_runtime.orchestration.models import OrchestrationModel, TaskNode


READY_NODE_SELECTION_DECISION_TYPE = "orchestration.ready_node_selection"
READY_NODE_SELECT_OPERATION = "node.select"

_SELECTION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/orchestration/ready-node-selection",
)


def stable_ready_node_selection_id(*parts: object) -> UUID:
    return uuid5(_SELECTION_NAMESPACE, "|".join(str(item) for item in parts))


def ready_node_selection_fingerprint(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReadyNodeSelectionExecutionPolicy(OrchestrationModel):
    timeout_seconds: float = Field(default=15.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_revision_count: int = Field(default=0, ge=0, le=0)


class ReadyNodeCandidateBinding(OrchestrationModel):
    """Runtime-only binding from a semantic key to one ready TaskNode."""

    node_key: str = Field(min_length=1)
    node: TaskNode
    node_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_node_fingerprint(self) -> ReadyNodeCandidateBinding:
        if self.node_fingerprint != ready_node_selection_fingerprint(self.node):
            raise ValueError("ready-node fingerprint does not match TaskNode")
        return self


class ReadyNodeSelectionDecisionPayload(OrchestrationModel):
    task_description: str = Field(min_length=1)
    candidates: tuple[ReadyNodeCandidateBinding, ...] = Field(min_length=1)
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_revision: int = Field(ge=0)
    execution_policy: ReadyNodeSelectionExecutionPolicy = Field(
        default_factory=ReadyNodeSelectionExecutionPolicy
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> ReadyNodeSelectionDecisionPayload:
        keys = tuple(item.node_key for item in self.candidates)
        node_ids = tuple(item.node.node_id for item in self.candidates)
        if len(set(keys)) != len(keys) or len(set(node_ids)) != len(node_ids):
            raise ValueError("ready-node candidate bindings must be one-to-one")
        if any(item.node.status.value != "pending" for item in self.candidates):
            raise ValueError("ready-node candidates must be pending")
        expected = ready_node_candidate_set_fingerprint(self.candidates)
        if self.candidate_set_fingerprint != expected:
            raise ValueError("ready-node candidate-set fingerprint does not match")
        return self

    def binding_for(self, node_key: str) -> ReadyNodeCandidateBinding:
        for item in self.candidates:
            if item.node_key == node_key:
                return item
        raise ValueError("ready-node selection escaped the Runtime candidate set")


class ReadyNodeSelectionEffect(OrchestrationModel):
    """Runtime-generated ordering choice; it cannot dispatch the node."""

    node_id: UUID
    node_key: str = Field(min_length=1)
    node_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_reason: str = Field(min_length=1)


def ready_node_candidate_set_fingerprint(
    candidates: tuple[ReadyNodeCandidateBinding, ...],
) -> str:
    return ready_node_selection_fingerprint(
        [
            {
                "node_key": item.node_key,
                "node_id": str(item.node.node_id),
                "node_fingerprint": item.node_fingerprint,
            }
            for item in candidates
        ]
    )
