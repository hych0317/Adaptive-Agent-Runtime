"""Runtime-owned contracts for governed Agent-proposed Tool invocation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from adaptive_agent_runtime.tool_ecosystem.models import (
    CapabilityRequirement,
    ImmutableJsonObject,
    ProviderAvailability,
    ToolInvocation,
    ToolModel,
    ToolObservation,
    ToolProviderMetadata,
)


TOOL_INVOCATION_DECISION_TYPE = "tool.invocation"
TOOL_INVOCATION_OPERATION = "tool.call"
TOOL_INVOCATION_INPUT_SOURCE_TYPE = "tool_invocation_input"

_TOOL_INVOCATION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/tool-ecosystem/invocation-decision",
)


def stable_tool_invocation_decision_id(*parts: object) -> UUID:
    return uuid5(_TOOL_INVOCATION_NAMESPACE, "|".join(str(item) for item in parts))


def tool_invocation_fingerprint(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ToolInvocationProposalDraft(ToolModel):
    """Agent-authored intent with no Runtime identity or execution authority."""

    call_key: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    arguments: ImmutableJsonObject = Field(default_factory=dict)


class ToolInvocationDecisionPayload(ToolModel):
    """Runtime snapshot retained outside the isolated Agent projection."""

    task_description: str = Field(min_length=1)
    node_goal: str = Field(min_length=1)
    invocation: ToolInvocation
    requirement: CapabilityRequirement
    provider_metadata: ToolProviderMetadata
    provider_metadata_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_argument_constraints: ImmutableJsonObject = Field(default_factory=dict)
    state_revision: int = Field(ge=0)
    privileged: bool = False

    @model_validator(mode="after")
    def validate_runtime_binding(self) -> ToolInvocationDecisionPayload:
        invocation = self.invocation
        metadata = self.provider_metadata
        if invocation.requirement_id != self.requirement.requirement_id:
            raise ValueError("Tool invocation requirement does not match Runtime request")
        if invocation.capability_id != self.requirement.capability_id:
            raise ValueError("Tool invocation changed the required capability")
        if metadata.provider_id != invocation.provider_id:
            raise ValueError("Tool invocation Provider does not match Runtime metadata")
        if metadata.capability_id != invocation.capability_id:
            raise ValueError("Tool Provider implements another capability")
        if metadata.availability is not ProviderAvailability.AVAILABLE:
            raise ValueError("Tool invocation Provider is not available")
        if self.provider_metadata_fingerprint != tool_invocation_fingerprint(metadata):
            raise ValueError("Tool Provider metadata fingerprint does not match")
        _validate_arguments(metadata, invocation.arguments)
        _validate_exact_constraints(
            invocation.arguments,
            self.exact_argument_constraints,
        )
        return self


class ToolInvocationEffect(ToolModel):
    """Final Runtime invocation submitted to Governance and governed Apply."""

    invocation: ToolInvocation
    agent_call_key: str = Field(min_length=1)
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_metadata_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ToolInvocationDecisionOutcome(ToolModel):
    request_id: UUID
    effect: ToolInvocationEffect
    observation: ToolObservation

    @model_validator(mode="after")
    def validate_observation(self) -> ToolInvocationDecisionOutcome:
        invocation = self.effect.invocation
        observation = self.observation
        if observation.invocation_id != invocation.invocation_id:
            raise ValueError("Tool observation belongs to another invocation")
        if observation.requirement_id != invocation.requirement_id:
            raise ValueError("Tool observation requirement does not match invocation")
        if observation.capability_id != invocation.capability_id:
            raise ValueError("Tool observation capability does not match invocation")
        if observation.provider_id != invocation.provider_id:
            raise ValueError("Tool observation Provider does not match invocation")
        return self


def tool_invocation_candidate_set_fingerprint(
    candidates: Sequence[ToolProviderMetadata],
) -> str:
    return tool_invocation_fingerprint(
        [
            {
                "provider_id": item.provider_id,
                "metadata_fingerprint": tool_invocation_fingerprint(item),
            }
            for item in sorted(candidates, key=lambda candidate: candidate.provider_id)
        ]
    )


def _validate_arguments(
    metadata: ToolProviderMetadata,
    arguments: Mapping[str, object],
) -> None:
    schema = metadata.model_dump(mode="json")["input_schema"]
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        raise ValueError("Tool Provider input schema is invalid") from exc
    if not validator.is_valid(dict(arguments)):
        raise ValueError("Tool invocation arguments do not match Provider schema")


def _validate_exact_constraints(
    arguments: Mapping[str, object],
    constraints: Mapping[str, object],
) -> None:
    for key, expected in constraints.items():
        if arguments.get(key) != expected:
            raise ValueError(f"Tool invocation violates Runtime constraint '{key}'")
