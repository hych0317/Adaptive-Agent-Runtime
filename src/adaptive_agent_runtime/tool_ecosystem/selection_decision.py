"""Runtime-owned contracts for governed, Agent-proposed Tool selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from adaptive_agent_runtime.tool_ecosystem.models import (
    CapabilityRequest,
    CapabilityRequirement,
    ProviderAvailability,
    ToolModel,
    ToolProviderMetadata,
    ToolSelection,
    ToolSelectionContext,
)


TOOL_SELECTION_DECISION_TYPE = "tool.provider_selection"
TOOL_SELECTION_BIND_OPERATION = "tool.selection.bind"

_TOOL_SELECTION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/tool-ecosystem/selection-decision",
)


def stable_tool_selection_id(*parts: object) -> UUID:
    return uuid5(_TOOL_SELECTION_NAMESPACE, "|".join(str(item) for item in parts))


def tool_selection_fingerprint(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ToolSelectionExecutionPolicy(ToolModel):
    """Runtime-owned bound around one semantic Provider selection."""

    timeout_seconds: float = Field(default=15.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_revision_count: int = Field(default=0, ge=0, le=0)


class ToolCandidateBinding(ToolModel):
    """Runtime-only binding from an opaque candidate reference to metadata."""

    candidate_ref: str = Field(min_length=1)
    metadata: ToolProviderMetadata
    metadata_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_metadata(self) -> ToolCandidateBinding:
        expected = tool_selection_fingerprint(self.metadata)
        if self.metadata_fingerprint != expected:
            raise ValueError("Tool candidate metadata fingerprint does not match")
        if self.metadata.availability is not ProviderAvailability.AVAILABLE:
            raise ValueError("Agent-visible Tool candidates must be available")
        return self


class ToolSelectionDecisionPayload(ToolModel):
    """Full Runtime snapshot retained outside the isolated Agent context."""

    invocation_id: UUID
    task_description: str = Field(min_length=1)
    node_goal: str = Field(min_length=1)
    requirement: CapabilityRequirement
    arguments_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_context: ToolSelectionContext
    candidates: tuple[ToolCandidateBinding, ...] = Field(min_length=1)
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_revision: int = Field(ge=0)
    execution_policy: ToolSelectionExecutionPolicy = Field(
        default_factory=ToolSelectionExecutionPolicy
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> ToolSelectionDecisionPayload:
        refs = tuple(item.candidate_ref for item in self.candidates)
        provider_ids = tuple(item.metadata.provider_id for item in self.candidates)
        if len(set(refs)) != len(refs) or len(set(provider_ids)) != len(provider_ids):
            raise ValueError("Tool candidate bindings must be one-to-one")
        required_tags = set(self.requirement.required_provider_tags)
        for item in self.candidates:
            metadata = item.metadata
            if metadata.capability_id != self.requirement.capability_id:
                raise ValueError("Tool candidate implements another capability")
            if not required_tags.issubset(metadata.tags):
                raise ValueError("Tool candidate does not satisfy required tags")
        expected = tool_selection_fingerprint(
            [
                {
                    "candidate_ref": item.candidate_ref,
                    "metadata_fingerprint": item.metadata_fingerprint,
                }
                for item in self.candidates
            ]
        )
        if self.candidate_set_fingerprint != expected:
            raise ValueError("Tool candidate-set fingerprint does not match")
        return self

    def binding_for(self, candidate_ref: str) -> ToolCandidateBinding:
        for item in self.candidates:
            if item.candidate_ref == candidate_ref:
                return item
        raise ValueError("Tool selection escaped the Runtime candidate set")


class ToolSelectionEffect(ToolModel):
    """Runtime-generated Provider binding; it cannot invoke the Provider."""

    invocation_id: UUID
    requirement_id: UUID
    capability_id: str = Field(min_length=1)
    candidate_ref: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_metadata_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_reason: str = Field(min_length=1)
    agent_confidence: float = Field(ge=0.0, le=1.0)


class ToolSelectionDecisionOutcome(ToolModel):
    request_id: UUID
    effect: ToolSelectionEffect
    selection: ToolSelection

    @model_validator(mode="after")
    def validate_selection(self) -> ToolSelectionDecisionOutcome:
        if self.selection.requirement_id != self.effect.requirement_id:
            raise ValueError("Tool Selection outcome requirement does not match effect")
        if self.selection.capability_id != self.effect.capability_id:
            raise ValueError("Tool Selection outcome capability does not match effect")
        if self.selection.provider_id != self.effect.provider_id:
            raise ValueError("Tool Selection outcome Provider does not match effect")
        return self


def eligible_tool_candidate_bindings(
    *,
    invocation_id: UUID,
    request: CapabilityRequest,
    candidates: Sequence[ToolProviderMetadata],
) -> tuple[ToolCandidateBinding, ...]:
    """Fail closed while projecting Resolver output into selection candidates."""

    arguments = request.model_dump(mode="json")["arguments"]
    required_tags = set(request.requirement.required_provider_tags)
    eligible: list[ToolCandidateBinding] = []
    for metadata in sorted(candidates, key=lambda item: item.provider_id):
        if metadata.capability_id != request.requirement.capability_id:
            continue
        if not required_tags.issubset(metadata.tags):
            continue
        if metadata.availability is not ProviderAvailability.AVAILABLE:
            continue
        schema = metadata.model_dump(mode="json")["input_schema"]
        try:
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema)
        except SchemaError:
            continue
        if not validator.is_valid(arguments):
            continue
        eligible.append(
            ToolCandidateBinding(
                candidate_ref=str(
                    stable_tool_selection_id(
                        "candidate",
                        invocation_id,
                        metadata.provider_id,
                    )
                ),
                metadata=metadata,
                metadata_fingerprint=tool_selection_fingerprint(metadata),
            )
        )
    return tuple(eligible)


def candidate_set_fingerprint(
    candidates: Sequence[ToolCandidateBinding],
) -> str:
    return tool_selection_fingerprint(
        [
            {
                "candidate_ref": item.candidate_ref,
                "metadata_fingerprint": item.metadata_fingerprint,
            }
            for item in candidates
        ]
    )
