"""Governed models for projecting Runtime context into model-safe packages."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from adaptive_agent_runtime.context_memory.context_models import (
    ContextAssembly,
    ContextLayer,
    ContextSource,
)
from adaptive_agent_runtime.llm.json_types import ImmutableJsonValue, LLMModel
from adaptive_agent_runtime.llm.models import InferenceTargetProfile


class ContextSensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class LLMContextRole(StrEnum):
    """Context units never become trusted system or developer instructions."""

    USER = "user"
    CONTEXT = "context"


class ContextOmissionReason(StrEnum):
    CAPABILITY_POLICY = "capability_policy"
    EGRESS_POLICY = "egress_policy"
    TOKEN_BUDGET = "token_budget"


class ContextUnitClassification(LLMModel):
    context_id: UUID
    sensitivity: ContextSensitivity


class CapabilityContextPolicy(LLMModel):
    cognitive_capability_id: str = Field(min_length=1)
    allowed_layers: tuple[ContextLayer, ...] = tuple(ContextLayer)
    allowed_sources: tuple[ContextSource, ...] = tuple(ContextSource)
    excluded_tags: tuple[str, ...] = ()
    max_context_tokens: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def validate_policy(self) -> CapabilityContextPolicy:
        _ensure_unique(self.allowed_layers, "allowed context layers")
        _ensure_unique(self.allowed_sources, "allowed context sources")
        _ensure_unique(self.excluded_tags, "excluded context tags")
        if not self.allowed_layers:
            raise ValueError("a capability context policy requires allowed layers")
        if not self.allowed_sources:
            raise ValueError("a capability context policy requires allowed sources")
        return self


class ContextEgressPolicy(LLMModel):
    allowed_target_ids: tuple[str, ...] = Field(min_length=1)
    allowed_sensitivities: tuple[ContextSensitivity, ...] = (
        ContextSensitivity.PUBLIC,
        ContextSensitivity.INTERNAL,
    )
    blocked_tags: tuple[str, ...] = ()
    redact_keys: tuple[str, ...] = ()
    redaction_placeholder: str = Field(default="[REDACTED]", min_length=1)
    max_context_tokens: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def validate_policy(self) -> ContextEgressPolicy:
        _ensure_unique(self.allowed_target_ids, "allowed inference targets")
        _ensure_unique(
            self.allowed_sensitivities,
            "allowed context sensitivities",
        )
        _ensure_unique(self.blocked_tags, "blocked context tags")
        _ensure_unique(self.redact_keys, "redacted context keys")
        if not self.allowed_sensitivities:
            raise ValueError("an egress policy requires allowed sensitivities")
        return self


class ContextProjectionRequest(LLMModel):
    assembly: ContextAssembly
    target: InferenceTargetProfile
    capability_policy: CapabilityContextPolicy
    egress_policy: ContextEgressPolicy
    classifications: tuple[ContextUnitClassification, ...]

    @model_validator(mode="after")
    def validate_request(self) -> ContextProjectionRequest:
        classified = tuple(item.context_id for item in self.classifications)
        if len(set(classified)) != len(classified):
            raise ValueError("context classifications must be unique")
        assembled = {unit.context_id for unit in self.assembly.units}
        if set(classified) != assembled:
            raise ValueError(
                "context classifications must exactly cover assembled units"
            )
        return self


class LLMContextBlock(LLMModel):
    context_id: UUID
    role: LLMContextRole
    source: ContextSource
    layer: ContextLayer
    source_reference: str | None = Field(default=None, min_length=1)
    sensitivity: ContextSensitivity
    content: ImmutableJsonValue
    estimated_tokens: int = Field(ge=1)
    redacted_keys: tuple[str, ...] = ()


class ContextOmission(LLMModel):
    context_id: UUID
    reason: ContextOmissionReason


class LLMContextPackage(LLMModel):
    cognitive_capability_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    blocks: tuple[LLMContextBlock, ...]
    omissions: tuple[ContextOmission, ...] = ()
    used_tokens: int = Field(ge=0)
    max_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_package(self) -> LLMContextPackage:
        block_ids = tuple(block.context_id for block in self.blocks)
        omitted_ids = tuple(item.context_id for item in self.omissions)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("LLM context blocks must be unique")
        if len(set(omitted_ids)) != len(omitted_ids):
            raise ValueError("LLM context omissions must be unique")
        if set(block_ids).intersection(omitted_ids):
            raise ValueError("context cannot be both projected and omitted")
        expected = sum(block.estimated_tokens for block in self.blocks)
        if self.used_tokens != expected:
            raise ValueError("LLM context token usage does not match its blocks")
        if self.used_tokens > self.max_tokens:
            raise ValueError("LLM context package exceeds its token budget")
        return self


def _ensure_unique(values: tuple[object, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
