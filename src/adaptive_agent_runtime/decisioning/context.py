"""Deny-by-default Agent context projection and isolation boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from adaptive_agent_runtime.core.models import ImmutableJsonObject
from adaptive_agent_runtime.decisioning.models import (
    AgentContextManifest,
    DecisionBudget,
    DecisionConstraint,
    DecisionEvidenceReference,
    DecisionModel,
    DecisionRequest,
    DecisionTarget,
    decision_fingerprint,
)


RUNTIME_SHARED_SCOPE = "runtime_shared"


class ContextSensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class ContextOmissionReason(StrEnum):
    DECISION_TYPE = "decision_type"
    SOURCE_TYPE = "source_type"
    AGENT_SCOPE = "agent_scope"
    MEMORY_SCOPE = "memory_scope"
    TRACE_CATEGORY = "trace_category"
    SENSITIVITY = "sensitivity"
    BLOCKED_TAG = "blocked_tag"
    ITEM_BUDGET = "item_budget"
    TOKEN_BUDGET = "token_budget"


class ProjectionSource(DecisionModel):
    """A read-only candidate supplied by Runtime, never a Store reference."""

    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    agent_scope: str = Field(min_length=1)
    content: ImmutableJsonObject
    sensitivity: ContextSensitivity = ContextSensitivity.INTERNAL
    memory_scope: str | None = None
    trace_category: str | None = None
    evidence_id: str | None = None
    tags: frozenset[str] = frozenset()
    priority: int = 0
    estimated_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_tags(self) -> ProjectionSource:
        if any(not tag for tag in self.tags):
            raise ValueError("projection source tags cannot be empty")
        return self


class ProjectionSources(DecisionModel):
    """Finite source snapshot prepared outside the isolation boundary."""

    items: tuple[ProjectionSource, ...] = ()

    @model_validator(mode="after")
    def validate_unique_sources(self) -> ProjectionSources:
        source_ids = tuple(item.source_id for item in self.items)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("projection source ids must be unique")
        return self


class ContextProjectionPolicy(DecisionModel):
    """Small declarative policy; this is not a general policy engine."""

    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    agent_scope: str = Field(min_length=1)
    allowed_decision_types: frozenset[str] = Field(min_length=1)
    allowed_source_types: frozenset[str] = Field(min_length=1)
    allowed_memory_scopes: frozenset[str] = frozenset()
    allowed_trace_categories: frozenset[str] = frozenset()
    allowed_sensitivity_levels: frozenset[ContextSensitivity] = Field(
        default_factory=lambda: frozenset({ContextSensitivity.PUBLIC})
    )
    blocked_tags: frozenset[str] = frozenset()
    redact_keys: frozenset[str] = frozenset()
    max_items: int = Field(default=16, ge=0)
    max_context_tokens: int = Field(default=4096, ge=0)

    @model_validator(mode="after")
    def validate_policy(self) -> ContextProjectionPolicy:
        string_sets = (
            self.allowed_decision_types,
            self.allowed_source_types,
            self.allowed_memory_scopes,
            self.allowed_trace_categories,
            self.blocked_tags,
            self.redact_keys,
        )
        if any(any(not item for item in values) for values in string_sets):
            raise ValueError("context policy selectors cannot be empty")
        return self


class ProjectedContextBlock(DecisionModel):
    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    content: ImmutableJsonObject
    evidence_id: str | None = None
    estimated_tokens: int = Field(ge=0)


class ContextOmission(DecisionModel):
    source_id: str = Field(min_length=1)
    reason: ContextOmissionReason
    detail: str = Field(min_length=1)


class AgentContext(DecisionModel):
    """The only decision input an Agent is permitted to receive."""

    context_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    decision_type: str = Field(min_length=1)
    agent_scope: str = Field(min_length=1)
    target: DecisionTarget
    constraints: tuple[DecisionConstraint, ...] = ()
    budget: DecisionBudget
    evidence: tuple[DecisionEvidenceReference, ...] = ()
    blocks: tuple[ProjectedContextBlock, ...] = ()
    omissions: tuple[ContextOmission, ...] = ()
    projection_policy_id: str = Field(min_length=1)
    projection_policy_version: str = Field(min_length=1)
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_context(self) -> AgentContext:
        block_ids = tuple(item.source_id for item in self.blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("Agent context block ids must be unique")
        expected = _agent_context_fingerprint(self)
        if self.context_fingerprint != expected:
            raise ValueError("Agent context fingerprint does not match projection")
        return self

    def manifest(self) -> AgentContextManifest:
        return AgentContextManifest(
            context_id=self.context_id,
            request_id=self.request_id,
            agent_scope=self.agent_scope,
            policy_id=self.projection_policy_id,
            policy_version=self.projection_policy_version,
            context_fingerprint=self.context_fingerprint,
            included_source_ids=tuple(item.source_id for item in self.blocks),
            omitted_source_ids=tuple(item.source_id for item in self.omissions),
            estimated_tokens=sum(item.estimated_tokens for item in self.blocks),
        )


def _agent_context_fingerprint(context: AgentContext) -> str:
    return decision_fingerprint(
        {
            "context_id": context.context_id,
            "request_id": context.request_id,
            "decision_type": context.decision_type,
            "agent_scope": context.agent_scope,
            "target": context.target,
            "constraints": context.constraints,
            "budget": context.budget,
            "evidence": context.evidence,
            "blocks": context.blocks,
            "omissions": context.omissions,
            "projection_policy_id": context.projection_policy_id,
            "projection_policy_version": context.projection_policy_version,
            "basis_fingerprint": context.basis_fingerprint,
        }
    )


class PolicyAgentContextBuilder:
    """Project finite Runtime inputs through an explicit isolation policy."""

    module_id = "decision.context_builder.policy"

    def build(
        self,
        request: DecisionRequest[Any],
        sources: ProjectionSources,
        policy: ContextProjectionPolicy,
    ) -> AgentContext:
        if request.decision_type not in policy.allowed_decision_types:
            raise ValueError(
                f"decision type '{request.decision_type}' is not allowed by "
                f"context policy '{policy.policy_id}'"
            )

        included: list[ProjectedContextBlock] = []
        omissions: list[ContextOmission] = []
        used_tokens = 0
        ordered = sorted(
            sources.items,
            key=lambda item: (-item.priority, item.source_id),
        )
        for source in ordered:
            rejection = self._reject_reason(request, source, policy)
            if rejection is not None:
                omissions.append(rejection)
                continue
            if len(included) >= policy.max_items:
                omissions.append(
                    ContextOmission(
                        source_id=source.source_id,
                        reason=ContextOmissionReason.ITEM_BUDGET,
                        detail="context item budget was exhausted",
                    )
                )
                continue
            if used_tokens + source.estimated_tokens > policy.max_context_tokens:
                omissions.append(
                    ContextOmission(
                        source_id=source.source_id,
                        reason=ContextOmissionReason.TOKEN_BUDGET,
                        detail="context token budget was exhausted",
                    )
                )
                continue
            content = _redact_mapping(source.content, policy.redact_keys)
            included.append(
                ProjectedContextBlock(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    content=content,
                    evidence_id=source.evidence_id,
                    estimated_tokens=source.estimated_tokens,
                )
            )
            used_tokens += source.estimated_tokens

        included_evidence_ids = {
            item.evidence_id
            for item in included
            if item.evidence_id is not None
        }
        evidence = tuple(
            item
            for item in request.evidence
            if item.evidence_id in included_evidence_ids
        )
        values: dict[str, object] = {
            "context_id": uuid4(),
            "request_id": request.request_id,
            "decision_type": request.decision_type,
            "agent_scope": policy.agent_scope,
            "target": request.target,
            "constraints": request.constraints,
            "budget": request.budget,
            "evidence": evidence,
            "blocks": tuple(included),
            "omissions": tuple(omissions),
            "projection_policy_id": policy.policy_id,
            "projection_policy_version": policy.version,
            "basis_fingerprint": request.basis.snapshot_fingerprint,
        }
        provisional = decision_fingerprint(values)
        values["context_fingerprint"] = provisional
        return AgentContext.model_validate(values)

    @staticmethod
    def _reject_reason(
        request: DecisionRequest[Any],
        source: ProjectionSource,
        policy: ContextProjectionPolicy,
    ) -> ContextOmission | None:
        del request
        reason: ContextOmissionReason | None = None
        detail = ""
        if source.source_type not in policy.allowed_source_types:
            reason = ContextOmissionReason.SOURCE_TYPE
            detail = "source type is not allowlisted"
        elif source.agent_scope not in {policy.agent_scope, RUNTIME_SHARED_SCOPE}:
            reason = ContextOmissionReason.AGENT_SCOPE
            detail = "source belongs to another Agent scope"
        elif (
            source.memory_scope is not None
            and source.memory_scope not in policy.allowed_memory_scopes
        ):
            reason = ContextOmissionReason.MEMORY_SCOPE
            detail = "memory scope is not allowlisted"
        elif (
            source.trace_category is not None
            and source.trace_category not in policy.allowed_trace_categories
        ):
            reason = ContextOmissionReason.TRACE_CATEGORY
            detail = "trace category is not allowlisted"
        elif source.sensitivity not in policy.allowed_sensitivity_levels:
            reason = ContextOmissionReason.SENSITIVITY
            detail = "source sensitivity is not allowlisted"
        elif source.tags & policy.blocked_tags:
            reason = ContextOmissionReason.BLOCKED_TAG
            detail = "source contains a blocked tag"
        if reason is None:
            return None
        return ContextOmission(
            source_id=source.source_id,
            reason=reason,
            detail=detail,
        )


def _redact_mapping(
    value: Mapping[str, JsonValue],
    redact_keys: frozenset[str],
) -> dict[str, JsonValue]:
    return {
        key: "[REDACTED]" if key in redact_keys else _redact_value(item, redact_keys)
        for key, item in value.items()
    }


def _redact_value(value: JsonValue, redact_keys: frozenset[str]) -> JsonValue:
    if isinstance(value, Mapping):
        return _redact_mapping(value, redact_keys)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, redact_keys) for item in value]
    return value
