"""Context Unit, lifecycle, scheduling, and assembly models."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonValue,
    utc_now,
)


class ContextSource(StrEnum):
    CONVERSATION = "conversation"
    OBSERVATION = "observation"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
    MEMORY_RECALL = "memory_recall"
    INTERMEDIATE_RESULT = "intermediate_result"
    WORKING_STATE = "working_state"
    EXTERNAL_RESULT = "external_result"


class ContextLayer(StrEnum):
    WORKING = "working"
    TASK = "task"
    SEMANTIC = "semantic"


class ContextLifecycleState(StrEnum):
    ACTIVE = "active"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"


class ContextLifecycleAction(StrEnum):
    COMPRESS = "compress"
    ARCHIVE = "archive"
    RESTORE = "restore"


class ResidencyPolicy(StrEnum):
    PINNED = "pinned"
    SESSION = "session"
    TRANSIENT = "transient"


class ContextArchiveReference(ContextMemoryModel):
    archive_id: UUID = Field(default_factory=uuid4)
    context_id: UUID
    archived_at: AwareDatetime = Field(default_factory=utc_now)


class ContextMetadata(ContextMemoryModel):
    source: ContextSource
    layer: ContextLayer
    run_id: UUID
    task_id: UUID | None = None
    node_id: UUID | None = None
    source_reference: str | None = None
    tags: tuple[str, ...] = ()
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_tokens: int = Field(default=1, ge=1)
    access_count: int = Field(default=0, ge=0)
    last_accessed_at: AwareDatetime | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_metadata(self) -> ContextMetadata:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("context tags must be unique")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if (
            self.last_accessed_at is not None
            and self.last_accessed_at < self.created_at
        ):
            raise ValueError("last_accessed_at cannot precede created_at")
        return self


class ContextUnit(ContextMemoryModel):
    context_id: UUID = Field(default_factory=uuid4)
    content: ImmutableJsonValue
    metadata: ContextMetadata
    lifecycle_state: ContextLifecycleState = ContextLifecycleState.ACTIVE
    residency_policy: ResidencyPolicy = ResidencyPolicy.SESSION
    revision: int = Field(default=0, ge=0)
    core_conclusions: tuple[str, ...] = ()
    recovery_reference: ContextArchiveReference | None = None
    last_effect_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ContextUnit:
        if (
            self.recovery_reference is not None
            and self.recovery_reference.context_id != self.context_id
        ):
            raise ValueError("recovery reference belongs to a different context")
        if self.lifecycle_state is ContextLifecycleState.COMPRESSED:
            if not self.core_conclusions:
                raise ValueError("compressed context requires core conclusions")
            if self.recovery_reference is None:
                raise ValueError("compressed context requires a recovery reference")
        return self


class ContextCompressionResult(ContextMemoryModel):
    content: ImmutableJsonValue
    core_conclusions: tuple[str, ...] = Field(min_length=1)
    estimated_tokens: int = Field(ge=1)


class ContextPressure(ContextMemoryModel):
    run_id: UUID
    resident_units: int = Field(ge=0)
    active_units: int = Field(ge=0)
    compressed_units: int = Field(ge=0)
    resident_tokens: int = Field(ge=0)
    max_resident_tokens: int = Field(ge=1)
    pressure_ratio: float = Field(ge=0.0)


class ContextLifecycleDecision(ContextMemoryModel):
    action: ContextLifecycleAction
    context_id: UUID
    run_id: UUID
    task_id: UUID | None = None
    node_id: UUID | None = None
    source_revision: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)
    pressure_ratio: float = Field(ge=0.0)


class ContextLifecycleResult(ContextMemoryModel):
    decision: ContextLifecycleDecision
    unit: ContextUnit | None = None
    archive_reference: ContextArchiveReference | None = None
    completed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_result(self) -> ContextLifecycleResult:
        if self.decision.action is ContextLifecycleAction.ARCHIVE:
            if self.archive_reference is None:
                raise ValueError("archive result requires an archive reference")
        elif self.unit is None:
            raise ValueError("compress/restore result requires a context unit")
        return self


class ContextRequirement(ContextMemoryModel):
    run_id: UUID
    task_id: UUID | None = None
    node_id: UUID | None = None
    goal: str = Field(min_length=1)
    preferred_tags: tuple[str, ...] = ()
    layers: tuple[ContextLayer, ...] = (
        ContextLayer.WORKING,
        ContextLayer.TASK,
        ContextLayer.SEMANTIC,
    )
    required_context_ids: tuple[UUID, ...] = ()
    max_units: int = Field(default=16, ge=1)
    max_tokens: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def validate_requirement(self) -> ContextRequirement:
        if len(set(self.preferred_tags)) != len(self.preferred_tags):
            raise ValueError("preferred context tags must be unique")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("context layers must be unique")
        if len(set(self.required_context_ids)) != len(
            self.required_context_ids
        ):
            raise ValueError("required context ids must be unique")
        return self


class ContextSchedule(ContextMemoryModel):
    requirement: ContextRequirement
    selected: tuple[ContextUnit, ...]
    omitted_context_ids: tuple[UUID, ...] = ()
    used_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_budget(self) -> ContextSchedule:
        selected_ids = tuple(unit.context_id for unit in self.selected)
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("scheduled context units must be unique")
        if len(set(self.omitted_context_ids)) != len(self.omitted_context_ids):
            raise ValueError("omitted context ids must be unique")
        if set(selected_ids).intersection(self.omitted_context_ids):
            raise ValueError("context cannot be both selected and omitted")
        if not set(self.requirement.required_context_ids).issubset(selected_ids):
            raise ValueError("required context must be selected")
        for unit in self.selected:
            if unit.lifecycle_state is ContextLifecycleState.ARCHIVED:
                raise ValueError("archived context cannot be scheduled")
            if unit.metadata.run_id != self.requirement.run_id:
                raise ValueError("scheduled context belongs to another run")
            if unit.metadata.layer not in self.requirement.layers:
                raise ValueError("scheduled context uses an unrequested layer")
            if (
                self.requirement.task_id is not None
                and unit.metadata.task_id
                not in {None, self.requirement.task_id}
            ):
                raise ValueError("scheduled context belongs to another task")
        expected = sum(unit.metadata.estimated_tokens for unit in self.selected)
        if self.used_tokens != expected:
            raise ValueError("scheduled token usage does not match selected units")
        if self.used_tokens > self.requirement.max_tokens:
            raise ValueError("context schedule exceeds its token budget")
        if len(self.selected) > self.requirement.max_units:
            raise ValueError("context schedule exceeds its unit budget")
        return self


class ContextAssembly(ContextMemoryModel):
    requirement: ContextRequirement
    units: tuple[ContextUnit, ...]
    working_context: tuple[ContextUnit, ...]
    task_context: tuple[ContextUnit, ...]
    semantic_context: tuple[ContextUnit, ...]
    omitted_context_ids: tuple[UUID, ...] = ()
    used_tokens: int = Field(ge=0)
    assembled_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_assembly(self) -> ContextAssembly:
        unit_ids = tuple(unit.context_id for unit in self.units)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("assembled context units must be unique")
        if set(unit_ids).intersection(self.omitted_context_ids):
            raise ValueError("context cannot be both assembled and omitted")
        expected_layers = (
            (self.working_context, ContextLayer.WORKING),
            (self.task_context, ContextLayer.TASK),
            (self.semantic_context, ContextLayer.SEMANTIC),
        )
        for bucket, expected_layer in expected_layers:
            if any(unit.metadata.layer is not expected_layer for unit in bucket):
                raise ValueError(
                    f"{expected_layer.value} context contains a unit from another layer"
                )
        layered = (
            *self.working_context,
            *self.task_context,
            *self.semantic_context,
        )
        if layered != self.units:
            raise ValueError("assembled units must follow working/task/semantic order")
        expected = sum(unit.metadata.estimated_tokens for unit in self.units)
        if expected != self.used_tokens:
            raise ValueError("assembly token usage does not match its units")
        return self


class ContextPreparation(ContextMemoryModel):
    """One closed lifecycle-and-assembly result for the next execution."""

    pressure_before: ContextPressure
    pressure_after: ContextPressure
    lifecycle_results: tuple[ContextLifecycleResult, ...] = ()
    assembly: ContextAssembly
