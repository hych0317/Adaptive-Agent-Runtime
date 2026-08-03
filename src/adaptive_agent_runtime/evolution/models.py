"""Immutable models for replay-validated Runtime evolution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from adaptive_agent_runtime.context_memory.json_types import utc_now
from adaptive_agent_runtime.evaluation.models import (
    ImmutableJsonObject,
    OptimizationProposal,
)


_EVOLUTION_NAMESPACE = UUID("78e616ef-31d9-4e92-ae35-22fb5c780060")


def stable_evolution_id(*parts: object) -> UUID:
    return uuid5(_EVOLUTION_NAMESPACE, "|".join(str(part) for part in parts))


class EvolutionModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        values = self.model_dump(mode="python")
        if update:
            values.update(update)
        return self.__class__.model_validate(values)


class OptimizationApplicationStatus(StrEnum):
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class RuntimeConfigurationSnapshot(EvolutionModel):
    component: str = Field(min_length=1)
    version: int = Field(ge=0)
    config: ImmutableJsonObject
    source_proposal_id: UUID | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)


class ReplayCase(EvolutionModel):
    case_id: UUID
    source_run_id: UUID
    task_id: UUID
    input_payload: ImmutableJsonObject
    baseline_score: float = Field(ge=0.0, le=1.0)
    minimum_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: AwareDatetime = Field(default_factory=utc_now)


class ReplayExecutionResult(EvolutionModel):
    replay_run_id: UUID
    score: float = Field(ge=0.0, le=1.0)
    succeeded: bool
    diagnostics: tuple[str, ...] = ()
    completed_at: AwareDatetime = Field(default_factory=utc_now)


class ReplayObservation(EvolutionModel):
    case_id: UUID
    source_run_id: UUID
    candidate_component: str
    candidate_version: int = Field(ge=1)
    execution: ReplayExecutionResult


class ReplayValidation(EvolutionModel):
    passed: bool
    baseline_score: float = Field(ge=0.0, le=1.0)
    candidate_score: float = Field(ge=0.0, le=1.0)
    findings: tuple[str, ...] = ()
    observations: tuple[ReplayObservation, ...] = Field(min_length=1)
    validated_at: AwareDatetime = Field(default_factory=utc_now)


class OptimizationDeployment(EvolutionModel):
    deployment_id: UUID
    proposal: OptimizationProposal
    baseline: RuntimeConfigurationSnapshot
    candidate: RuntimeConfigurationSnapshot
    validation: ReplayValidation
    prepared_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_deployment(self) -> OptimizationDeployment:
        if self.baseline.component != self.candidate.component:
            raise ValueError("candidate configuration changes component identity")
        if self.candidate.version != self.baseline.version + 1:
            raise ValueError("candidate configuration must increment one version")
        if self.candidate.source_proposal_id != self.proposal.proposal_id:
            raise ValueError("candidate configuration has another source proposal")
        return self


class OptimizationApplication(EvolutionModel):
    application_id: UUID
    deployment: OptimizationDeployment
    status: OptimizationApplicationStatus
    revision: int = Field(default=0, ge=0)
    applied_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_application(self) -> OptimizationApplication:
        if self.updated_at < self.applied_at:
            raise ValueError("application update precedes apply")
        return self

