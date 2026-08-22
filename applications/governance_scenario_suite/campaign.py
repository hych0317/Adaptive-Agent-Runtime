"""Repeatable, traceable live-model governance experiment campaigns."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from pydantic import AwareDatetime, Field, StrictInt, model_validator

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.governance_scenario_suite.baseline import (
    FullAARScenarioExecutor,
    PlainAgentExecutor,
)
from applications.governance_scenario_suite.context_memory_executors import (
    NoAuthoritativeConstraintExecutor,
    NoMemoryScopeFilterExecutor,
)
from applications.governance_scenario_suite.contracts import (
    ScenarioContractModel,
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.e2e import (
    E2EModelConfig,
    gateway_model_factory,
)
from applications.governance_scenario_suite.e2e_metrics import (
    E2EAggregate,
    aggregate_e2e_results,
)
from applications.governance_scenario_suite.evidence import ScenarioRunResult
from applications.governance_scenario_suite.executors import (
    NoExactEffectBindingExecutor,
)
from applications.governance_scenario_suite.profiles import (
    ProfileManifest,
    build_profile_manifest,
)
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import (
    ScenarioExecutor,
    ScenarioExecutorRegistry,
)


_E2E_EXECUTORS: dict[ScenarioProfile, type[ScenarioExecutor]] = {
    ScenarioProfile.FULL_AAR: FullAARScenarioExecutor,
    ScenarioProfile.PLAIN_AGENT: PlainAgentExecutor,
    ScenarioProfile.NO_EXACT_EFFECT_BINDING: NoExactEffectBindingExecutor,
    ScenarioProfile.NO_MEMORY_SCOPE_FILTER: NoMemoryScopeFilterExecutor,
    ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK: (
        NoAuthoritativeConstraintExecutor
    ),
}


class E2ECampaignConfig(ScenarioContractModel):
    suite_id: str = Field(default="aar-governance-e2e-v1", min_length=1)
    scenario_ids: tuple[str, ...] = Field(min_length=1)
    profile: ScenarioProfile = ScenarioProfile.FULL_AAR
    repeat: StrictInt = Field(default=1, ge=1, le=100)
    model: E2EModelConfig

    @model_validator(mode="after")
    def validate_campaign(self) -> E2ECampaignConfig:
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("E2E scenario IDs must be unique")
        if self.profile not in _E2E_EXECUTORS:
            raise ValueError(
                f"profile '{self.profile.value}' is not a model-loop E2E profile"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return decision_fingerprint(self)


class ScenarioTrace(ScenarioContractModel):
    scenario_id: str = Field(min_length=1)
    spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class E2ECampaignResult(ScenarioContractModel):
    schema_version: int = Field(default=1, ge=1)
    suite_id: str = Field(min_length=1)
    generated_at: AwareDatetime
    campaign_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_manifest: ProfileManifest
    scenario_traces: tuple[ScenarioTrace, ...]
    results: tuple[ScenarioRunResult, ...]
    aggregate: E2EAggregate


def run_e2e_campaign(
    scenarios: Sequence[ScenarioSpec],
    config: E2ECampaignConfig,
    *,
    generated_at: datetime,
) -> E2ECampaignResult:
    by_id = {item.id: item for item in scenarios}
    missing = tuple(item for item in config.scenario_ids if item not in by_id)
    if missing:
        raise ValueError(f"unknown E2E scenario IDs: {', '.join(missing)}")
    selected = tuple(by_id[item] for item in config.scenario_ids)
    recovery = tuple(item.id for item in selected if item.family_id.startswith("R"))
    if recovery:
        raise ValueError(
            "cross-process recovery scenarios use the deterministic recovery runner: "
            + ", ".join(recovery)
        )
    executor = _E2E_EXECUTORS[config.profile]()
    runner = GovernanceScenarioRunner(
        executors=ScenarioExecutorRegistry({config.profile: executor}),
        model_factory=gateway_model_factory(config.model),
    )
    results = tuple(
        runner.run(
            scenario.model_copy(update={"profile": config.profile}),
            run_label=f"repeat-{repeat_index + 1}",
        )
        for scenario in selected
        for repeat_index in range(config.repeat)
    )
    profiled_scenarios = {
        item.id: item.model_copy(update={"profile": config.profile}) for item in selected
    }
    traces = tuple(
        ScenarioTrace(
            scenario_id=item.id,
            spec_fingerprint=decision_fingerprint(profiled_scenarios[item.id]),
        )
        for item in selected
    )
    return E2ECampaignResult(
        suite_id=config.suite_id,
        generated_at=generated_at,
        campaign_fingerprint=config.fingerprint,
        model_config_fingerprint=config.model.fingerprint,
        profile_manifest=build_profile_manifest(config.profile),
        scenario_traces=traces,
        results=results,
        aggregate=aggregate_e2e_results(results, profiled_scenarios),
    )


def write_e2e_campaign_result(
    campaign: E2ECampaignResult,
    output_directory: str | Path,
) -> Path:
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "governance-e2e-campaign.json"
    destination.write_text(campaign.model_dump_json(indent=2), encoding="utf-8")
    E2ECampaignResult.model_validate_json(destination.read_text(encoding="utf-8"))
    return destination
