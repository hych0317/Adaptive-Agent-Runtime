"""Auditable test-only Profile manifests and boundary fingerprints."""

from __future__ import annotations

from pydantic import Field

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.governance_scenario_suite.contracts import (
    ScenarioContractModel,
    ScenarioProfile,
)


class ProfileManifest(ScenarioContractModel):
    profile: ScenarioProfile
    disabled_boundaries: tuple[str, ...] = ()
    system_prompt_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_schema_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_flow_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain_store_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    gateway_contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


_DISABLED: dict[ScenarioProfile, tuple[str, ...]] = {
    ScenarioProfile.FULL_AAR: (),
    ScenarioProfile.PLAIN_AGENT: (
        "exact_effect_binding",
        "memory_scope_filter",
        "reconcile_before_retry",
        "resource_scope_projection",
    ),
    ScenarioProfile.NO_EXACT_EFFECT_BINDING: ("exact_effect_binding",),
    ScenarioProfile.NO_MEMORY_SCOPE_FILTER: ("memory_scope_filter",),
    ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK: (
        "authoritative_constraint_check",
    ),
    ScenarioProfile.NO_RECONCILE_FAIL_CLOSED: ("retry_known_not_committed",),
    ScenarioProfile.NO_RECONCILE_BLIND_RETRY: ("reconcile_before_retry",),
}


def build_profile_manifest(profile: ScenarioProfile) -> ProfileManifest:
    disabled = _DISABLED[profile]
    return ProfileManifest(
        profile=profile,
        disabled_boundaries=disabled,
        system_prompt_fingerprint=decision_fingerprint(
            {"prompt_contract": "ecommerce-support-v1"}
        ),
        tool_schema_fingerprint=decision_fingerprint(
            {"tools": "ecommerce-support-tools-v1"}
        ),
        control_flow_fingerprint=decision_fingerprint(
            {"profile": profile.value, "disabled": disabled}
        ),
        domain_store_contract_fingerprint=decision_fingerprint(
            {"contract": "EcommerceStorePort-v1"}
        ),
        gateway_contract_fingerprint=decision_fingerprint(
            {"contract": "PaymentGatewayPort-v1"}
        ),
    )


def all_profile_manifests() -> tuple[ProfileManifest, ...]:
    return tuple(build_profile_manifest(profile) for profile in ScenarioProfile)
