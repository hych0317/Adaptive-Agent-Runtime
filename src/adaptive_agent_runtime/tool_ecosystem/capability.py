"""Capability catalog, exact matching, and Runtime candidate filtering."""

from __future__ import annotations

from adaptive_agent_runtime.tool_ecosystem.contracts import (
    CapabilityCatalog,
    CapabilityMatcher,
    ToolRegistry,
)
from adaptive_agent_runtime.tool_ecosystem.errors import (
    CapabilityAlreadyRegisteredError,
)
from adaptive_agent_runtime.tool_ecosystem.models import (
    Capability,
    CapabilityMatch,
    CapabilityRequirement,
    ToolProviderMetadata,
)


class InMemoryCapabilityCatalog:
    module_id = "capability.catalog.in_memory"

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if capability.capability_id in self._capabilities:
            raise CapabilityAlreadyRegisteredError(
                f"capability '{capability.capability_id}' is already registered"
            )
        self._capabilities[capability.capability_id] = capability

    def get(self, capability_id: str) -> Capability | None:
        return self._capabilities.get(capability_id)

    def list_all(self) -> tuple[Capability, ...]:
        return tuple(
            self._capabilities[capability_id]
            for capability_id in sorted(self._capabilities)
        )


class ExactCapabilityMatcher:
    module_id = "capability.matcher.exact"

    def match(
        self,
        requirement: CapabilityRequirement,
        capability: Capability,
    ) -> CapabilityMatch | None:
        if requirement.capability_id != capability.capability_id:
            return None
        if not set(requirement.required_capability_tags).issubset(
            capability.tags
        ):
            return None
        return CapabilityMatch(
            requirement_id=requirement.requirement_id,
            capability=capability,
            score=1.0,
        )


class CapabilityResolver:
    """Filter provider metadata by explicit Capability requirements."""

    module_id = "capability.resolver.runtime"

    def __init__(
        self,
        *,
        catalog: CapabilityCatalog,
        registry: ToolRegistry,
        matcher: CapabilityMatcher,
    ) -> None:
        self._catalog = catalog
        self._registry = registry
        self._matcher = matcher

    def candidates(
        self,
        requirement: CapabilityRequirement,
    ) -> tuple[ToolProviderMetadata, ...]:
        matches = tuple(
            capability
            for capability in self._catalog.list_all()
            if self._matcher.match(requirement, capability) is not None
        )
        required_tags = set(requirement.required_provider_tags)
        candidates: dict[str, ToolProviderMetadata] = {}
        for capability in matches:
            for metadata in self._registry.metadata_for_capability(
                capability.capability_id
            ):
                if required_tags.issubset(metadata.tags):
                    candidates[metadata.provider_id] = metadata
        return tuple(candidates[item] for item in sorted(candidates))

