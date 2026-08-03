"""Deterministic provider selection over Runtime-filtered candidates."""

from __future__ import annotations

from typing import Sequence

from adaptive_agent_runtime.tool_ecosystem.errors import (
    ProviderUnavailableError,
    ToolSelectionError,
)
from adaptive_agent_runtime.tool_ecosystem.models import (
    CapabilityRequirement,
    ProviderAvailability,
    ToolProviderMetadata,
    ToolSelection,
    ToolSelectionContext,
)


class DeterministicToolSelector:
    module_id = "tool.selector.deterministic"

    def select(
        self,
        requirement: CapabilityRequirement,
        candidates: Sequence[ToolProviderMetadata],
        context: ToolSelectionContext,
    ) -> ToolSelection:
        matching = tuple(
            metadata
            for metadata in candidates
            if metadata.capability_id == requirement.capability_id
            and set(requirement.required_provider_tags).issubset(metadata.tags)
        )
        if not matching:
            raise ToolSelectionError(
                f"no provider matches capability '{requirement.capability_id}'"
            )
        available = tuple(
            metadata
            for metadata in matching
            if metadata.availability is ProviderAvailability.AVAILABLE
        )
        if not available:
            raise ProviderUnavailableError(
                f"all providers for '{requirement.capability_id}' are unavailable"
            )
        preferred_tags = set(requirement.preferred_provider_tags).union(
            context.tags
        )
        selected = min(
            available,
            key=lambda metadata: (
                -len(preferred_tags.intersection(metadata.tags)),
                -metadata.selection_priority,
                metadata.provider_id,
            ),
        )
        return ToolSelection(
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=selected.provider_id,
            reason="deterministic tag, priority, and provider-id ordering",
        )

