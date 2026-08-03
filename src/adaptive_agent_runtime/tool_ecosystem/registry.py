"""In-memory Tool Provider registry and metadata management."""

from __future__ import annotations

from adaptive_agent_runtime.tool_ecosystem.contracts import (
    CapabilityCatalog,
    ToolProvider,
)
from adaptive_agent_runtime.tool_ecosystem.errors import (
    CapabilityNotFoundError,
    ProviderAlreadyRegisteredError,
    ProviderNotFoundError,
)
from adaptive_agent_runtime.tool_ecosystem.models import (
    ProviderAvailability,
    ToolProviderMetadata,
)


class InMemoryToolRegistry:
    module_id = "tool.registry.in_memory"

    def __init__(self, capabilities: CapabilityCatalog) -> None:
        self._capabilities = capabilities
        self._metadata: dict[str, ToolProviderMetadata] = {}
        self._providers: dict[str, ToolProvider] = {}

    def register(
        self,
        metadata: ToolProviderMetadata,
        provider: ToolProvider,
    ) -> None:
        if metadata.provider_id in self._providers:
            raise ProviderAlreadyRegisteredError(
                f"provider '{metadata.provider_id}' is already registered"
            )
        if provider.provider_id != metadata.provider_id:
            raise ValueError("provider identity does not match its metadata")
        if self._capabilities.get(metadata.capability_id) is None:
            raise CapabilityNotFoundError(
                f"capability '{metadata.capability_id}' is not registered"
            )
        self._metadata[metadata.provider_id] = metadata
        self._providers[metadata.provider_id] = provider

    def metadata_for(self, provider_id: str) -> ToolProviderMetadata:
        metadata = self._metadata.get(provider_id)
        if metadata is None:
            raise ProviderNotFoundError(f"provider '{provider_id}' was not found")
        return metadata

    def provider_for(self, provider_id: str) -> ToolProvider:
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ProviderNotFoundError(f"provider '{provider_id}' was not found")
        return provider

    def metadata_for_capability(
        self,
        capability_id: str,
    ) -> tuple[ToolProviderMetadata, ...]:
        return tuple(
            sorted(
                (
                    metadata
                    for metadata in self._metadata.values()
                    if metadata.capability_id == capability_id
                ),
                key=lambda metadata: metadata.provider_id,
            )
        )

    def set_availability(
        self,
        provider_id: str,
        availability: ProviderAvailability,
    ) -> ToolProviderMetadata:
        current = self.metadata_for(provider_id)
        updated = current.model_copy(update={"availability": availability})
        self._metadata[provider_id] = updated
        return updated

    def list_metadata(self) -> tuple[ToolProviderMetadata, ...]:
        return tuple(
            self._metadata[provider_id]
            for provider_id in sorted(self._metadata)
        )

