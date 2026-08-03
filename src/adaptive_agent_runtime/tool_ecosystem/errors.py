"""Errors owned by the Tool Ecosystem."""


class ToolEcosystemError(Exception):
    """Base exception for provider-neutral Tool Ecosystem failures."""


class CapabilityAlreadyRegisteredError(ToolEcosystemError):
    """Raised when a capability identifier is registered twice."""


class CapabilityNotFoundError(ToolEcosystemError):
    """Raised when a requested capability is unknown."""


class ProviderAlreadyRegisteredError(ToolEcosystemError):
    """Raised when a provider identifier is registered twice."""


class ProviderNotFoundError(ToolEcosystemError):
    """Raised when a provider identifier is unknown."""


class ProviderUnavailableError(ToolEcosystemError):
    """Raised when matching providers exist but none are available."""


class ToolSelectionError(ToolEcosystemError):
    """Raised when a deterministic provider selection cannot be made."""


class ToolIntegrationError(ToolEcosystemError):
    """Raised when a TaskNode has no external capability request mapping."""
