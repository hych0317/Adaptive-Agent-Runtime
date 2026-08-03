from __future__ import annotations

import unittest
from typing import Any, cast

from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    CapabilityAlreadyRegisteredError,
    CapabilityRequirement,
    CapabilityResolver,
    DeterministicToolSelector,
    ExactCapabilityMatcher,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    ProviderAlreadyRegisteredError,
    ProviderAvailability,
    ProviderUnavailableError,
    ToolInvocation,
    ToolProviderMetadata,
    ToolProviderResult,
    ToolSelectionContext,
)


class StubProvider:
    module_id = "test.provider.stub"

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.calls = 0

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        self.calls += 1
        return ToolProviderResult.ok(
            output=invocation.model_dump(mode="json")["arguments"]
        )


def capability() -> Capability:
    return Capability(
        capability_id="financial_information",
        name="Financial Information",
        description="Retrieve provider-neutral financial information.",
        tags=("finance", "data"),
    )


def metadata(
    provider_id: str,
    *,
    tags: tuple[str, ...] = (),
    priority: int = 0,
    availability: ProviderAvailability = ProviderAvailability.AVAILABLE,
    schema: dict[str, Any] | None = None,
) -> ToolProviderMetadata:
    return ToolProviderMetadata(
        provider_id=provider_id,
        name=f"Provider {provider_id}",
        capability_id="financial_information",
        description=f"Financial data from {provider_id}.",
        input_schema=schema or {"type": "object"},
        availability=availability,
        tags=tags,
        selection_priority=priority,
    )


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = InMemoryCapabilityCatalog()
        self.catalog.register(capability())
        self.registry = InMemoryToolRegistry(self.catalog)
        self.resolver = CapabilityResolver(
            catalog=self.catalog,
            registry=self.registry,
            matcher=ExactCapabilityMatcher(),
        )

    def test_capability_registration_and_query(self) -> None:
        registered = self.catalog.get("financial_information")

        self.assertEqual(registered, capability())
        self.assertEqual(self.catalog.list_all(), (registered,))
        with self.assertRaises(CapabilityAlreadyRegisteredError):
            self.catalog.register(capability())

    def test_provider_metadata_is_registered_as_an_immutable_snapshot(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "required": ["symbol"],
        }
        provider = StubProvider("primary")
        descriptor = metadata("primary", schema=schema)
        self.registry.register(descriptor, provider)
        schema["required"][0] = "changed"

        stored = self.registry.metadata_for("primary")
        stored_schema = cast(dict[str, Any], stored.input_schema)
        self.assertEqual(stored.name, "Provider primary")
        self.assertEqual(stored.capability_id, "financial_information")
        self.assertEqual(stored_schema["required"], ("symbol",))
        with self.assertRaises(TypeError):
            stored_schema["required"][0] = "mutation"
        self.assertEqual(provider.calls, 0)

    def test_one_capability_keeps_multiple_providers(self) -> None:
        second = StubProvider("second")
        first = StubProvider("first")
        self.registry.register(metadata("second"), second)
        self.registry.register(metadata("first"), first)

        candidates = self.registry.metadata_for_capability(
            "financial_information"
        )

        self.assertEqual(
            tuple(item.provider_id for item in candidates),
            ("first", "second"),
        )

    def test_runtime_capability_filtering_uses_explicit_constraints(self) -> None:
        self.registry.register(
            metadata("primary", tags=("primary", "cn")),
            StubProvider("primary"),
        )
        self.registry.register(
            metadata("secondary", tags=("secondary", "cn")),
            StubProvider("secondary"),
        )
        requirement = CapabilityRequirement(
            capability_id="financial_information",
            required_capability_tags=("finance",),
            required_provider_tags=("primary",),
        )

        candidates = self.resolver.candidates(requirement)

        self.assertEqual(
            tuple(item.provider_id for item in candidates),
            ("primary",),
        )
        missing = self.resolver.candidates(
            CapabilityRequirement(capability_id="unknown")
        )
        self.assertEqual(missing, ())

    def test_deterministic_selection_is_stable_across_registration_order(self) -> None:
        self.registry.register(
            metadata("backup", tags=("fast",), priority=10),
            StubProvider("backup"),
        )
        self.registry.register(
            metadata("preferred", tags=("audited",), priority=1),
            StubProvider("preferred"),
        )
        requirement = CapabilityRequirement(
            capability_id="financial_information",
            preferred_provider_tags=("audited",),
        )
        selector = DeterministicToolSelector()

        first = selector.select(
            requirement,
            self.resolver.candidates(requirement),
            ToolSelectionContext(),
        )
        second = selector.select(
            requirement,
            tuple(reversed(self.resolver.candidates(requirement))),
            ToolSelectionContext(),
        )

        self.assertEqual(first.provider_id, "preferred")
        self.assertEqual(second.provider_id, "preferred")

    def test_selector_skips_unavailable_provider_and_reports_all_unavailable(self) -> None:
        self.registry.register(
            metadata("preferred", priority=10),
            StubProvider("preferred"),
        )
        self.registry.register(
            metadata("backup", priority=1),
            StubProvider("backup"),
        )
        requirement = CapabilityRequirement(
            capability_id="financial_information"
        )
        selector = DeterministicToolSelector()
        self.registry.set_availability(
            "preferred",
            ProviderAvailability.UNAVAILABLE,
        )

        selected = selector.select(
            requirement,
            self.resolver.candidates(requirement),
            ToolSelectionContext(),
        )
        self.assertEqual(selected.provider_id, "backup")

        self.registry.set_availability(
            "backup",
            ProviderAvailability.UNAVAILABLE,
        )
        with self.assertRaises(ProviderUnavailableError):
            selector.select(
                requirement,
                self.resolver.candidates(requirement),
                ToolSelectionContext(),
            )

    def test_duplicate_provider_is_rejected_without_overwrite(self) -> None:
        first = StubProvider("same")
        self.registry.register(metadata("same"), first)

        with self.assertRaises(ProviderAlreadyRegisteredError):
            self.registry.register(metadata("same"), StubProvider("same"))

        self.assertIs(self.registry.provider_for("same"), first)
        self.assertEqual(len(self.registry.list_metadata()), 1)


if __name__ == "__main__":
    unittest.main()
