from __future__ import annotations

import unittest

from adaptive_agent_runtime.llm import (
    BackendKind,
    BackendTransportFeatures,
    CapabilityInferenceSettings,
    FakeInferenceBackend,
    InferenceTargetProfile,
    ManagedCognitiveCapabilitySet,
    StructuredOutputLevel,
    compose_managed_capabilities,
    compose_managed_inference,
)


def fake_backend() -> FakeInferenceBackend:
    return FakeInferenceBackend(
        InferenceTargetProfile(
            target_id="fake/composed",
            backend_id="fake",
            backend_kind=BackendKind.LOCAL,
            adapter_version="1",
            model_id="fake-model",
            features=BackendTransportFeatures(
                structured_output=StructuredOutputLevel.JSON_SCHEMA,
            ),
        )
    )


class LLMCompositionTests(unittest.TestCase):
    def test_managed_stack_registers_backend_and_exposes_trace(self) -> None:
        backend = fake_backend()

        composition = compose_managed_inference((backend,))

        self.assertEqual(
            composition.registry.list_profiles(),
            (backend.profile,),
        )
        self.assertEqual(composition.trace.entries(), ())

    def test_capability_composition_applies_per_capability_settings(self) -> None:
        backend = fake_backend()
        composition = compose_managed_inference((backend,))

        capabilities = compose_managed_capabilities(
            composition.gateway,
            settings={
                "artifact_generation": CapabilityInferenceSettings(
                    required_target_id=backend.target_id,
                )
            },
        )

        self.assertIsInstance(capabilities, ManagedCognitiveCapabilitySet)
        self.assertEqual(capabilities.generator.capability_id, "artifact_generation")

    def test_empty_or_unknown_composition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a backend"):
            compose_managed_inference(())
        composition = compose_managed_inference((fake_backend(),))
        with self.assertRaisesRegex(ValueError, "unknown cognitive"):
            compose_managed_capabilities(
                composition.gateway,
                settings={"unknown": CapabilityInferenceSettings()},
            )


if __name__ == "__main__":
    unittest.main()
