from __future__ import annotations

import unittest

from pydantic import ValidationError

from adaptive_agent_runtime.llm import (
    BackendKind,
    BackendMetering,
    InferenceContractError,
    InferenceRequest,
    InferenceRequirements,
    InferenceResponseValidator,
    InferenceTargetProfile,
    InferenceUsage,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
    ProviderNeutralResponseValidator,
    ResponseSchemaValidationError,
    StructuredOutputLevel,
)


def profile(*, reports_usage: bool = False) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="fake/default",
        backend_id="fake",
        backend_kind=BackendKind.LOCAL,
        adapter_version="1",
        model_id="fake-model",
        metering=BackendMetering(reports_token_usage=reports_usage),
    )


class ResponseValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = ProviderNeutralResponseValidator()
        self.assertIsInstance(self.validator, InferenceResponseValidator)

    def test_request_rejects_invalid_json_schema(self) -> None:
        with self.assertRaisesRegex(ValidationError, "invalid response schema"):
            InferenceRequest(
                cognitive_capability_id="generation",
                input="write",
                response_schema={"type": "not-a-json-schema-type"},
            )

    def test_json_schema_is_validated_locally(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            response_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
        )
        invalid = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.OUTPUT,
            output={"count": "one"},
        )
        valid = invalid.model_copy(update={"output": {"count": 1}})

        with self.assertRaisesRegex(
            ResponseSchemaValidationError,
            "is not of type 'integer'",
        ):
            self.validator.validate(request, profile(), invalid)
        self.assertIs(self.validator.validate(request, profile(), valid), valid)

    def test_json_object_requirement_rejects_scalar_output(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="judge",
            input="judge",
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_OBJECT
            ),
        )
        response = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.OUTPUT,
            output="not an object",
        )

        with self.assertRaisesRegex(
            ResponseSchemaValidationError,
            "expected a JSON object",
        ):
            self.validator.validate(request, profile(), response)

    def test_target_metering_claim_requires_complete_usage(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        response = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.OUTPUT,
            output="done",
            usage=InferenceUsage(input_tokens=2),
        )

        with self.assertRaisesRegex(
            InferenceContractError,
            "promised token usage",
        ):
            self.validator.validate(request, profile(reports_usage=True), response)

    def test_finish_reason_must_match_response_kind(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        response = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.OUTPUT,
            output="done",
            finish_reason=NormalizedFinishReason.TOOL_INTENT,
        )

        with self.assertRaisesRegex(
            InferenceContractError,
            "tool-intent finish reason",
        ):
            self.validator.validate(request, profile(), response)


if __name__ == "__main__":
    unittest.main()
