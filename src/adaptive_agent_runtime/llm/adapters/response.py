"""Provider-neutral validation after backend response normalization."""

from __future__ import annotations

from collections.abc import Mapping

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import (  # type: ignore[import-untyped]
    ValidationError as JsonSchemaValidationError,
)

from adaptive_agent_runtime.llm.errors import (
    InferenceContractError,
    InferenceUsageAccountingError,
    ResponseSchemaValidationError,
)
from adaptive_agent_runtime.llm.models import (
    InferenceRequest,
    InferenceTargetProfile,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
    StructuredOutputLevel,
    ToolIntentMode,
)


class ProviderNeutralResponseValidator:
    """Enforce identity, turn-state, metering, and JSON Schema contracts."""

    module_id = "llm.response_validator.provider_neutral"

    def validate(
        self,
        request: InferenceRequest,
        target: InferenceTargetProfile,
        response: NormalizedModelResponse,
    ) -> NormalizedModelResponse:
        self._validate_identity(request, target, response)
        self._validate_turn(request, target, response)
        self._validate_usage(target, response)
        self._validate_structured_output(request, target, response)
        return response

    @staticmethod
    def _validate_identity(
        request: InferenceRequest,
        target: InferenceTargetProfile,
        response: NormalizedModelResponse,
    ) -> None:
        if response.request_id != request.request_id:
            raise InferenceContractError(
                target.target_id,
                "response references a different request",
            )
        if response.target_id != target.target_id:
            raise InferenceContractError(
                target.target_id,
                "response references a different target",
            )
        if target.model_id is not None and response.model_id != target.model_id:
            raise InferenceContractError(
                target.target_id,
                "response references a different model",
            )

    @staticmethod
    def _validate_turn(
        request: InferenceRequest,
        target: InferenceTargetProfile,
        response: NormalizedModelResponse,
    ) -> None:
        mode = request.requirements.tool_intent
        if response.kind is ModelResponseKind.TOOL_INTENT:
            if mode is ToolIntentMode.DISABLED:
                raise InferenceContractError(
                    target.target_id,
                    "backend returned tool intent when it was disabled",
                )
            if (
                response.finish_reason is not None
                and response.finish_reason is not NormalizedFinishReason.TOOL_INTENT
            ):
                raise InferenceContractError(
                    target.target_id,
                    "tool-intent response has an incompatible finish reason",
                )
        else:
            if mode is ToolIntentMode.REQUIRED:
                raise InferenceContractError(
                    target.target_id,
                    "backend returned output when tool intent was required",
                )
            if response.finish_reason is NormalizedFinishReason.TOOL_INTENT:
                raise InferenceContractError(
                    target.target_id,
                    "output response cannot use a tool-intent finish reason",
                )
        eligible = {tool.capability_id for tool in request.eligible_tools}
        unknown = sorted(
            {
                intent.capability_id
                for intent in response.tool_intents
                if intent.capability_id not in eligible
            }
        )
        if unknown:
            raise InferenceContractError(
                target.target_id,
                "backend proposed ineligible tool capabilities: "
                + ", ".join(unknown),
            )

    @staticmethod
    def _validate_usage(
        target: InferenceTargetProfile,
        response: NormalizedModelResponse,
    ) -> None:
        usage = response.usage
        if target.metering.reports_token_usage and (
            usage.input_tokens is None
            or usage.output_tokens is None
            or usage.total_tokens is None
        ):
            raise InferenceUsageAccountingError(
                target.target_id,
                "target promised token usage but the response omitted it",
            )
        if target.metering.reports_monetary_cost and (
            usage.monetary_cost is None or usage.currency is None
        ):
            raise InferenceUsageAccountingError(
                target.target_id,
                "target promised monetary cost but the response omitted it",
            )

    @staticmethod
    def _validate_structured_output(
        request: InferenceRequest,
        target: InferenceTargetProfile,
        response: NormalizedModelResponse,
    ) -> None:
        if response.kind is not ModelResponseKind.OUTPUT:
            return
        required = request.requirements.required_structured_output
        if (
            required is StructuredOutputLevel.JSON_OBJECT
            and not isinstance(response.output, Mapping)
        ):
            raise ResponseSchemaValidationError(
                target.target_id,
                "expected a JSON object",
            )
        if request.response_schema is None:
            return
        dumped = response.model_dump(mode="json")
        schema = request.model_dump(mode="json")["response_schema"]
        try:
            Draft202012Validator(schema).validate(dumped["output"])
        except JsonSchemaValidationError as exc:
            raise ResponseSchemaValidationError(
                target.target_id,
                exc.message,
            ) from exc
