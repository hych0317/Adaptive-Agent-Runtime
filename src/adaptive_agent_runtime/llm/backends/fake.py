"""Deterministic, request-addressed inference backend for offline tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias
from uuid import UUID

from adaptive_agent_runtime.llm.errors import (
    AuthenticationRequiredError,
    BackendUnavailableError,
    FakeInferenceResponseNotFoundError,
    InferenceBackendError,
    InferenceContractError,
    UnsupportedFeatureError,
)
from adaptive_agent_runtime.llm.models import (
    BackendAvailability,
    BackendExecutionMode,
    BackendProbeResult,
    InferenceRequest,
    InferenceTargetProfile,
    ModelResponseKind,
    NormalizedModelResponse,
    StructuredOutputLevel,
    ToolIntentMode,
)


FakeInferenceOutcome: TypeAlias = NormalizedModelResponse | InferenceBackendError

_STRUCTURED_OUTPUT_RANK = {
    StructuredOutputLevel.NONE: 0,
    StructuredOutputLevel.JSON_OBJECT: 1,
    StructuredOutputLevel.JSON_SCHEMA: 2,
}


class FakeInferenceBackend:
    """Return preconfigured outcomes by request ID without network access."""

    module_id = "llm.backend.fake"

    def __init__(
        self,
        profile: InferenceTargetProfile,
        responses: Mapping[UUID, FakeInferenceOutcome] | None = None,
        *,
        probe_result: BackendProbeResult | None = None,
    ) -> None:
        if profile.execution_mode is not BackendExecutionMode.INFERENCE:
            raise ValueError("FakeInferenceBackend requires an inference profile")
        if probe_result is not None and probe_result.target_id != profile.target_id:
            raise ValueError("probe result belongs to a different inference target")
        self._profile = profile
        self._responses = dict(responses or {})
        for outcome in self._responses.values():
            self._validate_outcome(outcome)
        self._probe_result = probe_result or BackendProbeResult(
            target_id=profile.target_id,
            availability=BackendAvailability.AVAILABLE,
            runtime_version="fake",
            protocol_version="1",
        )
        self._requests: list[InferenceRequest] = []

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> InferenceTargetProfile:
        return self._profile

    @property
    def recorded_requests(self) -> tuple[InferenceRequest, ...]:
        return tuple(self._requests)

    def set_response(
        self,
        request_id: UUID,
        outcome: FakeInferenceOutcome,
    ) -> None:
        if request_id in self._responses:
            raise ValueError(f"request '{request_id}' already has a fake response")
        self._validate_outcome(outcome)
        self._responses[request_id] = outcome

    async def probe(self) -> BackendProbeResult:
        return self._probe_result

    async def invoke(
        self,
        request: InferenceRequest,
    ) -> NormalizedModelResponse:
        self._requests.append(request)
        self._ensure_available()
        self._ensure_supported(request)
        try:
            outcome = self._responses[request.request_id]
        except KeyError as exc:
            raise FakeInferenceResponseNotFoundError(
                self.target_id,
                request.request_id,
            ) from exc
        if isinstance(outcome, InferenceBackendError):
            if outcome.target_id != self.target_id:
                raise InferenceContractError(
                    self.target_id,
                    (
                        "fake inference error references a different target: "
                        f"expected '{self.target_id}', got '{outcome.target_id}'"
                    ),
                )
            raise outcome
        if outcome.request_id != request.request_id:
            raise InferenceContractError(
                self.target_id,
                (
                    "fake inference response references a different request: "
                    f"expected '{request.request_id}', got '{outcome.request_id}'"
                ),
            )
        if outcome.target_id != self.target_id:
            raise InferenceContractError(
                self.target_id,
                (
                    "fake inference response references a different target: "
                    f"expected '{self.target_id}', got '{outcome.target_id}'"
                ),
            )
        if (
            self.profile.model_id is not None
            and outcome.model_id != self.profile.model_id
        ):
            raise InferenceContractError(
                self.target_id,
                (
                    "fake inference response references a different model: "
                    f"expected '{self.profile.model_id}', got '{outcome.model_id}'"
                ),
            )
        self._ensure_response_allowed(request, outcome)
        return outcome

    def _ensure_available(self) -> None:
        availability = self._probe_result.availability
        if availability is BackendAvailability.AUTH_REQUIRED:
            raise AuthenticationRequiredError(self.target_id)
        if availability is BackendAvailability.UNAVAILABLE:
            raise BackendUnavailableError(
                self.target_id,
                reason="fake probe reports unavailable",
                retryable=False,
            )

    def _ensure_supported(self, request: InferenceRequest) -> None:
        required = request.requirements
        supported = self._profile.features
        if (
            _STRUCTURED_OUTPUT_RANK[required.required_structured_output]
            > _STRUCTURED_OUTPUT_RANK[supported.structured_output]
        ):
            raise UnsupportedFeatureError(self.target_id, "structured_output")
        if required.tool_intent is not ToolIntentMode.DISABLED:
            if not supported.tool_intent:
                raise UnsupportedFeatureError(self.target_id, "tool_intent")
        requested_tokens = required.max_output_tokens
        supported_tokens = self._profile.limits.max_output_tokens
        if (
            requested_tokens is not None
            and supported_tokens is not None
            and requested_tokens > supported_tokens
        ):
            raise UnsupportedFeatureError(self.target_id, "max_output_tokens")

    def _ensure_response_allowed(
        self,
        request: InferenceRequest,
        response: NormalizedModelResponse,
    ) -> None:
        mode = request.requirements.tool_intent
        if (
            response.kind is ModelResponseKind.TOOL_INTENT
            and mode is ToolIntentMode.DISABLED
        ):
            raise InferenceContractError(
                self.target_id,
                "backend returned tool intent when it was disabled",
            )
        if (
            response.kind is ModelResponseKind.OUTPUT
            and mode is ToolIntentMode.REQUIRED
        ):
            raise InferenceContractError(
                self.target_id,
                "backend returned output when tool intent was required",
            )
        eligible = {item.capability_id for item in request.eligible_tools}
        unknown = {
            intent.capability_id
            for intent in response.tool_intents
            if intent.capability_id not in eligible
        }
        if unknown:
            raise InferenceContractError(
                self.target_id,
                "backend proposed ineligible tool capabilities: "
                + ", ".join(sorted(unknown)),
            )

    @staticmethod
    def _validate_outcome(outcome: FakeInferenceOutcome) -> None:
        if not isinstance(outcome, (NormalizedModelResponse, InferenceBackendError)):
            raise TypeError(
                "fake outcomes must be NormalizedModelResponse or "
                "InferenceBackendError"
            )
