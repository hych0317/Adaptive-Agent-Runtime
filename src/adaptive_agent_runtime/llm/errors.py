"""Normalized failures raised by the LLM integration layer."""

from __future__ import annotations

from enum import StrEnum


class InferenceFailureCode(StrEnum):
    AUTH_REQUIRED = "auth_required"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    CONTEXT_OVERFLOW = "context_overflow"
    MALFORMED_OUTPUT = "malformed_output"
    SCHEMA_VIOLATION = "schema_violation"
    TOOL_CALL_REJECTED = "tool_call_rejected"
    PROCESS_FAILED = "process_failed"
    PROTOCOL_ERROR = "protocol_error"


class LLMIntegrationError(Exception):
    """Base exception for provider-neutral LLM integration failures."""


class AgentBackendError(LLMIntegrationError):
    """Base failure for an isolated autonomous Agent backend."""

    def __init__(self, target_id: str, message: str) -> None:
        self.target_id = target_id
        super().__init__(message)


class AgentExecutionPolicyError(AgentBackendError):
    def __init__(self, target_id: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            target_id,
            f"Agent target '{target_id}' rejected execution policy: {reason}",
        )


class AgentExecutionProtocolError(AgentBackendError):
    def __init__(self, target_id: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            target_id,
            f"Agent target '{target_id}' protocol error: {reason}",
        )


class AgentExecutionBudgetError(AgentBackendError):
    def __init__(self, target_id: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            target_id,
            f"Agent target '{target_id}' exceeded execution budget: {reason}",
        )


class AgentProcessFailedError(AgentBackendError):
    def __init__(self, target_id: str, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__(
            target_id,
            f"Agent target '{target_id}' process failed with exit code {exit_code}",
        )


class AgentExecutionTimeoutError(AgentBackendError):
    def __init__(self, target_id: str) -> None:
        super().__init__(
            target_id,
            f"Agent target '{target_id}' timed out",
        )


class AgentBackendUnavailableError(AgentBackendError):
    def __init__(self, target_id: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            target_id,
            f"Agent target '{target_id}' is unavailable: {reason}",
        )


class AgentAuthenticationRequiredError(AgentBackendError):
    def __init__(self, target_id: str) -> None:
        super().__init__(
            target_id,
            f"Agent target '{target_id}' requires authentication",
        )


class CapabilityResultValidationError(LLMIntegrationError):
    """An untrusted cognitive draft conflicts with its originating request."""

    def __init__(
        self,
        capability_id: str,
        violations: tuple[str, ...],
    ) -> None:
        if not violations:
            raise ValueError("capability validation errors require violations")
        self.capability_id = capability_id
        self.violations = violations
        super().__init__(
            f"capability '{capability_id}' returned an invalid draft: "
            + "; ".join(violations)
        )


class CapabilityExecutionPolicyError(LLMIntegrationError):
    def __init__(self, capability_id: str, reason: str) -> None:
        self.capability_id = capability_id
        self.reason = reason
        super().__init__(
            f"capability '{capability_id}' execution policy is invalid: {reason}"
        )


class ContextAdapterError(LLMIntegrationError):
    """Base failure at the Runtime-to-model information boundary."""


class ContextEgressDeniedError(ContextAdapterError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"context egress denied: {reason}")


class ContextProjectionBudgetError(ContextAdapterError):
    def __init__(self, required_tokens: int, available_tokens: int) -> None:
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        super().__init__(
            "required context needs "
            f"{required_tokens} tokens but only {available_tokens} are available"
        )


class InferenceRoutingError(LLMIntegrationError):
    """No registered inference target can safely satisfy a request."""


class InferenceTargetAlreadyRegisteredError(InferenceRoutingError):
    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        super().__init__(f"inference target '{target_id}' is already registered")


class InferenceTargetNotFoundError(InferenceRoutingError):
    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        super().__init__(f"inference target '{target_id}' was not found")


class NoEligibleInferenceTargetError(InferenceRoutingError):
    def __init__(self, request_id: object) -> None:
        self.request_id = request_id
        super().__init__(
            f"no inference target can satisfy request '{request_id}'"
        )


class InferenceResponseBudgetError(LLMIntegrationError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"inference response budget rejected: {reason}")


class InferenceExecutionBudgetError(LLMIntegrationError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"inference execution budget exhausted: {reason}")


class InferenceBackendError(LLMIntegrationError):
    """A normalized failure produced while using one inference target."""

    def __init__(
        self,
        target_id: str,
        *,
        code: InferenceFailureCode,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.target_id = target_id
        self.code = code
        self.retryable = retryable


class ResponseSchemaValidationError(InferenceBackendError):
    def __init__(self, target_id: str, reason: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.SCHEMA_VIOLATION,
            message=(
                f"inference target '{target_id}' returned schema-invalid output: "
                f"{reason}"
            ),
            retryable=False,
        )


class RateLimitedError(InferenceBackendError):
    def __init__(self, target_id: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.RATE_LIMITED,
            message=f"inference target '{target_id}' is rate limited",
            retryable=True,
        )


class InferenceTimeoutError(InferenceBackendError):
    def __init__(self, target_id: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.TIMEOUT,
            message=f"inference target '{target_id}' timed out",
            retryable=True,
        )


class ContextOverflowError(InferenceBackendError):
    def __init__(self, target_id: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.CONTEXT_OVERFLOW,
            message=f"inference target '{target_id}' rejected context length",
            retryable=False,
        )


class MalformedModelOutputError(InferenceBackendError):
    def __init__(self, target_id: str, reason: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.MALFORMED_OUTPUT,
            message=f"inference target '{target_id}' returned malformed output: {reason}",
            retryable=False,
        )


class BackendRequestRejectedError(InferenceBackendError):
    def __init__(self, target_id: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(
            target_id,
            code=InferenceFailureCode.PROCESS_FAILED,
            message=(
                f"inference target '{target_id}' rejected the request "
                f"with HTTP {status_code}"
            ),
            retryable=False,
        )


class BackendProcessFailedError(InferenceBackendError):
    def __init__(self, target_id: str, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__(
            target_id,
            code=InferenceFailureCode.PROCESS_FAILED,
            message=(
                f"inference target '{target_id}' process failed "
                f"with exit code {exit_code}"
            ),
            retryable=False,
        )


class AuthenticationRequiredError(InferenceBackendError):
    def __init__(self, target_id: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.AUTH_REQUIRED,
            message=f"inference target '{target_id}' requires authentication",
            retryable=False,
        )


class BackendUnavailableError(InferenceBackendError):
    def __init__(
        self,
        target_id: str,
        *,
        reason: str = "backend is unavailable",
        retryable: bool = True,
    ) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.BACKEND_UNAVAILABLE,
            message=f"inference target '{target_id}' is unavailable: {reason}",
            retryable=retryable,
        )


class UnsupportedFeatureError(InferenceBackendError):
    def __init__(self, target_id: str, feature: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.UNSUPPORTED_FEATURE,
            message=(
                f"inference target '{target_id}' does not support required "
                f"feature '{feature}'"
            ),
            retryable=False,
        )
        self.feature = feature


class BackendProtocolError(InferenceBackendError):
    def __init__(self, target_id: str, message: str) -> None:
        super().__init__(
            target_id,
            code=InferenceFailureCode.PROTOCOL_ERROR,
            message=f"inference target '{target_id}' protocol error: {message}",
            retryable=False,
        )


class InferenceContractError(BackendProtocolError):
    """Raised when a Backend violates request/response identity."""


class InferenceUsageAccountingError(InferenceContractError):
    """Raised when a Backend omits usage it promised to report."""


class FakeInferenceResponseNotFoundError(BackendProtocolError):
    def __init__(self, target_id: str, request_id: object) -> None:
        super().__init__(
            target_id,
            f"no fake response for request '{request_id}'",
        )
