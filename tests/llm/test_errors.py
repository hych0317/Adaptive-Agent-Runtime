from __future__ import annotations

import unittest
from uuid import uuid4

from adaptive_agent_runtime.llm import (
    AuthenticationRequiredError,
    BackendProtocolError,
    BackendUnavailableError,
    FakeInferenceResponseNotFoundError,
    InferenceBackendError,
    InferenceContractError,
    InferenceFailureCode,
    LLMIntegrationError,
    UnsupportedFeatureError,
)


class InferenceErrorTests(unittest.TestCase):
    def test_authentication_error_is_normalized_and_not_retryable(self) -> None:
        error = AuthenticationRequiredError("codex/inference")

        self.assertIsInstance(error, InferenceBackendError)
        self.assertIsInstance(error, LLMIntegrationError)
        self.assertEqual(error.code, InferenceFailureCode.AUTH_REQUIRED)
        self.assertEqual(error.target_id, "codex/inference")
        self.assertFalse(error.retryable)

    def test_unavailable_error_preserves_retry_policy(self) -> None:
        transient = BackendUnavailableError("api/model", retryable=True)
        permanent = BackendUnavailableError("api/model", retryable=False)

        self.assertTrue(transient.retryable)
        self.assertFalse(permanent.retryable)
        self.assertEqual(
            transient.code,
            InferenceFailureCode.BACKEND_UNAVAILABLE,
        )

    def test_unsupported_error_preserves_feature(self) -> None:
        error = UnsupportedFeatureError("local/model", "json_schema")

        self.assertEqual(error.feature, "json_schema")
        self.assertEqual(error.code, InferenceFailureCode.UNSUPPORTED_FEATURE)
        self.assertFalse(error.retryable)

    def test_contract_and_fake_errors_are_protocol_failures(self) -> None:
        errors = (
            BackendProtocolError("fake", "bad protocol"),
            InferenceContractError("fake", "wrong identity"),
            FakeInferenceResponseNotFoundError("fake", uuid4()),
        )

        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(error.code, InferenceFailureCode.PROTOCOL_ERROR)
                self.assertFalse(error.retryable)
                self.assertIn("fake", str(error))


if __name__ == "__main__":
    unittest.main()
