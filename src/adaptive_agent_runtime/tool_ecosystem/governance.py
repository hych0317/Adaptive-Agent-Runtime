"""Operational Tool governance: timeout and retry decisions only."""

from __future__ import annotations

from pydantic import Field

from adaptive_agent_runtime.tool_ecosystem.models import (
    ToolAttempt,
    ToolAttemptStatus,
    ToolModel,
)


class RetryPolicy(ToolModel):
    max_retries: int = Field(default=0, ge=0)
    retry_failures: bool = True
    retry_timeouts: bool = True
    delay_seconds: float = Field(default=0.0, ge=0.0)


class ToolExecutionPolicy(ToolModel):
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class OperationalToolGovernor:
    """Decide how to retry execution; authorization belongs to Phase 6."""

    module_id = "tool.governor.operational"

    def should_retry(
        self,
        attempt: ToolAttempt,
        policy: ToolExecutionPolicy,
    ) -> bool:
        if not attempt.retryable:
            return False
        if attempt.attempt_number > policy.retry.max_retries:
            return False
        if attempt.status is ToolAttemptStatus.TIMED_OUT:
            return policy.retry.retry_timeouts
        if attempt.status is ToolAttemptStatus.FAILED:
            return policy.retry.retry_failures
        return False

    def retry_delay(self, policy: ToolExecutionPolicy) -> float:
        return policy.retry.delay_seconds

