"""Managed inference execution with routing, fallback, budgets, and trace."""

from __future__ import annotations

from asyncio import timeout as async_timeout
from collections.abc import Callable
from time import monotonic

from adaptive_agent_runtime.llm.adapters.contracts import (
    InferenceResponseValidator,
)
from adaptive_agent_runtime.llm.errors import (
    AuthenticationRequiredError,
    BackendProtocolError,
    BackendUnavailableError,
    InferenceBackendError,
    InferenceContractError,
    InferenceExecutionBudgetError,
    InferenceResponseBudgetError,
    NoEligibleInferenceTargetError,
)
from adaptive_agent_runtime.llm.gateway.contracts import (
    InferenceBackendRegistry,
    InferenceGatewayTraceSink,
    InferenceRouter,
)
from adaptive_agent_runtime.llm.gateway.models import (
    InferenceExecutionBudget,
    InferenceGatewayPolicy,
    InferenceGatewayTraceEvent,
    InferenceGatewayTraceEventKind,
)
from adaptive_agent_runtime.llm.models import (
    BackendAvailability,
    BackendProbeResult,
    InferenceRequest,
    InferenceTargetProfile,
    NormalizedModelResponse,
)


class ManagedInferenceGateway:
    """Execute bounded inference while Runtime retains routing authority."""

    module_id = "llm.inference_gateway.managed"

    def __init__(
        self,
        *,
        registry: InferenceBackendRegistry,
        router: InferenceRouter,
        response_validator: InferenceResponseValidator,
        trace_sink: InferenceGatewayTraceSink,
        probe_cache_ttl_seconds: float = 30.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if probe_cache_ttl_seconds <= 0:
            raise ValueError("probe cache TTL must be positive")
        self._registry = registry
        self._router = router
        self._response_validator = response_validator
        self._trace_sink = trace_sink
        self._probe_cache_ttl_seconds = probe_cache_ttl_seconds
        self._clock = clock
        self._probe_cache: dict[str, tuple[float, BackendProbeResult]] = {}

    async def probe_target(
        self,
        target_id: str,
        *,
        force: bool = False,
    ) -> BackendProbeResult:
        now = self._clock()
        cached = self._probe_cache.get(target_id)
        if not force and cached is not None and cached[0] > now:
            return cached[1]
        backend = self._registry.backend_for(target_id)
        profile = backend.profile
        try:
            probe = await backend.probe()
            self._validate_probe(profile, probe)
        except InferenceBackendError:
            self._probe_cache.pop(target_id, None)
            raise
        except Exception as exc:
            self._probe_cache.pop(target_id, None)
            raise BackendProtocolError(
                profile.target_id,
                f"backend probe raised unnormalized {type(exc).__name__}",
            ) from exc
        if probe.availability is BackendAvailability.AVAILABLE:
            self._probe_cache[target_id] = (
                now + self._probe_cache_ttl_seconds,
                probe,
            )
        else:
            self._probe_cache.pop(target_id, None)
        return probe

    async def execute(
        self,
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
    ) -> NormalizedModelResponse:
        execution_started_at = self._clock()
        candidates = self._router.candidates(
            request,
            policy,
            self._registry.list_profiles(),
        )
        await self._trace_sink.record(
            InferenceGatewayTraceEvent(
                request_id=request.request_id,
                correlation=request.correlation,
                trace_attributes=request.trace_attributes,
                kind=InferenceGatewayTraceEventKind.ROUTING_COMPLETED,
                candidate_target_ids=tuple(
                    profile.target_id for profile in candidates
                ),
            )
        )
        if not candidates:
            raise NoEligibleInferenceTargetError(request.request_id)

        queue = list(candidates)
        last_error: InferenceBackendError | None = None
        attempt = 0
        while queue and attempt < policy.budget.max_attempts:
            profile = queue.pop(0)
            attempt += 1
            await self._trace_sink.record(
                InferenceGatewayTraceEvent(
                    request_id=request.request_id,
                    correlation=request.correlation,
                    trace_attributes=request.trace_attributes,
                    kind=InferenceGatewayTraceEventKind.ATTEMPT_STARTED,
                    target_id=profile.target_id,
                    attempt_number=attempt,
                )
            )
            try:
                response = await self._invoke_with_budget(
                    request,
                    profile,
                    policy.budget,
                    execution_started_at,
                )
                validated = self._response_validator.validate(
                    request,
                    profile,
                    response,
                )
                self._enforce_budget(policy.budget, validated)
            except (
                InferenceExecutionBudgetError,
                InferenceResponseBudgetError,
            ) as exc:
                await self._trace_sink.record(
                    InferenceGatewayTraceEvent(
                        request_id=request.request_id,
                        correlation=request.correlation,
                        trace_attributes=request.trace_attributes,
                        kind=InferenceGatewayTraceEventKind.BUDGET_REJECTED,
                        target_id=profile.target_id,
                        attempt_number=attempt,
                        message=str(exc),
                    )
                )
                raise
            except InferenceBackendError as exc:
                last_error = exc
                await self._trace_sink.record(
                    InferenceGatewayTraceEvent(
                        request_id=request.request_id,
                        correlation=request.correlation,
                        trace_attributes=request.trace_attributes,
                        kind=InferenceGatewayTraceEventKind.ATTEMPT_FAILED,
                        target_id=profile.target_id,
                        attempt_number=attempt,
                        failure_code=exc.code,
                        message=str(exc),
                    )
                )
                if (
                    exc.retryable
                    and policy.retry.retry_same_target
                    and attempt < policy.budget.max_attempts
                ):
                    queue.append(profile)
                can_fallback = (
                    exc.code in policy.retry.fallback_failure_codes
                )
                if can_fallback and queue:
                    continue
                if exc.retryable and policy.retry.retry_same_target:
                    queue = [
                        candidate
                        for candidate in queue
                        if candidate.target_id == profile.target_id
                    ]
                    if queue:
                        continue
                raise
            else:
                await self._trace_sink.record(
                    InferenceGatewayTraceEvent(
                        request_id=request.request_id,
                        correlation=request.correlation,
                        trace_attributes=request.trace_attributes,
                        kind=InferenceGatewayTraceEventKind.ATTEMPT_SUCCEEDED,
                        target_id=profile.target_id,
                        attempt_number=attempt,
                        usage=validated.usage,
                    )
                )
                return validated

        if last_error is not None:
            raise last_error
        raise NoEligibleInferenceTargetError(request.request_id)

    async def _invoke_with_budget(
        self,
        request: InferenceRequest,
        profile: InferenceTargetProfile,
        budget: InferenceExecutionBudget,
        execution_started_at: float,
    ) -> NormalizedModelResponse:
        if budget.max_elapsed_seconds is None:
            return await self._invoke(request, profile)
        remaining = budget.max_elapsed_seconds - (
            self._clock() - execution_started_at
        )
        if remaining <= 0:
            raise InferenceExecutionBudgetError(
                "elapsed-time limit was reached before backend invocation"
            )
        try:
            async with async_timeout(remaining):
                return await self._invoke(request, profile)
        except TimeoutError as exc:
            raise InferenceExecutionBudgetError(
                f"elapsed-time limit {budget.max_elapsed_seconds} seconds"
            ) from exc

    async def _invoke(
        self,
        request: InferenceRequest,
        profile: InferenceTargetProfile,
    ) -> NormalizedModelResponse:
        backend = self._registry.backend_for(profile.target_id)
        try:
            probe = await self.probe_target(profile.target_id)
            if probe.availability is BackendAvailability.AUTH_REQUIRED:
                raise AuthenticationRequiredError(profile.target_id)
            if probe.availability is BackendAvailability.UNAVAILABLE:
                raise BackendUnavailableError(
                    profile.target_id,
                    reason="backend probe reports unavailable",
                )
            return await backend.invoke(request)
        except InferenceBackendError:
            raise
        except Exception as exc:
            raise BackendProtocolError(
                profile.target_id,
                f"backend raised unnormalized {type(exc).__name__}",
            ) from exc

    @staticmethod
    def _validate_probe(
        profile: InferenceTargetProfile,
        probe: BackendProbeResult,
    ) -> None:
        if probe.target_id != profile.target_id:
            raise InferenceContractError(
                profile.target_id,
                "probe references a different target",
            )

    @staticmethod
    def _enforce_budget(
        budget: InferenceExecutionBudget,
        response: NormalizedModelResponse,
    ) -> None:
        usage = response.usage
        if budget.max_total_tokens is not None:
            if usage.total_tokens is None:
                raise InferenceResponseBudgetError("token usage is missing")
            if usage.total_tokens > budget.max_total_tokens:
                raise InferenceResponseBudgetError(
                    f"{usage.total_tokens} tokens exceed limit "
                    f"{budget.max_total_tokens}"
                )
        if budget.max_response_cost is not None:
            if usage.monetary_cost is None or usage.currency is None:
                raise InferenceResponseBudgetError("monetary usage is missing")
            if usage.currency != budget.currency:
                raise InferenceResponseBudgetError(
                    f"currency '{usage.currency}' does not match "
                    f"budget currency '{budget.currency}'"
                )
            if usage.monetary_cost > budget.max_response_cost:
                raise InferenceResponseBudgetError(
                    f"cost {usage.monetary_cost} exceeds limit "
                    f"{budget.max_response_cost} {budget.currency}"
                )
