"""Deterministic capability negotiation over registered inference profiles."""

from __future__ import annotations

from typing import Sequence

from adaptive_agent_runtime.llm.gateway.models import InferenceGatewayPolicy
from adaptive_agent_runtime.llm.models import (
    InferenceRequest,
    InferenceTargetProfile,
    StructuredOutputLevel,
    ToolIntentMode,
)


_STRUCTURED_OUTPUT_RANK = {
    StructuredOutputLevel.NONE: 0,
    StructuredOutputLevel.JSON_OBJECT: 1,
    StructuredOutputLevel.JSON_SCHEMA: 2,
}


class DeterministicInferenceRouter:
    module_id = "llm.inference_router.deterministic"

    def candidates(
        self,
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
        profiles: Sequence[InferenceTargetProfile],
    ) -> tuple[InferenceTargetProfile, ...]:
        routing = policy.routing
        eligible = tuple(
            profile
            for profile in profiles
            if self._is_eligible(request, policy, profile)
        )
        preferred_ids = {
            target_id: index
            for index, target_id in enumerate(routing.preferred_target_ids)
        }
        preferred_tags = set(routing.preferred_target_tags)
        return tuple(
            sorted(
                eligible,
                key=lambda profile: (
                    preferred_ids.get(profile.target_id, len(preferred_ids)),
                    -len(preferred_tags.intersection(profile.tags)),
                    profile.target_id,
                ),
            )
        )

    @staticmethod
    def _is_eligible(
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
        profile: InferenceTargetProfile,
    ) -> bool:
        routing = policy.routing
        if (
            request.required_target_id is not None
            and profile.target_id != request.required_target_id
        ):
            return False
        if (
            routing.allowed_target_ids
            and profile.target_id not in routing.allowed_target_ids
        ):
            return False
        if profile.backend_kind not in routing.allowed_backend_kinds:
            return False
        if not set(routing.required_target_tags).issubset(profile.tags):
            return False
        supported_capabilities = profile.supported_cognitive_capability_ids
        if (
            supported_capabilities is not None
            and request.cognitive_capability_id not in supported_capabilities
        ):
            return False
        required = request.requirements
        if (
            _STRUCTURED_OUTPUT_RANK[required.required_structured_output]
            > _STRUCTURED_OUTPUT_RANK[profile.features.structured_output]
        ):
            return False
        if (
            required.tool_intent is not ToolIntentMode.DISABLED
            and not profile.features.tool_intent
        ):
            return False
        requested_output = required.max_output_tokens
        target_output = profile.limits.max_output_tokens
        if (
            requested_output is not None
            and target_output is not None
            and requested_output > target_output
        ):
            return False
        budget = policy.budget
        if (
            budget.max_total_tokens is not None
            and not profile.metering.reports_token_usage
        ):
            return False
        if (
            budget.max_response_cost is not None
            and not profile.metering.reports_monetary_cost
        ):
            return False
        return True
