"""Deterministic capability-aware context projection with egress enforcement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from adaptive_agent_runtime.context_memory.context_models import (
    ContextSource,
    ContextUnit,
)
from adaptive_agent_runtime.llm.adapters.models import (
    ContextOmission,
    ContextOmissionReason,
    ContextProjectionRequest,
    ContextSensitivity,
    LLMContextBlock,
    LLMContextPackage,
    LLMContextRole,
)
from adaptive_agent_runtime.llm.errors import (
    ContextEgressDeniedError,
    ContextProjectionBudgetError,
)


class PolicyEnforcedContextAdapter:
    """Filter, redact, and budget Context Units before provider formatting."""

    module_id = "llm.context_adapter.policy_enforced"

    def project(self, request: ContextProjectionRequest) -> LLMContextPackage:
        self._ensure_target_allowed(request)
        classifications = {
            item.context_id: item.sensitivity
            for item in request.classifications
        }
        max_tokens = self._effective_budget(request)
        required_ids = set(request.assembly.requirement.required_context_ids)
        eligible: list[ContextUnit] = []
        omissions: dict[UUID, ContextOmissionReason] = {}

        for unit in request.assembly.units:
            omission = self._omission_reason(
                request,
                unit,
                classifications[unit.context_id],
            )
            if omission is None:
                eligible.append(unit)
                continue
            if unit.context_id in required_ids:
                raise ContextEgressDeniedError(
                    f"required context '{unit.context_id}' is blocked by "
                    f"{omission.value}"
                )
            omissions[unit.context_id] = omission

        required_tokens = sum(
            unit.metadata.estimated_tokens
            for unit in eligible
            if unit.context_id in required_ids
        )
        if required_tokens > max_tokens:
            raise ContextProjectionBudgetError(required_tokens, max_tokens)

        selected_ids = {
            unit.context_id
            for unit in eligible
            if unit.context_id in required_ids
        }
        remaining_tokens = max_tokens - required_tokens
        for unit in eligible:
            if unit.context_id in selected_ids:
                continue
            tokens = unit.metadata.estimated_tokens
            if tokens <= remaining_tokens:
                selected_ids.add(unit.context_id)
                remaining_tokens -= tokens
            else:
                omissions[unit.context_id] = ContextOmissionReason.TOKEN_BUDGET

        blocks = tuple(
            self._project_unit(
                request,
                unit,
                classifications[unit.context_id],
            )
            for unit in request.assembly.units
            if unit.context_id in selected_ids
        )
        ordered_omissions = tuple(
            ContextOmission(context_id=unit.context_id, reason=omissions[unit.context_id])
            for unit in request.assembly.units
            if unit.context_id in omissions
        )
        return LLMContextPackage(
            cognitive_capability_id=(
                request.capability_policy.cognitive_capability_id
            ),
            target_id=request.target.target_id,
            goal=request.assembly.requirement.goal,
            blocks=blocks,
            omissions=ordered_omissions,
            used_tokens=sum(block.estimated_tokens for block in blocks),
            max_tokens=max_tokens,
        )

    @staticmethod
    def _ensure_target_allowed(request: ContextProjectionRequest) -> None:
        if request.target.target_id not in request.egress_policy.allowed_target_ids:
            raise ContextEgressDeniedError(
                f"target '{request.target.target_id}' is not allowed"
            )

    @staticmethod
    def _effective_budget(request: ContextProjectionRequest) -> int:
        limits = [
            request.assembly.requirement.max_tokens,
            request.capability_policy.max_context_tokens,
            request.egress_policy.max_context_tokens,
        ]
        target_limit = request.target.limits.max_context_tokens
        if target_limit is not None:
            limits.append(target_limit)
        return min(limits)

    @staticmethod
    def _omission_reason(
        request: ContextProjectionRequest,
        unit: ContextUnit,
        sensitivity: ContextSensitivity,
    ) -> ContextOmissionReason | None:
        capability_policy = request.capability_policy
        if (
            unit.metadata.layer not in capability_policy.allowed_layers
            or unit.metadata.source not in capability_policy.allowed_sources
            or set(unit.metadata.tags).intersection(capability_policy.excluded_tags)
        ):
            return ContextOmissionReason.CAPABILITY_POLICY
        egress_policy = request.egress_policy
        if (
            sensitivity not in egress_policy.allowed_sensitivities
            or set(unit.metadata.tags).intersection(egress_policy.blocked_tags)
        ):
            return ContextOmissionReason.EGRESS_POLICY
        return None

    @staticmethod
    def _project_unit(
        request: ContextProjectionRequest,
        unit: ContextUnit,
        sensitivity: ContextSensitivity,
    ) -> LLMContextBlock:
        content, redacted_keys = _redact(
            unit.content,
            set(request.egress_policy.redact_keys),
            request.egress_policy.redaction_placeholder,
        )
        role = (
            LLMContextRole.USER
            if unit.metadata.source is ContextSource.CONVERSATION
            else LLMContextRole.CONTEXT
        )
        return LLMContextBlock(
            context_id=unit.context_id,
            role=role,
            source=unit.metadata.source,
            layer=unit.metadata.layer,
            source_reference=unit.metadata.source_reference,
            sensitivity=sensitivity,
            content=content,
            estimated_tokens=unit.metadata.estimated_tokens,
            redacted_keys=redacted_keys,
        )


def _redact(
    value: Any,
    redact_keys: set[str],
    placeholder: str,
) -> tuple[Any, tuple[str, ...]]:
    redacted: set[str] = set()
    normalized_redact_keys = {key.casefold() for key in redact_keys}

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            projected: dict[str, Any] = {}
            for key, child in item.items():
                if key.casefold() in normalized_redact_keys:
                    projected[key] = placeholder
                    redacted.add(key)
                else:
                    projected[key] = visit(child)
            return projected
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        return item

    return visit(value), tuple(sorted(redacted))
