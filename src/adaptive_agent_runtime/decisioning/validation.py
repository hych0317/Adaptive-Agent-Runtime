"""Runtime validation and effect normalization boundary."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Generic

from pydantic import BaseModel

from adaptive_agent_runtime.core.models import utc_now

from adaptive_agent_runtime.decisioning.budget import budget_violations
from adaptive_agent_runtime.decisioning.context import AgentContext
from adaptive_agent_runtime.decisioning.contracts import DecisionEffectNormalizer
from adaptive_agent_runtime.decisioning.models import (
    DecisionBasis,
    DecisionBudgetUsage,
    DecisionRequest,
    DecisionProposal,
    DecisionValidation,
    DecisionValidationCheck,
    DecisionValidationOutcome,
    DecisionValidationStatus,
    DecisionValidationViolation,
    NormalizedDecisionEffect,
    RequestPayloadT,
    ProposalPayloadT,
    EffectPayloadT,
)


class RuntimeDecisionValidator(
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
):
    """Validate authority constraints before producing a governed effect."""

    module_id = "decision.validator.runtime"

    def __init__(
        self,
        *,
        normalizer: DecisionEffectNormalizer[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._normalizer = normalizer
        self._clock = clock

    def validate(
        self,
        request: DecisionRequest[RequestPayloadT],
        context: AgentContext,
        proposal: DecisionProposal[ProposalPayloadT],
        *,
        current_basis: DecisionBasis,
        usage: DecisionBudgetUsage,
    ) -> DecisionValidationOutcome[EffectPayloadT]:
        checks: list[DecisionValidationCheck] = []
        violations: list[DecisionValidationViolation] = []

        def check(name: str, passed: bool, detail: str, code: str) -> None:
            checks.append(
                DecisionValidationCheck(name=name, passed=passed, detail=detail)
            )
            if not passed:
                violations.append(
                    DecisionValidationViolation(code=code, message=detail)
                )

        check(
            "request_binding",
            proposal.request_id == request.request_id,
            "proposal must reference the active decision request",
            "request_mismatch",
        )
        check(
            "proposal_type",
            proposal.proposal_type == request.decision_type,
            "proposal type must match requested decision type",
            "proposal_type_mismatch",
        )
        check(
            "schema_version",
            proposal.schema_version == request.schema_version,
            "proposal schema version must match the request",
            "schema_version_mismatch",
        )
        now = self._clock()
        is_expired = request.expires_at is not None and now >= request.expires_at
        check(
            "expiry",
            not is_expired,
            "decision request must not be expired",
            "request_expired",
        )
        check(
            "basis",
            current_basis == request.basis,
            "current Runtime basis must match the request snapshot",
            "stale_basis",
        )
        check(
            "proposal_snapshot",
            proposal.input_snapshot_fingerprint
            == request.basis.snapshot_fingerprint,
            "proposal must be based on the requested Runtime snapshot",
            "snapshot_mismatch",
        )
        check(
            "context_binding",
            proposal.context_fingerprint == context.context_fingerprint,
            "proposal must reference the isolated Agent context",
            "context_mismatch",
        )
        check(
            "allowed_action",
            proposal.selected_action in request.allowed_actions,
            "proposal action must be allowlisted by Runtime",
            "action_not_allowed",
        )
        visible_evidence = {item.evidence_id for item in context.evidence}
        check(
            "evidence_scope",
            set(proposal.evidence_refs).issubset(visible_evidence),
            "proposal may reference only evidence visible in Agent context",
            "evidence_not_visible",
        )
        check(
            "revision",
            proposal.revision == usage.revision_count
            and proposal.revision <= request.budget.max_revision_count,
            "proposal revision must match Runtime budget accounting",
            "revision_mismatch",
        )
        exhausted = budget_violations(request.budget, usage)
        check(
            "budget",
            not exhausted,
            (
                "decision budget must not be exhausted"
                if not exhausted
                else f"decision budget exceeded: {', '.join(exhausted)}"
            ),
            "budget_exhausted",
        )

        normalized: NormalizedDecisionEffect[EffectPayloadT] | None = None
        if not violations:
            try:
                normalized = self._normalizer.normalize(request, proposal)
            except Exception as exc:
                detail = str(exc) or exc.__class__.__name__
                checks.append(
                    DecisionValidationCheck(
                        name="effect_normalization",
                        passed=False,
                        detail=f"effect normalization failed: {detail}",
                    )
                )
                violations.append(
                    DecisionValidationViolation(
                        code="effect_normalization_failed",
                        message=f"{exc.__class__.__name__}: {detail}",
                    )
                )

        status = (
            DecisionValidationStatus.PASSED
            if not violations and normalized is not None
            else DecisionValidationStatus.FAILED
        )
        validation = DecisionValidation(
            request_id=request.request_id,
            proposal_id=proposal.proposal_id,
            status=status,
            checks=tuple(checks),
            violations=tuple(violations),
            validated_basis=current_basis,
            normalized_effect_fingerprint=(
                normalized.effect_fingerprint if normalized is not None else None
            ),
            created_at=now,
        )
        return DecisionValidationOutcome(
            validation=validation,
            normalized_effect=normalized,
        )
