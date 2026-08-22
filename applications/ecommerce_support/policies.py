"""Deterministic ecommerce policies kept outside model control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from applications.ecommerce_support.models import (
    AuthenticatedPrincipal,
    MemoryWriteRequest,
)
from applications.governance_scenario_suite.contracts import ReasonCode


class DomainPolicyError(ValueError):
    """Safe domain rejection with a stable machine-readable reason."""

    def __init__(self, reason_code: ReasonCode, public_message: str) -> None:
        super().__init__(public_message)
        self.reason_code = reason_code
        self.public_message = public_message


@dataclass(frozen=True)
class EcommercePolicy:
    policy_version: str = "ecommerce-policy-v1"
    automatic_refund_limit_cents: int = 2_000
    automatic_coupon_limit_cents: int = 1_000

    def __post_init__(self) -> None:
        if self.automatic_refund_limit_cents < 0:
            raise ValueError("automatic refund limit cannot be negative")
        if self.automatic_coupon_limit_cents < 0:
            raise ValueError("automatic coupon limit cannot be negative")

    def refund_requires_approval(self, amount_cents: int) -> bool:
        return amount_cents > self.automatic_refund_limit_cents

    def coupon_is_automatic(self, amount_cents: int) -> bool:
        return amount_cents <= self.automatic_coupon_limit_cents


class MemoryCandidateWritePolicy:
    """Bind user preferences to Principal and reject sensitive long-term writes."""

    _FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
        {
            "address",
            "authorization",
            "card_number",
            "credential",
            "cvv",
            "password",
            "payment_token",
            "secret",
            "shipping_address",
            "token",
        }
    )

    def validate(
        self,
        principal: AuthenticatedPrincipal,
        candidate: MemoryWriteRequest,
    ) -> None:
        if candidate.tenant_id != principal.tenant_id:
            raise DomainPolicyError(
                ReasonCode.MEMORY_SCOPE_DENIED,
                "Memory is not available in the current scope.",
            )
        if candidate.category == "user_preference":
            if candidate.subject_user_id != principal.user_id:
                raise DomainPolicyError(
                    ReasonCode.MEMORY_SCOPE_DENIED,
                    "Memory is not available in the current scope.",
                )
        if _contains_forbidden_key(candidate.content, self._FORBIDDEN_KEYS):
            raise DomainPolicyError(
                ReasonCode.SENSITIVE_MEMORY_WRITE_DENIED,
                "Sensitive information cannot be stored as long-term Memory.",
            )


def _contains_forbidden_key(
    value: object,
    forbidden: frozenset[str],
) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.strip().lower()
            if normalized in forbidden or _contains_forbidden_key(item, forbidden):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False
