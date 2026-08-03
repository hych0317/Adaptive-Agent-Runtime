"""Decision-only contracts for Runtime Governance."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.governance.models import (
    ConfidenceAssessment,
    ConfidencePolicy,
    GovernanceAuthorization,
    AuthorizationUse,
    GovernanceDecision,
    GovernancePolicy,
    GovernanceRequest,
    HumanReviewDecision,
    ReviewRequest,
    RuleEvaluation,
    GovernanceTarget,
)


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class RuleEvaluator(RuntimeModule, Protocol):
    def evaluate(
        self,
        request: GovernanceRequest,
        policy: GovernancePolicy,
    ) -> RuleEvaluation: ...


@runtime_checkable
class ConfidenceEvaluator(RuntimeModule, Protocol):
    def evaluate(
        self,
        request: GovernanceRequest,
        policy: ConfidencePolicy,
    ) -> ConfidenceAssessment: ...


@runtime_checkable
class HumanReviewService(RuntimeModule, Protocol):
    def open(
        self,
        request: GovernanceRequest,
        *,
        policy_id: str,
        policy_version: str,
        reason: str,
    ) -> ReviewRequest: ...

    def get(self, review_request_id: UUID) -> ReviewRequest | None: ...

    def resolve(
        self,
        review_request_id: UUID,
        decision: HumanReviewDecision,
    ) -> ReviewRequest: ...


@runtime_checkable
class GovernanceEvaluator(RuntimeModule, Protocol):
    def evaluate(self, request: GovernanceRequest) -> GovernanceDecision: ...

    def finalize_review(
        self,
        request: GovernanceRequest,
        review: ReviewRequest,
    ) -> GovernanceDecision: ...


@runtime_checkable
class AuthorizationIssuer(RuntimeModule, Protocol):
    def issue(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
    ) -> GovernanceAuthorization: ...


@runtime_checkable
class GovernedOperationTarget(RuntimeModule, Protocol[T_co]):
    """A bound operation whose subject and destination cannot change."""

    @property
    def operation(self) -> str: ...

    @property
    def target(self) -> GovernanceTarget: ...

    @property
    def subject_fingerprint(self) -> str: ...

    async def apply(self) -> T_co: ...


@runtime_checkable
class AuthorizationVerifier(Protocol):
    def verify(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
        authorization: GovernanceAuthorization,
        target: GovernedOperationTarget[object],
    ) -> None: ...


@runtime_checkable
class AuthorizationConsumptionStore(RuntimeModule, Protocol):
    async def reserve(self, use: AuthorizationUse) -> None: ...

    async def resolve(
        self,
        use: AuthorizationUse,
        *,
        expected_revision: int,
    ) -> None: ...

    async def load(self, authorization_id: UUID) -> AuthorizationUse | None: ...
