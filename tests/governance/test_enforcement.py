from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from adaptive_agent_runtime.governance import (
    AuthorizationReplayError,
    AuthorizationUseStatus,
    AuthorizationVerificationError,
    BoundGovernedOperation,
    ConfidenceSignals,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceRequest,
    GovernanceRule,
    GovernanceScope,
    GovernanceTarget,
    GovernedOperationError,
    GovernedOperationExecutor,
    HumanReviewDecision,
    ImpactAssessment,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    ReviewOutcome,
    RiskLevel,
    RuleEffect,
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    StrictAuthorizationVerifier,
    default_governance_policy,
    governance_fingerprint,
)
from adaptive_agent_runtime.governance.models import GovernancePolicy
from adaptive_agent_runtime.persistence import SQLitePersistence


NOW = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)


class SimulatedProcessCrash(BaseException):
    pass


def allowed_operation(subject: object):  # type: ignore[no-untyped-def]
    operation = "state.test_apply"
    target = GovernanceTarget(target_type="test_state", target_id="state-1")
    request = GovernanceRequest(
        request_id=UUID(int=101),
        scope=GovernanceScope.STATE,
        operation=operation,
        target=target,
        risk=RiskLevel.LOW,
        signals=ConfidenceSignals(
            stated_confidence=1.0,
            impact=ImpactAssessment(
                score=0.1,
                reversible=True,
                description="Bounded test mutation.",
            ),
        ),
        attributes={
            SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(subject)
        },
        requested_at=NOW,
    )
    reviews = InMemoryHumanReviewService(clock=lambda: NOW)
    evaluator = RuntimeGovernanceEvaluator(
        policy=GovernancePolicy(
            policy_id="test.enforcement",
            version="1",
            rules=(
                GovernanceRule(
                    rule_id="allow.test",
                    description="Allow bounded enforcement test.",
                    effect=RuleEffect.ALLOW,
                    scopes=(GovernanceScope.STATE,),
                    operations=(operation,),
                    risk_levels=(RiskLevel.LOW,),
                ),
            ),
        ),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=reviews,
        clock=lambda: NOW,
    )
    decision = evaluator.evaluate(request)
    authorization = GovernanceAuthorizationIssuer(clock=lambda: NOW).issue(
        request,
        decision,
    )
    return request, decision, authorization, target


class GovernanceEnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorization_is_verified_applied_and_consumed_once(self) -> None:
        subject = {"revision": 1, "value": "approved"}
        request, decision, authorization, target_identity = allowed_operation(
            subject
        )
        store = InMemoryAuthorizationConsumptionStore()
        executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=store,
            clock=lambda: NOW,
        )
        calls = 0

        async def apply() -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {"revision": 2}

        target = BoundGovernedOperation(
            module_id="test.target",
            operation=request.operation,
            target=target_identity,
            subject=subject,
            apply=apply,
        )

        result = await executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=target,
        )

        self.assertEqual(result, {"revision": 2})
        use = await store.load(authorization.authorization_id)
        self.assertIsNotNone(use)
        assert use is not None
        self.assertEqual(use.status, AuthorizationUseStatus.APPLIED)
        self.assertEqual(use.revision, 1)
        with self.assertRaises(AuthorizationReplayError):
            await executor.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=target,
            )
        self.assertEqual(calls, 1)

    async def test_resolution_time_does_not_regress_when_clock_moves_backward(self) -> None:
        subject = {"revision": 1, "value": "approved"}
        request, decision, authorization, target_identity = allowed_operation(
            subject
        )
        store = InMemoryAuthorizationConsumptionStore()
        clock_values = iter((NOW, NOW - timedelta(microseconds=1)))
        executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=store,
            clock=lambda: next(clock_values),
        )

        async def apply() -> dict[str, object]:
            return {"revision": 2}

        await executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="test.target",
                operation=request.operation,
                target=target_identity,
                subject=subject,
                apply=apply,
            ),
        )

        use = await store.load(authorization.authorization_id)
        self.assertIsNotNone(use)
        assert use is not None
        self.assertEqual(use.reserved_at, NOW)
        self.assertEqual(use.updated_at, NOW)
        self.assertEqual(use.status, AuthorizationUseStatus.APPLIED)

    async def test_changed_subject_is_rejected_before_reservation(self) -> None:
        approved_subject = {"revision": 1}
        request, decision, authorization, target_identity = allowed_operation(
            approved_subject
        )
        store = InMemoryAuthorizationConsumptionStore()
        executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=store,
            clock=lambda: NOW,
        )
        called = False

        async def apply() -> dict[str, int]:
            nonlocal called
            called = True
            return {"revision": 3}

        with self.assertRaises(AuthorizationVerificationError):
            await executor.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=BoundGovernedOperation(
                    module_id="test.target",
                    operation=request.operation,
                    target=target_identity,
                    subject={"revision": 2},
                    apply=apply,
                ),
            )
        self.assertFalse(called)
        self.assertIsNone(await store.load(authorization.authorization_id))

    async def test_failed_apply_consumes_authorization_as_failed(self) -> None:
        subject = {"revision": 1}
        request, decision, authorization, target_identity = allowed_operation(
            subject
        )
        store = InMemoryAuthorizationConsumptionStore()
        executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=store,
            clock=lambda: NOW,
        )

        async def apply() -> dict[str, int]:
            raise ValueError("mutation rejected")

        with self.assertRaises(GovernedOperationError):
            await executor.execute(
                request=request,
                decision=decision,
                authorization=authorization,
                target=BoundGovernedOperation(
                    module_id="test.target",
                    operation=request.operation,
                    target=target_identity,
                    subject=subject,
                    apply=apply,
                ),
            )
        use = await store.load(authorization.authorization_id)
        self.assertIsNotNone(use)
        assert use is not None
        self.assertEqual(use.status, AuthorizationUseStatus.FAILED)
        self.assertIn("mutation rejected", use.error or "")

    async def test_reserved_use_survives_restart_and_blocks_replay(self) -> None:
        subject = {"external_side_effect": True}
        request, decision, authorization, target_identity = allowed_operation(
            subject
        )
        with TemporaryDirectory() as directory:
            path = f"{directory}/governance.sqlite3"
            first = SQLitePersistence(path)
            executor = GovernedOperationExecutor(
                verifier=StrictAuthorizationVerifier(),
                consumption_store=first.authorization_store,
                clock=lambda: NOW,
            )

            async def crash() -> dict[str, bool]:
                raise SimulatedProcessCrash()

            with self.assertRaises(SimulatedProcessCrash):
                await executor.execute(
                    request=request,
                    decision=decision,
                    authorization=authorization,
                    target=BoundGovernedOperation(
                        module_id="test.external_target",
                        operation=request.operation,
                        target=target_identity,
                        subject=subject,
                        apply=crash,
                    ),
                )
            first.close()

            reopened = SQLitePersistence(path)
            replayed = False

            async def replay() -> dict[str, bool]:
                nonlocal replayed
                replayed = True
                return {"replayed": True}

            with self.assertRaises(AuthorizationReplayError):
                await GovernedOperationExecutor(
                    verifier=StrictAuthorizationVerifier(),
                    consumption_store=reopened.authorization_store,
                    clock=lambda: NOW,
                ).execute(
                    request=request,
                    decision=decision,
                    authorization=authorization,
                    target=BoundGovernedOperation(
                        module_id="test.external_target",
                        operation=request.operation,
                        target=target_identity,
                        subject=subject,
                        apply=replay,
                    ),
                )
            use = await reopened.authorization_store.load(
                authorization.authorization_id
            )
            reopened.close()

        self.assertFalse(replayed)
        self.assertIsNotNone(use)
        assert use is not None
        self.assertEqual(use.status, AuthorizationUseStatus.RESERVED)


class PersistentReviewTests(unittest.TestCase):
    def test_pending_review_can_be_resolved_after_database_reopen(self) -> None:
        request = GovernanceRequest(
            request_id=UUID(int=202),
            scope=GovernanceScope.EVOLUTION,
            operation="optimization.apply",
            target=GovernanceTarget(
                target_type="optimization_proposal",
                target_id="proposal-1",
            ),
            risk=RiskLevel.HIGH,
            signals=ConfidenceSignals(
                stated_confidence=0.9,
                impact=ImpactAssessment(
                    score=0.9,
                    reversible=True,
                    description="Runtime policy change.",
                ),
            ),
            requested_at=NOW,
        )
        with TemporaryDirectory() as directory:
            path = f"{directory}/review.sqlite3"
            first = SQLitePersistence(path)
            evaluator = RuntimeGovernanceEvaluator(
                policy=default_governance_policy(),
                rule_evaluator=DeterministicRuleEvaluator(),
                confidence_evaluator=DeterministicConfidenceEvaluator(),
                review_service=first.human_review_service,
                clock=lambda: NOW,
            )
            preliminary = evaluator.evaluate(request)
            self.assertEqual(preliminary.outcome, DecisionOutcome.REVIEW_REQUIRED)
            review_id = preliminary.review_request_id
            assert review_id is not None
            first.close()

            reopened = SQLitePersistence(path)
            review = reopened.human_review_service.get(review_id)
            self.assertIsNotNone(review)
            assert review is not None
            resolved_at = review.requested_at + timedelta(minutes=1)
            resolved = reopened.human_review_service.resolve(
                review_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="human-reviewer",
                    rationale="Bounded rollout with rollback plan.",
                    decided_at=resolved_at,
                ),
            )
            resumed_evaluator = RuntimeGovernanceEvaluator(
                policy=default_governance_policy(),
                rule_evaluator=DeterministicRuleEvaluator(),
                confidence_evaluator=DeterministicConfidenceEvaluator(),
                review_service=reopened.human_review_service,
                clock=lambda: resolved_at,
            )
            final = resumed_evaluator.finalize_review(request, resolved)
            authorization = GovernanceAuthorizationIssuer(
                clock=lambda: resolved_at
            ).issue(request, final)
            reopened.close()

        self.assertEqual(final.outcome, DecisionOutcome.ALLOW)
        self.assertEqual(authorization.request_id, request.request_id)


if __name__ == "__main__":
    unittest.main()
