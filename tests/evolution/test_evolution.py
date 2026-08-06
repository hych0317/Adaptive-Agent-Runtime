from __future__ import annotations

from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunStatus,
)
from adaptive_agent_runtime.evaluation import (
    EvaluationComponent,
    OptimizationProposal,
)
from adaptive_agent_runtime.evolution import (
    DeterministicReplayValidator,
    OptimizationApplicationStatus,
    ReplayCase,
    ReplayExecutionResult,
    ReplayValidationError,
    RuntimeConfigurationSnapshot,
    RuntimeReplayRunner,
    UnsupportedOptimizationError,
)
from adaptive_agent_runtime.evolution.apply import (
    ConfigurationPatchPlanner,
    InMemoryEvolutionStore,
    OptimizationDeploymentService,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernedOperationExecutor,
    HumanReviewDecision,
    ReviewOutcome,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.legacy.optimization_governance import (
    LegacyOptimizationApplyGovernanceAdapter,
    LegacyOptimizationRollbackGovernanceAdapter,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    MockExecutionStrategy,
    NodeExecutionResult,
    StrategyActionExecutor,
    TaskNode,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from adaptive_agent_runtime.persistence.evolution import SQLiteEvolutionStore


def proposal(*, recover: bool = True) -> OptimizationProposal:
    now = datetime.now(timezone.utc)
    return OptimizationProposal(
        proposal_id=uuid4(),
        source_pattern_ids=(uuid4(),),
        source_candidate_ids=(uuid4(),),
        target_component=EvaluationComponent.ORCHESTRATION,
        change_kind="orchestration.strategy.review",
        change_spec={
            "proposal_only": True,
            "config_patch": {"failure_replanning_enabled": recover},
        },
        rationale="Repeated node failures require bounded recovery.",
        expected_benefit="Recover replayed workloads without score regression.",
        proposal_confidence=0.95,
        validation_plan=("Replay persisted failure workloads.",),
        rollback_plan=("Restore the prior configuration version.",),
        created_at=now,
    )


class AgentRuntimeReplayExecutor:
    """Replay through a fresh real Core + Orchestration Runtime instance."""

    module_id = "test.evolution.agent_runtime_replay"

    async def execute(
        self,
        case: ReplayCase,
        configuration: RuntimeConfigurationSnapshot,
    ) -> ReplayExecutionResult:
        node = TaskNode(
            goal=str(case.input_payload["goal"]),
            expected_output="replay result",
            strategy_id="mock",
        )
        recover = configuration.config.get("failure_replanning_enabled") is True
        outcome = (
            NodeExecutionResult.ok(output={"recovered": True})
            if recover
            else NodeExecutionResult.failed(error="recorded failure")
        )
        runtime = AgentRuntime(
            planner=DynamicTaskGraphPlanner(DynamicTaskGraph(nodes=(node,))),
            executor=StrategyActionExecutor(
                (MockExecutionStrategy({node.node_id: outcome}),)
            ),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            max_steps=3,
        )
        result = await runtime.run(
            AgentTask(
                task_id=case.task_id,
                description=str(case.input_payload["goal"]),
            )
        )
        succeeded = result.final_state.status is RunStatus.COMPLETED
        return ReplayExecutionResult(
            replay_run_id=result.final_state.run_id,
            score=1.0 if succeeded else 0.0,
            succeeded=succeeded,
            diagnostics=(result.final_state.status.value,),
        )


class FixedRegressionReplayExecutor:
    module_id = "test.evolution.regression_replay"

    async def execute(
        self,
        case: ReplayCase,
        configuration: RuntimeConfigurationSnapshot,
    ) -> ReplayExecutionResult:
        del case, configuration
        return ReplayExecutionResult(
            replay_run_id=uuid4(),
            score=0.4,
            succeeded=True,
        )


def replay_case(*, baseline_score: float = 0.0) -> ReplayCase:
    return ReplayCase(
        case_id=uuid4(),
        source_run_id=uuid4(),
        task_id=uuid4(),
        input_payload={"goal": "recover the failed task"},
        baseline_score=baseline_score,
        minimum_score=0.8,
    )


class OptimizationEvolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_configuration_key_is_rejected_before_replay(self) -> None:
        store = InMemoryEvolutionStore(allow_legacy_mutations=True)
        await store.initialize(
            RuntimeConfigurationSnapshot(
                component="orchestration",
                version=0,
                config={"failure_replanning_enabled": False},
            )
        )
        unsupported = proposal().model_copy(
            update={"change_spec": {"config_patch": {"unknown": True}}}
        )
        service = OptimizationDeploymentService(
            store=store,
            change_planner=ConfigurationPatchPlanner(),
            replay_runner=RuntimeReplayRunner(AgentRuntimeReplayExecutor()),
            validator=DeterministicReplayValidator(),
            allow_legacy_mutations=True,
        )

        with self.assertRaises(UnsupportedOptimizationError):
            await service.prepare(unsupported, (replay_case(),))

    async def test_replay_regression_blocks_configuration_mutation(self) -> None:
        store = InMemoryEvolutionStore(allow_legacy_mutations=True)
        baseline = RuntimeConfigurationSnapshot(
            component="orchestration",
            version=0,
            config={"failure_replanning_enabled": False},
        )
        await store.initialize(baseline)
        service = OptimizationDeploymentService(
            store=store,
            change_planner=ConfigurationPatchPlanner(),
            replay_runner=RuntimeReplayRunner(FixedRegressionReplayExecutor()),
            validator=DeterministicReplayValidator(),
            allow_legacy_mutations=True,
        )
        deployment = await service.prepare(
            proposal(),
            (replay_case(baseline_score=0.9),),
        )

        self.assertFalse(deployment.validation.passed)
        with self.assertRaises(ReplayValidationError):
            await service.apply(deployment)
        self.assertEqual(await store.load_active("orchestration"), baseline)

    async def test_governed_apply_persists_and_governed_rollback_restores(
        self,
    ) -> None:
        baseline = RuntimeConfigurationSnapshot(
            component="orchestration",
            version=0,
            config={"failure_replanning_enabled": False},
        )
        candidate_proposal = proposal()
        case = replay_case()
        with TemporaryDirectory() as directory:
            path = f"{directory}/evolution.sqlite3"
            first = SQLitePersistence(path)
            first_evolution = SQLiteEvolutionStore(
                first.database,
                allow_legacy_mutations=True,
            )
            await first_evolution.initialize(baseline)
            await first_evolution.save(case)
            service = OptimizationDeploymentService(
                store=first_evolution,
                change_planner=ConfigurationPatchPlanner(),
                replay_runner=RuntimeReplayRunner(AgentRuntimeReplayExecutor()),
                validator=DeterministicReplayValidator(),
                allow_legacy_mutations=True,
            )
            deployment = await service.prepare(
                candidate_proposal,
                await first_evolution.list_all(),
            )
            self.assertTrue(deployment.validation.passed)
            self.assertGreater(
                deployment.validation.candidate_score,
                deployment.validation.baseline_score,
            )

            request = LegacyOptimizationApplyGovernanceAdapter().to_request(
                candidate_proposal
            )
            evaluator = RuntimeGovernanceEvaluator(
                policy=default_governance_policy(),
                rule_evaluator=DeterministicRuleEvaluator(),
                confidence_evaluator=DeterministicConfidenceEvaluator(),
                review_service=first.human_review_service,
            )
            preliminary = evaluator.evaluate(request)
            review_id = preliminary.review_request_id
            assert review_id is not None
            review = first.human_review_service.get(review_id)
            assert review is not None
            resolved = first.human_review_service.resolve(
                review_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="evolution-reviewer",
                    rationale="Replay improved the persisted cases.",
                    decided_at=review.requested_at,
                ),
            )
            final = evaluator.finalize_review(request, resolved)
            authorization = GovernanceAuthorizationIssuer().issue(
                request,
                final,
            )

            async def apply():  # type: ignore[no-untyped-def]
                return await service.apply(deployment)

            operation_executor = GovernedOperationExecutor(
                verifier=StrictAuthorizationVerifier(),
                consumption_store=first.authorization_store,
            )
            application = await operation_executor.execute(
                request=request,
                decision=final,
                authorization=authorization,
                target=BoundGovernedOperation(
                    module_id="test.optimization_apply",
                    operation=request.operation,
                    target=request.target,
                    subject=candidate_proposal,
                    apply=apply,
                ),
            )
            self.assertEqual(
                application.status,
                OptimizationApplicationStatus.APPLIED,
            )
            first.close()

            reopened = SQLitePersistence(path)
            reopened_evolution = SQLiteEvolutionStore(
                reopened.database,
                allow_legacy_mutations=True,
            )
            active = await reopened_evolution.load_active(
                "orchestration"
            )
            self.assertEqual(active, deployment.candidate)
            assert active is not None
            replayed = await AgentRuntimeReplayExecutor().execute(case, active)
            self.assertTrue(replayed.succeeded)

            persisted_application = await reopened_evolution.load_application(
                application.application_id
            )
            self.assertEqual(persisted_application, application)
            assert persisted_application is not None
            rollback_request = LegacyOptimizationRollbackGovernanceAdapter().to_request(
                persisted_application
            )
            rollback_evaluator = RuntimeGovernanceEvaluator(
                policy=default_governance_policy(),
                rule_evaluator=DeterministicRuleEvaluator(),
                confidence_evaluator=DeterministicConfidenceEvaluator(),
                review_service=reopened.human_review_service,
            )
            rollback_preliminary = rollback_evaluator.evaluate(rollback_request)
            rollback_review_id = rollback_preliminary.review_request_id
            assert rollback_review_id is not None
            rollback_review = reopened.human_review_service.get(
                rollback_review_id
            )
            assert rollback_review is not None
            rollback_resolved = reopened.human_review_service.resolve(
                rollback_review_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="rollback-reviewer",
                    rationale="Post-apply guard requested rollback.",
                    decided_at=rollback_review.requested_at,
                ),
            )
            rollback_final = rollback_evaluator.finalize_review(
                rollback_request,
                rollback_resolved,
            )
            rollback_authorization = GovernanceAuthorizationIssuer().issue(
                rollback_request,
                rollback_final,
            )
            reopened_service = OptimizationDeploymentService(
                store=reopened_evolution,
                change_planner=ConfigurationPatchPlanner(),
                replay_runner=RuntimeReplayRunner(AgentRuntimeReplayExecutor()),
                validator=DeterministicReplayValidator(),
                allow_legacy_mutations=True,
            )

            async def rollback():  # type: ignore[no-untyped-def]
                return await reopened_service.rollback(application.application_id)

            rolled_back = await GovernedOperationExecutor(
                verifier=StrictAuthorizationVerifier(),
                consumption_store=reopened.authorization_store,
            ).execute(
                request=rollback_request,
                decision=rollback_final,
                authorization=rollback_authorization,
                target=BoundGovernedOperation(
                    module_id="test.optimization_rollback",
                    operation=rollback_request.operation,
                    target=rollback_request.target,
                    subject=persisted_application,
                    apply=rollback,
                ),
            )

            self.assertEqual(
                rolled_back.status,
                OptimizationApplicationStatus.ROLLED_BACK,
            )
            self.assertEqual(
                await reopened_evolution.load_active("orchestration"),
                baseline,
            )
            self.assertEqual(
                len(
                    await reopened_evolution.application_history(
                        application.application_id
                    )
                ),
                2,
            )
            reopened.close()


if __name__ == "__main__":
    unittest.main()
