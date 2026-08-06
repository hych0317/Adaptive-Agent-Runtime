from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionFaultPoint,
    DecisionResultStatus,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceTarget,
    RuntimeCommitPermit,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
)
from adaptive_agent_runtime.optimization import (
    OptimizationAssessmentAgentRequest,
    OptimizationAssessmentRequest,
    OptimizationEvidenceBinding,
    OptimizationProposalDraft,
    OptimizationProposalEffect,
    OptimizationRiskClassification,
    OptimizationScope,
    OptimizationTarget,
    OptimizationTargetConstraints,
    OptimizationTargetKey,
    OptimizationTargetType,
    stable_optimization_request_id,
    validate_optimization_value,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.optimization import (
    SQLiteOptimizationProposalStore,
)
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.optimization import (
    DeterministicOptimizationAssessmentCapability,
    OptimizationAssessmentCapability,
    ResearchOptimizationProposalHandler,
    ResearchPlanningOptimizationBaselineProvider,
    research_initial_planning_optimization_scope,
)
from applications.research_agent.planning import RESEARCH_PLANNING_GRAPH_LIMITS


class RecordingOptimizationCapability:
    module_id = "test.optimization.recording_capability"
    capability_id = "test.optimization.recording"

    def __init__(self, *, outside_candidate: bool = False) -> None:
        self.calls = 0
        self.outside_candidate = outside_candidate
        self.requests: list[OptimizationAssessmentAgentRequest] = []

    async def assess_optimization(
        self,
        request: OptimizationAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[OptimizationProposalDraft]:
        del invocation
        self.calls += 1
        self.requests.append(request)
        target = request.targets[0]
        candidate_ref = (
            "optimization-evidence-" + ("f" * 32)
            if self.outside_candidate
            else request.evidence[0].candidate_ref
        )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=OptimizationProposalDraft(
                target_ref=target.target_ref,
                proposed_value=target.constraints.allowed_values[0],
                expected_impact="Reduce bounded Initial Planning complexity.",
                applicable_conditions=("Same Initial Planning scope.",),
                limitations=("Proposal-only; Replay has not been performed.",),
                supporting_candidate_refs=(candidate_ref,),
            ),
        )


class FailingOptimizationCapability:
    module_id = "test.optimization.must_not_run"
    capability_id = "test.optimization.must_not_run"

    async def assess_optimization(
        self,
        request: OptimizationAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[OptimizationProposalDraft]:
        del request, invocation
        raise AssertionError("Optimization Agent was invoked during Resume")


class ChangingBaselineProvider(ResearchPlanningOptimizationBaselineProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def targets(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationTarget, ...]:
        self.calls += 1
        targets = await super().targets(scope)
        if self.calls < 3:
            return targets
        first = targets[0].model_copy(
            update={
                "current_value": 9,
                "current_value_fingerprint": (
                    "19581e27de7ced00ff1ce50b2047e7a567c76b1cbaeba8b915c362d0198554a4"
                ),
            }
        )
        return (first, *targets[1:])


class NoopPermitVerifier:
    async def verify(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class IndeterminateOptimizationStore:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def commit(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        return await self._delegate.commit(*args, **kwargs)

    async def load_by_effect(self, effect_fingerprint: str):  # type: ignore[no-untyped-def]
        del effect_fingerprint
        raise OSError("simulated indeterminate read")

    async def load_receipt(self, effect_fingerprint: str):  # type: ignore[no-untyped-def]
        return await self._delegate.load_receipt(effect_fingerprint)

    async def list_for_scope(self, scope: OptimizationScope):  # type: ignore[no-untyped-def]
        return await self._delegate.list_for_scope(scope)


async def _run_research(path: Path):  # type: ignore[no-untyped-def]
    agent = ResearchAgent(persistence_path=path)
    try:
        return await agent.run("分析 Tesla 投资价值")
    finally:
        agent.close()


def _governance(persistence: SQLitePersistence) -> RuntimeGovernanceEvaluator:
    return RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )


def _authority_fixture() -> tuple[
    OptimizationScope,
    OptimizationEvidenceBinding,
    OptimizationProposalEffect,
    GovernanceTarget,
    RuntimeCommitPermit,
]:
    scope = research_initial_planning_optimization_scope()
    counter_feedback_id = UUID(int=901)
    candidate = OptimizationEvidenceBinding(
        candidate_ref="optimization-evidence-" + ("b" * 32),
        candidate_fingerprint="c" * 64,
        scope=scope,
        learning_insight_id=UUID(int=902),
        learning_insight_version=1,
        learning_insight_effect_fingerprint="d" * 64,
        learning_insight_fingerprint="e" * 64,
        supporting_feedback_refs=(UUID(int=903),),
        supporting_experience_refs=(UUID(int=904),),
        supporting_evaluation_refs=(UUID(int=905),),
        supporting_artifact_effect_fingerprints=("f" * 64,),
        counterevidence_feedback_refs=(counter_feedback_id,),
        source_run_refs=(UUID(int=906), UUID(int=907)),
        observed_pattern="A verified cross-run pattern was observed.",
        applicable_conditions=("Same Initial Planning scope.",),
        limitations=("Association only.",),
    )
    target = OptimizationTarget(
        target_ref="target-" + ("a" * 32),
        target_type=OptimizationTargetType.INITIAL_PLANNING_POLICY,
        target_key=OptimizationTargetKey.PLANNER_MAX_NODES,
        current_value=8,
        current_value_fingerprint=(
            "2c624232cdd221771294dfbb310aca000a0df6ac8b66b696d90ef06fdefb64a3"
        ),
        configuration_source="test.planning.policy",
        constraints=OptimizationTargetConstraints(
            value_type="integer",
            minimum=1,
            maximum=32,
            allowed_values=(7, 9),
        ),
    )
    effect = OptimizationProposalEffect(
        proposal_id=UUID(int=908),
        scope=scope,
        target=target,
        proposed_value=7,
        evidence_snapshot=(candidate,),
        supporting_candidate_refs=(candidate.candidate_ref,),
        expected_impact="Bound Initial Planning complexity.",
        applicable_conditions=("Same Initial Planning scope.",),
        limitations=("Proposal only.",),
        risk_classification=OptimizationRiskClassification.LOW,
        rollback_requirements=("Retain the exact baseline.",),
        evidence_set_fingerprint=(
            "e1a46a7868407cc1f16f62f188ce6639a2a2396ee1a831af8c85ca9177b462ac"
        ),
        proposal_fingerprint="1" * 64,
        basis_fingerprint="2" * 64,
    )
    # Keep the fixture fingerprint tied to its candidate snapshot.
    from adaptive_agent_runtime.decisioning import decision_fingerprint

    effect = effect.model_copy(
        update={
            "evidence_set_fingerprint": decision_fingerprint(
                (candidate.candidate_fingerprint,)
            )
        }
    )
    target_binding = GovernanceTarget(
        target_type="optimization_proposal",
        target_id=str(effect.proposal_id),
    )
    permit = RuntimeCommitPermit(
        authorization_id=UUID(int=909),
        request_id=UUID(int=910),
        decision_id=UUID(int=911),
        operation="optimization.proposal.commit",
        target=target_binding,
        subject_fingerprint="3" * 64,
        issued_at=datetime.now(timezone.utc),
        integrity_seal="4" * 64,
    )
    return scope, candidate, effect, target_binding, permit


def _handler(
    persistence: SQLitePersistence,
    capability: OptimizationAssessmentCapability,
    *,
    baseline_provider: ResearchPlanningOptimizationBaselineProvider | None = None,
    fault_injector: Any = None,
) -> ResearchOptimizationProposalHandler:
    return ResearchOptimizationProposalHandler(
        evidence_resolver=persistence.optimization_evidence_resolver,
        baseline_provider=(
            baseline_provider or ResearchPlanningOptimizationBaselineProvider()
        ),
        store=persistence.optimization_proposal_store,
        capability=capability,
        governance=_governance(persistence),
        reviews=persistence.human_review_service,
        issuer=persistence.authorization_issuer,
        operation_executor=persistence.operation_executor,
        trace_sink=persistence.trace_sink,
        checkpoint_store=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                OptimizationAssessmentRequest,
                OptimizationProposalDraft,
                OptimizationProposalEffect,
            ]
        ),
        fault_injector=fault_injector,
    )


class OptimizationProposalContractTests(unittest.TestCase):
    def test_proposal_requires_explicit_scope(self) -> None:
        with self.assertRaises(ValidationError):
            OptimizationScope.model_validate(
                {"tenant": "default", "project": "research"}
            )

    def test_component_global_scope_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            OptimizationScope(
                tenant="global",
                project="research",
                application="research_agent",
                decision_type="planning.task_graph.initialize",
            )

    def test_proposal_requires_current_baseline(self) -> None:
        scope = research_initial_planning_optimization_scope()
        with self.assertRaises(ValidationError):
            OptimizationTarget.model_validate(
                {
                    "target_ref": "target-" + ("a" * 32),
                    "target_type": "initial_planning_policy",
                    "target_key": "planner.max_nodes",
                    "configuration_source": "test",
                    "constraints": {"value_type": "integer"},
                }
            )
        self.assertEqual(scope.application, "research_agent")

    def test_proposal_requires_learning_insight_evidence(self) -> None:
        scope, _candidate, effect, _target, _permit = _authority_fixture()
        with self.assertRaises(ValidationError):
            OptimizationAssessmentRequest.model_validate(
                {
                    "scope": scope.model_dump(mode="json"),
                    "trigger_run_id": str(uuid4()),
                    "targets": [effect.target.model_dump(mode="json")],
                    "candidates": [],
                    "evidence_set_fingerprint": "0" * 64,
                }
            )

    def test_proposal_preserves_counterevidence(self) -> None:
        _scope, candidate, effect, _target, _permit = _authority_fixture()
        self.assertEqual(
            effect.evidence_snapshot[0].counterevidence_feedback_refs,
            candidate.counterevidence_feedback_refs,
        )

    def test_arbitrary_config_patch_is_rejected(self) -> None:
        scope = research_initial_planning_optimization_scope()
        target = OptimizationTarget(
            target_ref="target-" + ("a" * 32),
            target_type=OptimizationTargetType.INITIAL_PLANNING_POLICY,
            target_key=OptimizationTargetKey.PLANNER_MAX_NODES,
            current_value=8,
            current_value_fingerprint=(
                "2c624232cdd221771294dfbb310aca000a0df6ac8b66b696d90ef06fdefb64a3"
            ),
            configuration_source="test",
            constraints=OptimizationTargetConstraints(
                value_type="integer", minimum=1, maximum=32
            ),
        )
        with self.assertRaises(ValueError):
            validate_optimization_value(target, {"max_nodes": 7})
        self.assertEqual(scope.project, "research")

    def test_agent_cannot_set_scope_or_current_value(self) -> None:
        base = {
            "target_ref": "target-" + ("a" * 32),
            "proposed_value": 7,
            "expected_impact": "Bound planning.",
            "applicable_conditions": ["same scope"],
            "limitations": ["proposal only"],
            "supporting_candidate_refs": [
                "optimization-evidence-" + ("b" * 32)
            ],
        }
        for injected in (
            {"scope": {"tenant": "other"}},
            {"current_value": 8},
        ):
            with self.assertRaises(ValidationError):
                OptimizationProposalDraft.model_validate({**base, **injected})

    def test_agent_cannot_request_apply_activation_or_rollback(self) -> None:
        for operation in (
            "optimization.apply",
            "configuration.activate",
            "optimization.rollback",
        ):
            with self.assertRaises(ValidationError):
                OptimizationProposalDraft.model_validate(
                    {
                        "target_ref": "target-" + ("a" * 32),
                        "proposed_value": 7,
                        "expected_impact": "Bound planning.",
                        "applicable_conditions": ["same scope"],
                        "limitations": ["proposal only"],
                        "supporting_candidate_refs": [
                            "optimization-evidence-" + ("b" * 32)
                        ],
                        "operation": operation,
                    }
                )


class OptimizationProposalStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_proposal_and_receipt_commit_atomically(self) -> None:
        scope, candidate, effect, target, permit = _authority_fixture()
        del scope
        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(Path(directory) / "runtime.sqlite3")
            store = SQLiteOptimizationProposalStore(
                database,
                permit_verifier=NoopPermitVerifier(),  # type: ignore[arg-type]
            )
            with database.transaction() as cursor:
                cursor.execute(
                    "CREATE TRIGGER fail_optimization_receipt "
                    "BEFORE INSERT ON optimization_proposal_receipts "
                    "BEGIN SELECT RAISE(ABORT, 'receipt failure'); END"
                )
            with patch(
                "adaptive_agent_runtime.persistence.optimization._resolve_candidates",
                return_value=(candidate,),
            ):
                with self.assertRaises(sqlite3.IntegrityError):
                    await store.commit(
                        effect,
                        source_decision_request_id=UUID(int=912),
                        effect_fingerprint="5" * 64,
                        permit=permit,
                        target=target,
                        subject_fingerprint="3" * 64,
                    )
            with database.reader() as cursor:
                proposal_count = cursor.execute(
                    "SELECT COUNT(*) AS count FROM optimization_proposals"
                ).fetchone()["count"]
                receipt_count = cursor.execute(
                    "SELECT COUNT(*) AS count FROM optimization_proposal_receipts"
                ).fetchone()["count"]
            self.assertEqual((proposal_count, receipt_count), (0, 0))
            database.close()

    async def test_duplicate_effect_returns_same_proposal(self) -> None:
        _scope, candidate, effect, target, permit = _authority_fixture()
        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(Path(directory) / "runtime.sqlite3")
            store = SQLiteOptimizationProposalStore(
                database,
                permit_verifier=NoopPermitVerifier(),  # type: ignore[arg-type]
            )
            with patch(
                "adaptive_agent_runtime.persistence.optimization._resolve_candidates",
                return_value=(candidate,),
            ):
                first = await store.commit(
                    effect,
                    source_decision_request_id=UUID(int=913),
                    effect_fingerprint="6" * 64,
                    permit=permit,
                    target=target,
                    subject_fingerprint="3" * 64,
                )
                second = await store.commit(
                    effect,
                    source_decision_request_id=UUID(int=913),
                    effect_fingerprint="6" * 64,
                    permit=permit,
                    target=target,
                    subject_fingerprint="3" * 64,
                )
            self.assertEqual(second, first)
            self.assertIn("feedback:00000000-0000-0000-0000-000000000385", first.counterevidence_refs)
            receipt = await store.load_receipt("6" * 64)
            self.assertIsNotNone(receipt)
            database.close()

    async def test_proposal_store_does_not_write_active_configuration(self) -> None:
        _scope, candidate, effect, target, permit = _authority_fixture()
        with TemporaryDirectory() as directory:
            database = SQLiteDatabase(Path(directory) / "runtime.sqlite3")
            store = SQLiteOptimizationProposalStore(
                database,
                permit_verifier=NoopPermitVerifier(),  # type: ignore[arg-type]
            )
            with patch(
                "adaptive_agent_runtime.persistence.optimization._resolve_candidates",
                return_value=(candidate,),
            ):
                await store.commit(
                    effect,
                    source_decision_request_id=UUID(int=914),
                    effect_fingerprint="7" * 64,
                    permit=permit,
                    target=target,
                    subject_fingerprint="3" * 64,
                )
            with database.reader() as cursor:
                active = cursor.execute(
                    "SELECT COUNT(*) AS count FROM runtime_configuration_active"
                ).fetchone()["count"]
                history = cursor.execute(
                    "SELECT COUNT(*) AS count FROM runtime_configuration_snapshots"
                ).fetchone()["count"]
                deployments = cursor.execute(
                    "SELECT COUNT(*) AS count FROM optimization_application_current"
                ).fetchone()["count"]
            self.assertEqual((active, history, deployments), (0, 0, 0))
            database.close()


class GovernedOptimizationProposalTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end_phase3_evidence_commits_proposal_only(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            first = await _run_research(path)
            second = await _run_research(path)
            self.assertEqual(first.optimization_proposals, ())
            self.assertEqual(len(second.optimization_proposals), 1)
            proposal = second.optimization_proposals[0]
            self.assertEqual(proposal.scope, research_initial_planning_optimization_scope())
            self.assertEqual(proposal.target_key, OptimizationTargetKey.PLANNER_MAX_NODES)
            self.assertEqual(proposal.current_value, 8)
            self.assertEqual(proposal.proposed_value, 9)
            self.assertTrue(proposal.supporting_learning_insight_refs)
            self.assertTrue(proposal.supporting_feedback_refs)
            self.assertTrue(proposal.supporting_experience_refs)
            self.assertTrue(proposal.supporting_evaluation_refs)
            operation = next(
                record.request.operation
                for record in second.governance_records
                if record.scenario == "optimization_proposal"
            )
            self.assertEqual(operation, "optimization.proposal.commit")

            persistence = SQLitePersistence(path)
            try:
                with persistence.database.reader() as cursor:
                    proposal_count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM optimization_proposals"
                    ).fetchone()["count"]
                    receipt_count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM optimization_proposal_receipts"
                    ).fetchone()["count"]
                    active_count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM runtime_configuration_active"
                    ).fetchone()["count"]
                    application_count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM optimization_application_current"
                    ).fetchone()["count"]
                self.assertEqual(proposal_count, receipt_count)
                self.assertEqual(active_count, 0)
                self.assertEqual(application_count, 0)
                stored = await persistence.optimization_proposal_store.load_by_effect(
                    proposal.effect_fingerprint
                )
                self.assertEqual(stored, proposal)
            finally:
                persistence.close()
            self.assertEqual(RESEARCH_PLANNING_GRAPH_LIMITS.max_nodes, 8)

    async def test_agent_cannot_select_evidence_outside_candidate_set(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            await _run_research(path)
            persistence = SQLitePersistence(path)
            try:
                handler = _handler(
                    persistence,
                    RecordingOptimizationCapability(outside_candidate=True),
                )
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await handler.assess(
                        trigger_run_id=uuid4(),
                        task_id=uuid4(),
                        scope=research_initial_planning_optimization_scope(),
                    )
            finally:
                persistence.close()

    async def test_stale_baseline_rejects_effect(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            await _run_research(path)
            persistence = SQLitePersistence(path)
            try:
                before = await persistence.optimization_proposal_store.list_for_scope(
                    research_initial_planning_optimization_scope()
                )
                handler = _handler(
                    persistence,
                    RecordingOptimizationCapability(),
                    baseline_provider=ChangingBaselineProvider(),
                )
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await handler.assess(
                        trigger_run_id=uuid4(),
                        task_id=uuid4(),
                        scope=research_initial_planning_optimization_scope(),
                    )
                after = await persistence.optimization_proposal_store.list_for_scope(
                    research_initial_planning_optimization_scope()
                )
                self.assertEqual(after, before)
            finally:
                persistence.close()

    async def test_source_change_before_apply_rejects_stale_effect(self) -> None:
        class CrashAfterAuthorization(RuntimeError):
            pass

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            await _run_research(path)
            trigger_run_id = uuid4()
            task_id = uuid4()

            def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                if point is DecisionFaultPoint.AUTHORIZED:
                    raise CrashAfterAuthorization("simulated crash")

            first = SQLitePersistence(path)
            try:
                with self.assertRaises(CrashAfterAuthorization):
                    await _handler(
                        first,
                        RecordingOptimizationCapability(),
                        fault_injector=inject,
                    ).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=task_id,
                        scope=research_initial_planning_optimization_scope(),
                    )
                with first.database.transaction() as cursor:
                    row = cursor.execute(
                        "SELECT effect_fingerprint, insight_json "
                        "FROM learning_insights ORDER BY version DESC LIMIT 1"
                    ).fetchone()
                    assert row is not None
                    payload = json.loads(row["insight_json"])
                    payload["observed_pattern"] = "tampered source evidence"
                    cursor.execute(
                        "UPDATE learning_insights SET insight_json = ? "
                        "WHERE effect_fingerprint = ?",
                        (json.dumps(payload), row["effect_fingerprint"]),
                    )
            finally:
                first.close()

            reopened = SQLitePersistence(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await _handler(
                        reopened,
                        FailingOptimizationCapability(),
                    ).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=task_id,
                        scope=research_initial_planning_optimization_scope(),
                    )
                request_id = stable_optimization_request_id(
                    trigger_run_id,
                    research_initial_planning_optimization_scope(),
                )
                checkpoint = await reopened.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationAssessmentRequest,
                        OptimizationProposalDraft,
                        OptimizationProposalEffect,
                    ]
                ).load(request_id)
                assert checkpoint is not None and checkpoint.result is not None
                self.assertEqual(checkpoint.result.status, DecisionResultStatus.FAILED)
                with reopened.database.reader() as cursor:
                    count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM optimization_proposals "
                        "WHERE source_decision_request_id = ?",
                        (str(request_id),),
                    ).fetchone()["count"]
                self.assertEqual(count, 0)
            finally:
                reopened.close()

    async def test_proposal_resume_does_not_call_agent_again(self) -> None:
        class CrashAfterCommit(RuntimeError):
            pass

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            await _run_research(path)
            trigger_run_id = uuid4()
            task_id = uuid4()
            capability = RecordingOptimizationCapability()

            def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                if point is DecisionFaultPoint.EFFECT_COMMITTED:
                    raise CrashAfterCommit("simulated crash")

            first = SQLitePersistence(path)
            try:
                with self.assertRaises(CrashAfterCommit):
                    await _handler(
                        first,
                        capability,
                        fault_injector=inject,
                    ).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=task_id,
                        scope=research_initial_planning_optimization_scope(),
                    )
                self.assertEqual(capability.calls, 1)
            finally:
                first.close()

            reopened = SQLitePersistence(path)
            try:
                resumed = await _handler(
                    reopened,
                    FailingOptimizationCapability(),
                ).assess(
                    trigger_run_id=trigger_run_id,
                    task_id=task_id,
                    scope=research_initial_planning_optimization_scope(),
                )
                self.assertIsNotNone(resumed.proposal)
                request_id = stable_optimization_request_id(
                    trigger_run_id,
                    research_initial_planning_optimization_scope(),
                )
                checkpoint = await reopened.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationAssessmentRequest,
                        OptimizationProposalDraft,
                        OptimizationProposalEffect,
                    ]
                ).load(request_id)
                assert checkpoint is not None and checkpoint.result is not None
                self.assertEqual(checkpoint.result.status, DecisionResultStatus.APPLIED)
                with reopened.database.reader() as cursor:
                    count = cursor.execute(
                        "SELECT COUNT(*) AS count FROM optimization_proposals "
                        "WHERE source_decision_request_id = ?",
                        (str(request_id),),
                    ).fetchone()["count"]
                self.assertEqual(count, 1)
            finally:
                reopened.close()

    async def test_same_fingerprint_different_payload_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            second = await _run_research(path)
            proposal = second.optimization_proposals[0]
            scope = research_initial_planning_optimization_scope()
            request_id = stable_optimization_request_id(
                second.runtime_result.final_state.run_id,
                scope,
            )
            persistence = SQLitePersistence(path)
            try:
                checkpoint = await persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationAssessmentRequest,
                        OptimizationProposalDraft,
                        OptimizationProposalEffect,
                    ]
                ).load(request_id)
                assert checkpoint is not None and checkpoint.validated_decision is not None
                normalized = checkpoint.validated_decision.normalized_effect
                altered = normalized.payload.model_copy(update={"proposed_value": 10})
                target = GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                )
                permit = RuntimeCommitPermit(
                    authorization_id=uuid4(),
                    request_id=uuid4(),
                    decision_id=uuid4(),
                    operation="optimization.proposal.commit",
                    target=target,
                    subject_fingerprint="0" * 64,
                    issued_at=datetime.now(timezone.utc),
                    integrity_seal="0" * 64,
                )
                permissive_store = SQLiteOptimizationProposalStore(
                    persistence.database,
                    permit_verifier=NoopPermitVerifier(),  # type: ignore[arg-type]
                )
                with self.assertRaises(PersistenceConflictError):
                    await permissive_store.commit(
                        altered,
                        source_decision_request_id=request_id,
                        effect_fingerprint=proposal.effect_fingerprint,
                        permit=permit,
                        target=target,
                        subject_fingerprint="0" * 64,
                    )
            finally:
                persistence.close()

    async def test_unknown_proposal_apply_is_durable(self) -> None:
        class CrashWhileApplying(RuntimeError):
            pass

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            await _run_research(path)
            trigger_run_id = uuid4()
            task_id = uuid4()

            def inject(point: DecisionFaultPoint, checkpoint: Any) -> None:
                if point is DecisionFaultPoint.APPLYING:
                    raise CrashWhileApplying("simulated crash")

            first = SQLitePersistence(path)
            try:
                with self.assertRaises(CrashWhileApplying):
                    await _handler(
                        first,
                        RecordingOptimizationCapability(),
                        fault_injector=inject,
                    ).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=task_id,
                        scope=research_initial_planning_optimization_scope(),
                    )
            finally:
                first.close()

            reopened = SQLitePersistence(path)
            try:
                handler = ResearchOptimizationProposalHandler(
                    evidence_resolver=reopened.optimization_evidence_resolver,
                    baseline_provider=ResearchPlanningOptimizationBaselineProvider(),
                    store=IndeterminateOptimizationStore(
                        reopened.optimization_proposal_store
                    ),  # type: ignore[arg-type]
                    capability=FailingOptimizationCapability(),
                    governance=_governance(reopened),
                    reviews=reopened.human_review_service,
                    issuer=reopened.authorization_issuer,
                    operation_executor=reopened.operation_executor,
                    trace_sink=reopened.trace_sink,
                    checkpoint_store=reopened.create_decision_checkpoint_store(
                        DecisionCheckpoint[
                            OptimizationAssessmentRequest,
                            OptimizationProposalDraft,
                            OptimizationProposalEffect,
                        ]
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await handler.assess(
                        trigger_run_id=trigger_run_id,
                        task_id=task_id,
                        scope=research_initial_planning_optimization_scope(),
                    )
                request_id = stable_optimization_request_id(
                    trigger_run_id,
                    research_initial_planning_optimization_scope(),
                )
                checkpoint_store = reopened.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        OptimizationAssessmentRequest,
                        OptimizationProposalDraft,
                        OptimizationProposalEffect,
                    ]
                )
                checkpoint = await checkpoint_store.load(request_id)
                assert checkpoint is not None and checkpoint.result is not None
                self.assertEqual(checkpoint.result.status, DecisionResultStatus.FAILED)
                self.assertEqual(
                    checkpoint.result.reconciliation_status,
                    "unknown",
                )
                # A second Resume observes the same terminal UNKNOWN result and
                # still cannot invoke the Agent or apply the Effect.
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await handler.assess(
                        trigger_run_id=trigger_run_id,
                        task_id=task_id,
                        scope=research_initial_planning_optimization_scope(),
                    )
                repeated = await checkpoint_store.load(request_id)
                self.assertEqual(repeated, checkpoint)
            finally:
                reopened.close()

    async def test_stored_proposal_does_not_change_planner_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            second = await _run_research(path)
            self.assertEqual(len(second.optimization_proposals), 1)
            targets = await ResearchPlanningOptimizationBaselineProvider().targets(
                research_initial_planning_optimization_scope()
            )
            current = {item.target_key: item.current_value for item in targets}
            self.assertEqual(current[OptimizationTargetKey.PLANNER_MAX_NODES], 8)
            self.assertEqual(current[OptimizationTargetKey.PLANNER_MAX_DEPTH], 5)
            self.assertEqual(RESEARCH_PLANNING_GRAPH_LIMITS.max_nodes, 8)

    async def test_stored_proposal_does_not_change_runtime_behavior(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_research(path)
            second = await _run_research(path)
            self.assertEqual(len(second.optimization_proposals), 1)
            third = await _run_research(path)
            self.assertEqual(third.runtime_result.final_state.status.value, "completed")
            self.assertEqual(len(third.task_graph.nodes), 8)
            self.assertEqual(RESEARCH_PLANNING_GRAPH_LIMITS.max_nodes, 8)


if __name__ == "__main__":
    unittest.main()
