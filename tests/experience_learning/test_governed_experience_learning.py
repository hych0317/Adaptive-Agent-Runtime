from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from pydantic import ValidationError

from adaptive_agent_runtime import InMemoryTraceSink
from adaptive_agent_runtime.context_memory import MemoryScope
from adaptive_agent_runtime.decision_feedback import DecisionFeedbackAttributionType
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionFaultPoint,
    decision_fingerprint,
)
from adaptive_agent_runtime.experience_learning import (
    LearningAssessmentRequest,
    LearningInsightDraft,
    LearningInsightEffect,
    stable_learning_request_id,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    ConfidenceSignals,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    ImpactAssessment,
    RiskLevel,
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    default_governance_policy,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import CapabilityTurnKind, CapabilityTurnResult
from adaptive_agent_runtime.orchestration import PLANNING_DECISION_TYPE
from adaptive_agent_runtime.persistence import SQLitePersistence

from applications.research_agent.agent import ResearchAgent
from applications.research_agent.experience_learning import (
    DeterministicLearningAssessmentCapability,
    LearningAssessmentCapability,
    ResearchExperienceLearningHandler,
    _validate_non_behavioral_draft,
)
from applications.research_agent.report import ResearchRunResult


LEARNING_SCOPE = MemoryScope(
    tenant_id="default", project_id="research", agent_scope="planner"
)


async def _run_once(
    path: Path,
    task: str = "分析 Acme 的投资价值",
) -> ResearchRunResult:
    agent = ResearchAgent(persistence_path=path)
    try:
        return await agent.run(task)
    finally:
        agent.close()


async def _run_twice(
    path: Path,
) -> tuple[ResearchRunResult, ResearchRunResult]:
    first = await _run_once(path, "分析 Acme 的投资价值")
    second = await _run_once(path, "分析 Acme 的长期投资价值")
    return first, second


def _handler(
    persistence: SQLitePersistence,
    capability: LearningAssessmentCapability | None = None,
    *,
    fault_injector: Callable[..., None] | None = None,
) -> ResearchExperienceLearningHandler:
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )
    return ResearchExperienceLearningHandler(
        store=persistence.learning_insight_store,
        capability=capability or DeterministicLearningAssessmentCapability(),
        governance=governance,
        reviews=persistence.human_review_service,
        issuer=persistence.authorization_issuer,
        operation_executor=persistence.operation_executor,
        trace_sink=InMemoryTraceSink(),
        checkpoint_store=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                LearningAssessmentRequest,
                LearningInsightDraft,
                LearningInsightEffect,
            ]
        ),
        fault_injector=fault_injector,
    )


class OutsideCandidateCapability:
    module_id = "test.learning.outside"
    capability_id = "test.learning.outside"

    async def assess_learning(self, request, *, invocation=None):  # type: ignore[no-untyped-def]
        del invocation
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=LearningInsightDraft(
                observed_pattern="A non-causal pattern was observed.",
                applicable_conditions=("Same verified scope.",),
                limitations=("Association only.",),
                supporting_candidate_refs=(
                    request.evidence[0].candidate_ref,
                    "evidence-00000000000000000000000000000000",
                ),
            ),
        )


class ConflictingEvidenceCapability:
    module_id = "test.learning.conflict"
    capability_id = "test.learning.conflict"

    async def assess_learning(self, request, *, invocation=None):  # type: ignore[no-untyped-def]
        del invocation
        eligible = tuple(item for item in request.evidence if item.conclusion_eligible)
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=LearningInsightDraft(
                observed_pattern="Different verified outcomes were observed.",
                applicable_conditions=("Same verified scope.",),
                limitations=("Association only; evidence is conflicting.",),
                supporting_candidate_refs=(eligible[0].candidate_ref,),
                counterevidence_candidate_refs=(eligible[1].candidate_ref,),
            ),
        )


class CountingLearningCapability(DeterministicLearningAssessmentCapability):
    module_id = "test.learning.counting"
    capability_id = "test.learning.counting"

    def __init__(self) -> None:
        self.calls = 0

    async def assess_learning(self, request, *, invocation=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return await super().assess_learning(request, invocation=invocation)


class GovernedExperienceLearningTests(unittest.IsolatedAsyncioTestCase):
    async def test_learning_requires_multiple_completed_runs(self) -> None:
        with TemporaryDirectory() as directory:
            result = await _run_once(Path(directory) / "runtime.sqlite3")
            self.assertEqual(result.learning_insights, ())

    async def test_learning_requires_applied_feedback_and_experience(self) -> None:
        for missing_table in ("decision_feedback", "experience_metadata"):
            with self.subTest(missing_table=missing_table), TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.sqlite3"
                first, second = await _run_twice(path)
                target_run = str(second.runtime_result.final_state.run_id)
                connection = sqlite3.connect(path)
                try:
                    if missing_table == "decision_feedback":
                        connection.execute(
                            "DELETE FROM decision_feedback WHERE source_run_id = ?",
                            (target_run,),
                        )
                    else:
                        effects = tuple(
                            row[0]
                            for row in connection.execute(
                                "SELECT effect_fingerprint FROM experience_metadata "
                                "WHERE source_run_id = ?",
                                (target_run,),
                            ).fetchall()
                        )
                        for effect in effects:
                            connection.execute(
                                "DELETE FROM experience_memory_links "
                                "WHERE effect_fingerprint = ?",
                                (effect,),
                            )
                        connection.execute(
                            "DELETE FROM experience_metadata WHERE source_run_id = ?",
                            (target_run,),
                        )
                    connection.commit()
                finally:
                    connection.close()
                persistence = SQLitePersistence(path)
                try:
                    candidates = await persistence.learning_insight_store.resolve_candidates(
                        scope=LEARNING_SCOPE,
                        subject_decision_type=PLANNING_DECISION_TYPE,
                    )
                    self.assertEqual(
                        {item.source_run_id for item in candidates},
                        {first.runtime_result.final_state.run_id},
                    )
                finally:
                    persistence.close()

    async def test_insufficient_evidence_does_not_become_positive_signal(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_twice(path)
            persistence = SQLitePersistence(path)
            try:
                candidates = await persistence.learning_insight_store.resolve_candidates(
                    scope=LEARNING_SCOPE,
                    subject_decision_type=PLANNING_DECISION_TYPE,
                )
                changed = candidates[1].model_copy(
                    update={
                        "attribution_type": (
                            DecisionFeedbackAttributionType.INSUFFICIENT_EVIDENCE
                        )
                    }
                )
                with self.assertRaisesRegex(ValidationError, "multiple independent"):
                    LearningAssessmentRequest(
                        scope=LEARNING_SCOPE,
                        subject_decision_type=PLANNING_DECISION_TYPE,
                        trigger_run_id=uuid4(),
                        candidates=(candidates[0], changed),
                        evidence_set_fingerprint="0" * 64,
                    )
            finally:
                persistence.close()

    async def test_missing_feedback_is_not_interpreted_as_success(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_once(path)
            persistence = SQLitePersistence(path)
            try:
                with persistence.database.transaction() as cursor:
                    cursor.execute("DELETE FROM decision_feedback")
                result = await _handler(persistence).assess(
                    trigger_run_id=uuid4(),
                    task_id=uuid4(),
                    scope=LEARNING_SCOPE,
                    subject_decision_type=PLANNING_DECISION_TYPE,
                )
                self.assertIsNone(result.insight)
                self.assertEqual(result.eligible_candidate_count, 0)
            finally:
                persistence.close()

    def test_agent_cannot_propose_reward_ranking_or_configuration(self) -> None:
        base = {
            "observed_pattern": "A pattern was observed.",
            "applicable_conditions": ["Same scope."],
            "limitations": ["Association only."],
            "supporting_candidate_refs": [
                "evidence-00000000000000000000000000000000"
            ],
        }
        for forbidden in ("reward", "strategy_ranking", "configuration_patch"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValidationError):
                LearningInsightDraft.model_validate({**base, forbidden: "forbidden"})
        with self.assertRaisesRegex(ValueError, "behavior-changing"):
            _validate_non_behavioral_draft(
                LearningInsightDraft.model_validate(
                    {**base, "observed_pattern": "Must change configuration."}
                )
            )

    async def test_agent_cannot_select_evidence_outside_candidate_set(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _, second = await _run_twice(path)
            persistence = SQLitePersistence(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await _handler(persistence, OutsideCandidateCapability()).assess(
                        trigger_run_id=uuid4(),
                        task_id=second.runtime_result.final_state.task.task_id,
                        scope=LEARNING_SCOPE,
                        subject_decision_type=PLANNING_DECISION_TYPE,
                    )
            finally:
                persistence.close()

    async def test_cross_scope_evidence_is_excluded(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await _run_twice(path)
            persistence = SQLitePersistence(path)
            try:
                for scope in (
                    MemoryScope(tenant_id="other", project_id="research", agent_scope="planner"),
                    MemoryScope(tenant_id="default", project_id="other", agent_scope="planner"),
                    MemoryScope(tenant_id="default", project_id="research", agent_scope="other"),
                ):
                    self.assertEqual(
                        await persistence.learning_insight_store.resolve_candidates(
                            scope=scope,
                            subject_decision_type=PLANNING_DECISION_TYPE,
                        ),
                        (),
                    )
            finally:
                persistence.close()

    async def test_conflicting_evidence_is_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _, second = await _run_twice(path)
            persistence = SQLitePersistence(path)
            try:
                result = await _handler(
                    persistence, ConflictingEvidenceCapability()
                ).assess(
                    trigger_run_id=uuid4(),
                    task_id=second.runtime_result.final_state.task.task_id,
                    scope=LEARNING_SCOPE,
                    subject_decision_type=PLANNING_DECISION_TYPE,
                )
                assert result.insight is not None
                self.assertEqual(len(result.insight.supporting_evidence_refs), 1)
                self.assertEqual(len(result.insight.counterevidence_refs), 1)
            finally:
                persistence.close()

    async def test_learning_insight_is_append_only(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _, second = await _run_twice(path)
            first_insight = second.learning_insights[0]
            third = await _run_once(path, "分析 Acme 的风险调整后投资价值")
            persistence = SQLitePersistence(path)
            try:
                records = await persistence.learning_insight_store.list_for_scope(
                    LEARNING_SCOPE,
                    subject_decision_type=PLANNING_DECISION_TYPE,
                )
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0], first_insight)
                self.assertEqual(third.learning_insights, (records[1],))
            finally:
                persistence.close()

    async def test_duplicate_effect_returns_same_insight_and_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _, second = await _run_twice(path)
            insight = second.learning_insights[0]
            persistence = SQLitePersistence(path)
            try:
                receipt_before = await persistence.learning_insight_store.load_receipt(
                    insight.effect_fingerprint
                )
                resumed = await _handler(persistence).assess(
                    trigger_run_id=second.runtime_result.final_state.run_id,
                    task_id=second.runtime_result.final_state.task.task_id,
                    scope=LEARNING_SCOPE,
                    subject_decision_type=PLANNING_DECISION_TYPE,
                )
                receipt_after = await persistence.learning_insight_store.load_receipt(
                    insight.effect_fingerprint
                )
                self.assertEqual(resumed.insight, insight)
                self.assertEqual(receipt_before, receipt_after)
            finally:
                persistence.close()

    async def test_same_fingerprint_different_payload_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _, second = await _run_twice(path)
            insight = second.learning_insights[0]
            persistence = SQLitePersistence(path)
            try:
                checkpoints = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        LearningAssessmentRequest,
                        LearningInsightDraft,
                        LearningInsightEffect,
                    ]
                )
                checkpoint = await checkpoints.load(
                    stable_learning_request_id(
                        second.runtime_result.final_state.run_id,
                        LEARNING_SCOPE,
                        PLANNING_DECISION_TYPE,
                    )
                )
                assert checkpoint is not None
                assert checkpoint.validated_decision is not None
                normalized = checkpoint.validated_decision.normalized_effect
                changed = normalized.payload.model_copy(
                    update={"observed_pattern": "A different observed pattern."}
                )
                target = GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                )
                subject = {"tampered_learning": changed.model_dump(mode="json")}
                subject_fingerprint = governance_fingerprint(subject)
                request = GovernanceRequest(
                    scope=GovernanceScope.STATE,
                    operation="learning.insight.commit",
                    target=target,
                    risk=RiskLevel.LOW,
                    signals=ConfidenceSignals(
                        stated_confidence=1.0,
                        impact=ImpactAssessment(
                            score=0.1,
                            reversible=False,
                            description="Learning conflict test.",
                        ),
                    ),
                    attributes={
                        SUBJECT_FINGERPRINT_ATTRIBUTE: subject_fingerprint
                    },
                )
                evaluator = RuntimeGovernanceEvaluator(
                    policy=default_governance_policy(),
                    rule_evaluator=DeterministicRuleEvaluator(),
                    confidence_evaluator=DeterministicConfidenceEvaluator(),
                    review_service=persistence.human_review_service,
                )
                decision = evaluator.evaluate(request)
                authorization = persistence.authorization_issuer.issue(
                    request, decision
                )

                async def forbidden_raw() -> None:
                    raise AssertionError("Permit callback required")

                async def commit_changed(permit):  # type: ignore[no-untyped-def]
                    return await persistence.learning_insight_store.commit(
                        changed,
                        effect_fingerprint=insight.effect_fingerprint,
                        permit=permit,
                        target=target,
                        subject_fingerprint=subject_fingerprint,
                    )

                with self.assertRaisesRegex(Exception, "fingerprint conflicts"):
                    await persistence.operation_executor.execute(
                        request=request,
                        decision=decision,
                        authorization=authorization,
                        target=BoundGovernedOperation(
                            module_id="test.learning.conflict",
                            operation="learning.insight.commit",
                            target=target,
                            subject=subject,
                            apply=forbidden_raw,
                            apply_with_permit=commit_changed,
                        ),
                    )
            finally:
                persistence.close()

    async def test_learning_insight_does_not_change_planning_or_recall(self) -> None:
        project = Path(__file__).resolve().parents[2]
        for relative in (
            "applications/research_agent/planning.py",
            "applications/research_agent/memory_recall.py",
        ):
            tree = ast.parse((project / relative).read_text(encoding="utf-8"))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            self.assertNotIn("adaptive_agent_runtime.experience_learning", imported)

    async def test_unknown_learning_apply_is_durable_and_not_retried(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            _, second = await _run_twice(path)
            persistence = SQLitePersistence(path)
            capability = CountingLearningCapability()
            trigger_run_id = uuid4()

            def fault(point, checkpoint):  # type: ignore[no-untyped-def]
                if point is DecisionFaultPoint.EFFECT_COMMITTED:
                    raise RuntimeError("crash after Learning commit")

            try:
                with self.assertRaisesRegex(RuntimeError, "crash"):
                    await _handler(
                        persistence,
                        capability,
                        fault_injector=fault,
                    ).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=second.runtime_result.final_state.task.task_id,
                        scope=LEARNING_SCOPE,
                        subject_decision_type=PLANNING_DECISION_TYPE,
                    )
                request_id = stable_learning_request_id(
                    trigger_run_id, LEARNING_SCOPE, PLANNING_DECISION_TYPE
                )
                checkpoints = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        LearningAssessmentRequest,
                        LearningInsightDraft,
                        LearningInsightEffect,
                    ]
                )
                checkpoint = await checkpoints.load(request_id)
                assert checkpoint is not None
                assert checkpoint.validated_decision is not None
                fingerprint = (
                    checkpoint.validated_decision.normalized_effect.effect_fingerprint
                )
                with persistence.database.transaction() as cursor:
                    cursor.execute(
                        "UPDATE learning_insights SET insight_json = ? "
                        "WHERE effect_fingerprint = ?",
                        ("{corrupt", fingerprint),
                    )
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await _handler(persistence, capability).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=second.runtime_result.final_state.task.task_id,
                        scope=LEARNING_SCOPE,
                        subject_decision_type=PLANNING_DECISION_TYPE,
                    )
                calls = capability.calls
                with self.assertRaisesRegex(RuntimeError, "did not apply"):
                    await _handler(persistence, capability).assess(
                        trigger_run_id=trigger_run_id,
                        task_id=second.runtime_result.final_state.task.task_id,
                        scope=LEARNING_SCOPE,
                        subject_decision_type=PLANNING_DECISION_TYPE,
                    )
                self.assertEqual(capability.calls, calls)
                completed = await checkpoints.load(request_id)
                assert completed is not None and completed.result is not None
                assert completed.result.reconciliation_status is not None
                self.assertEqual(
                    completed.result.reconciliation_status.value,
                    "unknown",
                )
            finally:
                persistence.close()

    def test_core_event_loop_remains_unchanged(self) -> None:
        project = Path(__file__).resolve().parents[2]
        tree = ast.parse(
            (project / "src/adaptive_agent_runtime/core/runtime.py").read_text(
                encoding="utf-8"
            )
        )
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("adaptive_agent_runtime.experience_learning", imported)

    async def test_end_to_end_multiple_runs_commit_auditable_insight(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            first, second = await _run_twice(path)
            self.assertEqual(first.learning_insights, ())
            self.assertEqual(len(second.learning_insights), 1)
            insight = second.learning_insights[0]
            self.assertEqual(
                set(insight.source_run_refs),
                {
                    first.runtime_result.final_state.run_id,
                    second.runtime_result.final_state.run_id,
                },
            )
            persistence = SQLitePersistence(path)
            try:
                readback = await persistence.learning_insight_store.load_by_effect(
                    insight.effect_fingerprint
                )
                receipt = await persistence.learning_insight_store.load_receipt(
                    insight.effect_fingerprint
                )
                self.assertEqual(readback, insight)
                self.assertEqual(receipt.insight_fingerprint, decision_fingerprint(insight))  # type: ignore[union-attr]
            finally:
                persistence.close()


if __name__ == "__main__":
    unittest.main()
