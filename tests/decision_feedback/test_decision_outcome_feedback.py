from __future__ import annotations

import ast
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from adaptive_agent_runtime import RunStatus
from adaptive_agent_runtime.decision_feedback import (
    DECISION_FEEDBACK_DECISION_TYPE,
    DecisionFeedbackAttributionType,
    DecisionFeedbackDraft,
    DecisionFeedbackEffect,
    DecisionFeedbackRequest,
)
from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionCheckpoint,
    DecisionFaultPoint,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    ConfidenceSignals,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    GovernedOperationError,
    ImpactAssessment,
    RiskLevel,
    RuntimeGovernanceEvaluator,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    default_governance_policy,
    governance_fingerprint,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.decision_feedback import (
    FEEDBACK_INPUT_SOURCE_TYPE,
    ResearchDecisionFeedbackHandler,
    _DeterministicFeedbackProducer,
)
from tests.research_agent.test_initial_memory_recall import (
    RecallAwarePlanner,
    SelectingRecallAgent,
)
from applications.research_agent.cognition import ResearchCognitiveCapabilities


class FeedbackCrash(RuntimeError):
    pass


def _handler(persistence: SQLitePersistence) -> ResearchDecisionFeedbackHandler:
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )
    return ResearchDecisionFeedbackHandler(
        state_store=persistence.state_store,
        evaluation_store=persistence.evaluation_report_store,
        feedback_store=persistence.decision_feedback_store,
        governance=governance,
        reviews=persistence.human_review_service,
        issuer=persistence.authorization_issuer,
        operation_executor=persistence.operation_executor,
        trace_sink=persistence.trace_sink,
        checkpoint_store=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                DecisionFeedbackRequest,
                DecisionFeedbackDraft,
                DecisionFeedbackEffect,
            ]
        ),
    )


class DecisionOutcomeFeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, path: Path):  # type: ignore[no-untyped-def]
        agent = ResearchAgent(persistence_path=path)
        try:
            return await agent.run("分析 Acme 的投资价值")
        finally:
            agent.close()

    async def test_planning_feedback_requires_completed_run(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                running = result.runtime_result.final_state.model_copy(
                    update={"status": RunStatus.RUNNING, "output": None}
                )
                with self.assertRaisesRegex(ValueError, "completed Run"):
                    await _handler(persistence).record_for_run(
                        final_state=running,
                        evaluation=result.evaluation,
                        artifact_receipt=result.report_commit_receipt,
                        experience=result.experience_metadata,  # type: ignore[arg-type]
                        recall_bundle=None,
                    )
            finally:
                persistence.close()

    async def test_feedback_requires_applied_subject_decision(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            connection = sqlite3.connect(path)
            try:
                planning_id = str(result.decision_feedback[0].subject_decision_id)
                earliest = connection.execute(
                    "SELECT MIN(revision) FROM decision_checkpoints WHERE request_id = ?",
                    (planning_id,),
                ).fetchone()[0]
                connection.execute(
                    "UPDATE decision_checkpoint_current SET revision = ? "
                    "WHERE request_id = ?",
                    (earliest, planning_id),
                )
                connection.commit()
            finally:
                connection.close()
            persistence = SQLitePersistence(path)
            try:
                with self.assertRaisesRegex(ValueError, "APPLIED Planning"):
                    await _handler(persistence).record_for_run(
                        final_state=result.runtime_result.final_state,
                        evaluation=result.evaluation,
                        artifact_receipt=result.report_commit_receipt,
                        experience=result.experience_metadata,  # type: ignore[arg-type]
                        recall_bundle=None,
                    )
            finally:
                persistence.close()

    async def test_mismatched_evaluation_run_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            first = await self._run(path)
            second = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                with self.assertRaisesRegex(ValueError, "Evaluation identity"):
                    await _handler(persistence).record_for_run(
                        final_state=first.runtime_result.final_state,
                        evaluation=second.evaluation,
                        artifact_receipt=first.report_commit_receipt,
                        experience=first.experience_metadata,  # type: ignore[arg-type]
                        recall_bundle=None,
                    )
            finally:
                persistence.close()

    async def test_mismatched_artifact_provenance_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                forged = result.report_commit_receipt.model_copy(
                    update={"run_id": uuid4()}
                )
                with self.assertRaisesRegex(ValueError, "another run"):
                    await _handler(persistence).record_for_run(
                        final_state=result.runtime_result.final_state,
                        evaluation=result.evaluation,
                        artifact_receipt=forged,
                        experience=result.experience_metadata,  # type: ignore[arg-type]
                        recall_bundle=None,
                    )
            finally:
                persistence.close()

    async def test_planning_feedback_records_runtime_outcome(self) -> None:
        with TemporaryDirectory() as directory:
            result = await self._run(Path(directory) / "runtime.sqlite3")
            self.assertEqual(len(result.decision_feedback), 1)
            feedback = result.decision_feedback[0]
            self.assertEqual(feedback.runtime_outcome.value, "succeeded")
            self.assertEqual(
                feedback.evaluation_verdict, result.evaluation.outcome.verdict.value
            )
            self.assertEqual(feedback.subject_decision_type, "planning.task_graph.initialize")

    async def test_recall_feedback_binds_committed_bundle(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await self._run(path)
            agent = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    memory_recall=SelectingRecallAgent(),
                    task_planner=RecallAwarePlanner("Acme"),
                ),
            )
            try:
                result = await agent.run("分析 Acme 的投资价值")
            finally:
                agent.close()
            self.assertEqual(len(result.decision_feedback), 2)
            recall = next(
                item
                for item in result.decision_feedback
                if item.subject_decision_type == "memory.recall"
            )
            assert result.memory_recall_bundle is not None
            assert recall.recall_ref is not None
            self.assertEqual(recall.recall_ref.bundle_id, result.memory_recall_bundle.bundle_id)
            self.assertEqual(
                recall.recall_ref.bundle_fingerprint,
                decision_fingerprint(result.memory_recall_bundle),
            )

    async def test_recall_feedback_never_claims_causality(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            await self._run(path)
            agent = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    memory_recall=SelectingRecallAgent(),
                    task_planner=RecallAwarePlanner("Acme"),
                ),
            )
            try:
                result = await agent.run("分析 Acme 的投资价值")
            finally:
                agent.close()
            recall = next(
                item for item in result.decision_feedback
                if item.subject_decision_type == "memory.recall"
            )
            self.assertIn(
                recall.attribution_type,
                {
                    DecisionFeedbackAttributionType.ASSOCIATED,
                    DecisionFeedbackAttributionType.INSUFFICIENT_EVIDENCE,
                },
            )
            serialized = recall.model_dump_json().lower()
            self.assertNotIn("caused_success", serialized)
            self.assertNotIn("caused_failure", serialized)
            self.assertNotIn("reward", serialized)

    async def test_insufficient_evidence_is_recorded_without_guessing(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                feedback = result.decision_feedback[0]
                checkpoint_store = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        DecisionFeedbackRequest,
                        DecisionFeedbackDraft,
                        DecisionFeedbackEffect,
                    ]
                )
                from adaptive_agent_runtime.decision_feedback import stable_feedback_request_id

                checkpoint = await checkpoint_store.load(
                    stable_feedback_request_id(
                        feedback.source_run_id, feedback.subject_decision_id
                    )
                )
                assert checkpoint is not None
                payload = checkpoint.request.payload.model_copy(
                    update={
                        "evaluation": checkpoint.request.payload.evaluation.model_copy(
                            update={"verdict": "inconclusive", "score": None}
                        )
                    }
                )
                request = checkpoint.request.model_copy(update={"payload": payload})
                sources = ProjectionSources(
                    items=(
                        ProjectionSource(
                            source_id="decision-feedback-input",
                            source_type=FEEDBACK_INPUT_SOURCE_TYPE,
                            agent_scope="deterministic_feedback_builder",
                            content=payload.model_dump(mode="json"),
                            sensitivity=ContextSensitivity.INTERNAL,
                            estimated_tokens=256,
                        ),
                    )
                )
                policy = ContextProjectionPolicy(
                    policy_id="test.feedback",
                    version="1",
                    agent_scope="deterministic_feedback_builder",
                    allowed_decision_types=frozenset({DECISION_FEEDBACK_DECISION_TYPE}),
                    allowed_source_types=frozenset({FEEDBACK_INPUT_SOURCE_TYPE}),
                    allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
                    max_items=1,
                    max_context_tokens=512,
                )
                context = PolicyAgentContextBuilder().build(request, sources, policy)
                proposed = await _DeterministicFeedbackProducer().propose(context)
                self.assertEqual(
                    proposed.proposal.payload.attribution_type,
                    DecisionFeedbackAttributionType.INSUFFICIENT_EVIDENCE,
                )
                self.assertNotIn("caused", proposed.proposal.payload.summary.lower())
            finally:
                persistence.close()

    async def test_feedback_is_append_only(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                before = await persistence.decision_feedback_store.list_for_run(
                    result.runtime_result.final_state.run_id
                )
                resumed = ResearchAgent(persistence_path=path)
                try:
                    after_result = await resumed.resume(
                        result.runtime_result.final_state.run_id
                    )
                finally:
                    resumed.close()
                after = await persistence.decision_feedback_store.list_for_run(
                    result.runtime_result.final_state.run_id
                )
                self.assertEqual(before, after)
                self.assertEqual(after_result.decision_feedback, before)
            finally:
                persistence.close()

    async def test_duplicate_feedback_effect_returns_same_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                feedback = result.decision_feedback[0]
                before = await persistence.decision_feedback_store.load_receipt(
                    feedback.effect_fingerprint
                )
                resumed = await _handler(persistence).record_for_run(
                    final_state=result.runtime_result.final_state,
                    evaluation=result.evaluation,
                    artifact_receipt=result.report_commit_receipt,
                    experience=result.experience_metadata,  # type: ignore[arg-type]
                    recall_bundle=None,
                )
                after = await persistence.decision_feedback_store.load_receipt(
                    feedback.effect_fingerprint
                )
                self.assertEqual(before, after)
                self.assertEqual(resumed.records, result.decision_feedback)
            finally:
                persistence.close()

    async def test_same_fingerprint_different_payload_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                feedback = result.decision_feedback[0]
                from adaptive_agent_runtime.decision_feedback import stable_feedback_request_id

                checkpoints = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        DecisionFeedbackRequest,
                        DecisionFeedbackDraft,
                        DecisionFeedbackEffect,
                    ]
                )
                checkpoint = await checkpoints.load(
                    stable_feedback_request_id(
                        feedback.source_run_id, feedback.subject_decision_id
                    )
                )
                assert checkpoint is not None
                assert checkpoint.validated_decision is not None
                normalized = checkpoint.validated_decision.normalized_effect
                changed = normalized.payload.model_copy(
                    update={"evaluation_verdict": "different"}
                )
                target = GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                )
                subject = {"tampered_feedback": changed.model_dump(mode="json")}
                subject_fingerprint = governance_fingerprint(subject)
                request = GovernanceRequest(
                    scope=GovernanceScope.STATE,
                    operation="decision.feedback.commit",
                    target=target,
                    risk=RiskLevel.LOW,
                    signals=ConfidenceSignals(
                        stated_confidence=1.0,
                        impact=ImpactAssessment(
                            score=0.1,
                            reversible=False,
                            description="conflict test",
                        ),
                    ),
                    attributes={SUBJECT_FINGERPRINT_ATTRIBUTE: subject_fingerprint},
                )
                evaluator = RuntimeGovernanceEvaluator(
                    policy=default_governance_policy(),
                    rule_evaluator=DeterministicRuleEvaluator(),
                    confidence_evaluator=DeterministicConfidenceEvaluator(),
                    review_service=persistence.human_review_service,
                )
                decision = evaluator.evaluate(request)
                self.assertEqual(decision.outcome, DecisionOutcome.ALLOW)
                authorization = persistence.authorization_issuer.issue(request, decision)

                async def rejected_raw():  # type: ignore[no-untyped-def]
                    raise AssertionError("permit callback required")

                async def apply_with_permit(permit):  # type: ignore[no-untyped-def]
                    return await persistence.decision_feedback_store.commit(
                        changed,
                        effect_fingerprint=feedback.effect_fingerprint,
                        permit=permit,
                        target=target,
                        subject_fingerprint=subject_fingerprint,
                    )

                with self.assertRaisesRegex(GovernedOperationError, "fingerprint conflicts"):
                    await persistence.operation_executor.execute(
                        request=request,
                        decision=decision,
                        authorization=authorization,
                        target=BoundGovernedOperation(
                            module_id="test.feedback.conflict",
                            operation="decision.feedback.commit",
                            target=target,
                            subject=subject,
                            apply=rejected_raw,
                            apply_with_permit=apply_with_permit,
                        ),
                    )
            finally:
                persistence.close()

    async def test_feedback_resume_after_commit_does_not_duplicate(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            captured_run = None

            def fault(point, checkpoint):  # type: ignore[no-untyped-def]
                nonlocal captured_run
                if (
                    point is DecisionFaultPoint.EFFECT_COMMITTED
                    and checkpoint.request.decision_type
                    == DECISION_FEEDBACK_DECISION_TYPE
                ):
                    captured_run = checkpoint.run_id
                    raise FeedbackCrash("after feedback commit")

            agent = ResearchAgent(
                persistence_path=path, decision_fault_injector=fault
            )
            try:
                with self.assertRaises(FeedbackCrash):
                    await agent.run("分析 Acme 的投资价值")
            finally:
                agent.close()
            assert captured_run is not None
            resumed = ResearchAgent(persistence_path=path)
            try:
                result = await resumed.resume(captured_run)
            finally:
                resumed.close()
            persistence = SQLitePersistence(path)
            try:
                records = await persistence.decision_feedback_store.list_for_run(
                    captured_run
                )
                self.assertEqual(len(records), 1)
                self.assertEqual(result.decision_feedback, records)
            finally:
                persistence.close()

    async def test_feedback_does_not_mutate_decision_memory_or_experience(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path)
            persistence = SQLitePersistence(path)
            try:
                memories_before = await persistence.memory_store.list_all()
                experience_before = await persistence.experience_metadata_store.list_for_run(
                    result.runtime_result.final_state.run_id
                )
                bundle_before = await persistence.memory_recall_bundle_store.load_for_run(
                    result.runtime_result.final_state.run_id
                )
                await _handler(persistence).record_for_run(
                    final_state=result.runtime_result.final_state,
                    evaluation=result.evaluation,
                    artifact_receipt=result.report_commit_receipt,
                    experience=result.experience_metadata,  # type: ignore[arg-type]
                    recall_bundle=None,
                )
                self.assertEqual(memories_before, await persistence.memory_store.list_all())
                self.assertEqual(
                    experience_before,
                    await persistence.experience_metadata_store.list_for_run(
                        result.runtime_result.final_state.run_id
                    ),
                )
                self.assertEqual(
                    bundle_before,
                    await persistence.memory_recall_bundle_store.load_for_run(
                        result.runtime_result.final_state.run_id
                    ),
                )
            finally:
                persistence.close()

    def test_feedback_is_not_consumed_by_planning_or_recall(self) -> None:
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
            self.assertNotIn("adaptive_agent_runtime.decision_feedback", imported)


if __name__ == "__main__":
    unittest.main()
