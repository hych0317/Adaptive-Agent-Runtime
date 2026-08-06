from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_agent_runtime import InMemoryTraceSink, RunStatus
from adaptive_agent_runtime.context_memory import (
    ExperienceAssessmentDraft,
    ExperienceAssessmentRequest,
    ExperienceExecutionOutcome,
    ExperienceMetadataEffect,
    MemoryRecallEligibilityPolicy,
    MemoryScope,
    RuntimeMemoryCandidateResolver,
    RuntimeExecutionObservation,
    stable_experience_request_id,
)
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionFaultPoint,
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
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    TaskGraphDraft,
    TaskPlanningRequest,
)
from adaptive_agent_runtime.persistence import (
    SQLitePersistence,
)
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.cognition import ResearchCognitiveCapabilities
from applications.research_agent.experience_assessment import (
    ExperienceAssessmentAgentRequest,
    ResearchExperienceAssessmentHandler,
)
from applications.research_agent.tasks import build_research_task_draft


class RecordingExperienceAssessor:
    module_id = "test.experience.assessor"
    capability_id = "experience.assessment.test"

    def __init__(self, *, pattern: str = "The strategy succeeded.") -> None:
        self.calls = 0
        self.requests: list[ExperienceAssessmentAgentRequest] = []
        self.pattern = pattern

    async def assess_experience(
        self,
        request: ExperienceAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ExperienceAssessmentDraft]:
        del invocation
        self.calls += 1
        self.requests.append(request)
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=ExperienceAssessmentDraft(
                observed_pattern=self.pattern,
                possible_relevance="Potentially relevant to a similar research task.",
                explanation="Advisory semantic assessment only.",
            ),
        )


class CapturingPlanner:
    module_id = "test.experience.planner"
    capability_id = "experience.planner.test"

    def __init__(self) -> None:
        self.requests: list[TaskPlanningRequest] = []

    async def propose(
        self,
        request: TaskPlanningRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[TaskGraphDraft]:
        del invocation
        self.requests.append(request)
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=build_research_task_draft("Acme"),
        )


def _handler(
    persistence: SQLitePersistence,
    capability: RecordingExperienceAssessor,
    *,
    fault_injector=None,  # type: ignore[no-untyped-def]
) -> ResearchExperienceAssessmentHandler:
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )
    return ResearchExperienceAssessmentHandler(
        state_store=persistence.state_store,
        artifact_readback=persistence.workspace_artifact_committer,
        metadata_store=persistence.experience_metadata_store,
        capability=capability,
        governance=governance,
        reviews=persistence.human_review_service,
        issuer=persistence.authorization_issuer,
        operation_executor=persistence.operation_executor,
        trace_sink=InMemoryTraceSink(),
        checkpoint_store=persistence.create_decision_checkpoint_store(
            DecisionCheckpoint[
                ExperienceAssessmentRequest,
                ExperienceAssessmentDraft,
                ExperienceMetadataEffect,
            ]
        ),
        fault_injector=fault_injector,
    )


class ExperienceMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, path: Path, assessor: RecordingExperienceAssessor):  # type: ignore[no-untyped-def]
        agent = ResearchAgent(
            persistence_path=path,
            cognitive_capabilities=ResearchCognitiveCapabilities(
                experience_assessor=assessor
            ),
        )
        try:
            return await agent.run("分析 Acme 的投资价值")
        finally:
            agent.close()

    async def test_experience_assessment_requires_completed_execution(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            initial = RecordingExperienceAssessor()
            result = await self._run(path, initial)
            persistence = SQLitePersistence(path)
            try:
                assessor = RecordingExperienceAssessor()
                running = result.runtime_result.final_state.model_copy(
                    update={"status": RunStatus.RUNNING, "output": None}
                )
                with self.assertRaisesRegex(ValueError, "terminal"):
                    await _handler(persistence, assessor).assess(
                        final_state=running,
                        evaluation=result.evaluation,
                        artifact_receipt=result.report_commit_receipt,
                        assessment_revision=2,
                    )
                self.assertEqual(assessor.calls, 0)
            finally:
                persistence.close()

    async def test_running_execution_cannot_create_experience(self) -> None:
        await self.test_experience_assessment_requires_completed_execution()

    async def test_uncommitted_artifact_cannot_create_experience(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            try:
                assessor = RecordingExperienceAssessor()
                forged = result.report_commit_receipt.model_copy(
                    update={"effect_fingerprint": decision_fingerprint("forged")}
                )
                with self.assertRaisesRegex(ValueError, "not committed"):
                    await _handler(persistence, assessor).assess(
                        final_state=result.runtime_result.final_state,
                        evaluation=result.evaluation,
                        artifact_receipt=forged,
                        assessment_revision=2,
                    )
                self.assertEqual(assessor.calls, 0)
            finally:
                persistence.close()

    async def test_failed_execution_without_evaluation_creates_no_positive_experience(self) -> None:
        with self.assertRaises(ValidationError):
            ExperienceAssessmentRequest.model_validate(
                {
                    "execution_id": str(uuid4()),
                    "run_id": str(uuid4()),
                    "task_id": str(uuid4()),
                    "related_decision_ids": [str(uuid4())],
                    "source_artifacts": [],
                    "evaluation_refs": [],
                    "runtime_observation": RuntimeExecutionObservation(
                        state_revision=1,
                        state_fingerprint=decision_fingerprint("failed"),
                        final_status="failed",
                        step_count=1,
                        artifact_verified=False,
                        evaluation_available=False,
                    ).model_dump(mode="json"),
                    "execution_outcome": ExperienceExecutionOutcome.FAILED.value,
                    "success_signal": False,
                    "failure_signal": True,
                }
            )

    async def test_agent_assessment_cannot_override_runtime_outcome(self) -> None:
        with TemporaryDirectory() as directory:
            assessor = RecordingExperienceAssessor(pattern="The strategy failed.")
            result = await self._run(Path(directory) / "runtime.sqlite3", assessor)
            metadata = result.experience_metadata
            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertTrue(metadata.success_signal)
            self.assertFalse(metadata.failure_signal)
            self.assertEqual(
                metadata.execution_outcome,
                ExperienceExecutionOutcome.SUCCEEDED,
            )
            self.assertEqual(
                metadata.agent_assessment_summary["observed_pattern"],
                "The strategy failed.",
            )

    def test_agent_cannot_select_memory_or_experience_identity(self) -> None:
        for forbidden in ("memory_id", "experience_id", "success_signal"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValidationError):
                ExperienceAssessmentDraft.model_validate(
                    {
                        "observed_pattern": "pattern",
                        "possible_relevance": "relevance",
                        "explanation": "explanation",
                        forbidden: str(uuid4()),
                    }
                )

    async def test_experience_context_excludes_global_memory_and_policy(self) -> None:
        with TemporaryDirectory() as directory:
            assessor = RecordingExperienceAssessor()
            await self._run(Path(directory) / "runtime.sqlite3", assessor)
            self.assertEqual(assessor.calls, 1)
            projected = json.dumps(
                assessor.requests[0].model_dump(mode="json"),
                sort_keys=True,
            ).lower()
            for forbidden in (
                "memory_id",
                "global_memory",
                "governance",
                "authorization",
                "policy",
            ):
                self.assertNotIn(forbidden, projected)

    async def test_experience_metadata_is_append_only(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            try:
                second = await _handler(
                    persistence,
                    RecordingExperienceAssessor(pattern="A later assessment."),
                ).assess(
                    final_state=result.runtime_result.final_state,
                    evaluation=result.evaluation,
                    artifact_receipt=result.report_commit_receipt,
                    source_memories=result.memories,
                    assessment_revision=2,
                )
                records = await persistence.experience_metadata_store.list_for_run(
                    result.runtime_result.final_state.run_id
                )
                self.assertEqual(tuple(item.version for item in records), (1, 2))
                self.assertNotEqual(records[0].experience_id, second.metadata.experience_id)
                self.assertNotEqual(
                    records[0].agent_assessment_summary,
                    records[1].agent_assessment_summary,
                )
            finally:
                persistence.close()

    async def test_duplicate_effect_returns_same_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            try:
                metadata = result.experience_metadata
                assert metadata is not None
                before = await persistence.experience_metadata_store.load_receipt(
                    metadata.effect_fingerprint
                )
                resumed = await _handler(
                    persistence,
                    RecordingExperienceAssessor(),
                ).resume(result.runtime_result.final_state.run_id)
                after = await persistence.experience_metadata_store.load_receipt(
                    metadata.effect_fingerprint
                )
                assert resumed is not None
                self.assertEqual(resumed.metadata, metadata)
                self.assertEqual(before, after)
                self.assertEqual(
                    len(
                        await persistence.experience_metadata_store.list_for_run(
                            result.runtime_result.final_state.run_id
                        )
                    ),
                    1,
                )
            finally:
                persistence.close()

    async def test_same_fingerprint_different_payload_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            try:
                metadata = result.experience_metadata
                assert metadata is not None
                checkpoints = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        ExperienceAssessmentRequest,
                        ExperienceAssessmentDraft,
                        ExperienceMetadataEffect,
                    ]
                )
                checkpoint = await checkpoints.load(
                    stable_experience_request_id(result.runtime_result.final_state.run_id)
                )
                assert checkpoint is not None
                assert checkpoint.validated_decision is not None
                normalized = checkpoint.validated_decision.normalized_effect
                changed = normalized.payload.model_copy(
                    update={
                        "agent_assessment_summary": {
                            "observed_pattern": "different",
                            "possible_relevance": "different",
                            "explanation": "different",
                        }
                    }
                )
                target = GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                )
                subject = {"tampered_effect": changed.model_dump(mode="json")}
                subject_fingerprint = governance_fingerprint(subject)
                request = GovernanceRequest(
                    scope=GovernanceScope.STATE,
                    operation="experience.metadata.commit",
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
                self.assertEqual(decision.outcome, DecisionOutcome.ALLOW)
                authorization = persistence.authorization_issuer.issue(
                    request,
                    decision,
                )

                async def rejected_raw():  # type: ignore[no-untyped-def]
                    raise AssertionError("permit callback required")

                async def apply_with_permit(permit):  # type: ignore[no-untyped-def]
                    return await persistence.experience_metadata_store.commit(
                        changed,
                        effect_fingerprint=metadata.effect_fingerprint,
                        permit=permit,
                        target=target,
                        subject_fingerprint=subject_fingerprint,
                    )

                with self.assertRaisesRegex(
                    GovernedOperationError,
                    "fingerprint conflicts",
                ):
                    await persistence.operation_executor.execute(
                        request=request,
                        decision=decision,
                        authorization=authorization,
                        target=BoundGovernedOperation(
                            module_id="test.experience.conflict",
                            operation="experience.metadata.commit",
                            target=target,
                            subject=subject,
                            apply=rejected_raw,
                            apply_with_permit=apply_with_permit,
                        ),
                    )
            finally:
                persistence.close()

    async def test_experience_creation_does_not_mutate_memory(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            try:
                before = await persistence.memory_store.list_all()
                await _handler(persistence, RecordingExperienceAssessor()).assess(
                    final_state=result.runtime_result.final_state,
                    evaluation=result.evaluation,
                    artifact_receipt=result.report_commit_receipt,
                    source_memories=before,
                    assessment_revision=2,
                )
                after = await persistence.memory_store.list_all()
                self.assertEqual(before, after)
            finally:
                persistence.close()

    async def test_recall_can_read_experience_metadata_without_memory_write(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            first = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            try:
                before = await persistence.memory_store.list_all()
                candidates = await RuntimeMemoryCandidateResolver(
                    persistence.memory_store,
                    persistence.experience_metadata_store,
                ).resolve(
                    request_id=uuid4(),
                    goal="分析 Acme 的投资价值",
                    facts={"domain": "financial_research"},
                    tags=("research",),
                    policy=MemoryRecallEligibilityPolicy(
                        scope=MemoryScope(
                            project_id="research",
                            agent_scope="planner",
                        )
                    ),
                )
                after = await persistence.memory_store.list_all()
                self.assertEqual(before, after)
                self.assertTrue(
                    any(
                        value.startswith("experience:succeeded:")
                        for candidate in candidates
                        for value in candidate.candidate.provenance_summary
                    )
                )
            finally:
                persistence.close()
            planner = CapturingPlanner()
            agent = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    task_planner=planner,
                    experience_assessor=RecordingExperienceAssessor(),
                ),
            )
            try:
                second = await agent.run("分析 Acme 的投资价值")
            finally:
                agent.close()
            self.assertIsNotNone(second.memory_recall_bundle)
            assert second.memory_recall_bundle is not None
            self.assertTrue(
                any(
                    value.startswith("experience:succeeded:")
                    for value in second.memory_recall_bundle.provenance
                )
            )
            self.assertEqual(len(planner.requests), 1)
            projected = planner.requests[0].committed_memory_recall_bundle
            self.assertIsNotNone(projected)
            assert projected is not None
            self.assertEqual(
                projected["bundle_fingerprint"],
                decision_fingerprint(second.memory_recall_bundle),
            )
            self.assertIn("historical_experience", projected)

    async def test_unknown_experience_apply_is_durable_and_not_retried(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            result = await self._run(path, RecordingExperienceAssessor())
            persistence = SQLitePersistence(path)
            assessor = RecordingExperienceAssessor()

            def fault(point, checkpoint):  # type: ignore[no-untyped-def]
                if point is DecisionFaultPoint.EFFECT_COMMITTED:
                    raise RuntimeError("crash after Experience commit")

            try:
                handler = _handler(
                    persistence,
                    assessor,
                    fault_injector=fault,
                )
                with self.assertRaisesRegex(RuntimeError, "crash"):
                    await handler.assess(
                        final_state=result.runtime_result.final_state,
                        evaluation=result.evaluation,
                        artifact_receipt=result.report_commit_receipt,
                        source_memories=result.memories,
                        assessment_revision=2,
                    )
                request_id = stable_experience_request_id(
                    result.runtime_result.final_state.run_id,
                    2,
                )
                checkpoint_store = persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        ExperienceAssessmentRequest,
                        ExperienceAssessmentDraft,
                        ExperienceMetadataEffect,
                    ]
                )
                checkpoint = await checkpoint_store.load(request_id)
                assert checkpoint is not None
                assert checkpoint.validated_decision is not None
                effect_fingerprint = (
                    checkpoint.validated_decision.normalized_effect.effect_fingerprint
                )
                with persistence.database.transaction() as cursor:
                    cursor.execute(
                        "UPDATE experience_metadata SET metadata_json = ? "
                        "WHERE effect_fingerprint = ?",
                        ("{corrupt", effect_fingerprint),
                    )
                resumed = await _handler(persistence, assessor).resume(
                    result.runtime_result.final_state.run_id,
                    assessment_revision=2,
                )
                self.assertIsNone(resumed)
                calls = assessor.calls
                resumed_again = await _handler(persistence, assessor).resume(
                    result.runtime_result.final_state.run_id,
                    assessment_revision=2,
                )
                self.assertIsNone(resumed_again)
                self.assertEqual(assessor.calls, calls)
                completed = await checkpoint_store.load(request_id)
                assert completed is not None and completed.result is not None
                self.assertEqual(completed.result.reconciliation_status.value, "unknown")
            finally:
                persistence.close()


if __name__ == "__main__":
    unittest.main()
