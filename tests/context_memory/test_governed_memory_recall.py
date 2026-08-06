from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime.context_memory import (
    MemoryCondition,
    MemoryEvidence,
    MemoryRecallDraft,
    MemoryScope,
    MemorySensitivity,
    MemoryStatus,
    MemoryUnit,
)
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionFaultPoint,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
)
from adaptive_agent_runtime.persistence import SQLitePersistence

from applications.research_agent.memory_recall import (
    ResearchInitialMemoryRecallHandler,
)


class RecordingRecallAgent:
    module_id = "test.memory_recall.agent"
    capability_id = "test.memory_recall"

    def __init__(self, *, outsider: bool = False) -> None:
        self.requests = []
        self.outsider = outsider

    async def propose_memory_recall(
        self,
        request,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ):
        del invocation
        self.requests.append(request)
        refs = (
            ("candidate:not-runtime-approved",)
            if self.outsider
            else (request.candidates[0].candidate_ref,)
        )
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=MemoryRecallDraft(
                selected_candidate_refs=refs,
                selection_reason="Select relevant historical experience.",
            ),
        )


def memory(
    *,
    key: str,
    content,
    scope: MemoryScope | None = None,
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    confidence: float = 0.9,
    facts=None,
    tags=("research",),
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryUnit:
    candidate_id = uuid4()
    created = created_at or datetime.now(timezone.utc)
    return MemoryUnit(
        memory_key=key,
        content=content,
        condition=MemoryCondition(
            facts=facts or {"domain": "financial_research"},
            required_tags=tags,
        ),
        evidence=(
            MemoryEvidence(
                source_reference=f"test:{key}",
                note="durable test provenance",
            ),
        ),
        confidence=confidence,
        status=status,
        scope=scope or MemoryScope(project_id="research", agent_scope="planner"),
        sensitivity=sensitivity,
        expires_at=expires_at,
        last_candidate_id=candidate_id,
        last_candidate_fingerprint="a" * 64,
        created_at=created,
        updated_at=created,
    )


class GovernedMemoryRecallTests(unittest.IsolatedAsyncioTestCase):
    def _handler(
        self,
        persistence: SQLitePersistence,
        agent: RecordingRecallAgent,
        *,
        fault=None,
    ) -> ResearchInitialMemoryRecallHandler:
        governance = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=persistence.human_review_service,
        )
        from adaptive_agent_runtime.context_memory import (
            MemoryRecallRequest,
            MemoryRecallEffect,
        )
        return ResearchInitialMemoryRecallHandler(
            memory_store=persistence.memory_store,
            bundle_store=persistence.memory_recall_bundle_store,
            capability=agent,
            governance=governance,
            reviews=persistence.human_review_service,
            issuer=persistence.authorization_issuer,
            operation_executor=persistence.operation_executor,
            trace_sink=persistence.trace_sink,
            checkpoint_store=persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    MemoryRecallRequest, MemoryRecallDraft, MemoryRecallEffect
                ]
            ),
            fault_injector=fault,
        )

    async def _recall(self, handler, *, run_id=None):
        return await handler.recall(
            goal="分析 Acme 投资价值",
            run_id=run_id or uuid4(),
            task_id=uuid4(),
            scope=MemoryScope(project_id="research", agent_scope="planner"),
            facts={"domain": "financial_research"},
            tags=("research",),
            max_items=2,
            max_tokens=256,
        )

    async def test_recall_only_exposes_runtime_eligible_candidates(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            now = datetime.now(timezone.utc)
            candidates = (
                memory(
                    key="research.valid",
                    content={"principle": "compare margins", "token": "hidden"},
                ),
                memory(key="research.retired", content="x", status=MemoryStatus.RETIRED),
                memory(
                    key="research.expired",
                    content="x",
                    created_at=now - timedelta(days=2),
                    expires_at=now - timedelta(days=1),
                ),
                memory(key="research.condition_false", content="x", facts={"market": "CN"}),
                memory(
                    key="research.other_agent",
                    content="x",
                    scope=MemoryScope(project_id="research", agent_scope="recovery"),
                ),
                memory(
                    key="research.sensitive",
                    content={"secret": "never expose"},
                    sensitivity=MemorySensitivity.CONFIDENTIAL,
                ),
            )
            for item in candidates:
                await persistence.memory_store.save(item, expected_revision=None)
            agent = RecordingRecallAgent()
            result = await self._recall(self._handler(persistence, agent))
            self.assertIsNotNone(result)
            self.assertEqual(len(agent.requests), 1)
            exposed = agent.requests[0].candidates
            self.assertEqual(len(exposed), 1)
            self.assertNotIn("memory_id", exposed[0].model_dump())
            self.assertEqual(exposed[0].sanitized_content["token"], "[REDACTED]")
            persistence.close()

    async def test_agent_cannot_select_candidate_outside_candidate_set(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            await persistence.memory_store.save(
                memory(key="research.valid", content="useful"),
                expected_revision=None,
            )
            run_id = uuid4()
            result = await self._recall(
                self._handler(persistence, RecordingRecallAgent(outsider=True)),
                run_id=run_id,
            )
            self.assertIsNone(result)
            self.assertEqual(
                await persistence.memory_recall_bundle_store.count_for_run(run_id), 0
            )
            persistence.close()

    async def test_revoked_expired_or_condition_false_memory_is_excluded(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            now = datetime.now(timezone.utc)
            values = (
                memory(key="research.valid", content="valid"),
                memory(key="research.revoked", content="revoked", status=MemoryStatus.RETIRED),
                memory(
                    key="research.expired",
                    content="expired",
                    created_at=now - timedelta(days=2),
                    expires_at=now - timedelta(days=1),
                ),
                memory(
                    key="research.false_condition",
                    content="wrong market",
                    facts={"market": "CN"},
                ),
            )
            for item in values:
                await persistence.memory_store.save(item, expected_revision=None)
            agent = RecordingRecallAgent()
            await self._recall(self._handler(persistence, agent))
            self.assertEqual(
                tuple(item.sanitized_content for item in agent.requests[0].candidates),
                ("valid",),
            )
            persistence.close()

    async def test_cross_agent_memory_is_not_exposed(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            for item in (
                memory(key="research.valid", content="planner"),
                memory(
                    key="research.recovery_only",
                    content="recovery",
                    scope=MemoryScope(project_id="research", agent_scope="recovery"),
                ),
            ):
                await persistence.memory_store.save(item, expected_revision=None)
            agent = RecordingRecallAgent()
            await self._recall(self._handler(persistence, agent))
            self.assertEqual(len(agent.requests[0].candidates), 1)
            self.assertEqual(agent.requests[0].candidates[0].sanitized_content, "planner")
            persistence.close()

    async def test_sensitive_memory_is_redacted_or_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            for item in (
                memory(
                    key="research.internal",
                    content={"principle": "safe", "secret": "redact-me"},
                ),
                memory(
                    key="research.confidential",
                    content={"principle": "never-visible"},
                    sensitivity=MemorySensitivity.CONFIDENTIAL,
                ),
            ):
                await persistence.memory_store.save(item, expected_revision=None)
            agent = RecordingRecallAgent()
            await self._recall(self._handler(persistence, agent))
            exposed = agent.requests[0].candidates
            self.assertEqual(len(exposed), 1)
            self.assertEqual(exposed[0].sanitized_content["secret"], "[REDACTED]")
            self.assertNotIn("never-visible", str(exposed))
            persistence.close()

    async def test_no_candidate_planning_continues_without_recall(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            agent = RecordingRecallAgent()
            result = await self._recall(self._handler(persistence, agent))
            self.assertIsNone(result)
            self.assertEqual(agent.requests, [])
            persistence.close()

    async def test_source_change_before_apply_rejects_stale_effect(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            original = memory(key="research.valid", content="version zero")
            await persistence.memory_store.save(original, expected_revision=None)
            request_id: UUID | None = None

            def crash(point, checkpoint):
                nonlocal request_id
                if point is DecisionFaultPoint.AUTHORIZED:
                    request_id = checkpoint.request_id
                    raise KeyboardInterrupt("after authorization")

            run_id = uuid4()
            handler = self._handler(persistence, RecordingRecallAgent(), fault=crash)
            with self.assertRaises(KeyboardInterrupt):
                await self._recall(handler, run_id=run_id)
            assert request_id is not None
            changed = original.model_copy(
                update={
                    "content": "version one",
                    "revision": 1,
                    "updated_at": datetime.now(timezone.utc),
                    "last_candidate_id": uuid4(),
                    "last_candidate_fingerprint": "b" * 64,
                }
            )
            await persistence.memory_store.save(changed, expected_revision=0)
            resumed = await handler.resume(request_id)
            self.assertIsNone(resumed)
            self.assertEqual(
                await persistence.memory_recall_bundle_store.count_for_run(
                    run_id
                ),
                0,
            )
            persistence.close()

    async def test_recall_does_not_mutate_memory_records(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            original = memory(key="research.valid", content="immutable")
            await persistence.memory_store.save(original, expected_revision=None)
            before = await persistence.memory_store.list_all()
            result = await self._recall(
                self._handler(persistence, RecordingRecallAgent())
            )
            after = await persistence.memory_store.list_all()
            self.assertIsNotNone(result)
            self.assertEqual(before, after)
            persistence.close()


if __name__ == "__main__":
    unittest.main()
