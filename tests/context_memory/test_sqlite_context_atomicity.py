from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime, timezone
from uuid import UUID, uuid4, uuid5

from adaptive_agent_runtime.context_memory import (
    ContextArchiveReference,
    ContextCompressionExecutionPolicy,
    ContextCompressionTransaction,
    ContextLifecycleManager,
    ContextLifecycleState,
    ContextUnit,
)
from adaptive_agent_runtime.decisioning import DecisionCheckpoint, decision_fingerprint
from adaptive_agent_runtime.governance import (
    AuthorizationUseStatus,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import CompressedContextDraft
from adaptive_agent_runtime.persistence import SQLitePersistence
from applications.research_agent.context_compression import (
    ContextCompressionDecisionPayload,
    ContextCompressionEffect,
    ResearchContextCompressionDecisionHandler,
)
from applications.research_agent.strategies import ResearchWorkspace
from applications.research_agent.tasks import build_research_task
from tests.context_memory.test_context_compression_decision import (
    FakeCompressionAgent,
    context_unit,
)


CHECKPOINT_TYPE = DecisionCheckpoint[
    ContextCompressionDecisionPayload,
    CompressedContextDraft,
    ContextCompressionEffect,
]


class CrashBeforeContextFinalize(ContextCompressionTransaction):
    module_id = "test.context_archive.crash_before_finalize"

    def __init__(self, delegate) -> None:  # type: ignore[no-untyped-def]
        self._delegate = delegate
        self._crash = True

    async def archive(self, unit, *, reference=None):  # type: ignore[no-untyped-def]
        return await self._delegate.archive(unit, reference=reference)

    async def discard(self, reference):  # type: ignore[no-untyped-def]
        await self._delegate.discard(reference)

    async def restore(self, reference):  # type: ignore[no-untyped-def]
        return await self._delegate.restore(reference)

    async def find_latest(self, context_id):  # type: ignore[no-untyped-def]
        return await self._delegate.find_latest(context_id)

    async def stage_compression(self, unit, **kwargs):  # type: ignore[no-untyped-def]
        return await self._delegate.stage_compression(unit, **kwargs)

    async def commit_compression(self, source, compressed, **kwargs):  # type: ignore[no-untyped-def]
        if self._crash:
            self._crash = False
            raise KeyboardInterrupt("crash after pending Archive stage")
        return await self._delegate.commit_compression(source, compressed, **kwargs)

    async def verify_compression(self, unit, **kwargs):  # type: ignore[no-untyped-def]
        return await self._delegate.verify_compression(unit, **kwargs)


class SQLiteContextAtomicityTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _governance(persistence: SQLitePersistence) -> RuntimeGovernanceEvaluator:
        return RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=persistence.human_review_service,
        )

    @staticmethod
    def _handler(
        persistence: SQLitePersistence,
        *,
        archive,
        capability: FakeCompressionAgent,
        workspace: ResearchWorkspace,
    ) -> ResearchContextCompressionDecisionHandler:
        return ResearchContextCompressionDecisionHandler(
            capability=capability,
            store=persistence.context_store,
            archive=archive,
            execution_policy=ContextCompressionExecutionPolicy(
                timeout_seconds=2.0,
                target_token_ratio=0.5,
            ),
            governance=SQLiteContextAtomicityTests._governance(persistence),
            reviews=persistence.human_review_service,
            issuer=persistence.authorization_issuer,
            operation_executor=persistence.operation_executor,
            trace_sink=persistence.trace_sink,
            workspace=workspace,
            checkpoint_store=persistence.create_decision_checkpoint_store(
                CHECKPOINT_TYPE
            ),
        )

    async def test_pending_archive_is_invisible_and_resume_finishes_same_effect(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            run_id = uuid4()
            unit = context_unit(run_id)
            task = build_research_task("Acme")
            first = SQLitePersistence(path)
            await first.context_store.save(unit, expected_revision=None)
            first_agent = FakeCompressionAgent()
            first_handler = self._handler(
                first,
                archive=CrashBeforeContextFinalize(first.context_archive),
                capability=first_agent,
                workspace=ResearchWorkspace(task),
            )
            with self.assertRaisesRegex(
                KeyboardInterrupt,
                "crash after pending Archive stage",
            ):
                await first_handler.compress(unit)

            self.assertEqual(await first.context_store.load(unit.context_id), unit)
            self.assertIsNone(await first.context_archive.find_latest(unit.context_id))
            self.assertEqual(await first.context_archive.pending_count(), 1)
            with first.database.reader() as cursor:
                published = cursor.execute(
                    "SELECT COUNT(*) AS count FROM context_archives"
                ).fetchone()
            assert published is not None
            self.assertEqual(int(published["count"]), 0)
            first.close()

            reopened = SQLitePersistence(path)
            resumed_agent = FakeCompressionAgent()
            resumed_handler = self._handler(
                reopened,
                archive=reopened.context_archive,
                capability=resumed_agent,
                workspace=ResearchWorkspace(task),
            )
            await resumed_handler.compress(unit)
            committed = await reopened.context_store.load(unit.context_id)
            assert committed is not None
            self.assertEqual(committed.lifecycle_state, ContextLifecycleState.COMPRESSED)
            self.assertEqual(committed.revision, unit.revision + 1)
            self.assertEqual(len(first_agent.requests), 1)
            self.assertEqual(len(resumed_agent.requests), 0)
            self.assertEqual(await reopened.context_archive.pending_count(), 0)
            assert committed.last_effect_fingerprint is not None
            self.assertTrue(
                await reopened.context_archive.verify_compression(
                    committed,
                    effect_fingerprint=committed.last_effect_fingerprint,
                    source_fingerprint=decision_fingerprint(unit),
                )
            )
            reference = await reopened.context_archive.find_latest(unit.context_id)
            assert isinstance(reference, ContextArchiveReference)
            self.assertEqual(
                reference.archive_id,
                uuid5(unit.context_id, committed.last_effect_fingerprint),
            )
            self.assertEqual(
                reference.archive_id,
                committed.recovery_reference.archive_id
                if committed.recovery_reference is not None
                else None,
            )
            with reopened.database.reader() as cursor:
                counts = cursor.execute(
                    "SELECT COUNT(*) AS count FROM context_archives"
                ).fetchone()
            assert counts is not None
            self.assertEqual(int(counts["count"]), 1)

            await resumed_handler.compress(unit)
            with reopened.database.reader() as cursor:
                counts = cursor.execute(
                    "SELECT COUNT(*) AS count FROM context_archives"
                ).fetchone()
            assert counts is not None
            self.assertEqual(int(counts["count"]), 1)
            reopened.close()

    async def test_pending_archive_cannot_publish_after_authorization_expires(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            run_id = uuid4()
            unit = context_unit(run_id)
            task = build_research_task("Acme")
            first = SQLitePersistence(path)
            await first.context_store.save(unit, expected_revision=None)
            handler = self._handler(
                first,
                archive=CrashBeforeContextFinalize(first.context_archive),
                capability=FakeCompressionAgent(),
                workspace=ResearchWorkspace(task),
            )
            with self.assertRaises(KeyboardInterrupt):
                await handler.compress(unit)
            with first.database.reader() as cursor:
                row = cursor.execute(
                    "SELECT authorization_id FROM governance_authorization_uses"
                ).fetchone()
            assert row is not None
            use = await first.authorization_store.load(
                UUID(row["authorization_id"])
            )
            assert use is not None
            expired = use.model_copy(
                update={
                    "status": AuthorizationUseStatus.FAILED,
                    "revision": use.revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                    "error": "authorization expired before Context finalize",
                }
            )
            await first.authorization_store.resolve(
                expired,
                expected_revision=use.revision,
            )
            first.close()

            reopened = SQLitePersistence(path)
            resumed = self._handler(
                reopened,
                archive=reopened.context_archive,
                capability=FakeCompressionAgent(),
                workspace=ResearchWorkspace(task),
            )
            with self.assertRaises(Exception):
                await resumed.compress(unit)
            self.assertEqual(await reopened.context_store.load(unit.context_id), unit)
            self.assertIsNone(await reopened.context_archive.find_latest(unit.context_id))
            self.assertEqual(await reopened.context_archive.pending_count(), 1)
            with reopened.database.reader() as cursor:
                published = cursor.execute(
                    "SELECT COUNT(*) AS count FROM context_archives"
                ).fetchone()
            assert published is not None
            self.assertEqual(int(published["count"]), 0)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
