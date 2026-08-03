from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    ContextCompressionResult,
    ContextLayer,
    ContextLifecycleManager,
    ContextLifecycleState,
    ContextMetadata,
    ContextSource,
    ContextUnit,
    EvidenceDrivenMemoryConsolidator,
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryRecallQuery,
)
from adaptive_agent_runtime.persistence import SQLitePersistence


class DeterministicCompressor:
    module_id = "test.persistence_compressor"

    async def compress(self, unit: ContextUnit) -> ContextCompressionResult:
        return ContextCompressionResult(
            content={"summary": f"compressed:{unit.context_id}"},
            core_conclusions=("durable conclusion",),
            estimated_tokens=4,
        )


class SQLiteContextMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_archive_restores_after_database_reopen(self) -> None:
        run_id = uuid4()
        unit = ContextUnit(
            content={"large": ["fact"] * 20},
            metadata=ContextMetadata(
                source=ContextSource.OBSERVATION,
                layer=ContextLayer.TASK,
                run_id=run_id,
                estimated_tokens=40,
            ),
        )
        with TemporaryDirectory() as directory:
            path = f"{directory}/context.sqlite3"
            first = SQLitePersistence(path)
            lifecycle = ContextLifecycleManager(
                store=first.context_store,
                archive=first.context_archive,
                compressor=DeterministicCompressor(),
            )
            await lifecycle.add(unit)
            compressed = await lifecycle.compress(unit.context_id)
            self.assertEqual(
                compressed.lifecycle_state,
                ContextLifecycleState.COMPRESSED,
            )
            archive_reference = await lifecycle.archive(unit.context_id)
            self.assertEqual(await first.context_store.list_for_run(run_id), ())
            first.close()

            reopened = SQLitePersistence(path)
            restored_lifecycle = ContextLifecycleManager(
                store=reopened.context_store,
                archive=reopened.context_archive,
                compressor=DeterministicCompressor(),
            )
            latest_reference = await reopened.context_archive.find_latest(
                unit.context_id
            )
            self.assertIsNotNone(latest_reference)
            assert latest_reference is not None
            self.assertEqual(
                latest_reference.archive_id,
                archive_reference.archive_id,
            )
            restored = await restored_lifecycle.restore_context(unit.context_id)

            self.assertEqual(restored.lifecycle_state, ContextLifecycleState.ACTIVE)
            self.assertEqual(restored.context_id, unit.context_id)
            self.assertEqual(restored.revision, compressed.revision + 1)
            self.assertEqual(
                await reopened.context_store.list_for_run(run_id),
                (restored,),
            )
            history = await reopened.context_store.history_for(unit.context_id)
            self.assertEqual(
                [snapshot.revision for snapshot in history],
                [0, 1, 2],
            )
            reopened.close()

    async def test_memory_candidate_is_idempotent_after_database_reopen(self) -> None:
        candidate = MemoryCandidate(
            memory_key="research.preference",
            content={"preference": "cite primary sources"},
            condition=MemoryCondition(
                facts={"domain": "finance"},
                required_tags=("research",),
            ),
            evidence=(
                MemoryEvidence(
                    source_reference="run:first",
                    note="user requested primary sources",
                ),
            ),
            confidence=0.9,
            evolution=MemoryEvolutionType.EXTEND,
        )
        with TemporaryDirectory() as directory:
            path = f"{directory}/memory.sqlite3"
            first = SQLitePersistence(path)
            initial = await EvidenceDrivenMemoryConsolidator(
                first.memory_store
            ).consolidate(candidate)
            first.close()

            reopened = SQLitePersistence(path)
            repeated = await EvidenceDrivenMemoryConsolidator(
                reopened.memory_store
            ).consolidate(candidate)
            recalled = await ConditionalMemoryRecall(
                reopened.memory_store
            ).recall(
                MemoryRecallQuery(
                    facts={"domain": "finance"},
                    tags=("research",),
                )
            )

            self.assertEqual(repeated.memory, initial.memory)
            self.assertEqual(recalled, (initial.memory,))
            history = await reopened.memory_store.history_for(
                initial.memory.memory_id
            )
            self.assertEqual(history, (initial.memory,))
            reopened.close()


if __name__ == "__main__":
    unittest.main()
