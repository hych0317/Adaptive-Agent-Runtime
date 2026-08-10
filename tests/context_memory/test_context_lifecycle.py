from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
from typing import Any, cast
from uuid import UUID, uuid4

from adaptive_agent_runtime.context_memory import (
    ContextArchiveReference,
    ContextCompressionResult,
    ContextLayer,
    ContextLifecycleManager,
    ContextLifecycleState,
    ContextMetadata,
    ContextRecoveryError,
    ContextSnapshotConflictError,
    ContextSource,
    ContextTransitionError,
    ContextUnit,
    InMemoryContextArchive,
    InMemoryContextStore,
    ResidencyPolicy,
)


class StubCompressor:
    module_id = "test.context_compressor"

    async def compress(self, unit: ContextUnit) -> ContextCompressionResult:
        return ContextCompressionResult(
            content={"summary": unit.metadata.source_reference},
            core_conclusions=("core conclusion",),
            estimated_tokens=2,
        )


class FailingContextStore(InMemoryContextStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_save = False
        self.fail_delete = False

    async def save(
        self,
        unit: ContextUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        if self.fail_save:
            raise RuntimeError("injected context save failure")
        await super().save(unit, expected_revision=expected_revision)

    async def delete(
        self,
        context_id: UUID,
        *,
        expected_revision: int,
    ) -> None:
        if self.fail_delete:
            raise RuntimeError("injected context delete failure")
        await super().delete(
            context_id,
            expected_revision=expected_revision,
        )


def context_unit(
    *,
    residency: ResidencyPolicy = ResidencyPolicy.SESSION,
) -> ContextUnit:
    return ContextUnit(
        content={"raw": ["alpha", "beta"]},
        metadata=ContextMetadata(
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.TASK,
            run_id=uuid4(),
            source_reference="document:1",
            estimated_tokens=10,
        ),
        residency_policy=residency,
    )


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


class ContextLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryContextStore()
        self.archive = InMemoryContextArchive()
        self.manager = ContextLifecycleManager(
            store=self.store,
            archive=self.archive,
            compressor=StubCompressor(),
        )

    async def test_metadata_defaults_share_one_timestamp(self) -> None:
        metadata = ContextMetadata(
            source=ContextSource.DOCUMENT,
            layer=ContextLayer.TASK,
            run_id=uuid4(),
        )

        self.assertEqual(metadata.updated_at, metadata.created_at)

        from adaptive_agent_runtime.context_memory.context_runtime import (
            _evolve_metadata,
        )
        with patch(
            "adaptive_agent_runtime.context_memory.context_runtime.utc_now",
            return_value=NOW - timedelta(microseconds=1),
        ):
            evolved = _evolve_metadata(
                metadata.model_copy(
                    update={"created_at": NOW, "updated_at": NOW}
                )
            )
        self.assertEqual(evolved.updated_at, NOW)

    async def test_context_content_is_deeply_immutable(self) -> None:
        source: dict[str, Any] = {"nested": [1, 2]}
        unit = ContextUnit(
            content=source,
            metadata=context_unit().metadata,
        )
        source["nested"][0] = 9

        immutable_content = cast(dict[str, Any], unit.content)
        self.assertEqual(immutable_content["nested"][0], 1)
        with self.assertRaises(TypeError):
            immutable_content["nested"][0] = 9

        copied = unit.model_copy(update={"content": {"mutable": [1]}})
        copied_content = cast(dict[str, Any], copied.content)
        with self.assertRaises(TypeError):
            copied_content["mutable"][0] = 2

    async def test_active_context_compresses_with_recovery_entry(self) -> None:
        original = context_unit()
        await self.manager.add(original)

        compressed = await self.manager.compress(original.context_id)

        self.assertEqual(original.lifecycle_state, ContextLifecycleState.ACTIVE)
        self.assertEqual(
            compressed.lifecycle_state,
            ContextLifecycleState.COMPRESSED,
        )
        self.assertEqual(compressed.revision, original.revision + 1)
        self.assertEqual(compressed.core_conclusions, ("core conclusion",))
        self.assertIsNotNone(compressed.recovery_reference)
        assert compressed.recovery_reference is not None
        archived_original = await self.archive.restore(
            compressed.recovery_reference
        )
        self.assertEqual(archived_original.content, original.content)
        self.assertEqual(
            archived_original.lifecycle_state,
            ContextLifecycleState.ARCHIVED,
        )
        self.assertEqual(archived_original.revision, original.revision)
        self.assertEqual(await self.manager.active_for_run(original.metadata.run_id), ())
        self.assertEqual(
            await self.manager.resident_for_run(original.metadata.run_id),
            (compressed,),
        )

        with self.assertRaises(ContextSnapshotConflictError):
            await self.manager.add(original)
        self.assertEqual(await self.store.load(original.context_id), compressed)

    async def test_compressed_context_archives_and_restores(self) -> None:
        original = context_unit()
        await self.manager.add(original)
        compressed = await self.manager.compress(original.context_id)

        reference = await self.manager.archive(compressed.context_id)
        self.assertIsNone(await self.store.load(compressed.context_id))

        restored = await self.manager.restore(reference)
        self.assertEqual(restored.lifecycle_state, ContextLifecycleState.ACTIVE)
        self.assertEqual(restored.context_id, original.context_id)
        self.assertEqual(restored.content, compressed.content)

        with self.assertRaises(ContextRecoveryError):
            await self.manager.restore(reference)

    async def test_compression_recovery_can_replace_its_compressed_snapshot(self) -> None:
        original = context_unit()
        await self.manager.add(original)
        compressed = await self.manager.compress(original.context_id)
        assert compressed.recovery_reference is not None

        restored = await self.manager.restore(compressed.recovery_reference)

        self.assertEqual(restored.lifecycle_state, ContextLifecycleState.ACTIVE)
        self.assertEqual(restored.content, original.content)
        self.assertGreater(restored.revision, compressed.revision)

    async def test_active_context_cannot_skip_compression(self) -> None:
        unit = context_unit()
        await self.manager.add(unit)

        with self.assertRaises(ContextTransitionError):
            await self.manager.archive(unit.context_id)

        self.assertEqual(await self.store.load(unit.context_id), unit)

    async def test_pinned_context_cannot_be_compressed(self) -> None:
        unit = context_unit(residency=ResidencyPolicy.PINNED)
        await self.manager.add(unit)

        with self.assertRaises(ContextTransitionError):
            await self.manager.compress(unit.context_id)

    async def test_unknown_archive_reference_cannot_restore(self) -> None:
        with self.assertRaises(ContextRecoveryError):
            await self.manager.restore(
                ContextArchiveReference(context_id=uuid4())
            )

    async def test_recovery_reference_must_belong_to_the_context(self) -> None:
        with self.assertRaises(ValueError):
            ContextUnit(
                content="summary",
                metadata=context_unit().metadata,
                lifecycle_state=ContextLifecycleState.COMPRESSED,
                core_conclusions=("conclusion",),
                recovery_reference=ContextArchiveReference(context_id=uuid4()),
            )

    async def test_compression_compensates_archive_when_store_save_fails(self) -> None:
        store = FailingContextStore()
        archive = InMemoryContextArchive()
        manager = ContextLifecycleManager(
            store=store,
            archive=archive,
            compressor=StubCompressor(),
        )
        original = context_unit()
        await manager.add(original)
        store.fail_save = True

        with self.assertRaises(RuntimeError):
            await manager.compress(original.context_id)

        self.assertEqual(await store.load(original.context_id), original)
        self.assertEqual(archive.record_count(), 0)

    async def test_archival_compensates_when_resident_delete_fails(self) -> None:
        store = FailingContextStore()
        archive = InMemoryContextArchive()
        manager = ContextLifecycleManager(
            store=store,
            archive=archive,
            compressor=StubCompressor(),
        )
        original = context_unit()
        await manager.add(original)
        compressed = await manager.compress(original.context_id)
        archive_count = archive.record_count()
        store.fail_delete = True

        with self.assertRaises(RuntimeError):
            await manager.archive(compressed.context_id)

        self.assertEqual(await store.load(compressed.context_id), compressed)
        self.assertEqual(archive.record_count(), archive_count)

    async def test_old_archive_reference_cannot_regress_context_history(self) -> None:
        original = context_unit()
        await self.manager.add(original)
        first_compressed = await self.manager.compress(original.context_id)
        first_reference = await self.manager.archive(first_compressed.context_id)
        first_restored = await self.manager.restore(first_reference)
        second_compressed = await self.manager.compress(first_restored.context_id)
        second_reference = await self.manager.archive(second_compressed.context_id)

        with self.assertRaises(ContextSnapshotConflictError):
            await self.manager.add(original)
        with self.assertRaises(ContextSnapshotConflictError):
            await self.manager.restore(first_reference)

        latest = await self.manager.restore(second_reference)
        self.assertEqual(latest.revision, second_compressed.revision + 1)
        self.assertEqual(
            tuple(unit.revision for unit in self.store.history_for(original.context_id)),
            (0, 1, 2, 3, 4),
        )


if __name__ == "__main__":
    unittest.main()
