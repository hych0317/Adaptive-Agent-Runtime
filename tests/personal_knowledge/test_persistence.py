from __future__ import annotations

from datetime import timedelta
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID, uuid4

from applications.personal_knowledge import (
    CategoryChangeType,
    CategoryStatus,
    CategoryStore,
    Citation,
    ConfirmedCategoryChange,
    ConfirmedKnowledgeChange,
    EvidenceArchiveStore,
    KnowledgeChangeType,
    KnowledgeConflictError,
    KnowledgeDocument,
    KnowledgeEntry,
    KnowledgeEntryStore,
    KnowledgeInvariantError,
    KnowledgeStatus,
    ReviewArchive,
    ReviewDraft,
    ReviewMessage,
    ReviewRole,
    SQLitePersonalKnowledgeStore,
    SourceKind,
    SourceRecord,
    SourceTextArtifact,
    TextArtifactKind,
    TranscriptSegment,
)
from applications.personal_knowledge.models import utc_now


class SQLitePersonalKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.store = SQLitePersonalKnowledgeStore(
            f"{self.directory.name}/knowledge.sqlite3"
        )

    async def asyncTearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    async def create_category(self, name: str) -> UUID:
        change = ConfirmedCategoryChange(
            change_type=CategoryChangeType.CREATE,
            name=name,
        )
        result = await self.store.apply_category_change(change)
        return result.category.category_id

    async def create_review(
        self,
        *,
        source_ids: tuple[UUID, ...] = (),
        artifact_ids: tuple[UUID, ...] = (),
    ) -> ReviewArchive:
        review = ReviewArchive(
            source_ids=source_ids,
            artifact_ids=artifact_ids,
            conversation=(
                ReviewMessage(role=ReviewRole.USER, content="我的想法"),
                ReviewMessage(role=ReviewRole.AGENT, content="整理后的讨论"),
            ),
            drafts=(ReviewDraft(content="知识草稿"),),
        )
        await self.store.save_review(review, expected_revision=None)
        return review

    async def publish(
        self,
        *,
        category_id: UUID,
        review: ReviewArchive,
        title: str = "Confirmed title",
        body: str = "User-edited final body",
        citations: tuple[Citation, ...] = (),
    ) -> KnowledgeEntry:
        return await self.store.apply_knowledge_change(
            ConfirmedKnowledgeChange(
                change_type=KnowledgeChangeType.PUBLISH,
                review_id=review.review_id,
                document=KnowledgeDocument(
                    title=title,
                    category_id=category_id,
                    body=body,
                    tags=("agent", "knowledge"),
                    citations=citations,
                ),
            )
        )

    async def new_update_review(self) -> ReviewArchive:
        return await self.create_review()

    async def test_store_satisfies_application_contracts(self) -> None:
        self.assertIsInstance(self.store, EvidenceArchiveStore)
        self.assertIsInstance(self.store, CategoryStore)
        self.assertIsInstance(self.store, KnowledgeEntryStore)

    async def test_source_transcript_and_review_are_archived_outside_entries(
        self,
    ) -> None:
        source = SourceRecord(
            kind=SourceKind.VIDEO,
            canonical_url="https://example.test/video/1",
            title="Video",
        )
        await self.store.save_source(source)
        artifact = SourceTextArtifact.from_text(
            source_id=source.source_id,
            kind=TextArtifactKind.TRANSCRIPT,
            text="timestamped transcript",
            segments=(
                TranscriptSegment(
                    start_seconds=0,
                    end_seconds=3,
                    text="timestamped transcript",
                ),
            ),
        )
        await self.store.save_artifact(artifact)
        review = await self.create_review(
            source_ids=(source.source_id,),
            artifact_ids=(artifact.artifact_id,),
        )

        self.assertEqual(await self.store.load_source(source.source_id), source)
        self.assertEqual(
            await self.store.load_artifact(artifact.artifact_id),
            artifact,
        )
        self.assertEqual(await self.store.load_review(review.review_id), review)
        self.assertEqual(await self.store.list_entries(), ())

    async def test_source_identity_is_idempotent_but_cannot_change(self) -> None:
        source = SourceRecord(kind=SourceKind.PLAIN_TEXT, title="Idea")
        await self.store.save_source(source)
        await self.store.save_source(source)

        with self.assertRaises(KnowledgeConflictError):
            await self.store.save_source(
                source.model_copy(update={"title": "Changed identity"})
            )

    async def test_review_updates_require_exact_revision(self) -> None:
        review = await self.create_review()
        updated = review.model_copy(
            update={
                "conversation": (
                    *review.conversation,
                    ReviewMessage(role=ReviewRole.USER, content="继续讨论"),
                ),
                "revision": 1,
                "updated_at": review.updated_at + timedelta(seconds=1),
            }
        )
        await self.store.save_review(updated, expected_revision=0)

        with self.assertRaises(KnowledgeConflictError):
            await self.store.save_review(
                updated.model_copy(
                    update={
                        "revision": 2,
                        "updated_at": updated.updated_at + timedelta(seconds=1),
                    }
                ),
                expected_revision=0,
            )

    async def test_confirmation_publishes_exact_document_and_is_idempotent(
        self,
    ) -> None:
        source = SourceRecord(
            kind=SourceKind.WEB_ARTICLE,
            canonical_url="https://example.test/article",
        )
        await self.store.save_source(source)
        artifact = SourceTextArtifact.from_text(
            source_id=source.source_id,
            kind=TextArtifactKind.ORIGINAL_TEXT,
            text="source text",
        )
        await self.store.save_artifact(artifact)
        review = await self.create_review(
            source_ids=(source.source_id,),
            artifact_ids=(artifact.artifact_id,),
        )
        category_id = await self.create_category("AI")
        confirmation = ConfirmedKnowledgeChange(
            change_type=KnowledgeChangeType.PUBLISH,
            review_id=review.review_id,
            document=KnowledgeDocument(
                title="用户编辑后的标题",
                category_id=category_id,
                body="用户编辑后的最终正文",
                tags=("AI",),
                citations=(
                    Citation(
                        source_id=source.source_id,
                        artifact_id=artifact.artifact_id,
                        source_url=source.canonical_url,
                        quote="source text",
                    ),
                ),
            ),
        )

        created = await self.store.apply_knowledge_change(confirmation)
        replayed = await self.store.apply_knowledge_change(confirmation)
        finalized_review = await self.store.load_review(review.review_id)

        self.assertEqual(created, replayed)
        self.assertEqual(created.document, confirmation.document)
        self.assertIsNotNone(finalized_review)
        assert finalized_review is not None
        self.assertEqual(
            finalized_review.final_confirmation_id,
            confirmation.confirmation_id,
        )

    async def test_keeps_current_plus_two_historical_versions(self) -> None:
        category_id = await self.create_category("Engineering")
        review = await self.create_review()
        publication = ConfirmedKnowledgeChange(
            change_type=KnowledgeChangeType.PUBLISH,
            review_id=review.review_id,
            document=KnowledgeDocument(
                title="Versioned entry",
                category_id=category_id,
                body="version 0",
            ),
        )
        entry = await self.store.apply_knowledge_change(publication)
        base_time = utc_now()

        for number in range(1, 4):
            update_review = await self.new_update_review()
            entry = await self.store.apply_knowledge_change(
                ConfirmedKnowledgeChange(
                    change_type=KnowledgeChangeType.UPDATE,
                    review_id=update_review.review_id,
                    target_entry_id=entry.entry_id,
                    expected_revision=entry.revision,
                    document=entry.document.model_copy(
                        update={"body": f"version {number}"}
                    ),
                    confirmed_at=base_time + timedelta(seconds=number),
                )
            )

        history = await self.store.history_for(entry.entry_id)
        replayed_old_confirmation = await self.store.apply_knowledge_change(
            publication
        )

        self.assertEqual([item.revision for item in history], [1, 2, 3])
        self.assertEqual(history[-1].document.body, "version 3")
        self.assertEqual(replayed_old_confirmation, history[-1])

    async def test_delete_uses_recoverable_trash(self) -> None:
        category_id = await self.create_category("Thinking")
        entry = await self.publish(
            category_id=category_id,
            review=await self.create_review(),
        )
        deleted = await self.store.apply_knowledge_change(
            ConfirmedKnowledgeChange(
                change_type=KnowledgeChangeType.DELETE,
                target_entry_id=entry.entry_id,
                expected_revision=entry.revision,
            )
        )

        self.assertEqual(deleted.status, KnowledgeStatus.TRASHED)
        self.assertEqual(await self.store.list_entries(), ())
        self.assertEqual(
            await self.store.list_entries(include_trashed=True),
            (deleted,),
        )

        restored = await self.store.apply_knowledge_change(
            ConfirmedKnowledgeChange(
                change_type=KnowledgeChangeType.RESTORE,
                target_entry_id=entry.entry_id,
                expected_revision=deleted.revision,
            )
        )
        self.assertEqual(restored.status, KnowledgeStatus.ACTIVE)
        self.assertEqual(await self.store.list_entries(), (restored,))

    async def test_confirmed_category_merge_reclassifies_existing_knowledge(
        self,
    ) -> None:
        source_category = await self.create_category("Old topic")
        target_category = await self.create_category("New topic")
        entry = await self.publish(
            category_id=source_category,
            review=await self.create_review(),
        )
        result = await self.store.apply_category_change(
            ConfirmedCategoryChange(
                change_type=CategoryChangeType.MERGE,
                category_id=source_category,
                target_category_id=target_category,
                expected_revision=0,
            )
        )
        updated = await self.store.load_entry(entry.entry_id)

        self.assertEqual(result.category.status, CategoryStatus.RETIRED)
        self.assertEqual(result.category.merged_into, target_category)
        self.assertEqual(result.affected_entry_ids, (entry.entry_id,))
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.document.category_id, target_category)

    async def test_category_with_entries_cannot_be_silently_retired(self) -> None:
        category_id = await self.create_category("In use")
        await self.publish(
            category_id=category_id,
            review=await self.create_review(),
        )

        with self.assertRaises(KnowledgeInvariantError):
            await self.store.apply_category_change(
                ConfirmedCategoryChange(
                    change_type=CategoryChangeType.RETIRE,
                    category_id=category_id,
                    expected_revision=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
