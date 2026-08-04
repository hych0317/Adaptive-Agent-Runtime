"""Narrow storage contracts owned by the personal knowledge application."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from applications.personal_knowledge.models import (
    Category,
    CategoryChangeResult,
    ConfirmedCategoryChange,
    ConfirmedKnowledgeChange,
    KnowledgeEntry,
    KnowledgeSearchHit,
    ReviewArchive,
    SourceRecord,
    SourceTextArtifact,
    SourceSubscription,
)


@runtime_checkable
class EvidenceArchiveStore(Protocol):
    async def save_source(self, source: SourceRecord) -> None: ...

    async def load_source(self, source_id: UUID) -> SourceRecord | None: ...

    async def save_artifact(self, artifact: SourceTextArtifact) -> None: ...

    async def load_artifact(
        self,
        artifact_id: UUID,
    ) -> SourceTextArtifact | None: ...

    async def save_review(
        self,
        review: ReviewArchive,
        *,
        expected_revision: int | None,
    ) -> None: ...

    async def load_review(self, review_id: UUID) -> ReviewArchive | None: ...

    async def list_reviews(
        self,
        *,
        include_finalized: bool = False,
    ) -> tuple[ReviewArchive, ...]: ...


@runtime_checkable
class CategoryStore(Protocol):
    async def apply_category_change(
        self,
        change: ConfirmedCategoryChange,
    ) -> CategoryChangeResult: ...

    async def load_category(self, category_id: UUID) -> Category | None: ...

    async def list_categories(
        self,
        *,
        include_retired: bool = False,
    ) -> tuple[Category, ...]: ...


@runtime_checkable
class KnowledgeEntryStore(Protocol):
    async def apply_knowledge_change(
        self,
        change: ConfirmedKnowledgeChange,
    ) -> KnowledgeEntry: ...

    async def load_entry(self, entry_id: UUID) -> KnowledgeEntry | None: ...

    async def list_entries(
        self,
        *,
        include_trashed: bool = False,
    ) -> tuple[KnowledgeEntry, ...]: ...

    async def history_for(self, entry_id: UUID) -> tuple[KnowledgeEntry, ...]: ...


@runtime_checkable
class PersonalKnowledgeStore(
    EvidenceArchiveStore,
    CategoryStore,
    KnowledgeEntryStore,
    Protocol,
):
    """Complete application store assembled from narrow domain contracts."""


@runtime_checkable
class KnowledgeSearchStore(Protocol):
    async def search(
        self,
        query: str,
        *,
        include_archive: bool = False,
        limit: int = 10,
    ) -> tuple[KnowledgeSearchHit, ...]: ...


@runtime_checkable
class SubscriptionStore(Protocol):
    async def save_subscription(
        self,
        subscription: SourceSubscription,
        *,
        expected_revision: int | None,
    ) -> None: ...

    async def list_subscriptions(
        self,
        *,
        enabled_only: bool = False,
    ) -> tuple[SourceSubscription, ...]: ...
