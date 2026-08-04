"""Frozen domain models for the personal knowledge application."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceKind(StrEnum):
    PLAIN_TEXT = "plain_text"
    WEB_ARTICLE = "web_article"
    VIDEO = "video"


class TextArtifactKind(StrEnum):
    ORIGINAL_TEXT = "original_text"
    TRANSCRIPT = "transcript"


class ReviewRole(StrEnum):
    USER = "user"
    AGENT = "agent"


class CategoryStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class CategoryChangeType(StrEnum):
    CREATE = "create"
    RENAME = "rename"
    RETIRE = "retire"
    MERGE = "merge"


class KnowledgeStatus(StrEnum):
    ACTIVE = "active"
    TRASHED = "trashed"


class SearchScope(StrEnum):
    KNOWLEDGE = "knowledge"
    ARCHIVE = "archive"


class SubscriptionKind(StrEnum):
    WEB = "web"
    VIDEO = "video"


class KnowledgeChangeType(StrEnum):
    PUBLISH = "publish"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"


class SourceRecord(KnowledgeModel):
    source_id: UUID = Field(default_factory=uuid4)
    kind: SourceKind
    canonical_url: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    creator: str | None = Field(default=None, min_length=1)
    published_at: AwareDatetime | None = None
    captured_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        if self.kind is not SourceKind.PLAIN_TEXT and self.canonical_url is None:
            raise ValueError("web and video sources require a canonical URL")
        return self


class TranscriptSegment(KnowledgeModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("transcript segment end must follow its start")
        return self


class SourceTextArtifact(KnowledgeModel):
    artifact_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    kind: TextArtifactKind
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    segments: tuple[TranscriptSegment, ...] = ()
    captured_at: AwareDatetime = Field(default_factory=utc_now)

    @classmethod
    def from_text(
        cls,
        *,
        source_id: UUID,
        kind: TextArtifactKind,
        text: str,
        segments: tuple[TranscriptSegment, ...] = (),
        artifact_id: UUID | None = None,
        captured_at: datetime | None = None,
    ) -> SourceTextArtifact:
        normalized = text.strip()
        return cls(
            artifact_id=artifact_id or uuid4(),
            source_id=source_id,
            kind=kind,
            text=normalized,
            content_hash=sha256(normalized.encode("utf-8")).hexdigest(),
            segments=segments,
            captured_at=captured_at or utc_now(),
        )

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        expected = sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash != expected:
            raise ValueError("artifact content hash does not match its text")
        if self.kind is TextArtifactKind.TRANSCRIPT and not self.segments:
            raise ValueError("transcript artifacts require timestamped segments")
        previous_end = 0.0
        for segment in self.segments:
            if segment.start_seconds < previous_end:
                raise ValueError("transcript segments cannot overlap or regress")
            previous_end = segment.end_seconds
        return self


class ReviewMessage(KnowledgeModel):
    message_id: UUID = Field(default_factory=uuid4)
    role: ReviewRole
    content: str = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=utc_now)


class ReviewDraft(KnowledgeModel):
    draft_id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=utc_now)


class KnowledgeProposal(KnowledgeModel):
    proposal_id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    suggested_category: str | None = Field(default=None, min_length=1)
    tags: tuple[str, ...] = ()
    citations: tuple["Citation", ...] = ()
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        normalized_tags = tuple(tag.strip().casefold() for tag in self.tags)
        if any(not tag for tag in normalized_tags):
            raise ValueError("proposal tags cannot be blank")
        if len(set(normalized_tags)) != len(normalized_tags):
            raise ValueError("proposal tags must be unique")
        return self


class ReviewArchive(KnowledgeModel):
    review_id: UUID = Field(default_factory=uuid4)
    source_ids: tuple[UUID, ...] = ()
    artifact_ids: tuple[UUID, ...] = ()
    target_entry_id: UUID | None = None
    conversation: tuple[ReviewMessage, ...] = ()
    drafts: tuple[ReviewDraft, ...] = ()
    proposals: tuple[KnowledgeProposal, ...] = ()
    final_confirmation_id: UUID | None = None
    revision: int = Field(default=0, ge=0)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_archive(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("review update cannot precede creation")
        for values, label in (
            (self.source_ids, "source ids"),
            (self.artifact_ids, "artifact ids"),
            (tuple(item.message_id for item in self.conversation), "message ids"),
            (tuple(item.draft_id for item in self.drafts), "draft ids"),
            (tuple(item.proposal_id for item in self.proposals), "proposal ids"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"review {label} must be unique")
        return self


class Category(KnowledgeModel):
    category_id: UUID
    name: str = Field(min_length=1)
    status: CategoryStatus = CategoryStatus.ACTIVE
    merged_into: UUID | None = None
    revision: int = Field(default=0, ge=0)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_category(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("category update cannot precede creation")
        if self.status is CategoryStatus.ACTIVE and self.merged_into is not None:
            raise ValueError("an active category cannot be merged")
        if self.merged_into == self.category_id:
            raise ValueError("a category cannot merge into itself")
        return self


class ConfirmedCategoryChange(KnowledgeModel):
    confirmation_id: UUID = Field(default_factory=uuid4)
    change_type: CategoryChangeType
    category_id: UUID = Field(default_factory=uuid4)
    name: str | None = Field(default=None, min_length=1)
    target_category_id: UUID | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    confirmed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_change(self) -> Self:
        if self.change_type is CategoryChangeType.CREATE:
            if self.name is None:
                raise ValueError("category creation requires a name")
            if self.expected_revision is not None or self.target_category_id is not None:
                raise ValueError("category creation cannot target existing state")
        elif self.change_type is CategoryChangeType.RENAME:
            if self.name is None or self.expected_revision is None:
                raise ValueError("category rename requires a name and revision")
            if self.target_category_id is not None:
                raise ValueError("category rename cannot target another category")
        elif self.change_type is CategoryChangeType.RETIRE:
            if self.expected_revision is None:
                raise ValueError("category retirement requires a revision")
            if self.name is not None or self.target_category_id is not None:
                raise ValueError("category retirement accepts no name or target")
        else:
            if self.expected_revision is None or self.target_category_id is None:
                raise ValueError("category merge requires a revision and target")
            if self.name is not None:
                raise ValueError("category merge accepts no name")
            if self.target_category_id == self.category_id:
                raise ValueError("a category cannot merge into itself")
        return self


class Citation(KnowledgeModel):
    citation_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    artifact_id: UUID | None = None
    source_url: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)
    quote: str | None = Field(default=None, min_length=1)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_locator(self) -> Self:
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("citation timestamps must be supplied together")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("citation end must follow its start")
        return self


class KnowledgeDocument(KnowledgeModel):
    title: str = Field(min_length=1)
    category_id: UUID
    body: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        normalized_tags = tuple(tag.strip().casefold() for tag in self.tags)
        if any(not tag for tag in normalized_tags):
            raise ValueError("knowledge tags cannot be blank")
        if len(set(normalized_tags)) != len(normalized_tags):
            raise ValueError("knowledge tags must be unique")
        citation_ids = tuple(item.citation_id for item in self.citations)
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("knowledge citation ids must be unique")
        return self


class ConfirmedKnowledgeChange(KnowledgeModel):
    confirmation_id: UUID = Field(default_factory=uuid4)
    change_type: KnowledgeChangeType
    review_id: UUID | None = None
    target_entry_id: UUID | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    document: KnowledgeDocument | None = None
    confirmed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_change(self) -> Self:
        if self.change_type is KnowledgeChangeType.PUBLISH:
            if self.review_id is None or self.document is None:
                raise ValueError("publication requires a review and document")
            if self.target_entry_id is not None or self.expected_revision is not None:
                raise ValueError("publication cannot target an existing entry")
        elif self.change_type is KnowledgeChangeType.UPDATE:
            if (
                self.review_id is None
                or self.document is None
                or self.target_entry_id is None
                or self.expected_revision is None
            ):
                raise ValueError("update requires review, document, target, and revision")
        else:
            if self.target_entry_id is None or self.expected_revision is None:
                raise ValueError("delete and restore require a target and revision")
            if self.document is not None or self.review_id is not None:
                raise ValueError("delete and restore accept no document or review")
        return self


class KnowledgeEntry(KnowledgeModel):
    entry_id: UUID
    document: KnowledgeDocument
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    revision: int = Field(default=0, ge=0)
    last_confirmation_id: UUID
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("knowledge update cannot precede creation")
        if self.status is KnowledgeStatus.ACTIVE and self.deleted_at is not None:
            raise ValueError("active knowledge cannot have deleted_at")
        if self.status is KnowledgeStatus.TRASHED and self.deleted_at is None:
            raise ValueError("trashed knowledge requires deleted_at")
        return self


class CategoryChangeResult(KnowledgeModel):
    category: Category
    affected_entry_ids: tuple[UUID, ...] = ()


class KnowledgeSearchHit(KnowledgeModel):
    scope: SearchScope
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    score: float
    entry_id: UUID | None = None
    source_id: UUID | None = None
    artifact_id: UUID | None = None
    source_url: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.scope is SearchScope.KNOWLEDGE:
            if self.entry_id is None or self.artifact_id is not None:
                raise ValueError("knowledge search hits require an entry only")
        elif self.artifact_id is None or self.source_id is None:
            raise ValueError("archive search hits require source and artifact")
        return self


class SourceSubscription(KnowledgeModel):
    subscription_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    kind: SubscriptionKind
    url: str = Field(min_length=1)
    interval_minutes: int = Field(default=60, ge=5, le=43_200)
    enabled: bool = True
    last_content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    last_polled_at: AwareDatetime | None = None
    next_poll_at: AwareDatetime = Field(default_factory=utc_now)
    last_error: str | None = Field(default=None, min_length=1)
    revision: int = Field(default=0, ge=0)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    def after_poll(
        self,
        *,
        content_hash: str | None,
        error: str | None,
        at: datetime | None = None,
    ) -> SourceSubscription:
        now = at or utc_now()
        return self.model_copy(
            update={
                "last_content_hash": content_hash or self.last_content_hash,
                "last_polled_at": now,
                "next_poll_at": now + timedelta(minutes=self.interval_minutes),
                "last_error": error,
                "revision": self.revision + 1,
                "updated_at": now,
            }
        )


KnowledgeProposal.model_rebuild()
ReviewArchive.model_rebuild()
