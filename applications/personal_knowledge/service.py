"""Application workflow from source acquisition to confirmed publication."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from applications.personal_knowledge.cognition import KnowledgeSynthesizer
from applications.personal_knowledge.contracts import PersonalKnowledgeStore
from applications.personal_knowledge.errors import (
    KnowledgeConflictError,
    KnowledgeNotFoundError,
)
from applications.personal_knowledge.governed_writes import GovernedKnowledgeWriter
from applications.personal_knowledge.models import (
    Citation,
    ConfirmedKnowledgeChange,
    KnowledgeChangeType,
    KnowledgeDocument,
    KnowledgeEntry,
    KnowledgeModel,
    KnowledgeProposal,
    ReviewArchive,
    ReviewDraft,
    ReviewMessage,
    ReviewRole,
    SourceRecord,
    SourceTextArtifact,
    TextArtifactKind,
    TranscriptSegment,
    utc_now,
)
from applications.personal_knowledge.source_tools import (
    FetchedSourceText,
    SourceToolRunner,
)


class EditableKnowledgeConfirmation(KnowledgeModel):
    """Exact payload submitted by the editable confirmation dialog."""

    confirmation_id: UUID = Field(default_factory=uuid4)
    review_id: UUID
    proposal_id: UUID
    expected_review_revision: int = Field(ge=0)
    title: str = Field(min_length=1)
    category_id: UUID
    body: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    citations: tuple[Citation, ...] = ()
    target_entry_id: UUID | None = None
    expected_entry_revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_target(self) -> EditableKnowledgeConfirmation:
        if (self.target_entry_id is None) != (self.expected_entry_revision is None):
            raise ValueError("entry target and revision must be supplied together")
        return self


@dataclass(frozen=True)
class ReviewSessionResult:
    source: SourceRecord
    artifact: SourceTextArtifact
    review: ReviewArchive
    proposal: KnowledgeProposal


class PersonalKnowledgeService:
    def __init__(
        self,
        *,
        store: PersonalKnowledgeStore,
        source_tools: SourceToolRunner,
        synthesizer: KnowledgeSynthesizer,
        writer: GovernedKnowledgeWriter | None = None,
    ) -> None:
        self._store = store
        self._source_tools = source_tools
        self._synthesizer = synthesizer
        self._writer = writer or GovernedKnowledgeWriter(store)

    async def ingest_text(
        self,
        text: str,
        *,
        title: str | None = None,
        user_note: str | None = None,
    ) -> ReviewSessionResult:
        fetched, _ = await self._source_tools.acquire_manual_text(text, title=title)
        return await self._start_review(fetched, user_note=user_note)

    async def capture_idea(
        self,
        idea: str,
        *,
        title: str | None = None,
    ) -> ReviewSessionResult:
        fetched, _ = await self._source_tools.acquire_manual_text(idea, title=title)
        return await self._start_review(fetched, user_note=idea)

    async def ingest_url(
        self,
        url: str,
        *,
        user_note: str | None = None,
    ) -> ReviewSessionResult:
        fetched, _ = await self._source_tools.acquire_web_text(url)
        return await self._start_review(fetched, user_note=user_note)

    async def ingest_video(
        self,
        url: str,
        *,
        user_note: str | None = None,
    ) -> ReviewSessionResult:
        fetched, _ = await self._source_tools.acquire_video_transcript(url)
        return await self._start_review(fetched, user_note=user_note)

    async def start_review_from_fetched(
        self,
        fetched: FetchedSourceText,
        *,
        user_note: str | None = None,
    ) -> ReviewSessionResult:
        return await self._start_review(fetched, user_note=user_note)

    async def append_review_message(
        self,
        review_id: UUID,
        *,
        content: str,
    ) -> ReviewArchive:
        current = await self._required_review(review_id)
        if current.final_confirmation_id is not None:
            raise KnowledgeConflictError("a finalized review cannot be continued")
        user_message = ReviewMessage(role=ReviewRole.USER, content=content.strip())
        conversation = (*current.conversation, user_message)
        source_text, source_title = await self._review_source_context(current)
        synthesis = await self._synthesizer.synthesize(
            source_text=source_text,
            source_title=source_title,
            conversation=conversation,
        )
        proposal = KnowledgeProposal(
            title=synthesis.title,
            body=synthesis.body,
            suggested_category=synthesis.suggested_category,
            tags=synthesis.tags,
            citations=self._citations_for_review(current),
        )
        agent_message = ReviewMessage(
            role=ReviewRole.AGENT,
            content="已根据对话更新待审知识提案。",
        )
        updated = current.model_copy(
            update={
                "conversation": (*conversation, agent_message),
                "drafts": (*current.drafts, self._draft_from(proposal)),
                "proposals": (*current.proposals, proposal),
                "revision": current.revision + 1,
                "updated_at": utc_now(),
            }
        )
        await self._store.save_review(updated, expected_revision=current.revision)
        return updated

    async def start_entry_revision(self, entry_id: UUID) -> ReviewArchive:
        entry = await self._store.load_entry(entry_id)
        if entry is None:
            raise KnowledgeNotFoundError(f"entry '{entry_id}' was not found")
        if entry.status.value != "active":
            raise KnowledgeConflictError("trashed knowledge must be restored before editing")
        proposal = KnowledgeProposal(
            title=entry.document.title,
            body=entry.document.body,
            tags=entry.document.tags,
            citations=entry.document.citations,
        )
        review = ReviewArchive(
            target_entry_id=entry.entry_id,
            conversation=(
                ReviewMessage(
                    role=ReviewRole.AGENT,
                    content="已载入当前知识版本；确认前不会覆盖现有内容。",
                ),
            ),
            drafts=(self._draft_from(proposal),),
            proposals=(proposal,),
        )
        await self._store.save_review(review, expected_revision=None)
        return review

    async def confirm_publication(
        self,
        confirmation: EditableKnowledgeConfirmation,
    ) -> KnowledgeEntry:
        review = await self._required_review(confirmation.review_id)
        if review.final_confirmation_id is not None:
            raise KnowledgeConflictError("review was already finalized")
        if review.revision != confirmation.expected_review_revision:
            raise KnowledgeConflictError("confirmation is based on a stale review")
        if not any(
            item.proposal_id == confirmation.proposal_id
            for item in review.proposals
        ):
            raise KnowledgeNotFoundError("proposal does not belong to this review")
        if confirmation.target_entry_id != review.target_entry_id:
            raise KnowledgeConflictError("confirmation targets another knowledge entry")
        change_type = (
            KnowledgeChangeType.UPDATE
            if review.target_entry_id is not None
            else KnowledgeChangeType.PUBLISH
        )
        change = ConfirmedKnowledgeChange(
            confirmation_id=confirmation.confirmation_id,
            change_type=change_type,
            review_id=review.review_id,
            target_entry_id=confirmation.target_entry_id,
            expected_revision=confirmation.expected_entry_revision,
            document=KnowledgeDocument(
                title=confirmation.title.strip(),
                category_id=confirmation.category_id,
                body=confirmation.body.strip(),
                tags=tuple(tag.strip() for tag in confirmation.tags),
                citations=confirmation.citations,
            ),
        )
        return await self._writer.apply(change)

    async def confirm_status_change(
        self,
        *,
        entry_id: UUID,
        expected_revision: int,
        restore: bool,
    ) -> KnowledgeEntry:
        change = ConfirmedKnowledgeChange(
            change_type=(
                KnowledgeChangeType.RESTORE if restore else KnowledgeChangeType.DELETE
            ),
            target_entry_id=entry_id,
            expected_revision=expected_revision,
        )
        return await self._writer.apply(change)

    async def _start_review(
        self,
        fetched: FetchedSourceText,
        *,
        user_note: str | None,
    ) -> ReviewSessionResult:
        source = SourceRecord(
            kind=fetched.kind,
            canonical_url=fetched.canonical_url,
            title=fetched.title,
            creator=fetched.creator,
        )
        artifact = SourceTextArtifact.from_text(
            source_id=source.source_id,
            kind=(
                TextArtifactKind.TRANSCRIPT
                if fetched.kind.value == "video"
                else TextArtifactKind.ORIGINAL_TEXT
            ),
            text=fetched.text,
            segments=fetched.segments,
        )
        conversation: tuple[ReviewMessage, ...] = ()
        if user_note is not None and user_note.strip():
            conversation = (
                ReviewMessage(role=ReviewRole.USER, content=user_note.strip()),
            )
        synthesis = await self._synthesizer.synthesize(
            source_text=artifact.text,
            source_title=source.title,
            conversation=conversation,
        )
        citations = self._initial_citations(source, artifact)
        proposal = KnowledgeProposal(
            title=synthesis.title,
            body=synthesis.body,
            suggested_category=synthesis.suggested_category,
            tags=synthesis.tags,
            citations=citations,
        )
        review = ReviewArchive(
            source_ids=(source.source_id,),
            artifact_ids=(artifact.artifact_id,),
            conversation=(
                *conversation,
                ReviewMessage(
                    role=ReviewRole.AGENT,
                    content="已生成待审提案；确认前不会写入个人知识库。",
                ),
            ),
            drafts=(self._draft_from(proposal),),
            proposals=(proposal,),
        )
        await self._store.save_source(source)
        await self._store.save_artifact(artifact)
        await self._store.save_review(review, expected_revision=None)
        return ReviewSessionResult(
            source=source,
            artifact=artifact,
            review=review,
            proposal=proposal,
        )

    @staticmethod
    def _initial_citations(
        source: SourceRecord,
        artifact: SourceTextArtifact,
    ) -> tuple[Citation, ...]:
        if not artifact.segments:
            return (
                Citation(
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    source_url=source.canonical_url,
                    label=source.title or "原始文本",
                ),
            )
        windows: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        window_start = artifact.segments[0].start_seconds
        for segment in artifact.segments:
            if current and segment.start_seconds - window_start >= 600:
                windows.append(current)
                current = []
                window_start = segment.start_seconds
            current.append(segment)
        if current:
            windows.append(current)
        citations: list[Citation] = []
        for index, segments in enumerate(windows, start=1):
            first = segments[0]
            last = segments[-1]
            quote = " ".join(item.text for item in segments)[:240]
            citations.append(
                Citation(
                    source_id=source.source_id,
                    artifact_id=artifact.artifact_id,
                    source_url=source.canonical_url,
                    label=f"{source.title or '视频转录'} · 窗口 {index}",
                    quote=quote,
                    start_seconds=first.start_seconds,
                    end_seconds=last.end_seconds,
                )
            )
        return tuple(citations)

    async def _required_review(self, review_id: UUID) -> ReviewArchive:
        review = await self._store.load_review(review_id)
        if review is None:
            raise KnowledgeNotFoundError(f"review '{review_id}' was not found")
        return review

    async def _review_source_context(
        self,
        review: ReviewArchive,
    ) -> tuple[str, str | None]:
        texts: list[str] = []
        titles: list[str] = []
        for artifact_id in review.artifact_ids:
            artifact = await self._store.load_artifact(artifact_id)
            if artifact is None:
                raise KnowledgeNotFoundError(
                    f"artifact '{artifact_id}' was not found"
                )
            texts.append(artifact.text)
        for source_id in review.source_ids:
            source = await self._store.load_source(source_id)
            if source is not None and source.title is not None:
                titles.append(source.title)
        source_text = "\n\n".join(texts)
        source_title = " / ".join(titles) or None
        if not source_text and review.proposals:
            source_text = review.proposals[-1].body
            source_title = review.proposals[-1].title
        return source_text, source_title

    def _citations_for_review(
        self,
        review: ReviewArchive,
    ) -> tuple[Citation, ...]:
        if review.proposals:
            return review.proposals[-1].citations
        return ()

    @staticmethod
    def _draft_from(proposal: KnowledgeProposal) -> ReviewDraft:
        return ReviewDraft(
            content=json.dumps(
                proposal.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
