from __future__ import annotations

import unittest
from uuid import uuid4

from pydantic import ValidationError

from applications.personal_knowledge import (
    CategoryChangeType,
    ConfirmedCategoryChange,
    ConfirmedKnowledgeChange,
    KnowledgeChangeType,
    KnowledgeDocument,
    SourceKind,
    SourceRecord,
    SourceTextArtifact,
    TextArtifactKind,
    TranscriptSegment,
)


class PersonalKnowledgeModelTests(unittest.TestCase):
    def test_video_source_requires_canonical_url(self) -> None:
        with self.assertRaises(ValidationError):
            SourceRecord(kind=SourceKind.VIDEO)

    def test_transcript_requires_ordered_timestamped_segments(self) -> None:
        source_id = uuid4()
        artifact = SourceTextArtifact.from_text(
            source_id=source_id,
            kind=TextArtifactKind.TRANSCRIPT,
            text="first second",
            segments=(
                TranscriptSegment(start_seconds=0, end_seconds=2, text="first"),
                TranscriptSegment(start_seconds=2, end_seconds=4, text="second"),
            ),
        )

        self.assertEqual(artifact.source_id, source_id)
        self.assertEqual(len(artifact.content_hash), 64)

        with self.assertRaises(ValidationError):
            SourceTextArtifact.from_text(
                source_id=source_id,
                kind=TextArtifactKind.TRANSCRIPT,
                text="missing timestamps",
            )

    def test_confirmed_knowledge_change_enforces_review_boundary(self) -> None:
        with self.assertRaises(ValidationError):
            ConfirmedKnowledgeChange(
                change_type=KnowledgeChangeType.PUBLISH,
                document=KnowledgeDocument(
                    title="Unreviewed",
                    category_id=uuid4(),
                    body="Must not publish.",
                ),
            )

    def test_category_change_requires_confirmation_payload(self) -> None:
        with self.assertRaises(ValidationError):
            ConfirmedCategoryChange(change_type=CategoryChangeType.RENAME)


if __name__ == "__main__":
    unittest.main()

