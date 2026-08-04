from __future__ import annotations

import unittest
from datetime import timedelta
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from uuid import uuid4

from adaptive_agent_runtime.context_memory import InMemoryMemoryStore
from adaptive_agent_runtime.llm.capabilities import (
    CapabilityTurnKind,
    CapabilityTurnResult,
    MemoryCandidateDraft,
    MemoryConditionDraft,
    MemoryEvolutionDraft,
)

from applications.personal_knowledge import (
    CategoryChangeType,
    ConfirmedCategoryChange,
    DeterministicKnowledgeSynthesizer,
    DeterministicKnowledgeQuestionAnswerer,
    EditableKnowledgeConfirmation,
    HTTPTextResponse,
    ExtractedVideoTranscript,
    PersonalKnowledgeService,
    PreferenceMemoryService,
    RuntimePreferenceInferenceService,
    SourceSubscription,
    KnowledgeRetrievalService,
    SQLitePersonalKnowledgeStore,
    SourceKind,
    SourceToolRunner,
    SubprocessVideoTranscriptExtractor,
    SubscriptionKind,
    SubscriptionPoller,
    TextArtifactKind,
    TranscriptSegment,
    build_source_tool_stack,
)
from applications.personal_knowledge.models import utc_now
from applications.personal_knowledge.source_tools import VIDEO_EXTRACT_TRANSCRIPT
from applications.personal_knowledge import source_tools as source_tools_module
from applications.personal_knowledge.source_tools import (
    resolve_video_transcript_script,
)
from applications.personal_knowledge.errors import KnowledgeConflictError


class FakeHTTPTextFetcher:
    def __init__(self) -> None:
        self.body = "原文证据。"

    async def fetch(self, url: str) -> HTTPTextResponse:
        return HTTPTextResponse(
            final_url=url,
            content_type="text/html; charset=utf-8",
            text=(
                "<html><head><title>可审阅的来源</title>"
                "<script>污染</script></head>"
                f"<body><article><h1>核心观点</h1><p>{self.body}</p>"
                "</article></body></html>"
            ),
        )


class FakeVideoTranscriptExtractor:
    async def extract(self, url: str) -> ExtractedVideoTranscript:
        return ExtractedVideoTranscript(
            canonical_url=url,
            title="视频标题",
            creator="作者",
            duration_seconds=1_205,
            text="第一段观点。第二段结论。",
            segments=(
                TranscriptSegment(start_seconds=0, end_seconds=600, text="第一段观点"),
                TranscriptSegment(start_seconds=600, end_seconds=1205, text="第二段结论"),
            ),
        )


class FakeMemoryExtractionCapability:
    module_id = "test.memory_extraction"
    capability_id = "memory_extraction"

    async def extract(self, request, *, invocation=None):  # type: ignore[no-untyped-def]
        del request, invocation
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=(
                MemoryCandidateDraft(
                    memory_key="personal_knowledge.preference.summary_style",
                    content={"preference": "先结论后证据"},
                    condition=MemoryConditionDraft(
                        facts={"application": "personal_knowledge"},
                        required_tags=("personal_knowledge",),
                    ),
                    evidence_reference_ids=("review:test",),
                    confidence=0.9,
                    evolution=MemoryEvolutionDraft.EXTEND,
                ),
            ),
        )


class PersonalKnowledgeWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = SQLitePersonalKnowledgeStore(":memory:")
        self.fetcher = FakeHTTPTextFetcher()
        self.stack = build_source_tool_stack(
            fetcher=self.fetcher,
            video_extractor=FakeVideoTranscriptExtractor(),
        )
        self.service = PersonalKnowledgeService(
            store=self.store,
            source_tools=SourceToolRunner(self.stack),
            synthesizer=DeterministicKnowledgeSynthesizer(),
        )

    def tearDown(self) -> None:
        self.store.close()

    async def _category_id(self):  # type: ignore[no-untyped-def]
        category_id = uuid4()
        await self.store.apply_category_change(
            ConfirmedCategoryChange(
                change_type=CategoryChangeType.CREATE,
                category_id=category_id,
                name="Agent 架构",
            )
        )
        return category_id

    async def test_unconfirmed_source_stays_outside_knowledge(self) -> None:
        session = await self.service.ingest_url("https://example.test/article")

        self.assertEqual(await self.store.list_entries(), ())
        self.assertEqual(session.source.kind, SourceKind.WEB_ARTICLE)
        self.assertNotIn("污染", session.artifact.text)
        archived = await self.store.load_review(session.review.review_id)
        self.assertIsNotNone(archived)
        self.assertEqual(archived.proposals[-1], session.proposal)  # type: ignore[union-attr]

        _, observation = await SourceToolRunner(self.stack).acquire_web_text(
            "https://example.test/traced"
        )
        trace = self.stack.trace_sink.entries_for(observation.invocation_id)
        self.assertGreaterEqual(len(trace), 2)

    async def test_confirmation_publishes_exact_edited_payload(self) -> None:
        category_id = await self._category_id()
        session = await self.service.ingest_text("初稿内容", title="初稿")

        entry = await self.service.confirm_publication(
            EditableKnowledgeConfirmation(
                review_id=session.review.review_id,
                proposal_id=session.proposal.proposal_id,
                expected_review_revision=session.review.revision,
                title="用户编辑后的标题",
                category_id=category_id,
                body="用户编辑后的正文",
                tags=("agent", "反思"),
                citations=session.proposal.citations,
            )
        )

        self.assertEqual(entry.document.title, "用户编辑后的标题")
        self.assertEqual(entry.document.body, "用户编辑后的正文")
        self.assertEqual(len(entry.document.citations), 1)
        archived = await self.store.load_review(session.review.review_id)
        self.assertEqual(archived.final_confirmation_id, entry.last_confirmation_id)  # type: ignore[union-attr]

    async def test_stale_dialog_cannot_publish(self) -> None:
        category_id = await self._category_id()
        session = await self.service.capture_idea("知识是经审阅的理解")
        await self.service.append_review_message(
            session.review.review_id,
            content="请强调这是我的判断",
        )

        with self.assertRaises(KnowledgeConflictError):
            await self.service.confirm_publication(
                EditableKnowledgeConfirmation(
                    review_id=session.review.review_id,
                    proposal_id=session.proposal.proposal_id,
                    expected_review_revision=session.review.revision,
                    title=session.proposal.title,
                    category_id=category_id,
                    body=session.proposal.body,
                    citations=session.proposal.citations,
                )
            )
        self.assertEqual(await self.store.list_entries(), ())

    async def test_fts_defaults_to_confirmed_knowledge_with_optional_archive(self) -> None:
        category_id = await self._category_id()
        pending = await self.service.ingest_text(
            "来源说知识库只是原料仓库",
            title="来源材料",
        )
        self.assertEqual(await self.store.search("原料仓库"), ())
        archive_hits = await self.store.search("原料仓库", include_archive=True)
        self.assertEqual(len(archive_hits), 1)
        self.assertEqual(archive_hits[0].scope.value, "archive")

        await self.service.confirm_publication(
            EditableKnowledgeConfirmation(
                review_id=pending.review.review_id,
                proposal_id=pending.proposal.proposal_id,
                expected_review_revision=pending.review.revision,
                title="个人知识库的边界",
                category_id=category_id,
                body="知识库必须体现个人思考，而不是原文堆积。",
                citations=pending.proposal.citations,
            )
        )
        hits = await self.store.search("个人思考")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].scope.value, "knowledge")
        self.assertEqual(len(await self.store.search("思考")), 1)

        retrieval = KnowledgeRetrievalService(
            store=self.store,
            answerer=DeterministicKnowledgeQuestionAnswerer(),
        )
        answer = await retrieval.answer("个人思考")
        self.assertIsNotNone(answer)
        self.assertEqual(len(answer.citations), 1)  # type: ignore[union-attr]

    async def test_video_is_a_registered_tool_and_archives_only_transcript(self) -> None:
        self.assertIsNotNone(self.stack.catalog.get(VIDEO_EXTRACT_TRANSCRIPT))
        providers = self.stack.registry.metadata_for_capability(
            VIDEO_EXTRACT_TRANSCRIPT
        )
        self.assertEqual(len(providers), 1)

        session = await self.service.ingest_video(
            "https://www.bilibili.com/video/BV1example/"
        )
        self.assertEqual(session.artifact.kind, TextArtifactKind.TRANSCRIPT)
        self.assertEqual(len(session.artifact.segments), 2)
        self.assertTrue(session.proposal.citations[0].start_seconds == 0)
        self.assertEqual(await self.store.list_entries(), ())

    async def test_app_native_subscription_creates_review_only_on_change(self) -> None:
        subscription = SourceSubscription(
            name="认可的频道",
            kind=SubscriptionKind.WEB,
            url="https://example.test/channel",
            interval_minutes=5,
            next_poll_at=utc_now() - timedelta(seconds=1),
        )
        await self.store.save_subscription(subscription, expected_revision=None)
        poller = SubscriptionPoller(
            store=self.store,
            source_tools=SourceToolRunner(self.stack),
            workflow=self.service,
        )

        first = await poller.poll_once()
        self.assertTrue(first[0].changed)
        self.assertEqual(len(await self.store.list_reviews()), 1)
        second = await poller.poll_once(force=True)
        self.assertFalse(second[0].changed)
        self.assertEqual(len(await self.store.list_reviews()), 1)
        self.fetcher.body = "更新后的新文章"
        third = await poller.poll_once(force=True)
        self.assertTrue(third[0].changed)
        self.assertEqual(len(await self.store.list_reviews()), 2)

    async def test_inferred_preference_waits_for_confirmation_in_runtime_memory(self) -> None:
        memory_store = InMemoryMemoryStore()
        preferences = PreferenceMemoryService(memory_store)
        inference = RuntimePreferenceInferenceService(
            FakeMemoryExtractionCapability()
        )
        suggestions = await inference.infer(
            observations=({"kind": "edit_pattern", "body_changed": True},),
            existing=(),
            evidence_reference="review:test",
        )
        self.assertEqual(await memory_store.list_all(), ())
        self.assertEqual(len(suggestions), 1)

        memory = await preferences.confirm_suggestion(suggestions[0])
        self.assertTrue(memory.memory_key.startswith("personal_knowledge.preference."))
        self.assertEqual(len(await preferences.recall()), 1)

    async def test_existing_entry_revision_and_recoverable_trash_use_confirmation(self) -> None:
        category_id = await self._category_id()
        session = await self.service.capture_idea("第一版思考")
        entry = await self.service.confirm_publication(
            EditableKnowledgeConfirmation(
                review_id=session.review.review_id,
                proposal_id=session.proposal.proposal_id,
                expected_review_revision=session.review.revision,
                title="会演进的知识",
                category_id=category_id,
                body="第一版",
                citations=session.proposal.citations,
            )
        )
        revision_review = await self.service.start_entry_revision(entry.entry_id)
        revision_proposal = revision_review.proposals[-1]
        updated = await self.service.confirm_publication(
            EditableKnowledgeConfirmation(
                review_id=revision_review.review_id,
                proposal_id=revision_proposal.proposal_id,
                expected_review_revision=revision_review.revision,
                target_entry_id=entry.entry_id,
                expected_entry_revision=entry.revision,
                title="会演进的知识",
                category_id=category_id,
                body="第二版，经重新审阅",
                citations=revision_proposal.citations,
            )
        )
        self.assertEqual(updated.revision, 1)
        self.assertEqual(len(await self.store.history_for(entry.entry_id)), 2)

        trashed = await self.service.confirm_status_change(
            entry_id=updated.entry_id,
            expected_revision=updated.revision,
            restore=False,
        )
        self.assertEqual(trashed.status.value, "trashed")
        restored = await self.service.confirm_status_change(
            entry_id=trashed.entry_id,
            expected_revision=trashed.revision,
            restore=True,
        )
        self.assertEqual(restored.status.value, "active")

    async def test_video_helper_temp_media_is_removed_after_extraction(self) -> None:
        before = set(
            Path(tempfile.gettempdir()).glob("personal-knowledge-video-*")
        )
        with tempfile.TemporaryDirectory() as helper_dir:
            script = Path(helper_dir) / "fake_video_helper.py"
            script.write_text(
                """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--url")
parser.add_argument("--output-dir")
parser.add_argument("--probe-only", action="store_true")
parser.add_argument("--json", action="store_true")
args = parser.parse_args()
output = Path(args.output_dir)
if args.probe_only:
    print(json.dumps({"duration_seconds": 10, "recommended_timeout_seconds": 60}))
else:
    transcript = output / "transcript.txt"
    segments = output / "segments.jsonl"
    transcript.write_text("转录正文", encoding="utf-8")
    segments.write_text(
        json.dumps(
            {"start": 0, "end": 10, "text": "转录正文"},
            ensure_ascii=False,
        ) + "\\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "success": True,
        "asr_status": "transcribed",
        "transcript_source": "asr",
        "transcript_char_count": 4,
        "transcript_path": str(transcript),
        "segments_path": str(segments),
        "duration_seconds": 10,
        "title": "测试视频",
    }, ensure_ascii=False))
""",
                encoding="utf-8",
            )
            extractor = SubprocessVideoTranscriptExtractor(
                script,
                python_executable=sys.executable,
            )
            extracted = await extractor.extract("https://example.com/video")
            self.assertEqual(extracted.text, "转录正文")
        after = set(
            Path(tempfile.gettempdir()).glob("personal-knowledge-video-*")
        )
        self.assertEqual(after, before)

    async def test_video_helper_is_discovered_in_sibling_hermes_repository(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            module_path = (
                root
                / "Adaptive_Agent_Runtime"
                / "applications"
                / "personal_knowledge"
                / "source_tools.py"
            )
            helper = (
                root
                / "Hermes"
                / "hermes-data"
                / "skills"
                / "media"
                / "video-summary"
                / "scripts"
                / "fetch_video_transcript.py"
            )
            helper.parent.mkdir(parents=True)
            helper.write_text("# test helper\n", encoding="utf-8")
            with (
                patch.object(source_tools_module, "__file__", str(module_path)),
                patch.dict(os.environ, {"PERSONAL_KNOWLEDGE_VIDEO_SCRIPT": ""}),
            ):
                resolved = resolve_video_transcript_script()
            self.assertEqual(resolved, helper.resolve())


if __name__ == "__main__":
    unittest.main()
