"""Knowledge synthesis backed by Runtime cognitive capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from pydantic import Field

from adaptive_agent_runtime.llm.capabilities import (
    ArtifactGenerationCapability,
    CapabilityTurnKind,
    EvidenceReference,
    GenerationRequest,
)
from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    ImmutableJsonValue,
)

from applications.personal_knowledge.models import KnowledgeModel, ReviewMessage


class KnowledgeSynthesisDraft(KnowledgeModel):
    """Authority-free draft; it cannot be persisted as knowledge directly."""

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    suggested_category: str | None = Field(default=None, min_length=1)
    tags: tuple[str, ...] = ()


class KnowledgeSynthesizer(Protocol):
    async def synthesize(
        self,
        *,
        source_text: str,
        source_title: str | None,
        conversation: Sequence[ReviewMessage],
    ) -> KnowledgeSynthesisDraft: ...


class DeterministicKnowledgeSynthesizer:
    """Offline-safe synthesizer for local demos and deterministic tests."""

    async def synthesize(
        self,
        *,
        source_text: str,
        source_title: str | None,
        conversation: Sequence[ReviewMessage],
    ) -> KnowledgeSynthesisDraft:
        normalized = "\n".join(
            line.strip() for line in source_text.splitlines() if line.strip()
        )
        title = source_title or normalized.splitlines()[0][:80]
        user_reflections = [
            item.content.strip() for item in conversation if item.role.value == "user"
        ]
        body = normalized
        if user_reflections:
            body = f"{normalized}\n\n我的思考\n" + "\n".join(
                f"- {item}" for item in user_reflections
            )
        return KnowledgeSynthesisDraft(title=title, body=body)


class RuntimeKnowledgeSynthesizer:
    """Adapter that asks Runtime's generation capability for a structured draft."""

    _OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "body": {"type": "string", "minLength": 1},
            "suggested_category": {"type": ["string", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "body", "suggested_category", "tags"],
        "additionalProperties": False,
    }

    def __init__(self, capability: ArtifactGenerationCapability) -> None:
        self._capability = capability

    async def synthesize(
        self,
        *,
        source_text: str,
        source_title: str | None,
        conversation: Sequence[ReviewMessage],
    ) -> KnowledgeSynthesisDraft:
        context = {
            "source_title": source_title,
            "source_text": source_text,
            "review_conversation": [
                {"role": item.role.value, "content": item.content}
                for item in conversation
            ],
        }
        turn = await self._capability.generate(
            GenerationRequest(
                instruction=(
                    "整理为个人知识提案。区分来源观点与用户思考，保留分歧和不确定性；"
                    "只输出符合 schema 的结构化内容，不执行入库。"
                ),
                context=cast(ImmutableJsonValue, context),
                media_type="application/json",
                output_schema=cast(ImmutableJsonObject, self._OUTPUT_SCHEMA),
                evidence=(
                    EvidenceReference(
                        reference_id="source-text",
                        kind="source.archive.text",
                        summary="Archived source text for the review session.",
                        reliability=1.0,
                    ),
                ),
            )
        )
        if turn.kind is not CapabilityTurnKind.COMPLETED or turn.result is None:
            raise ValueError("knowledge synthesis returned unresolved Tool intents")
        content = turn.result.content
        if not isinstance(content, Mapping):
            raise ValueError("knowledge synthesis did not return structured content")
        return KnowledgeSynthesisDraft.model_validate(dict(content))


class ChunkedKnowledgeSynthesizer:
    """Ensure every long-source window contributes to final synthesis."""

    def __init__(
        self,
        delegate: KnowledgeSynthesizer,
        *,
        max_chunk_characters: int = 24_000,
    ) -> None:
        if max_chunk_characters < 1_000:
            raise ValueError("synthesis chunks must be at least 1000 characters")
        self._delegate = delegate
        self._max_chunk_characters = max_chunk_characters

    async def synthesize(
        self,
        *,
        source_text: str,
        source_title: str | None,
        conversation: Sequence[ReviewMessage],
    ) -> KnowledgeSynthesisDraft:
        if len(source_text) <= self._max_chunk_characters:
            return await self._delegate.synthesize(
                source_text=source_text,
                source_title=source_title,
                conversation=conversation,
            )
        chunks = self._chunks(source_text)
        notes: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            draft = await self._delegate.synthesize(
                source_text=chunk,
                source_title=f"{source_title or '长内容'} · 窗口 {index}/{len(chunks)}",
                conversation=(),
            )
            notes.append(
                f"## 覆盖窗口 {index}/{len(chunks)}\n标题：{draft.title}\n{draft.body}"
            )
        return await self._delegate.synthesize(
            source_text="\n\n".join(notes),
            source_title=source_title,
            conversation=conversation,
        )

    def _chunks(self, text: str) -> tuple[str, ...]:
        return tuple(
            text[index : index + self._max_chunk_characters]
            for index in range(0, len(text), self._max_chunk_characters)
        )
