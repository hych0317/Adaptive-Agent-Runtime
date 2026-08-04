"""FTS retrieval and evidence-bound knowledge question answering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Protocol, cast
from uuid import UUID

from pydantic import Field, model_validator

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

from applications.personal_knowledge.contracts import KnowledgeSearchStore
from applications.personal_knowledge.models import (
    KnowledgeModel,
    KnowledgeSearchHit,
    SearchScope,
)


def search_reference_key(hit: KnowledgeSearchHit) -> str:
    if hit.scope is SearchScope.KNOWLEDGE:
        return f"entry:{hit.entry_id}"
    return f"artifact:{hit.artifact_id}"


class KnowledgeAnswerDraft(KnowledgeModel):
    answer: str = Field(min_length=1)
    reference_keys: tuple[str, ...] = Field(min_length=1)


class KnowledgeAnswerCitation(KnowledgeModel):
    reference_key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    scope: SearchScope
    entry_id: UUID | None = None
    source_id: UUID | None = None
    artifact_id: UUID | None = None
    source_url: str | None = None


class KnowledgeAnswer(KnowledgeModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    citations: tuple[KnowledgeAnswerCitation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> KnowledgeAnswer:
        keys = tuple(item.reference_key for item in self.citations)
        if len(set(keys)) != len(keys):
            raise ValueError("answer citations must be unique")
        return self


class KnowledgeQuestionAnswerer(Protocol):
    async def answer(
        self,
        *,
        question: str,
        hits: Sequence[KnowledgeSearchHit],
    ) -> KnowledgeAnswerDraft: ...


class SwitchableKnowledgeQuestionAnswerer:
    def __init__(self, delegate: KnowledgeQuestionAnswerer) -> None:
        self._delegate = delegate
        self._lock = Lock()

    def use(self, delegate: KnowledgeQuestionAnswerer) -> None:
        with self._lock:
            self._delegate = delegate

    async def answer(
        self,
        *,
        question: str,
        hits: Sequence[KnowledgeSearchHit],
    ) -> KnowledgeAnswerDraft:
        with self._lock:
            delegate = self._delegate
        return await delegate.answer(question=question, hits=hits)


class DeterministicKnowledgeQuestionAnswerer:
    async def answer(
        self,
        *,
        question: str,
        hits: Sequence[KnowledgeSearchHit],
    ) -> KnowledgeAnswerDraft:
        del question
        selected = tuple(hits[:3])
        if not selected:
            raise ValueError("cannot answer without retrieved knowledge")
        body = "\n\n".join(
            f"{item.title}：{item.snippet}" for item in selected
        )
        return KnowledgeAnswerDraft(
            answer=body,
            reference_keys=tuple(search_reference_key(item) for item in selected),
        )


class RuntimeKnowledgeQuestionAnswerer:
    _OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "reference_keys": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
        },
        "required": ["answer", "reference_keys"],
        "additionalProperties": False,
    }

    def __init__(self, capability: ArtifactGenerationCapability) -> None:
        self._capability = capability

    async def answer(
        self,
        *,
        question: str,
        hits: Sequence[KnowledgeSearchHit],
    ) -> KnowledgeAnswerDraft:
        catalog = [
            {
                "reference_key": search_reference_key(item),
                "scope": item.scope.value,
                "title": item.title,
                "snippet": item.snippet,
                "source_url": item.source_url,
            }
            for item in hits
        ]
        turn = await self._capability.generate(
            GenerationRequest(
                instruction=(
                    "仅依据给定个人知识证据回答问题。无法支持的内容要明确说明；"
                    "reference_keys 只能使用证据目录中的键。"
                ),
                context=cast(
                    ImmutableJsonValue,
                    {"question": question, "evidence_catalog": catalog},
                ),
                media_type="application/json",
                output_schema=cast(ImmutableJsonObject, self._OUTPUT_SCHEMA),
                evidence=tuple(
                    EvidenceReference(
                        reference_id=search_reference_key(item),
                        kind=f"personal_knowledge.{item.scope.value}",
                        summary=item.title,
                        reliability=1.0,
                    )
                    for item in hits
                ),
            )
        )
        if turn.kind is not CapabilityTurnKind.COMPLETED or turn.result is None:
            raise ValueError("knowledge answer returned unresolved Tool intents")
        if not isinstance(turn.result.content, Mapping):
            raise ValueError("knowledge answer did not return structured content")
        return KnowledgeAnswerDraft.model_validate(dict(turn.result.content))


class KnowledgeRetrievalService:
    def __init__(
        self,
        *,
        store: KnowledgeSearchStore,
        answerer: KnowledgeQuestionAnswerer,
    ) -> None:
        self._store = store
        self._answerer = answerer

    async def search(
        self,
        query: str,
        *,
        include_archive: bool = False,
        limit: int = 10,
    ) -> tuple[KnowledgeSearchHit, ...]:
        return await self._store.search(
            query,
            include_archive=include_archive,
            limit=limit,
        )

    async def answer(
        self,
        question: str,
        *,
        include_archive: bool = False,
        limit: int = 8,
    ) -> KnowledgeAnswer | None:
        hits = await self.search(
            question,
            include_archive=include_archive,
            limit=limit,
        )
        if not hits:
            return None
        draft = await self._answerer.answer(question=question, hits=hits)
        by_key = {search_reference_key(item): item for item in hits}
        unknown = set(draft.reference_keys).difference(by_key)
        if unknown:
            raise ValueError("answer cited evidence that retrieval did not provide")
        citations = tuple(
            KnowledgeAnswerCitation(
                reference_key=key,
                title=by_key[key].title,
                snippet=by_key[key].snippet,
                scope=by_key[key].scope,
                entry_id=by_key[key].entry_id,
                source_id=by_key[key].source_id,
                artifact_id=by_key[key].artifact_id,
                source_url=by_key[key].source_url,
            )
            for key in draft.reference_keys
        )
        return KnowledgeAnswer(
            question=question.strip(),
            answer=draft.answer,
            citations=citations,
        )
