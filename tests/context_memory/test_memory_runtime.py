from __future__ import annotations

import asyncio
import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    EvidenceDrivenMemoryConsolidator,
    InMemoryMemoryStore,
    MemoryCandidate,
    MemoryCondition,
    MemoryConsolidationError,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryNotFoundError,
    MemoryRecallQuery,
    MemoryStatus,
    MemoryUnit,
)


class YieldingMemoryStore(InMemoryMemoryStore):
    async def load(self, memory_id: UUID) -> MemoryUnit | None:
        memory = await super().load(memory_id)
        await asyncio.sleep(0)
        return memory

    async def load_applied_candidate(
        self,
        candidate_id: UUID,
    ) -> MemoryUnit | None:
        memory = await super().load_applied_candidate(candidate_id)
        await asyncio.sleep(0)
        return memory

    async def save(
        self,
        memory: MemoryUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        await asyncio.sleep(0)
        await super().save(memory, expected_revision=expected_revision)


def evidence(note: str) -> MemoryEvidence:
    return MemoryEvidence(source_reference=f"source:{note}", note=note)


def candidate(
    *,
    evolution: MemoryEvolutionType,
    content: str,
    condition: MemoryCondition,
    target: UUID | None = None,
    confidence: float = 0.7,
    note: str = "evidence",
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_key="research.preference",
        content=content,
        condition=condition,
        evidence=(evidence(note),),
        confidence=confidence,
        evolution=evolution,
        target_memory_id=target,
    )


class MemoryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryMemoryStore()
        self.consolidator = EvidenceDrivenMemoryConsolidator(self.store)
        self.condition = MemoryCondition(
            facts={"market": "CN"},
            required_tags=("research",),
        )
        created = await self.consolidator.consolidate(
            candidate(
                evolution=MemoryEvolutionType.EXTEND,
                content="prefer audited sources",
                condition=self.condition,
                confidence=0.6,
                note="initial",
            )
        )
        self.initial = created.memory

    async def test_extend_creates_conditional_evidence_backed_memory(self) -> None:
        self.assertEqual(self.initial.status, MemoryStatus.ACTIVE)
        self.assertEqual(self.initial.condition, self.condition)
        self.assertEqual(len(self.initial.evidence), 1)
        self.assertEqual(self.initial.revision, 0)

    async def test_replaying_candidate_is_idempotent(self) -> None:
        store = InMemoryMemoryStore()
        consolidator = EvidenceDrivenMemoryConsolidator(store)
        proposal = candidate(
            evolution=MemoryEvolutionType.EXTEND,
            content="stable claim",
            condition=self.condition,
        )

        first = await consolidator.consolidate(proposal)
        replay = await consolidator.consolidate(proposal)

        self.assertEqual(replay.memory, first.memory)
        self.assertEqual(await store.list_all(), (first.memory,))
        self.assertEqual(store.history_for(first.memory.memory_id), (first.memory,))

    async def test_reused_candidate_id_with_changed_payload_is_rejected(self) -> None:
        store = InMemoryMemoryStore()
        consolidator = EvidenceDrivenMemoryConsolidator(store)
        proposal = candidate(
            evolution=MemoryEvolutionType.EXTEND,
            content="original claim",
            condition=self.condition,
        )
        first = await consolidator.consolidate(proposal)
        collision = proposal.model_copy(update={"content": "different claim"})

        with self.assertRaises(MemoryConsolidationError):
            await consolidator.consolidate(collision)

        self.assertEqual(await store.list_all(), (first.memory,))
        self.assertEqual(store.history_for(first.memory.memory_id), (first.memory,))

    async def test_support_adds_evidence_and_increases_confidence(self) -> None:
        result = await self.consolidator.consolidate(
            candidate(
                evolution=MemoryEvolutionType.SUPPORT,
                content="prefer audited sources",
                condition=self.condition,
                target=self.initial.memory_id,
                confidence=0.5,
                note="support",
            )
        )

        self.assertGreater(result.memory.confidence, self.initial.confidence)
        self.assertEqual(len(result.memory.evidence), 2)
        self.assertEqual(result.memory.revision, self.initial.revision + 1)
        self.assertEqual(len(self.initial.evidence), 1)

    async def test_concurrent_support_updates_preserve_both_evidence_items(self) -> None:
        store = YieldingMemoryStore()
        consolidator = EvidenceDrivenMemoryConsolidator(store)
        initial = (
            await consolidator.consolidate(
                candidate(
                    evolution=MemoryEvolutionType.EXTEND,
                    content="prefer audited sources",
                    condition=self.condition,
                    note="base",
                )
            )
        ).memory
        first = candidate(
            evolution=MemoryEvolutionType.SUPPORT,
            content="prefer audited sources",
            condition=self.condition,
            target=initial.memory_id,
            note="parallel-a",
        )
        second = candidate(
            evolution=MemoryEvolutionType.SUPPORT,
            content="prefer audited sources",
            condition=self.condition,
            target=initial.memory_id,
            note="parallel-b",
        )

        await asyncio.gather(
            consolidator.consolidate(first),
            consolidator.consolidate(second),
        )

        current = await store.load(initial.memory_id)
        assert current is not None
        self.assertEqual(current.revision, 2)
        self.assertEqual(
            {item.note for item in current.evidence},
            {"base", "parallel-a", "parallel-b"},
        )

    async def test_modify_replaces_claim_but_preserves_evidence(self) -> None:
        result = await self.consolidator.consolidate(
            candidate(
                evolution=MemoryEvolutionType.MODIFY,
                content="prefer audited primary sources",
                condition=self.condition,
                target=self.initial.memory_id,
                confidence=0.8,
                note="modify",
            )
        )

        self.assertEqual(result.memory.content, "prefer audited primary sources")
        self.assertEqual(len(result.memory.evidence), 2)
        self.assertEqual(result.previous_memory, self.initial)

    async def test_conflict_preserves_incumbent_and_records_counterclaim(self) -> None:
        result = await self.consolidator.consolidate(
            candidate(
                evolution=MemoryEvolutionType.CONFLICT,
                content="prefer fast secondary sources",
                condition=self.condition,
                target=self.initial.memory_id,
                confidence=0.9,
                note="conflict",
            )
        )

        self.assertEqual(result.memory.status, MemoryStatus.CONFLICTED)
        self.assertEqual(result.memory.content, self.initial.content)
        self.assertEqual(result.memory.conflicts[0].content, "prefer fast secondary sources")
        self.assertEqual(result.memory.evidence, self.initial.evidence)
        self.assertEqual(result.memory.conflicts[0].evidence[0].note, "conflict")
        self.assertLess(result.memory.confidence, self.initial.confidence)

        recall = ConditionalMemoryRecall(self.store)
        query = MemoryRecallQuery(facts={"market": "CN"}, tags=("research",))
        self.assertEqual(await recall.recall(query), ())
        included = await recall.recall(
            MemoryRecallQuery(
                facts={"market": "CN"},
                tags=("research",),
                include_conflicted=True,
            )
        )
        self.assertEqual(included, (result.memory,))

    async def test_conditional_recall_requires_exact_facts_and_tags(self) -> None:
        recall = ConditionalMemoryRecall(self.store)

        matched = await recall.recall(
            MemoryRecallQuery(facts={"market": "CN"}, tags=("research",))
        )
        wrong_fact = await recall.recall(
            MemoryRecallQuery(facts={"market": "US"}, tags=("research",))
        )
        missing_tag = await recall.recall(
            MemoryRecallQuery(facts={"market": "CN"})
        )

        self.assertEqual(matched, (self.initial,))
        self.assertEqual(wrong_fact, ())
        self.assertEqual(missing_tag, ())

    async def test_null_condition_requires_the_fact_to_be_present(self) -> None:
        condition = MemoryCondition(facts={"market": None})

        self.assertFalse(condition.matches({}, ()))
        self.assertTrue(condition.matches({"market": None}, ()))

    async def test_missing_target_is_rejected_without_writes(self) -> None:
        before = await self.store.list_all()
        with self.assertRaises(MemoryNotFoundError):
            await self.consolidator.consolidate(
                candidate(
                    evolution=MemoryEvolutionType.SUPPORT,
                    content="prefer audited sources",
                    condition=self.condition,
                    target=uuid4(),
                )
            )
        self.assertEqual(await self.store.list_all(), before)

    async def test_support_rejects_changed_content(self) -> None:
        before = await self.store.load(self.initial.memory_id)
        history = self.store.history_for(self.initial.memory_id)
        with self.assertRaises(MemoryConsolidationError):
            await self.consolidator.consolidate(
                candidate(
                    evolution=MemoryEvolutionType.SUPPORT,
                    content="different claim",
                    condition=self.condition,
                    target=self.initial.memory_id,
                )
            )
        self.assertEqual(await self.store.load(self.initial.memory_id), before)
        self.assertEqual(self.store.history_for(self.initial.memory_id), history)


if __name__ == "__main__":
    unittest.main()
