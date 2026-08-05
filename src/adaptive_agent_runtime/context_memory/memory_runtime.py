"""Evidence-driven Memory consolidation and conditional recall."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from adaptive_agent_runtime.context_memory.contracts import MemoryStore
from adaptive_agent_runtime.context_memory.errors import (
    MemoryConsolidationError,
    MemoryNotFoundError,
    MemorySnapshotConflictError,
)
from adaptive_agent_runtime.context_memory.memory_models import (
    MemoryCandidate,
    MemoryConflict,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryRecallQuery,
    MemoryStatus,
    MemoryUnit,
    MemoryUpdateResult,
    MemoryBatchWrite,
)


_MEMORY_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/memory/candidate",
)


def _evolve_memory(
    memory: MemoryUnit,
    *,
    changed_at: datetime,
    **changes: Any,
) -> MemoryUnit:
    values = memory.model_dump(mode="python")
    values.update(changes)
    values["revision"] = memory.revision + 1
    values["updated_at"] = max(memory.updated_at, changed_at)
    return MemoryUnit.model_validate(values)


def _new_evidence(
    memory: MemoryUnit,
    candidate: MemoryCandidate,
) -> tuple[MemoryEvidence, ...]:
    existing_ids = {item.evidence_id for item in memory.evidence}
    additions = tuple(
        item for item in candidate.evidence if item.evidence_id not in existing_ids
    )
    if not additions:
        raise MemoryConsolidationError("candidate adds no new evidence")
    return additions


def _evidence_strength(candidate: MemoryCandidate) -> float:
    average_weight = sum(item.weight for item in candidate.evidence) / len(
        candidate.evidence
    )
    return candidate.confidence * average_weight


def _candidate_fingerprint(candidate: MemoryCandidate) -> str:
    serialized = json.dumps(
        candidate.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


class InMemoryMemoryStore:
    module_id = "memory.store.in_memory"

    def __init__(self) -> None:
        self._current: dict[UUID, MemoryUnit] = {}
        self._history: defaultdict[UUID, list[MemoryUnit]] = defaultdict(list)
        self._applied_candidates: dict[UUID, MemoryUnit] = {}
        self._applied_effects: dict[str, tuple[MemoryUnit, ...]] = {}

    async def save(
        self,
        memory: MemoryUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        candidate_id = memory.last_candidate_id
        if candidate_id is None:
            raise MemorySnapshotConflictError(
                "memory writes require an originating candidate"
            )
        applied = self._applied_candidates.get(candidate_id)
        if applied is not None:
            if applied == memory:
                return
            raise MemorySnapshotConflictError(
                "candidate id was already applied with different content"
            )

        current = self._current.get(memory.memory_id)
        if expected_revision is None:
            if current is not None:
                raise MemorySnapshotConflictError(
                    "memory already has a current snapshot"
                )
        elif (
            current is None
            or current.revision != expected_revision
            or memory.revision != expected_revision + 1
        ):
            raise MemorySnapshotConflictError(
                "memory write is based on a stale snapshot"
            )
        self._current[memory.memory_id] = memory
        self._history[memory.memory_id].append(memory)
        self._applied_candidates[candidate_id] = memory

    async def load(self, memory_id: UUID) -> MemoryUnit | None:
        return self._current.get(memory_id)

    async def load_applied_candidate(
        self,
        candidate_id: UUID,
    ) -> MemoryUnit | None:
        return self._applied_candidates.get(candidate_id)

    async def list_all(self) -> tuple[MemoryUnit, ...]:
        return tuple(self._current.values())

    async def save_batch(
        self,
        writes: tuple[MemoryBatchWrite, ...],
        *,
        effect_fingerprint: str,
    ) -> tuple[MemoryUnit, ...]:
        applied_effect = self._applied_effects.get(effect_fingerprint)
        if applied_effect is not None:
            return applied_effect
        current = dict(self._current)
        history = {key: list(value) for key, value in self._history.items()}
        applied_candidates = dict(self._applied_candidates)
        committed: list[MemoryUnit] = []
        for write in writes:
            memory = write.memory
            candidate_id = memory.last_candidate_id
            if candidate_id is None or candidate_id in applied_candidates:
                raise MemorySnapshotConflictError(
                    "memory batch contains an absent or previously applied candidate"
                )
            existing = current.get(memory.memory_id)
            if write.expected_revision is None:
                if existing is not None:
                    raise MemorySnapshotConflictError(
                        "memory batch create collides with current state"
                    )
            elif (
                existing is None
                or existing.revision != write.expected_revision
                or memory.revision != write.expected_revision + 1
            ):
                raise MemorySnapshotConflictError(
                    "memory batch write is based on a stale snapshot"
                )
            current[memory.memory_id] = memory
            history.setdefault(memory.memory_id, []).append(memory)
            applied_candidates[candidate_id] = memory
            committed.append(memory)
        self._current = current
        self._history = defaultdict(list, history)
        self._applied_candidates = applied_candidates
        result = tuple(committed)
        self._applied_effects[effect_fingerprint] = result
        return result

    async def load_applied_effect(
        self,
        effect_fingerprint: str,
    ) -> tuple[MemoryUnit, ...] | None:
        return self._applied_effects.get(effect_fingerprint)

    def history_for(self, memory_id: UUID) -> tuple[MemoryUnit, ...]:
        return tuple(self._history.get(memory_id, ()))


class EvidenceDrivenMemoryConsolidator:
    module_id = "memory.evidence_consolidator"

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def consolidate(
        self,
        candidate: MemoryCandidate,
    ) -> MemoryUpdateResult:
        fingerprint = _candidate_fingerprint(candidate)
        for _ in range(8):
            applied = await self._store.load_applied_candidate(
                candidate.candidate_id
            )
            if applied is not None:
                if applied.last_candidate_fingerprint != fingerprint:
                    raise MemoryConsolidationError(
                        "candidate id was reused with a different payload"
                    )
                return MemoryUpdateResult(
                    evolution=candidate.evolution,
                    memory=applied,
                    candidate_id=candidate.candidate_id,
                )
            try:
                return await self._consolidate_once(candidate, fingerprint)
            except MemorySnapshotConflictError:
                continue
        raise MemoryConsolidationError(
            "memory changed repeatedly during consolidation"
        )

    async def consolidate_batch(
        self,
        candidates: tuple[MemoryCandidate, ...],
        *,
        effect_fingerprint: str,
    ) -> tuple[MemoryUpdateResult, ...]:
        if not candidates:
            raise MemoryConsolidationError("memory batch cannot be empty")
        existing = await self._store.load_applied_effect(effect_fingerprint)
        if existing is not None:
            by_candidate = {item.last_candidate_id: item for item in existing}
            if set(by_candidate) != {item.candidate_id for item in candidates}:
                raise MemoryConsolidationError(
                    "effect fingerprint was reused for another Memory batch"
                )
            return tuple(
                MemoryUpdateResult(
                    evolution=candidate.evolution,
                    memory=by_candidate[candidate.candidate_id],
                    candidate_id=candidate.candidate_id,
                )
                for candidate in candidates
            )
        for candidate in candidates:
            if await self._store.load_applied_candidate(candidate.candidate_id) is not None:
                raise MemoryConsolidationError(
                    "Memory batch has an indeterminate partial predecessor"
                )
        working = {item.memory_id: item for item in await self._store.list_all()}
        writes: list[MemoryBatchWrite] = []
        results: list[MemoryUpdateResult] = []
        for candidate in candidates:
            result, expected_revision = self._prepare_candidate(candidate, working)
            working[result.memory.memory_id] = result.memory
            writes.append(
                MemoryBatchWrite(
                    memory=result.memory,
                    expected_revision=expected_revision,
                )
            )
            results.append(result)
        committed = await self._store.save_batch(
            tuple(writes), effect_fingerprint=effect_fingerprint
        )
        if tuple(item.memory for item in results) != committed:
            raise MemoryConsolidationError("Memory batch failed authoritative read-back")
        return tuple(results)

    def _prepare_candidate(
        self,
        candidate: MemoryCandidate,
        working: dict[UUID, MemoryUnit],
    ) -> tuple[MemoryUpdateResult, int | None]:
        fingerprint = _candidate_fingerprint(candidate)
        strength = _evidence_strength(candidate)
        if candidate.evolution is MemoryEvolutionType.EXTEND:
            created = MemoryUnit(
                memory_id=uuid5(_MEMORY_NAMESPACE, str(candidate.candidate_id)),
                memory_key=candidate.memory_key,
                content=candidate.content,
                condition=candidate.condition,
                evidence=candidate.evidence,
                confidence=strength,
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
                created_at=candidate.created_at,
                updated_at=candidate.created_at,
            )
            if created.memory_id in working:
                raise MemoryConsolidationError("Memory batch create identity already exists")
            return (
                MemoryUpdateResult(
                    evolution=candidate.evolution,
                    memory=created,
                    candidate_id=candidate.candidate_id,
                ),
                None,
            )
        target_id = candidate.target_memory_id
        if target_id is None or target_id not in working:
            raise MemoryNotFoundError(f"memory '{target_id}' was not found")
        current = working[target_id]
        if current.status is not MemoryStatus.ACTIVE:
            raise MemoryConsolidationError(
                "only active memory can receive evidence-driven updates"
            )
        if current.memory_key != candidate.memory_key:
            raise MemoryConsolidationError(
                "candidate memory_key does not match its target"
            )
        additions = _new_evidence(current, candidate)
        if candidate.evolution is MemoryEvolutionType.SUPPORT:
            if current.content != candidate.content or current.condition != candidate.condition:
                raise MemoryConsolidationError(
                    "support evidence must preserve content and condition"
                )
            updated = _evolve_memory(
                current,
                changed_at=candidate.created_at,
                evidence=(*current.evidence, *additions),
                confidence=1.0 - ((1.0 - current.confidence) * (1.0 - strength)),
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
            )
        elif candidate.evolution is MemoryEvolutionType.MODIFY:
            updated = _evolve_memory(
                current,
                changed_at=candidate.created_at,
                content=candidate.content,
                condition=candidate.condition,
                evidence=(*current.evidence, *additions),
                confidence=strength,
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
            )
        elif candidate.evolution is MemoryEvolutionType.CONFLICT:
            if current.condition != candidate.condition or current.content == candidate.content:
                raise MemoryConsolidationError("invalid conflicting Memory evidence")
            conflict = MemoryConflict(
                candidate_id=candidate.candidate_id,
                content=candidate.content,
                condition=candidate.condition,
                evidence=additions,
                confidence=candidate.confidence,
            )
            updated = _evolve_memory(
                current,
                changed_at=candidate.created_at,
                status=MemoryStatus.CONFLICTED,
                conflicts=(*current.conflicts, conflict),
                confidence=max(0.0, current.confidence - (strength * 0.5)),
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
            )
        else:
            raise MemoryConsolidationError(
                f"unsupported evolution '{candidate.evolution.value}'"
            )
        return (
            MemoryUpdateResult(
                evolution=candidate.evolution,
                memory=updated,
                previous_memory=current,
                candidate_id=candidate.candidate_id,
            ),
            current.revision,
        )

    async def _consolidate_once(
        self,
        candidate: MemoryCandidate,
        fingerprint: str,
    ) -> MemoryUpdateResult:
        strength = _evidence_strength(candidate)
        if candidate.evolution is MemoryEvolutionType.EXTEND:
            created = MemoryUnit(
                memory_id=uuid5(_MEMORY_NAMESPACE, str(candidate.candidate_id)),
                memory_key=candidate.memory_key,
                content=candidate.content,
                condition=candidate.condition,
                evidence=candidate.evidence,
                confidence=strength,
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
                created_at=candidate.created_at,
                updated_at=candidate.created_at,
            )
            await self._store.save(created, expected_revision=None)
            return MemoryUpdateResult(
                evolution=candidate.evolution,
                memory=created,
                candidate_id=candidate.candidate_id,
            )

        target_id = candidate.target_memory_id
        if target_id is None:
            raise MemoryConsolidationError("candidate has no target memory")
        current = await self._store.load(target_id)
        if current is None:
            raise MemoryNotFoundError(f"memory '{target_id}' was not found")
        if current.status is not MemoryStatus.ACTIVE:
            raise MemoryConsolidationError(
                "only active memory can receive evidence-driven updates"
            )
        if current.memory_key != candidate.memory_key:
            raise MemoryConsolidationError(
                "candidate memory_key does not match its target"
            )

        additions = _new_evidence(current, candidate)
        if candidate.evolution is MemoryEvolutionType.SUPPORT:
            if (
                current.content != candidate.content
                or current.condition != candidate.condition
            ):
                raise MemoryConsolidationError(
                    "support evidence must preserve content and condition"
                )
            confidence = 1.0 - (
                (1.0 - current.confidence) * (1.0 - strength)
            )
            updated = _evolve_memory(
                current,
                changed_at=candidate.created_at,
                evidence=(*current.evidence, *additions),
                confidence=confidence,
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
            )
        elif candidate.evolution is MemoryEvolutionType.MODIFY:
            updated = _evolve_memory(
                current,
                changed_at=candidate.created_at,
                content=candidate.content,
                condition=candidate.condition,
                evidence=(*current.evidence, *additions),
                confidence=strength,
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
            )
        elif candidate.evolution is MemoryEvolutionType.CONFLICT:
            if current.condition != candidate.condition:
                raise MemoryConsolidationError(
                    "conflicting evidence must refer to the same condition"
                )
            if current.content == candidate.content:
                raise MemoryConsolidationError(
                    "identical content is support, not conflict"
                )
            conflict = MemoryConflict(
                candidate_id=candidate.candidate_id,
                content=candidate.content,
                condition=candidate.condition,
                evidence=additions,
                confidence=candidate.confidence,
            )
            updated = _evolve_memory(
                current,
                changed_at=candidate.created_at,
                status=MemoryStatus.CONFLICTED,
                conflicts=(*current.conflicts, conflict),
                confidence=max(
                    0.0,
                    current.confidence - (strength * 0.5),
                ),
                last_candidate_id=candidate.candidate_id,
                last_candidate_fingerprint=fingerprint,
            )
        else:
            raise MemoryConsolidationError(
                f"unsupported evolution '{candidate.evolution.value}'"
            )

        await self._store.save(
            updated,
            expected_revision=current.revision,
        )
        return MemoryUpdateResult(
            evolution=candidate.evolution,
            memory=updated,
            previous_memory=current,
            candidate_id=candidate.candidate_id,
        )


class ConditionalMemoryRecall:
    module_id = "memory.conditional_recall"

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def recall(
        self,
        query: MemoryRecallQuery,
    ) -> tuple[MemoryUnit, ...]:
        recalled: list[MemoryUnit] = []
        for memory in await self._store.list_all():
            allowed_statuses = {MemoryStatus.ACTIVE}
            if query.include_conflicted:
                allowed_statuses.add(MemoryStatus.CONFLICTED)
            if memory.status not in allowed_statuses:
                continue
            if memory.confidence < query.min_confidence:
                continue
            if not memory.condition.matches(query.facts, query.tags):
                continue
            recalled.append(memory)

        recalled.sort(
            key=lambda memory: (
                -memory.confidence,
                -memory.updated_at.timestamp(),
                str(memory.memory_id),
            )
        )
        return tuple(recalled[: query.limit])
