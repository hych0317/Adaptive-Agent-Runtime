"""Context storage, lifecycle management, scheduling, and assembly."""

from __future__ import annotations

from asyncio import Lock
from collections import defaultdict
from typing import Any, Sequence
from uuid import UUID, uuid5

from adaptive_agent_runtime.context_memory.context_models import (
    ContextArchiveReference,
    ContextAssembly,
    ContextLayer,
    ContextLifecycleAction,
    ContextLifecycleDecision,
    ContextLifecycleResult,
    ContextLifecycleState,
    ContextMetadata,
    ContextPressure,
    ContextRequirement,
    ContextSchedule,
    ContextUnit,
    ResidencyPolicy,
)
from adaptive_agent_runtime.context_memory.compression_decision import (
    ContextCompressionEffect,
)
from adaptive_agent_runtime.context_memory.contracts import (
    ContextArchive,
    ContextCompressor,
    ContextCompressionTransaction,
    ContextLifecycleActionExecutor,
    ContextLifecyclePlanning,
    ContextPressureMonitoring,
    ContextStore,
)
from adaptive_agent_runtime.context_memory.errors import (
    ContextBudgetExceededError,
    ContextNotFoundError,
    ContextRecoveryError,
    ContextSnapshotConflictError,
    ContextTransitionError,
)
from adaptive_agent_runtime.context_memory.json_types import utc_now
from adaptive_agent_runtime.governance.models import GovernanceTarget, RuntimeCommitPermit


def _evolve_metadata(metadata: ContextMetadata, **changes: Any) -> ContextMetadata:
    values = metadata.model_dump(mode="python")
    values.update(changes)
    values["updated_at"] = utc_now()
    return ContextMetadata.model_validate(values)


def _evolve_unit(unit: ContextUnit, **changes: Any) -> ContextUnit:
    values = unit.model_dump(mode="python")
    values.update(changes)
    values["revision"] = unit.revision + 1
    values["metadata"] = _evolve_metadata(
        changes.get("metadata", unit.metadata)
    )
    return ContextUnit.model_validate(values)


class InMemoryContextStore:
    module_id = "context.store.in_memory"

    def __init__(self) -> None:
        self._current: dict[UUID, ContextUnit] = {}
        self._history: defaultdict[UUID, list[ContextUnit]] = defaultdict(list)

    async def save(
        self,
        unit: ContextUnit,
        *,
        expected_revision: int | None,
    ) -> None:
        if unit.lifecycle_state is ContextLifecycleState.ARCHIVED:
            raise ContextTransitionError("archived context cannot remain resident")
        current = self._current.get(unit.context_id)
        if expected_revision is None:
            if current is not None:
                if current == unit:
                    return
                raise ContextSnapshotConflictError(
                    "context already has a resident snapshot"
                )
            history = self._history.get(unit.context_id)
            if history:
                raise ContextSnapshotConflictError(
                    "archived context must be restored through a versioned write"
                )
        else:
            base = current
            if base is None:
                history = self._history.get(unit.context_id)
                base = history[-1] if history else None
            if (
                base is None
                or base.revision != expected_revision
                or unit.revision != expected_revision + 1
            ):
                raise ContextSnapshotConflictError(
                    "context write is based on a stale snapshot"
                )
        self._current[unit.context_id] = unit
        self._history[unit.context_id].append(unit)

    async def load(self, context_id: UUID) -> ContextUnit | None:
        return self._current.get(context_id)

    async def delete(
        self,
        context_id: UUID,
        *,
        expected_revision: int,
    ) -> None:
        current = self._current.get(context_id)
        if current is None or current.revision != expected_revision:
            raise ContextSnapshotConflictError(
                "context delete is based on a stale snapshot"
            )
        del self._current[context_id]

    async def list_for_run(self, run_id: UUID) -> tuple[ContextUnit, ...]:
        return tuple(
            unit
            for unit in self._current.values()
            if unit.metadata.run_id == run_id
        )

    def history_for(self, context_id: UUID) -> tuple[ContextUnit, ...]:
        return tuple(self._history.get(context_id, ()))


class InMemoryContextArchive:
    module_id = "context.archive.in_memory"

    def __init__(self) -> None:
        self._records: dict[
            UUID,
            tuple[ContextArchiveReference, ContextUnit],
        ] = {}

    async def archive(
        self,
        unit: ContextUnit,
        *,
        reference: ContextArchiveReference | None = None,
    ) -> ContextArchiveReference:
        reference = reference or ContextArchiveReference(context_id=unit.context_id)
        if reference.context_id != unit.context_id:
            raise ContextRecoveryError("archive reference has a different Context identity")
        existing = self._records.get(reference.archive_id)
        if existing is not None:
            if existing[1].context_id != unit.context_id:
                raise ContextRecoveryError("archive id is bound to another Context")
            return existing[0]
        values = unit.model_dump(mode="python")
        values["lifecycle_state"] = ContextLifecycleState.ARCHIVED
        values["metadata"] = _evolve_metadata(unit.metadata)
        archived = ContextUnit.model_validate(values)
        self._records[reference.archive_id] = (reference, archived)
        return reference

    async def discard(self, reference: ContextArchiveReference) -> None:
        record = self._records.get(reference.archive_id)
        if record is not None and record[1].context_id == reference.context_id:
            del self._records[reference.archive_id]

    async def restore(
        self,
        reference: ContextArchiveReference,
    ) -> ContextUnit:
        record = self._records.get(reference.archive_id)
        if record is None or record[1].context_id != reference.context_id:
            raise ContextRecoveryError(
                f"archive reference '{reference.archive_id}' is unavailable"
            )
        return record[1]

    async def find_latest(
        self,
        context_id: UUID,
    ) -> ContextArchiveReference | None:
        for reference, archived in reversed(tuple(self._records.values())):
            if archived.context_id == context_id:
                return reference
        return None

    def record_count(self) -> int:
        return len(self._records)


class ContextLifecycleManager:
    module_id = "context.lifecycle_manager"

    def __init__(
        self,
        *,
        store: ContextStore,
        archive: ContextArchive,
        compressor: ContextCompressor,
    ) -> None:
        self._store = store
        self._archive = archive
        self._compressor = compressor
        self._lock = Lock()

    async def add(self, unit: ContextUnit) -> None:
        async with self._lock:
            if unit.lifecycle_state is ContextLifecycleState.ARCHIVED:
                raise ContextTransitionError("cannot add archived context as resident")
            await self._store.save(unit, expected_revision=None)

    async def compress(self, context_id: UUID) -> ContextUnit:
        async with self._lock:
            return await self._compress(context_id)

    async def _compress(self, context_id: UUID) -> ContextUnit:
        unit = await self._require(context_id)
        if unit.lifecycle_state is not ContextLifecycleState.ACTIVE:
            raise ContextTransitionError("only active context can be compressed")
        self._ensure_evictable(unit)

        result = await self._compressor.compress(unit)
        committed = await self._store.load(unit.context_id)
        if (
            committed is not None
            and committed.revision == unit.revision + 1
            and committed.lifecycle_state is ContextLifecycleState.COMPRESSED
            and committed.content == result.content
            and committed.core_conclusions == result.core_conclusions
            and committed.metadata.estimated_tokens == result.estimated_tokens
            and committed.recovery_reference is not None
        ):
            return committed
        recovery_reference = await self._archive.archive(unit)
        if recovery_reference.context_id != unit.context_id:
            await self._archive.discard(recovery_reference)
            raise ContextRecoveryError(
                "archive returned a reference for a different context"
            )
        metadata = _evolve_metadata(
            unit.metadata,
            estimated_tokens=result.estimated_tokens,
        )
        compressed = _evolve_unit(
            unit,
            content=result.content,
            metadata=metadata,
            lifecycle_state=ContextLifecycleState.COMPRESSED,
            core_conclusions=result.core_conclusions,
            recovery_reference=recovery_reference,
        )
        try:
            await self._store.save(
                compressed,
                expected_revision=unit.revision,
            )
        except Exception:
            await self._archive.discard(recovery_reference)
            raise
        return compressed


    async def archive(self, context_id: UUID) -> ContextArchiveReference:
        async with self._lock:
            return await self._archive_resident(context_id)

    async def _archive_resident(
        self,
        context_id: UUID,
    ) -> ContextArchiveReference:
        unit = await self._require(context_id)
        if unit.lifecycle_state is not ContextLifecycleState.COMPRESSED:
            raise ContextTransitionError(
                "context must be compressed before public archival"
            )
        self._ensure_evictable(unit)
        reference = await self._archive.archive(unit)
        if reference.context_id != unit.context_id:
            await self._archive.discard(reference)
            raise ContextRecoveryError(
                "archive returned a reference for a different context"
            )
        try:
            await self._store.delete(
                context_id,
                expected_revision=unit.revision,
            )
        except Exception:
            await self._archive.discard(reference)
            raise
        return reference

    async def restore(
        self,
        reference: ContextArchiveReference,
    ) -> ContextUnit:
        async with self._lock:
            return await self._restore(reference)

    async def restore_context(self, context_id: UUID) -> ContextUnit:
        """Restore the newest archived snapshot for a required Context Unit."""

        async with self._lock:
            reference = await self._archive.find_latest(context_id)
            if reference is None:
                raise ContextRecoveryError(
                    f"context '{context_id}' has no archive recovery entry"
                )
            return await self._restore(reference)

    async def resolve_action_subject(
        self,
        decision: ContextLifecycleDecision,
    ) -> ContextUnit | ContextArchiveReference:
        """Resolve the immutable subject that Governance must authorize."""

        async with self._lock:
            if decision.action is ContextLifecycleAction.RESTORE:
                reference = await self._archive.find_latest(decision.context_id)
                if reference is None:
                    raise ContextRecoveryError(
                        f"context '{decision.context_id}' has no archive recovery entry"
                    )
                return reference
            unit = await self._require(decision.context_id)
            self._require_decision_revision(decision, unit)
            return unit

    async def apply(
        self,
        decision: ContextLifecycleDecision,
    ) -> ContextLifecycleResult:
        """Apply exactly one policy decision through versioned lifecycle methods."""

        async with self._lock:
            if decision.action is ContextLifecycleAction.COMPRESS:
                unit = await self._require(decision.context_id)
                self._require_decision_revision(decision, unit)
                compressed = await self._compress(decision.context_id)
                return ContextLifecycleResult(
                    decision=decision,
                    unit=compressed,
                    archive_reference=compressed.recovery_reference,
                )
            if decision.action is ContextLifecycleAction.ARCHIVE:
                unit = await self._require(decision.context_id)
                self._require_decision_revision(decision, unit)
                reference = await self._archive_resident(decision.context_id)
                archived = await self._archive.restore(reference)
                return ContextLifecycleResult(
                    decision=decision,
                    unit=archived,
                    archive_reference=reference,
                )
            if decision.action is ContextLifecycleAction.RESTORE:
                restore_reference = await self._archive.find_latest(
                    decision.context_id
                )
                if restore_reference is None:
                    raise ContextRecoveryError(
                        f"context '{decision.context_id}' has no archive recovery entry"
                    )
                restored = await self._restore(restore_reference)
                return ContextLifecycleResult(
                    decision=decision,
                    unit=restored,
                    archive_reference=restore_reference,
                )
            raise ContextTransitionError(
                f"unsupported lifecycle action '{decision.action}'"
            )

    async def mark_accessed(
        self,
        context_ids: Sequence[UUID],
    ) -> tuple[ContextUnit, ...]:
        """Persist use metadata so future scheduling can prefer useful Context."""

        async with self._lock:
            touched: list[ContextUnit] = []
            for context_id in dict.fromkeys(context_ids):
                unit = await self._store.load(context_id)
                if unit is None:
                    continue
                metadata = _evolve_metadata(
                    unit.metadata,
                    access_count=unit.metadata.access_count + 1,
                    last_accessed_at=utc_now(),
                )
                updated = _evolve_unit(unit, metadata=metadata)
                await self._store.save(
                    updated,
                    expected_revision=unit.revision,
                )
                touched.append(updated)
            return tuple(touched)

    async def _restore(
        self,
        reference: ContextArchiveReference,
    ) -> ContextUnit:
        current = await self._store.load(reference.context_id)
        if current is not None and not (
            current.lifecycle_state is ContextLifecycleState.COMPRESSED
            and current.recovery_reference == reference
        ):
            raise ContextRecoveryError(
                "restoring this reference would overwrite resident context"
            )
        archived = await self._archive.restore(reference)
        if archived.lifecycle_state is not ContextLifecycleState.ARCHIVED:
            raise ContextRecoveryError("archive returned a non-archived context unit")
        if archived.context_id != reference.context_id:
            raise ContextRecoveryError(
                "archive returned a context unit for a different reference"
            )
        values = archived.model_dump(mode="python")
        values["lifecycle_state"] = ContextLifecycleState.ACTIVE
        values["revision"] = (
            current.revision + 1
            if current is not None
            else archived.revision + 1
        )
        values["metadata"] = _evolve_metadata(archived.metadata)
        restored = ContextUnit.model_validate(values)
        await self._store.save(
            restored,
            expected_revision=(
                current.revision if current is not None else archived.revision
            ),
        )
        return restored

    async def active_for_run(self, run_id: UUID) -> tuple[ContextUnit, ...]:
        return tuple(
            unit
            for unit in await self._store.list_for_run(run_id)
            if unit.lifecycle_state is ContextLifecycleState.ACTIVE
        )

    async def resident_for_run(self, run_id: UUID) -> tuple[ContextUnit, ...]:
        return await self._store.list_for_run(run_id)

    async def _require(self, context_id: UUID) -> ContextUnit:
        unit = await self._store.load(context_id)
        if unit is None:
            raise ContextNotFoundError(f"context '{context_id}' was not found")
        return unit

    @staticmethod
    def _ensure_evictable(unit: ContextUnit) -> None:
        if unit.residency_policy is ResidencyPolicy.PINNED:
            raise ContextTransitionError("pinned context cannot leave active residency")

    @staticmethod
    def _require_decision_revision(
        decision: ContextLifecycleDecision,
        unit: ContextUnit,
    ) -> None:
        if (
            decision.source_revision is not None
            and decision.source_revision != unit.revision
        ):
            raise ContextSnapshotConflictError(
                "context lifecycle decision is based on a stale snapshot"
            )


class ContextCompressionCommitter:
    """Trusted Context commit boundary used by a governed compression Apply."""

    module_id = "context.compression_committer"

    def __init__(self, *, store: ContextStore, archive: ContextArchive) -> None:
        self._store = store
        self._archive = archive

    async def commit(
        self,
        source: ContextUnit,
        effect: ContextCompressionEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> ContextUnit:
        current = await self._store.load(source.context_id)
        if current is not None and current.last_effect_fingerprint == effect_fingerprint:
            return current
        if current != source:
            raise ContextSnapshotConflictError(
                "compression commit source revision is no longer authoritative"
            )
        archive_reference = ContextArchiveReference(
            archive_id=uuid5(source.context_id, effect_fingerprint),
            context_id=source.context_id,
        )
        transactional_archive = (
            self._archive
            if isinstance(self._archive, ContextCompressionTransaction)
            else None
        )
        if transactional_archive is not None and (
            permit is None or target is None or subject_fingerprint is None
        ):
            raise ContextSnapshotConflictError(
                "transactional compression requires a Runtime Permit"
            )
        reference = (
            await transactional_archive.stage_compression(
                source,
                reference=archive_reference,
                effect_fingerprint=effect_fingerprint,
                source_fingerprint=effect.source_snapshot_fingerprint,
            )
            if transactional_archive is not None
            else await self._archive.archive(source, reference=archive_reference)
        )
        if reference.context_id != source.context_id:
            await self._archive.discard(reference)
            raise ContextRecoveryError("archive returned a different Context identity")
        metadata = _evolve_metadata(
            source.metadata,
            estimated_tokens=effect.estimated_tokens,
        )
        compressed = _evolve_unit(
            source,
            content=effect.content,
            metadata=metadata,
            lifecycle_state=ContextLifecycleState.COMPRESSED,
            core_conclusions=effect.core_conclusions,
            recovery_reference=reference,
            last_effect_fingerprint=effect_fingerprint,
        )
        if transactional_archive is not None:
            assert permit is not None
            assert target is not None
            assert subject_fingerprint is not None
            return await transactional_archive.commit_compression(
                source,
                compressed,
                reference=reference,
                effect_fingerprint=effect_fingerprint,
                source_fingerprint=effect.source_snapshot_fingerprint,
                permit=permit,
                target=target,
                subject_fingerprint=subject_fingerprint,
            )
        try:
            await self._store.save(compressed, expected_revision=source.revision)
        except Exception:
            await self._archive.discard(reference)
            raise
        readback = await self._store.load(source.context_id)
        if readback != compressed:
            raise ContextSnapshotConflictError(
                "compression authoritative commit failed read-back"
            )
        return readback

    async def load_effect(
        self,
        context_id: UUID,
        effect_fingerprint: str,
        *,
        source_fingerprint: str | None = None,
    ) -> ContextUnit | None:
        unit = await self._store.load(context_id)
        if unit is None or unit.last_effect_fingerprint != effect_fingerprint:
            return None
        if isinstance(self._archive, ContextCompressionTransaction):
            if source_fingerprint is None or not await self._archive.verify_compression(
                unit,
                effect_fingerprint=effect_fingerprint,
                source_fingerprint=source_fingerprint,
            ):
                return None
        return unit


class DeterministicContextPressureMonitor:
    """Measure resident token pressure without selecting lifecycle actions."""

    module_id = "context.pressure_monitor.deterministic"

    def __init__(self, *, max_resident_tokens: int | None = None) -> None:
        if max_resident_tokens is not None and max_resident_tokens < 1:
            raise ValueError("resident Context budget must be positive")
        self._max_resident_tokens = max_resident_tokens

    def measure(
        self,
        requirement: ContextRequirement,
        resident: Sequence[ContextUnit],
    ) -> ContextPressure:
        maximum = (
            requirement.max_tokens
            if self._max_resident_tokens is None
            else min(requirement.max_tokens, self._max_resident_tokens)
        )
        tokens = sum(unit.metadata.estimated_tokens for unit in resident)
        return ContextPressure(
            run_id=requirement.run_id,
            resident_units=len(resident),
            active_units=sum(
                unit.lifecycle_state is ContextLifecycleState.ACTIVE
                for unit in resident
            ),
            compressed_units=sum(
                unit.lifecycle_state is ContextLifecycleState.COMPRESSED
                for unit in resident
            ),
            resident_tokens=tokens,
            max_resident_tokens=maximum,
            pressure_ratio=tokens / maximum,
        )


class DeterministicContextLifecyclePolicy:
    """Semantic GC policy driven by pressure, value, use, and residency."""

    module_id = "context.lifecycle_policy.deterministic"

    def __init__(
        self,
        *,
        compression_trigger_ratio: float = 0.8,
        target_ratio: float = 0.6,
        estimated_compression_ratio: float = 0.35,
        max_compressed_units: int = 2,
        max_pressure_actions_per_pass: int = 1,
    ) -> None:
        if not 0.0 < target_ratio < compression_trigger_ratio <= 1.0:
            raise ValueError("Context lifecycle ratios are invalid")
        if not 0.0 < estimated_compression_ratio < 1.0:
            raise ValueError("estimated compression ratio must be between zero and one")
        if max_compressed_units < 0:
            raise ValueError("max compressed Context Units cannot be negative")
        if max_pressure_actions_per_pass < 1:
            raise ValueError("Context pressure actions per pass must be positive")
        self._compression_trigger_ratio = compression_trigger_ratio
        self._target_ratio = target_ratio
        self._estimated_compression_ratio = estimated_compression_ratio
        self._max_compressed_units = max_compressed_units
        self._max_pressure_actions_per_pass = max_pressure_actions_per_pass

    def decide(
        self,
        requirement: ContextRequirement,
        resident: Sequence[ContextUnit],
        pressure: ContextPressure,
    ) -> tuple[ContextLifecycleDecision, ...]:
        resident_by_id = {unit.context_id: unit for unit in resident}
        required = set(requirement.required_context_ids)
        missing = required.difference(resident_by_id)
        decisions = [
            ContextLifecycleDecision(
                action=ContextLifecycleAction.RESTORE,
                context_id=context_id,
                run_id=requirement.run_id,
                task_id=requirement.task_id,
                node_id=requirement.node_id,
                reason="required Context Unit is archived",
                pressure_ratio=pressure.pressure_ratio,
            )
            for context_id in sorted(missing, key=str)
        ]

        compressed = sorted(
            (
                unit
                for unit in resident
                if unit.lifecycle_state is ContextLifecycleState.COMPRESSED
                and unit.residency_policy is not ResidencyPolicy.PINNED
                and unit.context_id not in required
            ),
            key=self._eviction_key,
        )
        archive_count = max(0, len(compressed) - self._max_compressed_units)
        archived_ids: set[UUID] = set()
        for unit in compressed[
            : min(archive_count, self._max_pressure_actions_per_pass)
        ]:
            decisions.append(
                self._decision(
                    ContextLifecycleAction.ARCHIVE,
                    unit,
                    requirement,
                    pressure,
                    "compressed Context exceeded resident retention limit",
                )
            )
            archived_ids.add(unit.context_id)

        if archived_ids:
            return tuple(decisions)

        if pressure.pressure_ratio < self._compression_trigger_ratio:
            return tuple(decisions)

        projected_tokens = pressure.resident_tokens - sum(
            unit.metadata.estimated_tokens
            for unit in compressed
            if unit.context_id in archived_ids
        )
        target_tokens = int(
            pressure.max_resident_tokens * self._target_ratio
        )
        active = sorted(
            (
                unit
                for unit in resident
                if unit.lifecycle_state is ContextLifecycleState.ACTIVE
                and unit.residency_policy is not ResidencyPolicy.PINNED
                and unit.context_id not in required
            ),
            key=self._eviction_key,
        )
        for unit in active:
            if projected_tokens <= target_tokens:
                break
            decisions.append(
                self._decision(
                    ContextLifecycleAction.COMPRESS,
                    unit,
                    requirement,
                    pressure,
                    "resident token pressure crossed the compression threshold",
                )
            )
            estimated = max(
                1,
                int(
                    unit.metadata.estimated_tokens
                    * self._estimated_compression_ratio
                ),
            )
            projected_tokens -= unit.metadata.estimated_tokens - estimated
            if (
                sum(
                    item.action is not ContextLifecycleAction.RESTORE
                    for item in decisions
                )
                >= self._max_pressure_actions_per_pass
            ):
                break

        for unit in compressed[archive_count:]:
            if (
                sum(
                    item.action is not ContextLifecycleAction.RESTORE
                    for item in decisions
                )
                >= self._max_pressure_actions_per_pass
            ):
                break
            if projected_tokens <= target_tokens:
                break
            decisions.append(
                self._decision(
                    ContextLifecycleAction.ARCHIVE,
                    unit,
                    requirement,
                    pressure,
                    "compressed Context remained above the resident token target",
                )
            )
            projected_tokens -= unit.metadata.estimated_tokens
            if (
                sum(
                    item.action is not ContextLifecycleAction.RESTORE
                    for item in decisions
                )
                >= self._max_pressure_actions_per_pass
            ):
                break
        return tuple(decisions)

    @staticmethod
    def _decision(
        action: ContextLifecycleAction,
        unit: ContextUnit,
        requirement: ContextRequirement,
        pressure: ContextPressure,
        reason: str,
    ) -> ContextLifecycleDecision:
        return ContextLifecycleDecision(
            action=action,
            context_id=unit.context_id,
            run_id=requirement.run_id,
            task_id=requirement.task_id,
            node_id=requirement.node_id,
            source_revision=unit.revision,
            reason=reason,
            pressure_ratio=pressure.pressure_ratio,
        )

    @staticmethod
    def _eviction_key(unit: ContextUnit) -> tuple[Any, ...]:
        residency_order = {
            ResidencyPolicy.TRANSIENT: 0,
            ResidencyPolicy.SESSION: 1,
            ResidencyPolicy.PINNED: 2,
        }
        last_use = unit.metadata.last_accessed_at or unit.metadata.updated_at
        return (
            residency_order[unit.residency_policy],
            unit.metadata.importance,
            unit.metadata.access_count,
            last_use.timestamp(),
            -unit.metadata.estimated_tokens,
            str(unit.context_id),
        )


class DirectContextLifecycleExecutor:
    """Ungoverned mechanism adapter; applications may wrap this boundary."""

    module_id = "context.lifecycle_executor.direct"

    def __init__(self, manager: ContextLifecycleManager) -> None:
        self._manager = manager

    async def execute(
        self,
        decision: ContextLifecycleDecision,
    ) -> ContextLifecycleResult:
        return await self._manager.apply(decision)


class ContextLifecycleRuntime:
    """Close pressure -> policy -> apply, independent from Agent execution."""

    module_id = "context.lifecycle_runtime"

    def __init__(
        self,
        *,
        store: ContextStore,
        pressure_monitor: ContextPressureMonitoring,
        policy: ContextLifecyclePlanning,
        executor: ContextLifecycleActionExecutor,
        max_passes: int = 4,
    ) -> None:
        if max_passes < 1:
            raise ValueError("Context lifecycle passes must be positive")
        self._store = store
        self._pressure_monitor = pressure_monitor
        self._policy = policy
        self._executor = executor
        self._max_passes = max_passes
        self._lock = Lock()

    async def reconcile(
        self,
        requirement: ContextRequirement,
    ) -> tuple[
        ContextPressure,
        ContextPressure,
        tuple[ContextLifecycleResult, ...],
    ]:
        async with self._lock:
            resident = await self._store.list_for_run(requirement.run_id)
            pressure_before = self._pressure_monitor.measure(
                requirement,
                resident,
            )
            results: list[ContextLifecycleResult] = []
            for _ in range(self._max_passes):
                pressure = self._pressure_monitor.measure(requirement, resident)
                decisions = self._policy.decide(
                    requirement,
                    resident,
                    pressure,
                )
                if not decisions:
                    break
                for decision in decisions:
                    results.append(await self._executor.execute(decision))
                resident = await self._store.list_for_run(requirement.run_id)
            pressure_after = self._pressure_monitor.measure(requirement, resident)
            return pressure_before, pressure_after, tuple(results)


class ContextScheduler:
    module_id = "context.scheduler"

    _LAYER_ORDER = {
        ContextLayer.WORKING: 0,
        ContextLayer.TASK: 1,
        ContextLayer.SEMANTIC: 2,
    }

    def schedule(
        self,
        requirement: ContextRequirement,
        candidates: Sequence[ContextUnit],
    ) -> ContextSchedule:
        latest_by_id: dict[UUID, ContextUnit] = {}
        for unit in candidates:
            current = latest_by_id.get(unit.context_id)
            if current is None or self._version_key(unit) > self._version_key(current):
                latest_by_id[unit.context_id] = unit

        eligible_by_id: dict[UUID, ContextUnit] = {}
        omitted_ids: list[UUID] = []
        for unit in latest_by_id.values():
            if unit.metadata.run_id != requirement.run_id:
                continue
            if unit.lifecycle_state is ContextLifecycleState.ARCHIVED:
                omitted_ids.append(unit.context_id)
                continue
            if unit.metadata.layer not in requirement.layers:
                omitted_ids.append(unit.context_id)
                continue
            if (
                requirement.task_id is not None
                and unit.metadata.task_id not in {None, requirement.task_id}
            ):
                continue
            eligible_by_id[unit.context_id] = unit

        missing_required = set(requirement.required_context_ids) - set(
            eligible_by_id
        )
        if missing_required:
            missing = ", ".join(sorted(str(item) for item in missing_required))
            raise ContextNotFoundError(
                f"required context units are unavailable: {missing}"
            )

        mandatory_ids = set(requirement.required_context_ids)
        mandatory_ids.update(
            unit.context_id
            for unit in eligible_by_id.values()
            if unit.residency_policy is ResidencyPolicy.PINNED
        )
        ordered = sorted(
            eligible_by_id.values(),
            key=lambda unit: self._rank(unit, requirement, mandatory_ids),
        )
        mandatory = [unit for unit in ordered if unit.context_id in mandatory_ids]
        mandatory_tokens = sum(
            unit.metadata.estimated_tokens for unit in mandatory
        )
        if (
            len(mandatory) > requirement.max_units
            or mandatory_tokens > requirement.max_tokens
        ):
            raise ContextBudgetExceededError(
                "mandatory context exceeds the assembly budget"
            )

        selected = list(mandatory)
        used_tokens = mandatory_tokens
        selected_ids = {unit.context_id for unit in selected}
        for unit in ordered:
            if unit.context_id in selected_ids:
                continue
            if len(selected) >= requirement.max_units:
                omitted_ids.append(unit.context_id)
                continue
            next_tokens = used_tokens + unit.metadata.estimated_tokens
            if next_tokens > requirement.max_tokens:
                omitted_ids.append(unit.context_id)
                continue
            selected.append(unit)
            selected_ids.add(unit.context_id)
            used_tokens = next_tokens

        selected.sort(
            key=lambda unit: self._rank(unit, requirement, mandatory_ids)
        )
        return ContextSchedule(
            requirement=requirement,
            selected=tuple(selected),
            omitted_context_ids=tuple(
                context_id
                for context_id in dict.fromkeys(omitted_ids)
                if context_id not in selected_ids
            ),
            used_tokens=used_tokens,
        )

    @staticmethod
    def _version_key(unit: ContextUnit) -> tuple[int, float, int]:
        lifecycle_order = {
            ContextLifecycleState.ACTIVE: 0,
            ContextLifecycleState.COMPRESSED: 1,
            ContextLifecycleState.ARCHIVED: 2,
        }
        return (
            unit.revision,
            unit.metadata.updated_at.timestamp(),
            lifecycle_order[unit.lifecycle_state],
        )

    def _rank(
        self,
        unit: ContextUnit,
        requirement: ContextRequirement,
        mandatory_ids: set[UUID],
    ) -> tuple[Any, ...]:
        matching_tags = len(
            set(unit.metadata.tags).intersection(requirement.preferred_tags)
        )
        return (
            self._LAYER_ORDER[unit.metadata.layer],
            0 if unit.context_id in mandatory_ids else 1,
            0 if unit.metadata.node_id == requirement.node_id else 1,
            -matching_tags,
            -unit.metadata.importance,
            -unit.metadata.updated_at.timestamp(),
            str(unit.context_id),
        )


class ContextAssembler:
    module_id = "context.assembler"

    def assemble(self, schedule: ContextSchedule) -> ContextAssembly:
        working = tuple(
            unit
            for unit in schedule.selected
            if unit.metadata.layer is ContextLayer.WORKING
        )
        task = tuple(
            unit
            for unit in schedule.selected
            if unit.metadata.layer is ContextLayer.TASK
        )
        semantic = tuple(
            unit
            for unit in schedule.selected
            if unit.metadata.layer is ContextLayer.SEMANTIC
        )
        units = (*working, *task, *semantic)
        return ContextAssembly(
            requirement=schedule.requirement,
            units=units,
            working_context=working,
            task_context=task,
            semantic_context=semantic,
            omitted_context_ids=schedule.omitted_context_ids,
            used_tokens=schedule.used_tokens,
        )
