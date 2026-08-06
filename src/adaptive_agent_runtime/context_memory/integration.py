"""Adapters connecting Context-Memory to stable Core/Orchestration models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import NAMESPACE_URL, UUID, uuid5

from adaptive_agent_runtime import AgentState, Observation, RuntimeModule
from adaptive_agent_runtime.context_memory.context_models import (
    ContextAssembly,
    ContextLayer,
    ContextLifecycleState,
    ContextLifecycleResult,
    ContextMetadata,
    ContextPreparation,
    ContextPressure,
    ContextRequirement,
    ContextSource,
    ContextUnit,
    ResidencyPolicy,
)
from adaptive_agent_runtime.context_memory.contracts import (
    ContextAssemblyBuilder,
    ContextLifecycleManagement,
    ContextScheduling,
    ContextStore,
    MemoryRecall,
)
from adaptive_agent_runtime.context_memory.json_types import (
    ImmutableJsonValue,
    estimate_tokens,
    utc_now,
)
from adaptive_agent_runtime.context_memory.memory_models import (
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryRecallQuery,
    MemoryUnit,
)
from adaptive_agent_runtime.orchestration.models import TaskNode


_OBSERVATION_CONTEXT_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/context/observation",
)
_MEMORY_CONTEXT_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/context/memory-projection",
)


@runtime_checkable
class TaskContextRequirementProvider(RuntimeModule, Protocol):
    def requirement_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> ContextRequirement: ...


class DefaultTaskContextRequirementProvider:
    module_id = "context.task_requirement.default"

    def __init__(self, *, max_units: int = 16, max_tokens: int = 4096) -> None:
        self._max_units = max_units
        self._max_tokens = max_tokens

    def requirement_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> ContextRequirement:
        return ContextRequirement(
            run_id=state.run_id,
            task_id=state.task.task_id,
            node_id=node.node_id,
            goal=node.goal,
            preferred_tags=(f"node:{node.node_id}",),
            max_units=self._max_units,
            max_tokens=self._max_tokens,
        )


class ObservationContextAdapter:
    module_id = "context.adapter.observation"

    def convert(
        self,
        observation: Observation,
        state: AgentState,
        *,
        node: TaskNode | None = None,
    ) -> ContextUnit:
        content = observation.model_dump(mode="json")
        tags = ["observation", "success" if observation.succeeded else "failure"]
        if node is not None:
            tags.append(f"node:{node.node_id}")
        return ContextUnit(
            context_id=uuid5(
                _OBSERVATION_CONTEXT_NAMESPACE,
                f"{state.run_id}:{observation.action_id}",
            ),
            content=content,
            metadata=ContextMetadata(
                source=ContextSource.OBSERVATION,
                layer=ContextLayer.TASK,
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id if node is not None else None,
                source_reference=f"action:{observation.action_id}",
                tags=tuple(tags),
                importance=0.8 if observation.succeeded else 0.9,
                estimated_tokens=estimate_tokens(content),
                created_at=state.updated_at,
                updated_at=state.updated_at,
            ),
            lifecycle_state=ContextLifecycleState.ACTIVE,
            residency_policy=ResidencyPolicy.SESSION,
        )


class MemoryContextAdapter:
    module_id = "context.adapter.memory_recall"

    def convert(
        self,
        memory: MemoryUnit,
        state: AgentState,
        *,
        node: TaskNode,
    ) -> ContextUnit:
        content = {
            "memory_key": memory.memory_key,
            "content": memory.model_dump(mode="json")["content"],
            "confidence": memory.confidence,
            "condition": memory.condition.model_dump(mode="json"),
            "status": memory.status.value,
            "conflicts": [
                conflict.model_dump(mode="json") for conflict in memory.conflicts
            ],
        }
        tags = tuple(
            dict.fromkeys(("memory", *memory.condition.required_tags))
        )
        return ContextUnit(
            context_id=uuid5(
                _MEMORY_CONTEXT_NAMESPACE,
                (
                    f"{state.run_id}:{state.task.task_id}:{node.node_id}:"
                    f"{memory.memory_id}:{memory.revision}"
                ),
            ),
            content=content,
            metadata=ContextMetadata(
                source=ContextSource.MEMORY_RECALL,
                layer=ContextLayer.SEMANTIC,
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                source_reference=f"memory:{memory.memory_id}",
                tags=tags,
                importance=memory.confidence,
                estimated_tokens=estimate_tokens(content),
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            ),
            residency_policy=ResidencyPolicy.TRANSIENT,
        )


class ExplicitMemoryCandidateFactory:
    """Wrap explicit behavioral knowledge as a Candidate; never auto-generates it."""

    module_id = "memory.candidate_factory.explicit"

    def from_context(
        self,
        context: ContextUnit,
        *,
        memory_key: str,
        content: ImmutableJsonValue,
        condition: MemoryCondition,
        confidence: float,
        evolution: MemoryEvolutionType,
        note: str,
        target_memory_id: UUID | None = None,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            memory_key=memory_key,
            content=content,
            condition=condition,
            evidence=(
                MemoryEvidence(
                    source_context_id=context.context_id,
                    source_reference=context.metadata.source_reference,
                    note=note,
                ),
            ),
            confidence=confidence,
            evolution=evolution,
            target_memory_id=target_memory_id,
        )


class ContextMemoryCoordinator:
    """Explicit integration seam; it is not wired into the Core Event Loop."""

    module_id = "context_memory.coordinator"

    def __init__(
        self,
        *,
        context_store: ContextStore,
        scheduler: ContextScheduling,
        assembler: ContextAssemblyBuilder,
        requirements: TaskContextRequirementProvider,
        memory_recall: MemoryRecall,
        lifecycle: ContextLifecycleManagement | None = None,
        observation_adapter: ObservationContextAdapter | None = None,
        memory_adapter: MemoryContextAdapter | None = None,
    ) -> None:
        self._context_store = context_store
        self._scheduler = scheduler
        self._assembler = assembler
        self._requirements = requirements
        self._memory_recall = memory_recall
        self._lifecycle = lifecycle
        self._observation_adapter = observation_adapter or ObservationContextAdapter()
        self._memory_adapter = memory_adapter or MemoryContextAdapter()

    async def record_observation(
        self,
        observation: Observation,
        state: AgentState,
        *,
        node: TaskNode | None = None,
    ) -> ContextUnit:
        unit = self._observation_adapter.convert(
            observation,
            state,
            node=node,
        )
        await self._context_store.save(unit, expected_revision=None)
        return unit

    async def prepare_next_context(
        self,
        state: AgentState,
        node: TaskNode,
        *,
        memory_query: MemoryRecallQuery,
    ) -> ContextAssembly:
        preparation = await self.prepare(
            state,
            node,
            memory_query=memory_query,
        )
        return preparation.assembly

    async def prepare(
        self,
        state: AgentState,
        node: TaskNode,
        *,
        memory_query: MemoryRecallQuery,
    ) -> ContextPreparation:
        """Reconcile lifecycle, recall Memory, assemble, and persist usage."""

        requirement = self._requirements.requirement_for(node, state)
        lifecycle = self._lifecycle
        lifecycle_results: tuple[ContextLifecycleResult, ...]
        if lifecycle is None:
            resident = await self._context_store.list_for_run(state.run_id)
            pressure_before = self._pressure(requirement, resident)
            pressure_after = pressure_before
            lifecycle_results = ()
        else:
            (
                pressure_before,
                pressure_after,
                lifecycle_results,
            ) = await lifecycle.reconcile(requirement)
        resident = await self._context_store.list_for_run(state.run_id)
        memories = await self._memory_recall.recall(memory_query)
        recalled = tuple(
            self._memory_adapter.convert(memory, state, node=node)
            for memory in memories
        )
        schedule = self._scheduler.schedule(
            requirement,
            (*resident, *recalled),
        )
        selected_resident_ids = {
            unit.context_id
            for unit in schedule.selected
            if any(item.context_id == unit.context_id for item in resident)
        }
        await self._mark_accessed(selected_resident_ids)
        if selected_resident_ids:
            resident = await self._context_store.list_for_run(state.run_id)
            schedule = self._scheduler.schedule(
                requirement,
                (*resident, *recalled),
            )
        return ContextPreparation(
            pressure_before=pressure_before,
            pressure_after=pressure_after,
            lifecycle_results=lifecycle_results,
            assembly=self._assembler.assemble(schedule),
        )

    async def _mark_accessed(self, context_ids: set[UUID]) -> None:
        for context_id in sorted(context_ids, key=str):
            unit = await self._context_store.load(context_id)
            if unit is None:
                continue
            metadata_values = unit.metadata.model_dump(mode="python")
            now = utc_now()
            metadata_values.update(
                access_count=unit.metadata.access_count + 1,
                last_accessed_at=now,
                updated_at=now,
            )
            values = unit.model_dump(mode="python")
            values.update(
                metadata=ContextMetadata.model_validate(metadata_values),
                revision=unit.revision + 1,
            )
            updated = ContextUnit.model_validate(values)
            await self._context_store.save(
                updated,
                expected_revision=unit.revision,
            )

    @staticmethod
    def _pressure(
        requirement: ContextRequirement,
        resident: tuple[ContextUnit, ...],
    ) -> ContextPressure:
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
            max_resident_tokens=requirement.max_tokens,
            pressure_ratio=tokens / requirement.max_tokens,
        )
