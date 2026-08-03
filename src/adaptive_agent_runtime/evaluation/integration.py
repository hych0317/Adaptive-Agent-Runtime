"""Read-only adapters from existing Runtime snapshots into Evaluation facts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence, cast
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.context_memory.context_models import (
    ContextAssembly,
    ContextLifecycleResult,
    ContextUnit,
)
from adaptive_agent_runtime.context_memory.memory_models import MemoryUpdateResult
from adaptive_agent_runtime.core.models import (
    AgentState,
    CoreEventKind,
    RunResult,
    TraceEntry,
)
from adaptive_agent_runtime.evaluation.contracts import (
    ComponentEvaluator,
    OutcomeEvaluator,
    TraceCollector,
    TrajectoryEvaluator,
)
from adaptive_agent_runtime.evaluation.models import (
    EvaluationComponent,
    EvaluationCorrelation,
    EvaluationCriteria,
    EvaluationFact,
    EvaluationReport,
    EvaluationStateSnapshot,
    EvaluationSubject,
    ExecutionResultSnapshot,
    ExecutionStatus,
    TraceBatch,
    TraceCategory,
    TraceCompleteness,
    TraceCoverage,
    canonical_evaluation_json,
    stable_evaluation_id,
)
from adaptive_agent_runtime.evaluation.trace_collector import (
    TraceCorrelationError,
)
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.recovery import RecoveryRecord
from adaptive_agent_runtime.tool_ecosystem.models import ToolTraceEntry


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _uuid(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _nested_mapping(
    value: Mapping[str, Any] | None,
    key: str,
) -> Mapping[str, Any] | None:
    return _mapping(value.get(key)) if value is not None else None


class RuntimeTraceAdapter:
    """Normalize Core trace DTOs and extract explicit Orchestration links."""

    module_id = "evaluation.adapter.runtime_trace"

    def adapt(
        self,
        entries: Sequence[TraceEntry],
        *,
        run_id: UUID,
        task_id: UUID,
    ) -> TraceBatch:
        facts: list[EvaluationFact] = []
        selected_entries = tuple(
            entry for entry in entries if entry.event.run_id == run_id
        )
        if not selected_entries:
            return TraceBatch(
                coverage=(
                    TraceCoverage(
                        run_id=run_id,
                        component=EvaluationComponent.RUNTIME,
                        completeness=TraceCompleteness.MISSING,
                        diagnostics=("no Runtime trace entries were supplied",),
                    ),
                )
            )

        observed_kinds: set[str] = set()
        orchestration_observed = False

        for entry in selected_entries:
            event = entry.event
            observed_kinds.add(event.kind)
            payload = event.model_dump(mode="json")["payload"]
            correlation = self._correlation(
                payload,
                kind=event.kind,
                run_id=event.run_id,
                task_id=task_id,
            )
            component, category = self._classification(
                event.kind,
                correlation,
            )
            if component is EvaluationComponent.ORCHESTRATION:
                orchestration_observed = True
            facts.append(
                EvaluationFact(
                    fact_id=event.event_id,
                    component=component,
                    category=category,
                    kind=event.kind,
                    source=event.source,
                    occurred_at=event.occurred_at,
                    correlation=correlation,
                    source_scope=f"runtime:{event.run_id}",
                    source_sequence=entry.sequence,
                    source_record_id=str(event.event_id),
                    payload=payload,
                )
            )
            mutations = self._graph_mutations(payload)
            if mutations:
                orchestration_observed = True
                facts.append(
                    EvaluationFact(
                        fact_id=stable_evaluation_id(
                            "runtime-graph-mutation",
                            event.event_id,
                        ),
                        component=EvaluationComponent.ORCHESTRATION,
                        category=TraceCategory.TASK_GRAPH,
                        kind="orchestration.graph_mutated",
                        source=event.source,
                        occurred_at=event.occurred_at,
                        correlation=correlation,
                        source_scope=f"runtime:{event.run_id}",
                        source_sequence=entry.sequence,
                        source_record_id=f"{event.event_id}:mutations",
                        payload={"mutations": list(mutations)},
                    )
                )

        diagnostics = self._runtime_diagnostics(selected_entries, observed_kinds)
        coverage: list[TraceCoverage] = [
            TraceCoverage(
                run_id=run_id,
                component=EvaluationComponent.RUNTIME,
                completeness=(
                    TraceCompleteness.COMPLETE
                    if not diagnostics
                    else TraceCompleteness.PARTIAL
                ),
                diagnostics=diagnostics,
            )
        ]
        if orchestration_observed:
            coverage.append(
                TraceCoverage(
                    run_id=run_id,
                    component=EvaluationComponent.ORCHESTRATION,
                    completeness=TraceCompleteness.PARTIAL,
                    diagnostics=(
                        "Core events expose node actions but not full graph history",
                    ),
                )
            )
        return TraceBatch(facts=tuple(facts), coverage=tuple(coverage))

    @staticmethod
    def _runtime_diagnostics(
        entries: Sequence[TraceEntry],
        observed_kinds: set[str],
    ) -> tuple[str, ...]:
        diagnostics: list[str] = []
        sequences = tuple(sorted(entry.sequence for entry in entries))
        if len(set(sequences)) != len(sequences):
            diagnostics.append("Runtime trace contains duplicate sequence values")
        if sequences != tuple(range(1, len(entries) + 1)):
            diagnostics.append("Runtime trace sequence is not contiguous from 1")

        start_kind = CoreEventKind.RUNTIME_STARTED.value
        terminal_kinds = {
            CoreEventKind.RUNTIME_COMPLETED.value,
            CoreEventKind.RUNTIME_FAILED.value,
        }
        starts = tuple(
            entry for entry in entries if entry.event.kind == start_kind
        )
        terminals = tuple(
            entry for entry in entries if entry.event.kind in terminal_kinds
        )
        if start_kind not in observed_kinds:
            diagnostics.append("Runtime trace lacks a start event")
        elif len(starts) != 1:
            diagnostics.append("Runtime trace must contain one start event")
        elif starts[0].sequence != min(sequences):
            diagnostics.append("Runtime start event is not first")
        if not terminal_kinds.intersection(observed_kinds):
            diagnostics.append("Runtime trace lacks a terminal event")
        elif len(terminals) != 1:
            diagnostics.append("Runtime trace must contain one terminal event")
        elif terminals[0].sequence != max(sequences):
            diagnostics.append("Runtime terminal event is not last")
        return tuple(diagnostics)

    @staticmethod
    def _correlation(
        payload: Mapping[str, Any],
        *,
        kind: str,
        run_id: UUID,
        task_id: UUID,
    ) -> EvaluationCorrelation:
        state = _nested_mapping(payload, "state")
        state_task = _nested_mapping(state, "task")
        payload_task_id = _uuid(
            state_task.get("task_id") if state_task is not None else None
        )
        if payload_task_id not in {None, task_id}:
            raise TraceCorrelationError("Runtime trace task_id is inconsistent")

        plan = _nested_mapping(payload, "plan")
        action = _nested_mapping(payload, "action") or _nested_mapping(
            plan,
            "action",
        )
        observation = _nested_mapping(payload, "observation")
        if kind == CoreEventKind.STATE_UPDATED.value:
            state_plan = _nested_mapping(state, "last_plan")
            if (
                state_plan is not None
                and state_plan.get("decision") == "execute"
            ):
                plan = state_plan
                action = _nested_mapping(plan, "action")
                observation = _nested_mapping(state, "last_observation")
        action_from_request = _uuid(
            action.get("action_id") if action is not None else None
        )
        action_from_observation = _uuid(
            observation.get("action_id") if observation is not None else None
        )
        if (
            action_from_request is not None
            and action_from_observation is not None
            and action_from_request != action_from_observation
        ):
            raise TraceCorrelationError(
                "Runtime action and observation use different action_id values"
            )
        action_id = action_from_request or action_from_observation

        arguments = _nested_mapping(action, "arguments")
        node = _nested_mapping(arguments, "node")
        graph_id = _uuid(
            arguments.get("graph_id") if arguments is not None else None
        )
        node_from_action = _uuid(node.get("node_id") if node is not None else None)

        metadata = _nested_mapping(observation, "metadata")
        orchestration = _nested_mapping(metadata, "orchestration")
        node_from_observation = _uuid(
            orchestration.get("node_id")
            if orchestration is not None
            else None
        )
        if (
            node_from_action is not None
            and node_from_observation is not None
            and node_from_action != node_from_observation
        ):
            raise TraceCorrelationError(
                "Runtime action and observation use different node_id values"
            )
        node_id = node_from_action or node_from_observation

        tool = _nested_mapping(metadata, "tool")
        invocation_id = _uuid(
            tool.get("invocation_id") if tool is not None else None
        )
        tool_correlation = _nested_mapping(tool, "correlation")
        tool_action_id = _uuid(
            tool_correlation.get("action_id")
            if tool_correlation is not None
            else None
        )
        if action_id is not None and tool_action_id not in {None, action_id}:
            raise TraceCorrelationError(
                "Tool metadata action_id conflicts with Runtime action_id"
            )
        action_id = action_id or tool_action_id

        return EvaluationCorrelation(
            run_id=run_id,
            task_id=task_id,
            graph_id=graph_id,
            node_id=node_id,
            action_id=action_id,
            invocation_id=invocation_id,
        )

    @staticmethod
    def _classification(
        kind: str,
        correlation: EvaluationCorrelation,
    ) -> tuple[EvaluationComponent, TraceCategory]:
        if kind == CoreEventKind.RUNTIME_STARTED.value:
            return EvaluationComponent.RUNTIME, TraceCategory.TASK
        if kind in {
            CoreEventKind.RUNTIME_COMPLETED.value,
            CoreEventKind.RUNTIME_FAILED.value,
        }:
            return EvaluationComponent.RUNTIME, TraceCategory.FINAL_RESULT
        if correlation.graph_id is not None and kind == CoreEventKind.PLAN_CREATED.value:
            return EvaluationComponent.ORCHESTRATION, TraceCategory.TASK_GRAPH
        if correlation.node_id is not None:
            return EvaluationComponent.ORCHESTRATION, TraceCategory.NODE_EXECUTION
        return EvaluationComponent.RUNTIME, TraceCategory.RUNTIME

    @staticmethod
    def _graph_mutations(
        payload: Mapping[str, Any],
    ) -> tuple[JsonValue, ...]:
        observation = _nested_mapping(payload, "observation")
        metadata = _nested_mapping(observation, "metadata")
        orchestration = _nested_mapping(metadata, "orchestration")
        raw = orchestration.get("mutations") if orchestration is not None else None
        if isinstance(raw, (list, tuple)):
            return tuple(cast(JsonValue, item) for item in raw)
        return ()


class ToolTraceAdapter:
    module_id = "evaluation.adapter.tool_trace"

    def adapt(
        self,
        entries: Sequence[ToolTraceEntry],
    ) -> TraceBatch:
        facts: list[EvaluationFact] = []
        runs: set[UUID] = set()
        invocation_identity: dict[UUID, tuple[object, ...]] = {}
        invocation_sequences: defaultdict[UUID, set[int]] = defaultdict(set)
        for entry in entries:
            event = entry.event
            correlation = event.correlation
            if correlation.run_id is None:
                raise TraceCorrelationError(
                    f"Tool trace event '{event.event_id}' has no run_id"
                )
            identity = (
                correlation.run_id,
                correlation.task_id,
                event.requirement_id,
                event.capability_id,
                event.provider_id,
            )
            previous_identity = invocation_identity.setdefault(
                event.invocation_id,
                identity,
            )
            if previous_identity != identity:
                raise TraceCorrelationError(
                    f"Tool invocation '{event.invocation_id}' changes identity"
                )
            sequences = invocation_sequences[event.invocation_id]
            if entry.sequence in sequences:
                raise TraceCorrelationError(
                    f"Tool invocation '{event.invocation_id}' repeats sequence "
                    f"'{entry.sequence}'"
                )
            sequences.add(entry.sequence)
            runs.add(correlation.run_id)
            payload = {
                "requirement_id": str(event.requirement_id),
                "capability_id": event.capability_id,
                "provider_id": event.provider_id,
                "attempt_number": event.attempt_number,
                "data": event.model_dump(mode="json")["payload"],
            }
            facts.append(
                EvaluationFact(
                    fact_id=event.event_id,
                    component=EvaluationComponent.TOOL,
                    category=TraceCategory.TOOL_CALL,
                    kind=event.kind.value,
                    source="tool_ecosystem",
                    occurred_at=event.occurred_at,
                    correlation=EvaluationCorrelation(
                        run_id=correlation.run_id,
                        task_id=correlation.task_id,
                        node_id=correlation.node_id,
                        action_id=correlation.action_id,
                        invocation_id=event.invocation_id,
                    ),
                    source_scope=f"tool-invocation:{event.invocation_id}",
                    source_sequence=entry.sequence,
                    source_record_id=str(event.event_id),
                    payload=payload,
                )
            )
        coverage = tuple(
            TraceCoverage(
                run_id=run_id,
                component=EvaluationComponent.TOOL,
                completeness=TraceCompleteness.PARTIAL,
                diagnostics=(
                    "provided invocation traces may not enumerate every run call",
                ),
            )
            for run_id in sorted(runs, key=str)
        )
        return TraceBatch(facts=tuple(facts), coverage=coverage)


class GraphSnapshotAdapter:
    module_id = "evaluation.adapter.graph_snapshot"

    def adapt(
        self,
        graph: DynamicTaskGraph,
        *,
        run_id: UUID,
        task_id: UUID,
        observed_at: datetime,
    ) -> TraceBatch:
        fact = EvaluationFact(
            fact_id=stable_evaluation_id(
                "graph-snapshot",
                run_id,
                graph.graph_id,
                graph.version,
                observed_at.isoformat(),
                canonical_evaluation_json(graph),
            ),
            component=EvaluationComponent.ORCHESTRATION,
            category=TraceCategory.TASK_GRAPH,
            kind="orchestration.graph_snapshot",
            source=self.module_id,
            occurred_at=observed_at,
            correlation=EvaluationCorrelation(
                run_id=run_id,
                task_id=task_id,
                graph_id=graph.graph_id,
            ),
            source_scope=f"graph:{graph.graph_id}",
            source_sequence=graph.version,
            source_record_id=(
                f"{graph.graph_id}:{graph.version}:{observed_at.isoformat()}"
            ),
            payload={"graph": graph.model_dump(mode="json")},
        )
        coverage = TraceCoverage(
            run_id=run_id,
            component=EvaluationComponent.ORCHESTRATION,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=("graph snapshot does not provide complete version history",),
        )
        return TraceBatch(facts=(fact,), coverage=(coverage,))


class RecoveryTraceAdapter:
    """Project Failure-driven Replanning records into Runtime-native facts."""

    module_id = "evaluation.adapter.recovery"

    def adapt(
        self,
        records: Sequence[RecoveryRecord],
        *,
        run_id: UUID,
        task_id: UUID,
        graph_id: UUID,
    ) -> TraceBatch:
        facts = tuple(
            EvaluationFact(
                fact_id=stable_evaluation_id(
                    "recovery-record",
                    run_id,
                    record.plan.plan_id,
                    record.graph_version_before,
                    record.graph_version_after,
                ),
                component=EvaluationComponent.ORCHESTRATION,
                category=TraceCategory.TASK_GRAPH,
                kind=(
                    "orchestration.recovery_aborted"
                    if record.plan.aborts
                    else "orchestration.recovery_applied"
                ),
                source=self.module_id,
                occurred_at=record.recorded_at,
                correlation=EvaluationCorrelation(
                    run_id=run_id,
                    task_id=task_id,
                    graph_id=graph_id,
                    node_id=record.plan.analysis.node_id,
                    action_id=record.plan.analysis.action_id,
                ),
                source_scope=f"recovery:{record.plan.plan_id}",
                source_sequence=record.plan.attempt_number,
                source_record_id=str(record.plan.plan_id),
                payload={"recovery": record.model_dump(mode="json")},
            )
            for record in records
        )
        coverage = TraceCoverage(
            run_id=run_id,
            component=EvaluationComponent.ORCHESTRATION,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=(
                "recovery records cover only Failure-driven Replanning decisions",
            ),
        )
        return TraceBatch(facts=facts, coverage=(coverage,))


class ContextMemoryTraceAdapter:
    module_id = "evaluation.adapter.context_memory"

    def context_unit(
        self,
        unit: ContextUnit,
        *,
        action_id: UUID | None = None,
        kind: str | None = None,
    ) -> TraceBatch:
        metadata = unit.metadata
        fact = EvaluationFact(
            fact_id=stable_evaluation_id(
                "context-unit",
                metadata.run_id,
                unit.context_id,
                unit.revision,
                unit.lifecycle_state.value,
            ),
            component=EvaluationComponent.CONTEXT,
            category=TraceCategory.CONTEXT_CHANGE,
            kind=kind or f"context.{unit.lifecycle_state.value}",
            source=self.module_id,
            occurred_at=metadata.updated_at,
            correlation=EvaluationCorrelation(
                run_id=metadata.run_id,
                task_id=metadata.task_id,
                node_id=metadata.node_id,
                action_id=action_id,
                context_id=unit.context_id,
            ),
            source_scope=f"context:{unit.context_id}",
            source_sequence=unit.revision,
            source_record_id=f"{unit.context_id}:{unit.revision}",
            payload={
                "lifecycle_state": unit.lifecycle_state.value,
                "residency_policy": unit.residency_policy.value,
                "revision": unit.revision,
                "source": metadata.source.value,
                "layer": metadata.layer.value,
                "estimated_tokens": metadata.estimated_tokens,
                "tags": list(metadata.tags),
            },
        )
        coverage = TraceCoverage(
            run_id=metadata.run_id,
            component=EvaluationComponent.CONTEXT,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=("only explicitly supplied Context operations are visible",),
        )
        return TraceBatch(facts=(fact,), coverage=(coverage,))

    def context_assembly(
        self,
        assembly: ContextAssembly,
        *,
        action_id: UUID | None = None,
    ) -> TraceBatch:
        requirement = assembly.requirement
        fact = EvaluationFact(
            fact_id=stable_evaluation_id(
                "context-assembly",
                requirement.run_id,
                requirement.node_id,
                assembly.assembled_at.isoformat(),
                canonical_evaluation_json(assembly),
            ),
            component=EvaluationComponent.CONTEXT,
            category=TraceCategory.CONTEXT_CHANGE,
            kind="context.assembled",
            source=self.module_id,
            occurred_at=assembly.assembled_at,
            correlation=EvaluationCorrelation(
                run_id=requirement.run_id,
                task_id=requirement.task_id,
                node_id=requirement.node_id,
                action_id=action_id,
            ),
            source_scope=f"context-assembly:{requirement.run_id}",
            source_record_id=str(assembly.assembled_at.timestamp()),
            payload={
                "unit_count": len(assembly.units),
                "used_tokens": assembly.used_tokens,
                "omitted_count": len(assembly.omitted_context_ids),
                "context_ids": [str(unit.context_id) for unit in assembly.units],
            },
        )
        coverage = TraceCoverage(
            run_id=requirement.run_id,
            component=EvaluationComponent.CONTEXT,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=("only explicitly supplied Context assemblies are visible",),
        )
        return TraceBatch(facts=(fact,), coverage=(coverage,))

    def context_lifecycle(
        self,
        result: ContextLifecycleResult,
        *,
        run_id: UUID,
        task_id: UUID | None = None,
        node_id: UUID | None = None,
    ) -> TraceBatch:
        decision = result.decision
        fact = EvaluationFact(
            fact_id=stable_evaluation_id(
                "context-lifecycle",
                run_id,
                decision.context_id,
                decision.action.value,
                decision.source_revision,
                result.completed_at.isoformat(),
            ),
            component=EvaluationComponent.CONTEXT,
            category=TraceCategory.CONTEXT_CHANGE,
            kind=f"context.lifecycle.{decision.action.value}",
            source=self.module_id,
            occurred_at=result.completed_at,
            correlation=EvaluationCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                context_id=decision.context_id,
            ),
            source_scope=f"context-lifecycle:{decision.context_id}",
            source_record_id=(
                f"{decision.context_id}:{decision.action.value}:"
                f"{decision.source_revision}"
            ),
            payload={
                "action": decision.action.value,
                "source_revision": decision.source_revision,
                "reason": decision.reason,
                "pressure_ratio": decision.pressure_ratio,
                "result_revision": (
                    result.unit.revision if result.unit is not None else None
                ),
                "archive_id": (
                    str(result.archive_reference.archive_id)
                    if result.archive_reference is not None
                    else None
                ),
            },
        )
        coverage = TraceCoverage(
            run_id=run_id,
            component=EvaluationComponent.CONTEXT,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=(
                "lifecycle facts cover policy-applied Context transitions",
            ),
        )
        return TraceBatch(facts=(fact,), coverage=(coverage,))

    def memory_update(
        self,
        update: MemoryUpdateResult,
        *,
        correlation: EvaluationCorrelation,
    ) -> TraceBatch:
        memory = update.memory
        if correlation.memory_id not in {None, memory.memory_id}:
            raise TraceCorrelationError("Memory correlation has another memory_id")
        if correlation.candidate_id not in {None, update.candidate_id}:
            raise TraceCorrelationError("Memory correlation has another candidate_id")
        resolved = correlation.model_copy(
            update={
                "memory_id": memory.memory_id,
                "candidate_id": update.candidate_id,
            }
        )
        fact = EvaluationFact(
            fact_id=stable_evaluation_id(
                "memory-update",
                correlation.run_id,
                memory.memory_id,
                memory.revision,
                update.candidate_id,
            ),
            component=EvaluationComponent.MEMORY,
            category=TraceCategory.MEMORY_OPERATION,
            kind=f"memory.{update.evolution.value}",
            source=self.module_id,
            occurred_at=memory.updated_at,
            correlation=resolved,
            source_scope=f"memory:{memory.memory_id}",
            source_sequence=memory.revision,
            source_record_id=f"{memory.memory_id}:{memory.revision}",
            payload={
                "evolution": update.evolution.value,
                "status": memory.status.value,
                "revision": memory.revision,
                "confidence": memory.confidence,
                "conflict_count": len(memory.conflicts),
                "had_previous_snapshot": update.previous_memory is not None,
            },
        )
        coverage = TraceCoverage(
            run_id=correlation.run_id,
            component=EvaluationComponent.MEMORY,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=("Memory operations require explicit run correlation",),
        )
        return TraceBatch(facts=(fact,), coverage=(coverage,))


class EvaluationInputAssembler:
    """Compose post-run snapshots; it is not part of the Runtime Event Loop."""

    module_id = "evaluation.integration.input_assembler"

    def __init__(
        self,
        *,
        collector: TraceCollector,
        runtime_adapter: RuntimeTraceAdapter | None = None,
        tool_adapter: ToolTraceAdapter | None = None,
    ) -> None:
        self._collector = collector
        self._runtime_adapter = runtime_adapter or RuntimeTraceAdapter()
        self._tool_adapter = tool_adapter or ToolTraceAdapter()

    def assemble(
        self,
        result: RunResult,
        *,
        runtime_entries: Sequence[TraceEntry],
        tool_entries: Sequence[ToolTraceEntry] = (),
        extra_batches: Sequence[TraceBatch] = (),
    ) -> EvaluationSubject:
        state = result.final_state
        batches = [
            self._runtime_adapter.adapt(
                runtime_entries,
                run_id=state.run_id,
                task_id=state.task.task_id,
            ),
            *extra_batches,
        ]
        if tool_entries:
            batches.append(self._tool_adapter.adapt(tool_entries))
        trace = self._collector.collect(
            run_id=state.run_id,
            task_id=state.task.task_id,
            batches=batches,
        )
        return EvaluationSubject(
            trace=trace,
            state=self._state_snapshot(state),
            result=self._result_snapshot(state),
        )

    @staticmethod
    def _state_snapshot(state: AgentState) -> EvaluationStateSnapshot:
        dumped = state.model_dump(mode="json")
        return EvaluationStateSnapshot(
            run_id=state.run_id,
            task_id=state.task.task_id,
            task_description=state.task.description,
            status=ExecutionStatus(state.status.value),
            revision=state.revision,
            step_count=state.step_count,
            output=dumped["output"],
            error=state.error,
            captured_at=state.updated_at,
        )

    @staticmethod
    def _result_snapshot(state: AgentState) -> ExecutionResultSnapshot:
        dumped = state.model_dump(mode="json")
        return ExecutionResultSnapshot(
            run_id=state.run_id,
            task_id=state.task.task_id,
            succeeded=state.status.value == ExecutionStatus.COMPLETED.value,
            output=dumped["output"],
            error=state.error,
            completed_at=state.updated_at,
        )


class AgentEvaluationPipeline:
    """Run read-only evaluators and return one immutable report."""

    module_id = "evaluation.integration.pipeline"

    def __init__(
        self,
        *,
        outcome: OutcomeEvaluator,
        trajectory: TrajectoryEvaluator,
        components: Sequence[ComponentEvaluator],
    ) -> None:
        self._outcome = outcome
        self._trajectory = trajectory
        self._components = tuple(
            sorted(components, key=lambda evaluator: evaluator.component.value)
        )
        component_ids = tuple(item.component for item in self._components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("component evaluators must be unique")

    def evaluate(
        self,
        subject: EvaluationSubject,
        criteria: EvaluationCriteria,
    ) -> EvaluationReport:
        outcome = self._outcome.evaluate(subject, criteria)
        trajectory = self._trajectory.evaluate(subject)
        components = tuple(
            evaluator.evaluate(subject) for evaluator in self._components
        )
        trace_fact_ids = {fact.fact_id for fact in subject.trace.facts}
        for result in (outcome, trajectory, *components):
            evidence_ids = {
                evidence_id
                for finding in result.findings
                for evidence_id in finding.evidence_fact_ids
            }
            unknown = evidence_ids - trace_fact_ids
            if unknown:
                raise ValueError(
                    f"evaluator '{result.evaluator_id}' references unknown Trace "
                    "evidence"
                )
        scores = tuple(
            result.score
            for result in (outcome, trajectory, *components)
            if result.score is not None
        )
        overall = sum(scores) / len(scores) if scores else None
        report_id = stable_evaluation_id(
            "evaluation-report",
            subject.trace.trace_id,
            outcome.evaluation_id,
            trajectory.evaluation_id,
            *(item.evaluation_id for item in components),
        )
        return EvaluationReport(
            report_id=report_id,
            trace_id=subject.trace.trace_id,
            run_id=subject.trace.run_id,
            task_id=subject.trace.task_id,
            outcome=outcome,
            trajectory=trajectory,
            components=components,
            overall_score=overall,
            created_at=subject.result.completed_at,
        )
