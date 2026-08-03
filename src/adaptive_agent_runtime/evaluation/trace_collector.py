"""Merge normalized facts without controlling their execution sources."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from adaptive_agent_runtime.evaluation.models import (
    AgentExecutionTrace,
    EvaluationComponent,
    EvaluationCorrelation,
    EvaluationFact,
    TraceBatch,
    TraceCompleteness,
    TraceCoverage,
    canonical_evaluation_json,
    stable_evaluation_id,
)


class TraceCollectionError(ValueError):
    """Raised when normalized trace facts cannot form one valid snapshot."""


class TraceCorrelationError(TraceCollectionError):
    """Raised when explicit cross-module identifiers contradict each other."""


_TRACKED_COMPONENTS = (
    EvaluationComponent.RUNTIME,
    EvaluationComponent.ORCHESTRATION,
    EvaluationComponent.TOOL,
    EvaluationComponent.CONTEXT,
    EvaluationComponent.MEMORY,
)
_COMPLETENESS_RANK = {
    TraceCompleteness.MISSING: 0,
    TraceCompleteness.PARTIAL: 1,
    TraceCompleteness.COMPLETE: 2,
}
_EMPTY_TRACE_TIME = datetime(1970, 1, 1, tzinfo=timezone.utc)


class RuntimeNativeTraceCollector:
    """Build an immutable per-run partial order from read-only fact batches."""

    module_id = "evaluation.trace_collector.runtime_native"

    def collect(
        self,
        *,
        run_id: UUID,
        task_id: UUID,
        batches: Sequence[TraceBatch],
    ) -> AgentExecutionTrace:
        deduplicated: dict[UUID, EvaluationFact] = {}
        for batch in batches:
            for fact in batch.facts:
                if fact.correlation.run_id != run_id:
                    continue
                existing = deduplicated.get(fact.fact_id)
                if existing is None:
                    deduplicated[fact.fact_id] = fact
                elif existing != fact:
                    raise TraceCollectionError(
                        f"fact_id '{fact.fact_id}' has conflicting snapshots"
                    )

        facts = self._enrich_correlations(
            tuple(deduplicated.values()),
            run_id=run_id,
            task_id=task_id,
        )
        ordered = tuple(sorted(facts, key=self._view_key))
        coverage = self._merge_coverage(run_id, batches)
        collected_at = (
            max(fact.occurred_at for fact in ordered)
            if ordered
            else _EMPTY_TRACE_TIME
        )
        trace_id = stable_evaluation_id(
            "agent-trace",
            run_id,
            task_id,
            *(canonical_evaluation_json(fact) for fact in ordered),
            *(canonical_evaluation_json(item) for item in coverage),
        )
        return AgentExecutionTrace(
            trace_id=trace_id,
            run_id=run_id,
            task_id=task_id,
            facts=ordered,
            coverage=coverage,
            collected_at=collected_at,
        )

    @staticmethod
    def _view_key(fact: EvaluationFact) -> tuple[object, ...]:
        source_sequence = (
            fact.source_sequence
            if fact.source_sequence is not None
            else 2**63 - 1
        )
        return (
            fact.occurred_at,
            fact.component.value,
            fact.source_scope,
            source_sequence,
            str(fact.fact_id),
        )

    def _enrich_correlations(
        self,
        facts: tuple[EvaluationFact, ...],
        *,
        run_id: UUID,
        task_id: UUID,
    ) -> tuple[EvaluationFact, ...]:
        invocation_actions: defaultdict[UUID, set[UUID]] = defaultdict(set)
        invocation_nodes: defaultdict[UUID, set[UUID]] = defaultdict(set)
        action_nodes: defaultdict[UUID, set[UUID]] = defaultdict(set)
        node_actions: defaultdict[UUID, set[UUID]] = defaultdict(set)

        for fact in facts:
            correlation = fact.correlation
            if correlation.task_id not in {None, task_id}:
                raise TraceCorrelationError(
                    f"fact '{fact.fact_id}' belongs to another task"
                )
            if (
                correlation.invocation_id is not None
                and correlation.action_id is not None
            ):
                invocation_actions[correlation.invocation_id].add(
                    correlation.action_id
                )
            if (
                correlation.invocation_id is not None
                and correlation.node_id is not None
            ):
                invocation_nodes[correlation.invocation_id].add(
                    correlation.node_id
                )
            if correlation.action_id is not None and correlation.node_id is not None:
                action_nodes[correlation.action_id].add(correlation.node_id)
                node_actions[correlation.node_id].add(correlation.action_id)

        conflicting_invocations = tuple(
            invocation_id
            for invocation_id, action_ids in invocation_actions.items()
            if len(action_ids) > 1
        )
        if conflicting_invocations:
            raise TraceCorrelationError(
                "an invocation is correlated with multiple actions: "
                + ", ".join(sorted(map(str, conflicting_invocations)))
            )
        conflicting_invocation_nodes = tuple(
            invocation_id
            for invocation_id, node_ids in invocation_nodes.items()
            if len(node_ids) > 1
        )
        if conflicting_invocation_nodes:
            raise TraceCorrelationError(
                "an invocation is correlated with multiple task nodes: "
                + ", ".join(sorted(map(str, conflicting_invocation_nodes)))
            )
        conflicting_actions = tuple(
            action_id
            for action_id, node_ids in action_nodes.items()
            if len(node_ids) > 1
        )
        if conflicting_actions:
            raise TraceCorrelationError(
                "an action is correlated with multiple task nodes: "
                + ", ".join(sorted(map(str, conflicting_actions)))
            )

        enriched: list[EvaluationFact] = []
        for fact in facts:
            correlation = fact.correlation
            action_id = correlation.action_id
            node_id = correlation.node_id
            if correlation.invocation_id is not None:
                linked_invocation_nodes = invocation_nodes.get(
                    correlation.invocation_id,
                    set(),
                )
                if node_id is None and len(linked_invocation_nodes) == 1:
                    node_id = next(iter(linked_invocation_nodes))
                linked_actions = invocation_actions.get(
                    correlation.invocation_id,
                    set(),
                )
                if action_id is not None and linked_actions and linked_actions != {
                    action_id
                }:
                    raise TraceCorrelationError(
                        f"fact '{fact.fact_id}' conflicts with invocation linkage"
                    )
                if action_id is None and len(linked_actions) == 1:
                    action_id = next(iter(linked_actions))
            if action_id is None and node_id is not None:
                linked_node_actions = node_actions.get(node_id, set())
                if len(linked_node_actions) == 1:
                    action_id = next(iter(linked_node_actions))
            if node_id is None and action_id is not None:
                linked_nodes = action_nodes.get(action_id, set())
                if len(linked_nodes) == 1:
                    node_id = next(iter(linked_nodes))
            elif node_id is not None and action_id is not None:
                linked_nodes = action_nodes.get(action_id, set())
                if linked_nodes and node_id not in linked_nodes:
                    raise TraceCorrelationError(
                        f"fact '{fact.fact_id}' conflicts with action linkage"
                    )

            updated = EvaluationCorrelation(
                run_id=run_id,
                task_id=task_id,
                graph_id=correlation.graph_id,
                node_id=node_id,
                action_id=action_id,
                invocation_id=correlation.invocation_id,
                context_id=correlation.context_id,
                memory_id=correlation.memory_id,
                candidate_id=correlation.candidate_id,
            )
            enriched.append(fact.model_copy(update={"correlation": updated}))
        return tuple(enriched)

    @staticmethod
    def _merge_coverage(
        run_id: UUID,
        batches: Sequence[TraceBatch],
    ) -> tuple[TraceCoverage, ...]:
        by_component: defaultdict[
            EvaluationComponent,
            list[TraceCoverage],
        ] = defaultdict(list)
        for batch in batches:
            for coverage in batch.coverage:
                if coverage.run_id == run_id:
                    by_component[coverage.component].append(coverage)

        merged: list[TraceCoverage] = []
        for component in _TRACKED_COMPONENTS:
            entries = by_component.get(component, [])
            if not entries:
                merged.append(
                    TraceCoverage(
                        run_id=run_id,
                        component=component,
                        completeness=TraceCompleteness.MISSING,
                        diagnostics=("no trace source reported this component",),
                    )
                )
                continue
            completeness = max(
                (item.completeness for item in entries),
                key=_COMPLETENESS_RANK.__getitem__,
            )
            diagnostics = tuple(
                sorted(
                    {
                        diagnostic
                        for item in entries
                        for diagnostic in item.diagnostics
                    }
                )
            )
            merged.append(
                TraceCoverage(
                    run_id=run_id,
                    component=component,
                    completeness=completeness,
                    diagnostics=diagnostics,
                )
            )
        return tuple(merged)
