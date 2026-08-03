from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from adaptive_agent_runtime import (
    AgentState,
    AgentTask,
    CoreEventKind,
    Observation,
    PlanDecision,
    RunResult,
    RunStatus,
    RuntimeEvent,
    TraceEntry,
)
from adaptive_agent_runtime.context_memory import (
    ContextLayer,
    ContextLifecycleState,
    ContextMetadata,
    ContextSource,
    ContextUnit,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryUnit,
    MemoryUpdateResult,
)
from adaptive_agent_runtime.evaluation import (
    ContextMemoryTraceAdapter,
    EvaluationComponent,
    EvaluationCorrelation,
    EvaluationFact,
    EvaluationInputAssembler,
    GraphSnapshotAdapter,
    RuntimeNativeTraceCollector,
    RuntimeTraceAdapter,
    ToolTraceAdapter,
    TraceBatch,
    TraceCategory,
    TraceCompleteness,
    TraceCorrelationError,
    TraceCoverage,
)
from adaptive_agent_runtime.orchestration import DynamicTaskGraph, TaskNode
from adaptive_agent_runtime.tool_ecosystem import (
    ToolCorrelation,
    ToolTraceEntry,
    ToolTraceEvent,
    ToolTraceEventKind,
)


BASE_TIME = datetime(2026, 8, 2, 1, 0, tzinfo=timezone.utc)


def uid(value: int) -> UUID:
    return UUID(int=value)


def normalized_fact(
    fact_id: int,
    *,
    run_id: UUID,
    task_id: UUID,
    component: EvaluationComponent = EvaluationComponent.RUNTIME,
    kind: str = "runtime.fact",
    node_id: UUID | None = None,
    action_id: UUID | None = None,
    invocation_id: UUID | None = None,
) -> EvaluationFact:
    return EvaluationFact(
        fact_id=uid(fact_id),
        component=component,
        category=TraceCategory.RUNTIME,
        kind=kind,
        source="test",
        occurred_at=BASE_TIME + timedelta(seconds=fact_id),
        correlation=EvaluationCorrelation(
            run_id=run_id,
            task_id=task_id,
            node_id=node_id,
            action_id=action_id,
            invocation_id=invocation_id,
        ),
        source_scope=f"test:{run_id}",
        source_sequence=fact_id,
        source_record_id=str(fact_id),
    )


class TraceCollectorTests(unittest.TestCase):
    def test_runtime_adapter_isolates_run_and_detects_sequence_gaps(self) -> None:
        run_a = uid(90)
        run_b = uid(91)
        task_a = uid(92)
        task_b = uid(93)

        def entry(
            sequence: int,
            event_id: int,
            run_id: UUID,
            task_id: UUID,
            kind: CoreEventKind,
        ) -> TraceEntry:
            return TraceEntry(
                sequence=sequence,
                event=RuntimeEvent(
                    event_id=uid(event_id),
                    run_id=run_id,
                    kind=kind,
                    source="runtime.core",
                    occurred_at=BASE_TIME + timedelta(seconds=event_id),
                    payload={"state": {"task": {"task_id": str(task_id)}}},
                ),
                recorded_at=BASE_TIME + timedelta(seconds=event_id),
            )

        interleaved = (
            entry(1, 94, run_a, task_a, CoreEventKind.RUNTIME_STARTED),
            entry(1, 95, run_b, task_b, CoreEventKind.RUNTIME_STARTED),
            entry(2, 96, run_a, task_a, CoreEventKind.RUNTIME_COMPLETED),
            entry(2, 97, run_b, task_b, CoreEventKind.RUNTIME_COMPLETED),
        )
        adapter = RuntimeTraceAdapter()

        batch_a = adapter.adapt(interleaved, run_id=run_a, task_id=task_a)
        batch_b = adapter.adapt(interleaved, run_id=run_b, task_id=task_b)
        gapped = adapter.adapt(
            (
                entry(1, 98, run_a, task_a, CoreEventKind.RUNTIME_STARTED),
                entry(99, 99, run_a, task_a, CoreEventKind.RUNTIME_COMPLETED),
            ),
            run_id=run_a,
            task_id=task_a,
        )

        self.assertEqual({fact.correlation.run_id for fact in batch_a.facts}, {run_a})
        self.assertEqual({fact.correlation.run_id for fact in batch_b.facts}, {run_b})
        self.assertEqual(batch_a.coverage[0].completeness, TraceCompleteness.COMPLETE)
        self.assertEqual(batch_b.coverage[0].completeness, TraceCompleteness.COMPLETE)
        self.assertEqual(gapped.coverage[0].completeness, TraceCompleteness.PARTIAL)
        self.assertIn(
            "Runtime trace sequence is not contiguous from 1",
            gapped.coverage[0].diagnostics,
        )

    def test_invocation_link_disambiguates_actions_and_rejects_multiple_nodes(
        self,
    ) -> None:
        run_id = uid(100)
        task_id = uid(101)
        node_id = uid(102)
        invocation_id = uid(103)
        first_action = uid(104)
        selected_action = uid(105)
        facts = (
            normalized_fact(
                106,
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=first_action,
            ),
            normalized_fact(
                107,
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=selected_action,
                invocation_id=invocation_id,
            ),
            normalized_fact(
                108,
                run_id=run_id,
                task_id=task_id,
                component=EvaluationComponent.TOOL,
                node_id=node_id,
                invocation_id=invocation_id,
            ),
        )
        collector = RuntimeNativeTraceCollector()

        trace = collector.collect(
            run_id=run_id,
            task_id=task_id,
            batches=(TraceBatch(facts=facts),),
        )

        tool_fact = trace.facts_for(EvaluationComponent.TOOL)[0]
        self.assertEqual(tool_fact.correlation.action_id, selected_action)

        conflicting_node = normalized_fact(
            109,
            run_id=run_id,
            task_id=task_id,
            component=EvaluationComponent.TOOL,
            node_id=uid(110),
            invocation_id=invocation_id,
        )
        with self.assertRaises(TraceCorrelationError):
            collector.collect(
                run_id=run_id,
                task_id=task_id,
                batches=(TraceBatch(facts=(*facts, conflicting_node)),),
            )

    def test_trace_identity_includes_fact_content_and_coverage_diagnostics(
        self,
    ) -> None:
        run_id = uid(111)
        task_id = uid(112)
        original = normalized_fact(113, run_id=run_id, task_id=task_id)
        changed = original.model_copy(update={"payload": {"changed": True}})
        collector = RuntimeNativeTraceCollector()
        base_coverage = TraceCoverage(
            run_id=run_id,
            component=EvaluationComponent.RUNTIME,
            completeness=TraceCompleteness.PARTIAL,
            diagnostics=("first",),
        )
        changed_coverage = base_coverage.model_copy(
            update={"diagnostics": ("second",)}
        )

        original_trace = collector.collect(
            run_id=run_id,
            task_id=task_id,
            batches=(TraceBatch(facts=(original,), coverage=(base_coverage,)),),
        )
        changed_fact_trace = collector.collect(
            run_id=run_id,
            task_id=task_id,
            batches=(TraceBatch(facts=(changed,), coverage=(base_coverage,)),),
        )
        changed_coverage_trace = collector.collect(
            run_id=run_id,
            task_id=task_id,
            batches=(TraceBatch(facts=(original,), coverage=(changed_coverage,)),),
        )

        self.assertNotEqual(original_trace.trace_id, changed_fact_trace.trace_id)
        self.assertNotEqual(original_trace.trace_id, changed_coverage_trace.trace_id)

    def test_input_assembler_projects_terminal_core_snapshots(self) -> None:
        run_id = uid(80)
        task_id = uid(81)
        previous_action_id = uid(84)
        previous_node_id = uid(85)
        task = AgentTask(task_id=task_id, description="assemble evaluation input")
        final_state = AgentState(
            run_id=run_id,
            task=task,
            status=RunStatus.COMPLETED,
            revision=2,
            last_plan=PlanDecision.complete(output={"answer": 42}),
            last_observation=Observation.ok(
                previous_action_id,
                metadata={
                    "orchestration": {"node_id": str(previous_node_id)}
                },
            ),
            output={"answer": 42},
            created_at=BASE_TIME,
            updated_at=BASE_TIME + timedelta(seconds=2),
        )
        entries = (
            TraceEntry(
                sequence=1,
                event=RuntimeEvent(
                    event_id=uid(82),
                    run_id=run_id,
                    kind=CoreEventKind.RUNTIME_STARTED,
                    source="runtime.core",
                    occurred_at=BASE_TIME,
                    payload={"state": {"task": {"task_id": str(task_id)}}},
                ),
                recorded_at=BASE_TIME,
            ),
            TraceEntry(
                sequence=2,
                event=RuntimeEvent(
                    event_id=uid(86),
                    run_id=run_id,
                    kind=CoreEventKind.STATE_UPDATED,
                    source="runtime.core",
                    occurred_at=BASE_TIME + timedelta(seconds=1),
                    payload={"state": final_state.model_dump(mode="json")},
                ),
                recorded_at=BASE_TIME + timedelta(seconds=1),
            ),
            TraceEntry(
                sequence=3,
                event=RuntimeEvent(
                    event_id=uid(83),
                    run_id=run_id,
                    kind=CoreEventKind.RUNTIME_COMPLETED,
                    source="runtime.core",
                    occurred_at=BASE_TIME + timedelta(seconds=2),
                    payload={"state": final_state.model_dump(mode="json")},
                ),
                recorded_at=BASE_TIME + timedelta(seconds=2),
            ),
        )

        subject = EvaluationInputAssembler(
            collector=RuntimeNativeTraceCollector()
        ).assemble(
            RunResult(final_state=final_state),
            runtime_entries=entries,
        )

        self.assertEqual(subject.trace.run_id, run_id)
        self.assertEqual(subject.state.revision, final_state.revision)
        self.assertEqual(subject.state.output, {"answer": 42})
        self.assertTrue(subject.result.succeeded)
        self.assertEqual(subject.result.output, {"answer": 42})
        terminal_update = next(
            fact for fact in subject.trace.facts if fact.fact_id == uid(86)
        )
        terminal_result = next(
            fact for fact in subject.trace.facts if fact.fact_id == uid(83)
        )
        self.assertEqual(terminal_update.component, EvaluationComponent.RUNTIME)
        self.assertIsNone(terminal_update.correlation.action_id)
        self.assertIsNone(terminal_update.correlation.node_id)
        self.assertIsNone(terminal_result.correlation.action_id)
        self.assertIsNone(terminal_result.correlation.node_id)

    def test_runtime_and_tool_traces_merge_with_explicit_correlation(self) -> None:
        run_id = uid(1)
        task_id = uid(2)
        graph_id = uid(3)
        node_id = uid(4)
        action_id = uid(5)
        invocation_id = uid(6)
        runtime_entries = (
            TraceEntry(
                sequence=1,
                event=RuntimeEvent(
                    event_id=uid(10),
                    run_id=run_id,
                    kind=CoreEventKind.RUNTIME_STARTED,
                    source="runtime.core",
                    occurred_at=BASE_TIME,
                    payload={"state": {"task": {"task_id": str(task_id)}}},
                ),
                recorded_at=BASE_TIME,
            ),
            TraceEntry(
                sequence=2,
                event=RuntimeEvent(
                    event_id=uid(11),
                    run_id=run_id,
                    kind=CoreEventKind.ACTION_STARTED,
                    source="orchestration.strategy_executor",
                    occurred_at=BASE_TIME + timedelta(seconds=2),
                    payload={
                        "action": {
                            "action_id": str(action_id),
                            "name": "orchestration.execute_node",
                            "arguments": {
                                "graph_id": str(graph_id),
                                "node": {"node_id": str(node_id)},
                            },
                        }
                    },
                ),
                recorded_at=BASE_TIME + timedelta(seconds=2),
            ),
            TraceEntry(
                sequence=3,
                event=RuntimeEvent(
                    event_id=uid(12),
                    run_id=run_id,
                    kind=CoreEventKind.OBSERVATION_RECEIVED,
                    source="orchestration.strategy_executor",
                    occurred_at=BASE_TIME + timedelta(seconds=5),
                    payload={
                        "observation": {
                            "action_id": str(action_id),
                            "succeeded": True,
                            "metadata": {
                                "orchestration": {
                                    "node_id": str(node_id),
                                    "mutations": [],
                                },
                                "tool": {
                                    "invocation_id": str(invocation_id),
                                    "correlation": {
                                        "action_id": str(action_id),
                                    },
                                },
                            },
                        }
                    },
                ),
                recorded_at=BASE_TIME + timedelta(seconds=5),
            ),
        )
        tool_entries = (
            ToolTraceEntry(
                sequence=1,
                event=ToolTraceEvent(
                    event_id=uid(20),
                    invocation_id=invocation_id,
                    requirement_id=uid(7),
                    capability_id="financial_information",
                    provider_id="provider-a",
                    kind=ToolTraceEventKind.EXECUTION_STARTED,
                    correlation=ToolCorrelation(
                        run_id=run_id,
                        task_id=task_id,
                        node_id=node_id,
                    ),
                    occurred_at=BASE_TIME + timedelta(seconds=3),
                ),
                recorded_at=BASE_TIME + timedelta(seconds=3),
            ),
            ToolTraceEntry(
                sequence=2,
                event=ToolTraceEvent(
                    event_id=uid(21),
                    invocation_id=invocation_id,
                    requirement_id=uid(7),
                    capability_id="financial_information",
                    provider_id="provider-a",
                    kind=ToolTraceEventKind.EXECUTION_FINISHED,
                    correlation=ToolCorrelation(
                        run_id=run_id,
                        task_id=task_id,
                        node_id=node_id,
                    ),
                    occurred_at=BASE_TIME + timedelta(seconds=4),
                    payload={
                        "status": "succeeded",
                        "retry_status": "not_retried",
                        "attempt_count": 1,
                    },
                ),
                recorded_at=BASE_TIME + timedelta(seconds=4),
            ),
        )
        context_fact = EvaluationFact(
            fact_id=uid(30),
            component=EvaluationComponent.CONTEXT,
            category=TraceCategory.CONTEXT_CHANGE,
            kind="context.active",
            source="context_memory",
            occurred_at=BASE_TIME + timedelta(seconds=6),
            correlation=EvaluationCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=action_id,
                context_id=uid(31),
            ),
            source_scope="context:31",
            source_sequence=0,
            source_record_id="context-31-revision-0",
        )
        context_batch = TraceBatch(
            facts=(context_fact,),
            coverage=(
                TraceCoverage(
                    run_id=run_id,
                    component=EvaluationComponent.CONTEXT,
                    completeness=TraceCompleteness.PARTIAL,
                ),
            ),
        )
        runtime_batch = RuntimeTraceAdapter().adapt(
            runtime_entries,
            run_id=run_id,
            task_id=task_id,
        )
        tool_batch = ToolTraceAdapter().adapt(tool_entries)
        collector = RuntimeNativeTraceCollector()

        trace = collector.collect(
            run_id=run_id,
            task_id=task_id,
            batches=(runtime_batch, tool_batch, context_batch),
        )
        reversed_trace = collector.collect(
            run_id=run_id,
            task_id=task_id,
            batches=(context_batch, tool_batch, runtime_batch),
        )

        self.assertEqual(trace, reversed_trace)
        tool_facts = trace.facts_for(EvaluationComponent.TOOL)
        self.assertEqual(len(tool_facts), 2)
        self.assertTrue(
            all(fact.correlation.action_id == action_id for fact in tool_facts)
        )
        self.assertTrue(
            all(fact.correlation.node_id == node_id for fact in tool_facts)
        )
        self.assertEqual(
            tuple(fact.source_sequence for fact in tool_facts),
            (1, 2),
        )
        self.assertEqual(context_fact.correlation.action_id, action_id)
        self.assertEqual(trace.run_id, run_id)
        self.assertEqual(trace.task_id, task_id)

    def test_conflicting_invocation_action_links_are_rejected(self) -> None:
        run_id = uid(40)
        task_id = uid(41)
        invocation_id = uid(42)
        facts = (
            normalized_fact(
                43,
                run_id=run_id,
                task_id=task_id,
                invocation_id=invocation_id,
                action_id=uid(44),
            ),
            normalized_fact(
                45,
                run_id=run_id,
                task_id=task_id,
                invocation_id=invocation_id,
                action_id=uid(46),
            ),
        )

        with self.assertRaises(TraceCorrelationError):
            RuntimeNativeTraceCollector().collect(
                run_id=run_id,
                task_id=task_id,
                batches=(TraceBatch(facts=facts),),
            )

    def test_interleaved_runs_remain_isolated(self) -> None:
        run_a = uid(50)
        run_b = uid(51)
        task_id = uid(52)
        shared_node = uid(53)
        shared_action = uid(54)
        batch = TraceBatch(
            facts=(
                normalized_fact(
                    55,
                    run_id=run_b,
                    task_id=task_id,
                    node_id=shared_node,
                    action_id=shared_action,
                ),
                normalized_fact(
                    56,
                    run_id=run_a,
                    task_id=task_id,
                    node_id=shared_node,
                    action_id=shared_action,
                ),
            )
        )
        collector = RuntimeNativeTraceCollector()

        trace_a = collector.collect(
            run_id=run_a,
            task_id=task_id,
            batches=(batch,),
        )
        trace_b = collector.collect(
            run_id=run_b,
            task_id=task_id,
            batches=(batch,),
        )

        self.assertEqual(tuple(item.fact_id for item in trace_a.facts), (uid(56),))
        self.assertEqual(tuple(item.fact_id for item in trace_b.facts), (uid(55),))
        self.assertNotEqual(trace_a.trace_id, trace_b.trace_id)

    def test_graph_context_and_memory_snapshots_are_normalized(self) -> None:
        run_id = uid(60)
        task_id = uid(61)
        node_id = uid(62)
        action_id = uid(63)
        task_node = TaskNode(
            node_id=node_id,
            goal="research",
            expected_output="report",
            strategy_id="mock",
        )
        graph = DynamicTaskGraph(graph_id=uid(64), nodes=(task_node,))
        graph_batch = GraphSnapshotAdapter().adapt(
            graph,
            run_id=run_id,
            task_id=task_id,
            observed_at=BASE_TIME,
        )
        later_graph_batch = GraphSnapshotAdapter().adapt(
            graph,
            run_id=run_id,
            task_id=task_id,
            observed_at=BASE_TIME + timedelta(seconds=1),
        )
        context = ContextUnit(
            context_id=uid(65),
            content={"observation": "data"},
            metadata=ContextMetadata(
                source=ContextSource.OBSERVATION,
                layer=ContextLayer.TASK,
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                created_at=BASE_TIME,
                updated_at=BASE_TIME,
            ),
            lifecycle_state=ContextLifecycleState.ACTIVE,
        )
        adapter = ContextMemoryTraceAdapter()
        context_batch = adapter.context_unit(context, action_id=action_id)
        evidence = MemoryEvidence(
            evidence_id=uid(66),
            source_context_id=context.context_id,
            note="observed",
            observed_at=BASE_TIME,
        )
        memory = MemoryUnit(
            memory_id=uid(67),
            memory_key="research.preference",
            content="prefer audited sources",
            condition=MemoryCondition(),
            evidence=(evidence,),
            confidence=0.8,
            last_candidate_id=uid(68),
            last_candidate_fingerprint="fingerprint",
            created_at=BASE_TIME,
            updated_at=BASE_TIME,
        )
        update = MemoryUpdateResult(
            evolution=MemoryEvolutionType.EXTEND,
            memory=memory,
            candidate_id=uid(68),
        )
        memory_batch = adapter.memory_update(
            update,
            correlation=EvaluationCorrelation(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=action_id,
            ),
        )

        trace = RuntimeNativeTraceCollector().collect(
            run_id=run_id,
            task_id=task_id,
            batches=(
                graph_batch,
                later_graph_batch,
                context_batch,
                memory_batch,
            ),
        )

        self.assertEqual(
            {fact.component for fact in trace.facts},
            {
                EvaluationComponent.ORCHESTRATION,
                EvaluationComponent.CONTEXT,
                EvaluationComponent.MEMORY,
            },
        )
        memory_fact = trace.facts_for(EvaluationComponent.MEMORY)[0]
        self.assertEqual(memory_fact.correlation.memory_id, memory.memory_id)
        self.assertEqual(memory_fact.correlation.candidate_id, uid(68))
        graph_facts = trace.facts_for(EvaluationComponent.ORCHESTRATION)
        self.assertEqual(len(graph_facts), 2)
        self.assertNotEqual(graph_facts[0].fact_id, graph_facts[1].fact_id)


if __name__ == "__main__":
    unittest.main()
