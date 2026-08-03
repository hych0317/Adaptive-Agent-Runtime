from __future__ import annotations

import asyncio
import unittest
from typing import Any, Sequence, cast
from uuid import uuid4

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunStatus,
)
from adaptive_agent_runtime.context_memory import (
    ContextLayer,
    ContextSource,
    ObservationContextAdapter,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    ExecutionStrategy,
    StrategyActionExecutor,
    TaskNode,
    TaskNodeStatus,
)
from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    CapabilityRequest,
    CapabilityRequirement,
    CapabilityResolver,
    DeterministicToolSelector,
    ExactCapabilityMatcher,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
    MappedTaskCapabilityRequestProvider,
    ProviderAvailability,
    RetryPolicy,
    ToolCorrelation,
    ToolExecutionPolicy,
    ToolExecutionStatus,
    ToolExecutionStrategy,
    ToolInvocation,
    ToolProvider,
    ToolProviderMetadata,
    ToolProviderResult,
    ToolResultObservationAdapter,
    ToolSelection,
    ToolSelectionContext,
)


class EchoProvider:
    module_id = "test.provider.echo"

    def __init__(
        self,
        provider_id: str = "echo",
        *,
        fail: bool = False,
    ) -> None:
        self.provider_id = provider_id
        self.fail = fail
        self.invocations: list[ToolInvocation] = []

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        self.invocations.append(invocation)
        if self.fail:
            return ToolProviderResult.failed(error="echo failed", retryable=False)
        return ToolProviderResult.ok(
            output={"echo": invocation.model_dump(mode="json")["arguments"]}
        )


class HangingProvider:
    module_id = "test.provider.hanging"
    provider_id = "hanging"

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        del invocation
        await asyncio.sleep(10)
        return ToolProviderResult.ok(output="late")


class OutOfCandidatesSelector:
    module_id = "test.selector.out_of_candidates"

    def __init__(self, provider_id: str) -> None:
        self._provider_id = provider_id

    def select(
        self,
        requirement: CapabilityRequirement,
        candidates: Sequence[ToolProviderMetadata],
        context: ToolSelectionContext,
    ) -> ToolSelection:
        del candidates, context
        return ToolSelection(
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=self._provider_id,
            reason="malformed selector result for boundary test",
        )


def node(goal: str) -> TaskNode:
    return TaskNode(
        goal=goal,
        expected_output="provider-neutral result",
        strategy_id="tool",
    )


def tool_components(
    provider: ToolProvider,
    *,
    tags: tuple[str, ...] = (),
) -> tuple[
    InMemoryToolRegistry,
    CapabilityResolver,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
]:
    catalog = InMemoryCapabilityCatalog()
    catalog.register(
        Capability(
            capability_id="echo_capability",
            name="Echo",
            description="Echo provider-neutral arguments.",
        )
    )
    registry = InMemoryToolRegistry(catalog)
    registry.register(
        ToolProviderMetadata(
            provider_id=provider.provider_id,
            name="Echo Provider",
            capability_id="echo_capability",
            description="Deterministic test provider.",
            input_schema={"type": "object"},
            tags=tags,
        ),
        provider,
    )
    resolver = CapabilityResolver(
        catalog=catalog,
        registry=registry,
        matcher=ExactCapabilityMatcher(),
    )
    trace = InMemoryToolTraceSink()
    executor = ManagedToolExecutor(registry=registry, trace_sink=trace)
    return registry, resolver, trace, executor


class ToolIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_execution_strategy_runs_through_orchestration(self) -> None:
        task_node = node("echo financial query")
        provider = EchoProvider()
        _, resolver, trace, executor = tool_components(provider)
        request = CapabilityRequest(
            requirement=CapabilityRequirement(
                capability_id="echo_capability"
            ),
            arguments={"query": "ACME"},
        )
        strategy = ToolExecutionStrategy(
            requests=MappedTaskCapabilityRequestProvider(
                {task_node.node_id: request}
            ),
            resolver=resolver,
            selector=DeterministicToolSelector(),
            executor=executor,
            policy=ToolExecutionPolicy(retry=RetryPolicy(max_retries=1)),
        )
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(task_node,))
        )
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            max_steps=4,
        )

        result = await runtime.run(AgentTask(description="tool graph"))

        self.assertIsInstance(strategy, ExecutionStrategy)
        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        final_graph = planner.graph_for(result.final_state.run_id)
        self.assertEqual(final_graph.nodes[0].status, TaskNodeStatus.COMPLETED)
        self.assertEqual(len(provider.invocations), 1)
        correlation = provider.invocations[0].correlation
        self.assertEqual(correlation.run_id, result.final_state.run_id)
        self.assertEqual(correlation.task_id, result.final_state.task.task_id)
        self.assertEqual(correlation.node_id, task_node.node_id)
        self.assertTrue(
            all(
                entry.event.correlation == correlation
                for entry in trace.entries_for(
                    provider.invocations[0].invocation_id
                )
            )
        )
        observation = result.final_state.last_observation
        assert observation is not None
        output = cast(dict[str, Any], observation.output)
        echo = cast(dict[str, Any], output["echo"])
        self.assertEqual(echo["query"], "ACME")
        self.assertTrue(
            all(
                word not in field.lower()
                for field in TaskNode.model_fields
                for word in ("tool", "provider", "capability")
            )
        )

    async def test_tool_failure_becomes_failed_node_result(self) -> None:
        task_node = node("failing tool")
        provider = EchoProvider(fail=True)
        _, resolver, trace, executor = tool_components(provider)
        strategy = ToolExecutionStrategy(
            requests=MappedTaskCapabilityRequestProvider(
                {
                    task_node.node_id: CapabilityRequest(
                        requirement=CapabilityRequirement(
                            capability_id="echo_capability"
                        )
                    )
                }
            ),
            resolver=resolver,
            selector=DeterministicToolSelector(),
            executor=executor,
            policy=ToolExecutionPolicy(),
        )
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="failure"),
            status=RunStatus.RUNNING,
        )

        result = await strategy.execute(task_node, state)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.error, "echo failed")
        self.assertEqual(result.mutations, ())
        invocation = provider.invocations[0]
        self.assertEqual(invocation.correlation.run_id, state.run_id)
        self.assertEqual(invocation.correlation.node_id, task_node.node_id)
        self.assertTrue(
            all(
                entry.event.correlation == invocation.correlation
                for entry in trace.entries_for(invocation.invocation_id)
            )
        )

    async def test_selector_cannot_escape_runtime_candidates(self) -> None:
        task_node = node("candidate boundary")
        allowed = EchoProvider("allowed")
        registry, resolver, _, executor = tool_components(
            allowed,
            tags=("allowed",),
        )
        blocked = EchoProvider("blocked")
        registry.register(
            ToolProviderMetadata(
                provider_id="blocked",
                name="Blocked Provider",
                capability_id="echo_capability",
                description="Does not meet the task requirement.",
                input_schema={"type": "object"},
                tags=("blocked",),
            ),
            blocked,
        )
        request = CapabilityRequest(
            requirement=CapabilityRequirement(
                capability_id="echo_capability",
                required_provider_tags=("allowed",),
            )
        )
        strategy = ToolExecutionStrategy(
            requests=MappedTaskCapabilityRequestProvider(
                {task_node.node_id: request}
            ),
            resolver=resolver,
            selector=OutOfCandidatesSelector("blocked"),
            executor=executor,
            policy=ToolExecutionPolicy(),
        )
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="candidate boundary"),
            status=RunStatus.RUNNING,
        )

        result = await strategy.execute(task_node, state)

        self.assertFalse(result.succeeded)
        self.assertIn("outside Runtime candidates", result.error or "")
        self.assertEqual(allowed.invocations, [])
        self.assertEqual(blocked.invocations, [])

    async def test_tool_result_observation_flows_into_context_adapter(self) -> None:
        provider = EchoProvider()
        _, _, _, executor = tool_components(provider)
        action_id = uuid4()
        invocation = ToolInvocation(
            requirement_id=uuid4(),
            capability_id="echo_capability",
            provider_id="echo",
            arguments={"query": "context"},
            correlation=ToolCorrelation(action_id=action_id),
        )
        tool_result = await executor.execute(
            invocation,
            ToolExecutionPolicy(),
        )
        observation = ToolResultObservationAdapter().convert(
            tool_result,
            action_id=action_id,
        )
        task_node = node("consume observation")
        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="context integration"),
            status=RunStatus.RUNNING,
        )

        context = ObservationContextAdapter().convert(
            observation,
            state,
            node=task_node,
        )

        self.assertTrue(observation.succeeded)
        self.assertEqual(observation.action_id, action_id)
        self.assertIn("tool", observation.metadata)
        self.assertEqual(context.metadata.source, ContextSource.OBSERVATION)
        self.assertEqual(context.metadata.layer, ContextLayer.TASK)
        content = cast(dict[str, Any], context.content)
        tool_metadata = content["metadata"]["tool"]
        self.assertEqual(tool_metadata["provider_id"], "echo")
        self.assertEqual(tool_metadata["status"], "succeeded")
        self.assertEqual(
            tool_metadata["correlation"]["action_id"],
            str(action_id),
        )

    async def test_failed_tool_results_remain_typed_context_observations(
        self,
    ) -> None:
        failure = EchoProvider("failure", fail=True)
        _, _, _, failure_executor = tool_components(failure)
        failed = await failure_executor.execute(
            ToolInvocation(
                requirement_id=uuid4(),
                capability_id="echo_capability",
                provider_id="failure",
            ),
            ToolExecutionPolicy(),
        )

        hanging = HangingProvider()
        _, _, _, timeout_executor = tool_components(hanging)
        timed_out = await timeout_executor.execute(
            ToolInvocation(
                requirement_id=uuid4(),
                capability_id="echo_capability",
                provider_id="hanging",
            ),
            ToolExecutionPolicy(timeout_seconds=0.001),
        )

        unavailable_provider = EchoProvider("unavailable")
        registry, _, _, unavailable_executor = tool_components(
            unavailable_provider
        )
        registry.set_availability(
            "unavailable",
            ProviderAvailability.UNAVAILABLE,
        )
        unavailable = await unavailable_executor.execute(
            ToolInvocation(
                requirement_id=uuid4(),
                capability_id="echo_capability",
                provider_id="unavailable",
            ),
            ToolExecutionPolicy(),
        )

        state = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="failed tool context"),
            status=RunStatus.RUNNING,
        )
        task_node = node("consume failed tool result")
        expected_statuses = (
            ToolExecutionStatus.FAILED,
            ToolExecutionStatus.TIMED_OUT,
            ToolExecutionStatus.PROVIDER_UNAVAILABLE,
        )
        for result, expected_status in zip(
            (failed, timed_out, unavailable),
            expected_statuses,
            strict=True,
        ):
            with self.subTest(status=expected_status):
                observation = ToolResultObservationAdapter().convert(
                    result,
                    action_id=uuid4(),
                )
                context = ObservationContextAdapter().convert(
                    observation,
                    state,
                    node=task_node,
                )
                content = cast(dict[str, Any], context.content)
                tool_metadata = content["metadata"]["tool"]

                self.assertFalse(observation.succeeded)
                self.assertIsNotNone(observation.error)
                self.assertEqual(tool_metadata["status"], expected_status.value)
                self.assertEqual(
                    context.metadata.source,
                    ContextSource.OBSERVATION,
                )
                self.assertEqual(context.metadata.layer, ContextLayer.TASK)
                self.assertIn("failure", context.metadata.tags)


if __name__ == "__main__":
    unittest.main()
