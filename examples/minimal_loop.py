"""Run a deterministic two-decision Runtime Core loop."""

from __future__ import annotations

import asyncio
import json

from adaptive_agent_runtime import (
    ActionRequest,
    AgentRuntime,
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    Observation,
    PlanDecision,
)


class EchoPlanner:
    module_id = "example.echo_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        if state.last_observation is None:
            return PlanDecision.execute(
                ActionRequest(
                    name="echo",
                    arguments={"text": state.task.description},
                )
            )
        return PlanDecision.complete(output=state.last_observation.output)


class EchoExecutor:
    module_id = "example.echo_executor"

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        del state
        return Observation.ok(
            action.action_id,
            output={"message": action.arguments["text"]},
        )


async def main() -> None:
    state_store = InMemoryStateStore()
    trace_sink = InMemoryTraceSink()
    runtime = AgentRuntime(
        planner=EchoPlanner(),
        executor=EchoExecutor(),
        state_store=state_store,
        trace_sink=trace_sink,
    )

    result = await runtime.run(AgentTask(description="Phase 1 runtime loop"))
    trace = trace_sink.entries_for(result.final_state.run_id)

    print(result.model_dump_json(indent=2))
    print(
        json.dumps(
            [
                {"sequence": entry.sequence, "kind": entry.event.kind}
                for entry in trace
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

