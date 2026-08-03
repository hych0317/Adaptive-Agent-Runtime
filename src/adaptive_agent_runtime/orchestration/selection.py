"""Runtime-owned ready-node selection policies."""

from __future__ import annotations

from uuid import UUID

from adaptive_agent_runtime import AgentState
from adaptive_agent_runtime.orchestration.models import TaskNode


class FirstReadyTaskNodeSelector:
    """Preserve deterministic first-ready scheduling by default."""

    module_id = "orchestration.ready_node_selector.first"

    async def select_node_id(
        self,
        ready_nodes: tuple[TaskNode, ...],
        state: AgentState,
    ) -> UUID:
        del state
        if not ready_nodes:
            raise ValueError("ready-node selector requires candidates")
        return ready_nodes[0].node_id
