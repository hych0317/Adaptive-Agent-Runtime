"""Backend contract for isolated autonomous Agent execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.agent_backends.models import (
    AutonomousAgentRequest,
    AutonomousAgentResult,
)
from adaptive_agent_runtime.llm.models import AgentTargetProfile, BackendProbeResult


@runtime_checkable
class AutonomousAgentBackend(RuntimeModule, Protocol):
    @property
    def target_id(self) -> str: ...

    @property
    def profile(self) -> AgentTargetProfile: ...

    async def probe(self) -> BackendProbeResult: ...

    async def execute(
        self,
        request: AutonomousAgentRequest,
    ) -> AutonomousAgentResult: ...
