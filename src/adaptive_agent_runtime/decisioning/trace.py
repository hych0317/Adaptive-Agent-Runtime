"""Map decision-specific facts into the existing Runtime TraceSink."""

from __future__ import annotations

from adaptive_agent_runtime.core.contracts import TraceSink
from adaptive_agent_runtime.core.models import RuntimeEvent
from pydantic import JsonValue
from adaptive_agent_runtime.decisioning.models import DecisionTraceEvent


class RuntimeDecisionTraceWriter:
    module_id = "decision.trace.runtime"

    def __init__(self, trace_sink: TraceSink) -> None:
        self._trace_sink = trace_sink

    async def record(self, event: DecisionTraceEvent) -> None:
        correlations: dict[str, JsonValue] = {
            "request_id": str(event.request_id),
            "task_id": str(event.task_id) if event.task_id is not None else None,
            "node_id": str(event.node_id) if event.node_id is not None else None,
            "action_id": str(event.action_id) if event.action_id is not None else None,
            "proposal_id": (
                str(event.proposal_id) if event.proposal_id is not None else None
            ),
            "validation_id": (
                str(event.validation_id) if event.validation_id is not None else None
            ),
            "governance_decision_id": (
                str(event.governance_decision_id)
                if event.governance_decision_id is not None
                else None
            ),
            "review_request_id": (
                str(event.review_request_id)
                if event.review_request_id is not None
                else None
            ),
            "authorization_id": (
                str(event.authorization_id)
                if event.authorization_id is not None
                else None
            ),
        }
        await self._trace_sink.record(
            RuntimeEvent(
                run_id=event.run_id,
                kind=event.kind.value,
                source=event.source,
                occurred_at=event.occurred_at,
                payload={
                    "correlation": correlations,
                    "decision": dict(event.payload),
                },
            )
        )
