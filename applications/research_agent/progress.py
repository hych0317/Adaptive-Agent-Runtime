"""Application-owned progress events for Research Agent presentation layers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from adaptive_agent_runtime import InMemoryTraceSink, RuntimeEvent, TraceEntry


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchProgressKind(StrEnum):
    RUN_PREPARING = "run.preparing"
    GRAPH_INITIALIZED = "graph.initialized"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    GRAPH_NODE_ADDED = "graph.node_added"
    GRAPH_DEPENDENCY_ADDED = "graph.dependency_added"
    TRACE_RECORDED = "trace.recorded"
    RUNTIME_COMPLETED = "runtime.completed"
    RUNTIME_FAILED = "runtime.failed"


class ResearchProgressEvent(BaseModel):
    """One immutable Application-level progress fact."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    event_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    kind: ResearchProgressKind
    occurred_at: datetime = Field(default_factory=utc_now)
    trace_sequence: int | None = Field(default=None, ge=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class ResearchProgressSink(Protocol):
    async def publish(self, event: ResearchProgressEvent) -> None:
        """Publish one progress event without controlling execution."""

        ...


class ObservableTraceSink:
    """Record canonical Trace first, then best-effort progress notifications."""

    module_id = "research_agent.trace.observable"

    def __init__(
        self,
        primary: InMemoryTraceSink,
        progress: ResearchProgressSink,
    ) -> None:
        self._primary = primary
        self._progress = progress

    async def record(self, event: RuntimeEvent) -> TraceEntry:
        entry = await self._primary.record(event)
        for progress_event in progress_events_from_trace(entry):
            try:
                await self._progress.publish(progress_event)
            except Exception:
                # Presentation is observational and must never fail the Runtime.
                continue
        return entry


async def publish_progress(
    sink: ResearchProgressSink | None,
    event: ResearchProgressEvent,
) -> None:
    """Best-effort publication for non-Trace Application events."""

    if sink is None:
        return
    try:
        await sink.publish(event)
    except Exception:
        return


def progress_events_from_trace(entry: TraceEntry) -> tuple[ResearchProgressEvent, ...]:
    """Convert canonical Runtime Trace into small UI-oriented progress facts."""

    event = entry.event
    payload = event.payload
    progress: list[ResearchProgressEvent] = [
        ResearchProgressEvent(
            run_id=event.run_id,
            kind=ResearchProgressKind.TRACE_RECORDED,
            occurred_at=event.occurred_at,
            trace_sequence=entry.sequence,
            payload={
                "sequence": entry.sequence,
                "kind": event.kind,
                "source": event.source,
                "occurredAt": event.occurred_at.isoformat(),
                "nodeId": _trace_node_id(payload),
            },
        )
    ]

    if event.kind == "plan.created":
        node = _nested(payload, "plan", "action", "arguments", "node")
        if isinstance(node, Mapping):
            progress.append(
                ResearchProgressEvent(
                    run_id=event.run_id,
                    kind=ResearchProgressKind.NODE_STARTED,
                    occurred_at=event.occurred_at,
                    trace_sequence=entry.sequence,
                    payload={"node": _json(node)},
                )
            )

    if event.kind == "observation.received":
        observation = payload.get("observation")
        if isinstance(observation, Mapping):
            node_id = _nested(
                observation,
                "metadata",
                "orchestration",
                "node_id",
            )
            if node_id is not None:
                succeeded = observation.get("succeeded") is True
                progress.append(
                    ResearchProgressEvent(
                        run_id=event.run_id,
                        kind=(
                            ResearchProgressKind.NODE_COMPLETED
                            if succeeded
                            else ResearchProgressKind.NODE_FAILED
                        ),
                        occurred_at=event.occurred_at,
                        trace_sequence=entry.sequence,
                        payload={
                            "nodeId": str(node_id),
                            "output": _json(observation.get("output")),
                            "error": _json(observation.get("error")),
                        },
                    )
                )
            mutations = _nested(
                observation,
                "metadata",
                "orchestration",
                "mutations",
            )
            if isinstance(mutations, (tuple, list)):
                progress.extend(
                    _mutation_events(event.run_id, event.occurred_at, entry.sequence, mutations)
                )

    if event.kind == "runtime.completed":
        progress.append(
            ResearchProgressEvent(
                run_id=event.run_id,
                kind=ResearchProgressKind.RUNTIME_COMPLETED,
                occurred_at=event.occurred_at,
                trace_sequence=entry.sequence,
            )
        )
    elif event.kind == "runtime.failed":
        progress.append(
            ResearchProgressEvent(
                run_id=event.run_id,
                kind=ResearchProgressKind.RUNTIME_FAILED,
                occurred_at=event.occurred_at,
                trace_sequence=entry.sequence,
                payload={"error": _json(payload.get("error"))},
            )
        )
    return tuple(progress)


def _mutation_events(
    run_id: UUID,
    occurred_at: datetime,
    trace_sequence: int,
    mutations: tuple[Any, ...] | list[Any],
) -> tuple[ResearchProgressEvent, ...]:
    events: list[ResearchProgressEvent] = []
    for mutation in mutations:
        if not isinstance(mutation, Mapping):
            continue
        kind = mutation.get("mutation_type")
        if kind == "add_node" and isinstance(mutation.get("node"), Mapping):
            events.append(
                ResearchProgressEvent(
                    run_id=run_id,
                    kind=ResearchProgressKind.GRAPH_NODE_ADDED,
                    occurred_at=occurred_at,
                    trace_sequence=trace_sequence,
                    payload={"node": _json(mutation["node"])},
                )
            )
        elif kind == "add_dependency":
            node_id = mutation.get("node_id")
            dependency_id = mutation.get("dependency_id")
            if node_id is not None and dependency_id is not None:
                events.append(
                    ResearchProgressEvent(
                        run_id=run_id,
                        kind=ResearchProgressKind.GRAPH_DEPENDENCY_ADDED,
                        occurred_at=occurred_at,
                        trace_sequence=trace_sequence,
                        payload={
                            "nodeId": str(node_id),
                            "dependencyId": str(dependency_id),
                        },
                    )
                )
    return tuple(events)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _trace_node_id(payload: Mapping[str, Any]) -> str | None:
    candidates = (
        _nested(payload, "plan", "action", "arguments", "node", "node_id"),
        _nested(payload, "action", "arguments", "node", "node_id"),
        _nested(
            payload,
            "observation",
            "metadata",
            "orchestration",
            "node_id",
        ),
    )
    return next((str(item) for item in candidates if item), None)


def _json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
