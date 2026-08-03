"""Lightweight web console for the Research Agent application.

The console intentionally lives in the Application layer.  It serializes the
public ``ResearchRunResult`` snapshot and never reaches into Runtime internals.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from adaptive_agent_runtime.llm import BackendAvailability

from applications.research_agent.agent import ResearchAgent
from applications.research_agent.cli import load_llm_config_file
from applications.research_agent.llm_deployment import (
    ResearchLLMDeploymentConfig,
    build_research_llm_deployment,
)
from applications.research_agent.report import ResearchRunResult
from applications.research_agent.progress import (
    ResearchProgressEvent,
    ResearchProgressSink,
)


ASSET_ROOT = Path(__file__).with_name("web_assets")
MAX_REQUEST_BYTES = 16 * 1024


def _value(value: Any) -> Any:
    """Return a JSON-safe snapshot without retaining immutable wrappers."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    return value


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


def _dynamic_node_ids(result: ResearchRunResult) -> set[str]:
    dynamic: set[str] = set()
    for entry in result.runtime_trace:
        payload = entry.event.payload
        mutations = _nested(
            payload,
            "observation",
            "metadata",
            "orchestration",
            "mutations",
        )
        if not isinstance(mutations, (tuple, list)):
            continue
        for mutation in mutations:
            if not isinstance(mutation, Mapping):
                continue
            if mutation.get("mutation_type") != "add_node":
                continue
            node_id = _nested(mutation, "node", "node_id")
            if node_id:
                dynamic.add(str(node_id))
    return dynamic


def build_research_view(
    result: ResearchRunResult,
    *,
    task: str,
    elapsed_seconds: float,
    inference_label: str,
) -> dict[str, Any]:
    """Project one public application result into the console ViewModel."""

    state = result.runtime_result.final_state
    dynamic_ids = _dynamic_node_ids(result)
    nodes = []
    for node in result.task_graph.nodes:
        observation = node.observation
        nodes.append(
            {
                "id": str(node.node_id),
                "shortId": str(node.node_id)[:8],
                "goal": node.goal,
                "expectedOutput": node.expected_output,
                "strategy": node.strategy_id,
                "status": node.status.value,
                "dependencies": [str(item) for item in node.dependencies],
                "dynamic": str(node.node_id) in dynamic_ids,
                "output": _value(observation.output) if observation else None,
                "error": node.failure_reason,
            }
        )

    runtime_trace = []
    for entry in result.runtime_trace:
        payload = entry.event.payload
        runtime_trace.append(
            {
                "sequence": entry.sequence,
                "kind": entry.event.kind,
                "source": entry.event.source,
                "occurredAt": entry.event.occurred_at.isoformat(),
                "nodeId": _trace_node_id(payload),
            }
        )

    tool_observations = [
        {
            "invocationId": str(item.invocation_id),
            "capability": item.capability_id,
            "provider": item.provider_id,
            "status": item.status.value,
            "retryStatus": item.retry_status.value,
            "attempts": len(item.attempts),
            "nodeId": (
                str(item.correlation.node_id)
                if item.correlation.node_id is not None
                else None
            ),
            "output": _value(item.output),
            "error": item.error,
        }
        for item in result.tool_observations
    ]

    context_units = [
        {
            "id": str(item.context_id),
            "shortId": str(item.context_id)[:8],
            "state": item.lifecycle_state.value,
            "residency": item.residency_policy.value,
            "source": item.metadata.source.value,
            "layer": item.metadata.layer.value,
            "importance": item.metadata.importance,
            "tokens": item.metadata.estimated_tokens,
            "tags": list(item.metadata.tags),
            "nodeId": (
                str(item.metadata.node_id)
                if item.metadata.node_id is not None
                else None
            ),
            "content": _value(item.content),
            "conclusions": list(item.core_conclusions),
        }
        for item in result.context_units
    ]

    assemblies = [
        {
            "nodeId": (
                str(item.requirement.node_id)
                if item.requirement.node_id is not None
                else None
            ),
            "goal": item.requirement.goal,
            "usedTokens": item.used_tokens,
            "maxTokens": item.requirement.max_tokens,
            "units": len(item.units),
            "omitted": len(item.omitted_context_ids),
        }
        for item in result.context_assemblies
    ]

    memories = [
        {
            "id": str(item.memory_id),
            "key": item.memory_key,
            "status": item.status.value,
            "confidence": item.confidence,
            "revision": item.revision,
            "evidenceCount": len(item.evidence),
            "condition": _value(item.condition),
            "content": _value(item.content),
            "conflictCount": len(item.conflicts),
        }
        for item in result.memories
    ]

    evaluation = {
        "overallScore": result.evaluation.overall_score,
        "outcome": _evaluation_item(result.evaluation.outcome),
        "trajectory": _evaluation_item(result.evaluation.trajectory),
        "components": [
            _evaluation_item(item) for item in result.evaluation.components
        ],
        "failurePatterns": [
            {
                "key": item.pattern_key,
                "component": item.component.value,
                "description": item.description,
                "rootCause": item.root_cause.description,
                "confidence": item.pattern_confidence,
            }
            for item in result.failure_analysis.patterns
        ],
        "optimizationProposals": [
            {
                "id": str(item.proposal_id),
                "component": item.target_component.value,
                "changeKind": item.change_kind,
                "benefit": item.expected_benefit,
                "confidence": item.proposal_confidence,
                "status": item.status.value,
            }
            for item in result.optimization_proposals
        ],
        "historyRuns": result.evaluation_history_runs,
    }

    authorization_uses = {
        str(item.authorization_id): item for item in result.authorization_uses
    }
    governance = [
        {
            "scenario": record.scenario,
            "operation": record.request.operation,
            "risk": record.request.risk.value,
            "preliminary": record.preliminary.outcome.value,
            "final": record.final.outcome.value,
            "level": record.final.level.value,
            "reviewed": record.review is not None,
            "reviewStatus": (
                record.review.status.value if record.review is not None else None
            ),
            "authorizationId": (
                str(record.authorization.authorization_id)
                if record.authorization is not None
                else None
            ),
            "authorizationUse": (
                authorization_uses[
                    str(record.authorization.authorization_id)
                ].status.value
                if record.authorization is not None
                and str(record.authorization.authorization_id)
                in authorization_uses
                else None
            ),
            "reason": record.final.reason,
        }
        for record in result.governance_records
    ]

    return {
        "meta": {
            "runId": str(state.run_id),
            "taskId": str(state.task.task_id),
            "task": task,
            "company": result.report.company,
            "status": state.status.value,
            "steps": state.step_count,
            "elapsedSeconds": round(elapsed_seconds, 2),
            "inference": inference_label,
            "graphVersion": result.task_graph.version,
            "fixtureNotice": (
                "Research evidence uses illustrative demo fixtures; "
                "do not treat this report as investment advice."
            ),
        },
        "summary": {
            "nodes": len(nodes),
            "completedNodes": sum(
                1 for item in nodes if item["status"] == "completed"
            ),
            "runtimeEvents": len(runtime_trace),
            "toolTraceEntries": len(result.tool_trace),
            "contextAssemblies": len(assemblies),
            "governanceReviews": sum(
                1 for item in governance if item["reviewed"]
            ),
        },
        "graph": {
            "id": str(result.task_graph.graph_id),
            "version": result.task_graph.version,
            "nodes": nodes,
        },
        "runtimeTrace": runtime_trace,
        "tools": tool_observations,
        "context": {"units": context_units, "assemblies": assemblies},
        "memories": memories,
        "evaluation": evaluation,
        "governance": governance,
        "report": result.report.model_dump(mode="json"),
        "llm": {
            "reasoning": len(result.llm_reasoning),
            "contextPackages": len(result.llm_context_packages),
            "toolIntents": len(result.llm_tool_intents),
            "actionProposals": len(result.llm_action_proposals),
            "graphMutationProposals": len(
                result.llm_graph_mutation_proposals
            ),
            "autonomousExecutions": len(result.agent_executions),
            "judgeEnabled": result.llm_judgement is not None,
        },
    }


def _evaluation_item(item: Any) -> dict[str, Any]:
    component = item.component.value if item.component is not None else None
    return {
        "component": component,
        "verdict": item.verdict.value,
        "score": item.score,
        "confidence": item.assessment_confidence,
        "findings": [
            {
                "code": finding.code,
                "severity": finding.severity.value,
                "summary": finding.summary,
            }
            for finding in item.findings
        ],
    }


@dataclass(frozen=True)
class ResearchConsoleApplication:
    llm_config: ResearchLLMDeploymentConfig | None = None

    @property
    def inference_label(self) -> str:
        if self.llm_config is None:
            return "Deterministic Runtime"
        target = self.llm_config.target
        profile = target.build_profile()
        return f"{target.model_id} · {profile.backend_kind.value}"

    async def run(
        self,
        task: str,
        *,
        progress_sink: ResearchProgressSink | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.llm_config is None:
            result = await ResearchAgent().run_demo(
                task,
                progress_sink=progress_sink,
            )
        else:
            deployment = build_research_llm_deployment(self.llm_config)
            probe = await deployment.probe()
            if probe.availability is not BackendAvailability.AVAILABLE:
                diagnostics = ", ".join(probe.diagnostics) or "no diagnostics"
                raise RuntimeError(
                    f"LLM preflight failed: {probe.availability.value} "
                    f"({diagnostics})"
                )
            result = await ResearchAgent(
                cognitive_capabilities=deployment.cognitive_capabilities,
            ).run(task, progress_sink=progress_sink)
        return build_research_view(
            result,
            task=task,
            elapsed_seconds=time.perf_counter() - started,
            inference_label=self.inference_label,
        )


class RunChannel:
    """One in-memory, single-viewer event stream and terminal result."""

    def __init__(self, stream_id: UUID) -> None:
        self.stream_id = stream_id
        self._events: Queue[dict[str, Any]] = Queue()
        self._lock = Lock()
        self._done = Event()
        self._sequence = 0
        self._run_id: str | None = None
        self._result: dict[str, Any] | None = None
        self._error: str | None = None

    async def publish(self, event: ResearchProgressEvent) -> None:
        self._run_id = str(event.run_id)
        self._enqueue(
            event.kind.value,
            {
                "runId": str(event.run_id),
                "occurredAt": event.occurred_at.isoformat(),
                "traceSequence": event.trace_sequence,
                "payload": event.payload,
            },
        )

    def finish(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._result = result
            self._done.set()
        self._enqueue(
            "run.result_ready",
            {"runId": result["meta"]["runId"], "payload": {}},
        )

    def fail(self, error: str) -> None:
        with self._lock:
            self._error = error
            self._done.set()
        self._enqueue(
            "run.failed",
            {"runId": self._run_id, "payload": {"error": error}},
        )

    def next_event(self, timeout: float) -> dict[str, Any]:
        return self._events.get(timeout=timeout)

    def result(self) -> tuple[bool, dict[str, Any] | None, str | None]:
        with self._lock:
            return self._done.is_set(), self._result, self._error

    def _enqueue(self, kind: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        self._events.put(
            {
                "streamSequence": sequence,
                "streamId": str(self.stream_id),
                "kind": kind,
                **data,
            }
        )


class RunRegistry:
    """Small bounded registry for local Demo runs."""

    def __init__(self, *, max_runs: int = 16) -> None:
        self._max_runs = max_runs
        self._runs: dict[UUID, RunChannel] = {}
        self._lock = Lock()

    def create(self) -> RunChannel:
        with self._lock:
            if len(self._runs) >= self._max_runs:
                completed = [
                    run_id
                    for run_id, channel in self._runs.items()
                    if channel.result()[0]
                ]
                if not completed:
                    raise RuntimeError("too many active research runs")
                self._runs.pop(completed[0])
            stream_id = uuid4()
            channel = RunChannel(stream_id)
            self._runs[stream_id] = channel
            return channel

    def get(self, stream_id: str) -> RunChannel | None:
        try:
            identity = UUID(stream_id)
        except ValueError:
            return None
        with self._lock:
            return self._runs.get(identity)


class ResearchConsoleHandler(BaseHTTPRequestHandler):
    server_version = "AdaptiveResearchConsole/0.1"

    @property
    def application(self) -> ResearchConsoleApplication:
        return self.server.application  # type: ignore[attr-defined,no-any-return]

    @property
    def registry(self) -> RunRegistry:
        return self.server.registry  # type: ignore[attr-defined,no-any-return]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        request_path = urlparse(self.path).path
        if request_path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "inference": self.application.inference_label,
                },
            )
            return
        run_route = self._run_route(request_path)
        if run_route is not None:
            stream_id, operation = run_route
            channel = self.registry.get(stream_id)
            if channel is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
                return
            if operation == "events":
                self._stream_events(channel)
            else:
                self._run_result(channel)
            return
        asset = {
            "/": "index.html",
            "/index.html": "index.html",
            "/styles.css": "styles.css",
            "/app.js": "app.js",
        }.get(request_path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        asset_path = ASSET_ROOT / asset
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }[asset_path.suffix]
        data = asset_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlparse(self.path).path
        if path not in {"/api/research", "/api/runs"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            task = self._read_task()
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if path == "/api/runs":
            try:
                channel = self.registry.create()
            except RuntimeError as exc:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
                return
            Thread(
                target=self._execute_run,
                args=(task, channel),
                daemon=True,
                name=f"research-run-{channel.stream_id}",
            ).start()
            stream_id = str(channel.stream_id)
            self._json(
                HTTPStatus.ACCEPTED,
                {
                    "streamId": stream_id,
                    "eventsUrl": f"/api/runs/{stream_id}/events",
                    "resultUrl": f"/api/runs/{stream_id}/result",
                },
            )
            return
        try:
            result = asyncio.run(self.application.run(task))
        except Exception as exc:  # application boundary returns normalized error
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Research run failed: {exc}"},
            )
            return
        self._json(HTTPStatus.OK, result)

    def _read_task(self) -> str:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if len(task) > 500:
            raise ValueError("task must not exceed 500 characters")
        return task.strip()

    def _execute_run(self, task: str, channel: RunChannel) -> None:
        try:
            result = asyncio.run(
                self.application.run(task, progress_sink=channel)
            )
        except Exception as exc:
            channel.fail(f"Research run failed: {exc}")
            return
        channel.finish(result)

    @staticmethod
    def _run_route(path: str) -> tuple[str, str] | None:
        parts = path.strip("/").split("/")
        if (
            len(parts) == 4
            and parts[0] == "api"
            and parts[1] == "runs"
            and parts[3] in {"events", "result"}
        ):
            return parts[2], parts[3]
        return None

    def _run_result(self, channel: RunChannel) -> None:
        done, result, error = channel.result()
        if not done:
            self._json(HTTPStatus.ACCEPTED, {"status": "running"})
        elif error is not None:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": error})
        elif result is not None:
            self._json(HTTPStatus.OK, result)
        else:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "run completed without a result"},
            )

    def _stream_events(self, channel: RunChannel) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        terminal = {"run.result_ready", "run.failed"}
        try:
            while True:
                try:
                    event = channel.next_event(timeout=10.0)
                except Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                message = (
                    f"id: {event['streamSequence']}\n"
                    f"data: {data}\n\n"
                ).encode("utf-8")
                self.wfile.write(message)
                self.wfile.flush()
                if event["kind"] in terminal:
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        print(f"[research-console] {self.address_string()} {format % args}")

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class ResearchConsoleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        application: ResearchConsoleApplication,
    ) -> None:
        self.application = application
        self.registry = RunRegistry()
        super().__init__(address, ResearchConsoleHandler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Research Runtime Console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--llm-config", type=Path)
    parser.add_argument("--llm-target")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.llm_target is not None and args.llm_config is None:
        raise SystemExit("--llm-target requires --llm-config")
    llm_config = (
        load_llm_config_file(args.llm_config, target_name=args.llm_target)
        if args.llm_config is not None
        else None
    )
    application = ResearchConsoleApplication(llm_config=llm_config)
    server = ResearchConsoleServer((args.host, args.port), application)
    print(
        f"Research Runtime Console: http://{args.host}:{args.port} "
        f"[{application.inference_label}]"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
