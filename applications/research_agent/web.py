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
import ipaddress
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from adaptive_agent_runtime.llm import BackendAvailability

from applications.research_agent.agent import (
    ResearchAgent,
    ResearchInformationMode,
)
from applications.research_agent.llm_deployment import (
    ResearchLLMDeploymentConfig,
    build_research_llm_deployment,
)
from applications.research_agent.llm_tools import (
    RESEARCH_INFORMATION_RETRIEVAL_TOOL,
)
from applications.research_agent.report import ResearchRunResult
from applications.research_agent.progress import (
    ResearchProgressEvent,
    ResearchProgressSink,
)
from applications.research_agent.web_llm import WebLLMSettings


ASSET_ROOT = Path(__file__).with_name("web_assets")
DEFAULT_LLM_CONFIG = Path(__file__).parents[2] / "config" / "llm.toml"
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
        mutations = (
            payload.get("mutations")
            if entry.event.kind == "orchestration.graph_mutated"
            else _nested(
                payload,
                "observation",
                "metadata",
                "orchestration",
                "mutations",
            )
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
    execution_mode: ResearchInformationMode = (
        ResearchInformationMode.FIXTURE_DEMO
    ),
    tool_intent_enabled: bool = False,
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
                "scope": _value(item.scope),
                "targetKey": item.target_key.value,
                "currentValue": _value(item.current_value),
                "proposedValue": _value(item.proposed_value),
                "expectedImpact": item.expected_impact,
                "risk": item.risk_classification.value,
                "effectFingerprint": item.effect_fingerprint,
            }
            for item in result.optimization_proposals
        ],
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

    tool_specification = RESEARCH_INFORMATION_RETRIEVAL_TOOL.model_dump(
        mode="json"
    )
    llm_tool_intents = [
        {
            "nodeId": str(record.node_id),
            "callKey": record.intent.call_key,
            "capabilityId": record.intent.capability_id,
            "arguments": _value(record.intent.arguments),
            "provider": record.observation.provider_id,
            "status": record.observation.status.value,
            "output": _value(record.observation.output),
            "error": record.observation.error,
        }
        for record in result.llm_tool_intents
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
            "executionMode": execution_mode.value,
            "fixtureNotice": (
                (
                    "Information Retrieval and report generation used the "
                    "selected LLM. This is model synthesis, not live web or "
                    "market-data retrieval; verify material facts."
                )
                if execution_mode is ResearchInformationMode.LLM_RESEARCH
                else (
                    "Illustrative fixtures are active for stable Context & "
                    "Memory, Evaluation, and Governance demonstrations; do not "
                    "treat this report as investment advice."
                )
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
            "toolIntentEnabled": tool_intent_enabled,
            "candidateTools": (
                [
                    {
                        "capabilityId": tool_specification["capability_id"],
                        "name": tool_specification["name"],
                        "description": tool_specification["description"],
                        "inputSchema": tool_specification["input_schema"],
                    }
                ]
                if tool_intent_enabled
                else []
            ),
            "toolIntentRecords": llm_tool_intents,
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
    llm_settings: WebLLMSettings | None = None

    def _current_llm_config(self) -> ResearchLLMDeploymentConfig | None:
        if self.llm_settings is not None:
            configured = self.llm_settings.load_active()
            if configured is not None:
                return configured
        return self.llm_config

    @property
    def inference_label(self) -> str:
        config = self._current_llm_config()
        if config is None:
            return "Deterministic Runtime"
        target = config.target
        profile = target.build_profile()
        return f"{target.model_id} · {profile.backend_kind.value}"

    def llm_settings_view(self) -> dict[str, object]:
        if self.llm_settings is None:
            raise RuntimeError("web LLM settings are not enabled")
        return self.llm_settings.describe()

    def configure_llm(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, object]:
        if self.llm_settings is None:
            raise RuntimeError("web LLM settings are not enabled")
        self.llm_settings.activate(
            target_name,
            api_key=api_key,
            model_id=model_id,
        )
        return self.llm_settings.describe()

    async def discover_llm_models(
        self,
        target_name: str,
        *,
        api_key: str | None = None,
    ) -> dict[str, object]:
        """Probe one configured provider and expose its live model catalog."""

        if self.llm_settings is None:
            raise RuntimeError("web LLM settings are not enabled")
        config = self.llm_settings.load_target(target_name, api_key=api_key)
        deployment = build_research_llm_deployment(config)
        result = await deployment.probe()
        availability = result.availability
        if (
            availability is BackendAvailability.AVAILABLE
            and "authentication_missing" in result.diagnostics
        ):
            availability = BackendAvailability.AUTH_REQUIRED
        return {
            "availability": availability.value,
            "target": target_name,
            "currentModel": deployment.model_id,
            "models": list(result.available_model_ids),
            "runtimeVersion": result.runtime_version,
            "authMethod": result.active_auth_method,
            "diagnostics": list(result.diagnostics),
        }

    async def probe_llm(self) -> dict[str, object]:
        config = self._current_llm_config()
        if config is None:
            raise ValueError("select an LLM target before probing")
        deployment = build_research_llm_deployment(config)
        result = await deployment.probe()
        availability = result.availability
        if (
            availability is BackendAvailability.AVAILABLE
            and "authentication_missing" in result.diagnostics
        ):
            availability = BackendAvailability.AUTH_REQUIRED
        return {
            "availability": availability.value,
            "targetId": deployment.target_id,
            "model": deployment.model_id,
            "availableModels": list(result.available_model_ids),
            "runtimeVersion": result.runtime_version,
            "authMethod": result.active_auth_method,
            "diagnostics": list(result.diagnostics),
            "inference": self.inference_label,
        }

    async def run(
        self,
        task: str,
        *,
        progress_sink: ResearchProgressSink | None = None,
        execution_mode: str = "auto",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        llm_config = self._current_llm_config()
        if execution_mode not in {
            "auto",
            ResearchInformationMode.FIXTURE_DEMO.value,
            ResearchInformationMode.LLM_RESEARCH.value,
        }:
            raise ValueError("unsupported research execution mode")
        resolved_mode = (
            ResearchInformationMode.LLM_RESEARCH
            if execution_mode == "auto" and llm_config is not None
            else (
                ResearchInformationMode.FIXTURE_DEMO
                if execution_mode == "auto"
                else ResearchInformationMode(execution_mode)
            )
        )
        if resolved_mode is ResearchInformationMode.FIXTURE_DEMO:
            agent = ResearchAgent()
            try:
                result = await agent.run_demo(
                    task,
                    progress_sink=progress_sink,
                )
            finally:
                agent.close()
            inference_label = "Deterministic Runtime · Fixture Demo"
        else:
            if llm_config is None:
                raise ValueError(
                    "LLM Research mode requires an active LLM target"
                )
            deployment = build_research_llm_deployment(llm_config)
            probe = await deployment.probe()
            if (
                probe.availability is not BackendAvailability.AVAILABLE
                or "authentication_missing" in probe.diagnostics
            ):
                diagnostics = ", ".join(probe.diagnostics) or "no diagnostics"
                raise RuntimeError(
                    f"LLM preflight failed: {probe.availability.value} "
                    f"({diagnostics})"
                )
            agent = ResearchAgent(
                cognitive_capabilities=deployment.cognitive_capabilities,
                information_mode=ResearchInformationMode.LLM_RESEARCH,
            )
            try:
                result = await agent.run(task, progress_sink=progress_sink)
            finally:
                agent.close()
            inference_label = self.inference_label
        return build_research_view(
            result,
            task=task,
            elapsed_seconds=time.perf_counter() - started,
            inference_label=inference_label,
            execution_mode=resolved_mode,
            tool_intent_enabled=(
                llm_config is not None
                and llm_config.reasoner_tool_intent_limit > 0
            ),
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
        if request_path == "/api/llm/settings":
            if not self._require_local_client():
                return
            try:
                settings = self.application.llm_settings_view()
            except (RuntimeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, settings)
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
        if path in {
            "/api/llm/settings",
            "/api/llm/models",
            "/api/llm/probe",
        }:
            if not self._require_local_client():
                return
            self._handle_llm_post(path)
            return
        if path not in {"/api/research", "/api/runs"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            task, execution_mode = self._read_run_request()
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
                args=(task, execution_mode, channel),
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
            result = asyncio.run(
                self.application.run(task, execution_mode=execution_mode)
            )
        except Exception as exc:  # application boundary returns normalized error
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Research run failed: {exc}"},
            )
            return
        self._json(HTTPStatus.OK, result)

    def _read_run_request(self) -> tuple[str, str]:
        payload = self._read_json_payload()
        task = payload.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if len(task) > 500:
            raise ValueError("task must not exceed 500 characters")
        mode = payload.get("mode", "auto")
        if not isinstance(mode, str) or mode not in {
            "auto",
            ResearchInformationMode.FIXTURE_DEMO.value,
            ResearchInformationMode.LLM_RESEARCH.value,
        }:
            raise ValueError("mode must be auto, llm_research, or fixture_demo")
        return task.strip(), mode

    def _read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    def _handle_llm_post(self, path: str) -> None:
        if path == "/api/llm/settings":
            try:
                payload = self._read_json_payload()
                target = payload.get("target")
                api_key = payload.get("apiKey")
                model = payload.get("model")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("target must be a non-empty string")
                if api_key is not None and not isinstance(api_key, str):
                    raise ValueError("apiKey must be a string")
                if model is not None and not isinstance(model, str):
                    raise ValueError("model must be a string")
                settings = self.application.configure_llm(
                    target,
                    api_key=api_key,
                    model_id=model,
                )
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._json(
                HTTPStatus.OK,
                {
                    **settings,
                    "inference": self.application.inference_label,
                },
            )
            return
        if path == "/api/llm/models":
            try:
                payload = self._read_json_payload()
                target = payload.get("target")
                api_key = payload.get("apiKey")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError("target must be a non-empty string")
                if api_key is not None and not isinstance(api_key, str):
                    raise ValueError("apiKey must be a string")
                result = asyncio.run(
                    self.application.discover_llm_models(
                        target.strip(),
                        api_key=api_key,
                    )
                )
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"LLM model discovery failed: {exc}"},
                )
                return
            self._json(HTTPStatus.OK, result)
            return
        try:
            payload = self._read_json_payload()
            if payload:
                raise ValueError("probe request body must be an empty object")
            result = asyncio.run(self.application.probe_llm())
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"LLM probe failed: {exc}"},
            )
            return
        self._json(HTTPStatus.OK, result)

    def _require_local_client(self) -> bool:
        address = self.client_address[0]
        try:
            client = ipaddress.ip_address(address)
        except ValueError:
            self._json(HTTPStatus.FORBIDDEN, {"error": "local access required"})
            return False
        mapped = getattr(client, "ipv4_mapped", None)
        if not client.is_loopback and not (mapped is not None and mapped.is_loopback):
            self._json(HTTPStatus.FORBIDDEN, {"error": "local access required"})
            return False
        host_header = self.headers.get("Host", "")
        try:
            request_host = urlparse(f"//{host_header}").hostname
        except ValueError:
            request_host = None
        if request_host not in {"localhost", "127.0.0.1", "::1"}:
            self._json(HTTPStatus.FORBIDDEN, {"error": "local host required"})
            return False
        return True

    def _execute_run(
        self,
        task: str,
        execution_mode: str,
        channel: RunChannel,
    ) -> None:
        try:
            result = asyncio.run(
                self.application.run(
                    task,
                    progress_sink=channel,
                    execution_mode=execution_mode,
                )
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
    config_path = args.llm_config or DEFAULT_LLM_CONFIG
    llm_settings = WebLLMSettings(
        config_path,
        active_target=args.llm_target,
        activate_config_default=(
            args.llm_config is not None and args.llm_target is None
        ),
    )
    application = ResearchConsoleApplication(llm_settings=llm_settings)
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
