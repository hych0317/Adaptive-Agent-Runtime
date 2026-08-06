from __future__ import annotations

import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import patch

from adaptive_agent_runtime import RunStatus
from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    AnthropicMessagesConfig,
    BackendAvailability,
    BackendTransportFeatures,
    OpenAICompatibleService,
    OpenAICompatibleTargetDefinition,
    StructuredOutputLevel,
)

from applications.research_agent import (
    ResearchAgent as RuntimeResearchAgent,
    ResearchLLMCapability,
    ResearchLLMDeploymentConfig,
    build_research_llm_deployment,
)
from tests.runtime_database import register_test_resource, runtime_database_path


def ResearchAgent(**kwargs: object) -> RuntimeResearchAgent:
    kwargs.setdefault("persistence_path", runtime_database_path())
    kwargs.setdefault("run_kind", "test")
    kwargs.setdefault("disposable", True)
    return register_test_resource(
        RuntimeResearchAgent(**kwargs)  # type: ignore[arg-type]
    )
from applications.research_agent.report import ReportSection, ResearchReport


class ResearchLLMHTTPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_transport_probe_gateway_and_report(self) -> None:
        state: dict[str, list[Any]] = {"gets": [], "posts": []}
        report = ResearchReport(
            company="Tesla",
            executive_summary="Real loopback transport report.",
            sections=(
                ReportSection(
                    title="Evidence",
                    findings=("HTTP boundary remained Runtime-governed.",),
                ),
            ),
            risk_factors=("External providers still require live validation.",),
            investment_view="Balanced",
            markdown="# Tesla Real HTTP Report",
        )
        artifact = {
            "media_type": "application/json",
            "content": report.model_dump(mode="json"),
            "evidence_reference_ids": [],
            "warnings": [],
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                state["gets"].append((self.path, dict(self.headers)))
                if self.path != "/v1/models":
                    self._send(404, {"error": "not found"})
                    return
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "loopback-model", "object": "model"}
                        ],
                    },
                )

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                state["posts"].append(
                    (self.path, dict(self.headers), body)
                )
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": "not found"})
                    return
                self._send(
                    200,
                    {
                        "id": "loopback-chat",
                        "model": "loopback-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(artifact),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                            "total_tokens": 150,
                        },
                    },
                )

            def _send(self, status: int, body: dict[str, Any]) -> None:
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            if not isinstance(host, str):
                raise AssertionError("loopback server address is not text")
            target = OpenAICompatibleTargetDefinition(
                service=OpenAICompatibleService.LOCAL,
                target_id="research/loopback",
                model_id="loopback-model",
                base_url=f"http://{host}:{port}/v1",
                features=BackendTransportFeatures(
                    structured_output=StructuredOutputLevel.JSON_SCHEMA,
                ),
                supported_cognitive_capability_ids=("artifact_generation",),
            )
            deployment = build_research_llm_deployment(
                ResearchLLMDeploymentConfig(
                    target=target,
                    enabled_capabilities=(
                        ResearchLLMCapability.GENERATION,
                    ),
                )
            )

            with patch.dict(
                "os.environ",
                {"NO_PROXY": "127.0.0.1,localhost"},
                clear=False,
            ):
                probe = await deployment.probe()
                result = await ResearchAgent(
                    cognitive_capabilities=deployment.cognitive_capabilities,
                ).run("分析 Tesla 投资价值")
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await asyncio.to_thread(thread.join, 5.0)

        self.assertIs(probe.availability, BackendAvailability.AVAILABLE)
        self.assertEqual(
            probe.diagnostics, ("connectivity_and_model_verified",)
        )
        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.report.markdown, "# Tesla Real HTTP Report")
        self.assertEqual(len(state["gets"]), 1)
        self.assertEqual(len(state["posts"]), 1)
        post_path, post_headers, post_body = state["posts"][0]
        self.assertEqual(post_path, "/v1/chat/completions")
        self.assertNotIn("Authorization", post_headers)
        self.assertEqual(post_body["model"], "loopback-model")
        self.assertEqual(post_body["response_format"]["type"], "json_schema")
        self.assertEqual(len(result.llm_context_packages), 1)

    async def test_real_http_anthropic_protocol_and_report(self) -> None:
        state: dict[str, list[Any]] = {"gets": [], "posts": []}
        report = ResearchReport(
            company="Tesla",
            executive_summary="Real Anthropic loopback report.",
            sections=(
                ReportSection(
                    title="Evidence",
                    findings=("Native Messages boundary was preserved.",),
                ),
            ),
            risk_factors=("Live provider validation remains external.",),
            investment_view="Balanced",
            markdown="# Tesla Real Anthropic HTTP Report",
        )
        artifact = {
            "media_type": "application/json",
            "content": report.model_dump(mode="json"),
            "evidence_reference_ids": [],
            "warnings": [],
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                state["gets"].append((self.path, dict(self.headers)))
                if self.path != "/v1/models?limit=1000":
                    self._send(404, {"error": {"type": "not_found_error"}})
                    return
                self._send(
                    200,
                    {
                        "data": [
                            {
                                "id": "claude-loopback",
                                "type": "model",
                                "capabilities": {
                                    "structured_outputs": {"supported": True}
                                },
                            }
                        ],
                        "has_more": False,
                    },
                )

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                state["posts"].append(
                    (self.path, dict(self.headers), body)
                )
                if self.path != "/v1/messages":
                    self._send(404, {"error": {"type": "not_found_error"}})
                    return
                self._send(
                    200,
                    {
                        "id": "msg-loopback",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-loopback",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(artifact),
                            }
                        ],
                        "stop_reason": "end_turn",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                        },
                    },
                )

            def _send(self, status: int, body: dict[str, Any]) -> None:
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            if not isinstance(host, str):
                raise AssertionError("loopback server address is not text")
            target = AnthropicAPITargetDefinition(
                target_id="research/anthropic-loopback",
                model_id="claude-loopback",
                features=BackendTransportFeatures(
                    structured_output=StructuredOutputLevel.JSON_SCHEMA,
                ),
                supported_cognitive_capability_ids=(
                    "artifact_generation",
                ),
                config=AnthropicMessagesConfig(
                    base_url=f"http://{host}:{port}/v1",
                    api_key_env="AAR_ANTHROPIC_LOOPBACK_KEY",
                ),
            )
            deployment = build_research_llm_deployment(
                ResearchLLMDeploymentConfig(
                    target=target,
                    enabled_capabilities=(
                        ResearchLLMCapability.GENERATION,
                    ),
                )
            )

            with patch.dict(
                "os.environ",
                {
                    "AAR_ANTHROPIC_LOOPBACK_KEY": "loopback-secret",
                    "NO_PROXY": "127.0.0.1,localhost",
                },
                clear=False,
            ):
                probe = await deployment.probe()
                result = await ResearchAgent(
                    cognitive_capabilities=deployment.cognitive_capabilities,
                ).run("分析 Tesla 投资价值")
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await asyncio.to_thread(thread.join, 5.0)

        self.assertIs(probe.availability, BackendAvailability.AVAILABLE)
        self.assertEqual(
            result.runtime_result.final_state.status, RunStatus.COMPLETED
        )
        self.assertEqual(
            result.report.markdown,
            "# Tesla Real Anthropic HTTP Report",
        )
        self.assertEqual(len(state["gets"]), 1)
        self.assertEqual(len(state["posts"]), 1)
        get_path, get_headers = state["gets"][0]
        self.assertEqual(get_path, "/v1/models?limit=1000")
        lowered_get_headers = {
            key.lower(): value for key, value in get_headers.items()
        }
        self.assertEqual(
            lowered_get_headers["x-api-key"], "loopback-secret"
        )
        post_path, post_headers, post_body = state["posts"][0]
        self.assertEqual(post_path, "/v1/messages")
        lowered_post_headers = {
            key.lower(): value for key, value in post_headers.items()
        }
        self.assertEqual(
            lowered_post_headers["anthropic-version"], "2023-06-01"
        )
        self.assertNotIn("authorization", lowered_post_headers)
        self.assertEqual(post_body["model"], "claude-loopback")
        self.assertEqual(
            post_body["output_config"]["format"]["type"],
            "json_schema",
        )
        self.assertIsInstance(post_body["system"], str)
        self.assertEqual(len(result.llm_context_packages), 1)


if __name__ == "__main__":
    unittest.main()
