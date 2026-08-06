from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from applications.research_agent.agent import ResearchAgent as RuntimeResearchAgent
from tests.runtime_database import register_test_resource, runtime_database_path


def ResearchAgent(**kwargs: object) -> RuntimeResearchAgent:
    kwargs.setdefault("persistence_path", runtime_database_path())
    kwargs.setdefault("run_kind", "test")
    kwargs.setdefault("disposable", True)
    return register_test_resource(
        RuntimeResearchAgent(**kwargs)  # type: ignore[arg-type]
    )
from applications.research_agent.web import (
    ResearchConsoleApplication,
    ResearchConsoleServer,
    build_research_view,
)
from applications.research_agent.web_llm import WebLLMSettings
from adaptive_agent_runtime.llm import (
    BackendAvailability,
    BackendProbeResult,
)


TEST_LLM_CONFIG = """
schema_version = 1

[llm]
active_target = "deepseek-test"

[llm.targets.deepseek-test]
service = "deepseek"
model = "deepseek-chat"
structured_output = "json_object"
target_id = "research/deepseek/deepseek-chat"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"
api_probe = "models_endpoint"
capabilities = ["generation"]
context_tokens = 4096
max_attempts = 1

[llm.targets.codex-test]
service = "codex_cli"
executable = "codex"
model = "gpt-5.6-terra"
structured_output = "json_schema"
target_id = "research/codex-cli/gpt-5.6-terra"
capabilities = ["generation"]
context_tokens = 4096
max_attempts = 1
""".strip()


class ResearchWebViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_view_projects_runtime_result_without_changing_it(self) -> None:
        result = await ResearchAgent().run("分析 Tesla 投资价值")

        view = build_research_view(
            result,
            task="分析 Tesla 投资价值",
            elapsed_seconds=1.25,
            inference_label="Test Runtime",
            tool_intent_enabled=True,
        )

        self.assertEqual(view["meta"]["status"], "completed")
        self.assertEqual(view["meta"]["inference"], "Test Runtime")
        self.assertEqual(view["meta"]["executionMode"], "fixture_demo")
        self.assertIn("Context & Memory", view["meta"]["fixtureNotice"])
        self.assertEqual(view["summary"]["nodes"], 8)
        self.assertEqual(view["summary"]["completedNodes"], 8)
        self.assertEqual(
            len(view["runtimeTrace"]),
            len(result.runtime_trace),
        )
        self.assertEqual(
            sum(1 for node in view["graph"]["nodes"] if node["dynamic"]),
            1,
        )
        self.assertEqual(view["report"]["company"], "Tesla")
        self.assertTrue(view["context"]["assemblies"])
        self.assertTrue(view["memories"])
        self.assertTrue(view["governance"])
        self.assertEqual(
            view["llm"]["candidateTools"][0]["capabilityId"],
            "information_retrieval",
        )
        self.assertTrue(view["llm"]["toolIntentEnabled"])
        self.assertEqual(view["llm"]["toolIntentRecords"], [])


class ResearchWebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ResearchConsoleServer(
            ("127.0.0.1", 0),
            ResearchConsoleApplication(),
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        host_text = host.decode("ascii") if isinstance(host, bytes) else host
        self.base_url = f"http://{host_text}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_assets_health_and_research_endpoint(self) -> None:
        with urlopen(f"{self.base_url}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("Research Operations Console", html)
        self.assertIn("LLM Research", html)
        self.assertIn("Fixture Demo", html)

        with urlopen(f"{self.base_url}/api/health", timeout=5) as response:
            health = json.load(response)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["inference"], "Deterministic Runtime")

        request = Request(
            f"{self.base_url}/api/research",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"task": "分析 Tesla 投资价值"}).encode("utf-8"),
        )
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        self.assertEqual(result["meta"]["status"], "completed")
        self.assertEqual(result["meta"]["executionMode"], "fixture_demo")
        self.assertEqual(result["summary"]["completedNodes"], 8)
        self.assertFalse(result["llm"]["toolIntentEnabled"])
        self.assertEqual(result["llm"]["candidateTools"], [])

    def test_empty_task_is_rejected(self) -> None:
        request = Request(
            f"{self.base_url}/api/research",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b'{"task": ""}',
        )

        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)

        self.assertEqual(caught.exception.code, 400)

    def test_sse_stream_precedes_and_matches_terminal_result(self) -> None:
        request = Request(
            f"{self.base_url}/api/runs",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"task": "分析 Tesla 投资价值"}).encode("utf-8"),
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 202)
            session = json.load(response)

        events: list[dict[str, object]] = []
        with urlopen(f"{self.base_url}{session['eventsUrl']}", timeout=10) as stream:
            for raw_line in stream:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                events.append(event)
                if event["kind"] in {"run.result_ready", "run.failed"}:
                    break

        kinds = [event["kind"] for event in events]
        self.assertEqual(kinds[0], "run.preparing")
        self.assertIn("graph.initialized", kinds)
        self.assertIn("node.started", kinds)
        self.assertIn("node.completed", kinds)
        self.assertIn("graph.node_added", kinds)
        self.assertEqual(kinds[-1], "run.result_ready")
        self.assertEqual(
            [event["streamSequence"] for event in events],
            list(range(1, len(events) + 1)),
        )

        with urlopen(f"{self.base_url}{session['resultUrl']}", timeout=5) as response:
            result = json.load(response)
        self.assertEqual(result["meta"]["status"], "completed")
        self.assertEqual(result["summary"]["completedNodes"], 8)


class WebLLMSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary.name) / "llm.toml"
        self.config_path.write_text(TEST_LLM_CONFIG, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_private_key_is_persisted_but_never_returned(self) -> None:
        settings = WebLLMSettings(self.config_path)

        settings.activate("deepseek-test", api_key="private-browser-secret")
        view = settings.describe()

        private_text = (Path(self.temporary.name) / "llm.local.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("private-browser-secret", private_text)
        self.assertNotIn("private-browser-secret", json.dumps(view))
        self.assertEqual(view["activeTarget"], "deepseek-test")
        target = next(
            item
            for item in view["targets"]  # type: ignore[union-attr]
            if item["name"] == "deepseek-test"
        )
        self.assertTrue(target["credentialConfigured"])
        self.assertEqual(target["credentialSource"], "llm.local.toml")

    def test_cli_target_uses_oauth_without_private_key(self) -> None:
        settings = WebLLMSettings(self.config_path)

        config = settings.activate("codex-test")
        view = settings.describe()

        self.assertEqual(config.target.model_id, "gpt-5.6-terra")
        self.assertEqual(view["activeTarget"], "codex-test")
        target = next(
            item
            for item in view["targets"]  # type: ignore[union-attr]
            if item["name"] == "codex-test"
        )
        self.assertFalse(target["requiresApiKey"])
        self.assertEqual(target["credentialSource"], "oauth_cli")

    def test_dynamic_model_selection_uses_runtime_private_repository(self) -> None:
        settings = WebLLMSettings(self.config_path)

        config = settings.activate(
            "codex-test",
            model_id="gpt-5.6-sol",
        )
        view = settings.describe()

        self.assertEqual(config.target.model_id, "gpt-5.6-sol")
        target = next(
            item
            for item in view["targets"]  # type: ignore[union-attr]
            if item["name"] == "codex-test"
        )
        self.assertEqual(target["model"], "gpt-5.6-sol")
        private_text = (Path(self.temporary.name) / "llm.local.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("[selections.codex-test]", private_text)
        self.assertIn('model = "gpt-5.6-sol"', private_text)


class ResearchWebLLMSettingsServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        config_path = Path(self.temporary.name) / "llm.toml"
        config_path.write_text(TEST_LLM_CONFIG, encoding="utf-8")
        settings = WebLLMSettings(config_path)
        self.server = ResearchConsoleServer(
            ("127.0.0.1", 0),
            ResearchConsoleApplication(llm_settings=settings),
        )
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        host_text = host.decode("ascii") if isinstance(host, bytes) else host
        self.base_url = f"http://{host_text}:{port}"
        self.server_host = host_text
        self.server_port = port

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_browser_can_save_and_activate_sanitized_llm_settings(self) -> None:
        with urlopen(f"{self.base_url}/api/llm/settings", timeout=5) as response:
            initial = json.load(response)
        self.assertIsNone(initial["activeTarget"])

        request = Request(
            f"{self.base_url}/api/llm/settings",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "target": "deepseek-test",
                    "apiKey": "private-http-secret",
                }
            ).encode("utf-8"),
        )
        with urlopen(request, timeout=5) as response:
            configured = json.load(response)

        serialized = json.dumps(configured)
        self.assertNotIn("private-http-secret", serialized)
        self.assertEqual(configured["activeTarget"], "deepseek-test")
        self.assertIn("deepseek-chat", configured["inference"])
        private_text = (Path(self.temporary.name) / "llm.local.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("private-http-secret", private_text)

    def test_browser_receives_live_models_and_switches_selection(self) -> None:
        class FakeDeployment:
            target_id = "research/codex-cli/gpt-5.6-terra"
            model_id = "gpt-5.6-terra"

            async def probe(self) -> BackendProbeResult:
                return BackendProbeResult(
                    target_id=self.target_id,
                    availability=BackendAvailability.AVAILABLE,
                    runtime_version="0.146.0",
                    active_auth_method="chatgpt_oauth",
                    available_model_ids=("gpt-5.6-terra", "gpt-5.6-sol"),
                    diagnostics=("connectivity_and_model_verified",),
                )

        models_request = Request(
            f"{self.base_url}/api/llm/models",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"target": "codex-test"}).encode("utf-8"),
        )
        with patch(
            "applications.research_agent.web.build_research_llm_deployment",
            return_value=FakeDeployment(),
        ):
            with urlopen(models_request, timeout=5) as response:
                models = json.load(response)

        self.assertEqual(
            models["models"],
            ["gpt-5.6-terra", "gpt-5.6-sol"],
        )
        self.assertEqual(models["authMethod"], "chatgpt_oauth")

        settings_request = Request(
            f"{self.base_url}/api/llm/settings",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {"target": "codex-test", "model": "gpt-5.6-sol"}
            ).encode("utf-8"),
        )
        with urlopen(settings_request, timeout=5) as response:
            configured = json.load(response)

        self.assertIn("gpt-5.6-sol", configured["inference"])
        selected = next(
            target
            for target in configured["targets"]
            if target["name"] == "codex-test"
        )
        self.assertEqual(selected["model"], "gpt-5.6-sol")

    def test_llm_settings_reject_non_local_host_header(self) -> None:
        connection = HTTPConnection(self.server_host, self.server_port, timeout=5)
        connection.putrequest("GET", "/api/llm/settings", skip_host=True)
        connection.putheader("Host", "untrusted.example")
        connection.endheaders()

        response = connection.getresponse()
        response.read()
        connection.close()

        self.assertEqual(response.status, 403)

if __name__ == "__main__":
    unittest.main()
