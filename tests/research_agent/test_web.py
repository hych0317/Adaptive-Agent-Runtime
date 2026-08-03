from __future__ import annotations

import json
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from applications.research_agent.agent import ResearchAgent
from applications.research_agent.web import (
    ResearchConsoleApplication,
    ResearchConsoleServer,
    build_research_view,
)


class ResearchWebViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_view_projects_runtime_result_without_changing_it(self) -> None:
        result = await ResearchAgent().run("分析 Tesla 投资价值")

        view = build_research_view(
            result,
            task="分析 Tesla 投资价值",
            elapsed_seconds=1.25,
            inference_label="Test Runtime",
        )

        self.assertEqual(view["meta"]["status"], "completed")
        self.assertEqual(view["meta"]["inference"], "Test Runtime")
        self.assertEqual(view["summary"]["nodes"], 8)
        self.assertEqual(view["summary"]["completedNodes"], 8)
        self.assertEqual(len(view["runtimeTrace"]), 36)
        self.assertEqual(
            sum(1 for node in view["graph"]["nodes"] if node["dynamic"]),
            1,
        )
        self.assertEqual(view["report"]["company"], "Tesla")
        self.assertTrue(view["context"]["assemblies"])
        self.assertTrue(view["memories"])
        self.assertTrue(view["governance"])


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
        self.assertEqual(result["summary"]["completedNodes"], 8)

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


if __name__ == "__main__":
    unittest.main()
