from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from applications.personal_knowledge.web import (
    PersonalKnowledgeApplication,
    build_http_server,
)
from applications.personal_knowledge.llm_runtime import PersonalKnowledgeLLMManager


class PersonalKnowledgeWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        llm_config = root / "llm.toml"
        shutil.copyfile(
            Path(__file__).resolve().parents[2] / "config" / "llm.toml",
            llm_config,
        )
        self.app = PersonalKnowledgeApplication(
            root / "knowledge.sqlite3",
            runtime_database_path=root / "runtime.sqlite3",
            llm_manager=PersonalKnowledgeLLMManager(llm_config),
        )
        self.server = build_http_server(self.app, port=0)
        host, port = self.server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.app.close()
        self.temp.cleanup()

    def test_editable_confirmation_dialog_is_real_end_to_end_boundary(self) -> None:
        with httpx.Client(base_url=self.base_url, timeout=5) as client:
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("最终入库内容", page.text)
            self.assertIn("点击前不会写入知识库", page.text)
            self.assertIn('<select id="llm-model">', page.text)
            self.assertIn("测试并激活", page.text)
            self.assertNotIn("激活此模型", page.text)
            self.assertNotIn('id="probe-model"', page.text)
            self.assertNotIn("llm-model-list", page.text)

            category_response = client.post(
                "/api/categories/confirm",
                json={
                    "confirmed": True,
                    "change_type": "create",
                    "name": "认知与决策",
                },
            )
            self.assertEqual(category_response.status_code, 201)
            category_id = category_response.json()["category"]["category_id"]

            capture = client.post(
                "/api/ingest",
                json={"kind": "idea", "value": "确认之后才成为知识"},
            )
            self.assertEqual(capture.status_code, 201)
            session = capture.json()
            state_before = client.get("/api/state").json()
            self.assertEqual(state_before["entries"], [])

            review = session["review"]
            proposal = session["proposal"]
            confirmed = client.post(
                f"/api/reviews/{review['review_id']}/confirm",
                json={
                    "proposal_id": proposal["proposal_id"],
                    "expected_review_revision": review["revision"],
                    "title": "经我确认的知识",
                    "category_id": category_id,
                    "body": "这是用户在确认界面编辑后的最终正文。",
                    "tags": ["确认边界"],
                    "citations": proposal["citations"],
                },
            )
            self.assertEqual(confirmed.status_code, 201)
            state_after = client.get("/api/state").json()
            self.assertEqual(len(state_after["entries"]), 1)
            self.assertEqual(
                state_after["entries"][0]["document"]["body"],
                "这是用户在确认界面编辑后的最终正文。",
            )
            self.assertEqual(state_after["reviews"], [])

    def test_model_targets_can_be_selected_without_restarting(self) -> None:
        with httpx.Client(base_url=self.base_url, timeout=5) as client:
            settings = client.get("/api/llm/settings")
            self.assertEqual(settings.status_code, 200)
            payload = settings.json()
            self.assertTrue(payload["enabled"])
            self.assertGreaterEqual(len(payload["targets"]), 1)
            target = payload["targets"][0]
            self.assertNotIn("apiKey", target)

            activated = client.post(
                "/api/llm/settings",
                json={
                    "target": target["name"],
                    "model": target["model"],
                    "apiKey": None,
                },
            )
            self.assertEqual(activated.status_code, 200)
            self.assertEqual(activated.json()["activeTarget"], target["name"])
            self.assertNotEqual(activated.json()["inference"], "offline")

    def test_successful_probe_activates_the_selected_form_model(self) -> None:
        with httpx.Client(base_url=self.base_url, timeout=5) as client:
            settings = client.get("/api/llm/settings").json()
            target = next(
                item for item in settings["targets"]
                if item["service"] == "codex_cli"
            )
            manager = self.app.llm_manager
            assert manager is not None
            candidate = manager.activate(
                target["name"],
                model_id=target["model"],
            )
            probe_result = {
                "availability": "available",
                "target": target["name"],
                "targetId": candidate.target_id,
                "model": candidate.model_id,
                "availableModels": [candidate.model_id, "gpt-alternate"],
                "runtimeVersion": "test",
                "authMethod": "oauth_cli",
                "diagnostics": [],
            }
            with patch.object(
                manager,
                "probe_selection",
                AsyncMock(return_value=(probe_result, candidate)),
            ):
                response = client.post(
                    "/api/llm/probe",
                    json={
                        "target": target["name"],
                        "model": target["model"],
                        "apiKey": None,
                    },
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["activated"])
            self.assertEqual(payload["settings"]["activeTarget"], target["name"])
            self.assertNotEqual(payload["inference"], "offline")

    def test_source_tool_failure_returns_json_instead_of_dropping_fetch(self) -> None:
        with httpx.Client(base_url=self.base_url, timeout=5) as client:
            terminal = StringIO()
            with redirect_stderr(terminal):
                response = client.post(
                    "/api/ingest",
                    json={"kind": "url", "value": "http://127.0.0.1/private"},
                )
            self.assertEqual(response.status_code, 502)
            self.assertIn("error", response.json())
            self.assertIn("ToolIntegrationError", terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
