from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

import httpx

from applications.personal_knowledge.web import (
    PersonalKnowledgeApplication,
    build_http_server,
)


class PersonalKnowledgeWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        self.app = PersonalKnowledgeApplication(
            root / "knowledge.sqlite3",
            runtime_database_path=root / "runtime.sqlite3",
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


if __name__ == "__main__":
    unittest.main()
