from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "tests.experience_learning.process_learning_resume_worker"


class LearningCrossProcessResumeTests(unittest.TestCase):
    def test_learning_resume_does_not_call_agent_again(self) -> None:
        for fault_point in ("authorized", "applying", "effect_committed"):
            with self.subTest(fault_point=fault_point), TemporaryDirectory() as directory:
                database = Path(directory) / "runtime.sqlite3"
                marker = Path(directory) / "run-id.txt"
                crashed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        WORKER_MODULE,
                        "crash",
                        str(database),
                        str(marker),
                        fault_point,
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(crashed.returncode, 73, crashed.stderr)
                resumed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        WORKER_MODULE,
                        "resume",
                        str(database),
                        str(marker),
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                payload = json.loads(resumed.stdout.strip().splitlines()[-1])
                self.assertEqual(payload["invocation_count"], 1)
                self.assertEqual(payload["insight_count"], 1)
                connection = sqlite3.connect(database)
                try:
                    rows = connection.execute(
                        "SELECT learning_insight_id, effect_fingerprint, "
                        "evidence_set_fingerprint FROM learning_insights"
                    ).fetchall()
                    checkpoints = tuple(
                        json.loads(row[0])
                        for row in connection.execute(
                            "SELECT checkpoint_json FROM decision_checkpoints "
                            "WHERE checkpoint_json LIKE "
                            "'%experience.learning.assessment%'"
                        ).fetchall()
                    )
                finally:
                    connection.close()
                self.assertEqual(rows, [(
                    payload["learning_insight_id"],
                    payload["effect_fingerprint"],
                    payload["evidence_set_fingerprint"],
                )])
                current = max(checkpoints, key=lambda item: item["revision"])
                self.assertEqual(current["stage"], "completed")
                self.assertEqual(current["result"]["status"], "applied")
                self.assertEqual(
                    current["validated_decision"]["normalized_effect"]
                    ["effect_fingerprint"],
                    payload["effect_fingerprint"],
                )


if __name__ == "__main__":
    unittest.main()
