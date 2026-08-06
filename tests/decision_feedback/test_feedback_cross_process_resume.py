from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "tests.decision_feedback.process_feedback_resume_worker"


class DecisionFeedbackCrossProcessResumeTests(unittest.TestCase):
    def test_feedback_commit_resume_in_new_process_does_not_duplicate(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            marker = Path(directory) / "run-id.txt"
            crashed = subprocess.run(
                [sys.executable, "-m", WORKER_MODULE, "crash", str(database), str(marker)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(crashed.returncode, 73, crashed.stderr)
            self.assertTrue(marker.exists())

            resumed = subprocess.run(
                [sys.executable, "-m", WORKER_MODULE, "resume", str(database), str(marker)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(resumed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["feedback_count"], 1)

            connection = sqlite3.connect(database)
            try:
                record_count = connection.execute(
                    "SELECT COUNT(*) FROM decision_feedback WHERE source_run_id = ?",
                    (payload["run_id"],),
                ).fetchone()[0]
                feedback_checkpoints = [
                    json.loads(row[0])
                    for row in connection.execute(
                        "SELECT checkpoint_json FROM decision_checkpoints "
                        "WHERE checkpoint_json LIKE '%decision.outcome_feedback%'"
                    ).fetchall()
                ]
            finally:
                connection.close()
            self.assertEqual(record_count, 1)
            current = max(feedback_checkpoints, key=lambda item: item["revision"])
            self.assertEqual(current["stage"], "completed")
            self.assertEqual(current["result"]["status"], "applied")
            self.assertEqual(
                current["result"]["apply_receipt"]["effect_fingerprint"],
                payload["effect_fingerprints"][0],
            )


if __name__ == "__main__":
    unittest.main()
