from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class ReportCrossProcessResumeTests(unittest.TestCase):
    def _worker(self, database: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.research_agent.process_report_resume_worker",
                "--database",
                str(database),
                *args,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_report_commit_crash_resumes_without_agent_or_duplicate_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            crashed = self._worker(database, "--action", "start")
            self.assertEqual(crashed.returncode, 92, crashed.stderr)
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                run_row = connection.execute(
                    "SELECT run_id FROM application_run_manifests"
                ).fetchone()
                assert run_row is not None
                run_id = str(run_row["run_id"])
                before = connection.execute(
                    "SELECT effect_fingerprint, artifact_json, receipt_json "
                    "FROM workspace_artifacts"
                ).fetchone()
                assert before is not None
                calls = connection.execute(
                    "SELECT calls FROM closeout_report_invocations WHERE singleton = 1"
                ).fetchone()
                assert calls is not None
                self.assertEqual(int(calls["calls"]), 1)
            finally:
                connection.close()

            resumed = self._worker(
                database,
                "--action",
                "resume",
                "--run-id",
                run_id,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            result = json.loads(resumed.stdout)
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                artifacts = connection.execute(
                    "SELECT effect_fingerprint, artifact_json, receipt_json "
                    "FROM workspace_artifacts"
                ).fetchall()
                calls = connection.execute(
                    "SELECT calls FROM closeout_report_invocations WHERE singleton = 1"
                ).fetchone()
                report_decision = connection.execute(
                    "SELECT stage, result_status FROM decision_current "
                    "WHERE decision_type = 'artifact.report_commit'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["effect_fingerprint"], before["effect_fingerprint"])
            self.assertEqual(artifacts[0]["artifact_json"], before["artifact_json"])
            self.assertEqual(artifacts[0]["receipt_json"], before["receipt_json"])
            assert calls is not None
            self.assertEqual(int(calls["calls"]), 1)
            self.assertEqual(result["effect_fingerprint"], before["effect_fingerprint"])
            receipt = json.loads(before["receipt_json"])
            self.assertEqual(result["artifact_fingerprint"], receipt["artifact_fingerprint"])
            self.assertIsNotNone(receipt["source_decision_request_id"])
            self.assertIsNotNone(receipt["source_proposal_id"])
            self.assertTrue(receipt["provenance"])
            assert report_decision is not None
            self.assertEqual(report_decision["stage"], "completed")
            self.assertEqual(report_decision["result_status"], "applied")


if __name__ == "__main__":
    unittest.main()
