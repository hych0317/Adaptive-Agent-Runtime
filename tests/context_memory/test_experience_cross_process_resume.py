from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class ExperienceCrossProcessResumeTests(unittest.TestCase):
    def _worker(self, database: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.context_memory.process_experience_resume_worker",
                "--database",
                str(database),
                *args,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_experience_resume_after_commit_does_not_call_agent(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            crashed = self._worker(database, "--action", "start")
            self.assertEqual(crashed.returncode, 94, crashed.stderr)
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                manifest = connection.execute(
                    "SELECT run_id FROM application_run_manifests"
                ).fetchone()
                assert manifest is not None
                run_id = str(manifest["run_id"])
                before = connection.execute(
                    "SELECT effect_fingerprint, experience_id, metadata_json, "
                    "receipt_json FROM experience_metadata"
                ).fetchone()
                assert before is not None
                calls = connection.execute(
                    "SELECT calls FROM experience_agent_invocations WHERE singleton = 1"
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
                records = connection.execute(
                    "SELECT effect_fingerprint, experience_id, metadata_json, "
                    "receipt_json FROM experience_metadata"
                ).fetchall()
                calls = connection.execute(
                    "SELECT calls FROM experience_agent_invocations WHERE singleton = 1"
                ).fetchone()
                checkpoint_row = connection.execute(
                    "SELECT stage, result_status FROM decision_current "
                    "WHERE decision_type = 'experience.assessment'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["effect_fingerprint"], before["effect_fingerprint"])
            self.assertEqual(records[0]["experience_id"], before["experience_id"])
            self.assertEqual(records[0]["metadata_json"], before["metadata_json"])
            self.assertEqual(records[0]["receipt_json"], before["receipt_json"])
            assert calls is not None
            self.assertEqual(int(calls["calls"]), 1)
            self.assertEqual(result["experience_id"], before["experience_id"])
            self.assertEqual(
                result["effect_fingerprint"],
                before["effect_fingerprint"],
            )
            assert checkpoint_row is not None
            self.assertEqual(checkpoint_row["stage"], "completed")
            self.assertEqual(checkpoint_row["result_status"], "applied")


if __name__ == "__main__":
    unittest.main()
