from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class MemoryRecallCrossProcessResumeTests(unittest.TestCase):
    def worker(self, database: Path, action: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.context_memory.process_memory_recall_resume_worker",
                "--database",
                str(database),
                "--action",
                action,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def facts(database: Path) -> dict:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            checkpoint = connection.execute(
                "SELECT stage, result_status, proposal_ref, effect_ref, "
                "authorization_id FROM decision_current "
                "WHERE decision_type = 'memory.recall'"
            ).fetchone()
            calls = connection.execute(
                "SELECT calls FROM closeout_recall_invocations WHERE singleton = 1"
            ).fetchone()
            bundles = connection.execute(
                "SELECT bundle_json FROM memory_recall_bundles"
            ).fetchall()
            memories = connection.execute(
                "SELECT snapshot_json FROM memory_snapshots ORDER BY memory_id, revision"
            ).fetchall()
        finally:
            connection.close()
        assert checkpoint is not None and calls is not None
        return {
            "checkpoint": dict(checkpoint),
            "calls": int(calls["calls"]),
            "bundles": tuple(row["bundle_json"] for row in bundles),
            "memories": tuple(row["snapshot_json"] for row in memories),
        }

    def test_recall_commit_resume_does_not_call_agent_again(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            crashed = self.worker(database, "start")
            self.assertEqual(crashed.returncode, 93, crashed.stderr)
            before = self.facts(database)
            self.assertEqual(before["calls"], 1)
            self.assertEqual(len(before["bundles"]), 1)

            resumed = self.worker(database, "resume")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            after = self.facts(database)
            self.assertEqual(after["calls"], 1)
            self.assertEqual(after["bundles"], before["bundles"])
            self.assertEqual(after["memories"], before["memories"])
            self.assertEqual(after["checkpoint"]["stage"], "completed")
            self.assertEqual(after["checkpoint"]["result_status"], "applied")
            self.assertEqual(
                after["checkpoint"]["proposal_ref"],
                before["checkpoint"]["proposal_ref"],
            )
            self.assertEqual(
                after["checkpoint"]["effect_ref"],
                before["checkpoint"]["effect_ref"],
            )
            self.assertEqual(
                after["checkpoint"]["authorization_id"],
                before["checkpoint"]["authorization_id"],
            )

            repeated = self.worker(database, "resume")
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            final = self.facts(database)
            self.assertEqual(final["calls"], 1)
            self.assertEqual(final["bundles"], before["bundles"])

    def test_committed_recall_bundle_is_reused_without_duplicate_commit(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            self.assertEqual(self.worker(database, "start").returncode, 93)
            self.assertEqual(self.worker(database, "resume").returncode, 0)
            self.assertEqual(self.worker(database, "resume").returncode, 0)
            facts = self.facts(database)
            self.assertEqual(len(facts["bundles"]), 1)
            self.assertEqual(facts["calls"], 1)


if __name__ == "__main__":
    unittest.main()
