from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from applications.research_agent.agent import ResearchAgent


class ResearchDefaultSQLiteCompositionTests(unittest.TestCase):
    def test_default_constructor_uses_one_stable_persistent_database_identity(self) -> None:
        first = ResearchAgent()
        second = ResearchAgent()
        try:
            first_identity = first.persistence_identity
            second_identity = second.persistence_identity
            self.assertTrue(first_identity.is_persistent)
            self.assertNotEqual(first_identity.database_path, ":memory:")
            self.assertTrue(Path(first_identity.database_path).is_absolute())
            self.assertEqual(
                first_identity.database_path,
                second_identity.database_path,
            )
            adapter_paths = dict(first_identity.adapter_database_paths)
            self.assertEqual(
                set(adapter_paths),
                {
                    "state",
                    "graph",
                    "trace",
                    "decision",
                    "context",
                    "archive",
                    "memory",
                    "review",
                    "authorization",
                    "artifact",
                    "recall_bundle",
                    "experience",
                    "evaluation",
                    "decision_feedback",
                    "learning_insight",
                    "runtime_configuration",
                    "auto_adaptation",
                },
            )
            self.assertEqual(set(adapter_paths.values()), {first_identity.database_path})
            connection = sqlite3.connect(first_identity.database_path)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            connection.close()
            self.assertTrue(
                {
                    "agent_state_snapshots",
                    "task_graph_checkpoints",
                    "runtime_trace",
                    "decision_checkpoints",
                    "context_snapshots",
                    "context_archives",
                    "memory_snapshots",
                    "governance_reviews",
                    "governance_authorization_uses",
                    "workspace_artifacts",
                    "memory_recall_bundles",
                    "experience_metadata",
                    "experience_memory_links",
                    "evaluation_reports",
                    "decision_feedback",
                    "learning_insights",
                    "governed_runtime_configuration_snapshots",
                    "governed_runtime_configuration_active",
                    "optimization_configuration_receipts",
                    "auto_adaptation_triggers",
                    "auto_adaptation_trigger_history",
                }.issubset(tables)
            )
        finally:
            first.close()
            second.close()

    def test_default_database_identity_is_stable_across_processes(self) -> None:
        script = (
            "import json; from applications.research_agent.agent import ResearchAgent; "
            "a=ResearchAgent(); print(json.dumps(a.persistence_identity.__dict__)); a.close()"
        )
        paths = []
        for _ in range(2):
            process = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=True,
            )
            paths.append(json.loads(process.stdout)["database_path"])
        self.assertEqual(paths[0], paths[1])
        self.assertNotEqual(paths[0], ":memory:")

    def test_sqlite_initialization_failure_does_not_fall_back_to_memory(self) -> None:
        with TemporaryDirectory() as directory:
            blocking_file = Path(directory) / "not-a-directory"
            blocking_file.write_text("block", encoding="utf-8")
            with self.assertRaises((FileExistsError, OSError, sqlite3.Error)):
                ResearchAgent(persistence_path=blocking_file / "runtime.sqlite3")


if __name__ == "__main__":
    unittest.main()
