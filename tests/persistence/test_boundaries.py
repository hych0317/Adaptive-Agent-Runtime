from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from adaptive_agent_runtime import StateStore, TraceSink
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionCheckpointStore,
)
from adaptive_agent_runtime.context_memory import (
    ContextArchive,
    ContextStore,
    MemoryStore,
)
from adaptive_agent_runtime.orchestration import TaskGraphStore
from adaptive_agent_runtime.governance import (
    AuthorizationConsumptionStore,
    HumanReviewService,
)
from adaptive_agent_runtime.persistence import SQLitePersistence


SOURCE_ROOT = Path(__file__).parents[2] / "src" / "adaptive_agent_runtime"
PROJECT_ROOT = Path(__file__).parents[2]


class PersistenceBoundaryTests(unittest.TestCase):
    def test_runtime_modules_do_not_depend_on_persistence_implementations(
        self,
    ) -> None:
        for package in (
            "core",
            "orchestration",
            "context_memory",
            "decisioning",
            "evolution",
        ):
            for path in (SOURCE_ROOT / package).glob("*.py"):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "adaptive_agent_runtime.persistence",
                    source,
                    f"{path.name} imports a concrete persistence adapter",
                )

    def test_sqlite_adapters_implement_existing_narrow_contracts(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(f"{directory}/runtime.sqlite3")
            self.assertIsInstance(persistence.state_store, StateStore)
            self.assertIsInstance(persistence.trace_sink, TraceSink)
            self.assertIsInstance(persistence.task_graph_store, TaskGraphStore)
            self.assertIsInstance(persistence.context_store, ContextStore)
            self.assertIsInstance(persistence.context_archive, ContextArchive)
            self.assertIsInstance(persistence.memory_store, MemoryStore)
            self.assertIsInstance(
                persistence.human_review_service,
                HumanReviewService,
            )
            self.assertIsInstance(
                persistence.authorization_store,
                AuthorizationConsumptionStore,
            )
            self.assertFalse(hasattr(persistence, "runtime_configuration_store"))
            self.assertFalse(hasattr(persistence, "replay_case_store"))
            self.assertFalse(hasattr(persistence, "evolution_store"))
            self.assertTrue(
                hasattr(persistence.optimization_proposal_store, "commit")
            )
            self.assertIsInstance(
                persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint
                ),
                DecisionCheckpointStore,
            )
            persistence.close()

    def test_decision_consumers_use_the_proof_query_boundary(self) -> None:
        roots = (
            PROJECT_ROOT / "applications",
            SOURCE_ROOT / "persistence",
        )
        for root in roots:
            for path in root.rglob("*.py"):
                if path.name == "decisioning.py" and root == SOURCE_ROOT / "persistence":
                    continue
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("decision_checkpoints", source, str(path))
                self.assertNotIn("decision_checkpoint_current", source, str(path))
                self.assertNotIn("checkpoint_json LIKE", source, str(path))
        for path in (PROJECT_ROOT / "applications").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("SQLiteDecisionRecordReader", source, str(path))


if __name__ == "__main__":
    unittest.main()
