from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from adaptive_agent_runtime.evolution.errors import EvolutionBoundaryError
from adaptive_agent_runtime.evolution.apply import InMemoryEvolutionStore
from adaptive_agent_runtime.evolution.models import RuntimeConfigurationSnapshot
from adaptive_agent_runtime.optimization import OptimizationTargetKey
from adaptive_agent_runtime.persistence import SQLitePersistence
from applications.research_agent.agent import ResearchAgent
from applications.research_agent.web_llm import WebLLMSettings


ROOT = Path(__file__).parents[2]
APPLICATION_ROOT = ROOT / "applications"


class Phase4ReadinessTests(unittest.TestCase):
    def test_application_cannot_access_evolution_activate_or_rollback(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            try:
                self.assertFalse(hasattr(persistence, "evolution_store"))
                self.assertFalse(hasattr(persistence, "runtime_configuration_store"))
                self.assertFalse(hasattr(persistence, "replay_case_store"))
            finally:
                persistence.close()

        for path in APPLICATION_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            self.assertFalse(
                any(
                    value.startswith("adaptive_agent_runtime.evolution")
                    or value.startswith("adaptive_agent_runtime.legacy")
                    for value in imports
                ),
                f"{path} imports a legacy Evolution mutation capability",
            )

        async def assert_disabled() -> None:
            store = InMemoryEvolutionStore()
            with self.assertRaises(EvolutionBoundaryError):
                await store.initialize(
                    RuntimeConfigurationSnapshot(
                        component="orchestration",
                        version=0,
                        config={"failure_replanning_enabled": False},
                    )
                )

        asyncio.run(assert_disabled())

    def test_web_llm_settings_are_not_runtime_optimization_targets(self) -> None:
        self.assertEqual(WebLLMSettings.configuration_owner, "operator")
        self.assertEqual(
            WebLLMSettings.configuration_domain,
            "operator.web_llm",
        )
        self.assertTrue(
            all(not item.value.startswith("web_llm") for item in OptimizationTargetKey)
        )

    def test_evaluation_and_evolution_modules_import_independently(self) -> None:
        for module in (
            "adaptive_agent_runtime.evaluation",
            "adaptive_agent_runtime.evolution",
        ):
            completed = subprocess.run(
                [sys.executable, "-c", f"import {module}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_core_event_loop_remains_unchanged(self) -> None:
        source = (
            ROOT / "src" / "adaptive_agent_runtime" / "core" / "runtime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("optimization", source.lower())
        self.assertNotIn("evolution", source.lower())


class DefaultResearchReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_research_path_does_not_use_legacy_optimization(self) -> None:
        with TemporaryDirectory() as directory:
            agent = ResearchAgent(
                persistence_path=Path(directory) / "runtime.sqlite3"
            )
            try:
                result = await agent.run_demo("分析 Tesla 投资价值")
            finally:
                agent.close()
        self.assertEqual(result.optimization_proposals, ())

    async def test_no_optimization_apply_authorization_is_created(self) -> None:
        with TemporaryDirectory() as directory:
            agent = ResearchAgent(
                persistence_path=Path(directory) / "runtime.sqlite3"
            )
            try:
                result = await agent.run("分析 Tesla 投资价值")
            finally:
                agent.close()
        self.assertFalse(
            any(
                record.request.operation in {
                    "optimization.apply",
                    "configuration.activate",
                    "optimization.rollback",
                }
                for record in result.governance_records
            )
        )


if __name__ == "__main__":
    unittest.main()
