"""Architecture boundary checks for horizontal Runtime Governance."""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from adaptive_agent_runtime.governance import (
    GovernanceAuthorizationIssuer,
    RuntimeGovernanceEvaluator,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "adaptive_agent_runtime"
EXECUTION_PACKAGES = (
    "core",
    "orchestration",
    "context_memory",
    "tool_ecosystem",
    "evaluation",
    "decisioning",
)


def imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


class GovernanceBoundaryTests(unittest.TestCase):
    def test_runtime_modules_only_import_governance_commit_contracts(self) -> None:
        allowed = {
            "adaptive_agent_runtime.governance.contracts",
            "adaptive_agent_runtime.governance.models",
            "adaptive_agent_runtime.governance.errors",
        }
        violations: list[str] = []
        for package_name in EXECUTION_PACKAGES:
            for path in (PACKAGE_ROOT / package_name).rglob("*.py"):
                for module in imports(path):
                    if (
                        module.startswith("adaptive_agent_runtime.governance")
                        and module not in allowed
                    ):
                        violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_governance_algorithms_do_not_import_execution_implementations(
        self,
    ) -> None:
        governance_root = PACKAGE_ROOT / "governance"
        allowed_upstream = {
            "contracts.py": {"adaptive_agent_runtime.core.contracts"},
        }
        upstream_prefixes = tuple(
            f"adaptive_agent_runtime.{name}" for name in EXECUTION_PACKAGES
        )
        violations: list[str] = []
        for path in governance_root.glob("*.py"):
            if path.name in {"integration.py", "decisioning.py", "__init__.py"}:
                continue
            allowed = allowed_upstream.get(path.name, set())
            for module in imports(path):
                if module.startswith(upstream_prefixes) and module not in allowed:
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_integration_imports_only_upstream_dtos(self) -> None:
        integration_imports = set(
            imports(PACKAGE_ROOT / "governance" / "integration.py")
        )
        forbidden = {
            "adaptive_agent_runtime.core.runtime",
            "adaptive_agent_runtime.orchestration.graph",
            "adaptive_agent_runtime.orchestration.planner",
            "adaptive_agent_runtime.context_memory.context_runtime",
            "adaptive_agent_runtime.context_memory.memory_runtime",
            "adaptive_agent_runtime.tool_ecosystem.execution",
            "adaptive_agent_runtime.tool_ecosystem.governance",
            "adaptive_agent_runtime.evaluation.analyzer",
            "adaptive_agent_runtime.evaluation.optimization",
        }

        self.assertEqual(integration_imports & forbidden, set())
        self.assertNotIn(
            "adaptive_agent_runtime.evaluation.models",
            integration_imports,
        )

    def test_concrete_governance_services_do_not_apply_changes(self) -> None:
        evaluator_methods = {
            name
            for name, member in inspect.getmembers(
                RuntimeGovernanceEvaluator,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        issuer_methods = {
            name
            for name, member in inspect.getmembers(
                GovernanceAuthorizationIssuer,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }

        self.assertEqual(evaluator_methods, {"evaluate", "finalize_review"})
        self.assertEqual(issuer_methods, {"issue"})

    def test_applications_do_not_import_raw_authoritative_mutation_implementations(
        self,
    ) -> None:
        forbidden_symbols = {
            "ManagedToolExecutor",
            "SQLiteWorkspaceArtifactStore",
            "SQLiteMemoryStore",
            "SQLiteMemoryRecallBundleStore",
            "SQLiteExperienceMetadataStore",
            "SQLiteDecisionFeedbackStore",
            "SQLiteLearningInsightStore",
            "SQLiteContextArchive",
            "SQLiteTaskGraphStore",
        }
        violations: list[str] = []
        root = Path(__file__).resolve().parents[2] / "applications"
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    blocked = sorted(
                        {alias.name for alias in node.names} & forbidden_symbols
                    )
                    if blocked:
                        violations.append(
                            f"{path.relative_to(root)} imports {', '.join(blocked)}"
                        )
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
