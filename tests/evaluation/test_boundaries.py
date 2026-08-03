"""Architectural boundary checks for the read-only Evaluation layer."""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from adaptive_agent_runtime.evaluation import ConservativeOptimizationAgent


_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "adaptive_agent_runtime"
_EXECUTION_PACKAGES = (
    "core",
    "orchestration",
    "context_memory",
    "tool_ecosystem",
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


class EvaluationBoundaryTests(unittest.TestCase):
    def test_execution_packages_do_not_depend_on_evaluation(self) -> None:
        violations: list[str] = []
        for package_name in _EXECUTION_PACKAGES:
            for path in (_PACKAGE_ROOT / package_name).rglob("*.py"):
                for module in _imports(path):
                    if module.startswith("adaptive_agent_runtime.evaluation"):
                        violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_evaluator_modules_do_not_import_execution_implementations(self) -> None:
        evaluation_root = _PACKAGE_ROOT / "evaluation"
        allowed_upstream = {
            "contracts.py": {"adaptive_agent_runtime.core.contracts"},
        }
        upstream_prefixes = tuple(
            f"adaptive_agent_runtime.{name}" for name in _EXECUTION_PACKAGES
        )
        violations: list[str] = []
        for path in evaluation_root.glob("*.py"):
            if path.name in {"integration.py", "__init__.py"}:
                continue
            allowed = allowed_upstream.get(path.name, set())
            for module in _imports(path):
                if module.startswith(upstream_prefixes) and module not in allowed:
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_integration_adapter_does_not_import_runtime_services(self) -> None:
        imports = set(_imports(_PACKAGE_ROOT / "evaluation" / "integration.py"))
        forbidden = {
            "adaptive_agent_runtime.core.runtime",
            "adaptive_agent_runtime.core.state",
            "adaptive_agent_runtime.core.trace",
            "adaptive_agent_runtime.orchestration.execution",
            "adaptive_agent_runtime.orchestration.planner",
            "adaptive_agent_runtime.orchestration.scheduler",
            "adaptive_agent_runtime.context_memory.context_runtime",
            "adaptive_agent_runtime.context_memory.memory_runtime",
            "adaptive_agent_runtime.tool_ecosystem.execution",
            "adaptive_agent_runtime.tool_ecosystem.registry",
            "adaptive_agent_runtime.tool_ecosystem.selector",
        }

        self.assertEqual(imports & forbidden, set())

    def test_optimization_agent_exposes_proposal_generation_only(self) -> None:
        public_methods = {
            name
            for name, member in inspect.getmembers(
                ConservativeOptimizationAgent,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }

        self.assertEqual(public_methods, {"propose"})


if __name__ == "__main__":
    unittest.main()
