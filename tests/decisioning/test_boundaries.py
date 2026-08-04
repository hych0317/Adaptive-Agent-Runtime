from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from adaptive_agent_runtime.decisioning import (
    AgentContext,
    DecisionProposalProducer,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "adaptive_agent_runtime"


def imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


class DecisionBoundaryTests(unittest.TestCase):
    def test_core_does_not_depend_on_decisioning(self) -> None:
        violations: list[str] = []
        for path in (PACKAGE_ROOT / "core").rglob("*.py"):
            for module in imports(path):
                if module.startswith("adaptive_agent_runtime.decisioning"):
                    violations.append(f"{path.name}: {module}")
        self.assertEqual(violations, [])

    def test_decision_contract_and_lifecycle_are_provider_neutral(self) -> None:
        violations: list[str] = []
        for name in (
            "models.py",
            "contracts.py",
            "budget.py",
            "context.py",
            "validation.py",
            "lifecycle.py",
            "store.py",
        ):
            path = PACKAGE_ROOT / "decisioning" / name
            for module in imports(path):
                if module.startswith(
                    (
                        "adaptive_agent_runtime.llm",
                        "adaptive_agent_runtime.governance",
                        "adaptive_agent_runtime.persistence",
                        "adaptive_agent_runtime.orchestration",
                    )
                ):
                    violations.append(f"{name}: {module}")
        self.assertEqual(violations, [])

    def test_agent_proposal_port_has_no_runtime_authority(self) -> None:
        methods = {
            name
            for name, member in inspect.getmembers(
                DecisionProposalProducer,
                predicate=inspect.isfunction,
            )
            if not name.startswith("_")
        }
        self.assertEqual(methods, {"propose"})

    def test_agent_context_has_no_runtime_service_fields(self) -> None:
        forbidden = {
            "state",
            "state_store",
            "memory_store",
            "trace_sink",
            "governance",
            "other_agent_context",
        }
        self.assertEqual(set(AgentContext.model_fields) & forbidden, set())
        self.assertNotIn("allowed_actions", AgentContext.model_fields)


if __name__ == "__main__":
    unittest.main()
