"""Architecture boundaries for provider-neutral cognitive capabilities."""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from adaptive_agent_runtime.llm.capabilities.contracts import (
    ActionProposalCapability,
    ArtifactGenerationCapability,
    EvidenceJudgeCapability,
    GraphMutationProposalCapability,
    MemoryExtractionCapability,
    ReasoningCapability,
    RootCauseAnalysisCapability,
    SemanticCompressionCapability,
    TaskGraphProposalCapability,
)
from adaptive_agent_runtime.llm.contracts import InferenceBackend
from adaptive_agent_runtime.llm.adapters.models import LLMContextRole


_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "adaptive_agent_runtime"
_EXISTING_RUNTIME_PACKAGES = (
    "core",
    "orchestration",
    "context_memory",
    "tool_ecosystem",
    "evaluation",
    "governance",
    "decisioning",
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


class LLMBoundaryTests(unittest.TestCase):
    def test_existing_runtime_packages_do_not_depend_on_llm(self) -> None:
        violations: list[str] = []
        for package_name in _EXISTING_RUNTIME_PACKAGES:
            for path in (_PACKAGE_ROOT / package_name).rglob("*.py"):
                for module in _imports(path):
                    if module.startswith("adaptive_agent_runtime.llm"):
                        violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_capability_models_do_not_import_runtime_authority_models(self) -> None:
        imports = set(
            _imports(_PACKAGE_ROOT / "llm" / "capabilities" / "models.py")
        )
        forbidden_prefixes = tuple(
            f"adaptive_agent_runtime.{name}"
            for name in _EXISTING_RUNTIME_PACKAGES
        )

        self.assertFalse(
            any(module.startswith(forbidden_prefixes) for module in imports)
        )

    def test_capability_implementations_do_not_import_runtime_authority(self) -> None:
        forbidden_prefixes = tuple(
            f"adaptive_agent_runtime.{name}"
            for name in _EXISTING_RUNTIME_PACKAGES
            if name != "core"
        )
        violations: list[str] = []
        for path in (_PACKAGE_ROOT / "llm" / "capabilities").rglob("*.py"):
            for module in _imports(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_cognitive_contracts_expose_no_apply_methods(self) -> None:
        contracts = (
            ActionProposalCapability,
            ReasoningCapability,
            TaskGraphProposalCapability,
            ArtifactGenerationCapability,
            EvidenceJudgeCapability,
            RootCauseAnalysisCapability,
            GraphMutationProposalCapability,
            SemanticCompressionCapability,
            MemoryExtractionCapability,
        )
        forbidden = {
            "apply",
            "save",
            "publish",
            "execute_tool",
            "write_memory",
        }
        methods: set[str] = set()
        for contract in contracts:
            methods.update(
                name
                for name, member in inspect.getmembers(
                    contract,
                    predicate=inspect.isfunction,
                )
                if not name.startswith("_")
            )

        self.assertEqual(methods & forbidden, set())

    def test_external_operations_are_async_protocol_methods(self) -> None:
        methods = (
            ActionProposalCapability.propose_action,
            ReasoningCapability.analyze,
            TaskGraphProposalCapability.propose,
            ArtifactGenerationCapability.generate,
            EvidenceJudgeCapability.assess,
            RootCauseAnalysisCapability.analyze_root_cause,
            GraphMutationProposalCapability.propose_mutations,
            SemanticCompressionCapability.compress,
            MemoryExtractionCapability.extract,
            InferenceBackend.probe,
            InferenceBackend.invoke,
        )

        self.assertTrue(all(inspect.iscoroutinefunction(method) for method in methods))

    def test_gateway_does_not_import_runtime_state_or_execution(self) -> None:
        forbidden_prefixes = (
            "adaptive_agent_runtime.context_memory",
            "adaptive_agent_runtime.evaluation",
            "adaptive_agent_runtime.governance",
            "adaptive_agent_runtime.orchestration",
            "adaptive_agent_runtime.tool_ecosystem",
        )
        violations: list[str] = []
        for path in (_PACKAGE_ROOT / "llm" / "gateway").rglob("*.py"):
            for module in _imports(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_model_providers_do_not_import_runtime_state_or_execution(self) -> None:
        forbidden_prefixes = (
            "adaptive_agent_runtime.context_memory",
            "adaptive_agent_runtime.evaluation",
            "adaptive_agent_runtime.governance",
            "adaptive_agent_runtime.orchestration",
            "adaptive_agent_runtime.tool_ecosystem",
        )
        violations: list[str] = []
        for path in (_PACKAGE_ROOT / "llm" / "providers").rglob("*.py"):
            for module in _imports(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_agent_backends_do_not_import_runtime_authority_modules(self) -> None:
        forbidden_prefixes = (
            "adaptive_agent_runtime.context_memory",
            "adaptive_agent_runtime.evaluation",
            "adaptive_agent_runtime.governance",
            "adaptive_agent_runtime.orchestration",
            "adaptive_agent_runtime.tool_ecosystem",
        )
        violations: list[str] = []
        for path in (_PACKAGE_ROOT / "llm" / "agent_backends").rglob("*.py"):
            for module in _imports(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_only_context_adapter_imports_context_memory_models(self) -> None:
        violations: list[str] = []
        llm_root = _PACKAGE_ROOT / "llm"
        for path in llm_root.rglob("*.py"):
            if "adapters" in path.parts:
                continue
            for module in _imports(path):
                if module.startswith("adaptive_agent_runtime.context_memory"):
                    violations.append(f"{path.name}: {module}")

        self.assertEqual(violations, [])

    def test_runtime_context_cannot_be_elevated_to_instruction_roles(self) -> None:
        self.assertEqual(
            set(LLMContextRole),
            {LLMContextRole.USER, LLMContextRole.CONTEXT},
        )


if __name__ == "__main__":
    unittest.main()
