from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DecisionConsolidationBoundaryTests(unittest.TestCase):
    def test_runtime_affecting_agent_calls_exist_only_in_decision_adapters(
        self,
    ) -> None:
        strategies = (ROOT / "applications/research_agent/strategies.py").read_text(
            encoding="utf-8"
        )
        agent = (ROOT / "applications/research_agent/agent.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(".propose_mutations(", strategies)
        self.assertNotIn(".propose_action(", strategies)
        self.assertNotIn("_extract_llm_memories", agent)
        self.assertNotIn("memory_extractor.extract(", agent)
        self.assertNotIn("class GovernedToolIntentExecutor", strategies)
        self.assertNotIn("tool_intent_executor", strategies)
        self.assertNotIn("ToolInvocation(", strategies)

        tool_invocation = (
            ROOT / "applications/research_agent/tool_invocation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("DecisionLifecycleCoordinator", tool_invocation)
        self.assertIn("PolicyAgentContextBuilder", tool_invocation)
        self.assertIn("RuntimeDecisionGovernanceAdapter", tool_invocation)
        self.assertIn("GovernedDecisionApplier", tool_invocation)
        self.assertIn("effect.invocation", tool_invocation)

        adapters = {
            "ready": ROOT
            / "src/adaptive_agent_runtime/llm/adapters/ready_node_decision.py",
            "mutation": ROOT
            / "src/adaptive_agent_runtime/llm/adapters/graph_mutation_decision.py",
            "memory": ROOT
            / "src/adaptive_agent_runtime/llm/adapters/memory_extraction_decision.py",
        }
        self.assertIn(".propose_action(", adapters["ready"].read_text(encoding="utf-8"))
        self.assertIn(
            ".propose_mutations(",
            adapters["mutation"].read_text(encoding="utf-8"),
        )
        self.assertIn(".extract(", adapters["memory"].read_text(encoding="utf-8"))

    def test_risk_agent_remains_advisory_not_a_control_decision(self) -> None:
        strategies = (ROOT / "applications/research_agent/strategies.py").read_text(
            encoding="utf-8"
        )
        risk_section = strategies.split("class DeterministicRiskAgent:", 1)[1]
        risk_section = risk_section.split("class ReviewStrategy", 1)[0]
        self.assertIn("NodeExecutionResult.ok", risk_section)
        self.assertNotIn("DecisionLifecycleCoordinator", risk_section)
        self.assertNotIn("GraphMutation", risk_section)
        self.assertNotIn("MemoryCandidate", risk_section)


if __name__ == "__main__":
    unittest.main()
