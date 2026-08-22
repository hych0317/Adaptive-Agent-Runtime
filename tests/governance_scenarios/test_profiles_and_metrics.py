from __future__ import annotations

from pathlib import Path
from typing import ClassVar
import unittest

from applications.governance_scenario_suite.baseline import (
    FullAARScenarioExecutor,
    PlainAgentExecutor,
)
from applications.governance_scenario_suite.context_memory_executors import (
    NoAuthoritativeConstraintExecutor,
    NoMemoryScopeFilterExecutor,
)
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.executors import (
    NoExactEffectBindingExecutor,
)
from applications.governance_scenario_suite.loader import load_scenario_directory
from applications.governance_scenario_suite.metrics import build_evaluation_matrix
from applications.governance_scenario_suite.profiles import all_profile_manifests
from applications.governance_scenario_suite.recovery import CrossProcessRecoveryRunner
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import ScenarioExecutorRegistry


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIRECTORY = ROOT / "applications" / "governance_scenario_suite" / "scenarios"
RECOVERY_FAMILIES = {"R1_COMMITTED_RECOVERY", "R2_INCOMPLETE_RECOVERY"}


class ProfileAndMetricsTests(unittest.TestCase):
    scenarios: ClassVar[tuple[ScenarioSpec, ...]]
    by_id: ClassVar[dict[str, ScenarioSpec]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_scenario_directory(SCENARIO_DIRECTORY)
        cls.by_id = {item.id: item for item in cls.scenarios}

    @staticmethod
    def _runner(profile: ScenarioProfile, executor: object) -> GovernanceScenarioRunner:
        return GovernanceScenarioRunner(
            executors=ScenarioExecutorRegistry(
                {profile: executor}  # type: ignore[dict-item]
            )
        )

    def test_profile_manifests_share_domain_contracts_and_isolate_ablations(self) -> None:
        manifests = all_profile_manifests()
        precise = tuple(
            item
            for item in manifests
            if item.profile not in {ScenarioProfile.FULL_AAR, ScenarioProfile.PLAIN_AGENT}
        )

        self.assertEqual(len(manifests), len(ScenarioProfile))
        self.assertTrue(all(len(item.disabled_boundaries) == 1 for item in precise))
        self.assertEqual(
            len({item.domain_store_contract_fingerprint for item in manifests}),
            1,
        )
        self.assertEqual(len({item.gateway_contract_fingerprint for item in manifests}), 1)
        self.assertEqual(len({item.tool_schema_fingerprint for item in manifests}), 1)
        self.assertEqual(
            len({item.system_prompt_fingerprint for item in manifests}),
            1,
        )
        self.assertEqual(
            len({item.control_flow_fingerprint for item in manifests}),
            len(manifests),
        )

    def test_full_and_plain_agent_matrix_has_no_plain_availability_loss(self) -> None:
        non_recovery = tuple(
            item for item in self.scenarios if item.family_id not in RECOVERY_FAMILIES
        )
        full_runner = self._runner(
            ScenarioProfile.FULL_AAR,
            FullAARScenarioExecutor(),
        )
        plain_runner = self._runner(
            ScenarioProfile.PLAIN_AGENT,
            PlainAgentExecutor(),
        )
        full_results = tuple(full_runner.run(item) for item in non_recovery)
        plain_results = tuple(
            plain_runner.run(
                item.model_copy(update={"profile": ScenarioProfile.PLAIN_AGENT})
            )
            for item in non_recovery
        )
        matrix = build_evaluation_matrix(
            (*full_results, *plain_results),
            self.by_id,
        )
        summaries = {item.profile: item for item in matrix.profiles}

        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.PASS for item in full_results)
        )
        self.assertGreater(summaries[ScenarioProfile.PLAIN_AGENT].safety_loss_count, 0)
        self.assertGreater(
            summaries[ScenarioProfile.PLAIN_AGENT].unauthorized_effect_execution_rate,
            0.0,
        )
        self.assertGreater(
            summaries[ScenarioProfile.PLAIN_AGENT].memory_or_context_leak_count,
            0,
        )
        self.assertEqual(
            summaries[ScenarioProfile.PLAIN_AGENT].availability_loss_count,
            0,
        )

    def test_each_precise_ablation_breaks_target_and_preserves_positive_control(self) -> None:
        cases = (
            (
                ScenarioProfile.NO_EXACT_EFFECT_BINDING,
                NoExactEffectBindingExecutor(),
                "P2A_APPROVAL_TARGET_TAMPERING",
                "P2_APPROVED_REFUND",
            ),
            (
                ScenarioProfile.NO_MEMORY_SCOPE_FILTER,
                NoMemoryScopeFilterExecutor(),
                "M1B_CROSS_USER",
                "M1_OWN_MEMORY",
            ),
            (
                ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK,
                NoAuthoritativeConstraintExecutor(),
                "C1_CONTEXT_PRESSURE_OVER_REFUND",
                "C1_CONTEXT_PRESSURE_LEGAL_REFUND",
            ),
        )
        for profile, executor, dangerous_id, positive_id in cases:
            runner = self._runner(profile, executor)
            dangerous = runner.run(
                self.by_id[dangerous_id].model_copy(update={"profile": profile})
            )
            positive = runner.run(
                self.by_id[positive_id].model_copy(update={"profile": profile})
            )
            self.assertEqual(dangerous.evaluation_verdict, EvaluationVerdict.FAIL)
            self.assertEqual(positive.evaluation_verdict, EvaluationVerdict.PASS)

        recovery = CrossProcessRecoveryRunner()
        recovery_cases = (
            (
                ScenarioProfile.NO_RECONCILE_FAIL_CLOSED,
                "R2A_NOT_COMMITTED",
                "R2_NORMAL_REFUND",
            ),
            (
                ScenarioProfile.NO_RECONCILE_BLIND_RETRY,
                "R1_COMMITTED_RESPONSE_LOST",
                "R1_NORMAL_REFUND",
            ),
        )
        for profile, dangerous_id, positive_id in recovery_cases:
            dangerous = recovery.run(
                self.by_id[dangerous_id].model_copy(update={"profile": profile})
            )
            positive = recovery.run(
                self.by_id[positive_id].model_copy(update={"profile": profile})
            )
            self.assertEqual(dangerous.evaluation_verdict, EvaluationVerdict.FAIL)
            self.assertEqual(positive.evaluation_verdict, EvaluationVerdict.PASS)

    def test_unsafe_profile_switches_are_absent_from_production_runtime(self) -> None:
        forbidden = tuple(profile.value for profile in ScenarioProfile if profile is not ScenarioProfile.FULL_AAR)
        production_roots = (ROOT / "src", ROOT / "applications" / "ecommerce_support")
        for root in production_roots:
            corpus = "\n".join(
                path.read_text(encoding="utf-8")
                for path in root.rglob("*.py")
            )
            for value in forbidden:
                self.assertNotIn(value, corpus)


if __name__ == "__main__":
    unittest.main()
