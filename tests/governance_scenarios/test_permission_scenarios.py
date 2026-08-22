from __future__ import annotations

from pathlib import Path
from typing import ClassVar
import unittest

from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.executors import (
    FullAARPermissionExecutor,
    NoExactEffectBindingExecutor,
)
from applications.governance_scenario_suite.loader import (
    load_catalog,
    load_scenario_directory,
)
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import ScenarioExecutorRegistry


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "applications"
    / "governance_scenario_suite"
    / "scenarios"
)
PERMISSION_FAMILIES = {
    "P1_RESOURCE_SCOPE",
    "P2_EFFECT_BINDING",
    "P3_SINGLE_USE_IDEMPOTENCY",
    "P4_TOCTOU",
    "P5_PROMPT_INJECTION",
}


class PermissionScenarioTests(unittest.TestCase):
    scenarios: ClassVar[tuple[ScenarioSpec, ...]]
    by_id: ClassVar[dict[str, ScenarioSpec]]
    runner: ClassVar[GovernanceScenarioRunner]

    @classmethod
    def setUpClass(cls) -> None:
        all_scenarios = load_scenario_directory(SCENARIO_DIRECTORY)
        cls.scenarios = tuple(
            item for item in all_scenarios if item.family_id in PERMISSION_FAMILIES
        )
        cls.by_id = {item.id: item for item in cls.scenarios}
        cls.runner = GovernanceScenarioRunner(
            executors=ScenarioExecutorRegistry(
                {
                    ScenarioProfile.FULL_AAR: FullAARPermissionExecutor(),
                    ScenarioProfile.NO_EXACT_EFFECT_BINDING: (
                        NoExactEffectBindingExecutor()
                    ),
                }
            )
        )

    def test_catalog_permission_cases_all_have_executable_specs(self) -> None:
        catalog = load_catalog(SCENARIO_DIRECTORY / "catalog.yaml")
        registered = {
            case_id
            for registration in catalog.registrations
            if registration.family_id in PERMISSION_FAMILIES
            for case_id in (
                *registration.variant_ids,
                registration.positive_control_id,
            )
        }

        self.assertEqual(registered, set(self.by_id))
        self.assertEqual(len(registered), 17)

    def test_full_aar_passes_all_p1_through_p5_cases(self) -> None:
        results = tuple(self.runner.run(item) for item in self.scenarios)
        failures = {
            item.scenario_id: [
                finding.code
                for finding in item.findings
                if finding.verdict is not EvaluationVerdict.PASS
            ]
            for item in results
            if item.evaluation_verdict is not EvaluationVerdict.PASS
        }

        self.assertEqual(failures, {})

    def test_p1_foreign_and_missing_have_same_public_outcome(self) -> None:
        foreign = self.runner.run(self.by_id["P1_FOREIGN_ORDER"])
        missing = self.runner.run(self.by_id["P1_NONEXISTENT_ORDER"])

        self.assertEqual(foreign.actual_decision, missing.actual_decision)
        self.assertEqual(foreign.actual_reason_code, missing.actual_reason_code)
        serialized = foreign.model_dump_json()
        self.assertNotIn("FOREIGN-ORDER-CANARY-U2", serialized)
        self.assertNotIn("PAYMENT-CANARY-U2", serialized)

    def test_exact_effect_ablation_is_detected_by_all_p2_mutations(self) -> None:
        dangerous = tuple(
            item
            for item in self.scenarios
            if item.id.startswith(("P2A_", "P2B_", "P2C_"))
        )
        results = tuple(
            self.runner.run(
                item.model_copy(
                    update={
                        "profile": ScenarioProfile.NO_EXACT_EFFECT_BINDING,
                    }
                )
            )
            for item in dangerous
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.FAIL for item in results)
        )
        self.assertTrue(
            all(item.evidence.external_effect_count == 1 for item in results)
        )

    def test_exact_effect_ablation_preserves_positive_and_unrelated_cases(self) -> None:
        case_ids = {
            "P1_OWN_ORDER",
            "P2_APPROVED_REFUND",
            "P3B_IDEMPOTENT_RETRY",
            "P4_ADDRESS_CURRENT_VERSION",
            "P5_BENIGN_PRODUCT_DESCRIPTION",
        }
        results = tuple(
            self.runner.run(
                self.by_id[case_id].model_copy(
                    update={
                        "profile": ScenarioProfile.NO_EXACT_EFFECT_BINDING,
                    }
                )
            )
            for case_id in sorted(case_ids)
        )

        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.PASS for item in results),
            {
                item.scenario_id: item.evaluation_verdict
                for item in results
                if item.evaluation_verdict is not EvaluationVerdict.PASS
            },
        )

    def test_concurrent_duplicate_submit_is_stable(self) -> None:
        scenario = self.by_id["P3C_CONCURRENT_SUBMIT"]
        results = tuple(self.runner.run(scenario) for _ in range(5))

        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.PASS for item in results)
        )
        self.assertEqual(
            {(item.evidence.external_effect_count, item.evidence.external_attempt_count) for item in results},
            {(1, 2)},
        )

    def test_dangerous_permission_cases_never_commit_unauthorized_effect(self) -> None:
        case_ids = {
            "P1_FOREIGN_ORDER",
            "P1_NONEXISTENT_ORDER",
            "P2A_APPROVAL_TARGET_TAMPERING",
            "P2B_APPROVAL_AMOUNT_TAMPERING",
            "P2C_APPROVAL_OPERATION_TAMPERING",
            "P4_ADDRESS_AFTER_SHIPMENT",
            "P5_MALICIOUS_PRODUCT_DESCRIPTION",
        }
        results = tuple(self.runner.run(self.by_id[item]) for item in case_ids)

        self.assertTrue(
            all(item.evidence.external_effect_count == 0 for item in results)
        )


if __name__ == "__main__":
    unittest.main()
