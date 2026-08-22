from __future__ import annotations

from pathlib import Path
import unittest

from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
)
from applications.governance_scenario_suite.loader import load_scenario_directory
from applications.governance_scenario_suite.recovery import CrossProcessRecoveryRunner


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "applications"
    / "governance_scenario_suite"
    / "scenarios"
)


class CrossProcessRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = {
            item.id: item
            for item in load_scenario_directory(SCENARIO_DIRECTORY)
            if item.family_id in {"R1_COMMITTED_RECOVERY", "R2_INCOMPLETE_RECOVERY"}
        }
        cls.runner = CrossProcessRecoveryRunner()

    def test_full_aar_recovery_matrix_passes_in_new_processes(self) -> None:
        results = tuple(self.runner.run(item) for item in self.scenarios.values())

        self.assertEqual(len(results), 5)
        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.PASS for item in results),
            {item.scenario_id: item.evaluation_verdict for item in results},
        )
        self.assertTrue(all(item.metrics["model_call_count"] == 1 for item in results))

    def test_committed_recovery_does_not_repeat_external_attempt(self) -> None:
        result = self.runner.run(self.scenarios["R1_COMMITTED_RESPONSE_LOST"])

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertEqual(result.metrics["start_exit_code"], 91)
        self.assertEqual(result.evidence.external_effect_count, 1)
        self.assertEqual(result.evidence.external_attempt_count, 1)

    def test_not_committed_recovery_reuses_original_key(self) -> None:
        result = self.runner.run(self.scenarios["R2A_NOT_COMMITTED"])

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertEqual(result.metrics["start_exit_code"], 92)
        self.assertEqual(result.evidence.external_effect_count, 1)
        self.assertEqual(result.evidence.external_attempt_count, 1)

    def test_unknown_reconciliation_never_replays(self) -> None:
        result = self.runner.run(self.scenarios["R2B_RECONCILIATION_UNKNOWN"])

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertEqual(result.evidence.external_effect_count, 0)
        self.assertEqual(result.evidence.external_attempt_count, 0)

    def test_fail_closed_ablation_loses_legal_r2a_completion(self) -> None:
        scenario = self.scenarios["R2A_NOT_COMMITTED"].model_copy(
            update={"profile": ScenarioProfile.NO_RECONCILE_FAIL_CLOSED}
        )

        result = self.runner.run(scenario)

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.FAIL)
        self.assertEqual(result.evidence.external_effect_count, 0)

    def test_blind_retry_ablation_is_caught_by_duplicate_effect_oracle(self) -> None:
        scenario = self.scenarios["R1_COMMITTED_RESPONSE_LOST"].model_copy(
            update={"profile": ScenarioProfile.NO_RECONCILE_BLIND_RETRY}
        )

        result = self.runner.run(scenario)

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.FAIL)
        self.assertEqual(result.evidence.external_effect_count, 2)


if __name__ == "__main__":
    unittest.main()
