from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import ClassVar
import unittest

from applications.governance_scenario_suite.contracts import (
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.execution import (
    run_deterministic_suite,
)
from applications.governance_scenario_suite.final_reporting import (
    evaluate_release_gate,
)
from applications.governance_scenario_suite.loader import load_scenario_directory
from applications.governance_scenario_suite.metrics import build_evaluation_matrix


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIRECTORY = ROOT / "applications" / "governance_scenario_suite" / "scenarios"


class GovernanceCLIAndReleaseGateTests(unittest.TestCase):
    scenarios: ClassVar[tuple[ScenarioSpec, ...]]
    by_id: ClassVar[dict[str, ScenarioSpec]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_scenario_directory(SCENARIO_DIRECTORY)
        cls.by_id = {item.id: item for item in cls.scenarios}

    def test_release_gate_requires_full_controls_and_each_precise_ablation(self) -> None:
        full_cases = (
            "P2_APPROVED_REFUND",
            "P2B_APPROVAL_AMOUNT_TAMPERING",
            "M1B_CROSS_USER",
            "C1_CONTEXT_PRESSURE_OVER_REFUND",
            "R2A_NOT_COMMITTED",
            "R1_COMMITTED_RESPONSE_LOST",
        )
        results = list(
            run_deterministic_suite(
                self.scenarios,
                profile=ScenarioProfile.FULL_AAR,
                scenario_ids=full_cases,
            )
        )
        targets = (
            (
                ScenarioProfile.NO_EXACT_EFFECT_BINDING,
                "P2B_APPROVAL_AMOUNT_TAMPERING",
            ),
            (ScenarioProfile.NO_MEMORY_SCOPE_FILTER, "M1B_CROSS_USER"),
            (
                ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK,
                "C1_CONTEXT_PRESSURE_OVER_REFUND",
            ),
            (ScenarioProfile.NO_RECONCILE_FAIL_CLOSED, "R2A_NOT_COMMITTED"),
            (
                ScenarioProfile.NO_RECONCILE_BLIND_RETRY,
                "R1_COMMITTED_RESPONSE_LOST",
            ),
        )
        for profile, scenario_id in targets:
            results.extend(
                run_deterministic_suite(
                    self.scenarios,
                    profile=profile,
                    scenario_ids=(scenario_id,),
                )
            )
        gate = evaluate_release_gate(
            build_evaluation_matrix(tuple(results), self.by_id),
            production_database_unchanged=True,
        )

        self.assertTrue(gate.passed)
        self.assertTrue(all(item.passed for item in gate.checks))

    def test_demo_cli_writes_traceable_safe_json_and_markdown(self) -> None:
        with TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_governance_scenarios.py"),
                    "demo",
                    "--output-dir",
                    directory,
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            json_path = Path(directory) / "governance-report.json"
            markdown_path = Path(directory) / "governance-report.md"
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            raw = json_path.read_text(encoding="utf-8")
            report = json.loads(raw)
            self.assertEqual(report["manifest"]["mode"], "demo")
            self.assertEqual(len(report["batch"]["results"]), 3)
            self.assertEqual(len(report["manifest"]["git_revision"]), 40)
            self.assertNotIn("FOREIGN-U2-CANARY", raw)
            self.assertNotIn("ADDR-U2-SECRET", raw)
            self.assertIn("P2B_APPROVAL_AMOUNT_TAMPERING", raw)
            self.assertIn("R1_COMMITTED_RESPONSE_LOST", raw)
            self.assertIn("M1B_CROSS_USER", raw)


if __name__ == "__main__":
    unittest.main()
