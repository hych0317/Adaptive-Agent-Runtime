from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar
import json
import unittest

from applications.ecommerce_support.composition import compose_ecommerce_support
from applications.ecommerce_support.models import (
    AuthenticatedPrincipal,
    MemoryWriteRequest,
)
from applications.ecommerce_support.policies import DomainPolicyError
from applications.governance_scenario_suite.context_memory_executors import (
    FullAARContextMemoryExecutor,
    NoAuthoritativeConstraintExecutor,
    NoMemoryScopeFilterExecutor,
)
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
    ScenarioSpec,
)
from applications.governance_scenario_suite.loader import (
    load_catalog,
    load_scenario_directory,
)
from applications.governance_scenario_suite.reporting import (
    build_batch_result,
    write_reports,
)
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import ScenarioExecutorRegistry


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "applications"
    / "governance_scenario_suite"
    / "scenarios"
)
FAMILIES = {
    "C1_AUTHORITATIVE_CONSTRAINT",
    "M1_MEMORY_SCOPE",
    "M2_CONDITIONAL_MEMORY",
}


class ContextMemoryScenarioTests(unittest.TestCase):
    scenarios: ClassVar[tuple[ScenarioSpec, ...]]
    by_id: ClassVar[dict[str, ScenarioSpec]]
    runner: ClassVar[GovernanceScenarioRunner]

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = tuple(
            item
            for item in load_scenario_directory(SCENARIO_DIRECTORY)
            if item.family_id in FAMILIES
        )
        cls.by_id = {item.id: item for item in cls.scenarios}
        cls.runner = GovernanceScenarioRunner(
            executors=ScenarioExecutorRegistry(
                {
                    ScenarioProfile.FULL_AAR: FullAARContextMemoryExecutor(),
                    ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK: (
                        NoAuthoritativeConstraintExecutor()
                    ),
                    ScenarioProfile.NO_MEMORY_SCOPE_FILTER: NoMemoryScopeFilterExecutor(),
                }
            )
        )

    def test_catalog_stage5_cases_all_have_executable_specs(self) -> None:
        catalog = load_catalog(SCENARIO_DIRECTORY / "catalog.yaml")
        registered = {
            case_id
            for item in catalog.registrations
            if item.family_id in FAMILIES
            for case_id in (*item.variant_ids, item.positive_control_id)
        }
        self.assertEqual(registered, set(self.by_id))
        self.assertEqual(len(registered), 8)

    def test_full_aar_passes_all_context_and_memory_cases(self) -> None:
        results = tuple(self.runner.run(item) for item in self.scenarios)
        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.PASS for item in results),
            {
                item.scenario_id: [
                    finding.code
                    for finding in item.findings
                    if finding.verdict is not EvaluationVerdict.PASS
                ]
                for item in results
                if item.evaluation_verdict is not EvaluationVerdict.PASS
            },
        )

    def test_authoritative_constraint_ablation_is_detected_only_on_illegal_case(self) -> None:
        dangerous = self.by_id["C1_CONTEXT_PRESSURE_OVER_REFUND"].model_copy(
            update={"profile": ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK}
        )
        legal = self.by_id["C1_CONTEXT_PRESSURE_LEGAL_REFUND"].model_copy(
            update={"profile": ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK}
        )

        dangerous_result = self.runner.run(dangerous)
        legal_result = self.runner.run(legal)

        self.assertEqual(dangerous_result.evaluation_verdict, EvaluationVerdict.FAIL)
        self.assertEqual(dangerous_result.evidence.external_effect_count, 1)
        self.assertEqual(legal_result.evaluation_verdict, EvaluationVerdict.PASS)

    def test_memory_scope_ablation_leaks_both_tenant_and_user_canaries(self) -> None:
        results = tuple(
            self.runner.run(
                self.by_id[case_id].model_copy(
                    update={"profile": ScenarioProfile.NO_MEMORY_SCOPE_FILTER}
                )
            )
            for case_id in ("M1A_CROSS_TENANT", "M1B_CROSS_USER")
        )

        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.FAIL for item in results)
        )

    def test_memory_scope_ablation_preserves_own_and_conditional_behavior(self) -> None:
        case_ids = ("M1_OWN_MEMORY", "M2A_ORTHOGONAL_PREFERENCE", "M2_MATCHING_CONDITION")
        results = tuple(
            self.runner.run(
                self.by_id[case_id].model_copy(
                    update={"profile": ScenarioProfile.NO_MEMORY_SCOPE_FILTER}
                )
            )
            for case_id in case_ids
        )
        self.assertTrue(
            all(item.evaluation_verdict is EvaluationVerdict.PASS for item in results)
        )

    def test_sensitive_memory_candidates_are_rejected_without_audit_leak(self) -> None:
        scenario = self.by_id["M1_OWN_MEMORY"]
        principal = AuthenticatedPrincipal(tenant_id="T1", user_id="U1")
        canaries = (
            ("address", "ADDR-RAW-CANARY"),
            ("payment_token", "TOKEN-RAW-CANARY"),
            ("credential", "CREDENTIAL-RAW-CANARY"),
        )
        with TemporaryDirectory() as directory:
            composition = compose_ecommerce_support(directory, clock=lambda: scenario.clock)
            try:
                composition.store.seed(scenario.initial_authoritative_state)
                for key, value in canaries:
                    candidate = MemoryWriteRequest(
                        memory_key=f"sensitive-{key}",
                        content={key: value},
                        tenant_id="T1",
                        subject_user_id="U1",
                    )
                    with self.assertRaises(DomainPolicyError):
                        composition.tools.validate_memory_write(principal, candidate)
                audit_events = composition.store.snapshot().audit_events
                audit_json = json.dumps(
                    [item.model_dump(mode="json") for item in audit_events],
                    sort_keys=True,
                )
            finally:
                composition.close()
        self.assertEqual(len(audit_events), 3)
        self.assertTrue(
            all(item.event_type.value == "MEMORY_WRITE_DENIED" for item in audit_events)
        )
        for _, canary in canaries:
            self.assertNotIn(canary, audit_json)

    def test_persisted_reports_do_not_contain_any_memory_canary(self) -> None:
        results = tuple(self.runner.run(item) for item in self.scenarios)
        batch = build_batch_result(
            results,
            suite_id="stage5-safe-report",
            generated_at=self.scenarios[0].clock,
        )
        with TemporaryDirectory() as directory:
            json_path, markdown_path = write_reports(batch, directory)
            persisted = json_path.read_text(encoding="utf-8") + markdown_path.read_text(
                encoding="utf-8"
            )
        for scenario in self.scenarios:
            for memory in scenario.initial_authoritative_state.memories:
                if memory.canary:
                    self.assertNotIn(memory.canary, persisted)


if __name__ == "__main__":
    unittest.main()
