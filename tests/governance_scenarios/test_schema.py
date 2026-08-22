from __future__ import annotations

from pathlib import Path
import unittest

from pydantic import ValidationError
import yaml

from applications.governance_scenario_suite.contracts import (
    EffectSpec,
    FaultPoint,
    ReasonCode,
    ReconciliationStatus,
    ScenarioCatalogSpec,
    ScenarioProfile,
    ScenarioSpec,
    changed_effect_fields,
)


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "applications"
    / "governance_scenario_suite"
    / "scenarios"
)


def load_yaml(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class GovernanceScenarioContractTests(unittest.TestCase):
    def test_catalog_registers_ten_unique_core_families(self) -> None:
        catalog = ScenarioCatalogSpec.model_validate(
            load_yaml(SCENARIO_DIRECTORY / "catalog.yaml")
        )

        self.assertEqual(len(catalog.registrations), 10)
        self.assertEqual(
            {item.family_id for item in catalog.registrations},
            {
                "P1_RESOURCE_SCOPE",
                "P2_EFFECT_BINDING",
                "P3_SINGLE_USE_IDEMPOTENCY",
                "P4_TOCTOU",
                "P5_PROMPT_INJECTION",
                "R1_COMMITTED_RECOVERY",
                "R2_INCOMPLETE_RECOVERY",
                "C1_AUTHORITATIVE_CONSTRAINT",
                "M1_MEMORY_SCOPE",
                "M2_CONDITIONAL_MEMORY",
            },
        )

    def test_all_checked_in_scenarios_are_strictly_valid(self) -> None:
        paths = tuple(sorted(SCENARIO_DIRECTORY.glob("*.yaml")))
        scenario_paths = tuple(path for path in paths if path.name != "catalog.yaml")

        self.assertGreaterEqual(len(scenario_paths), 4)
        parsed = tuple(
            ScenarioSpec.model_validate(load_yaml(path)) for path in scenario_paths
        )
        self.assertEqual(len({item.id for item in parsed}), len(parsed))

    def test_p2_variants_change_exactly_one_unconfounded_field(self) -> None:
        expected = {
            "P2A_APPROVAL_TARGET_TAMPERING": ("order_id",),
            "P2B_APPROVAL_AMOUNT_TAMPERING": ("amount_cents",),
            "P2C_APPROVAL_OPERATION_TAMPERING": ("operation",),
        }
        for path in SCENARIO_DIRECTORY.glob("p2[abc]_*.yaml"):
            scenario = ScenarioSpec.model_validate(load_yaml(path))
            assert scenario.approved_effect is not None
            actual = changed_effect_fields(
                scenario.approved_effect,
                scenario.model_script.proposals[0],
            )
            self.assertEqual(actual, expected[scenario.id])
        self.assertEqual(set(expected), {
            ScenarioSpec.model_validate(load_yaml(path)).id
            for path in SCENARIO_DIRECTORY.glob("p2[abc]_*.yaml")
        })

    def test_unknown_scenario_field_is_rejected(self) -> None:
        payload = load_yaml(SCENARIO_DIRECTORY / "p2b_approval_amount_tampering.yaml")
        assert isinstance(payload, dict)
        payload["unexpected_switch"] = True

        with self.assertRaises(ValidationError):
            ScenarioSpec.model_validate(payload)

    def test_fractional_or_coerced_money_is_rejected(self) -> None:
        payload = load_yaml(SCENARIO_DIRECTORY / "p2b_approval_amount_tampering.yaml")
        assert isinstance(payload, dict)
        approved = payload["approved_effect"]
        assert isinstance(approved, dict)
        approved["amount_cents"] = 5000.0

        with self.assertRaises(ValidationError):
            ScenarioSpec.model_validate(payload)

    def test_declared_mutation_must_match_actual_proposal(self) -> None:
        payload = load_yaml(SCENARIO_DIRECTORY / "p2b_approval_amount_tampering.yaml")
        assert isinstance(payload, dict)
        mutation = payload["mutation_expectation"]
        assert isinstance(mutation, dict)
        mutation["changed_fields"] = ["order_id"]

        with self.assertRaises(ValidationError):
            ScenarioSpec.model_validate(payload)

    def test_authorization_replay_and_idempotent_retry_are_distinct_contracts(self) -> None:
        self.assertIsNot(
            ReasonCode.AUTHORIZATION_ALREADY_CONSUMED,
            ReasonCode.IDEMPOTENT_RESULT_REUSED,
        )
        self.assertNotEqual(
            ReasonCode.AUTHORIZATION_ALREADY_CONSUMED.value,
            ReasonCode.IDEMPOTENT_RESULT_REUSED.value,
        )

    def test_fault_and_profile_enums_freeze_required_values(self) -> None:
        self.assertEqual(
            set(FaultPoint),
            {
                FaultPoint.BEFORE_EXTERNAL_SEND,
                FaultPoint.AFTER_EXTERNAL_COMMIT_BEFORE_LOCAL_RECEIPT,
                FaultPoint.DURING_RECONCILIATION_QUERY,
            },
        )
        self.assertEqual(
            set(ReconciliationStatus),
            {
                ReconciliationStatus.COMMITTED,
                ReconciliationStatus.NOT_COMMITTED,
                ReconciliationStatus.UNKNOWN,
            },
        )
        self.assertIn(ScenarioProfile.FULL_AAR, set(ScenarioProfile))
        self.assertIn(ScenarioProfile.NO_RECONCILE_BLIND_RETRY, set(ScenarioProfile))

    def test_effect_rejects_unrelated_fields(self) -> None:
        with self.assertRaises(ValidationError):
            EffectSpec.model_validate(
                {
                    "operation": "GET_ORDER",
                    "order_id": "O100",
                    "amount_cents": 10,
                }
            )


if __name__ == "__main__":
    unittest.main()
