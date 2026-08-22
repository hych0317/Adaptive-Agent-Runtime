from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from applications.governance_scenario_suite.contracts import (
    AuditEventType,
    EvaluationVerdict,
    ReasonCode,
    ScenarioProfile,
    ScenarioSpec,
    ScenarioVerdict,
)
from applications.governance_scenario_suite.evidence import (
    EvidenceSource,
    ScenarioExecution,
)
from applications.governance_scenario_suite.loader import (
    ScenarioLoadError,
    load_scenario,
    load_scenario_directory,
)
from applications.governance_scenario_suite.reporting import (
    build_batch_result,
    write_reports,
)
from applications.governance_scenario_suite.runner import (
    GovernanceScenarioRunner,
    ScenarioRuntimeContext,
)
from applications.governance_scenario_suite.variants import (
    ScenarioExecutorRegistry,
)


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "applications"
    / "governance_scenario_suite"
    / "scenarios"
)


class RejectTamperingExecutor:
    profile = ScenarioProfile.FULL_AAR

    def __init__(self, projected_context: str = "scoped order summary") -> None:
        self._projected_context = projected_context

    def execute(self, context: object) -> ScenarioExecution:
        assert isinstance(context, ScenarioRuntimeContext)
        assert context.approval_id is not None
        proposal = context.model.propose(self._projected_context)
        context.composition.store.append_audit(
            occurred_at=context.scenario.clock,
            event_type=AuditEventType.PROPOSAL_RECORDED,
            effect_fingerprint=_fingerprint(proposal),
        )
        context.composition.store.append_audit(
            occurred_at=context.scenario.clock,
            event_type=AuditEventType.EFFECT_MISMATCH_REJECTED,
            reason_code=ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
            effect_fingerprint=_fingerprint(proposal),
        )
        return ScenarioExecution(
            decision=ScenarioVerdict.REJECT,
            reason_code=ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
            model_contexts=context.model.contexts,
            available_sources=frozenset(
                {EvidenceSource.AUDIT, EvidenceSource.MODEL_CONTEXT}
            ),
            model_call_count=context.model.calls,
        )


class MissingAuditExecutor:
    profile = ScenarioProfile.FULL_AAR

    def execute(self, context: object) -> ScenarioExecution:
        assert isinstance(context, ScenarioRuntimeContext)
        context.model.propose("scoped order summary")
        return ScenarioExecution(
            decision=ScenarioVerdict.REJECT,
            reason_code=ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
            model_contexts=context.model.contexts,
            available_sources=frozenset({EvidenceSource.MODEL_CONTEXT}),
            model_call_count=context.model.calls,
        )


class RejectAllExecutor:
    profile = ScenarioProfile.FULL_AAR

    def execute(self, context: object) -> ScenarioExecution:
        assert isinstance(context, ScenarioRuntimeContext)
        context.model.propose("scoped order summary")
        return ScenarioExecution(
            decision=ScenarioVerdict.REJECT,
            reason_code=ReasonCode.EFFECT_NOT_EQUAL_TO_APPROVAL,
            model_contexts=context.model.contexts,
            available_sources=frozenset({EvidenceSource.MODEL_CONTEXT}),
            model_call_count=context.model.calls,
        )


class ExtraExternalEffectsExecutor(RejectTamperingExecutor):
    def execute(self, context: object) -> ScenarioExecution:
        assert isinstance(context, ScenarioRuntimeContext)
        now = context.scenario.clock
        context.composition.gateway.refund(
            order_id="O100",
            amount_cents=1,
            idempotency_key="oracle-corruption-1",
            now=now,
        )
        context.composition.gateway.refund(
            order_id="O100",
            amount_cents=1,
            idempotency_key="oracle-corruption-2",
            now=now,
        )
        return super().execute(context)


def _fingerprint(value: object) -> str:
    from adaptive_agent_runtime.decisioning import decision_fingerprint

    return decision_fingerprint(value)


def _runner(executor: object) -> GovernanceScenarioRunner:
    return GovernanceScenarioRunner(
        executors=ScenarioExecutorRegistry(
            {ScenarioProfile.FULL_AAR: executor}  # type: ignore[dict-item]
        )
    )


class GovernanceInfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dangerous = load_scenario(
            SCENARIO_DIRECTORY / "p2a_approval_target_tampering.yaml"
        )
        self.positive = load_scenario(
            SCENARIO_DIRECTORY / "p2_approved_refund.yaml"
        )

    def test_runner_is_isolated_and_reproducible(self) -> None:
        runner = _runner(RejectTamperingExecutor())

        first = runner.run(self.dangerous)
        second = runner.run(self.dangerous)

        self.assertEqual(first.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.spec_fingerprint, second.spec_fingerprint)
        self.assertEqual(first.evidence, second.evidence)

    def test_oracle_detects_external_effects_even_when_decision_matches(self) -> None:
        result = _runner(ExtraExternalEffectsExecutor()).run(self.dangerous)

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.FAIL)
        self.assertEqual(result.evidence.external_effect_count, 2)
        self.assertTrue(
            any(
                item.code == "EXTERNAL_EFFECT_COUNT"
                and item.verdict is EvaluationVerdict.FAIL
                for item in result.findings
            )
        )

    def test_missing_required_audit_is_inconclusive(self) -> None:
        result = _runner(MissingAuditExecutor()).run(self.dangerous)

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.INCONCLUSIVE)
        self.assertTrue(
            any(item.code == "EVIDENCE_SOURCE_MISSING" for item in result.findings)
        )

    def test_positive_control_exposes_reject_all_executor(self) -> None:
        result = _runner(RejectAllExecutor()).run(self.positive)

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.FAIL)
        self.assertTrue(
            any(
                item.code == "DECISION_MATCH"
                and item.verdict is EvaluationVerdict.FAIL
                for item in result.findings
            )
        )

    def test_forbidden_context_canary_is_detected(self) -> None:
        canary = "PRIVATE-CONTEXT-CANARY-93A7"
        payload = self.dangerous.model_dump(mode="python")
        payload["expected"]["evidence"]["model_context"] = {
            "forbidden_values": [canary]
        }
        scenario = ScenarioSpec.model_validate(payload)

        result = _runner(RejectTamperingExecutor(canary)).run(scenario)

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.FAIL)
        self.assertNotIn(canary, result.model_dump_json())

    def test_strict_loader_rejects_multiple_documents(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("---\na: 1\n---\na: 2\n", encoding="utf-8")

            with self.assertRaisesRegex(ScenarioLoadError, "exactly one"):
                load_scenario(path)

    def test_scenario_id_cannot_be_reused_with_changed_content(self) -> None:
        source = (
            SCENARIO_DIRECTORY / "p2a_approval_target_tampering.yaml"
        ).read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text(source, encoding="utf-8")
            changed = source.replace(
                "title: Approved refund target is changed",
                "title: Reused identity with different content",
            )
            (root / "b.yaml").write_text(changed, encoding="utf-8")

            with self.assertRaisesRegex(ScenarioLoadError, "reused"):
                load_scenario_directory(root)

    def test_reports_are_safe_and_derived_from_typed_results(self) -> None:
        canary = "PAYMENT-TOKEN-CANARY-DO-NOT-PERSIST"
        payload = self.dangerous.model_dump(mode="python")
        payload["initial_authoritative_state"]["payments"] = [
            {
                "payment_id": "PAY100",
                "order_id": "O100",
                "display_reference": "payment ending 42",
                "sensitive_canary": canary,
            }
        ]
        payload["expected"]["evidence"]["audit"]["forbidden_raw_values"] = [
            canary
        ]
        scenario = ScenarioSpec.model_validate(payload)
        result = _runner(RejectTamperingExecutor()).run(scenario)
        batch = build_batch_result(
            (result,),
            suite_id="test-suite",
            generated_at=scenario.clock,
        )

        with TemporaryDirectory() as directory:
            json_path, markdown_path = write_reports(batch, directory)
            run_files = tuple((Path(directory) / "runs").glob("*.json"))
            combined = json_path.read_text(encoding="utf-8") + markdown_path.read_text(
                encoding="utf-8"
            )

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertTrue(batch.release_gate_passed)
        self.assertEqual(len(run_files), 1)
        self.assertNotIn(canary, combined)


if __name__ == "__main__":
    unittest.main()
