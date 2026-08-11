from __future__ import annotations

import tempfile
import unittest

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.contracts import TerminalExecutionError
from applications.terminal_bench.models import (
    TerminalCommandRole,
    TerminalCompletionDisposition,
    TerminalEvidenceAssurance,
    TerminalEvidenceProvenance,
    TerminalExecutionPolicy,
    TerminalRequirementState,
    TerminalVerificationContract,
)
from applications.terminal_bench.planner import (
    _contract_coverage_gaps,
    _task_requirements,
    _terminal_behavior_hints,
    _terminal_model_payload,
    _verification_assurance,
)
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    completed_result,
    execute_draft,
    verify_draft,
)


def self_check_contract() -> TerminalVerificationContract:
    return TerminalVerificationContract(
        evidence_kind="independent_check",
        evidence_provenance="agent_generated",
        evidence_sources=("agent-created synthetic fixture",),
        artifact_paths=("/app/result.json",),
        artifact_fingerprints=(
            "/app/result.json=sha256:" + "1" * 64,
        ),
        requirement_coverage=("req-001",),
        coverage_dimensions=(
            "artifact",
            "format",
            "semantic",
            "end_to_end",
        ),
        validation_methods=("synthetic end-to-end assertion",),
    )


class TerminalVerificationAssuranceTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_generated_check_submits_without_success_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-result", cwd="/app"),
                verify_draft(
                    "check-synthetic-result",
                    cwd="/app",
                    verification=self_check_contract(),
                ),
            )
            environment = FakeTerminalEnvironment(
                completed_result(stdout="created"),
                completed_result(stdout="self-check passed"),
            )
            app = build_terminal_application(
                trial_id="trial-self-checked-submission",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("Create one result.")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertIsNone(session.verified_checkpoint)
                self.assertEqual(len(capability.requests), 2)
                assert session.task_ledger is not None
                entry = session.task_ledger.entries[0]
                self.assertIs(entry.state, TerminalRequirementState.SATISFIED)
                self.assertIs(
                    entry.assurance,
                    TerminalEvidenceAssurance.SELF_CHECKED,
                )
                self.assertEqual(
                    entry.artifact_fingerprints,
                    ("/app/result.json=sha256:" + "b" * 64,),
                )
                self.assertNotEqual(
                    entry.artifact_fingerprints,
                    self_check_contract().artifact_fingerprints,
                )
                self.assertEqual(len(environment.runtime_evidence_calls), 1)
            finally:
                app.close()

    async def test_runtime_probe_cannot_extend_verify_role_timeout(self) -> None:
        exhausted_verify = completed_result(stdout="verified").model_copy(
            update={"duration_ms": 120_000}
        )
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-result", cwd="/app"),
                verify_draft(
                    "check-result",
                    cwd="/app",
                    timeout_sec=120,
                    verification=self_check_contract(),
                ),
            )
            environment = FakeTerminalEnvironment(
                completed_result(stdout="created"),
                exhausted_verify,
            )
            app = build_terminal_application(
                trial_id="trial-runtime-evidence-budget",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("Create one result.")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(environment.runtime_evidence_calls, [])
                receipt = session.latest_verification_receipt
                assert receipt is not None
                self.assertIs(
                    receipt.assurance,
                    TerminalEvidenceAssurance.SELF_CHECKED,
                )
            finally:
                app.close()

    async def test_agent_test_cannot_masquerade_as_task_provided(self) -> None:
        verification = TerminalVerificationContract(
            evidence_kind="official_tests",
            evidence_provenance="task_provided",
            evidence_sources=("/tests/claimed-official.py",),
            artifact_paths=("/app/result.json",),
            artifact_fingerprints=("/app/result.json=sha256:" + "1" * 64,),
            requirement_coverage=("req-001",),
            coverage_dimensions=("artifact", "format", "semantic", "end_to_end"),
            validation_methods=("claimed official test",),
        )
        probe = "\n".join(
            (
                "AAR_RUNTIME_EVIDENCE_V1",
                "S\t0\t/app/agent-written-test.py\t0\t" + "a" * 64,
                "A\t0\t" + "b" * 64,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(),
                completed_result(),
                runtime_evidence_stdout=probe,
            )
            app = build_terminal_application(
                trial_id="trial-false-task-provenance",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("create-result"),
                    verify_draft("agent-test", verification=verification),
                ),
            )
            try:
                artifacts = await app.run("Create one result.")
                receipt = app.journal.snapshot().latest_verification_receipt
                assert receipt is not None
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    receipt.assurance,
                    TerminalEvidenceAssurance.SELF_CHECKED,
                )
                self.assertIs(
                    receipt.evidence_provenance,
                    TerminalEvidenceProvenance.RUNTIME_OBSERVED,
                )
                self.assertEqual(
                    receipt.artifact_fingerprints,
                    ("/app/result.json=sha256:" + "b" * 64,),
                )
                self.assertIsNone(app.journal.snapshot().verified_checkpoint)
            finally:
                app.close()

    async def test_agent_tool_cannot_masquerade_as_external_standard(self) -> None:
        verification = self_check_contract().model_copy(
            update={
                "evidence_provenance": TerminalEvidenceProvenance.EXTERNAL_STANDARD,
                "evidence_sources": ("/usr/bin/claimed-standard",),
            }
        )
        probe = "\n".join(
            (
                "AAR_RUNTIME_EVIDENCE_V1",
                "S\t0\t/app/agent-tool\t0\t" + "a" * 64,
                "A\t0\t" + "b" * 64,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-false-external-provenance",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                    runtime_evidence_stdout=probe,
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("create-result"),
                    verify_draft("agent-tool", verification=verification),
                ),
            )
            try:
                artifacts = await app.run("Create one result.")
                receipt = app.journal.snapshot().latest_verification_receipt
                assert receipt is not None
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    receipt.assurance,
                    TerminalEvidenceAssurance.SELF_CHECKED,
                )
                self.assertIs(
                    receipt.evidence_provenance,
                    TerminalEvidenceProvenance.RUNTIME_OBSERVED,
                )
            finally:
                app.close()

    async def test_reconciliation_rejections_have_an_independent_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("uncertain-work", call_key="work-1"),
                execute_draft(
                    "rm -f /tmp/uncertain.pid",
                    call_key="bad-reconcile",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                execute_draft(
                    "test ! -e /tmp/uncertain.pid && test -e /app",
                    call_key="good-reconcile",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                verify_draft("true", call_key="verify-stable"),
            )
            app = build_terminal_application(
                trial_id="trial-reconciliation-rejection-budget",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    TerminalExecutionError(
                        "connection lost",
                        command_started=True,
                    ),
                    completed_result(stdout="stopped and stable"),
                    completed_result(stdout="verified"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_reconciliation_proposal_rejections=1,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("Create and verify one artifact.")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(session.proposal_rejections, 0)
                self.assertEqual(session.reconciliation_proposal_rejections, 1)
                self.assertIsNotNone(session.latest_reconciliation_receipt)
                self.assertEqual(len(app.journal.records()), 3)
                self.assertTrue(capability.requests[3].verification_due)
            finally:
                app.close()

    async def test_model_payload_uses_ledger_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-result"),
                verify_draft("true"),
            )
            app = build_terminal_application(
                trial_id="trial-ledger-projection",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("Create exact output in /app/result.")
                self.assertTrue(artifacts.runtime_result.succeeded)
                request = capability.requests[1]
                payload = _terminal_model_payload(request)
                session_payload = payload["session"]
                assert isinstance(session_payload, dict)
                self.assertNotIn("task_ledger", session_payload)
                self.assertGreaterEqual(len(payload["ledger_projection"]), 1)
                self.assertTrue(
                    any(
                        "exact-output" in hint
                        for hint in request.behavior_hints
                    )
                )
            finally:
                app.close()

    def test_behavior_hints_are_conditional_and_task_agnostic(self) -> None:
        requirements = _task_requirements(
            "Parse each line into one of the allowed labels and produce exact output."
        )
        hints = _terminal_behavior_hints(requirements)

        self.assertEqual(len(hints), 3)
        self.assertTrue(any("first failing line" in hint for hint in hints))
        self.assertTrue(any("expected and actual bytes" in hint for hint in hints))
        self.assertTrue(any("complete allowed set" in hint for hint in hints))
        self.assertFalse(
            any("terminal-bench" in hint.lower() for hint in hints)
        )

    def test_task_provided_independent_claim_is_not_automatically_trusted(
        self,
    ) -> None:
        verification = self_check_contract().model_copy(
            update={
                "evidence_provenance": TerminalEvidenceProvenance.TASK_PROVIDED
            }
        )

        self.assertIs(
            _verification_assurance(verification),
            TerminalEvidenceAssurance.SELF_CHECKED,
        )

    async def test_incomplete_contract_cannot_create_success_lock(self) -> None:
        instruction = (
            "# Must produce /app/result.json\n"
            "Do not modify /app/input.txt."
        )
        requirements = _task_requirements(instruction)
        gaps = _contract_coverage_gaps(instruction, requirements)
        self.assertEqual(gaps, ("# Must produce /app/result.json",))
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-incomplete-contract",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("preserve-input-and-create-result"),
                    verify_draft("true"),
                ),
            )
            try:
                artifacts = await app.run(instruction)
                session = app.journal.snapshot()
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertFalse(session.contract_coverage_complete)
                self.assertEqual(session.contract_unmapped_fragments, gaps)
                self.assertIsNone(session.verified_checkpoint)
            finally:
                app.close()
