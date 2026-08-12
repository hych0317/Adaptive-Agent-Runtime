from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from adaptive_agent_runtime.core import (
    AgentState,
    AgentTask,
    PlanDecisionType,
    RunStatus,
)

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.models import (
    TerminalExecResult,
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalCommandRole,
    TerminalExecutionPolicy,
    TerminalEvidenceAssurance,
    TerminalEvidenceProvenance,
    TerminalExecutionState,
    TerminalRequirementKind,
    TerminalRequirementState,
    TerminalSessionSnapshot,
    utc_now,
)
from applications.terminal_bench.planner import (
    JsonlTerminalTrialJournal,
    _task_requirements,
)
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    completed_result,
    execute_draft,
    verify_draft,
)


class TerminalTaskContractTests(unittest.TestCase):
    def test_explicit_markdown_sections_are_classified_conservatively(self) -> None:
        requirements = _task_requirements(
            """# Must achieve
- Parse every record.
# Must produce:
- /app/result.json
## Must preserve
- /app/input.log
### Must not do
- Do not contact the network.
#### Thresholds
- Coverage must be at least 95%.
# Notes
- This ordinary note must not acquire a privileged kind.
"""
        )

        self.assertEqual(
            [item.kind for item in requirements],
            [
                TerminalRequirementKind.ACHIEVE,
                TerminalRequirementKind.PRODUCE,
                TerminalRequirementKind.PRESERVE,
                TerminalRequirementKind.PROHIBIT,
                TerminalRequirementKind.THRESHOLD,
                TerminalRequirementKind.UNCLASSIFIED,
            ],
        )
        self.assertEqual(requirements[1].source_section, "Must produce")
        self.assertIsNone(requirements[-1].source_section)
        self.assertEqual(
            [item.requirement_id for item in requirements],
            [f"req-{index:03d}" for index in range(1, 7)],
        )

    def test_plain_language_remains_unclassified(self) -> None:
        requirements = _task_requirements(
            "Create /app/output and preserve all unrelated files."
        )

        self.assertEqual(len(requirements), 1)
        self.assertEqual(
            requirements[0].kind,
            TerminalRequirementKind.UNCLASSIFIED,
        )
        self.assertIsNone(requirements[0].source_section)

    def test_non_heading_hash_lines_keep_legacy_skip_behavior(self) -> None:
        requirements = _task_requirements(
            "#!/usr/bin/env bash\n#define MODE 1\n#Title\nCreate output."
        )

        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].description, "Create output.")
        self.assertEqual(
            requirements[0].kind,
            TerminalRequirementKind.UNCLASSIFIED,
        )

    def test_contract_binding_is_idempotent_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.jsonl"
            journal = JsonlTerminalTrialJournal("trial-contract", path)
            requirements = _task_requirements("Create one output.")

            journal.bind_task_contract(requirements)
            journal.bind_task_contract(requirements)

            events = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(events), 1)
            ledger = journal.snapshot().task_ledger
            contract = journal.task_contract()
            self.assertIsNotNone(ledger)
            self.assertIsNotNone(contract)
            assert ledger is not None
            assert contract is not None
            self.assertEqual(contract.requirements, requirements)
            self.assertEqual(
                tuple(item.requirement_id for item in ledger.entries),
                tuple(item.requirement_id for item in requirements),
            )
            event = json.loads(events[0])
            self.assertEqual(
                event["payload"]["contract"]["requirements"][0][
                    "description"
                ],
                "Create one output.",
            )
            self.assertNotIn(
                "Create one output.",
                ledger.model_dump_json(),
            )

            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                journal.bind_task_contract(
                    _task_requirements("Create a different output.")
                )
            self.assertFalse(journal.trace_consistent)

    def test_legacy_session_and_transcript_without_ledger_still_load(self) -> None:
        session = TerminalSessionSnapshot.model_validate(
            {"trial_id": "legacy-session"}
        )
        self.assertIsNone(session.task_ledger)
        self.assertIsNone(session.latest_verification_receipt)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "kind": "inference.usage",
                        "payload": {"session": {"trial_id": "legacy-trial"}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            journal = JsonlTerminalTrialJournal("legacy-trial", path)
            self.assertTrue(journal.trace_consistent)
            self.assertIsNone(journal.snapshot().task_ledger)


class TerminalTaskLedgerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_verification_creates_receipt_and_locks_ledger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("printf output", cwd="/app"),
                verify_draft("test -n output", cwd="/app"),
            )
            app = build_terminal_application(
                trial_id="trial-ledger-success",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="output"),
                    completed_result(stdout="verified"),
                ),
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("Create one output.")
                session = app.journal.snapshot()
                ledger = session.task_ledger
                receipt = session.latest_verification_receipt

                self.assertTrue(artifacts.summary.agent_complete)
                self.assertEqual(len(capability.requests), 2)
                self.assertIsNotNone(ledger)
                self.assertIsNotNone(receipt)
                assert ledger is not None
                assert receipt is not None
                checkpoint = session.verified_checkpoint
                assert checkpoint is not None
                self.assertTrue(receipt.passed)
                self.assertEqual(session.task_generation, 1)
                self.assertEqual(session.successful_work_generation, 1)
                self.assertEqual(receipt.task_generation, 1)
                self.assertEqual(ledger.generation, 1)
                self.assertEqual(
                    tuple(item.state for item in ledger.entries),
                    (TerminalRequirementState.SATISFIED,),
                )
                self.assertIs(
                    ledger.entries[0].assurance, TerminalEvidenceAssurance.TRUSTED
                )
                self.assertEqual(
                    ledger.entries[0].latest_evidence_action_id,
                    receipt.action_id,
                )
                self.assertEqual(ledger.entries[0].evidence_generation, 1)
                self.assertEqual(
                    checkpoint.receipt,
                    receipt,
                )
                self.assertIsNone(app.journal.completion_gate_error())
            finally:
                app.close()

    async def test_legacy_successful_checkpoint_is_migrated_from_records(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-ledger-migration",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="output"),
                    completed_result(stdout="verified"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("printf output", cwd="/app"),
                    verify_draft("true", cwd="/app"),
                ),
            )
            try:
                await app.run("Create one output.")
                session = app.journal.snapshot()
                checkpoint = session.verified_checkpoint
                assert checkpoint is not None
                app.journal._session = session.model_copy(
                    update={
                        "task_ledger": None,
                        "latest_verification_receipt": None,
                        "verified_checkpoint": checkpoint.model_copy(
                            update={"receipt": None}
                        ),
                    }
                )
                app.journal._task_contract = None

                app.journal.bind_task_contract(
                    _task_requirements("Create one output.")
                )

                migrated = app.journal.snapshot()
                self.assertIsNotNone(migrated.task_ledger)
                self.assertIsNotNone(migrated.latest_verification_receipt)
                assert migrated.verified_checkpoint is not None
                self.assertIsNotNone(migrated.verified_checkpoint.receipt)
                self.assertIsNone(app.journal.completion_gate_error())
            finally:
                app.close()

    async def test_failed_verification_does_not_verify_requirement_ledger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-ledger-failure",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="output"),
                    completed_result(return_code=1, stderr="not valid"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("printf output", cwd="/app"),
                    verify_draft("false", cwd="/app"),
                ),
                policy=TerminalExecutionPolicy(max_commands=2),
            )
            try:
                artifacts = await app.run("Create one output.")
                session = app.journal.snapshot()
                ledger = session.task_ledger
                receipt = session.latest_verification_receipt

                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIsNotNone(ledger)
                self.assertIsNotNone(receipt)
                assert ledger is not None
                assert receipt is not None
                self.assertFalse(receipt.passed)
                self.assertEqual(
                    tuple(item.state for item in ledger.entries),
                    (TerminalRequirementState.UNKNOWN,),
                )
                self.assertIsNone(ledger.entries[0].latest_evidence_action_id)
                self.assertEqual(
                    session.pending_repair_receipt_id,
                    receipt.action_id,
                )
                self.assertIsNone(session.repair_applied_action_id)
                self.assertIsNone(session.verified_checkpoint)
                self.assertIsNotNone(app.journal.completion_gate_error())
            finally:
                app.close()

    async def test_timed_out_zero_exit_does_not_create_success_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            now = utc_now()
            timed_out_zero = TerminalExecResult(
                stdout="late success",
                return_code=0,
                started_at=now,
                completed_at=now,
                duration_ms=0,
                execution_state=TerminalExecutionState.COMPLETED,
                timed_out=True,
            )
            app = build_terminal_application(
                trial_id="trial-ledger-timeout",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="output"),
                    timed_out_zero,
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("printf output", cwd="/app"),
                    verify_draft("true", cwd="/app"),
                ),
                policy=TerminalExecutionPolicy(max_commands=2),
            )
            try:
                artifacts = await app.run("Create one output.")
                session = app.journal.snapshot()

                self.assertFalse(artifacts.summary.agent_complete)
                observation = artifacts.runtime_result.final_state.last_observation
                assert observation is not None
                self.assertFalse(observation.succeeded)
                self.assertNotEqual(
                    observation.control.progress_kind.value,
                    "task_progress",
                )
                self.assertIsNone(session.verified_checkpoint)
                self.assertIsNotNone(session.latest_verification_receipt)
                assert session.latest_verification_receipt is not None
                self.assertFalse(session.latest_verification_receipt.passed)
                assert session.task_ledger is not None
                self.assertEqual(
                    session.task_ledger.entries[0].state,
                    TerminalRequirementState.UNKNOWN,
                )
            finally:
                app.close()

    async def test_work_after_failed_verification_invalidates_blocked_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-ledger-repair",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="initial"),
                    completed_result(return_code=1, stderr="broken"),
                    completed_result(stdout="repaired"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("printf initial", cwd="/app"),
                    verify_draft("false", cwd="/app"),
                    execute_draft(
                        "printf repaired",
                        call_key="repair-1",
                        cwd="/app",
                    ),
                ),
                policy=TerminalExecutionPolicy(max_commands=3),
            )
            try:
                await app.run("Create one output.")
                session = app.journal.snapshot()
                assert session.task_ledger is not None

                self.assertEqual(
                    session.task_ledger.entries[0].state,
                    TerminalRequirementState.UNKNOWN,
                )
                self.assertIsNone(
                    session.task_ledger.entries[0].latest_evidence_action_id
                )
                self.assertIsNone(session.verified_checkpoint)
                self.assertIsNotNone(session.pending_repair_receipt_id)
                self.assertEqual(
                    session.repair_applied_action_id,
                    app.journal.records()[-1].action_id,
                )
                self.assertEqual(session.task_generation, 2)
                self.assertEqual(session.successful_work_generation, 2)
            finally:
                app.close()

    def test_failed_to_start_work_preserves_generation_and_ledger(self) -> None:
        journal = JsonlTerminalTrialJournal("trial-work-not-started")
        journal.bind_task_contract(_task_requirements("Create one output."))
        session = self._verified_shadow_session(journal.snapshot())
        result = self._failed_to_start_result()

        advanced = journal._advance_session(
            session,
            self._work_record(journal.trial_id, result),
        )

        self.assertEqual(advanced.task_generation, 0)
        self.assertEqual(advanced.known_state_generation, 0)
        self.assertEqual(advanced.successful_work_generation, 0)
        assert advanced.task_ledger is not None
        self.assertEqual(advanced.task_ledger.generation, 0)
        self.assertIs(
            advanced.task_ledger.entries[0].state,
            TerminalRequirementState.SATISFIED,
        )
        self.assertEqual(advanced.task_ledger.entries[0].evidence_generation, 0)

    def test_nonzero_work_advances_generation_and_invalidates_ledger(self) -> None:
        journal = JsonlTerminalTrialJournal("trial-work-nonzero")
        journal.bind_task_contract(_task_requirements("Create one output."))
        session = self._verified_shadow_session(journal.snapshot())

        advanced = journal._advance_session(
            session,
            self._work_record(
                journal.trial_id,
                completed_result(return_code=1, stderr="partial change"),
            ),
        )

        self.assertEqual(advanced.task_generation, 1)
        self.assertEqual(advanced.known_state_generation, 1)
        self.assertIsNone(advanced.successful_work_generation)
        assert advanced.task_ledger is not None
        self.assertEqual(advanced.task_ledger.generation, 1)
        self.assertIs(
            advanced.task_ledger.entries[0].state,
            TerminalRequirementState.UNKNOWN,
        )
        self.assertIsNone(advanced.task_ledger.entries[0].evidence_generation)

    async def test_unmigratable_legacy_checkpoint_fails_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("printf output", cwd="/app"),
                verify_draft("true", cwd="/app"),
            )
            app = build_terminal_application(
                trial_id="trial-ledger-invalid-migration",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="output"),
                    completed_result(stdout="verified"),
                ),
                proposal_capability=capability,
            )
            try:
                await app.run("Create one output.")
                session = app.journal.snapshot()
                checkpoint = session.verified_checkpoint
                assert checkpoint is not None
                app.journal._records = app.journal._records[:-1]
                app.journal._session = session.model_copy(
                    update={
                        "task_ledger": None,
                        "latest_verification_receipt": None,
                        "verified_checkpoint": checkpoint.model_copy(
                            update={"receipt": None}
                        ),
                    }
                )
                app.journal._task_contract = None

                decision = await app.runtime._planner.plan(
                    AgentState(
                        run_id=uuid4(),
                        task=AgentTask(description="Create one output."),
                        status=RunStatus.RUNNING,
                    )
                )

                self.assertIs(decision.decision, PlanDecisionType.FAIL)
                self.assertIn("trace is inconsistent", decision.error or "")
                self.assertFalse(app.journal.trace_consistent)
                self.assertEqual(len(capability.requests), 2)
            finally:
                app.close()

    async def test_completion_gate_rejects_a_tampered_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-ledger-tamper",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="output"),
                    completed_result(stdout="verified"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("printf output", cwd="/app"),
                    verify_draft("true", cwd="/app"),
                ),
            )
            try:
                await app.run("Create one output.")
                session = app.journal.snapshot()
                receipt = session.latest_verification_receipt
                checkpoint = session.verified_checkpoint
                assert receipt is not None
                assert checkpoint is not None
                tampered = receipt.model_copy(
                    update={"result_fingerprint": "0" * 64}
                )
                app.journal._session = session.model_copy(
                    update={
                        "latest_verification_receipt": tampered,
                        "verified_checkpoint": checkpoint.model_copy(
                            update={"receipt": tampered}
                        ),
                    }
                )

                self.assertIn(
                    "result fingerprint",
                    app.journal.completion_gate_error() or "",
                )
            finally:
                app.close()

    @staticmethod
    def _verified_shadow_session(
        session: TerminalSessionSnapshot,
    ) -> TerminalSessionSnapshot:
        assert session.task_ledger is not None
        evidence_action_id = uuid4()
        ledger = session.task_ledger.model_copy(
            update={
                "entries": tuple(
                    entry.model_copy(
                        update={
                            "state": TerminalRequirementState.SATISFIED,
                            "assurance": TerminalEvidenceAssurance.TRUSTED,
                            "evidence_provenance": TerminalEvidenceProvenance.TASK_PROVIDED,
                            "latest_evidence_action_id": evidence_action_id,
                            "latest_result_fingerprint": "1" * 64,
                            "evidence_generation": 0,
                        }
                    )
                    for entry in session.task_ledger.entries
                )
            }
        )
        return session.model_copy(
            update={
                "task_ledger": ledger,
                "known_state_generation": 0,
                "successful_work_generation": 0,
            }
        )

    @staticmethod
    def _work_record(
        trial_id: str,
        result: TerminalExecResult,
    ) -> TerminalCommandRecord:
        return TerminalCommandRecord(
            action_id=uuid4(),
            invocation_id=uuid4(),
            decision_request_id=uuid4(),
            effect_fingerprint="0" * 64,
            intent=TerminalCommandIntent(
                trial_id=trial_id,
                call_key="work-boundary",
                command="change-output",
                command_role=TerminalCommandRole.WORK,
                timeout_sec=30,
            ),
            result=result,
            governance_status="applied",
        )

    @staticmethod
    def _failed_to_start_result() -> TerminalExecResult:
        now = utc_now()
        return TerminalExecResult(
            stderr="provider unavailable",
            started_at=now,
            completed_at=now,
            duration_ms=0,
            execution_state=TerminalExecutionState.FAILED_TO_START,
            transport_failed=True,
        )


if __name__ == "__main__":
    unittest.main()
