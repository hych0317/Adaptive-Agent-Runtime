from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from adaptive_agent_runtime.decisioning import (
    DecisionCheckpoint,
    DecisionResultStatus,
    DecisionReconciliationStatus,
    decision_fingerprint,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from tests.decisioning.fakes import (
    FakeEffectPayload,
    FakeProposalPayload,
    FakeRequestPayload,
)


CHECKPOINT_TYPE = DecisionCheckpoint[
    FakeRequestPayload,
    FakeProposalPayload,
    FakeEffectPayload,
]


class CrossProcessDecisionResumeTests(unittest.TestCase):
    def _worker(self, database: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.decisioning.process_resume_worker",
                "--database",
                str(database),
                *arguments,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _facts(database: Path) -> dict[str, object]:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            checkpoint_row = connection.execute(
                "SELECT request_id FROM decision_current"
            ).fetchone()
            assert checkpoint_row is not None
            request_id = UUID(str(checkpoint_row["request_id"]))
            invocation = connection.execute(
                "SELECT calls FROM closeout_agent_invocations"
            ).fetchone()
            commit = connection.execute(
                "SELECT commit_count FROM closeout_effect_commits"
            ).fetchone()
            attempt = connection.execute(
                "SELECT attempts FROM closeout_apply_attempts"
            ).fetchone()
            trace_kinds = tuple(
                json.loads(row["entry_json"])["event"]["kind"]
                for row in connection.execute(
                    "SELECT entry_json FROM runtime_trace ORDER BY sequence"
                ).fetchall()
            )
            facts = {
                "agent_calls": int(invocation["calls"]) if invocation else 0,
                "commit_count": int(commit["commit_count"]) if commit else 0,
                "apply_attempts": int(attempt["attempts"]) if attempt else 0,
                "trace_kinds": trace_kinds,
            }
        finally:
            connection.close()
        persistence = SQLitePersistence(database)
        try:
            checkpoint = asyncio.run(
                persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE).load(
                    request_id
                )
            )
        finally:
            persistence.close()
        assert checkpoint is not None
        return {"checkpoint": checkpoint, **facts}

    def _assert_identity_stable(self, before, after) -> None:  # type: ignore[no-untyped-def]
        assert before.proposal is not None and after.proposal is not None
        assert before.validated_decision is not None
        assert after.validated_decision is not None
        assert before.governance_receipt is not None
        assert after.governance_receipt is not None
        self.assertEqual(
            decision_fingerprint(before.proposal),
            decision_fingerprint(after.proposal),
        )
        self.assertEqual(
            before.validated_decision.normalized_effect.effect_fingerprint,
            after.validated_decision.normalized_effect.effect_fingerprint,
        )
        if before.governance_receipt.authorization_id is not None:
            self.assertEqual(
                before.governance_receipt.authorization_id,
                after.governance_receipt.authorization_id,
            )
        if before.governance_receipt.review_request_id is not None:
            self.assertEqual(
                before.governance_receipt.review_request_id,
                after.governance_receipt.review_request_id,
            )

    def test_resume_at_authorized_applying_and_committed_boundaries(self) -> None:
        for fault in ("authorized", "applying", "effect_committed"):
            with self.subTest(fault=fault), TemporaryDirectory() as directory:
                database = Path(directory) / "runtime.sqlite3"
                crashed = self._worker(
                    database, "--action", "start", "--fault", fault
                )
                self.assertEqual(crashed.returncode, 91, crashed.stderr)
                before = self._facts(database)
                resumed = self._worker(database, "--action", "resume")
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
                after = self._facts(database)
                checkpoint = after["checkpoint"]
                assert isinstance(checkpoint, DecisionCheckpoint)
                assert checkpoint.result is not None
                self.assertEqual(checkpoint.result.status, DecisionResultStatus.APPLIED)
                self.assertEqual(after["agent_calls"], 1)
                self.assertEqual(after["commit_count"], 1)
                expected_attempts = 1
                self.assertEqual(after["apply_attempts"], expected_attempts)
                self.assertEqual(
                    after["trace_kinds"],
                    (
                        "decision.requested",
                        "decision.context_projected",
                        "decision.proposed",
                        "decision.validation_passed",
                        "decision.governance_requested",
                        "decision.authorized",
                        "decision.apply_started",
                        "decision.applied",
                    ),
                )
                self._assert_identity_stable(before["checkpoint"], checkpoint)

    def test_human_review_pending_survives_process_and_preserves_identity(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            crashed = self._worker(
                database,
                "--action",
                "start",
                "--high-risk",
                "--fault",
                "review_pending",
            )
            self.assertEqual(crashed.returncode, 91, crashed.stderr)
            before = self._facts(database)
            resumed = self._worker(
                database, "--action", "approve", "--high-risk"
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            after = self._facts(database)
            checkpoint = after["checkpoint"]
            assert isinstance(checkpoint, DecisionCheckpoint)
            assert checkpoint.result is not None
            self.assertEqual(checkpoint.result.status, DecisionResultStatus.APPLIED)
            self.assertEqual(after["agent_calls"], 1)
            self.assertEqual(after["commit_count"], 1)
            self.assertIn("decision.review_required", after["trace_kinds"])
            self.assertEqual(after["trace_kinds"][-1], "decision.applied")
            self._assert_identity_stable(before["checkpoint"], checkpoint)

    def test_unknown_is_durable_and_never_retries_non_idempotent_apply(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "runtime.sqlite3"
            crashed = self._worker(
                database,
                "--action",
                "start",
                "--unknown",
                "--fault",
                "applying",
            )
            self.assertEqual(crashed.returncode, 91, crashed.stderr)
            first_resume = self._worker(
                database, "--action", "resume", "--unknown"
            )
            self.assertEqual(first_resume.returncode, 0, first_resume.stderr)
            first = self._facts(database)
            checkpoint = first["checkpoint"]
            assert isinstance(checkpoint, DecisionCheckpoint)
            assert checkpoint.result is not None
            self.assertEqual(checkpoint.result.status, DecisionResultStatus.FAILED)
            self.assertEqual(
                checkpoint.result.reconciliation_status,
                DecisionReconciliationStatus.UNKNOWN,
            )
            second_resume = self._worker(
                database, "--action", "resume", "--unknown"
            )
            self.assertEqual(second_resume.returncode, 0, second_resume.stderr)
            second = self._facts(database)
            self.assertEqual(second["agent_calls"], 1)
            self.assertEqual(second["commit_count"], 0)
            self.assertEqual(second["apply_attempts"], 0)
            self.assertEqual(second["trace_kinds"][-1], "decision.failed")
            second_checkpoint = second["checkpoint"]
            assert isinstance(second_checkpoint, DecisionCheckpoint)
            assert second_checkpoint.result is not None
            self.assertEqual(
                second_checkpoint.result.reconciliation_status,
                DecisionReconciliationStatus.UNKNOWN,
            )


if __name__ == "__main__":
    unittest.main()
