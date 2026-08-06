from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import UUID

from adaptive_agent_runtime.decisioning import (
    DecisionApplyReceipt,
    DecisionBasis,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionCorrelation,
    DecisionGovernanceOutcome,
    DecisionGovernanceReceipt,
    DecisionGovernanceScope,
    DecisionModel,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionResult,
    DecisionResultStatus,
    DecisionRiskLevel,
    DecisionTarget,
    DecisionValidation,
    DecisionValidationStatus,
    NormalizedDecisionEffect,
    ValidatedDecision,
    decision_fingerprint,
)
from adaptive_agent_runtime.persistence import (
    PersistenceConflictError,
    SQLitePersistence,
)


NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


class LargeRequest(DecisionModel):
    candidates: tuple[str, ...]


class SmallProposal(DecisionModel):
    selected: int


class LargeEffect(DecisionModel):
    evidence_snapshot: tuple[str, ...]
    selected: int


CHECKPOINT_TYPE = DecisionCheckpoint[LargeRequest, SmallProposal, LargeEffect]


def _checkpoints() -> tuple[DecisionCheckpoint[LargeRequest, SmallProposal, LargeEffect], ...]:
    candidates = tuple(
        f"candidate-{index:02d}-" + (chr(65 + index % 26) * 28_000)
        for index in range(57)
    )
    request_id = UUID(int=1001)
    run_id = UUID(int=1002)
    proposal_id = UUID(int=1003)
    basis = DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(candidates),
        state_revision=1,
    )
    request = DecisionRequest[LargeRequest](
        request_id=request_id,
        decision_type="test.large_evidence",
        target=DecisionTarget(target_type="test", target_id="large-evidence"),
        correlation=DecisionCorrelation(run_id=run_id, task_id=UUID(int=1004)),
        basis=basis,
        payload=LargeRequest(candidates=candidates),
        allowed_actions=("apply",),
        created_at=NOW,
    )
    proposal = DecisionProposal[SmallProposal](
        proposal_id=proposal_id,
        request_id=request_id,
        proposal_type="test.large_evidence",
        producer=DecisionProducer(producer_id="test", capability="test"),
        input_snapshot_fingerprint=basis.snapshot_fingerprint,
        context_fingerprint="1" * 64,
        selected_action="apply",
        payload=SmallProposal(selected=1),
        rationale="Select one candidate without copying the evidence set.",
        confidence=1.0,
        created_at=NOW,
    )
    effect = NormalizedDecisionEffect[LargeEffect].create(
        payload=LargeEffect(evidence_snapshot=candidates, selected=1),
        operation="test.large.apply",
        target=request.target,
        governance_scope=DecisionGovernanceScope.STATE,
        risk=DecisionRiskLevel.LOW,
        impact_score=0.1,
        reversible=True,
        impact_description="Test only.",
    )
    validation = DecisionValidation(
        validation_id=UUID(int=1005),
        request_id=request_id,
        proposal_id=proposal_id,
        status=DecisionValidationStatus.PASSED,
        validated_basis=basis,
        normalized_effect_fingerprint=effect.effect_fingerprint,
        created_at=NOW,
    )
    validated = ValidatedDecision(
        request=request,
        proposal=proposal,
        validation=validation,
        normalized_effect=effect,
    )
    governance = DecisionGovernanceReceipt(
        outcome=DecisionGovernanceOutcome.ALLOW,
        governance_request_id=UUID(int=1006),
        governance_decision_id=UUID(int=1007),
        authorization_id=UUID(int=1008),
        reason="Bounded test authorization.",
        decided_at=NOW,
    )
    commit = DecisionApplyReceipt(
        effect_fingerprint=effect.effect_fingerprint,
        committed_state_fingerprint=decision_fingerprint({"selected": 1}),
        result={"selected": 1},
        applied_at=NOW,
    )
    result = DecisionResult(
        result_id=UUID(int=1009),
        request_id=request_id,
        proposal_id=proposal_id,
        status=DecisionResultStatus.APPLIED,
        reason="Applied once.",
        apply_receipt=commit,
        completed_at=NOW,
    )
    values: list[DecisionCheckpoint[LargeRequest, SmallProposal, LargeEffect]] = []
    checkpoint = CHECKPOINT_TYPE(
        request_id=request_id,
        run_id=run_id,
        revision=0,
        stage=DecisionCheckpointStage.REQUESTED,
        request=request,
        updated_at=NOW,
    )
    values.append(checkpoint)
    for revision, stage, updates in (
        (1, DecisionCheckpointStage.CONTEXT_PROJECTED, {}),
        (2, DecisionCheckpointStage.PROPOSED, {"proposal": proposal}),
        (
            3,
            DecisionCheckpointStage.VALIDATED,
            {"validation": validation, "validated_decision": validated},
        ),
        (4, DecisionCheckpointStage.AUTHORIZED, {"governance_receipt": governance}),
        (5, DecisionCheckpointStage.APPLYING, {}),
        (6, DecisionCheckpointStage.EFFECT_COMMITTED, {"commit_receipt": commit}),
        (
            7,
            DecisionCheckpointStage.COMPLETED,
            {"commit_receipt": commit, "result": result},
        ),
    ):
        checkpoint = checkpoint.model_copy(
            update={
                "revision": revision,
                "stage": stage,
                "updated_at": NOW + timedelta(seconds=revision),
                **updates,
            }
        )
        values.append(checkpoint)
    return tuple(values)


class NormalizedDecisionPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_immutable_effect_and_commit_receipt_are_shared_across_decisions(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            persistence = SQLitePersistence(path)
            store = persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE)
            checkpoints = _checkpoints()
            for index, checkpoint in enumerate(checkpoints):
                await store.save(
                    checkpoint,
                    expected_revision=None if index == 0 else index - 1,
                )

            original = checkpoints[-1]
            assert original.proposal is not None
            assert original.validation is not None
            assert original.validated_decision is not None
            assert original.governance_receipt is not None
            assert original.commit_receipt is not None
            second_request_id = UUID(int=3001)
            second_proposal_id = UUID(int=3002)
            second_run_id = UUID(int=3003)
            second_request = original.request.model_copy(
                update={
                    "request_id": second_request_id,
                    "correlation": DecisionCorrelation(
                        run_id=second_run_id,
                        task_id=UUID(int=3004),
                    ),
                }
            )
            second_proposal = original.proposal.model_copy(
                update={
                    "proposal_id": second_proposal_id,
                    "request_id": second_request_id,
                }
            )
            second_validation = original.validation.model_copy(
                update={
                    "validation_id": UUID(int=3005),
                    "request_id": second_request_id,
                    "proposal_id": second_proposal_id,
                }
            )
            second_validated = ValidatedDecision(
                request=second_request,
                proposal=second_proposal,
                validation=second_validation,
                normalized_effect=original.validated_decision.normalized_effect,
            )
            second_governance = original.governance_receipt.model_copy(
                update={
                    "governance_request_id": UUID(int=3006),
                    "governance_decision_id": UUID(int=3007),
                    "authorization_id": UUID(int=3008),
                }
            )
            second_result = DecisionResult(
                result_id=UUID(int=3009),
                request_id=second_request_id,
                proposal_id=second_proposal_id,
                status=DecisionResultStatus.APPLIED,
                reason="Same authoritative effect reused.",
                apply_receipt=original.commit_receipt,
                completed_at=NOW,
            )
            second = CHECKPOINT_TYPE(
                request_id=second_request_id,
                run_id=second_run_id,
                revision=0,
                stage=DecisionCheckpointStage.COMPLETED,
                request=second_request,
                proposal=second_proposal,
                validation=second_validation,
                validated_decision=second_validated,
                governance_receipt=second_governance,
                commit_receipt=original.commit_receipt,
                result=second_result,
                updated_at=NOW,
            )
            await store.save(second, expected_revision=None)
            with persistence.database.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO runtime_run_metadata "
                    "(run_id, run_kind, disposable, created_at) VALUES (?, 'test', 1, ?)",
                    (str(original.run_id), NOW.isoformat()),
                )
                cursor.execute(
                    "INSERT INTO runtime_run_metadata "
                    "(run_id, run_kind, disposable, created_at) "
                    "VALUES (?, 'runtime', 0, ?)",
                    (str(second_run_id), NOW.isoformat()),
                )
            persistence.disposable_runs.purge_run(original.run_id, dry_run=False)
            self.assertEqual(await store.load(second_request_id), second)
            persistence.close()

            connection = sqlite3.connect(path)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM decision_effects"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM decision_commit_receipts"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM decision_evidence_snapshots"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM decision_results"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM decision_current"
                    ).fetchone()[0],
                    1,
                )
            finally:
                connection.close()

    async def test_large_evidence_is_stored_once_across_all_transitions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            persistence = SQLitePersistence(path)
            store = persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE)
            checkpoints = _checkpoints()
            for index, checkpoint in enumerate(checkpoints):
                await store.save(
                    checkpoint,
                    expected_revision=None if index == 0 else index - 1,
                )
            restored = await store.load(checkpoints[0].request_id)
            transitions = await store.transitions_for(checkpoints[0].request_id)
            persistence.close()

            self.assertEqual(restored, checkpoints[-1])
            self.assertEqual(len(transitions), 8)
            connection = sqlite3.connect(path)
            try:
                current_count = connection.execute(
                    "SELECT COUNT(*) FROM decision_current"
                ).fetchone()[0]
                evidence_count, evidence_bytes = connection.execute(
                    "SELECT COUNT(*), SUM(payload_size) "
                    "FROM decision_evidence_snapshots"
                ).fetchone()
                request_json = connection.execute(
                    "SELECT canonical_payload FROM decision_requests"
                ).fetchone()[0]
                validation_json = connection.execute(
                    "SELECT canonical_payload FROM decision_validations"
                ).fetchone()[0]
                effect_json = connection.execute(
                    "SELECT canonical_payload FROM decision_effects"
                ).fetchone()[0]
                result_json = connection.execute(
                    "SELECT canonical_payload FROM decision_results"
                ).fetchone()[0]
                commit_count = connection.execute(
                    "SELECT COUNT(*) FROM decision_commit_receipts"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(current_count, 1)
            self.assertEqual(evidence_count, 1)
            self.assertGreater(evidence_bytes, 1_500_000)
            for payload in (request_json, validation_json, effect_json, result_json):
                self.assertNotIn("candidate-00-", payload)
                self.assertNotIn('"candidates":[', payload)
            self.assertIn("$decision_evidence_ref", request_json)
            self.assertIn("$decision_evidence_ref", effect_json)
            self.assertNotIn("apply_receipt", result_json)
            self.assertEqual(commit_count, 1)
            assert restored is not None and restored.validated_decision is not None
            self.assertEqual(
                restored.validated_decision.normalized_effect.effect_fingerprint,
                checkpoints[-1].validated_decision.normalized_effect.effect_fingerprint,  # type: ignore[union-attr]
            )

    async def test_unsupported_codec_version_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.sqlite3"
            persistence = SQLitePersistence(path)
            store = persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE)
            checkpoint = _checkpoints()[0]
            await store.save(checkpoint, expected_revision=None)
            with persistence.database.transaction() as cursor:
                cursor.execute(
                    "UPDATE decision_requests SET codec_version = 'unsupported'"
                )
            with self.assertRaisesRegex(PersistenceConflictError, "unsupported"):
                await store.load(checkpoint.request_id)
            persistence.close()

    async def test_disposable_run_cleanup_refuses_external_reference(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            store = persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE)
            checkpoint = _checkpoints()[0]
            await store.save(checkpoint, expected_revision=None)
            external_run = UUID(int=2001)
            with persistence.database.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO runtime_run_metadata "
                    "(run_id, run_kind, disposable, created_at) VALUES (?, 'test', 1, ?)",
                    (str(checkpoint.run_id), NOW.isoformat()),
                )
                cursor.execute(
                    "INSERT INTO runtime_run_metadata "
                    "(run_id, run_kind, disposable, created_at) VALUES (?, 'runtime', 0, ?)",
                    (str(external_run), NOW.isoformat()),
                )
                cursor.execute(
                    "INSERT INTO decision_feedback "
                    "(effect_fingerprint, feedback_id, source_run_id, "
                    "subject_decision_id, subject_decision_type, version, "
                    "payload_fingerprint, record_json, receipt_json) "
                    "VALUES (?, ?, ?, ?, ?, 1, ?, '{}', '{}')",
                    (
                        "a" * 64,
                        str(UUID(int=2002)),
                        str(external_run),
                        str(checkpoint.request_id),
                        checkpoint.request.decision_type,
                        "b" * 64,
                    ),
                )
            with self.assertRaisesRegex(
                PersistenceConflictError, "external Decision Feedback"
            ):
                persistence.disposable_runs.purge_run(checkpoint.run_id)
            persistence.close()

    async def test_disposable_run_cleanup_is_explicit_and_dry_run_by_default(self) -> None:
        with TemporaryDirectory() as directory:
            persistence = SQLitePersistence(Path(directory) / "runtime.sqlite3")
            store = persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE)
            checkpoint = _checkpoints()[0]
            await store.save(checkpoint, expected_revision=None)
            with persistence.database.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO runtime_run_metadata "
                    "(run_id, run_kind, disposable, created_at) VALUES (?, 'test', 1, ?)",
                    (str(checkpoint.run_id), NOW.isoformat()),
                )
            preview = persistence.disposable_runs.purge_run(checkpoint.run_id)
            self.assertTrue(preview.dry_run)
            self.assertFalse(preview.deleted)
            self.assertIsNotNone(await store.load(checkpoint.request_id))
            deleted = persistence.disposable_runs.purge_run(
                checkpoint.run_id, dry_run=False
            )
            self.assertTrue(deleted.deleted)
            self.assertIsNone(await store.load(checkpoint.request_id))
            persistence.close()


if __name__ == "__main__":
    unittest.main()
