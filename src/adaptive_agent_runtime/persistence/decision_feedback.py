"""SQLite authority boundary for append-only Decision outcome feedback."""

from __future__ import annotations

import json
from collections.abc import Mapping
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    ExperienceMetadata,
    MemoryRecallBundle,
)
from adaptive_agent_runtime.core import AgentState, RunStatus
from adaptive_agent_runtime.decision_feedback import (
    DecisionFeedbackCommitReceipt,
    DecisionFeedbackEffect,
    DecisionFeedbackRecord,
    DecisionFeedbackStore,
    DecisionFeedbackSubjectReference,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.evaluation import EvaluationReport
from adaptive_agent_runtime.governance import (
    AuthorizationVerificationError,
    CommitPermitValidation,
    GovernanceTarget,
    RuntimeCommitPermit,
)
from adaptive_agent_runtime.orchestration import PLANNING_DECISION_TYPE
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.decisioning import (
    DecisionProof,
    SQLiteDecisionRecordReader,
)
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class SQLiteDecisionFeedbackStore(DecisionFeedbackStore):
    module_id = "decision.feedback_store.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier
        self._decisions = SQLiteDecisionRecordReader(database)

    async def find_applied_subject(
        self,
        run_id: UUID,
        decision_type: str,
    ) -> DecisionFeedbackSubjectReference | None:
        subjects = []
        for proof in self._decisions.find_completed_decisions(
            run_id, decision_type
        ):
            subject = self._subject_from_proof(proof)
            if subject is not None:
                subjects.append(subject)
        if len(subjects) > 1:
            raise PersistenceConflictError(
                f"Run has multiple APPLIED '{decision_type}' Decisions"
            )
        return subjects[0] if subjects else None

    async def commit(
        self,
        effect: DecisionFeedbackEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> DecisionFeedbackRecord:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Decision Feedback commit requires a Runtime Permit"
            )
        await self._permit_verifier.verify(
            permit,
            operation="decision.feedback.commit",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        payload_fingerprint = decision_fingerprint(effect)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT payload_fingerprint, record_json FROM decision_feedback "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                if prior["payload_fingerprint"] != payload_fingerprint:
                    raise PersistenceConflictError(
                        "Feedback Effect fingerprint conflicts with stored payload"
                    )
                return DecisionFeedbackRecord.model_validate_json(prior["record_json"])

            state_row = cursor.execute(
                "SELECT snapshots.snapshot_json FROM agent_state_current AS current "
                "JOIN agent_state_snapshots AS snapshots "
                "ON snapshots.run_id = current.run_id "
                "AND snapshots.revision = current.revision WHERE current.run_id = ?",
                (str(effect.source_run_id),),
            ).fetchone()
            if state_row is None:
                raise PersistenceConflictError("Feedback source Run is not persisted")
            state = AgentState.model_validate_json(state_row["snapshot_json"])
            if (
                state.status
                not in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.TERMINATED,
                }
                or state.task.task_id != effect.source_task_id
                or state.revision != effect.runtime_observation.state_revision
                or decision_fingerprint(state)
                != effect.runtime_observation.state_fingerprint
            ):
                raise PersistenceConflictError(
                    "Feedback source Run is non-terminal or changed"
                )

            subject_checkpoint = self._load_checkpoint(cursor, effect.subject.decision_id)
            self._verify_subject_checkpoint(subject_checkpoint, effect.subject, effect)
            planning_checkpoint = self._load_checkpoint(
                cursor, effect.planning_subject.decision_id
            )
            self._verify_subject_checkpoint(
                planning_checkpoint, effect.planning_subject, effect
            )
            if effect.planning_subject.decision_type != PLANNING_DECISION_TYPE:
                raise PersistenceConflictError("Feedback Planning binding is invalid")

            evaluation_row = cursor.execute(
                "SELECT report_fingerprint, report_json FROM evaluation_reports "
                "WHERE report_id = ? AND run_id = ? AND task_id = ?",
                (
                    str(effect.evaluation.report_id),
                    str(effect.source_run_id),
                    str(effect.source_task_id),
                ),
            ).fetchone()
            if evaluation_row is None:
                raise PersistenceConflictError("Feedback Evaluation is not persisted")
            report = EvaluationReport.model_validate_json(evaluation_row["report_json"])
            if (
                evaluation_row["report_fingerprint"]
                != effect.evaluation.report_fingerprint
                or decision_fingerprint(report) != effect.evaluation.report_fingerprint
                or report.trace_id != effect.evaluation.trace_id
                or report.outcome.evaluation_id
                != effect.evaluation.outcome_evaluation_id
                or decision_fingerprint(report.outcome)
                != effect.evaluation.outcome_fingerprint
                or report.outcome.verdict.value != effect.evaluation.verdict
                or report.outcome.score != effect.evaluation.score
            ):
                raise PersistenceConflictError("Feedback Evaluation evidence changed")

            experience_row = cursor.execute(
                "SELECT metadata_json FROM experience_metadata "
                "WHERE experience_id = ? AND effect_fingerprint = ?",
                (
                    str(effect.experience.experience_id),
                    effect.experience.effect_fingerprint,
                ),
            ).fetchone()
            if experience_row is None:
                raise PersistenceConflictError("Feedback Experience is not committed")
            experience = ExperienceMetadata.model_validate_json(
                experience_row["metadata_json"]
            )
            if (
                experience.source_run_id != effect.source_run_id
                or experience.version != effect.experience.version
                or decision_fingerprint(experience)
                != effect.experience.metadata_fingerprint
                or not any(
                    item.evaluation_id == effect.evaluation.outcome_evaluation_id
                    and item.evaluation_fingerprint
                    == effect.evaluation.outcome_fingerprint
                    for item in experience.source_evaluations
                )
            ):
                raise PersistenceConflictError(
                    "Feedback Experience does not bind the Evaluation"
                )

            artifact_row = cursor.execute(
                "SELECT receipt_json FROM workspace_artifacts "
                "WHERE run_id = ? AND node_id = ? AND artifact_type = ? "
                "AND effect_fingerprint = ?",
                (
                    str(effect.source_run_id),
                    str(effect.artifact.node_id),
                    effect.artifact.artifact_type,
                    effect.artifact.effect_fingerprint,
                ),
            ).fetchone()
            if artifact_row is None:
                raise PersistenceConflictError("Feedback Artifact is not committed")
            artifact_receipt = json.loads(artifact_row["receipt_json"])
            if (
                artifact_receipt.get("artifact_fingerprint")
                != effect.artifact.artifact_fingerprint
                or artifact_receipt.get("source_decision_request_id")
                != str(effect.artifact.source_decision_id)
            ):
                raise PersistenceConflictError(
                    "Feedback Artifact provenance does not match"
                )

            if effect.recall is not None:
                self._verify_recall(cursor, effect, planning_checkpoint)

            version_row = cursor.execute(
                "SELECT COALESCE(MAX(version), 0) AS version "
                "FROM decision_feedback WHERE source_run_id = ? "
                "AND subject_decision_id = ?",
                (str(effect.source_run_id), str(effect.subject.decision_id)),
            ).fetchone()
            version = int(version_row["version"]) + 1
            record = DecisionFeedbackRecord(
                feedback_id=effect.feedback_id,
                version=version,
                source_run_id=effect.source_run_id,
                source_task_id=effect.source_task_id,
                subject_decision_id=effect.subject.decision_id,
                subject_decision_type=effect.subject.decision_type,
                subject_effect_fingerprint=effect.subject.effect_fingerprint,
                planning_decision_id=effect.planning_subject.decision_id,
                planning_effect_fingerprint=(
                    effect.planning_subject.effect_fingerprint
                ),
                evaluation_ref=effect.evaluation,
                experience_metadata_ref=effect.experience,
                artifact_ref=effect.artifact,
                runtime_outcome=effect.runtime_outcome,
                evaluation_verdict=effect.evaluation_verdict,
                failure_count=effect.runtime_observation.failure_count,
                retry_count=effect.runtime_observation.retry_count,
                evidence_refs=effect.evidence_refs,
                attribution_type=effect.attribution_type,
                recall_ref=effect.recall,
                effect_fingerprint=effect_fingerprint,
            )
            receipt = DecisionFeedbackCommitReceipt(
                feedback_id=record.feedback_id,
                version=version,
                effect_fingerprint=effect_fingerprint,
                payload_fingerprint=payload_fingerprint,
                record_fingerprint=decision_fingerprint(record),
                committed_at=record.created_at,
            )
            cursor.execute(
                "INSERT INTO decision_feedback "
                "(effect_fingerprint, feedback_id, source_run_id, "
                "subject_decision_id, subject_decision_type, version, "
                "payload_fingerprint, record_json, receipt_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    effect_fingerprint,
                    str(record.feedback_id),
                    str(record.source_run_id),
                    str(record.subject_decision_id),
                    record.subject_decision_type,
                    version,
                    payload_fingerprint,
                    record.model_dump_json(),
                    receipt.model_dump_json(),
                ),
            )
            return record

    async def load_by_effect(
        self, effect_fingerprint: str
    ) -> DecisionFeedbackRecord | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT record_json, receipt_json FROM decision_feedback "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        record = DecisionFeedbackRecord.model_validate_json(row["record_json"])
        receipt = DecisionFeedbackCommitReceipt.model_validate_json(
            row["receipt_json"]
        )
        if (
            record.effect_fingerprint != effect_fingerprint
            or receipt.effect_fingerprint != effect_fingerprint
            or receipt.feedback_id != record.feedback_id
            or receipt.version != record.version
            or receipt.record_fingerprint != decision_fingerprint(record)
        ):
            raise PersistenceConflictError("Decision Feedback read-back is corrupted")
        return record

    async def load_receipt(
        self, effect_fingerprint: str
    ) -> DecisionFeedbackCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT receipt_json FROM decision_feedback "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        return (
            None
            if row is None
            else DecisionFeedbackCommitReceipt.model_validate_json(
                row["receipt_json"]
            )
        )

    async def list_for_run(
        self, run_id: UUID
    ) -> tuple[DecisionFeedbackRecord, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT record_json FROM decision_feedback "
                "WHERE source_run_id = ? ORDER BY subject_decision_type, version",
                (str(run_id),),
            ).fetchall()
        return tuple(
            DecisionFeedbackRecord.model_validate_json(row["record_json"])
            for row in rows
        )

    @staticmethod
    def _load_checkpoint(cursor: object, decision_id: UUID) -> DecisionProof:
        proof = SQLiteDecisionRecordReader.load_proof_in_transaction(
            cursor, decision_id
        )
        if proof is None:
            raise PersistenceConflictError("Feedback subject Decision is missing")
        return proof

    @staticmethod
    def _subject_from_proof(
        proof: DecisionProof,
    ) -> DecisionFeedbackSubjectReference | None:
        if not proof.is_applied or proof.effect_fingerprint is None:
            return None
        return DecisionFeedbackSubjectReference(
            decision_id=proof.request_id,
            decision_type=proof.decision_type,
            effect_fingerprint=proof.effect_fingerprint,
        )

    def _verify_subject_checkpoint(
        self,
        checkpoint: DecisionProof,
        expected: DecisionFeedbackSubjectReference,
        effect: DecisionFeedbackEffect,
    ) -> None:
        subject = self._subject_from_proof(checkpoint)
        if subject != expected:
            raise PersistenceConflictError(
                "Feedback subject Decision is not APPLIED or changed"
            )
        if (
            checkpoint.run_id != effect.source_run_id
            or checkpoint.task_id != effect.source_task_id
        ):
            raise PersistenceConflictError(
                "Feedback subject Decision belongs to another run or task"
            )
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT entry_json FROM runtime_trace WHERE run_id = ? "
                "AND entry_json LIKE ?",
                (str(effect.source_run_id), f'%{expected.decision_id}%'),
            ).fetchall()
        trace_match = False
        for row in rows:
            entry = json.loads(row["entry_json"])
            event = entry.get("event", {})
            payload = event.get("payload", {}) if isinstance(event, Mapping) else {}
            correlation_payload = (
                payload.get("correlation", {}) if isinstance(payload, Mapping) else {}
            )
            decision_payload = (
                payload.get("decision", {}) if isinstance(payload, Mapping) else {}
            )
            if (
                isinstance(correlation_payload, Mapping)
                and isinstance(decision_payload, Mapping)
                and event.get("kind") == "decision.applied"
                and correlation_payload.get("request_id") == str(expected.decision_id)
                and decision_payload.get("effect_fingerprint")
                == expected.effect_fingerprint
            ):
                trace_match = True
                break
        if not trace_match:
            raise PersistenceConflictError(
                "Feedback subject Effect is not confirmed by Decision Trace"
            )

    @staticmethod
    def _verify_recall(
        cursor: object,
        effect: DecisionFeedbackEffect,
        planning_checkpoint: DecisionProof,
    ) -> None:
        recall = effect.recall
        if recall is None:
            return
        if (
            recall.recall_decision_id != effect.subject.decision_id
            or recall.recall_effect_fingerprint
            != effect.subject.effect_fingerprint
            or recall.planning_decision_id
            != effect.planning_subject.decision_id
            or recall.planning_effect_fingerprint
            != effect.planning_subject.effect_fingerprint
        ):
            raise PersistenceConflictError("Recall Feedback Decision binding is invalid")
        row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT bundle_json FROM memory_recall_bundles "
            "WHERE effect_fingerprint = ? AND run_id = ?",
            (recall.recall_effect_fingerprint, str(effect.source_run_id)),
        ).fetchone()
        if row is None:
            raise PersistenceConflictError("Recall Feedback Bundle is not committed")
        bundle = MemoryRecallBundle.model_validate_json(row["bundle_json"])
        if (
            bundle.bundle_id != recall.bundle_id
            or bundle.recall_decision_id != recall.recall_decision_id
            or decision_fingerprint(bundle) != recall.bundle_fingerprint
        ):
            raise PersistenceConflictError("Recall Feedback Bundle changed")
        expected = f"recall-bundle:{bundle.bundle_id}"
        if expected not in planning_checkpoint.evidence_ids:
            raise PersistenceConflictError(
                "Planning Decision did not consume the Recall Bundle"
            )
