"""SQLite authority boundary for governed, append-only Learning Insights."""

from __future__ import annotations

import json
from collections.abc import Mapping
from uuid import UUID

from adaptive_agent_runtime.context_memory import ExperienceMetadata, MemoryScope
from adaptive_agent_runtime.core import AgentState, RunStatus
from adaptive_agent_runtime.decision_feedback import (
    DecisionFeedbackAttributionType,
    DecisionFeedbackCommitReceipt,
    DecisionFeedbackRecord,
    stable_feedback_request_id,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.evaluation import EvaluationReport
from adaptive_agent_runtime.experience_learning import (
    LearningEvidenceBinding,
    LearningEvidenceCandidate,
    LearningInsight,
    LearningInsightCommitReceipt,
    LearningInsightEffect,
    LearningInsightStore,
)
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


_RESEARCH_LEARNING_SCOPE = MemoryScope(
    tenant_id="default",
    project_id="research",
    agent_scope="planner",
)


class SQLiteLearningInsightStore(LearningInsightStore):
    """Resolve verified evidence and commit immutable Insights behind a Permit."""

    module_id = "learning.insight_store.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def resolve_candidates(
        self,
        *,
        scope: MemoryScope,
        subject_decision_type: str,
    ) -> tuple[LearningEvidenceCandidate, ...]:
        if scope != _RESEARCH_LEARNING_SCOPE:
            return ()
        if subject_decision_type != PLANNING_DECISION_TYPE:
            return ()
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT record_json, receipt_json FROM decision_feedback "
                "WHERE subject_decision_type = ? ORDER BY source_run_id, version",
                (subject_decision_type,),
            ).fetchall()
            candidates: list[LearningEvidenceCandidate] = []
            for row in rows:
                try:
                    record = DecisionFeedbackRecord.model_validate_json(
                        row["record_json"]
                    )
                    self._verify_feedback_receipt(record, row["receipt_json"])
                    candidate = self._verified_candidate(cursor, record, scope)
                except (KeyError, TypeError, ValueError, PersistenceConflictError):
                    # Candidate resolution is deny-by-default. Missing or corrupt
                    # evidence is absence, never a positive learning signal.
                    continue
                candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda item: item.candidate_ref))

    async def commit(
        self,
        effect: LearningInsightEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> LearningInsight:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Learning Insight commit requires a Runtime Permit"
            )
        await self._permit_verifier.verify(
            permit,
            operation="learning.insight.commit",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        payload_fingerprint = decision_fingerprint(effect)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT payload_fingerprint, insight_json FROM learning_insights "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                if prior["payload_fingerprint"] != payload_fingerprint:
                    raise PersistenceConflictError(
                        "Learning Effect fingerprint conflicts with stored payload"
                    )
                return LearningInsight.model_validate_json(prior["insight_json"])

            if effect.scope != _RESEARCH_LEARNING_SCOPE:
                raise PersistenceConflictError("Learning scope is not authorized")
            if effect.subject_decision_type != PLANNING_DECISION_TYPE:
                raise PersistenceConflictError(
                    "Phase 3-D only permits Initial Planning learning"
                )
            self._verify_effect_evidence(cursor, effect)

            scope_values = (
                effect.scope.tenant_id,
                effect.scope.project_id,
                effect.scope.agent_scope,
                effect.subject_decision_type,
            )
            version_row = cursor.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM learning_insights "
                "WHERE tenant_id = ? AND project_id = ? AND agent_scope = ? "
                "AND subject_decision_type = ?",
                scope_values,
            ).fetchone()
            version = int(version_row["version"]) + 1
            insight = LearningInsight(
                learning_insight_id=effect.learning_insight_id,
                version=version,
                scope=effect.scope,
                subject_decision_type=effect.subject_decision_type,
                observed_pattern=effect.observed_pattern,
                applicable_conditions=effect.applicable_conditions,
                limitations=effect.limitations,
                supporting_evidence_refs=tuple(
                    item.candidate_ref for item in effect.supporting_evidence
                ),
                counterevidence_refs=tuple(
                    item.candidate_ref for item in effect.counterevidence
                ),
                source_run_refs=effect.source_run_refs,
                source_feedback_refs=effect.source_feedback_refs,
                source_experience_refs=effect.source_experience_refs,
                evidence_set_fingerprint=effect.evidence_set_fingerprint,
                effect_fingerprint=effect_fingerprint,
            )
            receipt = LearningInsightCommitReceipt(
                learning_insight_id=insight.learning_insight_id,
                version=version,
                effect_fingerprint=effect_fingerprint,
                payload_fingerprint=payload_fingerprint,
                insight_fingerprint=decision_fingerprint(insight),
                committed_at=insight.created_at,
            )
            cursor.execute(
                "INSERT INTO learning_insights "
                "(effect_fingerprint, learning_insight_id, tenant_id, project_id, "
                "agent_scope, subject_decision_type, version, "
                "evidence_set_fingerprint, payload_fingerprint, insight_json, "
                "receipt_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    effect_fingerprint,
                    str(insight.learning_insight_id),
                    effect.scope.tenant_id,
                    effect.scope.project_id,
                    effect.scope.agent_scope,
                    effect.subject_decision_type,
                    version,
                    effect.evidence_set_fingerprint,
                    payload_fingerprint,
                    insight.model_dump_json(),
                    receipt.model_dump_json(),
                ),
            )
        return insight

    async def load_by_effect(self, effect_fingerprint: str) -> LearningInsight | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT insight_json, receipt_json FROM learning_insights "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        insight = LearningInsight.model_validate_json(row["insight_json"])
        receipt = LearningInsightCommitReceipt.model_validate_json(
            row["receipt_json"]
        )
        if (
            insight.effect_fingerprint != effect_fingerprint
            or receipt.effect_fingerprint != effect_fingerprint
            or receipt.learning_insight_id != insight.learning_insight_id
            or receipt.version != insight.version
            or receipt.insight_fingerprint != decision_fingerprint(insight)
        ):
            raise PersistenceConflictError("Learning Insight read-back is corrupted")
        return insight

    async def load_receipt(
        self, effect_fingerprint: str
    ) -> LearningInsightCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT receipt_json FROM learning_insights "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        return (
            None
            if row is None
            else LearningInsightCommitReceipt.model_validate_json(
                row["receipt_json"]
            )
        )

    async def list_for_scope(
        self,
        scope: MemoryScope,
        *,
        subject_decision_type: str | None = None,
    ) -> tuple[LearningInsight, ...]:
        query = (
            "SELECT insight_json FROM learning_insights WHERE tenant_id = ? "
            "AND project_id = ? AND agent_scope = ?"
        )
        values: list[str] = [scope.tenant_id, scope.project_id, scope.agent_scope]
        if subject_decision_type is not None:
            query += " AND subject_decision_type = ?"
            values.append(subject_decision_type)
        query += " ORDER BY version"
        with self._database.reader() as cursor:
            rows = cursor.execute(query, tuple(values)).fetchall()
        return tuple(
            LearningInsight.model_validate_json(row["insight_json"])
            for row in rows
        )

    def _verify_effect_evidence(
        self,
        cursor: object,
        effect: LearningInsightEffect,
    ) -> None:
        snapshot_by_ref = {item.candidate_ref: item for item in effect.evidence_snapshot}
        if len(snapshot_by_ref) != len(effect.evidence_snapshot):
            raise PersistenceConflictError("Learning evidence snapshot has duplicates")
        expected_set_fingerprint = decision_fingerprint(
            tuple(
                item.candidate_fingerprint
                for item in sorted(
                    effect.evidence_snapshot, key=lambda value: value.candidate_ref
                )
            )
        )
        if expected_set_fingerprint != effect.evidence_set_fingerprint:
            raise PersistenceConflictError("Learning evidence set fingerprint changed")
        selected = (*effect.supporting_evidence, *effect.counterevidence)
        if len({item.candidate_ref for item in selected}) != len(selected):
            raise PersistenceConflictError("Learning selected evidence has duplicates")
        if any(snapshot_by_ref.get(item.candidate_ref) != item for item in selected):
            raise PersistenceConflictError(
                "Learning selected evidence is outside the authorized snapshot"
            )
        if any(
            item.attribution_type is not DecisionFeedbackAttributionType.ASSOCIATED
            for item in selected
        ):
            raise PersistenceConflictError(
                "Insufficient evidence cannot support or refute a Learning Insight"
            )
        if len({item.source_run_id for item in selected}) < 2:
            raise PersistenceConflictError(
                "Learning Insight requires two independent selected runs"
            )
        if tuple(dict.fromkeys(item.source_run_id for item in selected)) != (
            effect.source_run_refs
        ):
            raise PersistenceConflictError("Learning source Run references changed")
        if tuple(dict.fromkeys(item.feedback_id for item in selected)) != (
            effect.source_feedback_refs
        ):
            raise PersistenceConflictError("Learning Feedback references changed")
        if tuple(dict.fromkeys(item.experience_id for item in selected)) != (
            effect.source_experience_refs
        ):
            raise PersistenceConflictError("Learning Experience references changed")
        for binding in effect.evidence_snapshot:
            record_row = cursor.execute(  # type: ignore[attr-defined]
                "SELECT record_json, receipt_json FROM decision_feedback "
                "WHERE feedback_id = ?",
                (str(binding.feedback_id),),
            ).fetchone()
            if record_row is None:
                raise PersistenceConflictError("Learning Feedback evidence is missing")
            record = DecisionFeedbackRecord.model_validate_json(
                record_row["record_json"]
            )
            self._verify_feedback_receipt(record, record_row["receipt_json"])
            candidate = self._verified_candidate(cursor, record, effect.scope)
            if self._binding(candidate) != binding:
                raise PersistenceConflictError("Learning evidence changed before Apply")

    def _verified_candidate(
        self,
        cursor: object,
        record: DecisionFeedbackRecord,
        scope: MemoryScope,
    ) -> LearningEvidenceCandidate:
        if record.subject_decision_type != PLANNING_DECISION_TYPE:
            raise PersistenceConflictError("Learning only accepts Planning Feedback")
        manifest = cursor.execute(  # type: ignore[attr-defined]
            "SELECT 1 FROM application_run_manifests "
            "WHERE application_id = 'research_agent' AND run_id = ?",
            (str(record.source_run_id),),
        ).fetchone()
        if manifest is None or scope != _RESEARCH_LEARNING_SCOPE:
            raise PersistenceConflictError("Learning evidence is outside scope")

        state_row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT snapshots.snapshot_json FROM agent_state_current AS current "
            "JOIN agent_state_snapshots AS snapshots ON snapshots.run_id = current.run_id "
            "AND snapshots.revision = current.revision WHERE current.run_id = ?",
            (str(record.source_run_id),),
        ).fetchone()
        if state_row is None:
            raise PersistenceConflictError("Learning source Run is missing")
        state = AgentState.model_validate_json(state_row["snapshot_json"])
        if state.status not in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.TERMINATED,
        }:
            raise PersistenceConflictError("Learning source Run is non-terminal")
        if state.task.task_id != record.source_task_id:
            raise PersistenceConflictError("Learning source task changed")

        feedback_checkpoint = self._checkpoint(
            cursor,
            stable_feedback_request_id(
                record.source_run_id, record.subject_decision_id
            ),
        )
        feedback_effect = self._applied_effect(cursor, feedback_checkpoint)
        if (
            feedback_effect.get("effect_fingerprint") != record.effect_fingerprint
            or not self._trace_applied(
                cursor,
                record.source_run_id,
                feedback_checkpoint.request_id,
                record.effect_fingerprint,
            )
        ):
            raise PersistenceConflictError("Learning Feedback is not APPLIED")
        subject_checkpoint = self._checkpoint(cursor, record.subject_decision_id)
        subject_effect = self._applied_effect(cursor, subject_checkpoint)
        if (
            subject_effect.get("effect_fingerprint")
            != record.subject_effect_fingerprint
        ):
            raise PersistenceConflictError("Learning subject Decision changed")

        evaluation_row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT report_fingerprint, report_json FROM evaluation_reports "
            "WHERE report_id = ? AND run_id = ?",
            (str(record.evaluation_ref.report_id), str(record.source_run_id)),
        ).fetchone()
        if evaluation_row is None:
            raise PersistenceConflictError("Learning Evaluation is missing")
        evaluation = EvaluationReport.model_validate_json(
            evaluation_row["report_json"]
        )
        if (
            evaluation_row["report_fingerprint"]
            != record.evaluation_ref.report_fingerprint
            or decision_fingerprint(evaluation)
            != record.evaluation_ref.report_fingerprint
            or decision_fingerprint(evaluation.outcome)
            != record.evaluation_ref.outcome_fingerprint
        ):
            raise PersistenceConflictError("Learning Evaluation changed")

        experience_row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT metadata_json FROM experience_metadata "
            "WHERE experience_id = ? AND effect_fingerprint = ?",
            (
                str(record.experience_metadata_ref.experience_id),
                record.experience_metadata_ref.effect_fingerprint,
            ),
        ).fetchone()
        if experience_row is None:
            raise PersistenceConflictError("Learning Experience is missing")
        experience = ExperienceMetadata.model_validate_json(
            experience_row["metadata_json"]
        )
        if (
            experience.source_run_id != record.source_run_id
            or experience.version != record.experience_metadata_ref.version
            or decision_fingerprint(experience)
            != record.experience_metadata_ref.metadata_fingerprint
            or not any(
                item.evaluation_id == record.evaluation_ref.outcome_evaluation_id
                and item.evaluation_fingerprint
                == record.evaluation_ref.outcome_fingerprint
                for item in experience.source_evaluations
            )
        ):
            raise PersistenceConflictError("Learning Experience provenance changed")

        artifact_row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT receipt_json FROM workspace_artifacts WHERE run_id = ? "
            "AND node_id = ? AND artifact_type = ? AND effect_fingerprint = ?",
            (
                str(record.source_run_id),
                str(record.artifact_ref.node_id),
                record.artifact_ref.artifact_type,
                record.artifact_ref.effect_fingerprint,
            ),
        ).fetchone()
        if artifact_row is None:
            raise PersistenceConflictError("Learning Artifact is missing")
        artifact_receipt = json.loads(artifact_row["receipt_json"])
        if (
            artifact_receipt.get("artifact_fingerprint")
            != record.artifact_ref.artifact_fingerprint
            or artifact_receipt.get("source_decision_request_id")
            != str(record.artifact_ref.source_decision_id)
        ):
            raise PersistenceConflictError("Learning Artifact provenance changed")

        summary = experience.agent_assessment_summary
        observed = str(summary.get("observed_pattern", "Verified execution evidence"))
        relevance = str(summary.get("possible_relevance", "Evidence is associated"))
        limitation = (
            "This record is limitation-only because deterministic Evaluation was "
            "inconclusive."
            if record.attribution_type
            is DecisionFeedbackAttributionType.INSUFFICIENT_EVIDENCE
            else "Association only; no causal or policy-quality conclusion."
        )
        candidate_subject = {
            "scope": scope,
            "feedback_fingerprint": record.effect_fingerprint,
            "feedback_record_fingerprint": decision_fingerprint(record),
            "experience_fingerprint": decision_fingerprint(experience),
            "evaluation_fingerprint": decision_fingerprint(evaluation),
            "artifact_fingerprint": record.artifact_ref.artifact_fingerprint,
            "subject_effect_fingerprint": record.subject_effect_fingerprint,
        }
        candidate_fingerprint = decision_fingerprint(candidate_subject)
        return LearningEvidenceCandidate(
            candidate_ref=f"evidence-{candidate_fingerprint[:32]}",
            scope=scope,
            subject_decision_type=record.subject_decision_type,
            source_run_id=record.source_run_id,
            source_task_id=record.source_task_id,
            subject_decision_id=record.subject_decision_id,
            subject_effect_fingerprint=record.subject_effect_fingerprint,
            feedback_id=record.feedback_id,
            feedback_effect_fingerprint=record.effect_fingerprint,
            feedback_record_fingerprint=decision_fingerprint(record),
            attribution_type=record.attribution_type,
            experience=record.experience_metadata_ref,
            evaluation=record.evaluation_ref,
            artifact=record.artifact_ref,
            runtime_outcome=record.runtime_outcome,
            evaluation_verdict=record.evaluation_verdict,
            runtime_observation={
                "failure_count": record.failure_count,
                "retry_count": record.retry_count,
            },
            experience_summary=experience.agent_assessment_summary,
            candidate_fingerprint=candidate_fingerprint,
        )

    @staticmethod
    def _binding(candidate: LearningEvidenceCandidate) -> LearningEvidenceBinding:
        return LearningEvidenceBinding(
            candidate_ref=candidate.candidate_ref,
            candidate_fingerprint=candidate.candidate_fingerprint,
            source_run_id=candidate.source_run_id,
            feedback_id=candidate.feedback_id,
            feedback_effect_fingerprint=candidate.feedback_effect_fingerprint,
            experience_id=candidate.experience.experience_id,
            experience_effect_fingerprint=candidate.experience.effect_fingerprint,
            evaluation_report_id=candidate.evaluation.report_id,
            evaluation_report_fingerprint=candidate.evaluation.report_fingerprint,
            artifact_effect_fingerprint=candidate.artifact.effect_fingerprint,
            runtime_outcome=candidate.runtime_outcome,
            attribution_type=candidate.attribution_type,
        )

    @staticmethod
    def _verify_feedback_receipt(
        record: DecisionFeedbackRecord,
        raw_receipt: str,
    ) -> None:
        receipt = DecisionFeedbackCommitReceipt.model_validate_json(raw_receipt)
        if (
            receipt.feedback_id != record.feedback_id
            or receipt.version != record.version
            or receipt.effect_fingerprint != record.effect_fingerprint
            or receipt.record_fingerprint != decision_fingerprint(record)
        ):
            raise PersistenceConflictError("Learning Feedback receipt is invalid")

    @staticmethod
    def _checkpoint(cursor: object, request_id: UUID) -> DecisionProof:
        proof = SQLiteDecisionRecordReader.load_proof_in_transaction(
            cursor, request_id
        )
        if proof is None:
            raise PersistenceConflictError("Learning Decision checkpoint is missing")
        return proof

    @staticmethod
    def _applied_effect(
        cursor: object, checkpoint: DecisionProof
    ) -> Mapping[str, object]:
        if not checkpoint.is_applied or checkpoint.effect_fingerprint is None:
            raise PersistenceConflictError("Learning evidence Decision is not APPLIED")
        normalized = SQLiteDecisionRecordReader.load_effect_in_transaction(
            cursor, checkpoint.effect_fingerprint
        )
        if normalized is None:
            raise PersistenceConflictError("Learning evidence Effect is missing")
        return normalized

    @staticmethod
    def _trace_applied(
        cursor: object,
        run_id: UUID,
        request_id: UUID,
        effect_fingerprint: str,
    ) -> bool:
        rows = cursor.execute(  # type: ignore[attr-defined]
            "SELECT entry_json FROM runtime_trace WHERE run_id = ? "
            "AND entry_json LIKE ?",
            (str(run_id), f"%{request_id}%"),
        ).fetchall()
        for row in rows:
            entry = json.loads(row["entry_json"])
            event = entry.get("event", {})
            payload = event.get("payload", {}) if isinstance(event, Mapping) else {}
            correlation = (
                payload.get("correlation", {}) if isinstance(payload, Mapping) else {}
            )
            decision = payload.get("decision", {}) if isinstance(payload, Mapping) else {}
            if (
                isinstance(correlation, Mapping)
                and isinstance(decision, Mapping)
                and event.get("kind") == "decision.applied"
                and correlation.get("request_id") == str(request_id)
                and decision.get("effect_fingerprint") == effect_fingerprint
            ):
                return True
        return False
