"""SQLite evidence resolution and immutable Phase 4-A Proposal authority store."""

from __future__ import annotations

import json
from collections.abc import Mapping
from uuid import UUID

from adaptive_agent_runtime.decision_feedback import (
    DecisionFeedbackAttributionType,
    DecisionFeedbackCommitReceipt,
    DecisionFeedbackRecord,
)
from adaptive_agent_runtime.decisioning import DecisionCheckpoint, decision_fingerprint
from adaptive_agent_runtime.evaluation import EvaluationReport
from adaptive_agent_runtime.experience_learning import (
    LearningAssessmentRequest,
    LearningInsight,
    LearningInsightCommitReceipt,
    LearningInsightDraft,
    LearningInsightEffect,
)
from adaptive_agent_runtime.context_memory import (
    ExperienceMetadata,
    ExperienceMetadataCommitReceipt,
)
from adaptive_agent_runtime.governance import (
    AuthorizationVerificationError,
    CommitPermitValidation,
    GovernanceTarget,
    RuntimeCommitPermit,
)
from adaptive_agent_runtime.optimization import (
    OptimizationEvidenceBinding,
    OptimizationEvidenceResolver,
    OptimizationProposal,
    OptimizationProposalCommitReceipt,
    OptimizationProposalEffect,
    OptimizationProposalStore,
    OptimizationScope,
    stable_optimization_evidence_ref,
)
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


# A Learning Insight is already a governed cross-run aggregation.  Projecting
# every historical version would both duplicate evidence and make the Agent
# context grow without bound.  Phase 4-A therefore assesses only the newest
# provenance-complete Insight for the exact Optimization scope.
_MAX_OPTIMIZATION_INSIGHT_CANDIDATES = 1


class SQLiteOptimizationEvidenceResolver(OptimizationEvidenceResolver):
    """Resolve only provenance-complete, persisted Phase 3 Learning evidence."""

    module_id = "optimization.evidence_resolver.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def resolve(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationEvidenceBinding, ...]:
        with self._database.reader() as cursor:
            return _resolve_candidates(cursor, scope)


class SQLiteOptimizationProposalStore(OptimizationProposalStore):
    """Permit-guarded append-only Proposal and Receipt transaction."""

    module_id = "optimization.proposal_store.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def commit(
        self,
        effect: OptimizationProposalEffect,
        *,
        source_decision_request_id: UUID,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> OptimizationProposal:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Optimization Proposal commit requires a Runtime Permit"
            )
        await self._permit_verifier.verify(
            permit,
            operation="optimization.proposal.commit",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        if target.target_type != "optimization_proposal":
            raise AuthorizationVerificationError(
                "Optimization Permit targets another authority object"
            )
        if target.target_id != str(effect.proposal_id):
            raise AuthorizationVerificationError(
                "Optimization Permit targets another Proposal"
            )
        payload_fingerprint = decision_fingerprint(effect)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT payload_fingerprint, proposal_json "
                "FROM optimization_proposals WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                if prior["payload_fingerprint"] != payload_fingerprint:
                    raise PersistenceConflictError(
                        "Optimization Effect fingerprint conflicts with stored payload"
                    )
                return OptimizationProposal.model_validate_json(
                    prior["proposal_json"]
                )

            duplicate = cursor.execute(
                "SELECT effect_fingerprint, payload_fingerprint, proposal_json "
                "FROM optimization_proposals WHERE proposal_id = ?",
                (str(effect.proposal_id),),
            ).fetchone()
            if duplicate is not None:
                if (
                    duplicate["effect_fingerprint"] != effect_fingerprint
                    or duplicate["payload_fingerprint"] != payload_fingerprint
                ):
                    raise PersistenceConflictError(
                        "Optimization Proposal identity conflicts with stored payload"
                    )
                return OptimizationProposal.model_validate_json(
                    duplicate["proposal_json"]
                )

            current = {
                item.candidate_ref: item
                for item in _resolve_candidates(cursor, effect.scope)
            }
            snapshot = {item.candidate_ref: item for item in effect.evidence_snapshot}
            selected_refs = (
                *effect.supporting_candidate_refs,
                *effect.counterevidence_candidate_refs,
            )
            if any(item not in snapshot for item in selected_refs):
                raise PersistenceConflictError(
                    "Optimization Effect selected evidence outside its snapshot"
                )
            for candidate_ref in selected_refs:
                if current.get(candidate_ref) != snapshot[candidate_ref]:
                    raise PersistenceConflictError(
                        "Optimization source evidence changed before Apply"
                    )
            expected_evidence_fingerprint = decision_fingerprint(
                tuple(item.candidate_fingerprint for item in effect.evidence_snapshot)
            )
            if expected_evidence_fingerprint != effect.evidence_set_fingerprint:
                raise PersistenceConflictError(
                    "Optimization evidence-set fingerprint is inconsistent"
                )

            supporting = tuple(
                snapshot[item] for item in effect.supporting_candidate_refs
            )
            explicit_counter = tuple(
                snapshot[item] for item in effect.counterevidence_candidate_refs
            )
            counter_refs = tuple(
                dict.fromkeys(
                    (
                        *(
                            f"feedback:{feedback_id}"
                            for item in supporting
                            for feedback_id in item.counterevidence_feedback_refs
                        ),
                        *(
                            f"learning_insight:{item.learning_insight_id}"
                            for item in explicit_counter
                        ),
                    )
                )
            )
            proposal = OptimizationProposal(
                proposal_id=effect.proposal_id,
                source_decision_request_id=source_decision_request_id,
                scope=effect.scope,
                target_type=effect.target.target_type,
                target_key=effect.target.target_key,
                current_value=effect.target.current_value,
                current_value_fingerprint=effect.target.current_value_fingerprint,
                current_configuration_revision=(
                    effect.target.current_configuration_revision
                ),
                current_configuration_fingerprint=(
                    effect.target.current_configuration_fingerprint
                ),
                configuration_source=effect.target.configuration_source,
                proposed_value=effect.proposed_value,
                supporting_learning_insight_refs=tuple(
                    dict.fromkeys(item.learning_insight_id for item in supporting)
                ),
                supporting_feedback_refs=tuple(
                    dict.fromkeys(
                        feedback_id
                        for item in supporting
                        for feedback_id in item.supporting_feedback_refs
                    )
                ),
                supporting_experience_refs=tuple(
                    dict.fromkeys(
                        experience_id
                        for item in supporting
                        for experience_id in item.supporting_experience_refs
                    )
                ),
                supporting_evaluation_refs=tuple(
                    dict.fromkeys(
                        evaluation_id
                        for item in supporting
                        for evaluation_id in item.supporting_evaluation_refs
                    )
                ),
                counterevidence_refs=counter_refs,
                applicable_conditions=effect.applicable_conditions,
                expected_impact=effect.expected_impact,
                limitations=effect.limitations,
                risk_classification=effect.risk_classification,
                rollback_requirements=effect.rollback_requirements,
                evidence_set_fingerprint=effect.evidence_set_fingerprint,
                effect_fingerprint=effect_fingerprint,
            )
            receipt = OptimizationProposalCommitReceipt(
                proposal_id=proposal.proposal_id,
                effect_fingerprint=effect_fingerprint,
                payload_fingerprint=payload_fingerprint,
                proposal_fingerprint=decision_fingerprint(proposal),
                committed_at=proposal.created_at,
            )
            cursor.execute(
                "INSERT INTO optimization_proposals "
                "(effect_fingerprint, proposal_id, source_decision_request_id, "
                "tenant_id, project_id, application_id, decision_type, target_key, "
                "baseline_fingerprint, evidence_set_fingerprint, payload_fingerprint, "
                "proposal_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    effect_fingerprint,
                    str(proposal.proposal_id),
                    str(source_decision_request_id),
                    proposal.scope.tenant,
                    proposal.scope.project,
                    proposal.scope.application,
                    proposal.scope.decision_type,
                    proposal.target_key.value,
                    proposal.current_value_fingerprint,
                    proposal.evidence_set_fingerprint,
                    payload_fingerprint,
                    proposal.model_dump_json(),
                ),
            )
            cursor.execute(
                "INSERT INTO optimization_proposal_receipts "
                "(effect_fingerprint, proposal_id, receipt_json) VALUES (?, ?, ?)",
                (
                    effect_fingerprint,
                    str(proposal.proposal_id),
                    receipt.model_dump_json(),
                ),
            )
        return proposal

    async def load_by_effect(
        self,
        effect_fingerprint: str,
    ) -> OptimizationProposal | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT proposal_json, payload_fingerprint FROM optimization_proposals "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            receipt_row = cursor.execute(
                "SELECT receipt_json FROM optimization_proposal_receipts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None and receipt_row is None:
            return None
        if row is None or receipt_row is None:
            raise PersistenceConflictError(
                "Optimization Proposal transaction is incomplete"
            )
        proposal = OptimizationProposal.model_validate_json(row["proposal_json"])
        receipt = OptimizationProposalCommitReceipt.model_validate_json(
            receipt_row["receipt_json"]
        )
        if (
            proposal.effect_fingerprint != effect_fingerprint
            or receipt.effect_fingerprint != effect_fingerprint
            or receipt.proposal_id != proposal.proposal_id
            or receipt.proposal_fingerprint != decision_fingerprint(proposal)
        ):
            raise PersistenceConflictError(
                "Optimization Proposal read-back is corrupted"
            )
        return proposal

    async def load_receipt(
        self,
        effect_fingerprint: str,
    ) -> OptimizationProposalCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT receipt_json FROM optimization_proposal_receipts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        return (
            None
            if row is None
            else OptimizationProposalCommitReceipt.model_validate_json(
                row["receipt_json"]
            )
        )

    async def load_by_id(
        self,
        proposal_id: UUID,
    ) -> OptimizationProposal | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT effect_fingerprint FROM optimization_proposals "
                "WHERE proposal_id = ?",
                (str(proposal_id),),
            ).fetchone()
        if row is None:
            return None
        return await self.load_by_effect(str(row["effect_fingerprint"]))

    async def list_for_scope(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationProposal, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT proposal_json FROM optimization_proposals "
                "WHERE tenant_id = ? AND project_id = ? AND application_id = ? "
                "AND decision_type = ? ORDER BY rowid",
                (
                    scope.tenant,
                    scope.project,
                    scope.application,
                    scope.decision_type,
                ),
            ).fetchall()
        return tuple(
            OptimizationProposal.model_validate_json(row["proposal_json"])
            for row in rows
        )

    async def verify_phase3_provenance(
        self,
        proposal_id: UUID,
    ) -> bool:
        """Re-read the complete authority chain used by automatic selection."""

        proposal = await self.load_by_id(proposal_id)
        if proposal is None:
            return False
        try:
            with self._database.reader() as cursor:
                verify_optimization_proposal_phase3_in_transaction(cursor, proposal)
        except (KeyError, TypeError, ValueError, PersistenceConflictError):
            return False
        return True


def _resolve_candidates(
    cursor: object,
    scope: OptimizationScope,
) -> tuple[OptimizationEvidenceBinding, ...]:
    rows = cursor.execute(  # type: ignore[attr-defined]
        "SELECT insight_json, receipt_json FROM learning_insights "
        "WHERE tenant_id = ? AND project_id = ? AND agent_scope = 'planner' "
        "AND subject_decision_type = ? ORDER BY version DESC LIMIT ?",
        (
            scope.tenant,
            scope.project,
            scope.decision_type,
            _MAX_OPTIMIZATION_INSIGHT_CANDIDATES,
        ),
    ).fetchall()
    candidates: list[OptimizationEvidenceBinding] = []
    for row in rows:
        try:
            insight = LearningInsight.model_validate_json(row["insight_json"])
            receipt = LearningInsightCommitReceipt.model_validate_json(
                row["receipt_json"]
            )
            if (
                receipt.learning_insight_id != insight.learning_insight_id
                or receipt.version != insight.version
                or receipt.effect_fingerprint != insight.effect_fingerprint
                or receipt.insight_fingerprint != decision_fingerprint(insight)
            ):
                raise PersistenceConflictError("Learning Insight receipt is invalid")
            effect = _learning_effect(cursor, insight)
            supporting = tuple(
                _verify_feedback_binding(cursor, binding)
                for binding in effect.supporting_evidence
            )
            counter = tuple(
                _verify_feedback_binding(cursor, binding)
                for binding in effect.counterevidence
            )
            if not supporting:
                raise PersistenceConflictError(
                    "Optimization evidence has no supporting Learning evidence"
                )
            supporting_feedback_refs = tuple(
                item.feedback_id for item in supporting
            )
            supporting_experience_refs = tuple(
                item.experience_metadata_ref.experience_id for item in supporting
            )
            supporting_evaluation_refs = tuple(
                item.evaluation_ref.report_id for item in supporting
            )
            supporting_artifact_effect_fingerprints = tuple(
                item.artifact_ref.effect_fingerprint for item in supporting
            )
            counterevidence_feedback_refs = tuple(
                item.feedback_id for item in counter
            )
            subject = {
                "scope": scope,
                "learning_insight_id": insight.learning_insight_id,
                "learning_insight_version": insight.version,
                "learning_insight_effect_fingerprint": insight.effect_fingerprint,
                "learning_insight_fingerprint": decision_fingerprint(insight),
                "supporting_feedback_refs": supporting_feedback_refs,
                "supporting_experience_refs": supporting_experience_refs,
                "supporting_evaluation_refs": supporting_evaluation_refs,
                "supporting_artifact_effect_fingerprints": (
                    supporting_artifact_effect_fingerprints
                ),
                "counterevidence_feedback_refs": counterevidence_feedback_refs,
                "source_run_refs": insight.source_run_refs,
                "observed_pattern": insight.observed_pattern,
                "applicable_conditions": insight.applicable_conditions,
                "limitations": insight.limitations,
            }
            candidates.append(
                OptimizationEvidenceBinding(
                    candidate_ref=stable_optimization_evidence_ref(
                        insight.learning_insight_id,
                        insight.version,
                    ),
                    candidate_fingerprint=decision_fingerprint(subject),
                    scope=scope,
                    learning_insight_id=insight.learning_insight_id,
                    learning_insight_version=insight.version,
                    learning_insight_effect_fingerprint=insight.effect_fingerprint,
                    learning_insight_fingerprint=decision_fingerprint(insight),
                    supporting_feedback_refs=supporting_feedback_refs,
                    supporting_experience_refs=supporting_experience_refs,
                    supporting_evaluation_refs=supporting_evaluation_refs,
                    supporting_artifact_effect_fingerprints=(
                        supporting_artifact_effect_fingerprints
                    ),
                    counterevidence_feedback_refs=counterevidence_feedback_refs,
                    source_run_refs=insight.source_run_refs,
                    observed_pattern=insight.observed_pattern,
                    applicable_conditions=insight.applicable_conditions,
                    limitations=insight.limitations,
                )
            )
        except (KeyError, TypeError, ValueError, PersistenceConflictError):
            # Runtime evidence resolution is deny-by-default.  Corrupt or
            # incomplete records never become Agent-visible candidates.
            continue
    return tuple(sorted(candidates, key=lambda item: item.candidate_ref))


def verify_optimization_proposal_phase3_in_transaction(
    cursor: object,
    proposal: OptimizationProposal,
) -> None:
    """Verify Proposal Decision plus every persisted Phase 3 evidence receipt."""

    decision_row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT checkpoints.checkpoint_json "
        "FROM decision_checkpoint_current AS current "
        "JOIN decision_checkpoints AS checkpoints "
        "ON checkpoints.request_id = current.request_id "
        "AND checkpoints.revision = current.revision "
        "WHERE current.request_id = ?",
        (str(proposal.source_decision_request_id),),
    ).fetchone()
    if decision_row is None:
        raise PersistenceConflictError("Optimization Proposal has no source Decision")
    checkpoint = json.loads(decision_row["checkpoint_json"])
    result = checkpoint.get("result") or {}
    normalized = (checkpoint.get("validated_decision") or {}).get(
        "normalized_effect"
    ) or {}
    if (
        checkpoint.get("stage") != "completed"
        or result.get("status") != "applied"
        or normalized.get("effect_fingerprint") != proposal.effect_fingerprint
    ):
        raise PersistenceConflictError("Optimization Proposal Decision is not APPLIED")
    effect = OptimizationProposalEffect.model_validate(normalized.get("payload"))
    if (
        effect.proposal_id != proposal.proposal_id
        or effect.scope != proposal.scope
        or effect.target.target_key != proposal.target_key
        or effect.target.current_value != proposal.current_value
        or effect.target.current_configuration_revision
        != proposal.current_configuration_revision
        or effect.target.current_configuration_fingerprint
        != proposal.current_configuration_fingerprint
        or effect.proposed_value != proposal.proposed_value
        or effect.evidence_set_fingerprint != proposal.evidence_set_fingerprint
    ):
        raise PersistenceConflictError("Optimization Proposal Decision payload is stale")

    snapshot = {item.candidate_ref: item for item in effect.evidence_snapshot}
    supporting = tuple(snapshot[item] for item in effect.supporting_candidate_refs)
    explicit_counter = tuple(
        snapshot[item] for item in effect.counterevidence_candidate_refs
    )
    supporting_feedback: list[UUID] = []
    supporting_experience: list[UUID] = []
    supporting_evaluation: list[UUID] = []
    derived_counter: list[str] = []
    learning_ids: list[UUID] = []
    for binding in supporting:
        insight_row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT insight_json, receipt_json FROM learning_insights "
            "WHERE learning_insight_id = ?",
            (str(binding.learning_insight_id),),
        ).fetchone()
        if insight_row is None:
            raise PersistenceConflictError("Optimization Learning evidence is missing")
        insight = LearningInsight.model_validate_json(insight_row["insight_json"])
        receipt = LearningInsightCommitReceipt.model_validate_json(
            insight_row["receipt_json"]
        )
        if (
            insight.version != binding.learning_insight_version
            or insight.effect_fingerprint
            != binding.learning_insight_effect_fingerprint
            or decision_fingerprint(insight) != binding.learning_insight_fingerprint
            or receipt.learning_insight_id != insight.learning_insight_id
            or receipt.effect_fingerprint != insight.effect_fingerprint
            or receipt.insight_fingerprint != decision_fingerprint(insight)
        ):
            raise PersistenceConflictError("Optimization Learning evidence is stale")
        learning_effect = _learning_effect(cursor, insight)
        verified_supporting = tuple(
            _verify_feedback_binding(cursor, item)
            for item in learning_effect.supporting_evidence
        )
        verified_counter = tuple(
            _verify_feedback_binding(cursor, item)
            for item in learning_effect.counterevidence
        )
        if not verified_supporting:
            raise PersistenceConflictError("Learning evidence has no supporting records")
        if tuple(item.feedback_id for item in verified_supporting) != (
            binding.supporting_feedback_refs
        ):
            raise PersistenceConflictError("Optimization Feedback binding changed")
        if tuple(item.feedback_id for item in verified_counter) != (
            binding.counterevidence_feedback_refs
        ):
            raise PersistenceConflictError("Optimization counterevidence changed")
        learning_ids.append(insight.learning_insight_id)
        supporting_feedback.extend(item.feedback_id for item in verified_supporting)
        supporting_experience.extend(
            item.experience_metadata_ref.experience_id for item in verified_supporting
        )
        supporting_evaluation.extend(
            item.evaluation_ref.report_id for item in verified_supporting
        )
        derived_counter.extend(
            f"feedback:{item.feedback_id}" for item in verified_counter
        )
    derived_counter.extend(
        f"learning_insight:{item.learning_insight_id}" for item in explicit_counter
    )
    if (
        tuple(dict.fromkeys(learning_ids))
        != proposal.supporting_learning_insight_refs
        or tuple(dict.fromkeys(supporting_feedback))
        != proposal.supporting_feedback_refs
        or tuple(dict.fromkeys(supporting_experience))
        != proposal.supporting_experience_refs
        or tuple(dict.fromkeys(supporting_evaluation))
        != proposal.supporting_evaluation_refs
        or tuple(dict.fromkeys(derived_counter)) != proposal.counterevidence_refs
    ):
        raise PersistenceConflictError("Optimization Phase 3 provenance changed")


def _learning_effect(cursor: object, insight: LearningInsight) -> LearningInsightEffect:
    rows = cursor.execute(  # type: ignore[attr-defined]
        "SELECT checkpoints.checkpoint_json FROM decision_checkpoint_current AS current "
        "JOIN decision_checkpoints AS checkpoints "
        "ON checkpoints.request_id = current.request_id "
        "AND checkpoints.revision = current.revision "
        "WHERE checkpoints.stage = 'completed' "
        "AND checkpoints.checkpoint_json LIKE ? "
        "AND checkpoints.checkpoint_json LIKE ?",
        (
            f"%{insight.effect_fingerprint}%",
            f"%{insight.learning_insight_id}%",
        ),
    ).fetchall()
    checkpoint_type = DecisionCheckpoint[
        LearningAssessmentRequest,
        LearningInsightDraft,
        LearningInsightEffect,
    ]
    for row in rows:
        try:
            checkpoint = checkpoint_type.model_validate_json(row["checkpoint_json"])
        except ValueError:
            continue
        validated = checkpoint.validated_decision
        if validated is None:
            continue
        normalized = validated.normalized_effect
        if (
            normalized.effect_fingerprint == insight.effect_fingerprint
            and normalized.payload.learning_insight_id == insight.learning_insight_id
        ):
            return normalized.payload
    raise PersistenceConflictError(
        "Learning Insight has no APPLIED Decision provenance"
    )


def _verify_feedback_binding(
    cursor: object,
    binding: object,
) -> DecisionFeedbackRecord:
    feedback_id = binding.feedback_id  # type: ignore[attr-defined]
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT record_json, receipt_json FROM decision_feedback WHERE feedback_id = ?",
        (str(feedback_id),),
    ).fetchone()
    if row is None:
        raise PersistenceConflictError("Optimization Feedback evidence is missing")
    record = DecisionFeedbackRecord.model_validate_json(row["record_json"])
    receipt = DecisionFeedbackCommitReceipt.model_validate_json(row["receipt_json"])
    if (
        record.attribution_type is not DecisionFeedbackAttributionType.ASSOCIATED
        or record.effect_fingerprint != binding.feedback_effect_fingerprint  # type: ignore[attr-defined]
        or receipt.feedback_id != record.feedback_id
        or receipt.effect_fingerprint != record.effect_fingerprint
        or receipt.record_fingerprint != decision_fingerprint(record)
    ):
        raise PersistenceConflictError("Optimization Feedback evidence is invalid")

    experience_row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT metadata_json, receipt_json FROM experience_metadata "
        "WHERE experience_id = ?",
        (str(record.experience_metadata_ref.experience_id),),
    ).fetchone()
    if experience_row is None:
        raise PersistenceConflictError("Optimization Experience evidence is missing")
    experience = ExperienceMetadata.model_validate_json(
        experience_row["metadata_json"]
    )
    experience_receipt = ExperienceMetadataCommitReceipt.model_validate_json(
        experience_row["receipt_json"]
    )
    if (
        experience.effect_fingerprint
        != record.experience_metadata_ref.effect_fingerprint
        or experience_receipt.metadata_fingerprint != decision_fingerprint(experience)
    ):
        raise PersistenceConflictError("Optimization Experience evidence is invalid")

    evaluation_row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT report_json, report_fingerprint FROM evaluation_reports "
        "WHERE report_id = ?",
        (str(record.evaluation_ref.report_id),),
    ).fetchone()
    if evaluation_row is None:
        raise PersistenceConflictError("Optimization Evaluation evidence is missing")
    evaluation = EvaluationReport.model_validate_json(evaluation_row["report_json"])
    if (
        evaluation_row["report_fingerprint"] != decision_fingerprint(evaluation)
        or evaluation_row["report_fingerprint"]
        != record.evaluation_ref.report_fingerprint
    ):
        raise PersistenceConflictError("Optimization Evaluation evidence is invalid")

    artifact_row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT receipt_json FROM workspace_artifacts WHERE effect_fingerprint = ?",
        (record.artifact_ref.effect_fingerprint,),
    ).fetchone()
    if artifact_row is None:
        raise PersistenceConflictError("Optimization Artifact provenance is missing")
    artifact_receipt = json.loads(artifact_row["receipt_json"])
    if not isinstance(artifact_receipt, Mapping) or (
        artifact_receipt.get("artifact_fingerprint")
        != record.artifact_ref.artifact_fingerprint
    ):
        raise PersistenceConflictError("Optimization Artifact provenance is invalid")
    return record
