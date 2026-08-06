"""SQLite authority boundary for append-only Experience Metadata."""

from __future__ import annotations

import json
from uuid import UUID

from adaptive_agent_runtime.context_memory import (
    ExperienceExecutionOutcome,
    ExperienceMetadata,
    ExperienceMetadataCommitReceipt,
    ExperienceMetadataEffect,
    MemoryUnit,
)
from adaptive_agent_runtime.core import AgentState, RunStatus
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance import (
    AuthorizationVerificationError,
    CommitPermitValidation,
    GovernanceTarget,
    RuntimeCommitPermit,
)
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.decisioning import SQLiteDecisionRecordReader
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class SQLiteExperienceMetadataStore:
    """Permit-guarded store exposing commit and read operations only."""

    module_id = "experience.metadata_store.sqlite"

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
        effect: ExperienceMetadataEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> ExperienceMetadata:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Experience Metadata commit requires a Runtime Permit"
            )
        await self._permit_verifier.verify(
            permit,
            operation="experience.metadata.commit",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        payload_fingerprint = decision_fingerprint(effect)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT payload_fingerprint, metadata_json FROM experience_metadata "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if prior is not None:
                if prior["payload_fingerprint"] != payload_fingerprint:
                    raise PersistenceConflictError(
                        "Experience Effect fingerprint conflicts with stored payload"
                    )
                return ExperienceMetadata.model_validate_json(prior["metadata_json"])

            state_row = cursor.execute(
                "SELECT snapshots.snapshot_json FROM agent_state_current AS current "
                "JOIN agent_state_snapshots AS snapshots "
                "ON snapshots.run_id = current.run_id "
                "AND snapshots.revision = current.revision "
                "WHERE current.run_id = ?",
                (str(effect.source_run_id),),
            ).fetchone()
            if state_row is None:
                raise PersistenceConflictError(
                    "Experience source execution has no persisted outcome"
                )
            state = AgentState.model_validate_json(state_row["snapshot_json"])
            expected_statuses = (
                {RunStatus.COMPLETED}
                if effect.execution_outcome is ExperienceExecutionOutcome.SUCCEEDED
                else {RunStatus.FAILED, RunStatus.TERMINATED}
            )
            if (
                state.status not in expected_statuses
                or state.revision != effect.runtime_observation.state_revision
                or decision_fingerprint(state)
                != effect.runtime_observation.state_fingerprint
            ):
                raise PersistenceConflictError(
                    "Experience source execution changed or is not terminal"
                )

            for artifact in effect.source_artifacts:
                row = cursor.execute(
                    "SELECT receipt_json FROM workspace_artifacts "
                    "WHERE run_id = ? AND node_id = ? AND artifact_type = ? "
                    "AND effect_fingerprint = ?",
                    (
                        str(effect.source_run_id),
                        str(artifact.node_id),
                        artifact.artifact_type,
                        artifact.effect_fingerprint,
                    ),
                ).fetchone()
                if row is None:
                    raise PersistenceConflictError(
                        "Experience source Artifact is not committed"
                    )
                receipt = json.loads(row["receipt_json"])
                if (
                    receipt.get("artifact_fingerprint")
                    != artifact.artifact_fingerprint
                    or receipt.get("source_decision_request_id")
                    != str(artifact.source_decision_id)
                ):
                    raise PersistenceConflictError(
                        "Experience source Artifact receipt does not match"
                    )

            for decision_id in effect.source_decision_ids:
                proof = SQLiteDecisionRecordReader.load_proof_in_transaction(
                    cursor, decision_id
                )
                if proof is None:
                    raise PersistenceConflictError(
                        "Experience references an uncommitted Decision"
                    )
                if not proof.is_applied:
                    raise PersistenceConflictError(
                        "Experience source Decision is not APPLIED"
                    )

            for source in effect.source_memories:
                row = cursor.execute(
                    "SELECT snapshots.snapshot_json FROM memory_current AS current "
                    "JOIN memory_snapshots AS snapshots "
                    "ON snapshots.memory_id = current.memory_id "
                    "AND snapshots.revision = current.revision "
                    "WHERE current.memory_id = ?",
                    (str(source.memory_id),),
                ).fetchone()
                if row is None:
                    raise PersistenceConflictError(
                        "Experience source Memory is unavailable"
                    )
                memory = MemoryUnit.model_validate_json(row["snapshot_json"])
                if (
                    memory.revision != source.revision
                    or decision_fingerprint(memory) != source.memory_fingerprint
                ):
                    raise PersistenceConflictError(
                        "Experience source Memory changed before commit"
                    )

            version_row = cursor.execute(
                "SELECT COALESCE(MAX(version), 0) AS version "
                "FROM experience_metadata WHERE source_run_id = ?",
                (str(effect.source_run_id),),
            ).fetchone()
            version = int(version_row["version"]) + 1
            metadata = ExperienceMetadata(
                experience_id=effect.experience_id,
                version=version,
                source_execution_id=effect.source_execution_id,
                source_run_id=effect.source_run_id,
                source_task_id=effect.source_task_id,
                source_decision_ids=effect.source_decision_ids,
                source_artifacts=effect.source_artifacts,
                source_evaluations=effect.source_evaluations,
                source_memory_refs=tuple(
                    item.memory_id for item in effect.source_memories
                ),
                runtime_observation=effect.runtime_observation,
                execution_outcome=effect.execution_outcome,
                success_signal=effect.success_signal,
                failure_signal=effect.failure_signal,
                evidence_refs=effect.evidence_refs,
                agent_assessment_summary=effect.agent_assessment_summary,
                effect_fingerprint=effect_fingerprint,
            )
            receipt = ExperienceMetadataCommitReceipt(
                experience_id=metadata.experience_id,
                version=metadata.version,
                effect_fingerprint=effect_fingerprint,
                payload_fingerprint=payload_fingerprint,
                metadata_fingerprint=decision_fingerprint(metadata),
                committed_at=metadata.committed_at,
            )
            cursor.execute(
                "INSERT INTO experience_metadata "
                "(effect_fingerprint, experience_id, source_run_id, version, "
                "payload_fingerprint, source_memory_refs_json, metadata_json, "
                "receipt_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    effect_fingerprint,
                    str(metadata.experience_id),
                    str(metadata.source_run_id),
                    version,
                    payload_fingerprint,
                    json.dumps(
                        [str(item) for item in metadata.source_memory_refs],
                        separators=(",", ":"),
                    ),
                    metadata.model_dump_json(),
                    receipt.model_dump_json(),
                ),
            )
            for memory_id in metadata.source_memory_refs:
                cursor.execute(
                    "INSERT INTO experience_memory_links "
                    "(effect_fingerprint, memory_id) VALUES (?, ?)",
                    (effect_fingerprint, str(memory_id)),
                )
            return metadata

    async def load_by_effect(
        self,
        effect_fingerprint: str,
    ) -> ExperienceMetadata | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT metadata_json, receipt_json FROM experience_metadata "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        metadata = ExperienceMetadata.model_validate_json(row["metadata_json"])
        receipt = ExperienceMetadataCommitReceipt.model_validate_json(
            row["receipt_json"]
        )
        if (
            metadata.effect_fingerprint != effect_fingerprint
            or receipt.effect_fingerprint != effect_fingerprint
            or receipt.experience_id != metadata.experience_id
            or receipt.version != metadata.version
            or receipt.metadata_fingerprint != decision_fingerprint(metadata)
        ):
            raise PersistenceConflictError(
                "Experience Metadata read-back is corrupted"
            )
        return metadata

    async def load_receipt(
        self,
        effect_fingerprint: str,
    ) -> ExperienceMetadataCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT receipt_json FROM experience_metadata "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        return (
            None
            if row is None
            else ExperienceMetadataCommitReceipt.model_validate_json(
                row["receipt_json"]
            )
        )

    async def list_for_run(
        self,
        run_id: UUID,
    ) -> tuple[ExperienceMetadata, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT metadata_json FROM experience_metadata "
                "WHERE source_run_id = ? ORDER BY version",
                (str(run_id),),
            ).fetchall()
        return tuple(
            ExperienceMetadata.model_validate_json(row["metadata_json"])
            for row in rows
        )

    async def list_for_memory(
        self,
        memory_id: UUID,
    ) -> tuple[ExperienceMetadata, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT metadata.metadata_json FROM experience_memory_links AS links "
                "JOIN experience_metadata AS metadata "
                "ON metadata.effect_fingerprint = links.effect_fingerprint "
                "WHERE links.memory_id = ? "
                "ORDER BY metadata.source_run_id, metadata.version",
                (str(memory_id),),
            ).fetchall()
        return tuple(
            ExperienceMetadata.model_validate_json(row["metadata_json"])
            for row in rows
        )
