"""Opt-in durable adapters for independent Runtime module contracts."""

from __future__ import annotations

from pathlib import Path
import secrets
from typing import TypeVar

from pydantic import BaseModel

from adaptive_agent_runtime.decisioning.models import DecisionCheckpoint
from adaptive_agent_runtime.governance import (
    CommitPermitVerifier,
    GovernanceAuthorizationIssuer,
    GovernedOperationExecutor,
    HMACCommitPermitAuthority,
    StrictAuthorizationVerifier,
)

from adaptive_agent_runtime.persistence.context_memory import (
    SQLiteContextArchive,
    SQLiteContextStore,
    SQLiteMemoryStore,
)
from adaptive_agent_runtime.persistence.core import (
    SQLiteStateStore,
    SQLiteTraceSink,
)
from adaptive_agent_runtime.persistence.errors import (
    PersistenceConflictError,
    PersistenceError,
    PersistenceSchemaError,
)
from adaptive_agent_runtime.persistence.governance import (
    SQLiteAuthorizationConsumptionStore,
    SQLiteHumanReviewService,
)
from adaptive_agent_runtime.persistence.decisioning import (
    DecisionProof,
    SQLiteDecisionCheckpointStore,
    SQLiteDecisionRecordReader,
)
from adaptive_agent_runtime.persistence.orchestration import SQLiteTaskGraphStore
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase
from adaptive_agent_runtime.persistence.artifacts import (
    SQLiteWorkspaceArtifactStore,
    WorkspaceArtifactCommitReceipt,
    WorkspaceArtifactCommitter,
)
from adaptive_agent_runtime.persistence.memory_recall import (
    SQLiteMemoryRecallBundleStore,
)
from adaptive_agent_runtime.persistence.experience import (
    SQLiteExperienceMetadataStore,
)
from adaptive_agent_runtime.persistence.evaluation_reports import (
    SQLiteEvaluationReportStore,
)
from adaptive_agent_runtime.persistence.decision_feedback import (
    SQLiteDecisionFeedbackStore,
)
from adaptive_agent_runtime.persistence.learning_insights import (
    SQLiteLearningInsightStore,
)
from adaptive_agent_runtime.persistence.optimization import (
    SQLiteOptimizationEvidenceResolver,
    SQLiteOptimizationProposalStore,
)
from adaptive_agent_runtime.persistence.optimization_configuration import (
    SQLiteGovernedRuntimeConfigurationStore,
    default_runtime_configuration_snapshot,
)
from adaptive_agent_runtime.persistence.auto_adaptation import (
    SQLiteAutoAdaptationTriggerStore,
)
from adaptive_agent_runtime.persistence.retention import (
    PurgeRunReport,
    SQLiteDisposableRunCleaner,
)


RequestCheckpointT = TypeVar("RequestCheckpointT", bound=BaseModel)
ProposalCheckpointT = TypeVar("ProposalCheckpointT", bound=BaseModel)
EffectCheckpointT = TypeVar("EffectCheckpointT", bound=BaseModel)


class SQLitePersistence:
    """Composition root exposing one database through narrow module stores."""

    def __init__(
        self,
        path: str | Path,
        *,
        enforce_authoritative_commits: bool = True,
    ) -> None:
        self.database = SQLiteDatabase(path)
        self.state_store = SQLiteStateStore(self.database)
        self.trace_sink = SQLiteTraceSink(self.database)
        self.decision_records = SQLiteDecisionRecordReader(self.database)
        self.disposable_runs = SQLiteDisposableRunCleaner(self.database)
        self.context_store = SQLiteContextStore(self.database)
        self.human_review_service = SQLiteHumanReviewService(self.database)
        self.authorization_store = SQLiteAuthorizationConsumptionStore(
            self.database
        )
        with self.database.transaction() as cursor:
            row = cursor.execute(
                "SELECT secret_hex FROM runtime_authority_keys WHERE singleton = 1"
            ).fetchone()
            if row is None:
                secret_hex = secrets.token_hex(32)
                cursor.execute(
                    "INSERT INTO runtime_authority_keys(singleton, secret_hex) "
                    "VALUES (1, ?)",
                    (secret_hex,),
                )
            else:
                secret_hex = str(row["secret_hex"])
        self._commit_permit_authority = HMACCommitPermitAuthority(
            bytes.fromhex(secret_hex)
        )
        self.commit_permit_verifier = CommitPermitVerifier(
            authority=self._commit_permit_authority,
            consumption_store=self.authorization_store,
        )
        self.authorization_issuer = GovernanceAuthorizationIssuer(
            authority=self._commit_permit_authority
        )
        self.operation_executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(self._commit_permit_authority),
            consumption_store=self.authorization_store,
            permit_authority=self._commit_permit_authority,
        )
        self.task_graph_store = SQLiteTaskGraphStore(
            self.database,
            permit_verifier=(
                self.commit_permit_verifier
                if enforce_authoritative_commits
                else None
            ),
        )
        self.context_archive = SQLiteContextArchive(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.memory_store = SQLiteMemoryStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.memory_recall_bundle_store = SQLiteMemoryRecallBundleStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.experience_metadata_store = SQLiteExperienceMetadataStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.evaluation_report_store = SQLiteEvaluationReportStore(self.database)
        self.decision_feedback_store = SQLiteDecisionFeedbackStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.learning_insight_store = SQLiteLearningInsightStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.optimization_evidence_resolver = SQLiteOptimizationEvidenceResolver(
            self.database
        )
        self.optimization_proposal_store = SQLiteOptimizationProposalStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.runtime_configuration = SQLiteGovernedRuntimeConfigurationStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.auto_adaptation_trigger_store = SQLiteAutoAdaptationTriggerStore(
            self.database
        )
        self.workspace_artifact_store = SQLiteWorkspaceArtifactStore(
            self.database,
            permit_verifier=self.commit_permit_verifier,
        )
        self.workspace_artifact_committer = WorkspaceArtifactCommitter(
            self.workspace_artifact_store
        )

    def create_decision_checkpoint_store(
        self,
        checkpoint_type: type[
            DecisionCheckpoint[
                RequestCheckpointT,
                ProposalCheckpointT,
                EffectCheckpointT,
            ]
        ],
    ) -> SQLiteDecisionCheckpointStore[
        RequestCheckpointT,
        ProposalCheckpointT,
        EffectCheckpointT,
    ]:
        """Create a typed store without introducing a global Draft registry."""

        return SQLiteDecisionCheckpointStore(self.database, checkpoint_type)

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> SQLitePersistence:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = [
    "DecisionProof",
    "PersistenceConflictError",
    "PersistenceError",
    "PersistenceSchemaError",
    "PurgeRunReport",
    "SQLiteContextArchive",
    "SQLiteContextStore",
    "SQLiteDatabase",
    "SQLiteDecisionCheckpointStore",
    "SQLiteDecisionRecordReader",
    "SQLiteDisposableRunCleaner",
    "SQLiteEvaluationReportStore",
    "SQLiteExperienceMetadataStore",
    "SQLiteDecisionFeedbackStore",
    "SQLiteLearningInsightStore",
    "SQLiteOptimizationEvidenceResolver",
    "SQLiteOptimizationProposalStore",
    "SQLiteGovernedRuntimeConfigurationStore",
    "SQLiteAutoAdaptationTriggerStore",
    "SQLiteMemoryStore",
    "SQLiteMemoryRecallBundleStore",
    "SQLiteAuthorizationConsumptionStore",
    "SQLiteHumanReviewService",
    "SQLitePersistence",
    "SQLiteStateStore",
    "SQLiteTaskGraphStore",
    "SQLiteTraceSink",
    "SQLiteWorkspaceArtifactStore",
    "WorkspaceArtifactCommitReceipt",
    "WorkspaceArtifactCommitter",
    "default_runtime_configuration_snapshot",
]
