"""Small shared SQLite backend for Runtime persistence adapters."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Iterator

from adaptive_agent_runtime.persistence.errors import PersistenceSchemaError


_SCHEMA_VERSION = 13

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runtime_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_authority_keys (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    secret_hex TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_state_snapshots (
    run_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (run_id, revision)
);
CREATE TABLE IF NOT EXISTS agent_state_current (
    run_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_trace (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    entry_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS task_graph_checkpoints (
    run_id TEXT PRIMARY KEY,
    graph_version INTEGER NOT NULL,
    state_revision INTEGER NOT NULL,
    checkpoint_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_graph_history (
    run_id TEXT NOT NULL,
    graph_version INTEGER NOT NULL,
    state_revision INTEGER NOT NULL,
    checkpoint_json TEXT NOT NULL,
    PRIMARY KEY (run_id, graph_version)
);
CREATE TABLE IF NOT EXISTS task_graph_checkpoint_journal (
    run_id TEXT NOT NULL,
    checkpoint_revision INTEGER NOT NULL,
    graph_version INTEGER NOT NULL,
    state_revision INTEGER NOT NULL,
    checkpoint_json TEXT NOT NULL,
    PRIMARY KEY (run_id, checkpoint_revision)
);

CREATE TABLE IF NOT EXISTS context_snapshots (
    context_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (context_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_context_snapshots_run
    ON context_snapshots (run_id);
CREATE TABLE IF NOT EXISTS context_current (
    context_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS context_archives (
    archive_id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL,
    snapshot_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS context_archive_transactions (
    effect_fingerprint TEXT PRIMARY KEY,
    archive_id TEXT NOT NULL UNIQUE,
    context_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'committed')),
    archive_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_snapshots (
    memory_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (memory_id, revision)
);
CREATE TABLE IF NOT EXISTS memory_current (
    memory_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_applied_candidates (
    candidate_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_applied_effects (
    effect_fingerprint TEXT PRIMARY KEY,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_batch_receipts (
    effect_fingerprint TEXT PRIMARY KEY,
    payload_fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_recall_bundles (
    effect_fingerprint TEXT PRIMARY KEY,
    bundle_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_recall_bundles_run
    ON memory_recall_bundles (run_id);
CREATE TABLE IF NOT EXISTS experience_metadata (
    effect_fingerprint TEXT PRIMARY KEY,
    experience_id TEXT NOT NULL UNIQUE,
    source_run_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    source_memory_refs_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    UNIQUE (source_run_id, version)
);
CREATE INDEX IF NOT EXISTS idx_experience_metadata_run
    ON experience_metadata (source_run_id, version);
CREATE TABLE IF NOT EXISTS experience_memory_links (
    effect_fingerprint TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (effect_fingerprint, memory_id),
    FOREIGN KEY (effect_fingerprint)
        REFERENCES experience_metadata(effect_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_experience_memory_links_memory
    ON experience_memory_links (memory_id);

CREATE TABLE IF NOT EXISTS evaluation_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    report_fingerprint TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_feedback (
    effect_fingerprint TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL UNIQUE,
    source_run_id TEXT NOT NULL,
    subject_decision_id TEXT NOT NULL,
    subject_decision_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    record_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    UNIQUE (source_run_id, subject_decision_id, version)
);
CREATE INDEX IF NOT EXISTS idx_decision_feedback_run
    ON decision_feedback (source_run_id, subject_decision_id, version);
CREATE TABLE IF NOT EXISTS learning_insights (
    effect_fingerprint TEXT PRIMARY KEY,
    learning_insight_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    agent_scope TEXT NOT NULL,
    subject_decision_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    evidence_set_fingerprint TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    insight_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    UNIQUE (
        tenant_id,
        project_id,
        agent_scope,
        subject_decision_type,
        version
    )
);
CREATE INDEX IF NOT EXISTS idx_learning_insights_scope
    ON learning_insights (
        tenant_id,
        project_id,
        agent_scope,
        subject_decision_type,
        version
    );

CREATE TABLE IF NOT EXISTS optimization_proposals (
    effect_fingerprint TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE,
    source_decision_request_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    application_id TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    target_key TEXT NOT NULL,
    baseline_fingerprint TEXT NOT NULL,
    evidence_set_fingerprint TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    proposal_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_optimization_proposals_scope
    ON optimization_proposals (
        tenant_id,
        project_id,
        application_id,
        decision_type
    );
CREATE TABLE IF NOT EXISTS optimization_proposal_receipts (
    effect_fingerprint TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE,
    receipt_json TEXT NOT NULL,
    FOREIGN KEY (effect_fingerprint)
        REFERENCES optimization_proposals(effect_fingerprint)
);

CREATE TABLE IF NOT EXISTS governed_runtime_configuration_snapshots (
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    application_id TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    target_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_fingerprint TEXT NOT NULL UNIQUE,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (
        tenant_id,
        project_id,
        application_id,
        decision_type,
        target_key,
        revision
    )
);
CREATE TABLE IF NOT EXISTS governed_runtime_configuration_active (
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    application_id TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    target_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_fingerprint TEXT NOT NULL,
    PRIMARY KEY (
        tenant_id,
        project_id,
        application_id,
        decision_type,
        target_key
    ),
    FOREIGN KEY (
        tenant_id,
        project_id,
        application_id,
        decision_type,
        target_key,
        revision
    ) REFERENCES governed_runtime_configuration_snapshots (
        tenant_id,
        project_id,
        application_id,
        decision_type,
        target_key,
        revision
    )
);
CREATE TABLE IF NOT EXISTS optimization_configuration_receipts (
    effect_fingerprint TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    receipt_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auto_adaptation_triggers (
    trigger_run_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    policy_fingerprint TEXT NOT NULL,
    selected_proposal_id TEXT UNIQUE,
    trigger_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auto_adaptation_trigger_history (
    trigger_run_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    trigger_json TEXT NOT NULL,
    PRIMARY KEY (trigger_run_id, revision),
    FOREIGN KEY (trigger_run_id)
        REFERENCES auto_adaptation_triggers(trigger_run_id)
);

CREATE TABLE IF NOT EXISTS governance_reviews (
    review_request_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS governance_review_history (
    review_request_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (review_request_id, revision)
);
CREATE TABLE IF NOT EXISTS governance_authorization_uses (
    authorization_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    use_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS governance_authorization_use_history (
    authorization_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    use_json TEXT NOT NULL,
    PRIMARY KEY (authorization_id, revision)
);

CREATE TABLE IF NOT EXISTS runtime_configuration_snapshots (
    component TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (component, version)
);
CREATE TABLE IF NOT EXISTS runtime_configuration_active (
    component TEXT PRIMARY KEY,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS optimization_application_current (
    application_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    application_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS optimization_application_history (
    application_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    application_json TEXT NOT NULL,
    PRIMARY KEY (application_id, revision)
);
CREATE TABLE IF NOT EXISTS replay_cases (
    case_id TEXT PRIMARY KEY,
    case_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_checkpoints (
    request_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    PRIMARY KEY (request_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_decision_checkpoints_run
    ON decision_checkpoints (run_id);
CREATE TABLE IF NOT EXISTS decision_checkpoint_current (
    request_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS application_run_manifests (
    application_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    PRIMARY KEY (application_id, run_id)
);
CREATE TABLE IF NOT EXISTS workspace_artifacts (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    effect_fingerprint TEXT NOT NULL UNIQUE,
    artifact_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    PRIMARY KEY (run_id, node_id, artifact_type)
);
"""


class SQLiteDatabase:
    """One locked connection shared by independent storage adapters."""

    module_id = "persistence.sqlite"

    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)
        if raw_path != ":memory:":
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(resolved)
        self.path = raw_path
        self._lock = RLock()
        self._connection = sqlite3.connect(
            raw_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)
            row = self._connection.execute(
                "SELECT version FROM runtime_schema WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO runtime_schema(singleton, version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row["version"]) > _SCHEMA_VERSION:
                raise PersistenceSchemaError(
                    f"unsupported persistence schema version {row['version']}"
                )
            elif int(row["version"]) < _SCHEMA_VERSION:
                self._connection.execute(
                    "UPDATE runtime_schema SET version = ? WHERE singleton = 1",
                    (_SCHEMA_VERSION,),
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Serialize a short write transaction on the shared connection."""

        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
