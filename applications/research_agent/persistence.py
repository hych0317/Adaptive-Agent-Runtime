"""Application snapshots required to reconstruct a Research Runtime composition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Protocol
from uuid import UUID

from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    TaskGraphCheckpoint,
    TaskNode,
    TaskNodeStatus,
)
from adaptive_agent_runtime.persistence import (
    DecisionProof,
    SQLiteDatabase,
    WorkspaceArtifactCommitReceipt,
)
from adaptive_agent_runtime.optimization import RuntimeConfigurationSnapshot

from applications.research_agent.tasks import ResearchTaskDefinition


@dataclass(frozen=True)
class ResearchPersistenceIdentity:
    database_path: str
    adapter_database_paths: tuple[tuple[str, str], ...]

    @property
    def is_persistent(self) -> bool:
        return self.database_path != ":memory:"


class DecisionProofQueries(Protocol):
    def load_proof_in_transaction(
        self, cursor: object, request_id: UUID
    ) -> DecisionProof | None: ...


class SQLiteResearchRunManifestStore:
    module_id = "research_agent.run_manifest.sqlite"
    application_id = "research_agent"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save(
        self,
        *,
        run_id: UUID,
        task: str,
        definition: ResearchTaskDefinition,
        configuration_snapshot: RuntimeConfigurationSnapshot,
        run_kind: str,
        disposable: bool,
    ) -> None:
        payload = json.dumps(
            {
                "task": task,
                "company": definition.company,
                "initial_graph": definition.initial_graph.model_dump(mode="json"),
                "nodes": {
                    role: node.model_dump(mode="json")
                    for role, node in definition.nodes.items()
                },
                "configuration_snapshot": configuration_snapshot.model_dump(
                    mode="json"
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._database.transaction() as cursor:
            metadata = cursor.execute(
                "SELECT run_kind, disposable FROM runtime_run_metadata "
                "WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            if metadata is None:
                cursor.execute(
                    "INSERT INTO runtime_run_metadata "
                    "(run_id, run_kind, disposable, created_at) VALUES (?, ?, ?, ?)",
                    (
                        str(run_id),
                        run_kind,
                        int(disposable),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            elif (
                metadata["run_kind"] != run_kind
                or int(metadata["disposable"]) != int(disposable)
            ):
                raise RuntimeError("Research run retention identity was reused")
            current = cursor.execute(
                "SELECT manifest_json FROM application_run_manifests "
                "WHERE application_id = ? AND run_id = ?",
                (self.application_id, str(run_id)),
            ).fetchone()
            if current is not None:
                if current["manifest_json"] != payload:
                    raise RuntimeError("Research run manifest identity was reused")
                return
            cursor.execute(
                "INSERT INTO application_run_manifests "
                "(application_id, run_id, manifest_json) VALUES (?, ?, ?)",
                (self.application_id, str(run_id), payload),
            )

    def load(
        self, run_id: UUID
    ) -> tuple[
        str,
        ResearchTaskDefinition,
        RuntimeConfigurationSnapshot | None,
    ] | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT manifest_json FROM application_run_manifests "
                "WHERE application_id = ? AND run_id = ?",
                (self.application_id, str(run_id)),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["manifest_json"])
        nodes = {
            role: TaskNode.model_validate(node)
            for role, node in payload["nodes"].items()
        }
        return (
            str(payload["task"]),
            ResearchTaskDefinition(
                company=str(payload["company"]),
                initial_graph=DynamicTaskGraph.model_validate(payload["initial_graph"]),
                nodes=MappingProxyType(nodes),
            ),
            (
                RuntimeConfigurationSnapshot.model_validate(
                    payload["configuration_snapshot"]
                )
                if payload.get("configuration_snapshot") is not None
                else None
            ),
        )


class SQLiteReportDispatchReconciler:
    """Reopen only a report dispatch whose exact Decision Effect is committed."""

    module_id = "research_agent.report_dispatch_reconciler.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        decisions: DecisionProofQueries,
    ) -> None:
        self._database = database
        self._decisions = decisions

    async def reconcile(self, run_id: UUID) -> bool:
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT checkpoint_json FROM task_graph_checkpoints WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            if row is None:
                return False
            checkpoint = TaskGraphCheckpoint.model_validate_json(
                row["checkpoint_json"]
            )
            reopened_ids: set[UUID] = set()
            for in_flight in checkpoint.in_flight:
                artifact_row = cursor.execute(
                    "SELECT artifact_json, receipt_json FROM workspace_artifacts "
                    "WHERE run_id = ? AND node_id = ? AND artifact_type = ?",
                    (str(run_id), str(in_flight.node_id), "research_report"),
                ).fetchone()
                if artifact_row is None:
                    continue
                artifact = json.loads(artifact_row["artifact_json"])
                receipt = WorkspaceArtifactCommitReceipt.model_validate_json(
                    artifact_row["receipt_json"]
                )
                request_id = receipt.source_decision_request_id
                proposal_id = receipt.source_proposal_id
                if (
                    request_id is None
                    or proposal_id is None
                    or receipt.run_id != run_id
                    or receipt.node_id != in_flight.node_id
                    or receipt.artifact_fingerprint != decision_fingerprint(artifact)
                ):
                    raise RuntimeError(
                        "report Artifact receipt is not bound to its persisted content"
                    )
                proof = self._decisions.load_proof_in_transaction(
                    cursor, request_id
                )
                if proof is None:
                    raise RuntimeError(
                        "report Artifact has no durable source Decision"
                    )
                if (
                    proof.stage.value not in {
                        "applying",
                        "effect_committed",
                        "completed",
                    }
                    or proof.decision_type != "artifact.report_commit"
                    or proof.request_id != request_id
                    or proof.target_id != f"{run_id}:{in_flight.node_id}"
                    or proof.proposal_id != proposal_id
                    or proof.effect_fingerprint != receipt.effect_fingerprint
                ):
                    raise RuntimeError(
                        "report dispatch reconciliation proof is inconsistent"
                    )
                reopened_ids.add(in_flight.node_id)
            if not reopened_ids:
                return False
            nodes = tuple(
                node.model_copy(
                    update={
                        "status": TaskNodeStatus.PENDING,
                        "observation": None,
                        "failure_reason": None,
                    }
                )
                if node.node_id in reopened_ids
                else node
                for node in checkpoint.graph.nodes
            )
            graph = DynamicTaskGraph(
                graph_id=checkpoint.graph.graph_id,
                version=checkpoint.graph.version,
                nodes=nodes,
            )
            reconciled = checkpoint.model_copy(
                update={
                    "graph": graph,
                    "in_flight": tuple(
                        item
                        for item in checkpoint.in_flight
                        if item.node_id not in reopened_ids
                    ),
                    "checkpoint_revision": checkpoint.checkpoint_revision + 1,
                }
            )
            payload = json.dumps(
                reconciled.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            cursor.execute(
                "INSERT INTO task_graph_checkpoint_journal "
                "(run_id, checkpoint_revision, graph_version, state_revision, "
                "checkpoint_json) VALUES (?, ?, ?, ?, ?)",
                (
                    str(run_id),
                    reconciled.checkpoint_revision,
                    reconciled.graph.version,
                    reconciled.state_revision,
                    payload,
                ),
            )
            cursor.execute(
                "UPDATE task_graph_checkpoints SET graph_version = ?, "
                "state_revision = ?, checkpoint_json = ? WHERE run_id = ?",
                (
                    reconciled.graph.version,
                    reconciled.state_revision,
                    payload,
                    str(run_id),
                ),
            )
        return True
