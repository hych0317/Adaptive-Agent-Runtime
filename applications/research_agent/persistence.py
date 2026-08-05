"""Application snapshots required to reconstruct a Research Runtime composition."""

from __future__ import annotations

import json
from types import MappingProxyType
from uuid import UUID

from adaptive_agent_runtime.orchestration import DynamicTaskGraph, TaskNode
from adaptive_agent_runtime.persistence import SQLiteDatabase

from applications.research_agent.tasks import ResearchTaskDefinition


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
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._database.transaction() as cursor:
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

    def load(self, run_id: UUID) -> tuple[str, ResearchTaskDefinition] | None:
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
        )
