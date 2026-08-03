"""Durable Replay cases and atomic Runtime configuration evolution."""

from __future__ import annotations

import json
from uuid import UUID

from adaptive_agent_runtime.context_memory.json_types import utc_now
from adaptive_agent_runtime.evolution import (
    EvolutionConflictError,
    OptimizationApplication,
    OptimizationApplicationStatus,
    OptimizationDeployment,
    ReplayCase,
    ReplayValidationError,
    RuntimeConfigurationSnapshot,
    stable_evolution_id,
)
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


def _json_text(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class SQLiteEvolutionStore:
    """One transactional store implementing configuration and Replay contracts."""

    module_id = "evolution.store.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def initialize(self, snapshot: RuntimeConfigurationSnapshot) -> None:
        payload = _json_text(snapshot)
        with self._database.transaction() as cursor:
            active = cursor.execute(
                "SELECT active.version, snapshots.snapshot_json "
                "FROM runtime_configuration_active AS active "
                "JOIN runtime_configuration_snapshots AS snapshots "
                "ON snapshots.component = active.component "
                "AND snapshots.version = active.version "
                "WHERE active.component = ?",
                (snapshot.component,),
            ).fetchone()
            if active is not None:
                if active["snapshot_json"] == payload:
                    return
                raise EvolutionConflictError(
                    "component configuration is already initialized"
                )
            cursor.execute(
                "INSERT INTO runtime_configuration_snapshots "
                "(component, version, snapshot_json) VALUES (?, ?, ?)",
                (snapshot.component, snapshot.version, payload),
            )
            cursor.execute(
                "INSERT INTO runtime_configuration_active(component, version) "
                "VALUES (?, ?)",
                (snapshot.component, snapshot.version),
            )

    async def load_active(
        self,
        component: str,
    ) -> RuntimeConfigurationSnapshot | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshots.snapshot_json "
                "FROM runtime_configuration_active AS active "
                "JOIN runtime_configuration_snapshots AS snapshots "
                "ON snapshots.component = active.component "
                "AND snapshots.version = active.version "
                "WHERE active.component = ?",
                (component,),
            ).fetchone()
        if row is None:
            return None
        return RuntimeConfigurationSnapshot.model_validate_json(
            row["snapshot_json"]
        )

    async def activate(
        self,
        deployment: OptimizationDeployment,
    ) -> OptimizationApplication:
        if not deployment.validation.passed:
            raise ReplayValidationError("candidate failed Replay validation")
        component = deployment.baseline.component
        application_id = stable_evolution_id(
            "optimization-application",
            deployment.deployment_id,
        )
        with self._database.transaction() as cursor:
            existing_application = cursor.execute(
                "SELECT application_json FROM optimization_application_current "
                "WHERE application_id = ?",
                (str(application_id),),
            ).fetchone()
            if existing_application is not None:
                application = OptimizationApplication.model_validate_json(
                    existing_application["application_json"]
                )
                if application.deployment == deployment:
                    return application
                raise EvolutionConflictError(
                    "Optimization application id was reused"
                )
            active = cursor.execute(
                "SELECT active.version, snapshots.snapshot_json "
                "FROM runtime_configuration_active AS active "
                "JOIN runtime_configuration_snapshots AS snapshots "
                "ON snapshots.component = active.component "
                "AND snapshots.version = active.version "
                "WHERE active.component = ?",
                (component,),
            ).fetchone()
            if (
                active is None
                or RuntimeConfigurationSnapshot.model_validate_json(
                    active["snapshot_json"]
                )
                != deployment.baseline
            ):
                raise EvolutionConflictError(
                    "active configuration changed after Replay"
                )
            candidate_payload = _json_text(deployment.candidate)
            existing_candidate = cursor.execute(
                "SELECT snapshot_json FROM runtime_configuration_snapshots "
                "WHERE component = ? AND version = ?",
                (component, deployment.candidate.version),
            ).fetchone()
            if existing_candidate is None:
                cursor.execute(
                    "INSERT INTO runtime_configuration_snapshots "
                    "(component, version, snapshot_json) VALUES (?, ?, ?)",
                    (
                        component,
                        deployment.candidate.version,
                        candidate_payload,
                    ),
                )
            elif existing_candidate["snapshot_json"] != candidate_payload:
                raise EvolutionConflictError(
                    "candidate configuration version already exists"
                )
            cursor.execute(
                "UPDATE runtime_configuration_active SET version = ? "
                "WHERE component = ?",
                (deployment.candidate.version, component),
            )
            now = utc_now()
            application = OptimizationApplication(
                application_id=application_id,
                deployment=deployment,
                status=OptimizationApplicationStatus.APPLIED,
                applied_at=now,
                updated_at=now,
            )
            application_payload = _json_text(application)
            cursor.execute(
                "INSERT INTO optimization_application_current "
                "(application_id, revision, application_json) VALUES (?, ?, ?)",
                (str(application_id), application.revision, application_payload),
            )
            cursor.execute(
                "INSERT INTO optimization_application_history "
                "(application_id, revision, application_json) VALUES (?, ?, ?)",
                (str(application_id), application.revision, application_payload),
            )
        return application

    async def rollback(
        self,
        application_id: UUID,
    ) -> OptimizationApplication:
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT application_json FROM optimization_application_current "
                "WHERE application_id = ?",
                (str(application_id),),
            ).fetchone()
            if row is None:
                raise EvolutionConflictError(
                    "Optimization application was not found"
                )
            current = OptimizationApplication.model_validate_json(
                row["application_json"]
            )
            if current.status is OptimizationApplicationStatus.ROLLED_BACK:
                return current
            candidate = current.deployment.candidate
            active = cursor.execute(
                "SELECT version FROM runtime_configuration_active "
                "WHERE component = ?",
                (candidate.component,),
            ).fetchone()
            if active is None or int(active["version"]) != candidate.version:
                raise EvolutionConflictError(
                    "candidate is no longer the active version"
                )
            baseline = current.deployment.baseline
            cursor.execute(
                "UPDATE runtime_configuration_active SET version = ? "
                "WHERE component = ?",
                (baseline.version, baseline.component),
            )
            rolled_back = current.model_copy(
                update={
                    "status": OptimizationApplicationStatus.ROLLED_BACK,
                    "revision": current.revision + 1,
                    "updated_at": utc_now(),
                }
            )
            payload = _json_text(rolled_back)
            cursor.execute(
                "UPDATE optimization_application_current "
                "SET revision = ?, application_json = ? "
                "WHERE application_id = ? AND revision = ?",
                (
                    rolled_back.revision,
                    payload,
                    str(application_id),
                    current.revision,
                ),
            )
            cursor.execute(
                "INSERT INTO optimization_application_history "
                "(application_id, revision, application_json) VALUES (?, ?, ?)",
                (str(application_id), rolled_back.revision, payload),
            )
        return rolled_back

    async def load_application(
        self,
        application_id: UUID,
    ) -> OptimizationApplication | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT application_json FROM optimization_application_current "
                "WHERE application_id = ?",
                (str(application_id),),
            ).fetchone()
        if row is None:
            return None
        return OptimizationApplication.model_validate_json(
            row["application_json"]
        )

    async def application_history(
        self,
        application_id: UUID,
    ) -> tuple[OptimizationApplication, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT application_json FROM optimization_application_history "
                "WHERE application_id = ? ORDER BY revision",
                (str(application_id),),
            ).fetchall()
        return tuple(
            OptimizationApplication.model_validate_json(row["application_json"])
            for row in rows
        )

    async def save(self, case: ReplayCase) -> None:
        payload = _json_text(case)
        with self._database.transaction() as cursor:
            row = cursor.execute(
                "SELECT case_json FROM replay_cases WHERE case_id = ?",
                (str(case.case_id),),
            ).fetchone()
            if row is not None:
                if row["case_json"] == payload:
                    return
                raise EvolutionConflictError("Replay case id was reused")
            cursor.execute(
                "INSERT INTO replay_cases(case_id, case_json) VALUES (?, ?)",
                (str(case.case_id), payload),
            )

    async def load(self, case_id: UUID) -> ReplayCase | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT case_json FROM replay_cases WHERE case_id = ?",
                (str(case_id),),
            ).fetchone()
        if row is None:
            return None
        return ReplayCase.model_validate_json(row["case_json"])

    async def list_all(self) -> tuple[ReplayCase, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT case_json FROM replay_cases ORDER BY case_id"
            ).fetchall()
        return tuple(ReplayCase.model_validate_json(row["case_json"]) for row in rows)

