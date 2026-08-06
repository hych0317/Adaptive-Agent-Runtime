"""Append-only SQLite persistence for deterministic Evaluation reports."""

from __future__ import annotations

from uuid import UUID

from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.evaluation import EvaluationReport
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


class SQLiteEvaluationReportStore:
    module_id = "evaluation.report_store.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def save(self, report: EvaluationReport) -> EvaluationReport:
        fingerprint = decision_fingerprint(report)
        with self._database.transaction() as cursor:
            prior = cursor.execute(
                "SELECT report_fingerprint, report_json FROM evaluation_reports "
                "WHERE report_id = ?",
                (str(report.report_id),),
            ).fetchone()
            if prior is not None:
                if prior["report_fingerprint"] != fingerprint:
                    raise PersistenceConflictError(
                        "Evaluation report identity conflicts with stored payload"
                    )
                return EvaluationReport.model_validate_json(prior["report_json"])
            duplicate = cursor.execute(
                "SELECT report_fingerprint, report_json FROM evaluation_reports "
                "WHERE run_id = ?",
                (str(report.run_id),),
            ).fetchone()
            if duplicate is not None:
                if duplicate["report_fingerprint"] != fingerprint:
                    raise PersistenceConflictError(
                        "Run already has a different Evaluation report"
                    )
                return EvaluationReport.model_validate_json(duplicate["report_json"])
            cursor.execute(
                "INSERT INTO evaluation_reports "
                "(report_id, run_id, task_id, report_fingerprint, report_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(report.report_id),
                    str(report.run_id),
                    str(report.task_id),
                    fingerprint,
                    report.model_dump_json(),
                ),
            )
        return report

    async def load(self, report_id: UUID) -> EvaluationReport | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT report_fingerprint, report_json FROM evaluation_reports "
                "WHERE report_id = ?",
                (str(report_id),),
            ).fetchone()
        if row is None:
            return None
        report = EvaluationReport.model_validate_json(row["report_json"])
        if row["report_fingerprint"] != decision_fingerprint(report):
            raise PersistenceConflictError("Evaluation report read-back is corrupted")
        return report

    async def load_for_run(self, run_id: UUID) -> EvaluationReport | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT report_fingerprint, report_json FROM evaluation_reports "
                "WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        report = EvaluationReport.model_validate_json(row["report_json"])
        if row["report_fingerprint"] != decision_fingerprint(report):
            raise PersistenceConflictError("Evaluation report read-back is corrupted")
        return report
