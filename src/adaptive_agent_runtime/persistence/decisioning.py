"""Normalized durable storage for typed Decision recovery and proof queries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Generic, cast
from uuid import UUID

from adaptive_agent_runtime.decisioning.errors import (
    DecisionCheckpointConflictError,
)
from adaptive_agent_runtime.decisioning.models import (
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionModel,
    DecisionResultStatus,
    DecisionTransition,
    EffectPayloadT,
    NormalizedDecisionEffect,
    ProposalPayloadT,
    RequestPayloadT,
    decision_fingerprint,
)
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase


_CODEC_VERSION = "1"
_STORAGE_SCHEMA_VERSION = "1"
_EVIDENCE_REF_KEY = "$decision_evidence_ref"


class DecisionProof(DecisionModel):
    """Small verified Decision projection for cross-domain provenance checks."""

    request_id: UUID
    run_id: UUID
    task_id: UUID | None = None
    decision_type: str
    target_id: str
    revision: int
    stage: DecisionCheckpointStage
    request_fingerprint: str
    proposal_id: UUID | None = None
    proposal_fingerprint: str | None = None
    validation_id: UUID | None = None
    validation_fingerprint: str | None = None
    effect_fingerprint: str | None = None
    governance_decision_id: UUID | None = None
    authorization_id: UUID | None = None
    review_request_id: UUID | None = None
    commit_receipt_id: str | None = None
    result_id: UUID | None = None
    result_status: DecisionResultStatus | None = None
    evidence_ids: tuple[str, ...] = ()

    @property
    def is_applied(self) -> bool:
        return (
            self.stage is DecisionCheckpointStage.COMPLETED
            and self.result_status is DecisionResultStatus.APPLIED
            and self.effect_fingerprint is not None
            and self.commit_receipt_id is not None
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _storage_fingerprint(
    object_type: str,
    codec_version: str,
    schema_version: str,
    canonical_payload: str,
) -> str:
    domain = f"{object_type}:{codec_version}:{schema_version}\0".encode("utf-8")
    return hashlib.sha256(domain + canonical_payload.encode("utf-8")).hexdigest()


def _metadata(
    object_type: str,
    logical_fingerprint: str,
    payload: object,
    *,
    schema_version: str = _STORAGE_SCHEMA_VERSION,
) -> dict[str, object]:
    canonical = _canonical_json(payload)
    return {
        "object_type": object_type,
        "codec_version": _CODEC_VERSION,
        "schema_version": schema_version,
        "logical_fingerprint": logical_fingerprint,
        "storage_payload_fingerprint": _storage_fingerprint(
            object_type,
            _CODEC_VERSION,
            schema_version,
            canonical,
        ),
        "canonical_payload": canonical,
        "payload_size": len(canonical.encode("utf-8")),
    }


def _externalize_evidence(
    payload: dict[str, object],
    *,
    field_name: str,
) -> tuple[dict[str, object], dict[str, object] | None]:
    nested = payload.get("payload")
    if not isinstance(nested, dict) or field_name not in nested:
        return payload, None
    evidence = nested[field_name]
    if not isinstance(evidence, list):
        return payload, None
    evidence_metadata = _metadata(
        "decision-evidence",
        decision_fingerprint(evidence),
        evidence,
    )
    nested[field_name] = {
        _EVIDENCE_REF_KEY: evidence_metadata["storage_payload_fingerprint"]
    }
    return payload, evidence_metadata


def _verify_storage_row(row: Mapping[str, object], expected_type: str) -> None:
    if row["object_type"] != expected_type:
        raise PersistenceConflictError(
            f"stored Decision object type is not {expected_type}"
        )
    if row["codec_version"] != _CODEC_VERSION:
        raise PersistenceConflictError(
            f"unsupported Decision codec version {row['codec_version']}"
        )
    if row["schema_version"] != _STORAGE_SCHEMA_VERSION:
        raise PersistenceConflictError(
            f"unsupported Decision object schema {row['schema_version']}"
        )
    canonical = str(row["canonical_payload"])
    if int(str(row["payload_size"])) != len(canonical.encode("utf-8")):
        raise PersistenceConflictError("stored Decision payload size is invalid")
    expected = _storage_fingerprint(
        expected_type,
        str(row["codec_version"]),
        str(row["schema_version"]),
        canonical,
    )
    if row["storage_payload_fingerprint"] != expected:
        raise PersistenceConflictError("stored Decision payload fingerprint is invalid")


def _insert_or_verify(
    cursor: object,
    *,
    table: str,
    identity_column: str,
    identity: str,
    columns: tuple[str, ...],
    values: tuple[object, ...],
) -> None:
    row = cursor.execute(  # type: ignore[attr-defined]
        f"SELECT {', '.join(columns)} FROM {table} "
        f"WHERE {identity_column} = ?",
        (identity,),
    ).fetchone()
    if row is not None:
        if any(row[column] != value for column, value in zip(columns, values)):
            raise DecisionCheckpointConflictError(
                f"immutable Decision identity in {table} was reused"
            )
        return
    placeholders = ", ".join("?" for _ in columns)
    cursor.execute(  # type: ignore[attr-defined]
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )


def _insert_evidence(cursor: object, metadata: Mapping[str, object] | None) -> str | None:
    if metadata is None:
        return None
    columns = (
        "storage_payload_fingerprint",
        "object_type",
        "codec_version",
        "schema_version",
        "logical_fingerprint",
        "canonical_payload",
        "payload_size",
    )
    values = tuple(metadata[column] for column in columns)
    identity = str(metadata["storage_payload_fingerprint"])
    _insert_or_verify(
        cursor,
        table="decision_evidence_snapshots",
        identity_column="storage_payload_fingerprint",
        identity=identity,
        columns=columns,
        values=values,
    )
    return identity


def _object_columns(*prefix: str) -> tuple[str, ...]:
    return (
        *prefix,
        "object_type",
        "codec_version",
        "schema_version",
        "logical_fingerprint",
        "storage_payload_fingerprint",
        "canonical_payload",
        "payload_size",
    )


def _object_values(
    metadata: Mapping[str, object], *prefix: object
) -> tuple[object, ...]:
    return (
        *prefix,
        metadata["object_type"],
        metadata["codec_version"],
        metadata["schema_version"],
        metadata["logical_fingerprint"],
        metadata["storage_payload_fingerprint"],
        metadata["canonical_payload"],
        metadata["payload_size"],
    )


def _load_object_row(
    cursor: object,
    table: str,
    identity_column: str,
    identity: str,
    expected_type: str,
) -> Mapping[str, object]:
    row = cursor.execute(  # type: ignore[attr-defined]
        f"SELECT * FROM {table} WHERE {identity_column} = ?",
        (identity,),
    ).fetchone()
    if row is None:
        raise PersistenceConflictError(f"referenced {expected_type} is missing")
    _verify_storage_row(row, expected_type)
    return cast(Mapping[str, object], row)


def _load_evidence(cursor: object, reference: str) -> object:
    row = _load_object_row(
        cursor,
        "decision_evidence_snapshots",
        "storage_payload_fingerprint",
        reference,
        "decision-evidence",
    )
    payload = json.loads(str(row["canonical_payload"]))
    if decision_fingerprint(payload) != row["logical_fingerprint"]:
        raise PersistenceConflictError("Evidence logical fingerprint is invalid")
    return payload


def _hydrate_evidence(cursor: object, value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {_EVIDENCE_REF_KEY}:
            reference = value[_EVIDENCE_REF_KEY]
            if not isinstance(reference, str):
                raise PersistenceConflictError("Evidence reference is invalid")
            return _load_evidence(cursor, reference)
        return {key: _hydrate_evidence(cursor, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hydrate_evidence(cursor, item) for item in value]
    return value


def _load_payload(
    cursor: object,
    table: str,
    identity_column: str,
    identity: str,
    expected_type: str,
) -> tuple[dict[str, object], Mapping[str, object]]:
    row = _load_object_row(
        cursor, table, identity_column, identity, expected_type
    )
    value = json.loads(str(row["canonical_payload"]))
    hydrated = _hydrate_evidence(cursor, value)
    if not isinstance(hydrated, dict):
        raise PersistenceConflictError(f"stored {expected_type} payload is invalid")
    return hydrated, row


class SQLiteDecisionCheckpointStore(
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
):
    """Persist one current recovery state plus immutable referenced objects."""

    module_id = "decision.recovery.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        checkpoint_type: type[
            DecisionCheckpoint[
                RequestPayloadT,
                ProposalPayloadT,
                EffectPayloadT,
            ]
        ],
    ) -> None:
        self._database = database
        self._checkpoint_type = checkpoint_type

    async def save(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        *,
        expected_revision: int | None,
    ) -> None:
        with self._database.transaction() as cursor:
            refs = self._persist_objects(cursor, checkpoint)
            values = self._current_values(checkpoint, refs)
            current = cursor.execute(
                "SELECT * FROM decision_current WHERE request_id = ?",
                (str(checkpoint.request_id),),
            ).fetchone()
            if current is None:
                if expected_revision is not None or checkpoint.revision != 0:
                    raise DecisionCheckpointConflictError(
                        "a new Decision recovery state must begin at revision 0"
                    )
                columns = tuple(values)
                cursor.execute(
                    f"INSERT INTO decision_current ({', '.join(columns)}) VALUES "
                    f"({', '.join('?' for _ in columns)})",
                    tuple(values[column] for column in columns),
                )
                from_revision = -1
                from_stage = None
            else:
                current_revision = int(current["revision"])
                if current_revision == checkpoint.revision:
                    if all(current[column] == value for column, value in values.items()):
                        return
                    raise DecisionCheckpointConflictError(
                        "Decision recovery revision was reused with different content"
                    )
                if expected_revision is None or current_revision != expected_revision:
                    raise DecisionCheckpointConflictError(
                        "Decision recovery write is based on a stale revision"
                    )
                if checkpoint.revision != current_revision + 1:
                    raise DecisionCheckpointConflictError(
                        "Decision recovery write is not the next revision"
                    )
                assignments = ", ".join(
                    f"{column} = ?" for column in values if column != "request_id"
                )
                update_values = tuple(
                    value for column, value in values.items() if column != "request_id"
                )
                cursor.execute(
                    f"UPDATE decision_current SET {assignments} "
                    "WHERE request_id = ? AND revision = ?",
                    (*update_values, str(checkpoint.request_id), expected_revision),
                )
                if cursor.rowcount != 1:
                    raise DecisionCheckpointConflictError(
                        "Decision recovery CAS update failed"
                    )
                from_revision = current_revision
                from_stage = str(current["stage"])
            self._insert_transition(
                cursor,
                checkpoint,
                refs,
                from_revision=from_revision,
                from_stage=from_stage,
            )

    def _persist_objects(
        self,
        cursor: object,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
    ) -> dict[str, str | None]:
        request_payload = checkpoint.request.model_dump(mode="json")
        request_payload, request_evidence = _externalize_evidence(
            request_payload, field_name="candidates"
        )
        request_evidence_ref = _insert_evidence(cursor, request_evidence)
        request_meta = _metadata(
            "decision-request",
            decision_fingerprint(checkpoint.request),
            request_payload,
            schema_version=checkpoint.request.schema_version,
        )
        request_columns = _object_columns(
            "request_id",
            "run_id",
            "task_id",
            "decision_type",
            "target_id",
            "evidence_snapshot_ref",
        )
        request_values = _object_values(
            request_meta,
            str(checkpoint.request_id),
            str(checkpoint.run_id),
            (
                str(checkpoint.request.correlation.task_id)
                if checkpoint.request.correlation.task_id is not None
                else None
            ),
            checkpoint.request.decision_type,
            checkpoint.request.target.target_id,
            request_evidence_ref,
        )
        _insert_or_verify(
            cursor,
            table="decision_requests",
            identity_column="request_id",
            identity=str(checkpoint.request_id),
            columns=request_columns,
            values=request_values,
        )

        proposal_ref: str | None = None
        proposal_fingerprint: str | None = None
        if checkpoint.proposal is not None:
            proposal_ref = str(checkpoint.proposal.proposal_id)
            proposal_fingerprint = decision_fingerprint(checkpoint.proposal)
            proposal_meta = _metadata(
                "decision-proposal",
                proposal_fingerprint,
                checkpoint.proposal.model_dump(mode="json"),
                schema_version=checkpoint.proposal.schema_version,
            )
            columns = _object_columns("proposal_id", "request_id")
            values = _object_values(
                proposal_meta, proposal_ref, str(checkpoint.request_id)
            )
            _insert_or_verify(
                cursor,
                table="decision_proposals",
                identity_column="proposal_id",
                identity=proposal_ref,
                columns=columns,
                values=values,
            )

        effect_ref: str | None = None
        effect_evidence_ref: str | None = None
        validated = checkpoint.validated_decision
        if validated is not None:
            if proposal_ref is None:
                raise DecisionCheckpointConflictError(
                    "validated Decision has no Proposal reference"
                )
            effect = validated.normalized_effect
            effect_ref = effect.effect_fingerprint
            effect_payload = effect.model_dump(mode="json")
            effect_payload, effect_evidence = _externalize_evidence(
                effect_payload, field_name="evidence_snapshot"
            )
            effect_evidence_ref = _insert_evidence(cursor, effect_evidence)
            effect_meta = _metadata(
                "decision-effect", effect_ref, effect_payload
            )
            columns = _object_columns(
                "effect_fingerprint",
                "evidence_snapshot_ref",
            )
            values = _object_values(
                effect_meta,
                effect_ref,
                effect_evidence_ref,
            )
            _insert_or_verify(
                cursor,
                table="decision_effects",
                identity_column="effect_fingerprint",
                identity=effect_ref,
                columns=columns,
                values=values,
            )

        validation_ref: str | None = None
        validation_fingerprint: str | None = None
        if checkpoint.validation is not None:
            if proposal_ref is None:
                raise DecisionCheckpointConflictError(
                    "Decision Validation has no Proposal reference"
                )
            validation_ref = str(checkpoint.validation.validation_id)
            validation_fingerprint = decision_fingerprint(checkpoint.validation)
            validation_meta = _metadata(
                "decision-validation",
                validation_fingerprint,
                checkpoint.validation.model_dump(mode="json"),
            )
            columns = _object_columns(
                "validation_id",
                "request_id",
                "proposal_id",
                "effect_fingerprint",
                "validation_status",
            )
            values = _object_values(
                validation_meta,
                validation_ref,
                str(checkpoint.request_id),
                proposal_ref,
                effect_ref,
                checkpoint.validation.status.value,
            )
            _insert_or_verify(
                cursor,
                table="decision_validations",
                identity_column="validation_id",
                identity=validation_ref,
                columns=columns,
                values=values,
            )

        governance_ref: str | None = None
        if checkpoint.governance_receipt is not None:
            receipt = checkpoint.governance_receipt
            governance_ref = decision_fingerprint(receipt)
            governance_meta = _metadata(
                "decision-governance-receipt",
                governance_ref,
                receipt.model_dump(mode="json"),
            )
            columns = _object_columns(
                "governance_receipt_id",
                "governance_decision_id",
                "request_id",
                "authorization_id",
                "review_request_id",
            )
            values = _object_values(
                governance_meta,
                governance_ref,
                str(receipt.governance_decision_id),
                str(checkpoint.request_id),
                str(receipt.authorization_id) if receipt.authorization_id else None,
                str(receipt.review_request_id) if receipt.review_request_id else None,
            )
            _insert_or_verify(
                cursor,
                table="decision_governance_receipts",
                identity_column="governance_receipt_id",
                identity=governance_ref,
                columns=columns,
                values=values,
            )

        commit_ref: str | None = None
        if checkpoint.commit_receipt is not None:
            commit = checkpoint.commit_receipt
            commit_ref = commit.effect_fingerprint
            commit_meta = _metadata(
                "decision-commit-receipt",
                decision_fingerprint(commit),
                commit.model_dump(mode="json"),
            )
            columns = _object_columns(
                "commit_receipt_id",
                "effect_fingerprint",
                "authority_type",
                "authority_receipt_ref",
                "committed_state_fingerprint",
            )
            values = _object_values(
                commit_meta,
                commit_ref,
                commit.effect_fingerprint,
                "effect-fingerprint",
                commit.effect_fingerprint,
                commit.committed_state_fingerprint,
            )
            _insert_or_verify(
                cursor,
                table="decision_commit_receipts",
                identity_column="commit_receipt_id",
                identity=commit_ref,
                columns=columns,
                values=values,
            )

        result_ref: str | None = None
        result_fingerprint: str | None = None
        if checkpoint.result is not None:
            result = checkpoint.result
            result_ref = str(result.result_id)
            result_fingerprint = decision_fingerprint(result)
            result_payload = result.model_dump(mode="json")
            result_payload.pop("apply_receipt", None)
            result_meta = _metadata(
                "decision-result", result_fingerprint, result_payload
            )
            columns = _object_columns(
                "result_id",
                "request_id",
                "result_status",
                "reconciliation_status",
                "commit_receipt_ref",
            )
            values = _object_values(
                result_meta,
                result_ref,
                str(checkpoint.request_id),
                result.status.value,
                (
                    result.reconciliation_status.value
                    if result.reconciliation_status is not None
                    else None
                ),
                commit_ref,
            )
            _insert_or_verify(
                cursor,
                table="decision_results",
                identity_column="result_id",
                identity=result_ref,
                columns=columns,
                values=values,
            )

        return {
            "request_ref": str(checkpoint.request_id),
            "request_fingerprint": str(request_meta["logical_fingerprint"]),
            "proposal_ref": proposal_ref,
            "proposal_fingerprint": proposal_fingerprint,
            "validation_ref": validation_ref,
            "validation_fingerprint": validation_fingerprint,
            "effect_ref": effect_ref,
            "governance_receipt_ref": governance_ref,
            "commit_receipt_ref": commit_ref,
            "result_ref": result_ref,
            "result_fingerprint": result_fingerprint,
        }

    @staticmethod
    def _current_values(
        checkpoint: DecisionCheckpoint[Any, Any, Any],
        refs: Mapping[str, str | None],
    ) -> dict[str, object]:
        governance = checkpoint.governance_receipt
        return {
            "request_id": str(checkpoint.request_id),
            "run_id": str(checkpoint.run_id),
            "task_id": (
                str(checkpoint.request.correlation.task_id)
                if checkpoint.request.correlation.task_id is not None
                else None
            ),
            "decision_type": checkpoint.request.decision_type,
            "target_id": checkpoint.request.target.target_id,
            "revision": checkpoint.revision,
            "stage": checkpoint.stage.value,
            "request_ref": refs["request_ref"],
            "proposal_ref": refs["proposal_ref"],
            "validation_ref": refs["validation_ref"],
            "effect_ref": refs["effect_ref"],
            "governance_receipt_ref": refs["governance_receipt_ref"],
            "authorization_id": (
                str(governance.authorization_id)
                if governance is not None and governance.authorization_id is not None
                else None
            ),
            "review_request_id": (
                str(governance.review_request_id)
                if governance is not None and governance.review_request_id is not None
                else None
            ),
            "commit_receipt_ref": refs["commit_receipt_ref"],
            "result_ref": refs["result_ref"],
            "result_status": (
                checkpoint.result.status.value if checkpoint.result is not None else None
            ),
            "reconciliation_status": (
                checkpoint.result.reconciliation_status.value
                if checkpoint.result is not None
                and checkpoint.result.reconciliation_status is not None
                else None
            ),
            "budget_usage_json": _canonical_json(
                checkpoint.budget_usage.model_dump(mode="json")
            ),
            "context_manifest_json": (
                _canonical_json(checkpoint.context_manifest.model_dump(mode="json"))
                if checkpoint.context_manifest is not None
                else None
            ),
            "updated_at": checkpoint.updated_at.isoformat(),
        }

    @staticmethod
    def _insert_transition(
        cursor: object,
        checkpoint: DecisionCheckpoint[Any, Any, Any],
        refs: Mapping[str, str | None],
        *,
        from_revision: int,
        from_stage: str | None,
    ) -> None:
        governance = checkpoint.governance_receipt
        values = (
            str(checkpoint.request_id),
            from_revision,
            checkpoint.revision,
            from_stage,
            checkpoint.stage.value,
            checkpoint.updated_at.isoformat(),
            refs["request_ref"],
            refs["proposal_ref"],
            refs["validation_ref"],
            refs["effect_ref"],
            refs["governance_receipt_ref"],
            (
                str(governance.authorization_id)
                if governance is not None and governance.authorization_id is not None
                else None
            ),
            refs["commit_receipt_ref"],
            refs["result_ref"],
            checkpoint.result.status.value if checkpoint.result is not None else None,
        )
        cursor.execute(  # type: ignore[attr-defined]
            "INSERT INTO decision_transitions (request_id, from_revision, "
            "to_revision, from_stage, to_stage, occurred_at, request_ref, "
            "proposal_ref, validation_ref, effect_ref, governance_receipt_ref, "
            "authorization_id, commit_receipt_ref, result_ref, result_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

    async def load(
        self,
        request_id: UUID,
    ) -> (
        DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]
        | None
    ):
        with self._database.reader() as cursor:
            current = cursor.execute(
                "SELECT * FROM decision_current WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
            if current is None:
                return None
            return self._hydrate_checkpoint(cursor, current)

    def _hydrate_checkpoint(
        self, cursor: object, current: Mapping[str, object]
    ) -> DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]:
        request_payload, request_row = _load_payload(
            cursor,
            "decision_requests",
            "request_id",
            str(current["request_ref"]),
            "decision-request",
        )
        proposal_payload = None
        proposal_row = None
        if current["proposal_ref"] is not None:
            proposal_payload, proposal_row = _load_payload(
                cursor,
                "decision_proposals",
                "proposal_id",
                str(current["proposal_ref"]),
                "decision-proposal",
            )
        validation_payload = None
        validation_row = None
        if current["validation_ref"] is not None:
            validation_payload, validation_row = _load_payload(
                cursor,
                "decision_validations",
                "validation_id",
                str(current["validation_ref"]),
                "decision-validation",
            )
        effect_payload = None
        effect_row = None
        if current["effect_ref"] is not None:
            effect_payload, effect_row = _load_payload(
                cursor,
                "decision_effects",
                "effect_fingerprint",
                str(current["effect_ref"]),
                "decision-effect",
            )
        governance_payload = None
        governance_row = None
        if current["governance_receipt_ref"] is not None:
            governance_payload, governance_row = _load_payload(
                cursor,
                "decision_governance_receipts",
                "governance_receipt_id",
                str(current["governance_receipt_ref"]),
                "decision-governance-receipt",
            )
        commit_payload = None
        commit_row = None
        if current["commit_receipt_ref"] is not None:
            commit_payload, commit_row = _load_payload(
                cursor,
                "decision_commit_receipts",
                "commit_receipt_id",
                str(current["commit_receipt_ref"]),
                "decision-commit-receipt",
            )
        result_payload = None
        result_row = None
        if current["result_ref"] is not None:
            result_payload, result_row = _load_payload(
                cursor,
                "decision_results",
                "result_id",
                str(current["result_ref"]),
                "decision-result",
            )
            result_payload["apply_receipt"] = (
                commit_payload
                if current["result_status"] == DecisionResultStatus.APPLIED.value
                else None
            )

        validated_payload = None
        if effect_payload is not None:
            if proposal_payload is None or validation_payload is None:
                raise PersistenceConflictError(
                    "stored Effect has incomplete validated Decision references"
                )
            validated_payload = {
                "request": request_payload,
                "proposal": proposal_payload,
                "validation": validation_payload,
                "normalized_effect": effect_payload,
            }
        checkpoint = self._checkpoint_type.model_validate(
            {
                "request_id": current["request_id"],
                "run_id": current["run_id"],
                "revision": current["revision"],
                "stage": current["stage"],
                "request": request_payload,
                "budget_usage": json.loads(str(current["budget_usage_json"])),
                "context_manifest": (
                    json.loads(str(current["context_manifest_json"]))
                    if current["context_manifest_json"] is not None
                    else None
                ),
                "proposal": proposal_payload,
                "validation": validation_payload,
                "validated_decision": validated_payload,
                "governance_receipt": governance_payload,
                "commit_receipt": commit_payload,
                "result": result_payload,
                "updated_at": current["updated_at"],
            }
        )
        checks: tuple[tuple[object | None, Mapping[str, object] | None], ...] = (
            (checkpoint.request, request_row),
            (checkpoint.proposal, proposal_row),
            (checkpoint.validation, validation_row),
            (
                checkpoint.validated_decision.normalized_effect
                if checkpoint.validated_decision is not None
                else None,
                effect_row,
            ),
            (checkpoint.governance_receipt, governance_row),
            (checkpoint.commit_receipt, commit_row),
            (checkpoint.result, result_row),
        )
        for model, row in checks:
            if model is None or row is None:
                continue
            if row["object_type"] == "decision-effect":
                if not isinstance(model, NormalizedDecisionEffect):
                    raise PersistenceConflictError(
                        "stored Decision effect has an invalid hydrated type"
                    )
                logical = model.effect_fingerprint
            else:
                logical = decision_fingerprint(model)
            if logical != row["logical_fingerprint"]:
                raise PersistenceConflictError(
                    f"{row['object_type']} logical fingerprint is invalid"
                )
        return checkpoint

    async def transitions_for(
        self, request_id: UUID
    ) -> tuple[DecisionTransition, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT * FROM decision_transitions WHERE request_id = ? "
                "ORDER BY to_revision",
                (str(request_id),),
            ).fetchall()
        return tuple(_transition_from_row(row) for row in rows)


def _transition_from_row(row: Mapping[str, object]) -> DecisionTransition:
    return DecisionTransition.model_validate(dict(row))


class SQLiteDecisionRecordReader:
    """Verified query boundary for non-Lifecycle Decision consumers."""

    module_id = "decision.proof.sqlite"

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def load_decision_proof(self, request_id: UUID) -> DecisionProof | None:
        with self._database.reader() as cursor:
            return self.load_proof_in_transaction(cursor, request_id)

    @staticmethod
    def load_proof_in_transaction(
        cursor: object, request_id: UUID
    ) -> DecisionProof | None:
        row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT current.*, requests.logical_fingerprint AS request_fp, "
            "proposals.logical_fingerprint AS proposal_fp, "
            "validations.logical_fingerprint AS validation_fp, "
            "governance.governance_decision_id AS governance_decision_id "
            "FROM decision_current AS current "
            "JOIN decision_requests AS requests "
            "ON requests.request_id = current.request_ref "
            "LEFT JOIN decision_proposals AS proposals "
            "ON proposals.proposal_id = current.proposal_ref "
            "LEFT JOIN decision_validations AS validations "
            "ON validations.validation_id = current.validation_ref "
            "LEFT JOIN decision_governance_receipts AS governance "
            "ON governance.governance_receipt_id = current.governance_receipt_ref "
            "WHERE current.request_id = ?",
            (str(request_id),),
        ).fetchone()
        if row is None:
            return None
        request_row = _load_object_row(
            cursor,
            "decision_requests",
            "request_id",
            str(row["request_ref"]),
            "decision-request",
        )
        request_payload = json.loads(str(request_row["canonical_payload"]))
        raw_evidence = (
            request_payload.get("evidence", [])
            if isinstance(request_payload, dict)
            else []
        )
        evidence_ids = tuple(
            str(item["evidence_id"])
            for item in raw_evidence
            if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
        )
        for table, column, identity, object_type in (
            ("decision_proposals", "proposal_id", row["proposal_ref"], "decision-proposal"),
            ("decision_validations", "validation_id", row["validation_ref"], "decision-validation"),
            ("decision_effects", "effect_fingerprint", row["effect_ref"], "decision-effect"),
            (
                "decision_governance_receipts",
                "governance_receipt_id",
                row["governance_receipt_ref"],
                "decision-governance-receipt",
            ),
            (
                "decision_commit_receipts",
                "commit_receipt_id",
                row["commit_receipt_ref"],
                "decision-commit-receipt",
            ),
            ("decision_results", "result_id", row["result_ref"], "decision-result"),
        ):
            if identity is not None:
                _load_object_row(cursor, table, column, str(identity), object_type)
        return DecisionProof(
            request_id=row["request_id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            decision_type=row["decision_type"],
            target_id=row["target_id"],
            revision=row["revision"],
            stage=row["stage"],
            request_fingerprint=row["request_fp"],
            proposal_id=row["proposal_ref"],
            proposal_fingerprint=row["proposal_fp"],
            validation_id=row["validation_ref"],
            validation_fingerprint=row["validation_fp"],
            effect_fingerprint=row["effect_ref"],
            governance_decision_id=(
                row["governance_decision_id"]
                if row["governance_receipt_ref"] is not None
                else None
            ),
            authorization_id=row["authorization_id"],
            review_request_id=row["review_request_id"],
            commit_receipt_id=row["commit_receipt_ref"],
            result_id=row["result_ref"],
            result_status=row["result_status"],
            evidence_ids=evidence_ids,
        )

    def find_completed_decisions(
        self, run_id: UUID, decision_type: str
    ) -> tuple[DecisionProof, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT request_id FROM decision_current WHERE run_id = ? "
                "AND decision_type = ? AND stage = 'completed'",
                (str(run_id), decision_type),
            ).fetchall()
            proofs = tuple(
                self.load_proof_in_transaction(cursor, UUID(str(row["request_id"])))
                for row in rows
            )
        return tuple(proof for proof in proofs if proof is not None)

    def load_effect(self, effect_fingerprint: str) -> Mapping[str, object] | None:
        with self._database.reader() as cursor:
            return self.load_effect_in_transaction(cursor, effect_fingerprint)

    @staticmethod
    def load_effect_in_transaction(
        cursor: object, effect_fingerprint: str
    ) -> Mapping[str, object] | None:
        row = cursor.execute(  # type: ignore[attr-defined]
            "SELECT 1 FROM decision_effects WHERE effect_fingerprint = ?",
            (effect_fingerprint,),
        ).fetchone()
        if row is None:
            return None
        payload, stored = _load_payload(
            cursor,
            "decision_effects",
            "effect_fingerprint",
            effect_fingerprint,
            "decision-effect",
        )
        if payload.get("effect_fingerprint") != stored["logical_fingerprint"]:
            raise PersistenceConflictError("Decision Effect identity is invalid")
        return payload

    def transitions_for(self, request_id: UUID) -> tuple[DecisionTransition, ...]:
        with self._database.reader() as cursor:
            rows = cursor.execute(
                "SELECT * FROM decision_transitions WHERE request_id = ? "
                "ORDER BY to_revision",
                (str(request_id),),
            ).fetchall()
        return tuple(_transition_from_row(row) for row in rows)
