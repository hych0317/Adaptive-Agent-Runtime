"""Permit-bound, atomic Phase 4-B Runtime configuration persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance import (
    AuthorizationVerificationError,
    CommitPermitValidation,
    GovernanceTarget,
    RuntimeCommitPermit,
)
from adaptive_agent_runtime.optimization import (
    OPTIMIZATION_APPLY_COMMIT_OPERATION,
    OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
    OptimizationApplyCommitReceipt,
    OptimizationApplyEffect,
    OptimizationProposal,
    OptimizationProposalCommitReceipt,
    OptimizationProposalStatus,
    OptimizationRiskClassification,
    OptimizationRollbackEffect,
    OptimizationScope,
    OptimizationTargetKey,
    RuntimeConfigurationSnapshot,
    RuntimeConfigurationActivationMode,
    AutoAdaptationPolicy,
    AutoAdaptationStatus,
    AutoAdaptationTriggerRecord,
    create_runtime_configuration_snapshot,
    runtime_configuration_target_id,
)
from adaptive_agent_runtime.persistence.errors import PersistenceConflictError
from adaptive_agent_runtime.persistence.sqlite import SQLiteDatabase
from adaptive_agent_runtime.persistence.optimization import (
    verify_optimization_proposal_phase3_in_transaction,
)


_DEFAULT_MAX_NODES = 8
_DEFAULT_CONFIGURATION_SOURCE = "research.planning.default"


class SQLiteGovernedRuntimeConfigurationStore:
    """Only Permit-bearing Effects can change the active configuration pointer."""

    module_id = "optimization.runtime_configuration.sqlite"

    def __init__(
        self,
        database: SQLiteDatabase,
        *,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._database = database
        self._permit_verifier = permit_verifier

    async def load_active(
        self,
        scope: OptimizationScope,
        target_key: OptimizationTargetKey,
    ) -> RuntimeConfigurationSnapshot | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshots.snapshot_json, "
                "active.snapshot_fingerprint AS active_fingerprint "
                "FROM governed_runtime_configuration_active AS active "
                "JOIN governed_runtime_configuration_snapshots AS snapshots "
                "ON snapshots.tenant_id = active.tenant_id "
                "AND snapshots.project_id = active.project_id "
                "AND snapshots.application_id = active.application_id "
                "AND snapshots.decision_type = active.decision_type "
                "AND snapshots.target_key = active.target_key "
                "AND snapshots.revision = active.revision "
                "WHERE active.tenant_id = ? AND active.project_id = ? "
                "AND active.application_id = ? AND active.decision_type = ? "
                "AND active.target_key = ?",
                (*_scope_values(scope), target_key.value),
            ).fetchone()
        return _snapshot_from_row(row)

    async def load_snapshot(
        self,
        scope: OptimizationScope,
        target_key: OptimizationTargetKey,
        revision: int,
    ) -> RuntimeConfigurationSnapshot | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT snapshot_json FROM governed_runtime_configuration_snapshots "
                "WHERE tenant_id = ? AND project_id = ? AND application_id = ? "
                "AND decision_type = ? AND target_key = ? AND revision = ?",
                (*_scope_values(scope), target_key.value, revision),
            ).fetchone()
        return _snapshot_from_row(row)

    async def load_receipt(
        self,
        effect_fingerprint: str,
    ) -> OptimizationApplyCommitReceipt | None:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT operation, payload_fingerprint, receipt_json "
                "FROM optimization_configuration_receipts "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        receipt = OptimizationApplyCommitReceipt.model_validate_json(
            row["receipt_json"]
        )
        if (
            receipt.effect_fingerprint != effect_fingerprint
            or receipt.operation != row["operation"]
            or receipt.payload_fingerprint != row["payload_fingerprint"]
        ):
            raise PersistenceConflictError(
                "Optimization configuration Receipt is corrupted"
            )
        return receipt

    async def commit_apply(
        self,
        effect: OptimizationApplyEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> OptimizationApplyCommitReceipt:
        payload_fingerprint = decision_fingerprint(effect)
        prior = await self._idempotent_receipt(
            effect_fingerprint,
            payload_fingerprint,
            OPTIMIZATION_APPLY_COMMIT_OPERATION,
        )
        if prior is not None:
            return prior
        await self._verify_permit(
            effect=effect,
            operation=OPTIMIZATION_APPLY_COMMIT_OPERATION,
            permit=permit,
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        _require_phase_4b_scope(effect.scope, effect.target_key)
        if effect.rollback_revision != effect.expected_current_revision:
            raise PersistenceConflictError("Apply rollback revision is inconsistent")
        if (
            effect.expected_current_value_fingerprint
            != decision_fingerprint(effect.expected_current_value)
        ):
            raise PersistenceConflictError("Apply baseline value fingerprint is invalid")

        with self._database.transaction() as cursor:
            prior = _load_receipt_in_transaction(cursor, effect_fingerprint)
            if prior is not None:
                _verify_receipt_payload(
                    prior,
                    payload_fingerprint,
                    OPTIMIZATION_APPLY_COMMIT_OPERATION,
                )
                return prior
            proposal = _load_and_verify_proposal(cursor, effect)
            if effect.trigger_mode is RuntimeConfigurationActivationMode.POLICY_AUTO:
                _verify_policy_auto_trigger(cursor, effect, proposal)
            current = _load_active_snapshot(cursor, effect.scope, effect.target_key)
            if current is None:
                current = _default_snapshot(effect.scope)
                if effect.expected_current_revision != 0:
                    raise PersistenceConflictError("Active configuration revision is stale")
            _verify_expected_current(
                current,
                revision=effect.expected_current_revision,
                value=effect.expected_current_value,
                snapshot_fingerprint=(
                    effect.expected_current_configuration_fingerprint
                ),
            )
            if (
                proposal.current_configuration_revision != current.revision
                or proposal.current_configuration_fingerprint
                != current.snapshot_fingerprint
                or proposal.current_value != current.value
                or proposal.current_value_fingerprint != current.value_fingerprint
            ):
                raise PersistenceConflictError(
                    "Optimization Proposal baseline no longer matches active configuration"
                )
            _insert_snapshot(cursor, current)
            activated = create_runtime_configuration_snapshot(
                scope=effect.scope,
                target_key=effect.target_key,
                revision=current.revision + 1,
                value=effect.proposed_value,
                previous_revision=current.revision,
                source_revision=None,
                source_proposal_id=effect.proposal_id,
                source_effect_fingerprint=effect_fingerprint,
                configuration_source=(
                    "governed.optimization.policy_auto"
                    if effect.trigger_mode
                    is RuntimeConfigurationActivationMode.POLICY_AUTO
                    else "governed.optimization.manual_apply"
                ),
                activation_mode=effect.trigger_mode,
                trigger_run_id=effect.trigger_run_id,
                auto_adaptation_policy_fingerprint=(
                    effect.auto_adaptation_policy_fingerprint
                ),
            )
            _insert_snapshot(cursor, activated)
            _cas_active_pointer(
                cursor,
                current=current,
                activated=activated,
                allow_missing=current.revision == 0,
            )
            receipt = OptimizationApplyCommitReceipt(
                effect_fingerprint=effect_fingerprint,
                operation=OPTIMIZATION_APPLY_COMMIT_OPERATION,
                scope=effect.scope,
                target_key=effect.target_key,
                active_revision=activated.revision,
                active_snapshot_id=activated.snapshot_id,
                active_snapshot_fingerprint=activated.snapshot_fingerprint,
                previous_revision=current.revision,
                payload_fingerprint=payload_fingerprint,
                activation_mode=effect.trigger_mode,
                trigger_run_id=effect.trigger_run_id,
                source_proposal_id=effect.proposal_id,
                auto_adaptation_policy_fingerprint=(
                    effect.auto_adaptation_policy_fingerprint
                ),
            )
            _insert_receipt(cursor, receipt)
        return receipt

    async def commit_rollback(
        self,
        effect: OptimizationRollbackEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> OptimizationApplyCommitReceipt:
        payload_fingerprint = decision_fingerprint(effect)
        prior = await self._idempotent_receipt(
            effect_fingerprint,
            payload_fingerprint,
            OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
        )
        if prior is not None:
            return prior
        await self._verify_permit(
            effect=effect,
            operation=OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
            permit=permit,
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        _require_phase_4b_scope(effect.scope, effect.target_key)

        with self._database.transaction() as cursor:
            prior = _load_receipt_in_transaction(cursor, effect_fingerprint)
            if prior is not None:
                _verify_receipt_payload(
                    prior,
                    payload_fingerprint,
                    OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
                )
                return prior
            source_receipt = _load_receipt_in_transaction(
                cursor, effect.source_apply_effect_fingerprint
            )
            if (
                source_receipt is None
                or source_receipt.operation != OPTIMIZATION_APPLY_COMMIT_OPERATION
            ):
                raise PersistenceConflictError("Rollback source Apply was not committed")
            current = _load_active_snapshot(cursor, effect.scope, effect.target_key)
            if current is None:
                raise PersistenceConflictError("Rollback requires an active configuration")
            _verify_expected_current(
                current,
                revision=effect.expected_current_revision,
                value=effect.expected_current_value,
                snapshot_fingerprint=(
                    effect.expected_current_configuration_fingerprint
                ),
            )
            if current.source_effect_fingerprint != effect.source_apply_effect_fingerprint:
                raise PersistenceConflictError(
                    "Rollback source is not the currently active Apply"
                )
            source = _load_snapshot_in_transaction(
                cursor,
                effect.scope,
                effect.target_key,
                effect.restore_source_revision,
            )
            if (
                source is None
                or source.snapshot_fingerprint != effect.restore_source_fingerprint
                or source.value != effect.restore_value
            ):
                raise PersistenceConflictError("Rollback source snapshot changed")
            activated = create_runtime_configuration_snapshot(
                scope=effect.scope,
                target_key=effect.target_key,
                revision=current.revision + 1,
                value=source.value,
                previous_revision=current.revision,
                source_revision=source.revision,
                source_proposal_id=None,
                source_effect_fingerprint=effect_fingerprint,
                configuration_source="governed.optimization.rollback",
                activation_mode=RuntimeConfigurationActivationMode.ROLLBACK,
            )
            _insert_snapshot(cursor, activated)
            _cas_active_pointer(
                cursor,
                current=current,
                activated=activated,
                allow_missing=False,
            )
            receipt = OptimizationApplyCommitReceipt(
                effect_fingerprint=effect_fingerprint,
                operation=OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
                scope=effect.scope,
                target_key=effect.target_key,
                active_revision=activated.revision,
                active_snapshot_id=activated.snapshot_id,
                active_snapshot_fingerprint=activated.snapshot_fingerprint,
                previous_revision=current.revision,
                payload_fingerprint=payload_fingerprint,
                activation_mode=RuntimeConfigurationActivationMode.ROLLBACK,
            )
            _insert_receipt(cursor, receipt)
        return receipt

    async def _idempotent_receipt(
        self,
        effect_fingerprint: str,
        payload_fingerprint: str,
        operation: str,
    ) -> OptimizationApplyCommitReceipt | None:
        receipt = await self.load_receipt(effect_fingerprint)
        if receipt is not None:
            _verify_receipt_payload(receipt, payload_fingerprint, operation)
        return receipt

    async def was_proposal_applied(self, proposal_id: UUID) -> bool:
        with self._database.reader() as cursor:
            row = cursor.execute(
                "SELECT 1 FROM governed_runtime_configuration_snapshots "
                "WHERE snapshot_json LIKE ? LIMIT 1",
                (f'%"source_proposal_id":"{proposal_id}"%',),
            ).fetchone()
        return row is not None

    async def _verify_permit(
        self,
        *,
        effect: OptimizationApplyEffect | OptimizationRollbackEffect,
        operation: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> None:
        if permit is None or target is None or subject_fingerprint is None:
            raise AuthorizationVerificationError(
                "Runtime configuration commit requires a Runtime Permit"
            )
        expected_target = GovernanceTarget(
            target_type="runtime_configuration",
            target_id=runtime_configuration_target_id(
                effect.scope, effect.target_key
            ),
        )
        if target != expected_target:
            raise AuthorizationVerificationError(
                "Runtime configuration Permit targets another authority object"
            )
        await self._permit_verifier.verify(
            permit,
            operation=operation,
            target=target,
            subject_fingerprint=subject_fingerprint,
        )


def default_runtime_configuration_snapshot(
    scope: OptimizationScope,
) -> RuntimeConfigurationSnapshot:
    _require_phase_4b_scope(scope, OptimizationTargetKey.PLANNER_MAX_NODES)
    return _default_snapshot(scope)


def _default_snapshot(scope: OptimizationScope) -> RuntimeConfigurationSnapshot:
    return create_runtime_configuration_snapshot(
        scope=scope,
        target_key=OptimizationTargetKey.PLANNER_MAX_NODES,
        revision=0,
        value=_DEFAULT_MAX_NODES,
        previous_revision=None,
        source_revision=None,
        source_proposal_id=None,
        source_effect_fingerprint=None,
        configuration_source=_DEFAULT_CONFIGURATION_SOURCE,
        created_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
    )


def _scope_values(scope: OptimizationScope) -> tuple[str, str, str, str]:
    return (scope.tenant, scope.project, scope.application, scope.decision_type)


def _require_phase_4b_scope(
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
) -> None:
    if _scope_values(scope) != (
        "default",
        "research",
        "research_agent",
        "planning.task_graph.initialize",
    ):
        raise PersistenceConflictError("Runtime configuration scope is not enabled")
    if target_key is not OptimizationTargetKey.PLANNER_MAX_NODES:
        raise PersistenceConflictError("Only planner.max_nodes is enabled in Phase 4-B")


def _snapshot_from_row(row: object | None) -> RuntimeConfigurationSnapshot | None:
    if row is None:
        return None
    snapshot = RuntimeConfigurationSnapshot.model_validate_json(
        row["snapshot_json"]  # type: ignore[index]
    )
    keys = row.keys()  # type: ignore[attr-defined]
    if (
        "active_fingerprint" in keys
        and row["active_fingerprint"] != snapshot.snapshot_fingerprint  # type: ignore[index]
    ):
        raise PersistenceConflictError(
            "Active configuration pointer fingerprint is corrupted"
        )
    return snapshot


def _load_active_snapshot(
    cursor: object,
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
) -> RuntimeConfigurationSnapshot | None:
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT snapshots.snapshot_json, "
        "active.snapshot_fingerprint AS active_fingerprint "
        "FROM governed_runtime_configuration_active AS active "
        "JOIN governed_runtime_configuration_snapshots AS snapshots "
        "ON snapshots.tenant_id = active.tenant_id "
        "AND snapshots.project_id = active.project_id "
        "AND snapshots.application_id = active.application_id "
        "AND snapshots.decision_type = active.decision_type "
        "AND snapshots.target_key = active.target_key "
        "AND snapshots.revision = active.revision "
        "WHERE active.tenant_id = ? AND active.project_id = ? "
        "AND active.application_id = ? AND active.decision_type = ? "
        "AND active.target_key = ?",
        (*_scope_values(scope), target_key.value),
    ).fetchone()
    return _snapshot_from_row(row)


def _load_snapshot_in_transaction(
    cursor: object,
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
    revision: int,
) -> RuntimeConfigurationSnapshot | None:
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT snapshot_json FROM governed_runtime_configuration_snapshots "
        "WHERE tenant_id = ? AND project_id = ? AND application_id = ? "
        "AND decision_type = ? AND target_key = ? AND revision = ?",
        (*_scope_values(scope), target_key.value, revision),
    ).fetchone()
    return _snapshot_from_row(row)


def _insert_snapshot(cursor: object, snapshot: RuntimeConfigurationSnapshot) -> None:
    existing = _load_snapshot_in_transaction(
        cursor, snapshot.scope, snapshot.target_key, snapshot.revision
    )
    if existing is not None:
        if existing != snapshot:
            raise PersistenceConflictError(
                "Runtime configuration revision conflicts with stored snapshot"
            )
        return
    cursor.execute(  # type: ignore[attr-defined]
        "INSERT INTO governed_runtime_configuration_snapshots "
        "(tenant_id, project_id, application_id, decision_type, target_key, "
        "revision, snapshot_fingerprint, snapshot_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            *_scope_values(snapshot.scope),
            snapshot.target_key.value,
            snapshot.revision,
            snapshot.snapshot_fingerprint,
            snapshot.model_dump_json(),
        ),
    )


def _cas_active_pointer(
    cursor: object,
    *,
    current: RuntimeConfigurationSnapshot,
    activated: RuntimeConfigurationSnapshot,
    allow_missing: bool,
) -> None:
    if allow_missing:
        inserted = cursor.execute(  # type: ignore[attr-defined]
            "INSERT OR IGNORE INTO governed_runtime_configuration_active "
            "(tenant_id, project_id, application_id, decision_type, target_key, "
            "revision, snapshot_fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                *_scope_values(activated.scope),
                activated.target_key.value,
                activated.revision,
                activated.snapshot_fingerprint,
            ),
        ).rowcount
        if inserted == 1:
            return
    updated = cursor.execute(  # type: ignore[attr-defined]
        "UPDATE governed_runtime_configuration_active "
        "SET revision = ?, snapshot_fingerprint = ? "
        "WHERE tenant_id = ? AND project_id = ? AND application_id = ? "
        "AND decision_type = ? AND target_key = ? AND revision = ? "
        "AND snapshot_fingerprint = ?",
        (
            activated.revision,
            activated.snapshot_fingerprint,
            *_scope_values(current.scope),
            current.target_key.value,
            current.revision,
            current.snapshot_fingerprint,
        ),
    ).rowcount
    if updated != 1:
        raise PersistenceConflictError("Active configuration CAS failed")


def _verify_expected_current(
    current: RuntimeConfigurationSnapshot,
    *,
    revision: int,
    value: object,
    snapshot_fingerprint: str,
) -> None:
    if (
        current.revision != revision
        or current.value != value
        or current.snapshot_fingerprint != snapshot_fingerprint
    ):
        raise PersistenceConflictError("Active configuration baseline is stale")


def _verify_policy_auto_trigger(
    cursor: object,
    effect: OptimizationApplyEffect,
    proposal: OptimizationProposal,
) -> None:
    if (
        effect.trigger_run_id is None
        or effect.auto_adaptation_policy_fingerprint is None
        or effect.auto_adaptation_trigger_id is None
        or effect.candidate_set_fingerprint is None
        or effect.selected_proposal_id != effect.proposal_id
    ):
        raise PersistenceConflictError("Policy-auto Effect provenance is incomplete")
    policy = AutoAdaptationPolicy(enabled=True, scope=effect.scope)
    if effect.auto_adaptation_policy_fingerprint != policy.fingerprint:
        raise PersistenceConflictError("Policy-auto Effect uses an unknown policy")
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT trigger_json FROM auto_adaptation_triggers "
        "WHERE trigger_run_id = ?",
        (str(effect.trigger_run_id),),
    ).fetchone()
    if row is None:
        raise PersistenceConflictError("Policy-auto Apply has no durable trigger claim")
    record = AutoAdaptationTriggerRecord.model_validate_json(row["trigger_json"])
    from adaptive_agent_runtime.optimization import (
        stable_auto_adaptation_trigger_identity,
        stable_optimization_apply_request_id,
    )

    expected_identity = stable_auto_adaptation_trigger_identity(
        effect.trigger_run_id,
        effect.proposal_id,
        effect.expected_current_revision,
    )
    expected_request_id = stable_optimization_apply_request_id(
        effect.proposal_id,
        trigger_run_id=effect.trigger_run_id,
        baseline_revision=effect.expected_current_revision,
    )
    if (
        record.status
        not in {
            AutoAdaptationStatus.SELECTED,
            AutoAdaptationStatus.REVIEW_PENDING,
        }
        or record.selected_proposal_id != effect.proposal_id
        or record.trigger_identity != expected_identity
        or effect.auto_adaptation_trigger_id != expected_identity
        or record.apply_request_id != expected_request_id
        or record.policy_fingerprint != effect.auto_adaptation_policy_fingerprint
        or record.baseline_revision != effect.expected_current_revision
        or record.baseline_fingerprint
        != effect.expected_current_configuration_fingerprint
        or record.candidate_set_fingerprint != effect.candidate_set_fingerprint
    ):
        raise PersistenceConflictError("Policy-auto trigger claim is inconsistent")
    authoritative = _authoritative_policy_auto_candidates(cursor, effect)
    if (
        len(authoritative) != 1
        or authoritative[0].proposal_id != effect.proposal_id
    ):
        raise PersistenceConflictError(
            "Policy-auto Apply no longer has exactly one authoritative candidate"
        )
    if (
        proposal.risk_classification is not OptimizationRiskClassification.LOW
        or proposal.counterevidence_refs
    ):
        raise PersistenceConflictError("Policy-auto Proposal is not low-risk and clean")
    current = effect.expected_current_value
    proposed = effect.proposed_value
    if (
        isinstance(current, bool)
        or isinstance(proposed, bool)
        or not isinstance(current, int)
        or not isinstance(proposed, int)
        or not 8 <= proposed <= 32
        or abs(proposed - current) != 1
    ):
        raise PersistenceConflictError("Policy-auto change violates the fixed delta")
    state_row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT snapshots.snapshot_json FROM agent_state_current AS current "
        "JOIN agent_state_snapshots AS snapshots ON snapshots.run_id = current.run_id "
        "AND snapshots.revision = current.revision WHERE current.run_id = ?",
        (str(effect.trigger_run_id),),
    ).fetchone()
    if state_row is None:
        raise PersistenceConflictError("Policy-auto trigger Run has no durable State")
    import json

    state = json.loads(state_row["snapshot_json"])
    if state.get("status") != "completed":
        raise PersistenceConflictError("Policy-auto trigger Run is not completed")


def _authoritative_policy_auto_candidates(
    cursor: object,
    effect: OptimizationApplyEffect,
) -> tuple[OptimizationProposal, ...]:
    rows = cursor.execute(  # type: ignore[attr-defined]
        "SELECT proposals.proposal_json, receipts.receipt_json "
        "FROM optimization_proposals AS proposals "
        "JOIN optimization_proposal_receipts AS receipts "
        "ON receipts.proposal_id = proposals.proposal_id "
        "WHERE proposals.tenant_id = ? AND proposals.project_id = ? "
        "AND proposals.application_id = ? AND proposals.decision_type = ?",
        _scope_values(effect.scope),
    ).fetchall()
    proposals: list[OptimizationProposal] = []
    for row in rows:
        candidate = OptimizationProposal.model_validate_json(row["proposal_json"])
        receipt = OptimizationProposalCommitReceipt.model_validate_json(
            row["receipt_json"]
        )
        if (
            receipt.proposal_id != candidate.proposal_id
            or receipt.effect_fingerprint != candidate.effect_fingerprint
            or receipt.proposal_fingerprint != decision_fingerprint(candidate)
        ):
            raise PersistenceConflictError(
                "Policy-auto candidate Proposal receipt is invalid"
            )
        verify_optimization_proposal_phase3_in_transaction(cursor, candidate)
        proposals.append(candidate)
    expected_set_fingerprint = decision_fingerprint(
        tuple(
            sorted(
                (str(item.proposal_id), decision_fingerprint(item))
                for item in proposals
            )
        )
    )
    if expected_set_fingerprint != effect.candidate_set_fingerprint:
        raise PersistenceConflictError("Policy-auto candidate set changed")

    eligible: list[OptimizationProposal] = []
    now = datetime.now(timezone.utc)
    for candidate in proposals:
        already_applied = cursor.execute(  # type: ignore[attr-defined]
            "SELECT 1 FROM governed_runtime_configuration_snapshots "
            "WHERE snapshot_json LIKE ? LIMIT 1",
            (f'%"source_proposal_id":"{candidate.proposal_id}"%',),
        ).fetchone()
        current = effect.expected_current_value
        proposed = candidate.proposed_value
        if (
            candidate.scope == effect.scope
            and candidate.target_key is effect.target_key
            and candidate.status is OptimizationProposalStatus.ACTIVE
            and (candidate.expires_at is None or candidate.expires_at > now)
            and candidate.current_configuration_revision
            == effect.expected_current_revision
            and candidate.current_configuration_fingerprint
            == effect.expected_current_configuration_fingerprint
            and candidate.current_value == current
            and candidate.risk_classification
            is OptimizationRiskClassification.LOW
            and not candidate.counterevidence_refs
            and already_applied is None
            and not isinstance(current, bool)
            and not isinstance(proposed, bool)
            and isinstance(current, int)
            and isinstance(proposed, int)
            and 8 <= proposed <= 32
            and abs(proposed - current) == 1
        ):
            eligible.append(candidate)
    return tuple(eligible)


def _load_and_verify_proposal(
    cursor: object,
    effect: OptimizationApplyEffect,
) -> OptimizationProposal:
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT proposal_json FROM optimization_proposals WHERE proposal_id = ?",
        (str(effect.proposal_id),),
    ).fetchone()
    receipt_row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT receipt_json FROM optimization_proposal_receipts "
        "WHERE proposal_id = ?",
        (str(effect.proposal_id),),
    ).fetchone()
    if row is None or receipt_row is None:
        raise PersistenceConflictError("Optimization Proposal is not committed")
    proposal = OptimizationProposal.model_validate_json(row["proposal_json"])
    from adaptive_agent_runtime.optimization import OptimizationProposalCommitReceipt

    receipt = OptimizationProposalCommitReceipt.model_validate_json(
        receipt_row["receipt_json"]
    )
    if (
        proposal.effect_fingerprint != effect.proposal_effect_fingerprint
        or decision_fingerprint(proposal) != effect.proposal_record_fingerprint
        or receipt.effect_fingerprint != proposal.effect_fingerprint
        or receipt.proposal_fingerprint != decision_fingerprint(proposal)
        or proposal.scope != effect.scope
        or proposal.target_key != effect.target_key
        or proposal.proposed_value != effect.proposed_value
        or proposal.evidence_set_fingerprint != effect.evidence_set_fingerprint
    ):
        raise PersistenceConflictError("Optimization Proposal authority proof is stale")
    if proposal.status is not OptimizationProposalStatus.ACTIVE:
        raise PersistenceConflictError("Optimization Proposal was revoked")
    if proposal.expires_at is not None and proposal.expires_at <= datetime.now(timezone.utc):
        raise PersistenceConflictError("Optimization Proposal expired")
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
    import json

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
    return proposal


def _load_receipt_in_transaction(
    cursor: object,
    effect_fingerprint: str,
) -> OptimizationApplyCommitReceipt | None:
    row = cursor.execute(  # type: ignore[attr-defined]
        "SELECT receipt_json FROM optimization_configuration_receipts "
        "WHERE effect_fingerprint = ?",
        (effect_fingerprint,),
    ).fetchone()
    return (
        None
        if row is None
        else OptimizationApplyCommitReceipt.model_validate_json(row["receipt_json"])
    )


def _insert_receipt(
    cursor: object,
    receipt: OptimizationApplyCommitReceipt,
) -> None:
    cursor.execute(  # type: ignore[attr-defined]
        "INSERT INTO optimization_configuration_receipts "
        "(effect_fingerprint, operation, payload_fingerprint, receipt_json) "
        "VALUES (?, ?, ?, ?)",
        (
            receipt.effect_fingerprint,
            receipt.operation,
            receipt.payload_fingerprint,
            receipt.model_dump_json(),
        ),
    )


def _verify_receipt_payload(
    receipt: OptimizationApplyCommitReceipt,
    payload_fingerprint: str,
    operation: str,
) -> None:
    if (
        receipt.payload_fingerprint != payload_fingerprint
        or receipt.operation != operation
    ):
        raise PersistenceConflictError(
            "Configuration Effect fingerprint conflicts with stored Receipt"
        )
